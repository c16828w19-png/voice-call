# app.py
import asyncio
import io
import re
import time
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from lynkio import Lynk, render_template

from core.cloner import ChatterboxCloner

SAMPLE_RATE = 24000

app = Lynk(
    host="0.0.0.0",
    port=8765,
    protocol="AUTO",
    serve_client=True,
    debug=True,
    max_payload_size=8 * 1024 * 1024,
    max_message_size=8 * 1024 * 1024,
)

rooms: Dict[str, dict] = {}
ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")

_cloner = ChatterboxCloner()


@app.task
async def init_model_task():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _cloner.load)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------
@app.get("/")
async def index(req):
    return render_template("call.html", context={
        "title": "VoiceClone Call",
        "cloner_ready": "true" if _cloner.ready else "false",
    })


# ------------------------------------------------------------------
# Control (JSON)  ← client.emit
# ------------------------------------------------------------------
@app.on("create_room")
async def create_room(client, data):
    room_id = (data.get("room_id") or "").strip()
    if not room_id:
        return await app.emit("error", {"message": "Enter a room code."}, client.id)
    if not ROOM_RE.match(room_id):
        return await app.emit("error", {"message": "Room code must be 3–64 chars: letters, digits, - or _."}, client.id)
    if room_id in rooms:
        return await app.emit("error", {"message": f'Room "{room_id}" already exists.'}, client.id)

    rooms[room_id] = {
        "host": client.id,
        "joiner": None,
        "voice_ref": None,
        "created_at": time.time(),
    }
    app.join_room(client.id, room_id)
    await app.emit("room_created", {
        "room_id": room_id,
        "cloner_ready": _cloner.ready,
    }, client.id)


@app.on("join_room")
async def join_room(client, data):
    room_id = (data.get("room_id") or "").strip()
    if not room_id:
        return await app.emit("error", {"message": "Enter a room code."}, client.id)
    room = rooms.get(room_id)
    if not room:
        return await app.emit("error", {"message": f'Room "{room_id}" does not exist.'}, client.id)
    if room["joiner"] is not None:
        return await app.emit("error", {"message": "Room is full (2 participants max)."}, client.id)
    if room["host"] == client.id:
        return await app.emit("error", {"message": "You are the host — wait for a joiner."}, client.id)

    room["joiner"] = client.id
    app.join_room(client.id, room_id)
    await app.emit("room_joined", {"room_id": room_id}, client.id)
    await app.emit_to_room(room_id, "joiner_connected", {}, exclude=[client.id])


@app.on("delete_room")
async def delete_room(client, data):
    room_id = data.get("room_id")
    room = rooms.get(room_id)
    if not room:
        return await app.emit("error", {"message": "Room not found."}, client.id)
    if room["host"] != client.id:
        return await app.emit("error", {"message": "Only the host can close the room."}, client.id)

    await app.emit_to_room(room_id, "room_closed", {"room_id": room_id})
    for cid in (room["host"], room["joiner"]):
        if cid:
            app.leave_room(cid, room_id)
    _cloner.clear_reference(room_id)
    del rooms[room_id]


@app.on("leave_room")
async def leave_room_event(client, data):
    room_id = next((rid for rid, r in rooms.items() if client.id in (r["host"], r["joiner"])), None)
    if not room_id:
        return
    room = rooms[room_id]
    if room["host"] == client.id:
        await app.emit_to_room(room_id, "room_closed", {"room_id": room_id}, exclude=client.id)
        if room["joiner"]:
            app.leave_room(room["joiner"], room_id)
        _cloner.clear_reference(room_id)
        del rooms[room_id]
    else:
        room["joiner"] = None
        await app.emit_to_room(room_id, "joiner_left", {}, exclude=client.id)
    app.leave_room(client.id, room_id)


# ------------------------------------------------------------------
# Binary: voice reference  ← client.sendBinaryEvent
# ------------------------------------------------------------------
@app.on_binary_event("voice_reference")
async def handle_voice_reference(client, payload: bytes):
    print(f"[voice_ref] received {len(payload)} bytes from {client.id}")
    room_id = next((rid for rid, r in rooms.items() if r["host"] == client.id), None)
    if not room_id:
        return await app.emit("error", {"message": "You are not hosting a room."}, client.id)

    try:
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise ValueError("Audio file is empty")
    except Exception as e:
        print(f"[voice_ref] decode FAILED: {e}")
        return await app.emit("error", {"message": f"Invalid audio: {e}"}, client.id)

    # Normalise to 24 kHz for consistency
    if sr != SAMPLE_RATE:
        n_out = int(len(audio) * SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    rooms[room_id]["voice_ref"] = audio
    duration = len(audio) / SAMPLE_RATE

    cloner_ok = False
    if _cloner.ready:
        loop = asyncio.get_running_loop()
        cloner_ok = await loop.run_in_executor(
            None, _cloner.set_reference_from_wav_bytes, room_id, payload
        )

    print(f"[voice_ref] OK: {duration:.2f}s @ {sr}Hz (cloner_ok={cloner_ok})")
    await app.emit("voice_ready", {
        "room_id": room_id,
        "duration": duration,
        "cloner_ready": _cloner.ready,
        "cloner_active": cloner_ok,
    }, client.id)


# ------------------------------------------------------------------
# Binary: raw mic  ← client.sendBinaryEvent
# ------------------------------------------------------------------
@app.on_binary_event("raw_mic")
async def handle_raw_mic(client, payload: bytes):
    room_id = next((rid for rid, r in rooms.items() if client.id in (r["host"], r["joiner"])), None)
    if not room_id:
        return
    room = rooms[room_id]
    is_host = client.id == room["host"]
    target_id = room["joiner"] if is_host else room["host"]
    if not target_id:
        return

    # ── Joiner → Host: immediate raw relay ──
    if not is_host:
        await app.send_binary_event(target_id, "raw_audio", payload)
        return

    # ── Host → Joiner: Chatterbox voice conversion ──
    if not (_cloner.ready and room["voice_ref"] is not None):
        await app.send_binary_event(target_id, "raw_audio", payload)
        return

    try:
        loop = asyncio.get_running_loop()
        converted = await loop.run_in_executor(
            None, _cloner.convert, room_id, payload
        )
        if converted:
            await app.send_binary_event(target_id, "cloned_audio", converted)
        else:
            await app.send_binary_event(target_id, "raw_audio", payload)
    except Exception as e:
        print(f"[convert] error: {e}")
        await app.send_binary_event(target_id, "raw_audio", payload)


# ------------------------------------------------------------------
# Housekeeping
# ------------------------------------------------------------------
@app.task
async def monitor_rooms():
    while app._running:
        await asyncio.sleep(30)
        stale = [rid for rid, r in rooms.items() if time.time() - r["created_at"] > 3600]
        for rid in stale:
            await app.emit_to_room(rid, "room_closed", {"room_id": rid})
            _cloner.clear_reference(rid)
            del rooms[rid]
        if stale:
            print(f"[monitor] cleaned {len(stale)}, {len(rooms)} active")


if __name__ == "__main__":
    print("VoiceClone Call → http://0.0.0.0:8765")
    app.run()
