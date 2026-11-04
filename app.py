import asyncio
import io
import re
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

import numpy as np
import soundfile as sf
from lynkio import Lynk, render_template

app = Lynk(
    host="0.0.0.0",
    port=8765,
    protocol="AUTO",
    serve_client=True,
    debug=True,
)

rooms: Dict[str, dict] = {}
inference_pool = ProcessPoolExecutor(max_workers=2)
_cloner_instance = None

ROOM_RE = re.compile(r"^[A-Za-z0-9_-]{3,64}$")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int = 24000) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    n = int(duration * target_sr)
    if n <= 0:
        return audio
    return np.interp(
        np.linspace(0, len(audio) - 1, n),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _run_kokoclone(pcm_bytes: bytes, ref_audio_np: np.ndarray) -> bytes:
    """Runs in a worker process — model cached per process."""
    global _cloner_instance
    try:
        if _cloner_instance is None:
            from core.cloner import KokoClone
            _cloner_instance = KokoClone()
        host_audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        cloned = _cloner_instance.clone(host_audio, ref_audio_np)
        return (cloned * 32767).astype(np.int16).tobytes()
    except Exception as e:
        print(f"[Inference] error: {e}")
        return pcm_bytes


async def clone_audio(pcm_bytes: bytes, ref_audio_np: np.ndarray) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(inference_pool, _run_kokoclone, pcm_bytes, ref_audio_np)


@app.get("/")
async def index(req):
    return render_template("call.html", context={"title": "VoiceClone Call"})


# ------------------------------------------------------------------
# CONTROL: JSON events (client.emit ↔ @app.on)
# ------------------------------------------------------------------
@app.on("create_room")
async def create_room(client, data):
    room_id = (data.get("room_id") or "").strip()
    if not room_id:
        await app.emit("error", {"message": "Enter a room code."}, client.id)
        return
    if not ROOM_RE.match(room_id):
        await app.emit("error", {"message": "Room code must be 3–64 chars: letters, digits, - or _."}, client.id)
        return
    if room_id in rooms:
        await app.emit("error", {"message": f'Room "{room_id}" already exists.'}, client.id)
        return

    rooms[room_id] = {
        "host": client.id,
        "joiner": None,
        "voice_ref": None,
        "created_at": time.time(),
    }
    app.join_room(client.id, room_id)
    await app.emit("room_created", {"room_id": room_id}, client.id)


@app.on("join_room")
async def join_room(client, data):
    room_id = (data.get("room_id") or "").strip()
    if not room_id:
        await app.emit("error", {"message": "Enter a room code."}, client.id)
        return
    room = rooms.get(room_id)
    if not room:
        await app.emit("error", {"message": f'Room "{room_id}" does not exist.'}, client.id)
        return
    if room["joiner"] is not None:
        await app.emit("error", {"message": "Room is full (2 participants max)."}, client.id)
        return
    if room["host"] == client.id:
        await app.emit("error", {"message": "You are the host — wait for a joiner."}, client.id)
        return

    room["joiner"] = client.id
    app.join_room(client.id, room_id)

    # CRITICAL: confirm to joiner so their UI can transition out of the spinner
    await app.emit("room_joined", {"room_id": room_id}, client.id)
    # Notify host
    await app.emit_to_room(room_id, "joiner_connected", {}, exclude=[client.id])


@app.on("delete_room")
async def delete_room(client, data):
    room_id = data.get("room_id")
    room = rooms.get(room_id)
    if not room:
        await app.emit("error", {"message": "Room not found."}, client.id)
        return
    if room["host"] != client.id:
        await app.emit("error", {"message": "Only the host can close the room."}, client.id)
        return

    await app.emit_to_room(room_id, "room_closed", {"room_id": room_id})
    for cid in (room["host"], room["joiner"]):
        if cid:
            app.leave_room(cid, room_id)
    del rooms[room_id]


@app.on("leave_room")
async def leave_room_event(client, data):
    room_id = next(
        (rid for rid, r in rooms.items() if client.id in (r["host"], r["joiner"])),
        None,
    )
    if not room_id:
        return
    room = rooms[room_id]
    if room["host"] == client.id:
        # Host leaving closes the room
        await app.emit_to_room(room_id, "room_closed", {"room_id": room_id}, exclude=client.id)
        if room["joiner"]:
            app.leave_room(room["joiner"], room_id)
        del rooms[room_id]
    else:
        # Joiner leaving — just notify host
        room["joiner"] = None
        await app.emit_to_room(room_id, "joiner_left", {}, exclude=client.id)
    app.leave_room(client.id, room_id)


# ------------------------------------------------------------------
# BINARY: audio (client.sendBinaryEvent ↔ @app.on_binary_event)
# ------------------------------------------------------------------
@app.on_binary_event("voice_reference")
async def handle_voice_reference(client, payload: bytes):
    """Host uploads a WAV — payload is the raw file bytes."""
    room_id = next((rid for rid, r in rooms.items() if r["host"] == client.id), None)
    if not room_id:
        await app.emit("error", {"message": "You are not hosting a room."}, client.id)
        return
    try:
        audio, sr = sf.read(io.BytesIO(payload), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = _resample(audio, sr, 24000)
        rooms[room_id]["voice_ref"] = audio.astype(np.float32)
        await app.emit("voice_ready", {"room_id": room_id, "duration": len(audio) / 24000}, client.id)
    except Exception as e:
        await app.emit("error", {"message": f"Invalid audio file: {e}"}, client.id)


@app.on_binary_event("raw_mic")
async def handle_raw_mic(client, payload: bytes):
    """Route PCM: joiner→host raw, host→joiner cloned."""
    room_id = next(
        (rid for rid, r in rooms.items() if client.id in (r["host"], r["joiner"])),
        None,
    )
    if not room_id:
        return
    room = rooms[room_id]
    is_host = client.id == room["host"]
    target_id = room["joiner"] if is_host else room["host"]
    if not target_id:
        return  # other side hasn't joined yet

    if is_host:
        ref = room["voice_ref"]
        if ref is not None:
            try:
                cloned = await clone_audio(payload, ref)
                await app.send_binary_event(target_id, "cloned_audio", cloned)
                return
            except Exception as e:
                print(f"[Clone] error: {e}")
        await app.send_binary_event(target_id, "raw_audio", payload)
    else:
        await app.send_binary_event(target_id, "raw_audio", payload)


# ------------------------------------------------------------------
# Background cleanup
# ------------------------------------------------------------------
@app.task
async def monitor_rooms():
    while app._running:
        await asyncio.sleep(30)
        stale = [rid for rid, r in rooms.items() if time.time() - r["created_at"] > 3600]
        for rid in stale:
            await app.emit_to_room(rid, "room_closed", {"room_id": rid})
            del rooms[rid]
        if stale:
            print(f"[monitor] cleaned {len(stale)}, {len(rooms)} active")


if __name__ == "__main__":
    print("VoiceClone Call → http://0.0.0.0:8765")
    app.run()
