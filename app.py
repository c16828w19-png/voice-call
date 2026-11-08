# app.py
import asyncio
import io
import re
import time
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from lynkio import Lynk, render_template

from core.stt import VoskSTT, resample_int16, STT_SAMPLE_RATE
from core.tts import PocketTTS

BROWSER_SAMPLE_RATE = 24000

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

_stt: Optional[VoskSTT] = None
_tts = PocketTTS()


def _load_models_sync():
    global _stt
    try:
        _stt = VoskSTT()
        print("[STT] Vosk ready (local)")
    except Exception as e:
        _stt = None
        print(f"[STT] DISABLED — {e}")
    _tts.load()


@app.task
async def init_models_task():
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _load_models_sync)


@app.get("/")
async def index(req):
    return render_template("call.html", context={
        "title": "VoiceClone Call",
        "stt_ready": "true" if _stt is not None else "false",
        "tts_ready": "true" if _tts.ready else "false",
    })


@app.on("create_room")
async def create_room(client, data):
    room_id = (data.get("room_id") or "").strip()
    if not room_id:
        return await app.emit("error", {"message": "Enter a room code."}, client.id)
    if not ROOM_RE.match(room_id):
        return await app.emit("error", {"message": "Room code must be 3–64 chars."}, client.id)
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
        "stt_ready": _stt is not None,
        "tts_ready": _tts.ready,
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
    _tts.clear_reference(room_id)
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
        _tts.clear_reference(room_id)
        del rooms[room_id]
    else:
        room["joiner"] = None
        await app.emit_to_room(room_id, "joiner_left", {}, exclude=client.id)
    app.leave_room(client.id, room_id)


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

    if sr != BROWSER_SAMPLE_RATE:
        n_out = int(len(audio) * BROWSER_SAMPLE_RATE / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    rooms[room_id]["voice_ref"] = audio
    duration = len(audio) / BROWSER_SAMPLE_RATE

    tts_ok = False
    if _tts.ready:
        loop = asyncio.get_running_loop()
        tts_ok = await loop.run_in_executor(None, _tts.set_reference, room_id, payload)

    print(f"[voice_ref] OK: {duration:.2f}s @ {sr}Hz (tts_ok={tts_ok})")
    await app.emit("voice_ready", {
        "room_id": room_id,
        "duration": duration,
        "tts_active": tts_ok,
    }, client.id)


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

    if not is_host:
        await app.send_binary_event(target_id, "raw_audio", payload)
        return

    if not (_tts.ready and _stt is not None and room["voice_ref"] is not None):
        await app.send_binary_event(target_id, "raw_audio", payload)
        return

    try:
        pcm_16k = resample_int16(payload, BROWSER_SAMPLE_RATE, STT_SAMPLE_RATE)
        loop = asyncio.get_running_loop()
        text, is_endpoint = await loop.run_in_executor(None, _stt.feed, pcm_16k)

        if not is_endpoint or not text:
            return

        print(f"[pipeline] STT: {text!r}")
        await _stream_tts_to_client(room_id, text, target_id)

    except Exception as e:
        print(f"[pipeline] error: {e}")
        await app.send_binary_event(target_id, "raw_audio", payload)


async def _stream_tts_to_client(room_id: str, text: str, target_id: str):
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _worker():
        try:
            for chunk in _tts.synthesize_stream(room_id, text):
                if chunk:
                    asyncio.run_coroutine_threadsafe(queue.put(chunk), loop).result(timeout=10)
        except Exception as e:
            print(f"[TTS] worker error: {e}")
        finally:
            asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    loop.run_in_executor(None, _worker)

    sent = 0
    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        await app.send_binary_event(target_id, "cloned_audio", chunk)
        sent += len(chunk)
    print(f"[pipeline] TTS streamed {sent} bytes for: {text!r}")


@app.task
async def monitor_rooms():
    while app._running:
        await asyncio.sleep(30)
        stale = [rid for rid, r in rooms.items() if time.time() - r["created_at"] > 3600]
        for rid in stale:
            await app.emit_to_room(rid, "room_closed", {"room_id": rid})
            _tts.clear_reference(rid)
            del rooms[rid]
        if stale:
            print(f"[monitor] cleaned {len(stale)}, {len(rooms)} active")


if __name__ == "__main__":
    print("VoiceClone Call → http://0.0.0.0:8765")
    app.run()