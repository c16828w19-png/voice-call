import asyncio
import io
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Dict, Optional

import numpy as np
import soundfile as sf
from lynkio import Lynk, render_template, json_response

# ------------------------------------------------------------------
# 1. Server Configuration (Origin & Auth Removed)
# ------------------------------------------------------------------
app = Lynk(
    host="0.0.0.0",
    port=8765,
    protocol="AUTO",
    serve_client=True,  # Serves /lynkio/client.js automatically
    debug=True,
    # Origin validation and JWT middleware have been removed.
)

# ------------------------------------------------------------------
# 2. In-Memory Room Store
# ------------------------------------------------------------------
rooms: Dict[str, dict] = {}

# ------------------------------------------------------------------
# 3. Inference Pool (Keeps the event loop unblocked)
# ------------------------------------------------------------------
inference_pool = ProcessPoolExecutor(max_workers=2)

# In server.py, replace the placeholder _run_kokoclone with this:

def _run_kokoclone(pcm_bytes: bytes, ref_audio_np: np.ndarray) -> bytes:
    """Synchronous KokoClone inference – runs in a separate process."""
    try:
        from core.cloner import KokoClone
        cloner = KokoClone()
        # Decode incoming Int16 PCM to float32
        host_audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Real-time voice conversion
        cloned = cloner.clone(host_audio, ref_audio_np)
        # Encode back to Int16 PCM bytes for the WebSocket
        return (cloned * 32767).astype(np.int16).tobytes()
    except Exception as e:
        print(f"[Inference] KokoClone error: {e}")
        return pcm_bytes  # Fallback: pass original audio
        
        
async def clone_audio(pcm_bytes: bytes, ref_audio_np: np.ndarray) -> bytes:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(inference_pool, _run_kokoclone, pcm_bytes, ref_audio_np)

# ------------------------------------------------------------------
# 4. Frontend Route (Serving via render_template)
# ------------------------------------------------------------------
@app.get("/")
async def index(req):
    """Serve the main call UI using Lynkio's render_template."""
    return render_template("call.html", context={"title": "Zoom Call"})

# ------------------------------------------------------------------
# 5. Room Lifecycle Events
# ------------------------------------------------------------------
@app.on("create_room")
async def create_room(client, data):
    room_id = data.get("room_id")
    if not room_id:
        await app.emit("error", {"message": "Room code required."}, client.id)
        return
    if room_id in rooms:
        await app.emit("error", {"message": "Room already exists."}, client.id)
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
    room_id = data.get("room_id")
    room = rooms.get(room_id)
    if not room:
        await app.emit("error", {"message": "Invalid room code."}, client.id)
        return
    if room["joiner"] is not None:
        await app.emit("error", {"message": "Room is full."}, client.id)
        return

    room["joiner"] = client.id
    app.join_room(client.id, room_id)
    # Notify the host that the joiner is ready
    await app.emit_to_room(room_id, "joiner_connected", {}, exclude=client.id)

# ------------------------------------------------------------------
# 6. Binary Audio Handlers (Selective Relay)
# ------------------------------------------------------------------
@app.on_binary_event("voice_reference")
async def handle_voice_reference(client, payload: bytes):
    """Host uploads a reference clip (e.g., 5s WAV) for cloning."""
    room_id = next((rid for rid, r in rooms.items() if r["host"] == client.id), None)
    if not room_id:
        return
    try:
        audio, sr = sf.read(io.BytesIO(payload))
        rooms[room_id]["voice_ref"] = audio.astype(np.float32)
        return await app.emit("voice_ready", {"room_id": room_id}, client.id)
        
    except Exception as e:
        return await app.emit("error", {"message": f"Invalid audio file: {e}"}, client.id)

@app.on_binary_event("raw_mic")
async def handle_raw_mic(client, payload: bytes):
    """
    Route raw microphone audio.
    - Joiner -> Host : Pass through raw.
    - Host -> Joiner : Apply KokoClone inference before sending.
    """
    room_id = next(
        (rid for rid, r in rooms.items() if client.id in (r["host"], r["joiner"])), None
    )
    if not room_id:
        return
    room = rooms[room_id]
    is_host = client.id == room["host"]
    target_id = room["joiner"] if is_host else room["host"]

    if is_host:
        ref = room["voice_ref"]
        if ref is not None:
            cloned = await clone_audio(payload, ref)
            await app.send_binary_event(target_id, "cloned_audio", cloned)
        else:
            # Host speaks but hasn't uploaded a reference voice yet
            await app.send_binary_event(target_id, "raw_audio", payload)
    else:
        # Joiner's audio: raw pass-through to Host
        await app.send_binary_event(target_id, "raw_audio", payload)

# ------------------------------------------------------------------
# 7. Background Tasks (Cleanup & Monitoring)
# ------------------------------------------------------------------
@app.task
async def monitor_rooms():
    """Remove rooms older than 1 hour."""
    while app._running:
        await asyncio.sleep(30)
        stale = [rid for rid, r in rooms.items() if time.time() - r["created_at"] > 3600]
        for rid in stale:
            del rooms[rid]
        if stale:
            print(f"[monitor] cleaned {len(stale)} stale rooms. Active: {len(rooms)}")

# ------------------------------------------------------------------
# 8. Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print(" VoiceClone Call Server")
    print(" UI:   http://localhost:8765")
    print(" WS:   ws://localhost:8765")
    print(" JS:   /lynkio/client.js")
    print("=" * 50)
    app.run()
