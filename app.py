import asyncio
import io
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


def _run_kokoclone(pcm_bytes: bytes, ref_audio_np: np.ndarray) -> bytes:
    try:
        from core.cloner import KokoClone
        cloner = KokoClone()
        host_audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        cloned = cloner.clone(host_audio, ref_audio_np)
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


# ---------------------------------------------------------------
# Room lifecycle
# ---------------------------------------------------------------
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

    # Confirm to the joiner
    await app.emit("room_joined", {"room_id": room_id}, client.id)
    # Notify the host
    await app.emit_to_room(room_id, "joiner_connected", {}, exclude=[client.id])


@app.on("delete_room")
async def delete_room(client, data):
    """Host closes the room and kicks everyone out."""
    room_id = data.get("room_id")
    room = rooms.get(room_id)
    if not room:
        await app.emit("error", {"message": "Room not found."}, client.id)
        return
    if room["host"] != client.id:
        await app.emit("error", {"message": "Only the host can close the room."}, client.id)
        return

    # Tell everyone in the room that it's closed
    await app.emit_to_room(room_id, "room_closed", {"room_id": room_id})

    # Clean up server state
    for cid in (room["host"], room["joiner"]):
        if cid:
            app.leave_room(cid, room_id)
    del rooms[room_id]


# ---------------------------------------------------------------
# Audio
# ---------------------------------------------------------------
@app.on_binary_event("voice_reference")
async def handle_voice_reference(client, payload: bytes):
    room_id = next((rid for rid, r in rooms.items() if r["host"] == client.id), None)
    if not room_id:
        await app.emit("error", {"message": "You are not a host in any room."}, client.id)
        return
    try:
        audio, _ = sf.read(io.BytesIO(payload))
        rooms[room_id]["voice_ref"] = audio.astype(np.float32)
        await app.emit("voice_ready", {"room_id": room_id}, client.id)
    except Exception as e:
        await app.emit("error", {"message": f"Invalid audio: {e}"}, client.id)


@app.on_binary_event("raw_mic")
async def handle_raw_mic(client, payload: bytes):
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
            await app.send_binary_event(target_id, "raw_audio", payload)
    else:
        await app.send_binary_event(target_id, "raw_audio", payload)


# ---------------------------------------------------------------
# Background cleanup
# ---------------------------------------------------------------
@app.task
async def monitor_rooms():
    while app._running:
        await asyncio.sleep(30)
        stale = [rid for rid, r in rooms.items() if time.time() - r["created_at"] > 3600]
        for rid in stale:
            await app.emit_to_room(rid, "room_closed", {"room_id": rid})
            del rooms[rid]
        if stale:
            print(f"[monitor] cleaned {len(stale)} rooms")


if __name__ == "__main__":
    print("VoiceClone Call running on http://0.0.0.0:8765")
    app.run()
