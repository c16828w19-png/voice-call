# core/tts.py
import os
import tempfile
import threading
from typing import Dict, Iterator, Optional

import numpy as np

TTS_SAMPLE_RATE = 24000
LOCAL_CONFIG = "/models/pocket-tts/local.yaml"


class PocketTTS:
    def __init__(self):
        self._model = None
        self._ready = False
        self._cloning_ready = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._voice_states: Dict[str, object] = {}
        self._ref_paths: Dict[str, str] = {}

    def load(self):
        if not os.path.isfile(LOCAL_CONFIG):
            self._error = f"Config not found: {LOCAL_CONFIG}"
            print(f"[TTS] DISABLED — {self._error}")
            return

        try:
            from pocket_tts import TTSModel
            print(f"[TTS] loading Pocket TTS from local config {LOCAL_CONFIG}…")
            self._model = TTSModel.load_model(
                config=LOCAL_CONFIG,
                quantize=True,
            )
            self._ready = True
            print("[TTS] READY (local weights)")
        except Exception as e:
            self._error = str(e)
            self._ready = False
            print(f"[TTS] DISABLED — {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def cloning_ready(self) -> bool:
        return self._cloning_ready

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------
    def set_reference(self, room_id: str, wav_bytes: bytes) -> bool:
        if not self._ready:
            return False
        try:
            fd, path = tempfile.mkstemp(suffix=".wav", prefix=f"ref_{room_id}_")
            os.write(fd, wav_bytes)
            os.close(fd)

            with self._lock:
                old = self._ref_paths.pop(room_id, None)
                if old and os.path.exists(old):
                    try:
                        os.remove(old)
                    except OSError:
                        pass
                self._ref_paths[room_id] = path
                self._voice_states[room_id] = self._model.get_state_for_audio_prompt(path)

            self._cloning_ready = True
            print(f"[TTS] voice reference set for room={room_id}")
            return True
        except Exception as e:
            print(f"[TTS] set_reference failed: {e}")
            return False

    def clear_reference(self, room_id: str):
        with self._lock:
            self._voice_states.pop(room_id, None)
            path = self._ref_paths.pop(room_id, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass
        if not self._voice_states:
            self._cloning_ready = False

    # ------------------------------------------------------------------
    def synthesize_stream(self, room_id: str, text: str) -> Iterator[bytes]:
        if not self._ready or not text:
            return
        with self._lock:
            state = self._voice_states.get(room_id)
        if state is None:
            return

        try:
            for chunk in self._model.generate_audio_stream(state, text):
                try:
                    arr = chunk.detach().cpu().numpy() if hasattr(chunk, "detach") else np.asarray(chunk)
                except Exception:
                    continue
                arr = np.asarray(arr, dtype=np.float32).flatten()
                if arr.size == 0:
                    continue
                arr = np.clip(arr, -1.0, 1.0)
                yield (arr * 32767).astype(np.int16).tobytes()
        except Exception as e:
            print(f"[TTS] stream failed: {e}")