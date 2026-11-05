# core/tts.py
import os
import tempfile
import threading
from typing import Dict, Iterator, Optional

import numpy as np

TTS_SAMPLE_RATE = 24000


class PocketTTS:
    """
    Wraps Pocket TTS. Loads once; caches per-room voice states from
    the host's reference WAV. Streaming synthesis yields Int16 PCM chunks.
    """

    def __init__(self):
        self._model = None
        self._ready = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._voice_states: Dict[str, object] = {}
        self._ref_paths: Dict[str, str] = {}

    def load(self):
        try:
            from pocket_tts import TTSModel
            print("[TTS] loading Pocket TTS (100M params, CPU)…")
            self._model = TTSModel.load_model()
            self._ready = True
            print(f"[TTS] READY — sample_rate={self._model.sample_rate}")
        except Exception as e:
            self._error = str(e)
            self._ready = False
            print(f"[TTS] DISABLED — {e}")

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> Optional[str]:
        return self._error

    # ------------------------------------------------------------------
    # Voice reference
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

    # ------------------------------------------------------------------
    # Streaming synthesis
    # ------------------------------------------------------------------
    def synthesize_stream(self, room_id: str, text: str) -> Iterator[bytes]:
        """
        Yields Int16 PCM chunks at 24 kHz as they are decoded.
        Empty generator if TTS is unavailable or no reference voice.
        """
        if not self._ready or not text:
            return
        with self._lock:
            state = self._voice_states.get(room_id)
        if state is None:
            return

        try:
            for chunk in self._model.generate_audio_stream(state, text):
                # Pocket TTS yields torch tensors
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