# core/cloner.py
import os
import tempfile
import threading
from typing import Dict, Optional

import numpy as np


class ChatterboxCloner:
    """
    Wraps Chatterbox's voice-conversion mode.
    Load once at startup; per-room reference WAVs are cached as temp files.
    """

    def __init__(self):
        self._synth = None
        self._ready = False
        self._error: Optional[str] = None
        self._lock = threading.Lock()
        self._ref_paths: Dict[str, str] = {}   # room_id -> path to reference WAV

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def load(self):
        try:
            from chatterbox_onnx import ChatterboxOnnx
            print("[Chatterbox] loading model (ONNX, quantized)…")
            # quantized=True keeps the LLM at ~350 MB instead of ~2 GB
            self._synth = ChatterboxOnnx(quantized=True)
            self._ready = True
            print("[Chatterbox] READY")
        except Exception as e:
            self._error = str(e)
            self._ready = False
            print(f"[Chatterbox] DISABLED — {e}")
            print("[Chatterbox] Host audio will be passed through unchanged.")

    @property
    def ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------
    # Reference voice
    # ------------------------------------------------------------------
    def set_reference_from_wav_bytes(self, room_id: str, wav_bytes: bytes) -> bool:
        """Persist the host's reference WAV to a temp file for Chatterbox."""
        if not self._ready:
            return False
        try:
            fd, path = tempfile.mkstemp(
                suffix=".wav", prefix=f"ref_{room_id}_"
            )
            os.write(fd, wav_bytes)
            os.close(fd)
            with self._lock:
                # Clean up any previous reference for this room
                old = self._ref_paths.pop(room_id, None)
                if old and os.path.exists(old):
                    try:
                        os.remove(old)
                    except OSError:
                        pass
                self._ref_paths[room_id] = path
            print(f"[Chatterbox] reference stored for room={room_id}")
            return True
        except Exception as e:
            print(f"[Chatterbox] set_reference failed: {e}")
            return False

    def clear_reference(self, room_id: str):
        with self._lock:
            path = self._ref_paths.pop(room_id, None)
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Voice conversion (audio → audio)
    # ------------------------------------------------------------------
    def convert(self, room_id: str, pcm_24k_int16: bytes) -> Optional[bytes]:
        """
        Convert a chunk of host speech to the cloned voice.
        Returns Int16 mono PCM at 24 kHz, or None on failure.
        """
        if not self._ready:
            return None

        with self._lock:
            ref_path = self._ref_paths.get(room_id)
        if not ref_path:
            return None

        try:
            # Chatterbox expects a file path for source audio as well.
            fd, src_path = tempfile.mkstemp(
                suffix=".wav", prefix=f"src_{room_id}_"
            )
            os.close(fd)
            # Write PCM as a WAV file
            import soundfile as sf
            arr = np.frombuffer(pcm_24k_int16, dtype=np.int16).astype(np.float32) / 32768.0
            sf.write(src_path, arr, 24000, subtype="PCM_16")

            fd, out_path = tempfile.mkstemp(
                suffix=".wav", prefix=f"out_{room_id}_"
            )
            os.close(fd)

            with self._lock:
                self._synth.voice_convert(
                    source_audio_path=src_path,
                    target_voice_path=ref_path,
                    output_file_name=out_path,
                )

            audio, sr = sf.read(out_path, dtype="int16")
            # Normalise to 24 kHz if Chatterbox returns a different rate
            if sr != 24000:
                audio_f = audio.astype(np.float32) / 32768.0
                n_out = int(len(audio_f) * 24000 / sr)
                audio_f = np.interp(
                    np.linspace(0, len(audio_f) - 1, n_out),
                    np.arange(len(audio_f)),
                    audio_f,
                )
                audio = (audio_f * 32767).astype(np.int16)

            for p in (src_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

            return audio.tobytes()
        except Exception as e:
            print(f"[Chatterbox] convert failed: {e}")
            return None
