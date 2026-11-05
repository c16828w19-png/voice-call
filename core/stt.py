# core/stt.py
import json
import os
import threading
import urllib.request
import zipfile
from typing import Optional

import numpy as np

MODEL_DIR = "vosk-model-small-en-us-0.15"
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
STT_SAMPLE_RATE = 16000


def _ensure_model() -> str:
    if os.path.isdir(MODEL_DIR):
        return MODEL_DIR
    print("[STT] downloading Vosk model (~50 MB)…")
    urllib.request.urlretrieve(MODEL_URL, "vosk_model.zip")
    with zipfile.ZipFile("vosk_model.zip", "r") as z:
        z.extractall(".")
    os.remove("vosk_model.zip")
    print(f"[STT] model extracted to {MODEL_DIR}")
    return MODEL_DIR


class VoskSTT:
    """Per-room streaming STT. Feed 16 kHz Int16 PCM."""

    def __init__(self):
        from vosk import Model, KaldiRecognizer, SetLogLevel
        SetLogLevel(-1)
        self._model = Model(_ensure_model())
        self._recognizer = KaldiRecognizer(self._model, STT_SAMPLE_RATE)
        self._lock = threading.Lock()

    def feed(self, pcm_16k: bytes):
        """
        Returns (final_text, is_endpoint).
        final_text is only set when Vosk detects an utterance boundary.
        """
        with self._lock:
            is_endpoint = self._recognizer.AcceptWaveform(pcm_16k)
            if is_endpoint:
                try:
                    text = json.loads(self._recognizer.Result()).get("text", "").strip()
                except Exception:
                    text = ""
                return text, True
            return "", False

    def reset(self):
        with self._lock:
            try:
                self._recognizer.Reset()
            except Exception:
                pass


def resample_int16(pcm: bytes, orig_sr: int, target_sr: int) -> bytes:
    if orig_sr == target_sr or len(pcm) < 4:
        return pcm
    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32)
    n_out = int(len(arr) * target_sr / orig_sr)
    if n_out <= 0:
        return b""
    out = np.interp(
        np.linspace(0, len(arr) - 1, n_out),
        np.arange(len(arr)),
        arr,
    )
    return out.astype(np.int16).tobytes()