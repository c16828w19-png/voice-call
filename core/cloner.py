# core/cloner.py
import os
import tempfile
import torch
import numpy as np
import soundfile as sf
from huggingface_hub import hf_hub_download
from kanade_tokenizer import KanadeModel, load_audio, load_vocoder, vocode

class KokoClone:
    def __init__(self, kanade_model="frothywater/kanade-25hz-clean", hf_repo="PatnaikAshish/kokoclone"):
        # Auto-detect GPU (CUDA) or fallback to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[KokoClone] Initializing on: {self.device.type.upper()}")
        self.hf_repo = hf_repo

        # Load Kanade Voice Conversion model
        print("[KokoClone] Loading Kanade Voice Conversion model...")
        self.kanade = KanadeModel.from_pretrained(kanade_model).to(self.device).eval()
        self.vocoder = load_vocoder(self.kanade.config.vocoder_name).to(self.device)
        self.sample_rate = self.kanade.config.sample_rate
        print(f"[KokoClone] Ready. Sample rate: {self.sample_rate}")

    def clone(self, source_audio: np.ndarray, reference_audio: np.ndarray) -> np.ndarray:
        """
        Real-time voice conversion.
        - source_audio: float32 numpy array (host microphone)
        - reference_audio: float32 numpy array (target voice)
        Returns: float32 numpy array of converted audio at self.sample_rate.
        """
        try:
            # Convert numpy arrays to torch tensors on the correct device
            source_tensor = torch.from_numpy(source_audio).float().to(self.device)
            ref_tensor = torch.from_numpy(reference_audio).float().to(self.device)

            # Ensure they are 1D
            if source_tensor.ndim > 1:
                source_tensor = source_tensor.squeeze()
            if ref_tensor.ndim > 1:
                ref_tensor = ref_tensor.squeeze()

            with torch.inference_mode():
                # Kanade voice conversion expects waveforms
                converted_mel = self.kanade.voice_conversion(
                    source_waveform=source_tensor.unsqueeze(0),  # add batch dim
                    reference_waveform=ref_tensor.unsqueeze(0)
                )
                # Vocode the mel spectrogram back to audio
                converted_wav = vocode(self.vocoder, converted_mel)
                # Remove batch dim and move to CPU numpy
                output = converted_wav.squeeze().cpu().numpy().astype(np.float32)
            return output
        except Exception as e:
            print(f"[KokoClone] Conversion error: {e}")
            # Fallback: return the original audio to keep the call alive
            return source_audio.astype(np.float32)
