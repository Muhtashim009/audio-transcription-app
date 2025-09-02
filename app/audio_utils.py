import os
import torch
import soundfile as sf
from demucs.apply import apply_model
from app.models import demucs_model
from app.config import DEVICE

def separate_vocals(audio_path: str, output_dir: str) -> str:
    """Separate vocals from background using Demucs."""
    try:
        wav, sr = sf.read(audio_path, always_2d=True)
        wav = torch.from_numpy(wav.T).float().unsqueeze(0)

        sources = apply_model(
            demucs_model,
            wav,
            device=DEVICE,
            progress=False,
            num_workers=0
        )

        sample_rate = demucs_model.samplerate
        vocals = sources[0, 3, :, :]  # Vocals source

        vocal_output = os.path.join(output_dir, "vocals.wav")
        sf.write(vocal_output, vocals.cpu().numpy().T, sample_rate)
        return vocal_output
    except Exception:
        # Fallback: return original audio if separation fails
        return audio_path
