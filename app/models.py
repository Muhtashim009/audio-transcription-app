from demucs.pretrained import get_model
import whisper
from app.config import DEVICE

def load_demucs():
    """Load Demucs model."""
    try:
        model = get_model(name="htdemucs")
        model.to(DEVICE)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Demucs: {str(e)}")

def load_whisper(model_size="small"):
    """Load Whisper model."""
    try:
        model = whisper.load_model(model_size).to(DEVICE)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper: {str(e)}")

# Eager load models
whisper_model = load_whisper("small")
demucs_model = load_demucs()
