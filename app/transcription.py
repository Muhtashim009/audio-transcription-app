import os
import tempfile
import uuid
import time
import soundfile as sf
import torch
import torchaudio
from fastapi import UploadFile, HTTPException, Form, File
from app.audio_utils import separate_vocals
from app.models import whisper_model
from app.config import DEVICE

async def handle_transcription(
    file: UploadFile,
    model_size: str = "small",
    language_hint: str = "en",
    enable_separation: bool = True
):
    request_id = str(uuid.uuid4())
    start_total = time.time()

    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    file_content = await file.read()
    if len(file_content) > 25 * 1024 * 1024:  # 25MB
        raise HTTPException(status_code=413, detail="File too large (max 25MB)")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        with open(input_path, "wb") as f:
            f.write(file_content)

        try:
            info = sf.info(input_path)
            duration_sec = info.duration
            sample_rate = info.samplerate
        except Exception:
            duration_sec = 0
            sample_rate = 16000

        # Separation step
        sep_start = time.time()
        if enable_separation:
            vocal_path = separate_vocals(input_path, tmpdir)
            separation_time = int((time.time() - sep_start) * 1000)
        else:
            vocal_path = input_path
            separation_time = 0

        # Transcription step
        trans_start = time.time()
        try:
            audio, sr = sf.read(vocal_path)
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                tensor = torch.from_numpy(audio).float().unsqueeze(0)
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio = resampler(tensor).squeeze().numpy()

            result = whisper_model.transcribe(
                audio,
                language=language_hint or None,
                fp16=(DEVICE == "cuda"),
                word_timestamps=True
            )
            text = result["text"]
            detected_language = result.get("language", "unknown")
            segments = [
                {"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()}
                for seg in result["segments"]
            ]
            transcription_time = int((time.time() - trans_start) * 1000)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

        total_time = int((time.time() - start_total) * 1000)

        return {
            "request_id": request_id,
            "duration_sec": round(duration_sec, 3),
            "sample_rate": sample_rate,
            "pipeline": {
                "separation": {"enabled": enable_separation, "method": "demucs"},
                "transcription": {"model": f"whisper-{model_size}"}
            },
            "text": text.strip(),
            "language": detected_language,
            "segments": segments,
            "timings_ms": {
                "separation": separation_time,
                "transcription": transcription_time,
                "total": total_time
            }
        }
