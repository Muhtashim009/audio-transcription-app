from fastapi import FastAPI, File, UploadFile, Form
from app.transcription import handle_transcription

app = FastAPI(title="Audio Transcription API", version="1.0")

@app.post("/v1/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form("small"),
    language_hint: str = Form("en"),
    enable_separation: bool = Form(True)
):
    return await handle_transcription(file, model_size, language_hint, enable_separation)

@app.get("/")
def health():
    return {"status": "ok", "message": "Transcription API is running"}
