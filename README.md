 # Audio Transcription App

This project provides a containerized microservice for automatic speech transcription.  
The service accepts an audio file, separates vocals from background noise, and transcribes the speech into text.  
It is designed to be portable and easy to run using Docker.

---

## How It Works

1. The service exposes a REST API (via FastAPI).
2. Users can upload an audio file (e.g., MP3, WAV).
3. The system processes the file:
   - Separates vocals from background using source separation models.
   - Transcribes the speech using a state-of-the-art speech recognition model.
4. Returns the transcription as structured JSON.

---

## Technologies Used

- **Python 3.10+**
- **FastAPI** for serving the API
- **Docker** for containerization
- **Whisper** (OpenAI) for speech recognition
- **Demucs** for vocal separation

---

## Getting Started

You can pull and run the pre-built Docker image directly.

### 1. Pull the Docker Image

```bash
docker pull muhtashim996/audio-transcription-app
```

### 2. Run the Container

```bash
docker run -d -p 8000:8000 muhtashim996/audio-transcription-app
```

### 3. Access the API

### Open your browser and go to:

```bash
http://localhost:8000/docs
```

This will open the interactive API documentation (Swagger UI) where you can test the endpoints.
You can upload an audio file (e.g., .mp3, .wav) and get the transcription in real time.

## Example Usage

### 1. Navigate to http://localhost:8000/docs

### 2. Use the POST /v1/transcribe endpoint

### 3. Upload an audio file (e.g., sample.mp3)

### 4. The service will respond with JSON containing the transcription.

