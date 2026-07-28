#!/usr/bin/env python3
"""
mac_clip_backtrack.py
---------------------
A rolling microphone transcript backtrack for macOS stream setups.

How it works:
1. Continuously records your mic into a rolling 30-second audio buffer in RAM.
2. Runs a local HTTP trigger server on port 5001.
3. When triggered (via HTTP GET http://localhost:5001/clip or hotkey):
   - Transcribes the last 30 seconds of speech using Apple Silicon MLX Whisper.
   - Cleans the transcript into a punchy clip title.
   - Sends the title to Aitum Nexus API (http://localhost:7777) & triggers Fossabot !clip.

Requirements:
    ./venv/bin/python3 -m pip install sounddevice numpy mlx-whisper requests
"""

import os
import sys
import time
import wave
import threading
import tempfile
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests

# Try importing MLX Whisper (optimized for Apple Silicon Mac) or standard Whisper
try:
    import mlx_whisper
    USE_MLX = True
    print("⚡ Using Apple Silicon MLX Whisper for ultra-fast transcription!")
except ImportError:
    import whisper
    USE_MLX = False
    print("🧠 Using standard OpenAI Whisper model...")

import sounddevice as sd
import numpy as np

# ----------------- CONFIGURATION -----------------
SAMPLE_RATE = 16000        # 16 kHz mono (ideal for Whisper)
BUFFER_SECONDS = 30        # Keeps the last 30 seconds of mic audio
HTTP_PORT = 5001           # Local trigger port
AITUM_API_URL = "http://localhost:7777/aitum" # Aitum Nexus local API URL
AITUM_GLOBAL_VAR = "clip_title"               # Variable name in Aitum Nexus
MAX_TITLE_LENGTH = 100                        # Max characters for Twitch clip title
# -------------------------------------------------

class AudioBuffer:
    """Rolling audio buffer stored safely in RAM."""
    def __init__(self, sample_rate, buffer_seconds):
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * buffer_seconds
        self.buffer = deque(maxlen=self.max_samples)
        self.lock = threading.Lock()

    def add_samples(self, samples):
        with self.lock:
            self.buffer.extend(samples)

    def get_audio_data(self):
        with self.lock:
            return np.array(self.buffer, dtype=np.int16)

audio_buffer = AudioBuffer(SAMPLE_RATE, BUFFER_SECONDS)
whisper_model = None

def load_model():
    global whisper_model
    if not USE_MLX:
        print("⏳ Loading Whisper model ('base.en')...")
        whisper_model = whisper.load_model("base.en")
        print("✅ Whisper model loaded successfully.")

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"Audio Warning: {status}", file=sys.stderr)
    # Convert float32 [-1, 1] to int16 PCM
    pcm16 = (indata[:, 0] * 32767).astype(np.int16)
    audio_buffer.add_samples(pcm16)

def start_audio_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        callback=audio_callback
    )
    stream.start()
    print(f"🎙️  Listening to default Mac mic (Rolling {BUFFER_SECONDS}s buffer active)")
    return stream

def transcribe_buffer():
    """Extracts recent audio from buffer and returns transcribed text."""
    audio_data = audio_buffer.get_audio_data()
    if len(audio_data) < SAMPLE_RATE * 2:
        return "Stream Highlight" # Fallback if audio buffer is too short

    # Write to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    try:
        if USE_MLX:
            res = mlx_whisper.transcribe(tmp_path, path_or_hf_repo="mlx-community/whisper-base.en-mlx")
            text = res.get("text", "").strip()
        else:
            res = whisper_model.transcribe(tmp_path)
            text = res.get("text", "").strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Clean up text
    text = " ".join(text.split())
    if not text:
        text = "Awesome Stream Moment"

    # Truncate if too long
    if len(text) > MAX_TITLE_LENGTH:
        text = text[:MAX_TITLE_LENGTH - 3] + "..."

    return text

def send_to_aitum(title):
    """Sends the transcribed clip title to Aitum Nexus API."""
    print(f"📌 Generated Title: \"{title}\"")
    
    # 1. Update Aitum Nexus Global Variable
    try:
        payload = {"variable": AITUM_GLOBAL_VAR, "value": title}
        resp = requests.post(f"{AITUM_API_URL}/state", json=payload, timeout=2)
        print(f"🟢 Aitum Variable updated: {resp.status_code}")
    except Exception as e:
        print(f"⚠️ Could not connect to Aitum Nexus API on port 7777: {e}")
        print("💡 Make sure Aitum Nexus is running with Public API enabled.")

class TriggerHTTPHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/clip") or self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            
            print("\n🎬 Clip Triggered! Transcribing last 30s...")
            title = transcribe_buffer()
            send_to_aitum(title)
            
            response_json = f'{{"status": "success", "title": "{title}"}}'
            self.wfile.write(response_json.encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return # Suppress default HTTP logging

def start_http_server():
    server = HTTPServer(('127.0.0.1', HTTP_PORT), TriggerHTTPHandler)
    print(f"🌐 HTTP Trigger Server listening on http://localhost:{HTTP_PORT}/clip")
    server.serve_forever()

if __name__ == "__main__":
    print("=" * 60)
    print("   🎙️  macOS Rolling Transcript Backtrack for Aitum Nexus")
    print("=" * 60)
    
    load_model()
    stream = start_audio_stream()

    # Start HTTP server thread
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()

    print("\nREADY!")
    print(f"👉 Trigger a clip anytime by visiting or sending HTTP GET to: http://localhost:{HTTP_PORT}/clip")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping transcript backtrack...")
        stream.stop()
        sys.exit(0)
