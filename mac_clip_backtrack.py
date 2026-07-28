#!/usr/bin/env python3
"""
mac_clip_backtrack.py
---------------------
A pro-grade rolling microphone transcript backtrack for macOS stream setups.

Features:
1. Native Apple AppKit macOS Sequoia Menu Bar App (Pause / Resume, Manual Trigger, Live Status)
2. 30-Second Rolling Audio RAM Buffer (Default, customizable via ?duration=15)
3. Local Apple Silicon MLX Whisper Speech-to-Text
4. Witty, Twitch-Chatter-Style AI Title Rewriting via Ollama
5. Native macOS Notifications & Auto-Copy to Clipboard (pbcopy)
6. Twitch TOS Profanity / Brand Safety Filtering
7. Game Category Context Support (?game=Valorant)
8. Automatic Aitum Nexus API Integration (http://localhost:7777)
"""

import os
import re
import sys
import json
import math
import time
import wave
import html
import threading
import tempfile
import subprocess
import urllib.parse
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests

try:
    import whisper  # type: ignore
    HAS_WHISPER = True
    print("🧠 Using standard OpenAI Whisper model ('base.en')...")
except ImportError:
    HAS_WHISPER = False
    print("⚠️ openai-whisper package not installed.")

import sounddevice as sd
import numpy as np
from scipy import signal

# ----------------- CONFIGURATION -----------------
STREAMER_NAME = "Mesut"                           # Your streamer name
TWITCH_CHANNEL = "mesutkaya"                      # Your Twitch channel username (for live category auto-detection)
CUSTOM_WORDS = []                                 # Custom game terms or jargon (streamer name is handled automatically)
DEFAULT_GAME = ""                                 # Default game category (e.g. "Valorant", "Just Chatting")
MICROPHONE_DEVICE = ""                            # Mic device name; empty = system default input. Set via dashboard.
PROFANITY_FILTER = True                           # Keep titles Twitch TOS brand-safe
ENABLE_NOTIFICATIONS = True                       # Display macOS desktop notifications
ENABLE_CLIPBOARD_COPY = True                      # Auto-copy titles to macOS Clipboard (Cmd+V)

ENABLE_YOUTUBE_UPLOAD = False                     # Auto-upload clips/shorts to YouTube in background (opt-in)
YOUTUBE_PRIVACY_STATUS = "unlisted"                # Privacy: 'public', 'unlisted', or 'private'
MAX_UPLOAD_KBPS = 1500                            # Upload speed cap in KB/s (1500 KB/s = ~12 Mbps limit, safe for dual streaming)
OBS_CLIPS_DIR = os.path.expanduser("~/Movies")     # Directory where OBS saves video clips
CLIENT_SECRET_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "client_secret.json")
TOKEN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "youtube_token.json")
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""

current_mic_volume = 0.0
live_realtime_transcript = "Listening for live speech..."

def load_config():
    """Loads persistent configuration settings from config.json."""
    global STREAMER_NAME, TWITCH_CHANNEL, CUSTOM_WORDS, DEFAULT_GAME, MICROPHONE_DEVICE
    global ENABLE_YOUTUBE_UPLOAD, YOUTUBE_PRIVACY_STATUS, MAX_UPLOAD_KBPS, OBS_CLIPS_DIR
    global ENABLE_NOTIFICATIONS, ENABLE_CLIPBOARD_COPY, PROFANITY_FILTER
    global GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                c = json.load(f)
                STREAMER_NAME = str(c.get("streamer_name", STREAMER_NAME))
                TWITCH_CHANNEL = str(c.get("twitch_channel", TWITCH_CHANNEL))
                CUSTOM_WORDS = list(c.get("custom_words", CUSTOM_WORDS))
                DEFAULT_GAME = str(c.get("default_game", DEFAULT_GAME))
                MICROPHONE_DEVICE = str(c.get("mic_device", MICROPHONE_DEVICE))
                ENABLE_YOUTUBE_UPLOAD = bool(c.get("enable_yt", ENABLE_YOUTUBE_UPLOAD))
                YOUTUBE_PRIVACY_STATUS = str(c.get("yt_privacy", YOUTUBE_PRIVACY_STATUS))
                MAX_UPLOAD_KBPS = int(c.get("max_upload_kbps", MAX_UPLOAD_KBPS))
                OBS_CLIPS_DIR = str(c.get("obs_clips_dir", OBS_CLIPS_DIR))
                ENABLE_NOTIFICATIONS = bool(c.get("enable_notif", ENABLE_NOTIFICATIONS))
                ENABLE_CLIPBOARD_COPY = bool(c.get("enable_clip", ENABLE_CLIPBOARD_COPY))
                PROFANITY_FILTER = bool(c.get("profanity_filter", PROFANITY_FILTER))
                GOOGLE_CLIENT_ID = str(c.get("google_client_id", GOOGLE_CLIENT_ID))
                GOOGLE_CLIENT_SECRET = str(c.get("google_client_secret", GOOGLE_CLIENT_SECRET))
                print(f"💾 Persistent Config loaded from config.json! Active mic: '{MICROPHONE_DEVICE}'")
        except Exception as e:
            print(f"⚠️ Config load note: {e}")

def save_config():
    """Saves current configuration persistently to config.json."""
    try:
        data = {
            "streamer_name": STREAMER_NAME,
            "twitch_channel": TWITCH_CHANNEL,
            "custom_words": CUSTOM_WORDS,
            "default_game": DEFAULT_GAME,
            "mic_device": MICROPHONE_DEVICE,
            "enable_yt": ENABLE_YOUTUBE_UPLOAD,
            "yt_privacy": YOUTUBE_PRIVACY_STATUS,
            "max_upload_kbps": MAX_UPLOAD_KBPS,
            "obs_clips_dir": OBS_CLIPS_DIR,
            "enable_notif": ENABLE_NOTIFICATIONS,
            "enable_clip": ENABLE_CLIPBOARD_COPY,
            "profanity_filter": PROFANITY_FILTER,
            "google_client_id": GOOGLE_CLIENT_ID,
            "google_client_secret": GOOGLE_CLIENT_SECRET
        }
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("💾 Settings saved to config.json successfully!")
    except Exception as e:
        print(f"⚠️ Config save error: {e}")

load_config()

def save_client_secret_json(client_id, client_secret):
    """Generates client_secret.json from provided OAuth credentials."""
    global GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
    GOOGLE_CLIENT_ID = client_id.strip()
    GOOGLE_CLIENT_SECRET = client_secret.strip()
    
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return False

    data = {
        "installed": {
            "client_id": GOOGLE_CLIENT_ID,
            "project_id": "clip-backtrack-app",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/base/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uris": ["http://localhost"]
        }
    }
    with open(CLIENT_SECRET_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"✅ Generated client_secret.json at '{CLIENT_SECRET_FILE}'")
    return True

SAMPLE_RATE = 16000                               # 16 kHz mono (ideal for Whisper)
MAX_BUFFER_SECONDS = 60                           # Max ring buffer length in RAM
DEFAULT_CLIP_SECONDS = 30                         # Default transcript backtrack window
HTTP_PORT = 5001                                  # Local trigger port
AITUM_API_URL = "http://localhost:7777/aitum"        # Aitum Nexus local API URL
AITUM_GLOBAL_VAR = "clip_title"                      # Variable name in Aitum Nexus
MAX_TITLE_LENGTH = 50                             # Max characters for Twitch clip title
# -------------------------------------------------

RECORDING_PAUSED = False
menu_app_instance = None

class AudioBuffer:
    """Fixed-size int16 ring buffer in RAM (no per-sample Python objects)."""
    def __init__(self, sample_rate, max_buffer_seconds):
        self.sample_rate = sample_rate
        self.capacity = sample_rate * max_buffer_seconds
        self.buffer = np.zeros(self.capacity, dtype=np.int16)
        self.write = 0      # next write position
        self.filled = 0     # number of valid samples, capped at capacity
        self.lock = threading.Lock()

    def add_samples(self, samples):
        if RECORDING_PAUSED:
            return  # Skip recording when paused
        samples = np.asarray(samples, dtype=np.int16)
        n = samples.size
        if n == 0:
            return
        if n >= self.capacity:          # a single block bigger than the buffer
            samples = samples[-self.capacity:]
            n = self.capacity
        with self.lock:
            end = self.write + n
            if end <= self.capacity:
                self.buffer[self.write:end] = samples
            else:                       # wrap around
                split = self.capacity - self.write
                self.buffer[self.write:] = samples[:split]
                self.buffer[:n - split] = samples[split:]
            self.write = (self.write + n) % self.capacity
            self.filled = min(self.capacity, self.filled + n)

    def get_audio_data(self, seconds=30):
        """Returns the last N seconds of recorded audio, oldest-to-newest."""
        with self.lock:
            want = min(self.filled, self.sample_rate * seconds)
            if want == 0:
                return np.array([], dtype=np.int16)
            start = (self.write - want) % self.capacity
            if start + want <= self.capacity:
                return self.buffer[start:start + want].copy()
            split = self.capacity - start
            return np.concatenate((self.buffer[start:], self.buffer[:want - split]))

audio_buffer = AudioBuffer(SAMPLE_RATE, MAX_BUFFER_SECONDS)
whisper_model = None

def load_model():
    global whisper_model
    if HAS_WHISPER and whisper_model is None:
        print("⏳ Loading Whisper model ('base.en') into RAM...")
        whisper_model = whisper.load_model("base.en")
        print("✅ Whisper model loaded successfully into RAM.")

active_audio_stream = None
stream_sample_rate = SAMPLE_RATE   # actual device rate; audio is resampled to SAMPLE_RATE for the buffer

def audio_callback(indata, frames, time_info, status):
    global current_mic_volume
    if status:
        print(f"Audio Warning: {status}", file=sys.stderr)

    # Store peak amplitude (float 0.0 to 1.0) for live VU meter
    current_mic_volume = float(np.max(np.abs(indata)))

    # Convert multi-channel input to clean mono PCM16
    if indata.ndim > 1 and indata.shape[1] > 1:
        mono_signal = np.mean(indata, axis=1)
    else:
        mono_signal = indata[:, 0] if indata.ndim > 1 else indata.flatten()

    # Resample device-native rate down to 16 kHz for the buffer/Whisper.
    # Virtual devices (Loopback/BlackHole) run at 48 kHz and don't resample
    # themselves, so opening them at 16 kHz yielded broken/empty capture.
    if stream_sample_rate != SAMPLE_RATE:
        g = math.gcd(int(stream_sample_rate), SAMPLE_RATE)
        mono_signal = signal.resample_poly(mono_signal, SAMPLE_RATE // g, int(stream_sample_rate) // g)

    pcm16 = (np.clip(mono_signal, -1.0, 1.0) * 32767).astype(np.int16)
    audio_buffer.add_samples(pcm16)

def get_audio_devices():
    """Lists available input audio devices on macOS."""
    devices = []
    try:
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                devices.append({"id": i, "name": d.get("name")})
    except Exception as e:
        print(f"⚠️ Audio device listing error: {e}")
    return devices

def start_audio_stream():
    global active_audio_stream
    if active_audio_stream:
        try:
            active_audio_stream.stop()
            active_audio_stream.close()
        except Exception:
            pass

    device_idx = None
    devices = sd.query_devices()

    # 1. Search for user-configured microphone
    if MICROPHONE_DEVICE:
        for i, d in enumerate(devices):
            if MICROPHONE_DEVICE.lower() in d['name'].lower() and d['max_input_channels'] > 0:
                device_idx = i
                break

    # 2. Fallback to system default input
    if device_idx is None:
        try:
            device_idx = sd.default.device[0]
        except Exception:
            device_idx = None

    global stream_sample_rate
    dev_info = devices[device_idx] if device_idx is not None and device_idx < len(devices) else {}
    dev_name = dev_info.get('name', 'System Default')
    chans = max(1, min(2, dev_info.get('max_input_channels', 1)))
    stream_sample_rate = int(dev_info.get('default_samplerate', SAMPLE_RATE)) or SAMPLE_RATE

    print(f"🎙️  Listening to Microphone Device #{device_idx}: '{dev_name}' "
          f"({chans} ch @ {stream_sample_rate} Hz -> {SAMPLE_RATE} Hz)...")

    active_audio_stream = sd.InputStream(
        device=device_idx,
        samplerate=stream_sample_rate,
        channels=chans,
        dtype='float32',
        callback=audio_callback
    )
    active_audio_stream.start()
    return active_audio_stream

def normalize_pcm16_gain(pcm16_samples):
    """Applies automatic digital gain boost so Whisper receives clear audio."""
    if len(pcm16_samples) == 0:
        return pcm16_samples
    peak = float(np.max(np.abs(pcm16_samples)))
    if peak > 5:
        gain = 24000.0 / peak
        return np.clip(pcm16_samples * gain, -32767, 32767).astype(np.int16)
    return pcm16_samples

def live_transcription_loop():
    """Background thread that continuously transcribes live speech in real time."""
    global live_realtime_transcript
    print("🎙️ Continuous Live Real-Time Transcription active...")
    while True:
        try:
            time.sleep(1.5)
            if RECORDING_PAUSED:
                continue

            audio_data = audio_buffer.get_audio_data(seconds=4)
            if len(audio_data) < SAMPLE_RATE * 1.0:
                continue

            # Apply digital gain normalization
            boosted_audio = normalize_pcm16_gain(audio_data)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                tmp_path = tmp_file.name
                with wave.open(tmp_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(boosted_audio.tobytes())

            try:
                if whisper_model:
                    res = whisper_model.transcribe(tmp_path, fp16=False)
                    text = res.get("text", "").strip()
                else:
                    text = ""

                text = " ".join(text.split())
                if text:
                    live_realtime_transcript = text
                    print(f"🗣️ [Whisper STT Output]: \"{text}\"")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except Exception as e:
            time.sleep(1.5)

def sanitize_profanity(text):
    """Simple profanity cleaner for Twitch TOS brand safety."""
    if not PROFANITY_FILTER or not text:
        return text
    bad_words = {
        r'\bfuck(ing|er|ed)?\b': 'f***',
        r'\bshit(ting|ty)?\b': 's***',
        r'\bbitch(es)?\b': 'b****',
        r'\basshole\b': 'a**hole',
        r'\bcunt\b': 'c***'
    }
    cleaned = text
    for pattern, replacement in bad_words.items():
        cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
    return cleaned

def copy_and_notify(title):
    """Copies generated title to macOS Clipboard and shows a desktop notification."""
    # 1. Auto-Copy to Clipboard
    if ENABLE_CLIPBOARD_COPY:
        try:
            proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
            proc.communicate(input=title.encode('utf-8'))
            print("📋 Title copied to macOS Clipboard!")
        except Exception as e:
            print(f"⚠️ Clipboard copy error: {e}")

    # 2. Native macOS Notification
    if ENABLE_NOTIFICATIONS:
        try:
            escaped_title = title.replace('"', '\\"').replace("'", "'")
            applescript = f'display notification "{escaped_title}" with title "🎬 Clip Backtrack" subtitle "Title copied to Clipboard!"'
            subprocess.run(['osascript', '-e', applescript], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"⚠️ Notification error: {e}")

current_detected_game = "Auto-Detecting..."
current_detected_jargon = []

# ----------------- AI DYNAMIC GAME JARGON GENERATOR & CACHE -----------------

JARGON_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jargon_cache.json")
jargon_memory_cache = {}

def load_jargon_cache():
    global jargon_memory_cache
    if os.path.exists(JARGON_CACHE_FILE):
        try:
            with open(JARGON_CACHE_FILE, "r", encoding="utf-8") as f:
                jargon_memory_cache = json.load(f)
        except Exception:
            jargon_memory_cache = {}

def save_jargon_cache():
    try:
        with open(JARGON_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(jargon_memory_cache, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

load_jargon_cache()

def get_game_jargon_ai(game_name=""):
    """Uses Ollama AI to dynamically generate jargon & character names for ANY video game category."""
    if not game_name or len(game_name) < 2:
        return []

    clean_game = game_name.strip()
    cache_key = clean_game.lower()

    # 1. Return from persistent local cache if already generated
    if cache_key in jargon_memory_cache:
        return jargon_memory_cache[cache_key]

    # 2. Check instant fallback database for instant response
    fallback_database = {
        "pokémon": ["Pikachu", "Charizard", "Pokéball", "Shiny", "Legendary", "Gym", "Evolution", "Pokédex", "Catch", "Trainer"],
        "valorant": ["Jett", "Reyna", "Raze", "Sova", "Viper", "Omen", "Chamber", "Fade", "Iso", "Clove", "Operator", "Op", "Vandal", "Phantom", "Ace", "Clutch", "Spike", "Ulting", "Headshot"],
        "league of legends": ["Faker", "Baron", "Dragon", "Mid", "Top", "Jungle", "Bot", "Ult", "Pentakill", "Gank", "Smite", "Flash", "Yasuo", "Zed", "Jinx"],
        "apex legends": ["Wraith", "Pathfinder", "Octane", "Horizon", "Bloodhound", "Gibby", "Shield Crack", "Kraber", "Peacekeeper", "Master", "Predator"],
        "counter-strike": ["AWP", "AK-47", "Deagle", "Clutch", "Ace", "Defuse", "Bomb", "Eco", "Dust2", "Mirage", "Rush B"],
        "fortnite": ["Victory Royale", "Crank", "Build", "Boxed", "Pump", "Chug", "Siphon", "Sweat", "Default"],
        "overwatch": ["Genji", "Tracer", "Reinhardt", "Mercy", "Ana", "Kiriko", "C9", "Nanoboost", "Shatter", "Team Kill"],
        "just chatting": ["Chat", "Bro", "Lmao", "W", "L", "Based", "Ratio", "Real", "GigaChad"]
    }

    for key, terms in fallback_database.items():
        if key in cache_key or cache_key in key:
            jargon_memory_cache[cache_key] = terms
            save_jargon_cache()
            return terms

    # 3. Query Ollama AI to generate game-specific jargon dynamically for new/indie games
    try:
        prompt = (
            f"You are a video game expert. Output 15 comma-separated key characters, items, abilities, and jargon words for the game '{clean_game}'. "
            "Respond ONLY with the comma-separated words."
        )
        payload = {
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2}
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=4)
        if resp.status_code == 200:
            raw = resp.json().get("response", "").strip()
            terms = [t.strip().strip('"').strip("'").replace("\n", " ") for t in raw.split(",") if t.strip() and len(t.strip()) < 30]
            if terms:
                print(f"🤖 AI Generated {len(terms)} dynamic jargon terms for category '{clean_game}': {terms[:6]}...")
                jargon_memory_cache[cache_key] = terms
                save_jargon_cache()
                return terms
    except Exception as e:
        print(f"⚠️ Ollama AI jargon generation note: {e}")

    return []

def get_live_twitch_game(streamer_name=None):
    """Auto-detects current live Twitch game/category for the streamer."""
    global current_detected_game, current_detected_jargon
    target_channel = TWITCH_CHANNEL if TWITCH_CHANNEL else (streamer_name if streamer_name else STREAMER_NAME)
    if not target_channel:
        current_detected_game = DEFAULT_GAME
        return DEFAULT_GAME

    try:
        url = f"https://decapi.me/twitch/game/{urllib.parse.quote(target_channel)}"
        resp = requests.get(url, timeout=2)
        if resp.status_code == 200:
            game = resp.text.strip()
            if game and "error" not in game.lower() and "user not found" not in game.lower():
                print(f"🎮 Live Twitch Category auto-detected for channel '{target_channel}': \"{game}\"")
                current_detected_game = game
                current_detected_jargon = get_game_jargon_ai(game)
                return game
    except Exception as e:
        print(f"⚠️ Live Twitch game query note: {e}")

    current_detected_game = DEFAULT_GAME if DEFAULT_GAME else "Just Chatting"
    current_detected_jargon = get_game_jargon_ai(current_detected_game)
    return current_detected_game

def ai_rewrite_title(raw_text, game_name=""):
    """Rewrites raw spoken transcript into a catchy Twitch clip title using AI."""
    if not raw_text or len(raw_text) < 5:
        return None

    auto_jargon = get_game_jargon_ai(game_name)
    combined_words = list(dict.fromkeys(CUSTOM_WORDS + auto_jargon))
    words_context = ", ".join(combined_words) if combined_words else STREAMER_NAME
    game_context = f"Game Category: '{game_name}'." if game_name else (f"Game Category: '{DEFAULT_GAME}'." if DEFAULT_GAME else "")
    
    if auto_jargon:
        print(f"🧠 Auto-populated {len(auto_jargon)} game jargon terms for category '{game_name}'")

    prompt = (
        f"You are a Twitch viewer creating a clip for streamer '{STREAMER_NAME}'s live stream.\n"
        f"{game_context}\n"
        f"Key names and jargon: {words_context}.\n"
        "Write a clip title exactly how an authentic Twitch chatter/viewer would name it.\n"
        "Style Guidelines:\n"
        f"- Twitch chatter tone: hype, funny, meme-y, or reacting to the moment (e.g. '{STREAMER_NAME.upper()} NOOO', 'WHAT WAS THAT SHOT', 'bro choked the 1v1', 'actual 200 IQ play').\n"
        f"- Refer to the streamer as '{STREAMER_NAME}' or 'bro' when natural.\n"
        "- Max 6 words (max 45 characters)\n"
        "- Keep profanity clean and family-friendly (Twitch TOS safe)\n"
        "- Capitalize appropriately, no quotes, no ending period\n\n"
        f"Spoken text: \"{raw_text}\"\nTitle:"
    )

    # 1. Try local Ollama AI (http://localhost:11434)
    try:
        payload = {
            "model": "llama3.2", 
            "prompt": prompt, 
            "stream": False,
            "options": {"temperature": 0.0}  # Deterministic output
        }
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=2)
        if resp.status_code == 200:
            ai_title = resp.json().get("response", "").strip().strip('"').strip("'")
            if ai_title and len(ai_title) <= MAX_TITLE_LENGTH:
                ai_title = sanitize_profanity(ai_title)
                print(f"🤖 AI Title (Ollama): {ai_title}")
                return ai_title
    except Exception:
        pass

    # 2. Try OpenAI API if OPENAI_API_KEY environment variable is set
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            headers = {"Authorization": f"Bearer {openai_key}", "Content-Type": "application/json"}
            payload = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 25,
                "temperature": 0.0
            }
            resp = requests.post("https://api.openai.com/v1/chat/completions", json=payload, headers=headers, timeout=2)
            if resp.status_code == 200:
                ai_title = resp.json()["choices"][0]["message"]["content"].strip().strip('"').strip("'")
                if ai_title and len(ai_title) <= MAX_TITLE_LENGTH:
                    ai_title = sanitize_profanity(ai_title)
                    print(f"🤖 AI Title (OpenAI): {ai_title}")
                    return ai_title
        except Exception:
            pass

    return None

last_clip_time = 0
last_clip_title = ""
last_clip_raw_transcript = ""

def transcribe_buffer(duration=30, game_name="", notify=True):
    """Extracts recent audio from buffer for custom duration and returns (title, raw_transcript)."""
    global last_clip_time, last_clip_title, last_clip_raw_transcript
    now = time.time()

    # Auto-detect live Twitch game category if not explicitly provided
    if not game_name or game_name == DEFAULT_GAME:
        game_name = get_live_twitch_game()

    # Title Cache: If re-triggered within 5 seconds, return the exact same title & raw transcript
    if now - last_clip_time < 5 and last_clip_title:
        print(f"🔄 Re-using stable title (cached): \"{last_clip_title}\"")
        return last_clip_title, last_clip_raw_transcript

    audio_data = audio_buffer.get_audio_data(seconds=duration)
    if len(audio_data) < SAMPLE_RATE * 2:
        last_clip_time = time.time()
        last_clip_title = "Stream Highlight"
        last_clip_raw_transcript = "No mic speech detected"
        return last_clip_title, last_clip_raw_transcript

    # Write to a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = tmp_file.name
        with wave.open(tmp_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2) # 16-bit
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_data.tobytes())

    # Build initial prompt hint for Whisper speech-to-text including game jargon
    auto_jargon = get_game_jargon_ai(game_name)
    combined_words = list(dict.fromkeys([STREAMER_NAME] + CUSTOM_WORDS + auto_jargon))
    whisper_prompt = f"Streamer {STREAMER_NAME}, game {game_name}, jargon: " + ", ".join(combined_words[:20])

    try:
        if whisper_model is None:
            return "Stream Highlight", "Whisper model not loaded"
        res = whisper_model.transcribe(tmp_path, initial_prompt=whisper_prompt, fp16=False)
        text = res.get("text", "").strip()
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # Clean up whitespace & profanity
    text = " ".join(text.split())
    if not text:
        last_clip_time = time.time()
        last_clip_title = "Awesome Stream Moment"
        last_clip_raw_transcript = "No clear speech transcribed"
        return last_clip_title, last_clip_raw_transcript

    # Try AI rewriting first!
    final_title = ai_rewrite_title(text, game_name=game_name)
    
    if not final_title:
        # Fallback: Extract last spoken sentence and cap at MAX_TITLE_LENGTH
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        title = sentences[-1] if sentences else text

        if len(title) > MAX_TITLE_LENGTH:
            words = title.split()
            short_title = ""
            for w in words:
                if len(short_title) + len(w) + 1 <= MAX_TITLE_LENGTH - 3:
                    short_title += (" " if short_title else "") + w
                else:
                    break
            final_title = (short_title + "...") if short_title else title[:MAX_TITLE_LENGTH - 3] + "..."
        else:
            final_title = title

    final_title = sanitize_profanity(final_title)
    
    # Save cache & push Notification / Clipboard ONLY if notify=True
    last_clip_time = time.time()
    last_clip_title = final_title
    last_clip_raw_transcript = text
    
    if notify:
        copy_and_notify(final_title)

    # Update Menu Bar App item if running
    if menu_app_instance:
        menu_app_instance.update_last_title(final_title)
    
    return final_title, text

# ----------------- YOUTUBE SHORTS AUTO-UPLOAD -----------------

def get_youtube_service():
    """Gets an authenticated YouTube API service instance using saved tokens or OAuth flow."""
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build

        creds = None
        if os.path.exists(TOKEN_FILE):
            try:
                creds = Credentials.from_authorized_user_file(TOKEN_FILE, YOUTUBE_SCOPES)
            except Exception as e:
                print(f"⚠️ YouTube Token load warning: {e}")

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as e:
                    print(f"⚠️ Token refresh warning: {e}")
                    creds = None

            if not creds:
                if not os.path.exists(CLIENT_SECRET_FILE):
                    print(f"⚠️ YouTube client_secret.json missing at '{CLIENT_SECRET_FILE}'")
                    return None
                print("🔑 Opening browser for 1-time YouTube OAuth Authorization...")
                import webbrowser
                flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, YOUTUBE_SCOPES)
                creds = flow.run_local_server(port=8080, open_browser=True)

            with open(TOKEN_FILE, "w") as token:
                token.write(creds.to_json())
            print("✅ YouTube OAuth token saved successfully!")

        return build("youtube", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ YouTube authentication error: {e}")
        return None

def generate_youtube_metadata(title, raw_transcript="", game_name=""):
    """Generates formatted YouTube Shorts title, description, and hashtags."""
    yt_title = title if "#Shorts" in title or "#shorts" in title else f"{title} #Shorts"
    if len(yt_title) > 100:
        yt_title = yt_title[:95] + "..."

    game_tag = f"#{game_name.replace(' ', '')}" if game_name else (f"#{DEFAULT_GAME.replace(' ', '')}" if DEFAULT_GAME else "#Gaming")
    hashtags = ["#Shorts", game_tag, "#TwitchClips", "#StreamHighlight"]
    
    description = (
        f"Stream highlight from {STREAMER_NAME}'s live stream!\n\n"
        f"🎙️ Spoken: \"{raw_transcript}\"\n\n"
        f"Tags: {' '.join(hashtags)}"
    )
    
    return yt_title, description, [h.strip('#') for h in hashtags]

def find_latest_video_clip(directory=OBS_CLIPS_DIR, max_age_seconds=300):
    """Finds the newest video file (.mp4/.mov/.mkv) created in the specified directory."""
    expanded_dir = os.path.expanduser(directory)
    if not os.path.exists(expanded_dir):
        return None

    video_extensions = ('.mp4', '.mov', '.mkv', '.webm')
    now = time.time()
    latest_file = None
    latest_mtime = 0

    for root, _, files in os.walk(expanded_dir):
        for f in files:
            if f.lower().endswith(video_extensions) and not f.startswith('.'):
                full_path = os.path.join(root, f)
                try:
                    mtime = os.path.getmtime(full_path)
                    if now - mtime <= max_age_seconds and mtime > latest_mtime:
                        latest_mtime = mtime
                        latest_file = full_path
                except OSError:
                    continue

    return latest_file

def upload_short_to_youtube_async(video_path, title, raw_transcript="", game_name=""):
    """Background worker function that uploads a video to YouTube Shorts."""
    def worker():
        if not video_path or not os.path.exists(video_path):
            print(f"⚠️ YouTube upload skipped: Video file not found at '{video_path}'")
            return

        print(f"🚀 Launching background YouTube upload for video: '{video_path}'...")
        if menu_app_instance:
            menu_app_instance.update_status_title("⏳ Uploading Short...")

        try:
            from googleapiclient.http import MediaFileUpload

            service = get_youtube_service()
            if not service:
                print("❌ YouTube upload failed: Could not authenticate YouTube service.")
                if menu_app_instance:
                    menu_app_instance.update_status_title("🎙️ Clip Backtrack")
                return

            yt_title, description, tags = generate_youtube_metadata(title, raw_transcript, game_name)

            body = {
                "snippet": {
                    "title": yt_title,
                    "description": description,
                    "tags": tags,
                    "categoryId": "20"
                },
                "status": {
                    "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                    "selfDeclaredMadeForKids": False
                }
            }

            # 1 MB chunk size for controlled rate limiting
            chunk_bytes = 1024 * 1024
            media = MediaFileUpload(video_path, chunksize=chunk_bytes, resumable=True)
            request = service.videos().insert(part="snippet,status", body=body, media_body=media)

            response = None
            print(f"⏳ Uploading to YouTube with rate limit of {MAX_UPLOAD_KBPS} KB/s...")
            
            while response is None:
                start_time = time.time()
                status, response = request.next_chunk()
                if status:
                    progress_pct = int(status.progress() * 100)
                    print(f"⏳ YouTube Upload Progress: {progress_pct}%")

                # Bandwidth Rate Limiter: Calculate required sleep to enforce MAX_UPLOAD_KBPS
                elapsed = time.time() - start_time
                target_time = chunk_bytes / (MAX_UPLOAD_KBPS * 1024)
                if elapsed < target_time:
                    time.sleep(target_time - elapsed)

            video_id = response.get("id")
            video_url = f"https://youtu.be/{video_id}"
            print(f"✅ YouTube Short Published! URL: {video_url}")

            # Notify user & copy URL
            if ENABLE_NOTIFICATIONS:
                escaped_url = video_url.replace('"', '\\"')
                applescript = f'display notification "{escaped_url}" with title "🚀 YouTube Short Published!" subtitle "{yt_title}"'
                subprocess.run(['osascript', '-e', applescript], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            if ENABLE_CLIPBOARD_COPY:
                try:
                    proc = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE)
                    proc.communicate(input=video_url.encode('utf-8'))
                except Exception:
                    pass

            if menu_app_instance:
                menu_app_instance.update_status_title("✅ Short Published!")
                # Reset status back after 10s
                threading.Timer(10.0, lambda: menu_app_instance.update_status_title("🎙️ Clip Backtrack")).start()

        except Exception as e:
            print(f"❌ YouTube upload error: {e}")
            if menu_app_instance:
                menu_app_instance.update_status_title("🎙️ Clip Backtrack")

    threading.Thread(target=worker, daemon=True).start()


def send_to_aitum(title, var_name=None):
    """Sends the transcribed clip title to Aitum Nexus API."""
    target_var = var_name if var_name else AITUM_GLOBAL_VAR
    print(f"📌 Generated Title: \"{title}\" (Target Aitum Variable: '{target_var}')")
    
    def normalize_var(name):
        return str(name).strip().lower().replace("_", " ")

    norm_target = normalize_var(target_var)

    try:
        # 1. Query Aitum Nexus state variables to find matching _id
        resp = requests.get("http://localhost:7777/aitum/state", timeout=1)
        if resp.status_code == 200:
            variables = resp.json().get("data", [])
            target_id = None
            matched_name = target_var

            # 1a. Exact / Normalized match
            for v in variables:
                v_name = v.get("name", "")
                if normalize_var(v_name) == norm_target:
                    target_id = v.get("_id")
                    matched_name = v_name
                    break

            # 1b. Fallback match (matches "Clip Title", "clip_title", "Clipt Title", etc.)
            if not target_id:
                for v in variables:
                    v_name = v.get("name", "")
                    v_norm = normalize_var(v_name)
                    if "clip" in v_norm and "title" in v_norm:
                        target_id = v.get("_id")
                        matched_name = v_name
                        break

            if target_id:
                # 2. PUT update to Aitum Nexus state variable ID
                update_url = f"http://localhost:7777/aitum/state/{target_id}"
                put_resp = requests.put(update_url, json={"value": title}, timeout=1)
                if put_resp.status_code == 200:
                    print(f"🟢 Aitum Variable '{matched_name}' (ID: {target_id}) updated successfully: \"{title}\"")
                    return
            else:
                print(f"⚠️ Aitum Variable '{target_var}' not found in Aitum Nexus. (Available: {[v.get('name') for v in variables]})")
    except Exception as e:
        print(f"⚠️ Aitum Nexus API notice: {e}")

from flask import Flask, request, jsonify, render_template_string
from waitress import serve
import logging

flask_app = Flask(__name__)
logging.getLogger('waitress').setLevel(logging.ERROR)

@flask_app.route('/favicon.ico', methods=['GET'])
def favicon():
    return '', 204

def render_unified_dashboard(new_title=None, new_transcript=None, new_duration=None):
    """Renders the single unified Dashboard, Settings, and Clip Control Center webpage."""
    live_game = get_live_twitch_game()
    live_jargon_str = ", ".join(current_detected_jargon) if current_detected_jargon else "None loaded"

    display_title = new_title if new_title else (last_clip_title if last_clip_title else "No clip generated yet")
    display_transcript = new_transcript if new_transcript else (last_clip_raw_transcript if last_clip_raw_transcript else "Speak into your mic and trigger a clip!")
    display_duration = new_duration if new_duration else 30

    html_page = """<!DOCTYPE html>
<html>
<head>
    <title>🎬 Clip Backtrack - Dashboard & Settings</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0b0c10; color: #e2e8f0; margin: 0; padding: 30px 15px; }
        .container { max-width: 660px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
        
        .header { display: flex; align-items: center; justify-content: space-between; background: #13151c; padding: 18px 24px; border-radius: 18px; border: 1px solid #232736; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
        .header h1 { font-size: 20px; color: #a78bfa; margin: 0; display: flex; align-items: center; gap: 10px; }

        .hero-card { background: linear-gradient(135deg, #1e1b4b, #0f172a); border: 1px solid #4338ca; border-radius: 20px; padding: 25px; box-shadow: 0 15px 35px rgba(0,0,0,0.6); }
        .hero-title { font-size: 13px; font-weight: 700; color: #818cf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px; }
        .title-box { font-size: 22px; font-weight: 800; color: #38bdf8; background: #0b0f19; padding: 16px 20px; border-radius: 12px; border: 1px solid #1e293b; margin-bottom: 12px; word-break: break-word; display: flex; align-items: center; justify-content: space-between; }
        .transcript-box { font-size: 13px; color: #94a3b8; font-style: italic; background: #111827; padding: 12px 16px; border-radius: 10px; border: 1px solid #1f2937; margin-bottom: 15px; line-height: 1.5; }
        
        .trigger-btn { background: linear-gradient(135deg, #8b5cf6, #ec4899); color: white; font-size: 16px; font-weight: 800; border: none; padding: 14px 24px; border-radius: 12px; cursor: pointer; width: 100%; box-shadow: 0 8px 20px rgba(139,92,246,0.4); transition: transform 0.15s, opacity 0.15s; }
        .trigger-btn:hover { transform: translateY(-2px); opacity: 0.95; }
        
        .section { background: #13151c; padding: 22px; border-radius: 18px; border: 1px solid #232736; }
        .section-title { font-size: 13px; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 15px; }
        
        .field { margin-bottom: 15px; display: flex; flex-direction: column; gap: 6px; text-align: left; }
        .field-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
        label { font-size: 13px; font-weight: 600; color: #94a3b8; }
        input[type="text"], input[type="number"], select { background: #0b0c10; border: 1px solid #334155; border-radius: 8px; color: #f8fafc; padding: 10px 14px; font-size: 14px; width: 100%; box-sizing: border-box; }
        input[type="checkbox"] { width: 20px; height: 20px; cursor: pointer; accent-color: #8b5cf6; }
        
        .save-btn { background: linear-gradient(135deg, #6366f1, #3b82f6); border: none; color: white; padding: 14px 24px; border-radius: 12px; font-size: 15px; font-weight: 700; cursor: pointer; width: 100%; margin-top: 10px; }
        .save-btn:hover { opacity: 0.9; }
        .auth-btn { background: #0284c7; margin-top: 8px; font-size: 13px; padding: 10px; border: none; color: white; border-radius: 8px; font-weight: 700; cursor: pointer; width: 100%; }
        
        .status-banner { display: none; background: #065f46; color: #34d399; padding: 12px; border-radius: 10px; font-weight: 600; text-align: center; margin-bottom: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎬 Clip Backtrack Dashboard</h1>
            <span style="font-size: 12px; font-weight: 700; color: #34d399; background: #064e3b; padding: 5px 12px; border-radius: 20px;">● Live</span>
        </div>

        <div id="status" class="status-banner">✅ Settings Saved Successfully!</div>

        <!-- HERO CARD: LIVE CLIP OUTPUT & TRIGGER -->
        <div class="hero-card">
            <div class="hero-title">🎙️ Continuous Live Real-Time Captions</div>
            <div class="transcript-box" id="liveRealtimeCaption" style="font-size: 15px; font-weight: 600; color: #38bdf8; background: #0f172a; border: 1px solid #1e3a8a; font-style: normal; margin-bottom: 20px;">"__DISPLAY_TRANSCRIPT__"</div>

            <div class="hero-title">📌 Latest Generated Clip Title</div>
            <div class="title-box">
                <span id="activeTitle">"__DISPLAY_TITLE__"</span>
                <button style="width: auto; margin:0; padding: 6px 12px; font-size: 12px; background: #334155; border: none; color: white; border-radius: 6px; cursor: pointer;" onclick="copyActiveTitle()">📋 Copy</button>
            </div>
            <button class="trigger-btn" onclick="triggerClipAjax()" style="margin-top: 15px;">✂️ Trigger Clip Now (30s Backtrack)</button>
        </div>

        <!-- LIVE STREAM STATUS CARD -->
        <div class="section" style="border: 1px solid #3b82f6; background: #0f172a;">
            <div class="section-title" style="color: #60a5fa;">🎮 Active Live Stream Status</div>
            <div class="field-row">
                <label>Detected Category:</label>
                <span style="font-weight: 700; color: #38bdf8; font-size: 15px;">__LIVE_GAME__</span>
            </div>
            <div class="field">
                <label>Active AI Jargon Terms:</label>
                <div style="font-size: 12px; color: #cbd5e1; background: #1e293b; padding: 10px; border-radius: 8px; font-family: monospace; line-height: 1.5;">
                    __LIVE_JARGON__
                </div>
            </div>
        </div>

        <!-- SETTINGS FORM -->
        <form id="settingsForm" onsubmit="event.preventDefault(); saveSettingsAjax();">
            <div class="section">
                <div class="section-title">🎙️ Streamer & Audio Input Context</div>
                <div class="field">
                    <label>Microphone Audio Input Device:</label>
                    <select id="mic_device" name="mic_device">
                        __MIC_OPTIONS__
                    </select>
                </div>
                <div class="field">
                    <label>🎙️ Live Microphone Input Test (VU Level Meter):</label>
                    <div style="background: #0b0c10; border: 1px solid #334155; border-radius: 10px; padding: 12px; display: flex; align-items: center; gap: 12px;">
                        <div style="flex-grow: 1; height: 16px; background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155;">
                            <div id="micMeterBar" style="width: 0%; height: 100%; background: linear-gradient(90deg, #10b981, #f59e0b, #ef4444); transition: width 0.1s ease-out; border-radius: 8px;"></div>
                        </div>
                        <span id="micMeterPercent" style="font-size: 13px; font-weight: 700; color: #38bdf8; min-width: 45px; text-align: right;">0%</span>
                    </div>
                </div>
                <div class="field">
                    <label>Streamer Display Name:</label>
                    <input type="text" id="streamer_name" name="streamer_name" value="__STREAMER_NAME__">
                </div>
                <div class="field">
                    <label>Twitch Channel Username (for Live Category Auto-Detection):</label>
                    <input type="text" id="twitch_channel" name="twitch_channel" value="__TWITCH_CHANNEL__" placeholder="e.g. mesutkaya">
                </div>
                <div class="field">
                    <label>Custom Words / Jargon (comma separated):</label>
                    <input type="text" id="custom_words" name="custom_words" value="__CUSTOM_WORDS__">
                </div>
                <div class="field">
                    <label>Default Game Fallback Category:</label>
                    <input type="text" id="default_game" name="default_game" value="__DEFAULT_GAME__">
                </div>
            </div>

            <div class="section">
                <div class="section-title">🚀 YouTube Shorts Auto-Upload</div>
                <div class="field-row">
                    <label for="enable_yt">Enable YouTube Auto-Upload:</label>
                    <input type="checkbox" id="enable_yt" name="enable_yt" __ENABLE_YT_CHECKED__>
                </div>
                <div class="field">
                    <label>Google OAuth Client ID:</label>
                    <input type="text" id="google_client_id" name="google_client_id" value="__GOOGLE_CLIENT_ID__" placeholder="xxxx.apps.googleusercontent.com">
                </div>
                <div class="field">
                    <label>Google OAuth Client Secret:</label>
                    <input type="password" id="google_client_secret" name="google_client_secret" value="" placeholder="__SECRET_PLACEHOLDER__" autocomplete="off">
                </div>
                <div class="field">
                    <label>Upload Privacy Status:</label>
                    <select id="yt_privacy" name="yt_privacy">
                        <option value="public" __PRIV_PUBLIC__>Public (Immediate Publish)</option>
                        <option value="unlisted" __PRIV_UNLISTED__>Unlisted (Private Review Link)</option>
                        <option value="private" __PRIV_PRIVATE__>Private</option>
                    </select>
                </div>
                <div class="field">
                    <label>Upload Bandwidth Speed Limit (KB/s):</label>
                    <input type="number" id="max_upload_kbps" name="max_upload_kbps" value="__MAX_UPLOAD_KBPS__">
                </div>
                <div class="field">
                    <label>OBS Clips Folder Directory:</label>
                    <input type="text" id="obs_clips_dir" name="obs_clips_dir" value="__OBS_CLIPS_DIR__">
                </div>
                <button type="button" class="auth-btn" onclick="triggerAuth()">🔑 Authenticate YouTube Account (Google OAuth)</button>
            </div>

            <div class="section">
                <div class="section-title">🔔 App Preferences</div>
                <div class="field-row">
                    <label for="enable_notif">macOS Desktop Notifications:</label>
                    <input type="checkbox" id="enable_notif" name="enable_notif" __ENABLE_NOTIF_CHECKED__>
                </div>
                <div class="field-row">
                    <label for="enable_clip">Auto-Copy Title to Clipboard:</label>
                    <input type="checkbox" id="enable_clip" name="enable_clip" __ENABLE_CLIP_CHECKED__>
                </div>
                <div class="field-row">
                    <label for="profanity_filter">Twitch TOS Profanity Filter:</label>
                    <input type="checkbox" id="profanity_filter" name="profanity_filter" __PROFANITY_CHECKED__>
                </div>
            </div>

            <button type="button" onclick="saveSettingsAjax()" class="save-btn">💾 Save Settings</button>
        </form>
    </div>

    <script>
        // High-frequency polling for smooth mic VU meter and real-time live captions
        setInterval(async () => {
            try {
                const resp = await fetch('/api/live_transcript');
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.live_transcript) {
                        const captionEl = document.getElementById('liveRealtimeCaption');
                        if (captionEl) {
                            captionEl.innerText = '"' + data.live_transcript + '"';
                        }
                    }
                    if (data.mic_volume !== undefined) {
                        const rawVol = data.mic_volume;
                        let pct = 0;
                        const noiseFloor = 0.0003;
                        if (rawVol > noiseFloor) {
                            pct = Math.min(100, Math.max(5, Math.round(Math.log10(rawVol / noiseFloor) * 40)));
                        }
                        const bar = document.getElementById('micMeterBar');
                        const txt = document.getElementById('micMeterPercent');
                        if (bar) bar.style.width = pct + '%';
                        if (txt) txt.innerText = pct + '%';
                    }
                }
            } catch (e) {}
        }, 120);

        async function saveSettingsAjax() {
            const data = {
                streamer_name: document.getElementById('streamer_name').value,
                mic_device: document.getElementById('mic_device').value,
                twitch_channel: document.getElementById('twitch_channel').value,
                custom_words: document.getElementById('custom_words').value.split(',').map(s => s.trim()).filter(Boolean),
                default_game: document.getElementById('default_game').value,
                enable_yt: document.getElementById('enable_yt').checked,
                google_client_id: document.getElementById('google_client_id').value,
                google_client_secret: document.getElementById('google_client_secret').value,
                yt_privacy: document.getElementById('yt_privacy').value,
                max_upload_kbps: parseInt(document.getElementById('max_upload_kbps').value) || 1500,
                obs_clips_dir: document.getElementById('obs_clips_dir').value,
                enable_notif: document.getElementById('enable_notif').checked,
                enable_clip: document.getElementById('enable_clip').checked,
                profanity_filter: document.getElementById('profanity_filter').checked
            };
            try {
                const resp = await fetch('/api/settings', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(data)
                });
                if (resp.ok) {
                    const status = document.getElementById('status');
                    status.style.display = 'block';
                    setTimeout(() => status.style.display = 'none', 3000);
                }
            } catch (e) {
                console.error(e);
            }
        }

        async function triggerClipAjax() {
            const btn = document.querySelector('.trigger-btn');
            if (btn) {
                btn.innerText = "⏳ Processing Clip Audio...";
                btn.style.opacity = "0.7";
            }
            try {
                const resp = await fetch('/clip?duration=30&trigger=true&json=true');
                const data = await resp.json();
                if (data.title) {
                    const titleEl = document.getElementById('activeTitle');
                    if (titleEl) titleEl.innerText = '"' + data.title + '"';
                    const capEl = document.getElementById('liveRealtimeCaption');
                    if (capEl && data.raw_transcript) capEl.innerText = '"' + data.raw_transcript + '"';
                }
            } catch (e) {
                console.error("Clip trigger error:", e);
            }
            if (btn) {
                btn.innerText = "✂️ Trigger Clip Now (30s Backtrack)";
                btn.style.opacity = "1";
            }
        }

        function copyActiveTitle() {
            const text = document.getElementById('activeTitle').innerText.replace(/^"|"$/g, '');
            navigator.clipboard.writeText(text);
            alert("📋 Copied to Clipboard: " + text);
        }

        document.getElementById('settingsForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const data = {
                streamer_name: document.getElementById('streamer_name').value,
                mic_device: document.getElementById('mic_device').value,
                twitch_channel: document.getElementById('twitch_channel').value,
                custom_words: document.getElementById('custom_words').value.split(',').map(s => s.trim()).filter(Boolean),
                default_game: document.getElementById('default_game').value,
                enable_yt: document.getElementById('enable_yt').checked,
                google_client_id: document.getElementById('google_client_id').value,
                google_client_secret: document.getElementById('google_client_secret').value,
                yt_privacy: document.getElementById('yt_privacy').value,
                max_upload_kbps: parseInt(document.getElementById('max_upload_kbps').value) || 1500,
                obs_clips_dir: document.getElementById('obs_clips_dir').value,
                enable_notif: document.getElementById('enable_notif').checked,
                enable_clip: document.getElementById('enable_clip').checked,
                profanity_filter: document.getElementById('profanity_filter').checked
            };
            const resp = await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            if (resp.ok) {
                const banner = document.getElementById('status');
                banner.style.display = 'block';
                setTimeout(() => banner.style.display = 'none', 3000);
            }
        });

        async function triggerAuth() {
            const resp = await fetch('/auth_youtube');
            const data = await resp.json();
            if (data.status === "missing_secret") {
                alert("⚠️ YouTube client_secret.json missing!\n\nPlease enter your Google OAuth Client ID and Secret in the fields above, click Save Settings, and click Authenticate again!");
            } else {
                alert("🔑 Google OAuth server started!\n\nOpening login page in your default browser. Please complete authentication there.");
            }
        }
    </script>
</body>
</html>"""
    def esc(v):
        return html.escape(str(v), quote=True)

    devices = get_audio_devices()
    mic_options = []
    for d in devices:
        selected = "selected" if MICROPHONE_DEVICE and MICROPHONE_DEVICE.lower() in d['name'].lower() else ""
        name = esc(d["name"])
        mic_options.append(f'<option value="{name}" {selected}>{name}</option>')
    mic_options_str = "\n".join(mic_options) if mic_options else '<option value="">Default Input</option>'

    rendered = html_page.replace("__DISPLAY_TITLE__", esc(display_title))\
                   .replace("__DISPLAY_TRANSCRIPT__", esc(display_transcript))\
                   .replace("__MIC_OPTIONS__", mic_options_str)\
                   .replace("__LIVE_GAME__", esc(live_game))\
                   .replace("__LIVE_JARGON__", esc(live_jargon_str))\
                   .replace("__STREAMER_NAME__", esc(STREAMER_NAME))\
                   .replace("__TWITCH_CHANNEL__", esc(TWITCH_CHANNEL))\
                   .replace("__CUSTOM_WORDS__", esc(", ".join(CUSTOM_WORDS)))\
                   .replace("__DEFAULT_GAME__", esc(DEFAULT_GAME))\
                   .replace("__ENABLE_YT_CHECKED__", "checked" if ENABLE_YOUTUBE_UPLOAD else "")\
                   .replace("__GOOGLE_CLIENT_ID__", esc(GOOGLE_CLIENT_ID))\
                   .replace("__SECRET_PLACEHOLDER__", "•••••• saved" if GOOGLE_CLIENT_SECRET else "GOCSPX-xxxx...")\
                   .replace("__PRIV_PUBLIC__", "selected" if YOUTUBE_PRIVACY_STATUS == "public" else "")\
                   .replace("__PRIV_UNLISTED__", "selected" if YOUTUBE_PRIVACY_STATUS == "unlisted" else "")\
                   .replace("__PRIV_PRIVATE__", "selected" if YOUTUBE_PRIVACY_STATUS == "private" else "")\
                   .replace("__MAX_UPLOAD_KBPS__", str(MAX_UPLOAD_KBPS))\
                   .replace("__OBS_CLIPS_DIR__", esc(OBS_CLIPS_DIR))\
                   .replace("__ENABLE_NOTIF_CHECKED__", "checked" if ENABLE_NOTIFICATIONS else "")\
                   .replace("__ENABLE_CLIP_CHECKED__", "checked" if ENABLE_CLIPBOARD_COPY else "")\
                   .replace("__PROFANITY_CHECKED__", "checked" if PROFANITY_FILTER else "")
    return rendered, 200, {'Content-Type': 'text/html; charset=utf-8'}

@flask_app.route('/', methods=['GET'])
@flask_app.route('/dashboard', methods=['GET'])
@flask_app.route('/settings', methods=['GET'])
def handle_unified_dashboard():
    return render_unified_dashboard()

@flask_app.route('/clip', methods=['GET'])
def handle_clip():
    try:
        # Check if this is an explicit trigger call (e.g. from button, hotkey, or trigger parameter)
        is_explicit_trigger = request.args.get('trigger', '') == 'true' or request.args.get('json', '') == 'true'

        duration_param = request.args.get('duration', DEFAULT_CLIP_SECONDS)
        try:
            duration = int(duration_param)
            duration = max(5, min(duration, MAX_BUFFER_SECONDS))
        except ValueError:
            duration = DEFAULT_CLIP_SECONDS

        game_name = request.args.get('game', DEFAULT_GAME)
        var_name = request.args.get('variable', AITUM_GLOBAL_VAR)

        if is_explicit_trigger:
            print(f"\n🎬 Explicit Clip Triggered! (Duration: {duration}s, Game: '{game_name}')")
            title, raw_transcript = transcribe_buffer(duration=duration, game_name=game_name, notify=True)
            send_to_aitum(title, var_name=var_name)

            video_path = request.args.get('video_path', '')
            if not video_path and ENABLE_YOUTUBE_UPLOAD:
                video_path = find_latest_video_clip()

            if video_path and ENABLE_YOUTUBE_UPLOAD:
                upload_short_to_youtube_async(video_path, title, raw_transcript=raw_transcript, game_name=game_name)

            is_json = request.args.get('json', '') == 'true'
            accept_header = request.headers.get('Accept', '')

            if is_json or ('application/json' in accept_header and 'text/html' not in accept_header):
                return jsonify({
                    "status": "success",
                    "title": title,
                    "raw_transcript": raw_transcript,
                    "duration": duration
                })
            else:
                return render_unified_dashboard(new_title=title, new_transcript=raw_transcript, new_duration=duration)
        else:
            # Passive URL load: Just render the unified dashboard without triggering notification/clipboard
            return render_unified_dashboard()
    except Exception as e:
        print(f"⚠️ Error processing clip: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@flask_app.route('/api/live_transcript', methods=['GET'])
def handle_live_transcript():
    return jsonify({
        "live_transcript": live_realtime_transcript,
        "mic_volume": current_mic_volume
    })

@flask_app.route('/api/settings', methods=['GET', 'POST'])
def handle_api_settings():
    global STREAMER_NAME, TWITCH_CHANNEL, CUSTOM_WORDS, DEFAULT_GAME, ENABLE_YOUTUBE_UPLOAD
    global YOUTUBE_PRIVACY_STATUS, MAX_UPLOAD_KBPS, OBS_CLIPS_DIR
    global ENABLE_NOTIFICATIONS, ENABLE_CLIPBOARD_COPY, PROFANITY_FILTER
    global GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET
    global GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, MICROPHONE_DEVICE

    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if "streamer_name" in data:
            STREAMER_NAME = str(data["streamer_name"])
        if "mic_device" in data and str(data["mic_device"]) != MICROPHONE_DEVICE:
            MICROPHONE_DEVICE = str(data["mic_device"])
            print(f"🎙️ Switching Microphone device to '{MICROPHONE_DEVICE}'...")
            threading.Thread(target=start_audio_stream, daemon=True).start()
        if "twitch_channel" in data:
            TWITCH_CHANNEL = str(data["twitch_channel"])
        if "custom_words" in data and isinstance(data["custom_words"], list):
            CUSTOM_WORDS = [str(w) for w in data["custom_words"]]
        if "default_game" in data:
            DEFAULT_GAME = str(data["default_game"])
        if "enable_yt" in data:
            ENABLE_YOUTUBE_UPLOAD = bool(data["enable_yt"])
        if "google_client_id" in data or "google_client_secret" in data:
            cid = str(data.get("google_client_id", ""))
            csec = str(data.get("google_client_secret", ""))
            if cid and csec:
                save_client_secret_json(cid, csec)
        if "yt_privacy" in data:
            YOUTUBE_PRIVACY_STATUS = str(data["yt_privacy"])
        if "max_upload_kbps" in data:
            MAX_UPLOAD_KBPS = int(data["max_upload_kbps"])
        if "obs_clips_dir" in data:
            OBS_CLIPS_DIR = os.path.expanduser(str(data["obs_clips_dir"]))
        if "enable_notif" in data:
            ENABLE_NOTIFICATIONS = bool(data["enable_notif"])
        if "enable_clip" in data:
            ENABLE_CLIPBOARD_COPY = bool(data["enable_clip"])
        if "profanity_filter" in data:
            PROFANITY_FILTER = bool(data["profanity_filter"])

        save_config()
        print(f"⚙️ Settings saved & updated live! Active mic: '{MICROPHONE_DEVICE}'")
        return jsonify({"status": "success"})

    return jsonify({
        "current_game": current_detected_game,
        "current_jargon": current_detected_jargon,
        "streamer_name": STREAMER_NAME,
        "twitch_channel": TWITCH_CHANNEL,
        "custom_words": CUSTOM_WORDS,
        "default_game": DEFAULT_GAME,
        "enable_yt": ENABLE_YOUTUBE_UPLOAD,
        "google_client_id": GOOGLE_CLIENT_ID,
        "google_client_secret_set": bool(GOOGLE_CLIENT_SECRET),
        "yt_privacy": YOUTUBE_PRIVACY_STATUS,
        "max_upload_kbps": MAX_UPLOAD_KBPS,
        "obs_clips_dir": OBS_CLIPS_DIR,
        "enable_notif": ENABLE_NOTIFICATIONS,
        "enable_clip": ENABLE_CLIPBOARD_COPY,
        "profanity_filter": PROFANITY_FILTER
    })

@flask_app.route('/pause', methods=['GET'])
def handle_pause():
    global RECORDING_PAUSED
    RECORDING_PAUSED = True
    print("⏸️ Mic recording PAUSED.")
    return jsonify({"status": "paused"})

@flask_app.route('/resume', methods=['GET'])
def handle_resume():
    global RECORDING_PAUSED
    RECORDING_PAUSED = False
    print("▶️ Mic recording RESUMED.")
    return jsonify({"status": "recording"})

@flask_app.route('/auth_youtube', methods=['GET'])
def handle_auth_youtube():
    if not os.path.exists(CLIENT_SECRET_FILE):
        return jsonify({"status": "missing_secret"})
    threading.Thread(target=get_youtube_service, daemon=True).start()
    return jsonify({"status": "started"})

def start_http_server():
    print(f"🌐 WSGI Production Server listening on http://localhost:{HTTP_PORT}/clip")
    serve(flask_app, host='127.0.0.1', port=HTTP_PORT, _quiet=True)

if __name__ == "__main__":
    print("=" * 60)
    print("   🎙️  Pro macOS Clip Backtrack & AI Titler Daemon")
    print("=" * 60)
    
    # Start HTTP server thread FIRST so port 5001 is listening instantly on app launch!
    server_thread = threading.Thread(target=start_http_server, daemon=True)
    server_thread.start()

    stream = start_audio_stream()
    load_model()

    # Start Continuous Live Real-Time Transcription thread
    live_thread = threading.Thread(target=live_transcription_loop, daemon=True)
    live_thread.start()

    print("\nREADY!")
    print(f"👉 Trigger: http://localhost:{HTTP_PORT}/clip")
    print(f"👉 Optional Params: http://localhost:{HTTP_PORT}/clip?duration=15&game=Valorant")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping transcript backtrack...")
        stream.stop()
        sys.exit(0)
