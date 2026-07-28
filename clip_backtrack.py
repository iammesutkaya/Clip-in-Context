#!/usr/bin/env python3
"""
clip_backtrack.py — rolling mic transcript backtrack + AI clip titler (macOS).

One terminal-launched process. That is the whole design decision: launched from
the terminal (or a LaunchAgent) it inherits microphone permission, so there is
no .app bundle, no code signing, and no TCC silence. Menu bar via rumps.

    mic → rolling 30s buffer → (trigger) → MLX Whisper → AI title
        → clipboard + notification → Aitum → optional YouTube Short

Trigger: menu bar item, or HTTP  GET http://localhost:5001/clip?duration=30&game=Valorant
Run:     python3 clip_backtrack.py
"""
import os, re, sys, json, math, time, threading, subprocess, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
import sounddevice as sd
from scipy import signal
import mlx_whisper
import requests
import rumps

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "config.json")
TOKEN_FILE = os.path.join(HERE, "youtube_token.json")
CLIENT_SECRET_FILE = os.path.join(HERE, "client_secret.json")
YOUTUBE_SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# ---------------- config (persisted to config.json) ----------------
cfg = {
    "streamer_name": "Mesut",
    "twitch_channel": "mesutkaya",
    "custom_words": [],
    "default_game": "",
    "mic_device": "Voice - Input 01",   # substring match; "" = system default
    "enable_yt": False,
    "yt_privacy": "unlisted",
    "max_upload_kbps": 1500,
    "obs_clips_dir": "~/Movies",
    "enable_notif": True,
    "enable_clip": True,
    "google_client_id": "",
    "google_client_secret": "",
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception as e:
            print(f"⚠️ config load: {e}")

def save_config():
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"⚠️ config save: {e}")

load_config()

# ---------------- audio ----------------
SAMPLE_RATE = 16000       # Whisper wants 16 kHz
BUFFER_SECONDS = 60       # ring buffer length
DEFAULT_CLIP_SECONDS = 30
HTTP_PORT = 5001
MODEL = "mlx-community/whisper-base.en-mlx"
MAX_TITLE_LENGTH = 50

recording_paused = False
mic_volume = 0.0

class RingBuffer:
    """Fixed float32 ring buffer, last BUFFER_SECONDS seconds at 16 kHz."""
    def __init__(self, seconds):
        self.cap = SAMPLE_RATE * seconds
        self.buf = np.zeros(self.cap, dtype=np.float32)
        self.write = 0
        self.filled = 0
        self.lock = threading.Lock()

    def add(self, samples):
        n = samples.size
        if n == 0:
            return
        if n >= self.cap:
            samples, n = samples[-self.cap:], self.cap
        with self.lock:
            end = self.write + n
            if end <= self.cap:
                self.buf[self.write:end] = samples
            else:
                split = self.cap - self.write
                self.buf[self.write:] = samples[:split]
                self.buf[:n - split] = samples[split:]
            self.write = (self.write + n) % self.cap
            self.filled = min(self.cap, self.filled + n)

    def last(self, seconds):
        with self.lock:
            want = min(self.filled, SAMPLE_RATE * seconds)
            if want == 0:
                return np.array([], dtype=np.float32)
            start = (self.write - want) % self.cap
            if start + want <= self.cap:
                return self.buf[start:start + want].copy()
            split = self.cap - start
            return np.concatenate((self.buf[start:], self.buf[:want - split]))

ring = RingBuffer(BUFFER_SECONDS)
_stream = None
_stream_rate = SAMPLE_RATE
_resample_g = 1

def _callback(indata, frames, t, status):
    global mic_volume
    if status:
        print(f"audio: {status}", file=sys.stderr)
    if recording_paused:
        return
    mono = indata.mean(axis=1) if indata.ndim > 1 else indata.ravel()
    mic_volume = float(np.max(np.abs(mono))) if mono.size else 0.0
    if _stream_rate != SAMPLE_RATE:
        mono = signal.resample_poly(mono, SAMPLE_RATE // _resample_g, _stream_rate // _resample_g)
    ring.add(mono.astype(np.float32))

def input_devices():
    return [{"id": i, "name": d["name"], "ch": d["max_input_channels"], "sr": int(d["default_samplerate"])}
            for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0]

def start_stream():
    global _stream, _stream_rate, _resample_g
    if _stream:
        try:
            _stream.stop(); _stream.close()
        except Exception:
            pass
    devs = input_devices()
    dev = next((d for d in devs if cfg["mic_device"] and cfg["mic_device"].lower() in d["name"].lower()), None)
    if dev is None:
        idx = sd.default.device[0]
        dev = next((d for d in devs if d["id"] == idx), devs[0] if devs else None)
    if dev is None:
        print("⚠️ no input device"); return
    _stream_rate = dev["sr"] or SAMPLE_RATE
    _resample_g = math.gcd(_stream_rate, SAMPLE_RATE)
    chans = min(2, dev["ch"])
    print(f"🎙️  {dev['name']} ({chans}ch @ {_stream_rate}Hz → {SAMPLE_RATE}Hz)")
    _stream = sd.InputStream(device=dev["id"], samplerate=_stream_rate, channels=chans,
                             dtype="float32", callback=_callback)
    _stream.start()

# ---------------- twitch category + jargon (title context) ----------------
detected_game = ""
JARGON_CACHE = os.path.join(HERE, "jargon_cache.json")
_jargon = {}
if os.path.exists(JARGON_CACHE):
    try:
        _jargon = json.load(open(JARGON_CACHE, encoding="utf-8"))
    except Exception:
        _jargon = {}

def game_jargon(game):
    if not game or len(game) < 2:
        return []
    key = game.strip().lower()
    if key in _jargon:
        return _jargon[key]
    try:
        r = requests.post("http://localhost:11434/api/generate", timeout=4, json={
            "model": "llama3.2", "stream": False, "options": {"temperature": 0.2},
            "prompt": f"Output 15 comma-separated key characters/items/jargon for the game '{game}'. Only the words."})
        if r.status_code == 200:
            terms = [t.strip().strip('"\'') for t in r.json().get("response", "").split(",") if 0 < len(t.strip()) < 30]
            if terms:
                _jargon[key] = terms
                json.dump(_jargon, open(JARGON_CACHE, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
                return terms
    except Exception:
        pass
    return []

def live_twitch_game():
    global detected_game
    ch = cfg["twitch_channel"] or cfg["streamer_name"]
    if ch:
        try:
            r = requests.get(f"https://decapi.me/twitch/game/{urllib.parse.quote(ch)}", timeout=2)
            g = r.text.strip()
            if r.status_code == 200 and g and "error" not in g.lower() and "not found" not in g.lower():
                detected_game = g
                return g
        except Exception:
            pass
    detected_game = cfg["default_game"] or "Just Chatting"
    return detected_game

# ---------------- AI title ----------------
BAD = {r'\bfuck(ing|er|ed)?\b': 'f***', r'\bshit(ting|ty)?\b': 's***',
       r'\bbitch(es)?\b': 'b****', r'\basshole\b': 'a**hole', r'\bcunt\b': 'c***'}

def clean(text):
    for p, r in BAD.items():
        text = re.sub(p, r, text, flags=re.IGNORECASE)
    return text

def ai_title(raw, game=""):
    if not raw or len(raw) < 5:
        return None
    jargon = game_jargon(game)
    words = ", ".join(dict.fromkeys(cfg["custom_words"] + jargon)) or cfg["streamer_name"]
    prompt = (
        f"You are a Twitch viewer clipping streamer '{cfg['streamer_name']}'s stream.\n"
        f"{'Game: ' + game if game else ''}\nKey names/jargon: {words}.\n"
        "Write the clip title exactly how a hype/funny Twitch chatter would.\n"
        "- Max 6 words (45 chars). Family-friendly. No quotes, no ending period.\n"
        f'Spoken: "{raw}"\nTitle:')
    # Ollama
    try:
        r = requests.post("http://localhost:11434/api/generate", timeout=6, json={
            "model": "llama3.2", "prompt": prompt, "stream": False, "options": {"temperature": 0.0}})
        if r.status_code == 200:
            t = r.json().get("response", "").strip().strip('"\'')
            if t and len(t) <= MAX_TITLE_LENGTH:
                return clean(t)
    except Exception:
        pass
    # OpenAI fallback
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=6,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "gpt-4o-mini", "temperature": 0.0, "max_tokens": 25,
                      "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"].strip().strip('"\'')
                if t and len(t) <= MAX_TITLE_LENGTH:
                    return clean(t)
        except Exception:
            pass
    return None

# ---------------- transcribe + orchestrate ----------------
last_title = ""
last_raw = ""

def make_clip(duration=DEFAULT_CLIP_SECONDS, game=""):
    """Transcribe last `duration` s → title. Returns (title, raw_transcript)."""
    global last_title, last_raw
    game = game or live_twitch_game()
    audio = ring.last(duration)
    if audio.size < SAMPLE_RATE or float(np.max(np.abs(audio))) < 0.005:
        last_title, last_raw = "Stream Highlight", "No mic speech detected"
        return last_title, last_raw
    jargon = ", ".join(list(dict.fromkeys([cfg["streamer_name"]] + cfg["custom_words"] + game_jargon(game)))[:20])
    res = mlx_whisper.transcribe(audio, path_or_hf_repo=MODEL,
                                 initial_prompt=f"Streamer {cfg['streamer_name']}, game {game}, jargon: {jargon}")
    text = " ".join(res.get("text", "").split())
    if not text:
        last_title, last_raw = "Awesome Stream Moment", "No clear speech"
        return last_title, last_raw
    title = ai_title(text, game)
    if not title:
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        title = (sentences[-1] if sentences else text)[:MAX_TITLE_LENGTH]
    title = clean(title)
    last_title, last_raw = title, text
    print(f'📌 "{title}"  ← "{text}"')
    if cfg["enable_clip"]:
        subprocess.run(["pbcopy"], input=title.encode())
    if cfg["enable_notif"]:
        subprocess.run(["osascript", "-e",
            f'display notification "{title}" with title "🎬 Clip Backtrack" subtitle "Copied to clipboard"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    send_to_aitum(title)
    if cfg["enable_yt"]:
        path = find_latest_clip()
        if path:
            upload_youtube_async(path, title, text, game)
    return title, text

# ---------------- Aitum ----------------
def send_to_aitum(title):
    try:
        r = requests.get("http://localhost:7777/aitum/state", timeout=1)
        if r.status_code != 200:
            return
        for v in r.json().get("data", []):
            n = str(v.get("name", "")).lower()
            if "clip" in n and "title" in n:
                requests.put(f"http://localhost:7777/aitum/state/{v['_id']}", json={"value": title}, timeout=1)
                print(f"🟢 Aitum '{v['name']}' ← {title}")
                return
    except Exception as e:
        print(f"⚠️ aitum: {e}")

# ---------------- YouTube ----------------
def find_latest_clip():
    d = os.path.expanduser(cfg["obs_clips_dir"])
    if not os.path.isdir(d):
        return None
    now, best, best_m = time.time(), None, 0
    for root, _, files in os.walk(d):
        for f in files:
            if f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")) and not f.startswith("."):
                p = os.path.join(root, f)
                try:
                    m = os.path.getmtime(p)
                except OSError:
                    continue
                if now - m <= 300 and m > best_m:
                    best, best_m = p, m
    return best

def youtube_service():
    try:
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        from googleapiclient.discovery import build
        creds = None
        if os.path.exists(TOKEN_FILE):
            creds = Credentials.from_authorized_user_file(TOKEN_FILE, YOUTUBE_SCOPES)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                if not os.path.exists(CLIENT_SECRET_FILE):
                    if cfg["google_client_id"] and cfg["google_client_secret"]:
                        write_client_secret()
                    else:
                        print("⚠️ set YouTube credentials first"); return None
                creds = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, YOUTUBE_SCOPES)\
                    .run_local_server(port=8080, open_browser=True)
            open(TOKEN_FILE, "w").write(creds.to_json())
        return build("youtube", "v3", credentials=creds)
    except Exception as e:
        print(f"❌ youtube auth: {e}")
        return None

def write_client_secret():
    json.dump({"installed": {
        "client_id": cfg["google_client_id"], "client_secret": cfg["google_client_secret"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"]}}, open(CLIENT_SECRET_FILE, "w"), indent=2)

def upload_youtube_async(path, title, raw, game):
    def work():
        if not path or not os.path.exists(path):
            return
        svc = youtube_service()
        if not svc:
            return
        from googleapiclient.http import MediaFileUpload
        yt_title = title if "#shorts" in title.lower() else f"{title} #Shorts"
        tag = f"#{game.replace(' ', '')}" if game else "#Gaming"
        body = {"snippet": {"title": yt_title[:100],
                            "description": f'{cfg["streamer_name"]} stream highlight.\n\n🎙️ "{raw}"\n\n#Shorts {tag} #TwitchClips',
                            "tags": ["Shorts", "TwitchClips", game or "Gaming"], "categoryId": "20"},
                "status": {"privacyStatus": cfg["yt_privacy"], "selfDeclaredMadeForKids": False}}
        chunk = 1024 * 1024
        req = svc.videos().insert(part="snippet,status", body=body,
                                  media_body=MediaFileUpload(path, chunksize=chunk, resumable=True))
        resp = None
        print(f"🚀 uploading {os.path.basename(path)}…")
        while resp is None:
            t0 = time.time()
            status, resp = req.next_chunk()
            wait = chunk / (cfg["max_upload_kbps"] * 1024) - (time.time() - t0)
            if wait > 0:
                time.sleep(wait)
        print(f"✅ https://youtu.be/{resp['id']}")
    threading.Thread(target=work, daemon=True).start()

# ---------------- HTTP trigger (stdlib, for Stream Deck / hotkey / Aitum) ----------------
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path == "/clip":
            dur = max(5, min(BUFFER_SECONDS, int(q.get("duration", [DEFAULT_CLIP_SECONDS])[0])))
            title, raw = make_clip(dur, q.get("game", [""])[0])
            body = json.dumps({"title": title, "raw_transcript": raw}).encode()
        elif u.path == "/pause":
            globals().__setitem__("recording_paused", True); body = b'{"status":"paused"}'
        elif u.path == "/resume":
            globals().__setitem__("recording_paused", False); body = b'{"status":"recording"}'
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

def start_http():
    HTTPServer(("127.0.0.1", HTTP_PORT), Handler).serve_forever()

# ---------------- menu bar (rumps) ----------------
class ClipApp(rumps.App):
    def __init__(self):
        super().__init__("🎙️", quit_button=None)
        self.title_item = rumps.MenuItem("Last Title: (none)")
        self.game_item = rumps.MenuItem("Category: (auto)")
        self.pause_item = rumps.MenuItem("Pause Recording", callback=self.toggle_pause)
        self.yt_item = rumps.MenuItem(f"YouTube Auto-Upload: {'ON' if cfg['enable_yt'] else 'OFF'}", callback=self.toggle_yt)
        self.mic_menu = rumps.MenuItem("Microphone")
        self.mic_items = {}
        for d in input_devices():
            it = rumps.MenuItem(d["name"], callback=self.pick_mic)
            it.state = 1 if cfg["mic_device"] and cfg["mic_device"].lower() in d["name"].lower() else 0
            self.mic_menu.add(it)
            self.mic_items[d["name"]] = it
        self.menu = [
            rumps.MenuItem("Trigger Clip Now", callback=self.trigger),
            self.pause_item, None,
            self.title_item, self.game_item, None,
            self.mic_menu,
            self.yt_item,
            rumps.MenuItem("Set YouTube Credentials…", callback=self.set_creds),
            rumps.MenuItem("Authenticate YouTube…", callback=self.auth_yt), None,
            rumps.MenuItem("Edit Settings (config.json)", callback=lambda _: subprocess.run(["open", "-t", CONFIG_FILE])),
            rumps.MenuItem("Reload Settings", callback=self.reload_cfg),
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]
        rumps.Timer(self.refresh, 5).start()

    def sync_mic_checks(self):
        for name, it in self.mic_items.items():
            it.state = 1 if cfg["mic_device"] and cfg["mic_device"].lower() in name.lower() else 0

    def pick_mic(self, sender):
        cfg["mic_device"] = sender.title
        save_config()
        self.sync_mic_checks()
        threading.Thread(target=start_stream, daemon=True).start()
        rumps.notification("Clip Backtrack", "Microphone", sender.title)

    def toggle_pause(self, sender):
        global recording_paused
        recording_paused = not recording_paused
        sender.title = "Resume Recording" if recording_paused else "Pause Recording"
        self.title = "⏸️" if recording_paused else "🎙️"

    def toggle_yt(self, sender):
        cfg["enable_yt"] = not cfg["enable_yt"]
        save_config()
        sender.title = f"YouTube Auto-Upload: {'ON' if cfg['enable_yt'] else 'OFF'}"

    def trigger(self, _):
        threading.Thread(target=make_clip, daemon=True).start()

    def set_creds(self, _):
        cid = rumps.Window("Google OAuth Client ID:", "YouTube Credentials", cfg["google_client_id"], dimensions=(360, 24)).run()
        if not cid.clicked:
            return
        sec = rumps.Window("Google OAuth Client Secret:", "YouTube Credentials", "", dimensions=(360, 24)).run()
        if not sec.clicked:
            return
        cfg["google_client_id"] = cid.text.strip()
        cfg["google_client_secret"] = sec.text.strip()
        save_config()
        if cfg["google_client_id"] and cfg["google_client_secret"]:
            write_client_secret()
        rumps.notification("Clip Backtrack", "YouTube", "Credentials saved")

    def auth_yt(self, _):
        threading.Thread(target=youtube_service, daemon=True).start()

    def reload_cfg(self, _):
        load_config()
        self.sync_mic_checks()
        self.yt_item.title = f"YouTube Auto-Upload: {'ON' if cfg['enable_yt'] else 'OFF'}"
        threading.Thread(target=start_stream, daemon=True).start()

    def refresh(self, _):
        t = last_title if last_title else "(none)"
        self.title_item.title = f"Last Title: {t[:30]}"
        self.game_item.title = f"Category: {detected_game or '(auto)'}"


if __name__ == "__main__":
    start_stream()
    threading.Thread(target=start_http, daemon=True).start()
    print(f"READY — trigger: http://localhost:{HTTP_PORT}/clip")
    ClipApp().run()
