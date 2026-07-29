#!/usr/bin/env python3
"""
clip_in_context.py — Clip in Context: speech → AI clip title → clipboard/overlay/YouTube (macOS).

One terminal-launched process. That is the whole design decision: launched from
the terminal (or a LaunchAgent) it inherits microphone permission, so there is
no .app bundle, no code signing, and no TCC silence. Menu bar via rumps.

    mic → rolling 30s buffer → (trigger) → MLX Whisper → AI title
        → clipboard + notification → Aitum → optional YouTube Short

Trigger: menu bar item, or HTTP  GET http://localhost:5001/clip?duration=30&game=Valorant
Run:     python3 clip_in_context.py
"""
import os, re, sys, json, math, time, threading, subprocess, urllib.parse, html
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

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
    "streamer_name": "",
    "twitch_channel": "",
    "custom_words": [],
    "default_game": "",
    "mic_device": "",                    # substring match; "" = system default
    "whisper_model": "mlx-community/whisper-large-v3-turbo",  # MLX Whisper repo
    "ollama_model": "llama3.2",          # model for jargon + title generation
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
MAX_TITLE_LENGTH = 50

# Whisper models offered in the dashboard (MLX community repos).
WHISPER_CHOICES = [
    "mlx-community/whisper-large-v3-turbo",   # best accuracy, still real-time on Apple Silicon
    "mlx-community/whisper-medium.en-mlx",
    "mlx-community/whisper-small.en-mlx",
    "mlx-community/whisper-base.en-mlx",       # fastest / lowest quality
]

recording_paused = False
mic_volume = 0.0
live_text = ""                       # continuous live caption (title is clip-only)
transcribe_lock = threading.Lock()   # MLX isn't reentrant; serialize live loop vs clip trigger
whisper_ok = False                   # True once MLX has transcribed successfully
whisper_err = False                  # True if the model failed to load
ollama_ok = False                    # True while Ollama is reachable (health thread)
notice = ""                          # transient banner shown on the dashboard
notice_ts = 0.0
notice_level = "warn"                # "warn" (red) or "ok" (green)
clip_history = []                    # recent clips: {"t":epoch,"title","raw"}, newest last
CLIPS_LOG = os.path.join(HERE, "clips.jsonl")
if os.path.exists(CLIPS_LOG):
    try:
        with open(CLIPS_LOG, encoding="utf-8") as _f:
            clip_history = [json.loads(l) for l in _f if l.strip()][-50:]
    except Exception:
        clip_history = []

def set_notice(msg, level="warn"):
    global notice, notice_ts, notice_level
    notice, notice_ts, notice_level = msg, time.time(), level
    print(("✅ " if level == "ok" else "⚠️ ") + msg)

def append_history(title, raw):
    global clip_history
    rec = {"t": time.time(), "title": title, "raw": raw}
    clip_history = (clip_history + [rec])[-50:]
    try:
        with open(CLIPS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass

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
            "model": cfg["ollama_model"], "stream": False, "options": {"temperature": 0.2},
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
    prompt = (
        f'A live streamer just said this on stream:\n"{raw}"\n\n'
        "Write ONE short, catchy clip title (max 8 words) that captures what they ACTUALLY said above.\n"
        "- Punchy and natural, like a real Twitch/YouTube clip title.\n"
        "- Base it ONLY on the words above. Do NOT invent games, characters, names, or events that were not said.\n"
        "- Family-friendly. No quotes, no hashtags, no emoji, no ending period.\n"
        "Title:")
    # Ollama
    try:
        r = requests.post("http://localhost:11434/api/generate", timeout=6, json={
            "model": cfg["ollama_model"], "prompt": prompt, "stream": False, "options": {"temperature": 0.4}})
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
                json={"model": "gpt-4o-mini", "temperature": 0.4, "max_tokens": 25,
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

def repetitive(text):
    """True if the transcript is a hallucinated loop (few unique words repeated)."""
    words = text.lower().split()
    return len(words) >= 8 and len(set(words)) / len(words) < 0.35

def dedup(text):
    """Collapse consecutive repeated words/phrases (Whisper stutter: 'tricked tricked')."""
    w = text.split()
    out = []
    for x in w:
        if out and out[-1].lower() == x.lower():
            continue
        out.append(x)
    # collapse an immediately repeated 2-3 word phrase (a b a b -> a b)
    for n in (3, 2):
        i = 0
        while i + 2 * n <= len(out):
            if [t.lower() for t in out[i:i + n]] == [t.lower() for t in out[i + n:i + 2 * n]]:
                del out[i + n:i + 2 * n]
            else:
                i += 1
    return " ".join(out)

def boost(audio):
    """Normalize quiet audio to ~0.9 peak so Whisper doesn't hallucinate on low levels."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    return np.clip(audio * (0.9 / peak), -1.0, 1.0).astype(np.float32) if peak > 1e-4 else audio

def transcribe(audio, **kw):
    global whisper_ok, whisper_err
    # condition_on_previous_text=False stops the runaway "word word word…" repeat loop.
    res = mlx_whisper.transcribe(audio, path_or_hf_repo=cfg["whisper_model"],
                                 condition_on_previous_text=False, **kw)
    whisper_ok, whisper_err = True, False
    return res

def warmup_whisper():
    """Load the MLX model at startup (sets status + removes first-clip latency)."""
    global whisper_err
    try:
        with transcribe_lock:
            transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32))
    except Exception as e:
        whisper_err = True
        set_notice(f"Whisper model '{cfg['whisper_model'].split('/')[-1]}' failed to load")
        print(f"⚠️ whisper warmup: {e}")

def health_loop():
    """Poll Ollama reachability for the dashboard status indicator."""
    global ollama_ok
    while True:
        try:
            ollama_ok = requests.get("http://localhost:11434/api/tags", timeout=1.5).status_code == 200
        except Exception:
            ollama_ok = False
        time.sleep(5)

def transcribe_long(audio, **kw):
    """Transcribe in 5s chunks (the window that stays stable) and stitch — used
    only as a fallback when the full-context pass hallucinates a repeat loop."""
    step = SAMPLE_RATE * 5
    parts = []
    for i in range(0, audio.size, step):
        seg = audio[i:i + step]
        if seg.size < SAMPLE_RATE // 2 or float(np.max(np.abs(seg))) < 0.005:
            continue
        t = " ".join(transcribe(boost(seg), **kw).get("text", "").split())
        if re.search(r"[a-z0-9]", t.lower()) and not repetitive(t):
            parts.append(t)
    return " ".join(parts)

def transcribe_clip(audio, **kw):
    """One full-context pass over the whole clip so the title sees everything
    that was said. Only fall back to stitched 5s chunks if that pass loops."""
    one = dedup(" ".join(transcribe(boost(audio), **kw).get("text", "").split()))
    if re.search(r"[a-z0-9]", one.lower()) and not repetitive(one):
        return one
    return dedup(transcribe_long(audio, **kw))

def make_clip(duration=DEFAULT_CLIP_SECONDS, game=""):
    """Transcribe last `duration` s → title. Returns (title, raw_transcript)."""
    global last_title, last_raw
    game = game or live_twitch_game()
    audio = ring.last(duration)
    if audio.size < SAMPLE_RATE or float(np.max(np.abs(audio))) < 0.005:
        last_title, last_raw = "Stream Highlight", "No mic speech detected"
        return last_title, last_raw
    jargon = ", ".join(list(dict.fromkeys([cfg["streamer_name"]] + cfg["custom_words"] + game_jargon(game)))[:20])
    with transcribe_lock:
        text = transcribe_clip(audio, initial_prompt=f"Streamer {cfg['streamer_name']}, game {game}, jargon: {jargon}")
    if not re.search(r"[a-z0-9]", text.lower()):   # nothing intelligible
        last_title, last_raw = "Awesome Stream Moment", "No clear speech"
        return last_title, last_raw
    title = ai_title(text, game)
    if not title:
        if not ollama_ok:
            set_notice("Ollama unreachable — used a fallback title")
        # Fallback: opening words of what was said (period-splitting mangled URLs
        # like "www.fema.org" into "org").
        title = " ".join(text.split()[:8])
        if len(title) > MAX_TITLE_LENGTH:
            title = title[:MAX_TITLE_LENGTH].rsplit(" ", 1)[0] + "…"
    title = clean(title)
    last_title, last_raw = title, text
    append_history(title, text)
    print(f'📌 "{title}"  ← "{text}"')
    if cfg["enable_clip"]:
        subprocess.run(["pbcopy"], input=title.encode())
    if cfg["enable_notif"]:
        safe = title.replace("\\", "\\\\").replace('"', '\\"')  # AppleScript string-escape
        subprocess.run(["osascript", "-e",
            f'display notification "{safe}" with title "🎬 Clip in Context" subtitle "Copied to clipboard"'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    send_to_aitum(title)
    # NB: no auto-upload here — it would race clip creation. YouTube upload is a
    # separate step: GET /upload (Aitum webhook after the vertical clip exports).
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
def find_latest_clip(max_age=None):
    """Newest video in the OBS clips folder. max_age (seconds) optionally limits
    to recently-modified files; None = newest regardless of age."""
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
                if max_age is not None and now - m > max_age:
                    continue
                if m > best_m:
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
      try:
        if not path or not os.path.exists(path):
            return
        svc = youtube_service()
        if not svc:
            set_notice("YouTube not authenticated — set credentials in Settings")
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
        url = f"https://youtu.be/{resp['id']}"
        print(f"✅ {url}")
        set_notice(f"Uploaded to YouTube: {url}", "ok")
      except Exception as e:
        set_notice(f"YouTube upload failed: {e}")
    threading.Thread(target=work, daemon=True).start()

def do_upload():
    """Upload the newest exported clip with the last generated title. Call this
    AFTER the vertical clip is exported (e.g. an Aitum webhook after 'Create
    vertical clip') so it isn't raced against clip creation."""
    if not cfg["enable_yt"]:
        set_notice("YouTube upload is off — enable it in Settings")
        return
    path = find_latest_clip()
    if not path:
        set_notice("No clip found in the OBS clips folder to upload")
        return
    set_notice(f"Uploading {os.path.basename(path)}…", "ok")
    upload_youtube_async(path, last_title or "Stream Highlight", last_raw, detected_game)

# ---------------- HTTP trigger (stdlib, for Stream Deck / hotkey / Aitum) ----------------
PAGE = """<!DOCTYPE html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clip in Context</title>
<style>
  :root{--bg:#0b0d12;--card:#141821;--line:#232a38;--txt:#e8edf5;--dim:#8b96a8;
    --accent:#8b5cf6;--accent2:#ec4899;--blue:#38bdf8;--ok:#34d399;--radius:16px;
    --chev:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%238b96a8' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E")}
  *{box-sizing:border-box}
  html{background:#0b0d12}
  body{margin:0;min-height:100vh;
    background:radial-gradient(1200px 600px at 50% -10%,#161b26,#0b0d12) no-repeat fixed #0b0d12;
    color:var(--txt);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
    padding:28px 16px}
  .wrap{max-width:640px;margin:0 auto;display:flex;flex-direction:column;gap:18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);padding:20px}
  .row{display:flex;align-items:center;justify-content:space-between;gap:12px}
  h1{font-size:19px;margin:0;display:flex;align-items:center;gap:9px}
  .pill{font-size:12px;font-weight:700;padding:4px 11px;border-radius:20px;background:#0e2a20;color:var(--ok)}
  .pill.off{background:#2a1416;color:#f87171}
  .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);font-weight:700;margin-bottom:10px}
  .vu{height:18px;border-radius:9px;background:#0b0f19;border:1px solid var(--line);overflow:hidden;flex:1}
  .vu>i{display:block;height:100%;width:0;border-radius:9px;
    background:linear-gradient(90deg,#10b981,#f59e0b,#ef4444);transition:width .08s linear}
  .pct{min-width:42px;text-align:right;font-weight:700;color:var(--blue)}
  .title{font-size:22px;font-weight:800;color:var(--blue);background:#0b0f19;border:1px solid var(--line);
    border-radius:12px;padding:14px 16px;word-break:break-word;display:flex;justify-content:space-between;align-items:center;gap:10px}
  .scriptbox{font-style:italic;color:var(--dim);background:#0e131c;border:1px solid var(--line);
    border-radius:10px;padding:11px 14px;margin-top:10px;min-height:20px}
  button{font:inherit;cursor:pointer;border:none;border-radius:12px;color:#fff;font-weight:700}
  .go{width:100%;padding:15px;font-size:16px;margin-top:14px;
    background:linear-gradient(135deg,var(--accent),var(--accent2));box-shadow:0 8px 22px rgba(139,92,246,.35)}
  .go:active{transform:translateY(1px)}
  .mini{background:#2a3345;padding:6px 12px;font-size:12px;border-radius:8px}
  .save{width:100%;padding:14px;font-size:15px;background:linear-gradient(135deg,#6366f1,#3b82f6);margin-top:6px}
  label{font-size:13px;color:var(--dim);font-weight:600;display:block;margin:14px 0 6px}
  input,select{width:100%;background:#0b0d12;border:1px solid #313b4d;border-radius:9px;
    color:var(--txt);padding:10px 12px;font:inherit}
  select{appearance:none;-webkit-appearance:none;padding-right:34px;
    background-image:var(--chev);background-repeat:no-repeat;background-position:right 12px center;background-size:13px}
  input[type=checkbox]{width:20px;height:20px;accent-color:var(--accent)}
  .flex{display:flex;gap:14px}.flex>div{flex:1}
  .chk{display:flex;align-items:center;justify-content:space-between;margin:12px 0}
  .toast{position:fixed;left:50%;bottom:24px;transform:translateX(-50%) translateY(80px);
    background:#0e2a20;color:var(--ok);border:1px solid #14503b;padding:11px 20px;border-radius:12px;
    font-weight:700;transition:transform .25s;opacity:0}
  .toast.show{transform:translateX(-50%) translateY(0);opacity:1}
  .cat{color:var(--blue);font-weight:700}
  details.set{background:var(--card);border:1px solid var(--line);border-radius:var(--radius)}
  details.set summary{list-style:none;cursor:pointer;padding:16px 20px;font-size:11px;letter-spacing:.08em;
    text-transform:uppercase;font-weight:700;color:var(--dim);display:flex;justify-content:space-between;align-items:center}
  details.set summary::-webkit-details-marker{display:none}
  details.set summary::after{content:'';width:14px;height:14px;background:var(--chev) center/14px no-repeat;transition:transform .2s}
  details.set[open] summary::after{transform:rotate(180deg)}
  details.set .setbody{padding:0 20px 22px}
  /* narrow widths (e.g. an OBS custom browser dock) */
  @media (max-width:520px){
    body{padding:14px 10px}
    .wrap{gap:12px}
    .card{padding:14px}
    .row{flex-wrap:wrap}
    .flex{flex-direction:column;gap:0}
    h1{font-size:17px}
    .go{font-size:15px;padding:13px}
    details.set summary,details.set .setbody{padding-left:14px;padding-right:14px}
  }
</style></head><body><div class="wrap">

  <div class="card row">
    <h1>🎬 Clip in Context</h1>
    <div style="display:flex;gap:8px;align-items:center">
      <span id="live" class="pill">● Running</span>
      <button id="pauseBtn" class="mini" onclick="togglePause()">Pause</button>
      <button class="mini" style="background:#3a1f2b;color:#f9a8c4" onclick="quitApp(this)">Quit</button>
    </div>
  </div>

  <div class="card row" style="padding:13px 20px;font-size:13px;color:var(--dim);flex-wrap:wrap;gap:14px">
    <span style="display:flex;gap:22px">
      <span><b id="whDot" style="color:#fbbf24">●</b> Whisper <span id="whTxt">…</span></span>
      <span><b id="olDot" style="color:#fbbf24">●</b> Ollama <span id="olTxt">…</span></span>
    </span>
    <span>🎮 <span id="cat" class="cat">…</span></span>
  </div>

  <div id="notice" class="card" style="display:none;border-color:#7c2d12;background:#2a1206;color:#fca5a5;padding:11px 16px;font-size:13px;align-items:center;justify-content:space-between;gap:12px;flex-direction:row">
    <span id="noticeMsg"></span>
    <button class="mini" style="background:#4a1d1d;color:#fca5a5" onclick="$('notice').style.display='none'">Dismiss</button>
  </div>

  <div class="card">
    <div class="row" style="margin-bottom:10px">
      <div class="lbl" style="margin:0">💬 Live captions</div>
      <div class="vu" title="Mic level" style="flex:none;width:96px;height:6px"><i id="vu"></i></div>
    </div>
    <div class="scriptbox" id="cap" style="font-style:normal;color:var(--txt);height:4.5em;line-height:1.5;overflow:hidden">Listening…</div>
  </div>

  <div class="card">
    <div class="lbl">📌 Latest clip title</div>
    <div class="title"><span id="ttl">No clip yet</span><button class="mini" onclick="copyTitle()">Copy</button></div>
    <div class="scriptbox" id="script">Trigger a clip after you speak.</div>
    <div class="row" style="margin-top:14px;gap:10px;align-items:stretch">
      <select id="dur" style="width:auto;padding:0 32px 0 14px;font-weight:700"><option value="15">15s</option><option value="30" selected>30s</option><option value="45">45s</option></select>
      <button class="go" style="margin:0" onclick="trigger()" id="gobtn">✂️ Trigger Clip Now</button>
    </div>
  </div>

  <details class="set">
    <summary>🕒 Recent clips</summary>
    <div class="setbody">
      <div style="display:flex;justify-content:flex-end;margin-bottom:8px"><button class="mini" onclick="clearHistory(this)">Clear all</button></div>
      <div id="history" style="display:flex;flex-direction:column;gap:2px;font-size:13px">No clips yet.</div>
    </div>
  </details>

  <details class="set">
    <summary>⚙️ Settings</summary>
    <form class="setbody" onsubmit="save(event)">
      <div class="lbl">🎙️ Streamer</div>
      <label>Microphone device</label>
      <div style="position:relative">
        <select id="mic_device" style="padding-left:26px">__MIC_OPTS__</select>
        <div title="Mic level" style="position:absolute;left:11px;top:0;bottom:0;margin:auto 0;width:8px;height:22px;background:#1b2230;border-radius:4px;overflow:hidden;display:flex;align-items:flex-end">
          <i id="vu2" style="display:block;width:100%;height:0;background:linear-gradient(0deg,#10b981,#f59e0b,#ef4444);transition:height .08s linear"></i>
        </div>
      </div>
      <div class="flex"><div><label>Streamer name</label><input id="streamer_name" value="__STREAMER__"></div>
        <div><label>Twitch channel</label><input id="twitch_channel" value="__TWITCH__"></div></div>
      <label>Custom words / jargon (comma separated)</label><input id="custom_words" value="__WORDS__">
      <label>Default game fallback</label><input id="default_game" value="__GAME__">

      <div class="lbl" style="margin-top:24px">🧠 AI models</div>
      <label>Transcription (Whisper MLX)</label><select id="whisper_model">__WHISPER_OPTS__</select>
      <label>Title generation (Ollama)</label><select id="ollama_model">__MODEL_OPTS__</select>

      <div class="lbl" style="margin-top:24px">🚀 YouTube Shorts</div>
      <div class="chk"><label style="margin:0">Enable auto-upload</label><input type="checkbox" id="enable_yt" __YT__></div>
      <label>Google OAuth Client ID</label><input id="google_client_id" value="__CID__" placeholder="xxxx.apps.googleusercontent.com">
      <label>Google OAuth Client Secret</label><input type="password" id="google_client_secret" value="" placeholder="__SEC_PH__" autocomplete="off">
      <div class="flex"><div><label>Privacy</label><select id="yt_privacy">
        <option value="public" __PUB__>Public</option><option value="unlisted" __UNL__>Unlisted</option><option value="private" __PRV__>Private</option></select></div>
        <div><label>Upload limit (KB/s)</label><input id="max_upload_kbps" value="__KBPS__"></div></div>
      <label>OBS clips folder</label><input id="obs_clips_dir" value="__OBS__">
      <button type="button" onclick="fetch('/upload')" style="width:100%;margin-top:12px;padding:12px;font-size:14px;font-weight:600;color:var(--blue);background:#14202e;border:1px solid #24425e;border-radius:10px">⬆&nbsp; Upload latest clip now</button>
      <div style="font-size:12px;color:var(--dim);margin-top:8px">Or call <code>GET /upload</code> from Aitum after the vertical clip exports.</div>

      <div class="lbl" style="margin-top:24px">🔔 Preferences</div>
      <div class="chk"><label style="margin:0">Desktop notifications</label><input type="checkbox" id="enable_notif" __NOTIF__></div>
      <div class="chk"><label style="margin:0">Auto-copy title to clipboard</label><input type="checkbox" id="enable_clip" __CLIP__></div>

      <button class="save" type="submit">💾 Save settings</button>
    </form>
  </details>
</div>
<div id="toast" class="toast">Saved</div>
<script>
const $=id=>document.getElementById(id);
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
async function clearHistory(btn){
  if(!btn.dataset.armed){btn.dataset.armed='1';btn.textContent='Confirm?';setTimeout(()=>{if(btn.dataset.armed){delete btn.dataset.armed;btn.textContent='Clear all';}},2500);return;}
  delete btn.dataset.armed;btn.textContent='Clear all';
  await fetch('/api/history/clear',{method:'POST'});loadHistory();
}
async function loadHistory(){try{const h=await(await fetch('/api/history')).json();
  $('history').innerHTML=h.length?h.map(c=>`<div style="display:flex;justify-content:space-between;gap:12px;border-bottom:1px solid var(--line);padding:8px 0"><span style="color:var(--blue);font-weight:600">${esc(c.title)}</span><span style="color:var(--dim);white-space:nowrap">${new Date(c.t*1000).toLocaleTimeString([],{hour:'2-digit',minute:'2-digit'})}</span></div>`).join(''):'No clips yet.';
}catch(e){}}
loadHistory();
let paused=false;
function togglePause(){fetch(paused?'/resume':'/pause');}
function quitApp(btn){
  // In-page confirm: OBS's embedded browser (CEF) crashes on window.confirm().
  if(btn.dataset.armed){fetch('/quit');document.body.innerHTML='<p style=\"text-align:center;margin-top:80px;color:#8b96a8\">Clip in Context stopped. You can close this tab.</p>';return;}
  btn.dataset.armed='1';btn.textContent='Confirm?';
  setTimeout(()=>{if(btn.dataset.armed){delete btn.dataset.armed;btn.textContent='Quit';}},2500);
}
setInterval(async()=>{try{
  const s=await(await fetch('/api/status')).json();
  const nf=0.0003,v=s.mic_volume||0;
  let p=v>nf?Math.min(100,Math.max(4,Math.round(Math.log10(v/nf)*40))):0;
  {const vu=$('vu');if(vu)vu.style.width=p+'%';const v2=$('vu2');if(v2)v2.style.height=p+'%';}
  $('cat').textContent=s.category||'auto';
  if(s.live)$('cap').textContent=s.live;
  $('whDot').style.color=s.whisper_err?'#f87171':(s.whisper?'var(--ok)':'#fbbf24');
  $('whTxt').textContent=s.whisper_err?'error':(s.whisper?'ready':'loading…');
  $('olDot').style.color=s.ollama?'var(--ok)':'#f87171';$('olTxt').textContent=s.ollama?'up':'down';
  if(s.notice){const n=$('notice'),ok=s.notice_level==='ok';
    n.style.borderColor=ok?'#14503b':'#7c2d12';n.style.background=ok?'#0e2a20':'#2a1206';n.style.color=ok?'#6ee7b7':'#fca5a5';
    $('noticeMsg').textContent=(ok?'✅ ':'⚠ ')+s.notice;n.style.display='flex';}
  const l=$('live');if(s.paused){l.textContent='⏸ Paused';l.classList.add('off');}else{l.textContent='● Running';l.classList.remove('off');}
  paused=s.paused;$('pauseBtn').textContent=paused?'Resume':'Pause';
  if(s.last_title){$('ttl').textContent=s.last_title;}
  if(s.last_raw){$('script').textContent='"'+s.last_raw+'"';}
}catch(e){}},150);
async function trigger(){const b=$('gobtn');b.textContent='⏳ Processing…';b.disabled=true;
  try{const d=await(await fetch('/clip?json=1&duration='+$('dur').value)).json();
    if(d.title)$('ttl').textContent=d.title;if(d.raw_transcript)$('script').textContent='"'+d.raw_transcript+'"';loadHistory();}catch(e){}
  b.textContent='✂️ Trigger Clip Now';b.disabled=false;}
function copyTitle(){navigator.clipboard.writeText($('ttl').textContent);toast('Copied');}
function toast(m){const t=$('toast');t.textContent=m;t.classList.add('show');setTimeout(()=>t.classList.remove('show'),1800);}
async function save(ev){ev.preventDefault();
  const b={streamer_name:$('streamer_name').value,twitch_channel:$('twitch_channel').value,
    mic_device:$('mic_device').value,default_game:$('default_game').value,
    whisper_model:$('whisper_model').value,ollama_model:$('ollama_model').value,
    custom_words:$('custom_words').value.split(',').map(s=>s.trim()).filter(Boolean),
    enable_yt:$('enable_yt').checked,google_client_id:$('google_client_id').value,
    google_client_secret:$('google_client_secret').value,yt_privacy:$('yt_privacy').value,
    max_upload_kbps:parseInt($('max_upload_kbps').value)||1500,obs_clips_dir:$('obs_clips_dir').value,
    enable_notif:$('enable_notif').checked,enable_clip:$('enable_clip').checked};
  const r=await fetch('/settings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b)});
  toast(r.ok?'Saved':'Error');if(r.ok)$('google_client_secret').value='';}
</script></body></html>"""

def apply_settings(data):
    """Update cfg from a settings dict (web form or API). Restarts mic if changed."""
    global recording_paused, whisper_ok
    mic_changed = "mic_device" in data and str(data["mic_device"]) != cfg["mic_device"]
    whisper_changed = "whisper_model" in data and str(data["whisper_model"]) != cfg["whisper_model"]
    for k in ("streamer_name", "twitch_channel", "default_game", "mic_device", "whisper_model", "ollama_model", "yt_privacy", "obs_clips_dir"):
        if k in data:
            cfg[k] = str(data[k])
    if "custom_words" in data:
        raw = data["custom_words"] if isinstance(data["custom_words"], list) else str(data["custom_words"]).split(",")
        cfg["custom_words"] = [str(w).strip() for w in raw if str(w).strip()]
    for k in ("enable_yt", "enable_notif", "enable_clip"):
        if k in data:
            cfg[k] = bool(data[k])
    if "max_upload_kbps" in data:
        try:
            cfg["max_upload_kbps"] = int(data["max_upload_kbps"])
        except (TypeError, ValueError):
            pass
    cid, sec = str(data.get("google_client_id", "")).strip(), str(data.get("google_client_secret", "")).strip()
    if cid:
        cfg["google_client_id"] = cid
    if sec:
        cfg["google_client_secret"] = sec
    if cfg["google_client_id"] and cfg["google_client_secret"]:
        write_client_secret()
    save_config()
    if mic_changed:
        threading.Thread(target=start_stream, daemon=True).start()
    if whisper_changed:
        whisper_ok = False                       # load the new model (first use downloads it)
        threading.Thread(target=warmup_whisper, daemon=True).start()

def live_loop():
    """Continuously transcribe the last few seconds for live captions (no title)."""
    global live_text
    while True:
        time.sleep(2.0)
        if recording_paused:
            continue
        audio = ring.last(5)
        if audio.size < SAMPLE_RATE or float(np.max(np.abs(audio))) < 0.01:
            continue
        try:
            with transcribe_lock:
                res = transcribe(boost(audio))
            t = " ".join(res.get("text", "").split())
            if re.search(r"[a-z0-9]", t.lower()) and not repetitive(t):
                live_text = dedup(t)
        except Exception:
            pass

def status_json():
    return {
        "mic_volume": mic_volume,
        "paused": recording_paused,
        "whisper": whisper_ok,
        "whisper_err": whisper_err,
        "ollama": ollama_ok,
        "notice": notice if (time.time() - notice_ts < 30) else "",
        "notice_level": notice_level,
        "live": live_text,
        "last_title": last_title,
        "last_raw": last_raw,
        "category": detected_game,
    }

def ollama_models():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []

def dashboard_html():
    e = lambda v: html.escape(str(v), quote=True)
    opts = "".join(
        f'<option value="{e(d["name"])}"{" selected" if cfg["mic_device"] and cfg["mic_device"].lower() in d["name"].lower() else ""}>{e(d["name"])}</option>'
        for d in input_devices()) or '<option value="">Default input</option>'
    models = ollama_models()
    if cfg["ollama_model"] not in models:
        models.insert(0, cfg["ollama_model"])
    model_opts = "".join(f'<option{" selected" if m == cfg["ollama_model"] else ""}>{e(m)}</option>' for m in models)
    whis = WHISPER_CHOICES[:]
    if cfg["whisper_model"] not in whis:
        whis.insert(0, cfg["whisper_model"])
    label = lambda m: e(m.split("/")[-1].replace("whisper-", "").replace("-mlx", ""))
    whisper_opts = "".join(f'<option value="{e(m)}"{" selected" if m == cfg["whisper_model"] else ""}>{label(m)}</option>' for m in whis)
    repl = {
        "__STREAMER__": e(cfg["streamer_name"]), "__TWITCH__": e(cfg["twitch_channel"]),
        "__WORDS__": e(", ".join(cfg["custom_words"])), "__GAME__": e(cfg["default_game"]),
        "__MIC_OPTS__": opts, "__MODEL_OPTS__": model_opts, "__WHISPER_OPTS__": whisper_opts,
        "__OBS__": e(cfg["obs_clips_dir"]), "__KBPS__": e(cfg["max_upload_kbps"]),
        "__CID__": e(cfg["google_client_id"]),
        "__SEC_PH__": "•••••• saved" if cfg["google_client_secret"] else "GOCSPX-…",
        "__YT__": "checked" if cfg["enable_yt"] else "", "__NOTIF__": "checked" if cfg["enable_notif"] else "",
        "__CLIP__": "checked" if cfg["enable_clip"] else "",
        "__PUB__": "selected" if cfg["yt_privacy"] == "public" else "",
        "__UNL__": "selected" if cfg["yt_privacy"] == "unlisted" else "",
        "__PRV__": "selected" if cfg["yt_privacy"] == "private" else "",
    }
    page = PAGE
    for k, v in repl.items():
        page = page.replace(k, str(v))
    return page

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/dashboard"):
            self._send(dashboard_html(), "text/html; charset=utf-8")
        elif u.path == "/api/status":
            self._send(json.dumps(status_json()))
        elif u.path == "/api/history":
            self._send(json.dumps(clip_history[-20:][::-1]))   # newest first
        elif u.path == "/clip":
            dur = max(5, min(BUFFER_SECONDS, int(q.get("duration", [DEFAULT_CLIP_SECONDS])[0])))
            title, raw = make_clip(dur, q.get("game", [""])[0])
            self._send(json.dumps({"title": title, "raw_transcript": raw}))
        elif u.path == "/upload":
            threading.Thread(target=do_upload, daemon=True).start()
            self._send(b'{"status":"uploading"}')
        elif u.path == "/pause":
            globals().__setitem__("recording_paused", True); self._send(b'{"status":"paused"}')
        elif u.path == "/resume":
            globals().__setitem__("recording_paused", False); self._send(b'{"status":"recording"}')
        elif u.path == "/quit":
            self._send(b'{"status":"quitting"}')
            threading.Timer(0.2, lambda: os._exit(0)).start()   # launchd KeepAlive=false → stays stopped
        else:
            self._send(b"not found", "text/plain", 404)

    def do_POST(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/settings":
            n = int(self.headers.get("Content-Length", 0))
            try:
                apply_settings(json.loads(self.rfile.read(n) or b"{}"))
                self._send(b'{"status":"saved"}')
            except Exception as ex:
                self._send(json.dumps({"status": "error", "message": str(ex)}).encode(), code=400)
        elif path == "/api/history/clear":
            global clip_history
            clip_history = []
            try:
                open(CLIPS_LOG, "w").close()
            except Exception:
                pass
            self._send(b'{"status":"cleared"}')
        else:
            self._send(b"not found", "text/plain", 404)

def start_http():
    ThreadingHTTPServer(("127.0.0.1", HTTP_PORT), Handler).serve_forever()

# ---------------- menu bar (rumps) ----------------
class ClipApp(rumps.App):
    def __init__(self):
        super().__init__("Clip in Context", quit_button=None)
        self._symbol("waveform")   # SF Symbol menu-bar icon (shown when the status bar builds)
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
            rumps.MenuItem("Open Dashboard…", callback=lambda _: subprocess.run(["open", f"http://localhost:{HTTP_PORT}/"])),
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
        rumps.notification("Clip in Context", "Microphone", sender.title)

    def _symbol(self, name):
        """Set the menu-bar icon to an SF Symbol (template so it adapts to light/dark)."""
        try:
            import AppKit
            img = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(name, None)
            if img is None:
                return
            img.setTemplate_(True)
            self._icon_nsimage = img          # rumps uses this when it builds the status bar
            item = getattr(getattr(self, "_nsapp", None), "nsstatusitem", None)
            if item is not None:              # already running → update live
                item.setTitle_("")
                item.setImage_(img)
        except Exception as e:
            print(f"icon: {e}")

    def toggle_pause(self, sender):
        global recording_paused
        recording_paused = not recording_paused
        sender.title = "Resume Recording" if recording_paused else "Pause Recording"
        self._symbol("pause.circle" if recording_paused else "waveform")

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
        rumps.notification("Clip in Context", "YouTube", "Credentials saved")

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
    threading.Thread(target=live_loop, daemon=True).start()
    threading.Thread(target=health_loop, daemon=True).start()
    threading.Thread(target=warmup_whisper, daemon=True).start()
    print(f"READY — trigger: http://localhost:{HTTP_PORT}/clip")
    app = ClipApp()
    import AppKit  # menu-bar-only: no Dock icon (otherwise Python shows a Dock rocket)
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.run()
