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
import os, re, sys, json, math, time, threading, subprocess, urllib.parse, html, queue, shutil
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Ensure standard brew / local bin paths are in PATH (needed when running under launchd / GUI app)
for _p in ["/opt/homebrew/bin", "/usr/local/bin", os.path.expanduser("~/.homebrew/bin")]:
    if _p not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = f"{_p}{os.pathsep}" + os.environ.get("PATH", "")

# Line-buffer stdout/stderr so /tmp/clipincontext.log is live (launchd block-buffers
# it otherwise, and every 📌/upload/error line sits in memory until the process exits).
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import sounddevice as sd
from scipy import signal
import mlx_whisper
import requests
import rumps

import clip_editor
import draw_showcase

HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(HERE, "config.json")
TOKEN_FILE = os.path.join(HERE, "youtube_token.json")
CLIENT_SECRET_FILE = os.path.join(HERE, "client_secret.json")
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl"
]

# ---------------- config (persisted to config.json) ----------------
cfg = {
    "streamer_name": "",
    "twitch_channel": "",
    "custom_words": [],
    "default_game": "",
    "default_duration": 30,              # default clip length in seconds (15, 30, 45, 60)
    "mic_device": "",                    # substring match; "" = system default
    "whisper_model": "mlx-community/whisper-large-v3-turbo",  # MLX Whisper repo
    "ollama_model": "llama3.2",          # model for jargon + title generation
    "enable_yt": False,
    "yt_privacy": "unlisted",
    "max_upload_kbps": 800,   # KB/s cap on YT upload; low so it yields uplink to the live stream
    "obs_clips_dir": "~/Movies",
    "enable_notif": True,
    "enable_clip": True,
    "enable_auto_editor": True,          # run AI video editor (subtitles, hook banner, loudnorm) before upload
    "smart_trim_silence": True,          # trim lead-in silence before first word
    "sub_color": "yellow",               # yellow, green, cyan, red, gold
    "sub_size": 64,                      # 48, 64, 76, 88
    "sub_position": "lower_third",       # lower_third, center, top_third
    "max_silence_gap_sec": 6.0,          # max gap in seconds to merge silence in story cuts
    "preserve_story_span": True,         # preserve continuous video span for story arcs <= 45s
    "segment_padding_sec": 0.5,          # pre/post padding around spoken segments
    "google_client_id": "",
    "google_client_secret": "",
    "review_uploads": True,              # /upload queues clips for review; off = edit + publish right away
    "publish_slots": ["12:00", "15:00", "18:00", "21:00"],   # local times approved Shorts go public
    # Twitch category (substring, case-insensitive; longest match wins) → hashtags
    "game_hashtags": {
        "tears of the kingdom": ["#zelda", "#totk"], "totk": ["#zelda", "#totk"],
        "breath of the wild": ["#zelda", "#botw"], "botw": ["#zelda", "#botw"],
        "zelda": ["#zelda"],
    },
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
BUFFER_SECONDS = 120      # ring buffer length; caps how far back /clip?duration= can reach (~8 MB RAM)
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
transcribe_lock = clip_editor.WHISPER_LOCK   # MLX isn't reentrant; shared with the editor
clip_action_lock = threading.Lock()   # Serialize clip renaming and file operations to prevent race conditions
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

def _llm(prompt, max_tokens=25):
    """Ollama first, OpenAI fallback. Returns the raw completion or ''."""
    try:
        r = requests.post("http://localhost:11434/api/generate", timeout=8, json={
            "model": cfg["ollama_model"], "prompt": prompt, "stream": False, "options": {"temperature": 0.5}})
        if r.status_code == 200:
            return r.json().get("response", "")
    except Exception:
        pass
    key = os.getenv("OPENAI_API_KEY")
    if key:
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", timeout=8,
                headers={"Authorization": f"Bearer {key}"},
                json={"model": "gpt-4o-mini", "temperature": 0.5, "max_tokens": max_tokens,
                      "messages": [{"role": "user", "content": prompt}]})
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
    return ""

def parse_titles(text):
    """LLM output → clean title lines (drops numbering, bullets, quotes, over-long lines)."""
    out = []
    for line in text.splitlines():
        t = re.sub(r"^\s*(?:\d+[.)]|[-*•])\s*", "", line).strip().strip('"\'').strip()
        if t and len(t) <= MAX_TITLE_LENGTH and not t.lower().startswith(("title", "here")):
            out.append(clean(t))
    return out

def ai_titles(raw, game="", n=1):
    """Up to n distinct title candidates for a transcript ([] if the LLM is unavailable)."""
    if not raw or len(raw) < 5:
        return []
    best = top_titles()
    style = ("Titles that did well on this channel (match the style, not the content):\n"
             + "\n".join(f"- {t}" for t in best) + "\n\n") if best else ""
    ask = "ONE short YouTube Shorts title" if n == 1 else f"{n} different short YouTube Shorts titles, one per line"
    prompt = (
        f'A streamer playing {game or "a game"} just said this during a highlight moment on stream:\n"{raw}"\n\n'
        + style +
        f"Write {ask} (max 8 words each) that make someone scrolling past stop and watch.\n"
        "- Describe the situation or stakes (clutch win, fail, surprise, close call), not a word-for-word quote.\n"
        "- Infer only from the words above. Do NOT invent characters, bosses, names, or events that weren't implied.\n"
        "- Ignore filler, grunts, and stray words that don't fit the sentence.\n"
        "- Family-friendly. No quotes, no hashtags, no emoji, no ending period, no numbering.\n"
        + ("Title:" if n == 1 else "Titles:"))
    return list(dict.fromkeys(parse_titles(_llm(prompt, 25 * n))))[:n]

def ai_title(raw, game=""):
    t = ai_titles(raw, game, 1)
    return t[0] if t else None

# ---------------- transcribe + orchestrate ----------------
last_title = ""
last_raw = ""
title_pending = False    # True while a /clip is transcribing, so /name and /upload wait for it
last_trigger_ts = 0.0    # time.time() when the last /clip started; /name & /upload wait for a clip newer than this

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

# Whisper emits a stray token on a breath/click at clip start ("Rom Okay. No!").
# ponytail: fixed word list; add new offenders here as they show up in clips.jsonl.
LEAD_JUNK = re.compile(r"^(?:(?:rom|um+|uh+|hm+|mm+)\b[\s,.!?…-]*)+", re.IGNORECASE)

def strip_lead_junk(text):
    return LEAD_JUNK.sub("", text).strip()

def transcribe_clip(audio, **kw):
    """One full-context pass over the whole clip so the title sees everything
    that was said. Only fall back to stitched 5s chunks if that pass loops."""
    one = strip_lead_junk(dedup(" ".join(transcribe(boost(audio), **kw).get("text", "").split())))
    if re.search(r"[a-z0-9]", one.lower()) and not repetitive(one):
        return one
    return strip_lead_junk(dedup(transcribe_long(audio, **kw)))

def make_clip(duration=DEFAULT_CLIP_SECONDS, game=""):
    """Transcribe last `duration` s → title. Returns (title, raw_transcript)."""
    global last_title, last_raw, title_pending, last_trigger_ts
    title_pending = True  # /name and /upload block on this so they wait for THIS clip's title
    last_trigger_ts = time.time()  # /name & /upload only act on a clip OBS exported after now
    try:
        game = game or live_twitch_game()
        audio = ring.last(duration)
        if audio.size < SAMPLE_RATE or float(np.max(np.abs(audio))) < 0.005:
            # Nothing was captured, so there's no clip to log — say so, otherwise the
            # trigger looks like it silently did nothing.
            set_notice("No mic audio in the last %ds — check the mic in Settings" % duration)
            last_title, last_raw = "Stream Highlight", "No mic speech detected"
            return last_title, last_raw
        jargon = ", ".join(list(dict.fromkeys([cfg["streamer_name"]] + cfg["custom_words"] + game_jargon(game)))[:20])
        set_notice(f"Transcribing the last {duration}s…", "work")
        with transcribe_lock:
            text = transcribe_clip(audio, initial_prompt=f"Streamer {cfg['streamer_name']}, game {game}, jargon: {jargon}")
        if not re.search(r"[a-z0-9]", text.lower()):   # nothing intelligible
            set_notice("Audio had no recognisable speech — nothing to title")
            last_title, last_raw = "Awesome Stream Moment", "No clear speech"
            return last_title, last_raw
        set_notice("Writing a title…", "work")
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
    finally:
        title_pending = False

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
EDITOR_SUFFIXES = ("_edited", "_STORY", "_story_raw")   # clip_editor outputs, not OBS exports

def find_latest_clip(max_age=None):
    """Newest video in the OBS clips folder. max_age (seconds) optionally limits
    to recently-modified files; None = newest regardless of age."""
    d = os.path.expanduser(cfg["obs_clips_dir"])
    if not os.path.isdir(d):
        return None
    now, best, best_m = time.time(), None, 0
    for root, _, files in os.walk(d):
        for f in files:
            stem = os.path.splitext(f)[0]
            if (f.lower().endswith((".mp4", ".mov", ".mkv", ".webm")) and not f.startswith(".")
                    and not stem.endswith(EDITOR_SUFFIXES)):
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

OAUTH_REDIRECT = f"http://localhost:{HTTP_PORT}/oauth2callback"
_oauth_flow = None   # pending google_auth_oauthlib Flow between /auth and /oauth2callback

def youtube_creds():
    """Return valid saved credentials (refreshing if needed). No interactive auth."""
    from google.oauth2.credentials import Credentials
    from google.auth.transport.requests import Request
    if not os.path.exists(TOKEN_FILE):
        return None
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, YOUTUBE_SCOPES)
    except Exception:
        return None
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            open(TOKEN_FILE, "w").write(creds.to_json())
        except Exception as e:
            print(f"⚠️ token refresh: {e}")
            return None
    return creds if creds and creds.valid else None

def youtube_auth_status():
    """Quick check of YouTube auth state without blocking or throwing exceptions."""
    if not os.path.exists(TOKEN_FILE):
        return {"authenticated": False, "status": "Not Connected"}
    try:
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, YOUTUBE_SCOPES)
        if creds and (creds.valid or (creds.expired and creds.refresh_token)):
            return {"authenticated": True, "status": "Connected"}
    except Exception:
        pass
    return {"authenticated": False, "status": "Needs Auth"}

def youtube_service():
    from googleapiclient.discovery import build
    creds = youtube_creds()
    return build("youtube", "v3", credentials=creds) if creds else None

def start_oauth():
    """Begin YouTube OAuth. The callback is caught on our own always-on server
    (port 5001), not run_local_server's flaky throwaway port."""
    global _oauth_flow
    if not os.path.exists(CLIENT_SECRET_FILE):
        if cfg["google_client_id"] and cfg["google_client_secret"]:
            write_client_secret()
        else:
            set_notice("Set your Google OAuth Client ID + Secret first")
            return
    try:
        from google_auth_oauthlib.flow import Flow
        _oauth_flow = Flow.from_client_secrets_file(CLIENT_SECRET_FILE, scopes=YOUTUBE_SCOPES,
                                                    redirect_uri=OAUTH_REDIRECT)
        url, _ = _oauth_flow.authorization_url(access_type="offline", prompt="consent",
                                               include_granted_scopes="true")
        set_notice("Opening browser to authorize YouTube…", "ok")
        subprocess.run(["open", url])
    except Exception as e:
        set_notice(f"YouTube auth error: {e}")

def finish_oauth(code):
    """Exchange the auth code (from /oauth2callback) for a saved token."""
    global _oauth_flow
    if not _oauth_flow:
        return "No authorization in progress — start it from the app first."
    try:
        _oauth_flow.fetch_token(code=code)
        open(TOKEN_FILE, "w").write(_oauth_flow.credentials.to_json())
        _oauth_flow = None
        set_notice("YouTube authenticated", "ok")
        return "Authenticated ✓ — you can close this tab and return to Clip in Context."
    except Exception as e:
        set_notice(f"YouTube auth failed: {e}")
        return f"Authorization failed: {e}"

def write_client_secret():
    json.dump({"installed": {
        "client_id": cfg["google_client_id"], "client_secret": cfg["google_client_secret"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "redirect_uris": ["http://localhost"]}}, open(CLIENT_SECRET_FILE, "w"), indent=2)

def get_game_hashtags(g_name):
    lower = (g_name or "").strip().lower()
    if not lower:
        return []
    # Longest key first so "tears of the kingdom" beats plain "zelda".
    for k in sorted(cfg.get("game_hashtags") or {}, key=len, reverse=True):
        if k.lower() in lower:
            v = cfg["game_hashtags"][k]
            return list(v) if isinstance(v, list) else [v]
    tag = re.sub(r"[^a-zA-Z0-9]", "", g_name)
    return [f"#{tag}"] if tag and tag.lower() not in ("justchatting", "gaming") else []

def build_metadata(title, raw, game):
    """YouTube snippet for a Short: (title ≤100 chars with hashtags, tags, description)."""
    game_tags = get_game_hashtags(game)
    hashtags = ["#Shorts", *game_tags, "#Gaming", "#TwitchClips", "#ShortsViral"]
    base = title.strip()
    yt_title = base
    for h in hashtags:                       # whole hashtags only, as many as fit in 100
        if h.lower() not in base.lower() and len(yt_title) + 1 + len(h) <= 100:
            yt_title += " " + h
    yt_title = yt_title[:100]

    tags = ([game] if game else []) + ["Shorts", "Gaming", "TwitchClips", "ViralShorts", "StreamHighlights"]
    if game:
        tags.append(f"{game} clips")
    for h in game_tags:
        if h.lstrip("#").lower() not in (t.lower() for t in tags):
            tags.append(h.lstrip("#"))
    if cfg.get("streamer_name"):
        tags.append(cfg["streamer_name"])

    description = (
        f'{base}{f" | {game}" if game else ""}\n\n'
        f'🎙️ "{clean(raw)}"\n\n'
        f'Highlight by {cfg.get("streamer_name") or "Streamer"}\n\n'
        f'{" ".join(hashtags)}'
    )
    return yt_title, tags, description

def _do_youtube_upload(path, title, raw, game, publish_at=None):
    """Blocking upload of one clip; returns the video id. publish_at (aware datetime)
    uploads it private and lets YouTube make it public then. Throttled to
    max_upload_kbps so it yields uplink to a live stream (~12 Mbps outbound)."""
    svc = youtube_service()
    if not svc:
        raise RuntimeError("YouTube not authenticated — click Authenticate in the YouTube tab")
    from googleapiclient.http import MediaFileUpload
    yt_title, tags, description = build_metadata(title, raw, game or cfg.get("default_game") or "")
    status = {"privacyStatus": cfg.get("yt_privacy", "public"), "selfDeclaredMadeForKids": False}
    if publish_at:
        status.update(privacyStatus="private", publishAt=publish_at.isoformat())
    body = {"snippet": {"title": yt_title, "description": description, "tags": tags,
                        "categoryId": "20"},   # 20 = Gaming
            "status": status}
    chunk = 1024 * 1024
    req = svc.videos().insert(part="snippet,status", body=body,
                              media_body=MediaFileUpload(path, chunksize=chunk, resumable=True))
    resp = None
    print(f"🚀 uploading {os.path.basename(path)}…")
    set_notice(f"Uploading {os.path.basename(path)} to YouTube…", "work")
    while resp is None:
        t0 = time.time()
        _, resp = req.next_chunk()
        wait = chunk / (cfg["max_upload_kbps"] * 1024) - (time.time() - t0)
        if wait > 0:
            time.sleep(wait)
    url = f"https://youtu.be/{resp['id']}"
    when = f" — goes public {publish_at:%a %H:%M}" if publish_at else ""
    print(f"✅ {url}{when}")
    set_notice(f"Uploaded to YouTube: {url}{when}", "ok")
    return resp["id"]

# ---------------- clip records: review queue → edit → scheduled upload ----------------
# One record per exported clip, persisted so a restart, crash, or the daily quota
# never loses a clip. status: pending (awaiting review) → approved (queued) →
# processing → scheduled/uploaded, or skipped / failed (retry from the dashboard).
RECORDS_FILE = os.path.join(HERE, "uploads.json")
_rec_lock = threading.Lock()
try:
    with open(RECORDS_FILE, encoding="utf-8") as _f:
        records = json.load(_f)
except (OSError, ValueError):
    records = []

def save_records():
    with _rec_lock:
        tmp = RECORDS_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=1, ensure_ascii=False)
        os.replace(tmp, RECORDS_FILE)   # atomic: a crash mid-write can't truncate the queue

def get_record(rid):
    return next((r for r in records if r["id"] == rid), None)

def top_titles(n=5):
    """Best-performing uploaded titles (by views) — few-shot style examples for new titles."""
    seen = sorted((r for r in records if r.get("views")), key=lambda r: r["views"], reverse=True)
    return [r["title"] for r in seen[:n]] if len(seen) >= 3 else []

def next_publish_slot(slots, taken, now):
    """First configured local HH:MM slot ≥30 min from now that no record already holds."""
    earliest = now + timedelta(minutes=30)
    for day in range(60):
        d = (now + timedelta(days=day)).date()
        for hm in sorted(slots):
            h, m = map(int, hm.split(":"))
            t = datetime(d.year, d.month, d.day, h, m).astimezone()
            if t >= earliest and t.isoformat() not in taken:
                return t
    return None

def next_quota_reset():
    """YouTube Data API quota resets at midnight Pacific; retry a few minutes after."""
    pt = datetime.now(ZoneInfo("America/Los_Angeles"))
    return (pt + timedelta(days=1)).replace(hour=0, minute=5, second=0, microsecond=0).timestamp()

QUOTA_ERRORS = ("quotaExceeded", "uploadLimitExceeded", "dailyLimitExceeded")
_upload_q = queue.Queue()
_queued = set()   # record ids waiting in _upload_q, so the retry loop never double-queues

def enqueue(rec):
    if rec["id"] not in _queued:
        _queued.add(rec["id"])
        _upload_q.put(rec["id"])

def process_record(rec):
    """Edit (captions, banner with the reviewed title) then upload. The edited file is
    kept on the record, so a quota retry tomorrow doesn't re-render it."""
    rec.update(status="processing", error="")
    save_records()
    if not (rec.get("edited_path") and os.path.exists(rec["edited_path"])):
        rec["edited_path"] = run_clip_editor_job(rec["path"], title=rec["title"])
        save_records()
    publish_at = None
    if cfg.get("review_uploads", True):
        publish_at = next_publish_slot(cfg["publish_slots"], {r.get("publish_at") for r in records},
                                       datetime.now().astimezone())
    rec["yt_id"] = _do_youtube_upload(rec["edited_path"], rec["title"], rec["raw"], rec["game"], publish_at)
    rec.update(status="scheduled" if publish_at else "uploaded",
               publish_at=publish_at.isoformat() if publish_at else None, uploaded_at=time.time())
    save_records()

# One serialized worker: uploads go out ONE at a time. Concurrent uploads
# multiplied uplink pressure and starved the live stream outputs.
def _upload_worker():
    while True:
        rid = _upload_q.get()
        _queued.discard(rid)
        rec = get_record(rid)
        try:
            if rec and rec["status"] == "approved":
                process_record(rec)
        except Exception as e:
            msg = str(e)
            print(f"❌ upload failed: {e}")
            if any(k in msg for k in QUOTA_ERRORS):
                # 10,000 units/day, an upload costs 1,600 → ~6/day. Stays approved; retried after reset.
                rec.update(status="approved", retry_after=next_quota_reset(), error="Daily upload limit — retrying after reset")
                set_notice("Daily YouTube upload limit reached (~6/day). Queued clips retry after midnight Pacific.")
            else:
                rec.update(status="failed", error=msg[:300])
                set_notice(f"YouTube upload failed: {e}")
            save_records()
        finally:
            _upload_q.task_done()

def retry_loop():
    """Re-queue approved clips (after a restart, or once the quota has reset)."""
    for r in records:
        if r["status"] == "processing":   # interrupted mid-upload by a restart
            r["status"] = "approved"
    while True:
        for r in records:
            if r["status"] == "approved" and (r.get("retry_after") or 0) <= time.time():
                enqueue(r)
        time.sleep(300)

def stats_loop():
    """Pull view counts for uploaded Shorts every 30 min (1 quota unit per 50 videos)."""
    while True:
        time.sleep(60)
        ids = [r["yt_id"] for r in records if r.get("yt_id")][-200:]
        svc = youtube_service() if ids else None
        if svc:
            try:
                views = {}
                for i in range(0, len(ids), 50):
                    res = svc.videos().list(part="statistics", id=",".join(ids[i:i + 50])).execute()
                    views.update({v["id"]: int(v["statistics"].get("viewCount", 0)) for v in res.get("items", [])})
                for r in records:
                    if r.get("yt_id") in views:
                        r["views"] = views[r["yt_id"]]
                save_records()
            except Exception as e:
                print(f"⚠️ stats: {e}")
        time.sleep(1800 - 60)

def suggest_titles():
    """For pending clips: transcribe the exported clip itself (game + mic audio, the
    real cut) and offer 3 title candidates. Run after the stream — it uses the GPU."""
    for r in [r for r in records if r["status"] == "pending" and not r.get("candidates")]:
        if not os.path.exists(r["path"]):
            continue
        set_notice(f"Suggesting titles for {os.path.basename(r['path'])}…", "work")
        try:
            text = clip_editor.transcribe_words(r["path"], cfg["whisper_model"])["text"]
            text = strip_lead_junk(dedup(" ".join(text.split())))
            if text:
                r["raw"] = text
            r["candidates"] = ai_titles(r["raw"], r["game"], 3)
            save_records()
        except Exception as e:
            print(f"⚠️ suggest: {e}")
    set_notice("Title suggestions ready", "ok")

def update_record(data):
    """Dashboard edits: title, status (approve / skip / back to pending / retry), related-video tick."""
    r = get_record(str(data.get("id", "")))
    if not r:
        raise ValueError("unknown clip")
    if "title" in data and str(data["title"]).strip():
        r["title"] = clean(str(data["title"]).strip())[:MAX_TITLE_LENGTH * 2]
    if "related" in data:
        r["related"] = bool(data["related"])
    st = data.get("status")
    if st in ("approved", "skipped", "pending") and r["status"] in ("pending", "skipped", "failed", "approved"):
        r.update(status=st, retry_after=None, error="")
        if st == "approved":
            enqueue(r)
    save_records()
    return r

def safe_filename(name):
    name = re.sub(r'[/:\\?%*|"<>\x00-\x1f]', "-", name).strip(" .-")
    return (name[:80].rstrip() or "Clip")

def wait_for_title(timeout=12.0):
    """Block while a /clip is mid-flight (title_pending) or none has run yet, so
    /name and /upload use the CURRENT clip's title — not the previous one. Fixes
    the off-by-one when Aitum fires /clip and /name/upload without waiting for
    /clip's response."""
    start = time.time()
    while (title_pending or not last_title) and time.time() - start < timeout:
        time.sleep(0.1)
    return bool(last_title)

def wait_for_fresh_clip(timeout=20.0):
    """Return the exported clip for the latest trigger: the newest video whose
    mtime is at/after last_trigger_ts. OBS finishes writing the vertical export a
    few seconds after /clip, so /name and /upload must WAIT for it — otherwise
    they grab a stale/previous file (or re-touch an already-uploaded one). Falls
    back to newest-overall on timeout, and to any file when no trigger has run."""
    start, told = time.time(), False
    while time.time() - start < timeout:
        p = find_latest_clip()
        if not last_trigger_ts:
            return p
        try:
            if p and os.path.getmtime(p) >= last_trigger_ts - 1:  # 1s fs-granularity slack
                return p
        except OSError:
            pass
        if not told:   # only announce if OBS hasn't finished the export yet
            set_notice("Waiting for OBS to finish exporting the clip…", "work")
            told = True
        time.sleep(0.3)
    return find_latest_clip()

def rename_latest_to_title():
    """Rename the newest clip in the OBS folder to the last AI title. Returns the
    new path, or None. mtime is preserved so it stays 'newest' for /upload."""
    wait_for_title()
    with clip_action_lock:
        path = wait_for_fresh_clip()
        if not path:
            set_notice("No clip found in the OBS clips folder to rename")
            return None
        if not last_title:
            set_notice("No title yet — trigger a clip first")
            return None
        d, ext = os.path.dirname(path), os.path.splitext(path)[1]
        base = safe_filename(last_title)
        new = os.path.join(d, base + ext)
        n = 2
        while os.path.exists(new) and os.path.abspath(new) != os.path.abspath(path):
            new = os.path.join(d, f"{base} ({n}){ext}"); n += 1
        if os.path.abspath(new) == os.path.abspath(path):
            return path
        try:
            os.rename(path, new)
            set_notice(f"Renamed clip → {os.path.basename(new)}", "ok")
            return new
        except OSError as e:
            set_notice(f"Rename failed: {e}")
            return None


def run_clip_editor_job(target_path, title=None, story_cut=True):
    """Refactored helper: Runs clip_editor pipeline with unified config options."""
    if not cfg.get("enable_auto_editor", True):
        return target_path
    try:
        import clip_editor
        set_notice(f"Editing clip {os.path.basename(target_path)}…", "work")
        edited = clip_editor.process_and_edit_clip(
            target_path,
            title=title or last_title or "Stream Highlight",
            model_repo=cfg.get("whisper_model", "mlx-community/whisper-large-v3-turbo"),
            options={
                "smart_trim": cfg.get("smart_trim_silence", True),
                "story_cut": bool(story_cut),
                "ollama_model": cfg.get("ollama_model", "qwen2.5:latest"),
                "sub_color": cfg.get("sub_color", "yellow"),
                "sub_size": cfg.get("sub_size", 64),
                "sub_position": cfg.get("sub_position", "lower_third"),
                "max_silence_gap_sec": cfg.get("max_silence_gap_sec", 6.0),
                "preserve_story_span": cfg.get("preserve_story_span", True),
                "segment_padding_sec": cfg.get("segment_padding_sec", 0.5),
                "enable_cta": True,
                "show_banner": cfg.get("show_banner", True),
            }
        )
        if edited and os.path.exists(edited):
            set_notice(f"Clip edited → {os.path.basename(edited)}", "ok")
            return edited
    except Exception as e:
        print(f"⚠️ Clip editor error: {e}")
        set_notice(f"Clip editing failed: {e}")
    return target_path

def do_upload():
    """Aitum webhook after OBS exports the vertical clip: record it for review
    (default) or, with review off, edit + publish right away."""
    if not cfg["enable_yt"]:
        set_notice("YouTube uploads disabled in settings", "ok")
        return
    path = rename_latest_to_title() or wait_for_fresh_clip()
    if not path or not os.path.exists(path):
        set_notice("No clip available to upload", "err")
        return
    if any(r["path"] == path for r in records):
        set_notice("Newest clip is already queued — skipping duplicate", "ok")
        return
    # Snapshot now: a later /clip must not relabel this one.
    rec = {"id": str(int(time.time() * 1000)), "created": time.time(), "path": path,
           "title": last_title or "Stream Highlight", "raw": last_raw, "game": detected_game,
           "candidates": [], "status": "pending"}
    records.append(rec)
    if cfg.get("review_uploads", True):
        save_records()
        n = sum(r["status"] == "pending" for r in records)
        set_notice(f"Clip queued for review ({n} waiting) — approve in the dashboard Queue tab", "ok")
    else:
        rec["status"] = "approved"
        save_records()
        enqueue(rec)

# ---------------- HTTP trigger (stdlib, for Stream Deck / hotkey / Aitum) ----------------

def apply_settings(data):
    """Update cfg from a settings dict (web form or API). Restarts mic if changed."""
    global recording_paused, whisper_ok
    mic_changed = "mic_device" in data and str(data["mic_device"]) != cfg["mic_device"]
    whisper_changed = "whisper_model" in data and str(data["whisper_model"]) != cfg["whisper_model"]
    for k in ("streamer_name", "twitch_channel", "default_game", "mic_device", "whisper_model", "ollama_model", "yt_privacy", "obs_clips_dir", "sub_color", "sub_position"):
        if k in data:
            cfg[k] = str(data[k])
    if "sub_size" in data:
        try:
            cfg["sub_size"] = int(data["sub_size"])
        except (TypeError, ValueError):
            pass
    if "custom_words" in data:
        raw = data["custom_words"] if isinstance(data["custom_words"], list) else str(data["custom_words"]).split(",")
        cfg["custom_words"] = [str(w).strip() for w in raw if str(w).strip()]
    for k in ("enable_yt", "enable_notif", "enable_clip", "enable_auto_editor", "smart_trim_silence", "review_uploads"):
        if k in data:
            cfg[k] = bool(data[k])
    if "publish_slots" in data:
        raw = data["publish_slots"] if isinstance(data["publish_slots"], list) else str(data["publish_slots"]).split(",")
        slots = [s.strip() for s in raw if re.fullmatch(r"([01]?\d|2[0-3]):[0-5]\d", str(s).strip())]
        if slots:
            cfg["publish_slots"] = slots
    if "max_upload_kbps" in data:
        try:
            cfg["max_upload_kbps"] = int(data["max_upload_kbps"])
        except (TypeError, ValueError):
            pass
    if "default_duration" in data:
        try:
            cfg["default_duration"] = int(data["default_duration"])
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

def status_json():
    yt_st = youtube_auth_status()
    return {
        "mic_volume": mic_volume,
        "paused": recording_paused,
        "whisper": whisper_ok,
        "whisper_err": whisper_err,
        "ollama": ollama_ok,
        "yt_authenticated": yt_st["authenticated"],
        "yt_status": yt_st["status"],
        "notice": notice if (time.time() - notice_ts < 30) else "",
        "notice_level": notice_level,
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
    dur_opts = "".join(f'<option value="{d}"{" selected" if cfg.get("default_duration", 30) == d else ""}>{d} seconds</option>' for d in (15, 30, 45, 60))
    repl = {
        "__STREAMER__": e(cfg["streamer_name"]), "__TWITCH__": e(cfg["twitch_channel"]),
        "__WORDS__": e(", ".join(cfg["custom_words"])), "__GAME__": e(cfg["default_game"]),
        "__MIC_OPTS__": opts, "__MODEL_OPTS__": model_opts, "__WHISPER_OPTS__": whisper_opts,
        "__DUR_OPTS__": dur_opts,
        "__OBS__": e(cfg["obs_clips_dir"]), "__KBPS__": e(cfg["max_upload_kbps"]),
        "__CID__": e(cfg["google_client_id"]),
        "__SEC_PH__": "•••••• saved" if cfg["google_client_secret"] else "GOCSPX-…",
        "__YT__": "checked" if cfg["enable_yt"] else "",
        "__REVIEW__": "checked" if cfg.get("review_uploads", True) else "",
        "__SLOTS__": e(", ".join(cfg["publish_slots"])), "__NOTIF__": "checked" if cfg["enable_notif"] else "",
        "__CLIP__": "checked" if cfg["enable_clip"] else "",
        "__PUB__": "selected" if cfg["yt_privacy"] == "public" else "",
        "__UNL__": "selected" if cfg["yt_privacy"] == "unlisted" else "",
        "__PRV__": "selected" if cfg["yt_privacy"] == "private" else "",
    }
    with open(os.path.join(HERE, "dashboard.html"), encoding="utf-8") as f:
        page = f.read()   # read per request: edit the page without restarting
    for k, v in repl.items():
        page = page.replace(k, str(v))
    return page

CROSS_SITE_OK = {"/oauth2callback"} | draw_showcase.READ_ONLY

class Handler(BaseHTTPRequestHandler):
    def handle_one_request(self):
        # The dashboard polls every 150ms; a reload mid-response raises
        # BrokenPipeError/ConnectionReset. Harmless — don't spam the log.
        try:
            super().handle_one_request()
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def log_message(self, *a):
        pass

    def _send(self, body, ctype="application/json", code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else body.encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def _forbidden(self, path):
        """Reject non-localhost Host headers (DNS rebinding) and cross-site browser
        requests (CSRF: any open web page can hit localhost). Only Google's OAuth
        redirect and the read-only overlay routes OBS browser sources load are exempt."""
        host = self.headers.get("Host", "").split(":")[0].lower()
        cross = self.headers.get("Sec-Fetch-Site", "").lower() == "cross-site"
        if host not in ("localhost", "127.0.0.1") or (cross and path not in CROSS_SITE_OK):
            self._send(b'{"error": "forbidden (cross-site/invalid host)"}', "application/json", 403)
            return True
        return False

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if self._forbidden(u.path):
            return

        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/dashboard"):
            self._send(dashboard_html(), "text/html; charset=utf-8")
        elif u.path in ("/icon.png", "/favicon.ico"):
            icon_path = os.path.join(HERE, "icon.png")
            if os.path.exists(icon_path):
                with open(icon_path, "rb") as f:
                    self._send(f.read(), "image/png")
            else:
                self._send(b"not found", "text/plain", 404)
        elif u.path == "/api/status":
            self._send(json.dumps(status_json()))
        elif u.path == "/api/queue":
            self._send(json.dumps([r for r in records if r["status"] != "skipped" or
                                   time.time() - r["created"] < 86400][-50:][::-1]))
        elif u.path == "/api/history":
            self._send(json.dumps(clip_history[-20:][::-1]))   # newest first
        elif u.path == "/clip":
            dur = max(5, min(BUFFER_SECONDS, int(q.get("duration", [cfg.get("default_duration", DEFAULT_CLIP_SECONDS)])[0])))
            title, raw = make_clip(dur, q.get("game", [""])[0])
            self._send(json.dumps({"title": title, "raw_transcript": raw}))
        elif u.path == "/category":
            self._send(json.dumps({"category": live_twitch_game()}))
        elif u.path == "/auth":
            threading.Thread(target=start_oauth, daemon=True).start()
            self._send(b'{"status":"authorizing"}')
        elif u.path == "/oauth2callback":
            err = q.get("error", [""])[0]
            msg = f"Authorization denied: {err}" if err else finish_oauth(q.get("code", [""])[0])
            self._send("<!doctype html><meta charset=utf-8><body style=\"font:16px -apple-system,sans-serif;"
                       "background:#0b0d12;color:#e8edf5;display:flex;align-items:center;justify-content:center;"
                       f"height:100vh;margin:0;text-align:center;padding:24px\"><div>🎬<br><br>{html.escape(msg)}</div></body>",
                       "text/html; charset=utf-8")
        elif u.path == "/upload":
            threading.Thread(target=do_upload, daemon=True).start()
            self._send(b'{"status":"uploading"}')
        elif u.path.startswith("/draw/"):
            draw_showcase.handle(self, u.path, q)
        elif u.path == "/edit":
            target_path = q.get("file", [""])[0] or wait_for_fresh_clip()
            # Security path sandbox (Bug #6): Restrict /edit target file paths to clip directories
            if target_path:
                abs_target = os.path.abspath(os.path.expanduser(target_path))
                allowed_dir = os.path.abspath(os.path.expanduser(cfg.get("obs_clips_dir", "~/Movies")))
                user_obs_dir = os.path.abspath(os.path.expanduser("~/OBS recordings"))
                if not (abs_target.startswith(allowed_dir) or abs_target.startswith(user_obs_dir)):
                    self._send(b'{"error": "forbidden: target file outside allowed clips directory"}', "application/json", 403)
                    return
                if os.path.exists(abs_target):
                    use_story = q.get("story_cut", ["true"])[0].lower() not in ("false", "0", "no")
                    threading.Thread(target=lambda: run_clip_editor_job(abs_target, story_cut=use_story), daemon=True).start()
                    self._send(json.dumps({"status": "editing", "file": abs_target, "story_cut": use_story}).encode())
                    return
            self._send(b'{"error": "file not found"}', "application/json", 404)
        elif u.path == "/name":
            threading.Thread(target=rename_latest_to_title, daemon=True).start()
            self._send(b'{"status":"renaming"}')
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
        if self._forbidden(path):
            return
        if path == "/settings":
            n = int(self.headers.get("Content-Length", 0))
            try:
                apply_settings(json.loads(self.rfile.read(n) or b"{}"))
                self._send(b'{"status":"saved"}')
            except Exception as ex:
                self._send(json.dumps({"status": "error", "message": str(ex)}).encode(), code=400)
        elif path == "/api/queue/update":
            n = int(self.headers.get("Content-Length", 0))
            try:
                self._send(json.dumps(update_record(json.loads(self.rfile.read(n) or b"{}"))))
            except Exception as ex:
                self._send(json.dumps({"error": str(ex)}), code=400)
        elif path == "/api/queue/suggest":
            threading.Thread(target=suggest_titles, daemon=True).start()
            self._send(b'{"status":"suggesting"}')
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
        self.yt_status_item = rumps.MenuItem("YouTube: Checking…", callback=self.auth_yt)
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
            self.yt_status_item,
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
        threading.Thread(target=start_oauth, daemon=True).start()

    def reload_cfg(self, _):
        load_config()
        self.sync_mic_checks()
        self.yt_item.title = f"YouTube Auto-Upload: {'ON' if cfg['enable_yt'] else 'OFF'}"
        threading.Thread(target=start_stream, daemon=True).start()

    def refresh(self, _):
        t = last_title if last_title else "(none)"
        self.title_item.title = f"Last Title: {t[:30]}"
        self.game_item.title = f"Category: {detected_game or '(auto)'}"
        st = youtube_auth_status()
        self.yt_status_item.title = f"YouTube: {'Connected ✅' if st['authenticated'] else 'Not Connected ⚠️'}"


if __name__ == "__main__":
    # Stream first: yield CPU to OBS and the game. Audio capture runs on CoreAudio's
    # real-time thread, which niceness doesn't touch, so the ring buffer never drops.
    os.nice(10)
    start_stream()
    threading.Thread(target=start_http, daemon=True).start()
    threading.Thread(target=health_loop, daemon=True).start()
    threading.Thread(target=warmup_whisper, daemon=True).start()
    threading.Thread(target=_upload_worker, daemon=True).start()
    threading.Thread(target=retry_loop, daemon=True).start()
    threading.Thread(target=stats_loop, daemon=True).start()
    print(f"READY — trigger: http://localhost:{HTTP_PORT}/clip")
    app = ClipApp()
    import AppKit  # menu-bar-only: no Dock icon (otherwise Python shows a Dock rocket)
    AppKit.NSApplication.sharedApplication().setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    app.run()
