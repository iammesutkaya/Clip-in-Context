"""
draw_showcase.py — Speed Draw overlay routes (/draw/*), served by clip_in_context's
HTTP server so the OBS overlays keep their localhost:5001 URLs.

Callers:
  - OBS browser sources (speed-draw-showcase.html, speed-draw-hud.html) read
    /draw/events, /draw/latest_status, /draw/latest_image, /draw/active.
  - ~/OBS source/04_misc/04_scripts/draw_game.py drives the rest.

Request screening: viewer text from the channel-point reward is read aloud (TTS),
shown on screen and posted to Discord, and Twitch AutoMod doesn't reliably hold it
before Aitum sees it. So Aitum's redemption rule calls /draw/request instead of
starting the flow; clean text starts the flow right away, flagged text waits for
/draw/approve or /draw/reject (menu bar, Stream Deck).
"""
import os, re, sys, json, time, queue, threading, subprocess
import requests

SCREENSHOTS_DIR = os.path.expanduser("~/OBS source/04_misc/03_screenshots")
DRAW_SCRIPT = os.path.expanduser("~/OBS source/04_misc/04_scripts/draw_game.py")
AITUM_STATE_DB = os.path.expanduser("~/.aitum/state.db")
AITUM_API = "http://localhost:7777/aitum"
FLOW_RULE_ID = "9vQv94t7VJPs06El"   # Aitum rule "Speed Draw - 60s Flow"
# Aitum global variables the flow reads (TTS, OBS text, screenshot file name).
STATE_IDS = {"prompt": "ZxEU6grJ9BqfYiF1", "requester": "fvPCK9poni8q1OlO", "number": "5LlJzC7jkquvSC27"}

# Set by clip_in_context at import time (avoids a circular import).
llm = lambda prompt, max_tokens=25: ""      # → completion text, "" when no model is reachable
notice = lambda msg, level="warn": print(msg)

# Read-only routes the overlays load from a browser source (cross-site allowed).
READ_ONLY = {"/draw/events", "/draw/latest_status", "/draw/latest_image", "/draw/active"}

_state = {"timestamp": 0, "event": "showcase", "number": 1, "prompt": "", "requester": "", "image_path": ""}
_sse_queues = set()


# ---------------- request screening ----------------
LEET = str.maketrans("013457@$!|", "oieastasil")
# After folding (lowercase, leetspeak, repeats squashed: "fuuuck" → "fuck"):
# ANYWHERE roots are matched even with spacing tricks ("f u c k", "motherfucker");
# WORD roots only as whole words, so "grape", "peacock", "Essex" stay clean.
# The LLM check below catches what a list can't (misspellings, innuendo, context).
ANYWHERE = re.compile(r"fuck|shit|cunt|bitch|whore|slut|porn|penis|vagina|retard|rapist|"
                      r"niga|niger|fagot|trany|nazi|hitler|suicide")
WORD = re.compile(r"\b(pusy|sex|sexy|nude|naked|tits?|fags?|spics?|chinks?|"
                  r"rape|raped|kys|kil yourself)\b")
PERSONAL = re.compile(r"\b\d{1,5}\s+\w+(\s\w+)?\s+(street|st|road|rd|avenue|ave|lane|ln|drive|dr|straße|str|weg|platz)\b"
                      r"|\d[\d\s()+-]{8,}\d|\b[\w.+-]+@[\w-]+\.\w+", re.IGNORECASE)   # address, phone, email
LINK = re.compile(r"https?://|www\.|\b[\w-]+\.(com|net|org|gg|tv|ly|io|xyz)\b", re.IGNORECASE)

def _fold(text, keep_spaces):
    t = text.lower().translate(LEET)
    t = re.sub(r"[^a-z ]" if keep_spaces else r"[^a-z]", "", t)
    return re.sub(r"(.)\1+", r"\1", t)     # squash repeats

def screen_prompt(text, ask=None):
    """None if the request is fine to show and read aloud, else a short reason.
    Fails closed: if the model can't answer, a human approves."""
    text = (text or "").strip()
    if not text:
        return None
    if LINK.search(text):
        return "contains a link"
    if PERSONAL.search(text):
        return "looks like personal info"
    m = ANYWHERE.search(_fold(text, False)) or WORD.search(_fold(text, True))
    if m:
        return f"blocked word ({m.group(0)[:2]}…)"
    ask = ask or llm
    verdict = ask(
        "A viewer submitted this drawing request on a family-friendly Twitch stream. It will be "
        "read aloud by text-to-speech, shown on screen and posted to Discord:\n"
        f'"{text}"\n\n'
        "Answer UNSAFE if it contains profanity (even disguised), slurs, sexual content or innuendo, "
        "harassment of a person, self-harm, violence against real people, personal information, or "
        "anything a streamer would regret reading aloud. Otherwise answer SAFE. One word only.", 5)
    v = verdict.strip().upper()
    if v.startswith("SAFE"):
        return None
    return "AI filter flagged it" if v.startswith("UNSAFE") else "filter unavailable (Ollama down)"

_pending = None   # {"prompt", "requester", "reason"} awaiting approval, or None
_current = None   # last APPROVED request {"prompt", "requester", "number"} — all the overlays ever see

def pending():
    return _pending

def current():
    global _current
    if _current is None:   # first call after a restart: last request Aitum saved
        _current = aitum_drawing_state()
        _current.pop("mtime", None)
    return _current

def _aitum_vars():
    """Aitum's live variables by name (its API), else what it last saved to state.db."""
    try:
        data = requests.get(f"{AITUM_API}/state", timeout=2).json().get("data", [])
        vals = {v.get("name"): v.get("value") for v in data}
        if "Drawing Request" in vals:
            return {"prompt": str(vals["Drawing Request"] or ""), "requester": str(vals.get("Drawing Requester") or ""),
                    "number": int(vals.get("Drawing Request Number") or 0)}
    except Exception:
        pass
    st = aitum_drawing_state()
    return {"prompt": st["prompt"], "requester": st["requester"], "number": st["number"]}

def _set_aitum_var(key, value):
    requests.put(f"{AITUM_API}/state/{STATE_IDS[key]}", json={"value": value}, timeout=2)

def start_flow():
    """Run the Speed Draw flow in Aitum (rules are called "commands" in its API)."""
    try:
        r = requests.get(f"{AITUM_API}/commands/{FLOW_RULE_ID}", timeout=3)
        if r.status_code == 200:
            return True
        notice(f"Aitum didn't start the draw flow (HTTP {r.status_code}) — run it from Aitum")
    except Exception as e:
        notice(f"Couldn't reach Aitum to start the draw flow: {e}")
    return False

def accept(req):
    """A request passed (or was approved): number it, write it back into Aitum — a newer
    redemption may have overwritten the variables while this one was held — show it on
    the overlays, and start the flow. Numbering here means rejected requests leave no gaps."""
    global _current
    num = _aitum_vars()["number"] + 1
    try:
        for key, value in (("prompt", req["prompt"]), ("requester", req["requester"]), ("number", num)):
            _set_aitum_var(key, value)
    except Exception as e:
        notice(f"Couldn't update Aitum's draw variables: {e}")
    _current = {"prompt": req["prompt"], "requester": req["requester"], "number": num}
    broadcast({"event": "request", **_current})
    start_flow()

def handle_request(prompt=None, requester=None):
    global _pending
    if prompt is None:
        time.sleep(0.5)   # let Aitum finish writing the variables its rule just set
        v = _aitum_vars()
        prompt, requester = v["prompt"], v["requester"]
    req = {"prompt": prompt, "requester": requester or ""}
    reason = screen_prompt(prompt)
    if reason is None:
        accept(req)   # leaves any other held request alone
        return
    _pending = {**req, "reason": reason}
    notice(f"Draw request from {requester or 'a viewer'} held: {reason} — Approve/Reject in the menu bar")
    subprocess.run(["osascript", "-e", 'display notification "Held for review — Approve or Reject '
                    'from the menu bar" with title "🎨 Draw request" sound name "Submarine"'],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def resolve(approve):
    """Approve (start the flow) or reject (drop it) the held request. Returns True if one was held."""
    global _pending
    if not _pending:
        return False
    held, _pending = _pending, None
    if approve:
        accept(held)
    else:
        notice(f"Rejected draw request from {held['requester'] or 'a viewer'} — refund it in Twitch's reward queue", "ok")
    return True

def _screenshot_path(p):
    """Only PNGs inside the screenshots folder may be served — /draw/latest_image
    is readable cross-site, so an arbitrary path here would leak any file."""
    if not p:
        return None
    real = os.path.realpath(os.path.expanduser(p))
    root = os.path.realpath(SCREENSHOTS_DIR)
    if real.lower().endswith(".png") and os.path.commonpath([real, root]) == root and os.path.isfile(real):
        return real
    return None


def _newest_screenshot():
    try:
        pngs = [os.path.join(SCREENSHOTS_DIR, f) for f in os.listdir(SCREENSHOTS_DIR) if f.endswith(".png")]
    except OSError:
        return None
    return max(pngs, key=os.path.getmtime) if pngs else None


def broadcast(event):
    payload = f"data: {json.dumps(event)}\n\n".encode("utf-8")
    for q in list(_sse_queues):
        try:
            q.put_nowait(payload)
        except Exception:
            pass


def notify_showcase(number, prompt, requester, image_path):
    global _state
    _state = {
        "timestamp": time.time(), "event": "showcase", "number": number,
        "prompt": prompt, "requester": requester,
        "image_path": _screenshot_path(image_path) or "",
        "image_url": f"http://localhost:5001/draw/latest_image?t={int(time.time()*1000)}",
    }
    broadcast(_state)


def aitum_drawing_state():
    data = {"number": 1, "prompt": "Freestyle Sketch", "requester": "Stream Viewer", "mtime": 0}
    if not os.path.exists(AITUM_STATE_DB):
        return data
    try:
        data["mtime"] = os.path.getmtime(AITUM_STATE_DB)
        with open(AITUM_STATE_DB, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    name, val = entry.get("name"), entry.get("value")
                    if name == "Drawing Request" and val:
                        data["prompt"] = str(val)
                    elif name == "Drawing Requester" and val:
                        data["requester"] = str(val)
                    elif name == "Drawing Request Number" and val is not None:
                        data["number"] = int(val)
                except Exception:
                    pass
    except Exception:
        pass
    return data


def _sse(h):
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Cache-Control", "no-cache")
    h.send_header("Connection", "keep-alive")
    h.send_header("Access-Control-Allow-Origin", "*")
    h.end_headers()
    q = queue.Queue()
    _sse_queues.add(q)
    try:
        h.wfile.write(b": keepalive\n\n")
        h.wfile.flush()
        while True:
            try:
                h.wfile.write(q.get(timeout=20))
            except queue.Empty:
                h.wfile.write(b": ping\n\n")
            h.wfile.flush()
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        _sse_queues.discard(q)


def _image(h):
    img = _screenshot_path(_state.get("image_path")) or _newest_screenshot()
    if not img:
        h._send(b'{"error":"no image found"}', "application/json", 404)
        return
    with open(img, "rb") as f:
        data = f.read()
    h.send_response(200)
    h.send_header("Content-Type", "image/png")
    h.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
    h.send_header("Access-Control-Allow-Origin", "*")
    h.end_headers()
    h.wfile.write(data)


def handle(h, path, q):
    """Serve one /draw/* GET on BaseHTTPRequestHandler `h` (needs h._send)."""
    if path == "/draw/events":
        _sse(h)
    elif path == "/draw/latest_status":
        h._send(json.dumps(_state))
    elif path == "/draw/latest_image":
        _image(h)
    elif path == "/draw/active":
        h._send(json.dumps(current()))   # approved requests only — never raw redemption text
    elif path == "/draw/request":
        # Aitum's redemption rule calls this; ?prompt= lets you test screening by hand.
        pr = q.get("prompt", [None])[0]
        threading.Thread(target=handle_request, args=(pr, q.get("requester", [None])[0]), daemon=True).start()
        h._send(b'{"status":"screening"}')
    elif path in ("/draw/approve", "/draw/reject"):
        ok = resolve(path == "/draw/approve")
        h._send(json.dumps({"status": ("approved" if path == "/draw/approve" else "rejected") if ok else "nothing held"}))
    elif path == "/draw/pending":
        h._send(json.dumps(_pending))
    elif path == "/draw/upload":
        subprocess.Popen([sys.executable, DRAW_SCRIPT, "upload"])
        h._send(b'{"status":"uploading_drawing"}')
    elif path == "/draw/notify_showcase":
        try:
            num = int(q.get("num", ["1"])[0])
        except ValueError:
            num = 1
        notify_showcase(num, q.get("prompt", ["Freestyle Sketch"])[0],
                        q.get("requester", ["Stream Viewer"])[0], q.get("path", [""])[0])
        h._send(b'{"status":"notified"}')
    elif path == "/draw/test_showcase":
        notify_showcase(5, "A test - the draw game works... I think", "MesutKaya", _newest_screenshot() or "")
        h._send(b'{"status":"test_triggered"}')
    elif path == "/draw/countdown":
        try:
            sec = int(q.get("seconds", ["60"])[0])
        except ValueError:
            sec = 60
        broadcast({"event": "countdown", "seconds": sec})
        h._send(json.dumps({"status": "countdown_started", "seconds": sec}))
    elif path == "/draw/stop_countdown":
        broadcast({"event": "stop_countdown"})
        h._send(b'{"status":"countdown_stopped"}')
    elif path in ("/draw/dismiss", "/draw/peel", "/draw/hide", "/draw/clear"):
        broadcast({"event": "dismiss"})
        h._send(b'{"status":"dismissed"}')
    elif path in ("/draw/restore", "/draw/show"):
        broadcast({"event": "restore"})
        h._send(b'{"status":"restored"}')
    else:
        h._send(b"not found", "text/plain", 404)
