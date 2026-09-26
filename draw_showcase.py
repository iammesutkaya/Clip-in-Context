"""
draw_showcase.py — Speed Draw overlay routes (/draw/*), served by clip_in_context's
HTTP server so the OBS overlays keep their localhost:5001 URLs.

Callers:
  - OBS browser sources (speed-draw-showcase.html, speed-draw-hud.html) read
    /draw/events, /draw/latest_status, /draw/latest_image, /draw/active.
  - ~/OBS source/04_misc/04_scripts/draw_game.py drives the rest.
"""
import os, sys, json, time, queue, subprocess

SCREENSHOTS_DIR = os.path.expanduser("~/OBS source/04_misc/03_screenshots")
DRAW_SCRIPT = os.path.expanduser("~/OBS source/04_misc/04_scripts/draw_game.py")
AITUM_STATE_DB = os.path.expanduser("~/.aitum/state.db")

# Read-only routes the overlays load from a browser source (cross-site allowed).
READ_ONLY = {"/draw/events", "/draw/latest_status", "/draw/latest_image", "/draw/active"}

_state = {"timestamp": 0, "event": "showcase", "number": 1, "prompt": "", "requester": "", "image_path": ""}
_sse_queues = set()


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
        h._send(json.dumps(aitum_drawing_state()))
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
