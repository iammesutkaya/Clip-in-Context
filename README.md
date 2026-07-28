# Clip Backtrack

Rolling mic transcript backtrack + AI clip titler for macOS streamers.

Press a button → the last 30s of your speech is transcribed locally (MLX
Whisper), rewritten into a Twitch-style clip title (Ollama), copied to the
clipboard, pushed to Aitum, and optionally uploaded as a YouTube Short.

## Design
One Python process, launched from the Terminal. That's deliberate: a
terminal-launched process inherits microphone permission, so there is **no .app
bundle, no Swift, no code signing, and no macOS TCC silence** — the exact trap
the earlier `.app` version fell into. Menu bar via `rumps`.

```
mic → rolling 30s buffer → (trigger) → MLX Whisper → AI title
    → clipboard + notification → Aitum → optional YouTube Short
```

## Run
```bash
pip install -r requirements.txt
python3 clip_backtrack.py
```
A 🎙️ icon appears in the menu bar. Grant microphone access the first time.

## Dashboard
Menu bar → **Open Dashboard…** (or visit `http://localhost:5001/`). Served by
the built-in stdlib HTTP server — no extra dependency. It has a **live mic VU
meter** (instant confirmation the mic works), the trigger button, latest title +
transcript, live Twitch category, and a full settings form (streamer, mic
device, custom words, YouTube, privacy, preferences). Settings save straight to
`config.json`; the OAuth secret is a password field and is never sent back to
the browser.

## Trigger
- Menu bar → **Trigger Clip Now**, dashboard button, or
- HTTP (Stream Deck / hotkey / Aitum): `GET http://localhost:5001/clip?duration=30&game=Valorant`
- Also `GET /pause` and `/resume`.

## Config
Edit from the dashboard, or the menu bar (mic, YouTube toggle/creds/auth), or
`config.json` directly (**Edit Settings** / **Reload Settings** in the menu).
Secrets (`config.json`, `client_secret.json`, `youtube_token.json`) are
gitignored.

## Optional: auto-start
Add a `~/Library/LaunchAgents` plist running `python3 .../clip_backtrack.py`.
Grant the mic prompt once on first launch.

## Requires
Apple Silicon (MLX). Ollama with `llama3.2` for AI titles (falls back to the
last spoken sentence, or `OPENAI_API_KEY` if set). Aitum Nexus + OBS optional.
