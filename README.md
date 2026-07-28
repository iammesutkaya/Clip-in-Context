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

## Trigger
- Menu bar → **Trigger Clip Now**, or
- HTTP (Stream Deck / hotkey / Aitum): `GET http://localhost:5001/clip?duration=30&game=Valorant`
- Also `GET /pause` and `/resume`.

## Config
Menu bar covers mic device, YouTube on/off, credentials, and auth. Everything
else (streamer name, Twitch channel, custom words, default game, upload privacy,
OBS clips dir) lives in `config.json` — **Edit Settings** opens it, **Reload
Settings** applies it. Secrets (`config.json`, `client_secret.json`,
`youtube_token.json`) are gitignored.

## Optional: auto-start
Add a `~/Library/LaunchAgents` plist running `python3 .../clip_backtrack.py`.
Grant the mic prompt once on first launch.

## Requires
Apple Silicon (MLX). Ollama with `llama3.2` for AI titles (falls back to the
last spoken sentence, or `OPENAI_API_KEY` if set). Aitum Nexus + OBS optional.
