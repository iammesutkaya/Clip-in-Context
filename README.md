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

## Launch without the terminal (only when you stream)
Double-click **Clip Backtrack.app** to start it; **Quit** from the menu bar to
stop. It does NOT run at login — only when you click. The app just tells an
idle LaunchAgent to start (launchd gives it the GUI session + mic access; a
plain detached process can't run the menu bar).

One-time setup:
```bash
# register the idle agent (does not run until kickstarted)
cp com.mesut.clipbacktrack.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.mesut.clipbacktrack.plist
# build the double-clickable launcher
osacompile -o "Clip Backtrack.app" -e 'do shell script "launchctl kickstart gui/$(id -u)/com.mesut.clipbacktrack"'
cp AppIcon.icns "Clip Backtrack.app/Contents/Resources/applet.icns"   # app icon
```
Move **Clip Backtrack.app** to /Applications if you like. Logs: `/tmp/clipbacktrack.log`.
If captions ever stay silent, grant **Python** in System Settings → Privacy &
Security → Microphone. After editing the code, just Quit and relaunch.

Remove it entirely:
```bash
launchctl unload -w ~/Library/LaunchAgents/com.mesut.clipbacktrack.plist
rm ~/Library/LaunchAgents/com.mesut.clipbacktrack.plist
```

## Tests
```bash
python3 test_logic.py   # dedup, repetitive, RingBuffer edge cases
```

## Requires
Apple Silicon (MLX). Ollama with `llama3.2` for AI titles (falls back to the
last spoken sentence, or `OPENAI_API_KEY` if set). Aitum Nexus + OBS optional.
