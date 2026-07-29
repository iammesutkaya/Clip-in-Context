# Clip in Context

Rolling mic transcript backtrack + AI clip titler for macOS streamers.

Press a button → the last 30s of your speech is transcribed locally (MLX
Whisper), rewritten into a Twitch-style clip title (Ollama), copied to the
clipboard, pushed to Aitum, and optionally uploaded as a YouTube Short. All
local — no cloud, no per-clip cost.

```
mic → rolling 30s buffer → (trigger) → MLX Whisper → AI title
    → clipboard + notification → Aitum → optional YouTube Short
```

## Requirements
- **Apple Silicon Mac** (MLX Whisper needs an M-series chip).
- **[Ollama](https://ollama.com)** for AI titles: `ollama serve` running, with a
  model pulled — `ollama pull llama3.2` (or `qwen2.5`). Without it, titles fall
  back to the last spoken sentence (or OpenAI if `OPENAI_API_KEY` is set).
- Optional: OBS (for YouTube upload), Aitum Nexus (for the title variable).

## Install
```bash
git clone <this-repo> clip-in-context && cd clip-in-context
./setup.sh
```
`setup.sh` creates a venv, installs dependencies, registers an on-demand
LaunchAgent, and builds a double-clickable **Clip in Context.app**. The Whisper
model downloads automatically on first run.

> **Don't put this in ~/Desktop, ~/Documents, or ~/Downloads.** macOS protects
> those folders (TCC); a terminal without Files-and-Folders access can't write
> there and setup fails with *"Operation not permitted"*. Clone it under your
> home directory (e.g. `~/clip-in-context`) instead.

## Use
- **Start:** double-click **Clip in Context.app** (move it to /Applications or the
  Dock). A waveform icon appears in the menu bar. Grant microphone access the
  first time. It does **not** run at login — only when you launch it.
- **Stop:** Quit from the menu bar.
- **Dashboard:** menu bar → **Open Dashboard…** or `http://localhost:5001/` —
  live captions, mic meter, trigger, recent clips, engine status, and settings.

### Dock it inside OBS
The dashboard works great as an OBS **custom browser dock** (fully interactive,
unlike a browser source): **OBS → Docks → Custom Browser Docks…** → name it
"Clip in Context", URL `http://localhost:5001` → Apply. The trigger button, live
captions, and recent clips now live right in your OBS window. The layout adapts
to narrow dock widths.

### Design note
One Python process, launched via launchd's GUI session, so it inherits
microphone permission — **no signed `.app`, no Swift, no code signing, no macOS
TCC silence** (the trap an earlier `.app` version fell into). Menu bar via
`rumps`; dashboard served by the stdlib HTTP server (no web framework).

## Trigger
- Menu bar / dashboard **Trigger Clip Now**, or
- HTTP (Stream Deck / Aitum): `GET http://localhost:5001/clip?duration=30&game=Valorant`
- `GET /pause`, `GET /resume`.

### Global hotkey (native macOS, no extra permission)
Bind any key with **Shortcuts.app** — no daemon, no Accessibility prompt:
1. Shortcuts.app → new Shortcut → **Get Contents of URL** → `http://localhost:5001/clip?duration=30`.
2. Shortcut Details → **Add Keyboard Shortcut** → pick your key (e.g. ⌃⌥C).

Raycast / BetterTouchTool work too — anything that runs a URL on a hotkey.

## Config
Everything is editable from the dashboard (or the menu bar for mic / YouTube).
Advanced fields live in `config.json`. Secrets (`config.json`,
`client_secret.json`, `youtube_token.json`) are gitignored and never sent back
to the browser.

## Recent clips & errors
The dashboard lists recent clips (persisted to `clips.jsonl`, **Clear all** to
wipe) and shows a warning banner when something degrades — Whisper failed to
load, Ollama unreachable, or a YouTube upload failed.

## Tests
```bash
./.venv/bin/python3 test_logic.py   # dedup, repetitive, RingBuffer edge cases
```

## Uninstall
```bash
launchctl unload -w ~/Library/LaunchAgents/com.clipincontext.app.plist
rm ~/Library/LaunchAgents/com.clipincontext.app.plist
```
Then delete the folder and `Clip in Context.app`. Logs: `/tmp/clipincontext.log`.

## License
MIT — see [LICENSE](LICENSE).
