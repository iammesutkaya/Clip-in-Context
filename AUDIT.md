# Clip Backtrack — Audit & Changes

## What it does
Menu-bar macOS app for streamers. Continuously records the mic into a rolling
30s RAM buffer. On trigger it transcribes the last N seconds with local Whisper,
rewrites the speech into a Twitch-style clip title via Ollama/OpenAI, copies it
to the clipboard + notifies, pushes it to Aitum Nexus, and can auto-upload a
YouTube Short of the newest OBS clip.

**Architecture:** Swift app (`ClipBacktrackApp.swift` → `ClipBacktrackExecutable`)
is the menu bar. It spawns the Python backend (`mac_clip_backtrack.py`, port 5001)
which does audio + STT + AI + a web dashboard. They talk over HTTP.

## Microphone bug — root cause
`"Voice - Input 01"` is a **Rogue Amoeba Loopback** virtual device (the user's
mic signal chain, gated → silent until speech). The app hard-coded
`SAMPLE_RATE = 16000` on the `InputStream`. Loopback runs natively at 48000 Hz
and does not resample itself, so opening it at 16 kHz produced broken/empty
capture. Hardware (Clarett) resamples in CoreAudio, which masked the bug on
that device.

Fixed by: open the device at its **native rate** and resample to 16 kHz for
Whisper (`scipy.signal.resample_poly` in the audio callback). Config points at
`Voice - Input 01`.

## Bugs fixed
| # | Severity | Bug | Fix |
|---|----------|-----|-----|
| 1 | Crash | `USE_MLX` / `mlx_whisper` referenced but never defined → NameError on **every** clip trigger | Removed dead MLX branch; use loaded `whisper_model`, guard `None` |
| 2 | Crash | `json` used in ~10 places (config, jargon, client_secret) but never imported | Added `import json` |
| 3 | Broken feature | Swift menu Pause/Resume hit `/pause` `/resume`; dashboard Auth hits `/auth_youtube` — **none existed** (404) | Added the three routes |
| 4 | Silent failure | 16 kHz forced on 48 kHz Loopback device → empty capture | Open at native rate, resample to 16 kHz |
| 5 | Security (XSS) | Spoken transcript, titles, device names injected into dashboard HTML unescaped | `html.escape(..., quote=True)` on all injected values |
| 6 | Footgun | YouTube auto-upload defaulted **on** + privacy **public** → a stray clip auto-publishes | Defaults now off + unlisted |
| 7 | Dead code | `NativeMenuBarController` (~110 lines) never instantiated — Swift owns the menu bar | Removed, plus its unused `AppKit` import |

## Still worth doing (not changed)
- **Secrets in plaintext.** Google client secret + OAuth token sit in the app
  dir and the secret is shown in a plaintext dashboard field. Move to Keychain.
- **Two competing implementations.** `run_backtrack.py` is an older MLX-based
  variant; the Python file also duplicates config/notify logic. Pick one.
- **Bundled copy drift.** `Clip Backtrack.app/.../mac_clip_backtrack.py` is a
  copy of the root file — they can silently diverge. Build step should copy, or
  the app should point at one source.
- **Buffer efficiency.** `AudioBuffer.get_audio_data` does `list(deque)[-n:]`
  (full O(n) copy of up to 960k samples) every 1.5s. Fine at this scale; switch
  to a numpy ring buffer if CPU matters.
- **No dependency manifest.** Needs a `requirements.txt`
  (`sounddevice numpy openai-whisper flask waitress requests google-api-python-client google-auth-oauthlib`).
- **Profanity filter** is a tiny regex list — trivially bypassed; fine as a
  courtesy, not brand-safety.
