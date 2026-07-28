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

## Second pass (done)
- **Secret exposure.** Dashboard no longer echoes the OAuth client secret:
  field is `type=password`, value never sent to the browser, `/api/settings`
  returns only a `google_client_secret_set` bool. (Full Keychain storage
  skipped: installed-app OAuth secrets aren't confidential and files are
  gitignored + home-owned.)
- **Two implementations.** Deleted `run_backtrack.py` (older MLX prototype,
  unreferenced). `mac_clip_backtrack.py` is the single source.
- **Bundle drift.** `build.sh` compiles the Swift binary and regenerates the
  whole `.app` (incl. Info.plist) from source, so the bundled Python copy can't
  diverge. The `.app` is gitignored and rebuildable.
- **Buffer efficiency.** `AudioBuffer` is now a fixed numpy int16 ring buffer —
  no per-sample Python objects, O(1) writes. Covered by `test_audiobuffer.py`.
- **Dependency manifest.** Added `requirements.txt`.

## Still open (intentionally not done)
- **Profanity filter** is a tiny regex list — courtesy filter, not real brand
  safety. Left as-is by design.
- **Swift hardcodes** a fallback path `/Users/Mesut/Desktop/clip/...`; harmless
  (bundle resource is primary) but not portable.
