#!/bin/bash
# One-time setup: create a venv, install deps, register the on-demand LaunchAgent,
# and build the double-clickable launcher. Re-run any time to update.
set -euo pipefail
cd "$(dirname "$0")"
DIR="$(pwd)"
LABEL="com.titledrop.app"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

command -v python3 >/dev/null || { echo "python3 not found — install it (e.g. brew install python)"; exit 1; }
[ "$(uname -m)" = "arm64" ] || echo "⚠️  Not Apple Silicon — MLX Whisper needs an M-series Mac."

echo "→ Creating venv + installing dependencies (this downloads a lot the first time)…"
python3 -m venv .venv
./.venv/bin/pip install -q --upgrade pip
./.venv/bin/pip install -q -r requirements.txt
PY="$DIR/.venv/bin/python3"

echo "→ Registering LaunchAgent ($LABEL, idle — runs only when you launch it)…"
mkdir -p "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key><string>$LABEL</string>
    <key>ProgramArguments</key>
    <array><string>$PY</string><string>$DIR/titledrop.py</string></array>
    <key>WorkingDirectory</key><string>$DIR</string>
    <key>RunAtLoad</key><false/>
    <key>KeepAlive</key><false/>
    <key>StandardOutPath</key><string>/tmp/titledrop.log</string>
    <key>StandardErrorPath</key><string>/tmp/titledrop.log</string>
</dict>
</plist>
EOF
launchctl unload "$PLIST" 2>/dev/null || true
launchctl load -w "$PLIST"

echo "→ Building 'TitleDrop.app' launcher…"
rm -rf "TitleDrop.app"
osacompile -o "TitleDrop.app" -e "do shell script \"launchctl kickstart gui/\$(id -u)/$LABEL\"" >/dev/null
[ -f AppIcon.icns ] && cp AppIcon.icns "TitleDrop.app/Contents/Resources/applet.icns"

cat <<DONE

✓ Done.
  • Start:  double-click "TitleDrop.app" (or move it to /Applications).
  • Stop:   Quit from the menu bar.
  • Needs:  Ollama running (ollama serve) with a model pulled (e.g. ollama pull llama3.2).
            The Whisper model downloads automatically on first run.
  • Logs:   /tmp/titledrop.log
DONE
