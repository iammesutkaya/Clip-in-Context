#!/bin/bash
# Build "Clip Backtrack.app" from source. The bundle is gitignored and fully
# regenerated here, so the bundled Python copy never drifts from the source.
set -euo pipefail
cd "$(dirname "$0")"

APP="Clip Backtrack.app"

swiftc ClipBacktrackApp.swift -o ClipBacktrackExecutable

mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp ClipBacktrackExecutable   "$APP/Contents/MacOS/ClipBacktrackExecutable"
cp mac_clip_backtrack.py      "$APP/Contents/Resources/mac_clip_backtrack.py"
cp AppIcon.icns               "$APP/Contents/Resources/AppIcon.icns"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key><string>ClipBacktrackExecutable</string>
    <key>CFBundleIdentifier</key><string>com.mesut.clipbacktrack</string>
    <key>CFBundleName</key><string>Clip Backtrack</string>
    <key>CFBundleDisplayName</key><string>Clip Backtrack</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleShortVersionString</key><string>1.0</string>
    <key>CFBundleVersion</key><string>1</string>
    <key>CFBundleIconFile</key><string>AppIcon</string>
    <key>LSUIElement</key><true/>
    <key>NSMicrophoneUsageDescription</key>
    <string>Clip Backtrack requires microphone access to record audio transcripts for stream clipping.</string>
</dict>
</plist>
PLIST

echo "Built $APP"
