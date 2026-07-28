import AppKit
import AVFoundation

class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    var statusItem: NSStatusItem!
    var pythonProcess: Process?
    var isPaused = false
    var lastTitleItem: NSMenuItem!
    var categoryItem: NSMenuItem!
    var pauseItem: NSMenuItem!
    var pollTimer: Timer?

    func applicationDidFinishLaunching(_ aNotification: Notification) {
        // Request macOS Microphone Permission explicitly on app launch
        if AVCaptureDevice.authorizationStatus(for: .audio) == .notDetermined {
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                print("🎙️ macOS Microphone Permission Granted: \(granted)")
            }
        }

        // 1. Create native macOS Status Bar item with ONLY the waveform.mid SF Symbol icon
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        if let button = statusItem.button {
            button.image = NSImage(systemSymbolName: "waveform.mid", accessibilityDescription: "Clip Backtrack")
            button.title = ""
        }

        // 2. Build Native NSMenu with native SF Symbols for every menu item
        let menu = NSMenu()
        menu.delegate = self

        pauseItem = NSMenuItem(title: "Pause Recording", action: #selector(togglePause), keyEquivalent: "")
        pauseItem.image = NSImage(systemSymbolName: "pause.circle", accessibilityDescription: nil)
        pauseItem.target = self
        menu.addItem(pauseItem)

        let triggerItem = NSMenuItem(title: "Trigger Clip Now", action: #selector(triggerClip), keyEquivalent: "")
        triggerItem.image = NSImage(systemSymbolName: "scissors", accessibilityDescription: nil)
        triggerItem.target = self
        menu.addItem(triggerItem)

        menu.addItem(NSMenuItem.separator())

        lastTitleItem = NSMenuItem(title: "Last Title: (None)", action: nil, keyEquivalent: "")
        lastTitleItem.image = NSImage(systemSymbolName: "doc.on.clipboard", accessibilityDescription: nil)
        menu.addItem(lastTitleItem)

        categoryItem = NSMenuItem(title: "Live Category: (Auto-Detecting...)", action: nil, keyEquivalent: "")
        categoryItem.image = NSImage(systemSymbolName: "gamecontroller", accessibilityDescription: nil)
        menu.addItem(categoryItem)

        menu.addItem(NSMenuItem.separator())

        let settingsItem = NSMenuItem(title: "Dashboard...", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.image = NSImage(systemSymbolName: "gearshape", accessibilityDescription: nil)
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(NSMenuItem.separator())

        let quitItem = NSMenuItem(title: "Quit Clip Backtrack", action: #selector(quitApp), keyEquivalent: "q")
        quitItem.image = NSImage(systemSymbolName: "xmark.circle", accessibilityDescription: nil)
        quitItem.target = self
        menu.addItem(quitItem)

        statusItem.menu = menu

        // 3. Launch Python Speech & AI Backend
        startPythonBackend()

        // 4. Start background polling timer (every 1 second) to keep Last Title live
        pollTimer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.fetchLastTitle()
        }
    }

    func menuWillOpen(_ menu: NSMenu) {
        fetchLastTitle()
    }

    func startPythonBackend() {
        // Clean up any lingering python backends before starting
        let pkill = Process()
        pkill.executableURL = URL(fileURLWithPath: "/usr/bin/pkill")
        pkill.arguments = ["-9", "-f", "mac_clip_backtrack.py"]
        try? pkill.run()
        pkill.waitUntilExit()

        let scriptPath = Bundle.main.path(forResource: "mac_clip_backtrack", ofType: "py") ?? "/Users/Mesut/Desktop/clip/mac_clip_backtrack.py"
        
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/opt/homebrew/bin/python3")
        process.arguments = [scriptPath, "--backend"]
        
        do {
            try process.run()
            self.pythonProcess = process
            print("🟢 Python Backend launched successfully.")
        } catch {
            print("⚠️ Failed to launch Python backend: \(error)")
        }
    }

    @objc func fetchLastTitle() {
        guard let url = URL(string: "http://localhost:5001/api/settings") else { return }
        
        URLSession.shared.dataTask(with: url) { [weak self] data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            
            let game = (json["current_game"] as? String) ?? "Auto-Detecting..."
            
            DispatchQueue.main.async {
                self?.categoryItem.title = "Live Category: \(game)"
            }
        }.resume()
    }

    @objc func togglePause() {
        isPaused.toggle()
        let endpoint = isPaused ? "pause" : "resume"
        
        if isPaused {
            statusItem.button?.image = NSImage(systemSymbolName: "waveform.slash", accessibilityDescription: "Paused") ?? NSImage(systemSymbolName: "waveform.low", accessibilityDescription: "Paused")
            pauseItem.title = "Resume Recording"
            pauseItem.image = NSImage(systemSymbolName: "play.circle", accessibilityDescription: nil)
        } else {
            statusItem.button?.image = NSImage(systemSymbolName: "waveform.mid", accessibilityDescription: "Clip Backtrack")
            pauseItem.title = "Pause Recording"
            pauseItem.image = NSImage(systemSymbolName: "pause.circle", accessibilityDescription: nil)
        }

        guard let url = URL(string: "http://localhost:5001/\(endpoint)") else { return }
        URLSession.shared.dataTask(with: url).resume()
    }

    @objc func triggerClip() {
        guard let url = URL(string: "http://localhost:5001/clip") else { return }
        
        URLSession.shared.dataTask(with: url) { [weak self] data, _, _ in
            guard let data = data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let title = json["title"] as? String else { return }
            
            DispatchQueue.main.async {
                let displayTitle = title.count <= 25 ? title : String(title.prefix(22)) + "..."
                self?.lastTitleItem.title = "Last Title: \"\(displayTitle)\""
            }
        }.resume()
    }

    @objc func toggleYouTube(_ sender: NSMenuItem) {
        if sender.title.contains("ON") {
            sender.title = "YouTube Auto-Upload: [ OFF ]"
        } else {
            sender.title = "YouTube Auto-Upload: [ ON ]"
        }
    }

    @objc func authYouTube() {
        print("🔑 Triggering YouTube OAuth...")
    }

    @objc func openSettings() {
        if let url = URL(string: "http://localhost:5001/settings") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc func openTriggerURL() {
        if let url = URL(string: "http://localhost:5001/clip") {
            NSWorkspace.shared.open(url)
        }
    }

    @objc func quitApp() {
        pollTimer?.invalidate()
        if let process = pythonProcess, process.isRunning {
            process.terminate()
        }
        NSApplication.shared.terminate(nil)
    }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
