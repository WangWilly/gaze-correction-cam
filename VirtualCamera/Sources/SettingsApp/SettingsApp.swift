//
//  SettingsApp.swift
//  SettingsApp
//
//  SwiftUI App definition for the Gaze Correction Settings application.
//

import SwiftUI

/// Main SwiftUI App for Gaze Correction Settings
struct GazeCorrectionSettingsApp: App {
    
    /// App delegate for macOS-specific functionality
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    
    var body: some Scene {
        WindowGroup {
            ContentView()
                .frame(minWidth: 400, minHeight: 500)
        }
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .appInfo) {
                Button("About Gaze Correction") {
                    NSApplication.shared.orderFrontStandardAboutPanel(
                        options: [
                            .applicationName: "Gaze Correction Camera",
                            .applicationVersion: "1.0.0",
                            .credits: NSAttributedString(
                                string: "Virtual camera with real-time gaze correction"
                            )
                        ]
                    )
                }
            }
        }
        
        Settings {
            SettingsView()
        }
    }
}

/// App Delegate for macOS-specific functionality
class AppDelegate: NSObject, NSApplicationDelegate {
    
    func applicationDidFinishLaunching(_ notification: Notification) {
        print("[GazeCorrection] Settings app launched")
    }
    
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        return true
    }
}

/// Settings View for app preferences
struct SettingsView: View {
    var body: some View {
        TabView {
            GeneralSettingsView()
                .tabItem {
                    Label("General", systemImage: "gear")
                }
            
            AdvancedSettingsView()
                .tabItem {
                    Label("Advanced", systemImage: "slider.horizontal.3")
                }
        }
        .frame(width: 400, height: 300)
    }
}

/// General settings view
struct GeneralSettingsView: View {
    @AppStorage("launchAtLogin") private var launchAtLogin = false
    @AppStorage("showInMenuBar") private var showInMenuBar = true
    
    var body: some View {
        Form {
            Toggle("Launch at Login", isOn: $launchAtLogin)
            Toggle("Show in Menu Bar", isOn: $showInMenuBar)
        }
        .padding()
    }
}

/// Advanced settings view
struct AdvancedSettingsView: View {
    @AppStorage("enableLogging") private var enableLogging = false
    @AppStorage("useSharedMemory") private var useSharedMemory = true
    
    var body: some View {
        Form {
            Toggle("Enable Debug Logging", isOn: $enableLogging)
            Toggle("Use Shared Memory IPC", isOn: $useSharedMemory)
        }
        .padding()
    }
}
