//
//  ContentView.swift
//  SettingsApp
//
//  Main content view for the Gaze Correction Settings application.
//  Provides controls for camera offset, focal length, IPD, and gaze correction toggle.
//

import SwiftUI

/// Main content view with all settings controls
struct ContentView: View {
    
    /// Settings manager
    @StateObject private var viewModel = SettingsViewModel()
    
    /// Selected tab
    @State private var selectedTab = 0
    
    var body: some View {
        VStack(spacing: 0) {
            // Header
            headerView
            
            Divider()
            
            // Tab selection
            tabBar
            
            Divider()
            
            // Content based on selected tab
            ScrollView {
                switch selectedTab {
                case 0:
                    CameraOffsetView(viewModel: viewModel)
                case 1:
                    CalibrationView(viewModel: viewModel)
                case 2:
                    StatusView(viewModel: viewModel)
                default:
                    EmptyView()
                }
            }
            
            Divider()
            
            // Footer with status
            footerView
        }
        .background(Color(NSColor.windowBackgroundColor))
    }
    
    /// Header view with title and gaze toggle
    private var headerView: some View {
        HStack {
            VStack(alignment: .leading) {
                Text("Gaze Correction Camera")
                    .font(.title2)
                    .fontWeight(.semibold)
                
                Text("Virtual camera settings")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }
            
            Spacer()
            
            // Gaze correction toggle
            Toggle("", isOn: $viewModel.gazeEnabled)
                .toggleStyle(.switch)
                .labelsHidden()
            
            Text(viewModel.gazeEnabled ? "ON" : "OFF")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(viewModel.gazeEnabled ? .green : .red)
        }
        .padding()
    }
    
    /// Tab bar for navigation
    private var tabBar: some View {
        HStack(spacing: 0) {
            tabButton(title: "Camera", icon: "camera", index: 0)
            tabButton(title: "Calibration", icon: "slider.horizontal.3", index: 1)
            tabButton(title: "Status", icon: "info.circle", index: 2)
        }
        .padding(.horizontal)
        .padding(.vertical, 8)
    }
    
    /// Tab button helper
    private func tabButton(title: String, icon: String, index: Int) -> some View {
        Button(action: { selectedTab = index }) {
            VStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.system(size: 16))
                Text(title)
                    .font(.caption)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .background(selectedTab == index ? Color.accentColor.opacity(0.2) : Color.clear)
            .cornerRadius(8)
        }
        .buttonStyle(.plain)
    }
    
    /// Footer view with status
    private var footerView: some View {
        HStack {
            Circle()
                .fill(viewModel.backendConnected ? Color.green : Color.red)
                .frame(width: 8, height: 8)
            
            Text(viewModel.backendConnected ? "Backend Connected" : "Backend Disconnected")
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Button("Reset All") {
                viewModel.resetToDefaults()
            }
            .font(.caption)
        }
        .padding()
    }
}

/// View model for settings
class SettingsViewModel: ObservableObject {
    
    /// Camera offset X in cm
    @Published var cameraOffsetX: Float = 0.0 {
        didSet { saveSettings() }
    }
    
    /// Camera offset Y in cm
    @Published var cameraOffsetY: Float = -21.0 {
        didSet { saveSettings() }
    }
    
    /// Camera offset Z in cm
    @Published var cameraOffsetZ: Float = -1.0 {
        didSet { saveSettings() }
    }
    
    /// Focal length in pixels
    @Published var focalLength: Float = 650.0 {
        didSet { saveSettings() }
    }
    
    /// Inter-pupillary distance in cm
    @Published var ipd: Float = 6.3 {
        didSet { saveSettings() }
    }
    
    /// Whether gaze correction is enabled
    @Published var gazeEnabled: Bool = true {
        didSet { saveSettings() }
    }
    
    /// Video resolution width
    @Published var videoWidth: Int = 1280 {
        didSet { saveSettings() }
    }
    
    /// Video resolution height
    @Published var videoHeight: Int = 720 {
        didSet { saveSettings() }
    }
    
    /// Frame rate
    @Published var frameRate: Int = 30 {
        didSet { saveSettings() }
    }
    
    /// Whether the backend is connected
    @Published var backendConnected: Bool = false
    
    /// UserDefaults suite
    private let defaults = UserDefaults(suiteName: "group.com.gazecorrection.camera") ?? .standard
    
    /// Timer for connection check
    private var connectionCheckTimer: Timer?
    
    /// Keys for tracking if settings have been initialized
    private let initializedKey = "settingsInitialized"
    
    init() {
        // Register defaults for first launch
        registerDefaults()
        loadSettings()
        startConnectionCheck()
    }
    
    /// Register default values for first-time launch
    private func registerDefaults() {
        defaults.register(defaults: [
            "focalLength": Float(650.0),
            "ipd": Float(6.3),
            "cameraOffsetX": Float(0.0),
            "cameraOffsetY": Float(-21.0),
            "cameraOffsetZ": Float(-1.0),
            "gazeEnabled": true,
            "videoWidth": 1280,
            "videoHeight": 720,
            "frameRate": 30
        ])
    }
    
    /// Load settings from UserDefaults
    private func loadSettings() {
        cameraOffsetX = defaults.float(forKey: "cameraOffsetX")
        cameraOffsetY = defaults.float(forKey: "cameraOffsetY")
        cameraOffsetZ = defaults.float(forKey: "cameraOffsetZ")
        focalLength = defaults.float(forKey: "focalLength")
        ipd = defaults.float(forKey: "ipd")
        gazeEnabled = defaults.bool(forKey: "gazeEnabled")
        videoWidth = defaults.integer(forKey: "videoWidth")
        videoHeight = defaults.integer(forKey: "videoHeight")
        frameRate = defaults.integer(forKey: "frameRate")
    }
    
    /// Save settings to UserDefaults
    private func saveSettings() {
        defaults.set(cameraOffsetX, forKey: "cameraOffsetX")
        defaults.set(cameraOffsetY, forKey: "cameraOffsetY")
        defaults.set(cameraOffsetZ, forKey: "cameraOffsetZ")
        defaults.set(focalLength, forKey: "focalLength")
        defaults.set(ipd, forKey: "ipd")
        defaults.set(gazeEnabled, forKey: "gazeEnabled")
        defaults.set(videoWidth, forKey: "videoWidth")
        defaults.set(videoHeight, forKey: "videoHeight")
        defaults.set(frameRate, forKey: "frameRate")
    }
    
    /// Reset to default values
    func resetToDefaults() {
        cameraOffsetX = 0.0
        cameraOffsetY = -21.0
        cameraOffsetZ = -1.0
        focalLength = 650.0
        ipd = 6.3
        gazeEnabled = true
        videoWidth = 1280
        videoHeight = 720
        frameRate = 30
    }
    
    /// Start periodic connection check
    private func startConnectionCheck() {
        connectionCheckTimer = Timer.scheduledTimer(withTimeInterval: 5.0, repeats: true) { [weak self] _ in
            self?.checkBackendConnection()
        }
        checkBackendConnection()
    }
    
    /// Check if backend is connected
    private func checkBackendConnection() {
        let fileManager = FileManager.default
        backendConnected = fileManager.fileExists(atPath: "/tmp/gaze_correction_frames") &&
                          fileManager.fileExists(atPath: "/tmp/gaze_correction_results")
    }
    
    deinit {
        connectionCheckTimer?.invalidate()
    }
}

// MARK: - Preview

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView()
    }
}
