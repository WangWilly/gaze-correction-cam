//
//  CalibrationView.swift
//  SettingsApp
//
//  Calibration controls and camera position visualization.
//

import SwiftUI

/// Calibration view with controls and visualization
struct CalibrationView: View {
    
    /// View model
    @ObservedObject var viewModel: SettingsViewModel
    
    var body: some View {
        VStack(spacing: 20) {
            // Video settings section
            GroupBox("Video Settings") {
                VStack(spacing: 12) {
                    // Resolution picker
                    HStack {
                        Text("Resolution")
                            .font(.caption)
                        
                        Spacer()
                        
                        Picker("", selection: $viewModel.videoWidth) {
                            Text("640×480").tag(640)
                            Text("1280×720").tag(1280)
                            Text("1920×1080").tag(1920)
                        }
                        .pickerStyle(.menu)
                        .frame(width: 120)
                        .onChange(of: viewModel.videoWidth) { newValue in
                            switch newValue {
                            case 640: viewModel.videoHeight = 480
                            case 1280: viewModel.videoHeight = 720
                            case 1920: viewModel.videoHeight = 1080
                            default: break
                            }
                        }
                    }
                    
                    // Frame rate picker
                    HStack {
                        Text("Frame Rate")
                            .font(.caption)
                        
                        Spacer()
                        
                        Picker("", selection: $viewModel.frameRate) {
                            Text("15 fps").tag(15)
                            Text("24 fps").tag(24)
                            Text("30 fps").tag(30)
                            Text("60 fps").tag(60)
                        }
                        .pickerStyle(.menu)
                        .frame(width: 120)
                    }
                }
                .padding(.vertical, 8)
            }
            
            // Camera position diagram
            GroupBox("Camera Position (Top View)") {
                CameraPositionDiagram(
                    cameraOffsetX: viewModel.cameraOffsetX,
                    cameraOffsetZ: viewModel.cameraOffsetZ
                )
                .frame(height: 150)
                .padding(.vertical, 8)
            }
            
            // Current values display
            GroupBox("Current Configuration") {
                VStack(spacing: 8) {
                    configRow(label: "Camera Offset X", value: String(format: "%+.1f cm", viewModel.cameraOffsetX))
                    configRow(label: "Camera Offset Y", value: String(format: "%+.1f cm", viewModel.cameraOffsetY))
                    configRow(label: "Camera Offset Z", value: String(format: "%+.1f cm", viewModel.cameraOffsetZ))
                    
                    Divider()
                    
                    configRow(label: "Focal Length", value: String(format: "%.0f px", viewModel.focalLength))
                    configRow(label: "IPD", value: String(format: "%.1f cm", viewModel.ipd))
                    
                    Divider()
                    
                    configRow(label: "Resolution", value: "\(viewModel.videoWidth)×\(viewModel.videoHeight)")
                    configRow(label: "Frame Rate", value: "\(viewModel.frameRate) fps")
                }
                .padding(.vertical, 8)
            }
            
            // Export/Import buttons
            HStack {
                Button("Export Settings") {
                    exportSettings()
                }
                
                Button("Import Settings") {
                    importSettings()
                }
            }
            .buttonStyle(.bordered)
            
            Spacer()
        }
        .padding()
    }
    
    /// Configuration row helper
    private func configRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .font(.caption.monospacedDigit())
        }
    }
    
    /// Export settings to file
    private func exportSettings() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = "gaze_correction_settings.json"
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                let settings: [String: Any] = [
                    "cameraOffsetX": viewModel.cameraOffsetX,
                    "cameraOffsetY": viewModel.cameraOffsetY,
                    "cameraOffsetZ": viewModel.cameraOffsetZ,
                    "focalLength": viewModel.focalLength,
                    "ipd": viewModel.ipd,
                    "gazeEnabled": viewModel.gazeEnabled,
                    "videoWidth": viewModel.videoWidth,
                    "videoHeight": viewModel.videoHeight,
                    "frameRate": viewModel.frameRate
                ]
                
                if let data = try? JSONSerialization.data(withJSONObject: settings, options: .prettyPrinted) {
                    try? data.write(to: url)
                }
            }
        }
    }
    
    /// Import settings from file
    private func importSettings() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        
        panel.begin { response in
            if response == .OK, let url = panel.url {
                if let data = try? Data(contentsOf: url),
                   let settings = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                    
                    if let x = settings["cameraOffsetX"] as? Float { viewModel.cameraOffsetX = x }
                    if let y = settings["cameraOffsetY"] as? Float { viewModel.cameraOffsetY = y }
                    if let z = settings["cameraOffsetZ"] as? Float { viewModel.cameraOffsetZ = z }
                    if let f = settings["focalLength"] as? Float { viewModel.focalLength = f }
                    if let i = settings["ipd"] as? Float { viewModel.ipd = i }
                    if let g = settings["gazeEnabled"] as? Bool { viewModel.gazeEnabled = g }
                    if let w = settings["videoWidth"] as? Int { viewModel.videoWidth = w }
                    if let h = settings["videoHeight"] as? Int { viewModel.videoHeight = h }
                    if let r = settings["frameRate"] as? Int { viewModel.frameRate = r }
                }
            }
        }
    }
}

/// Camera position diagram view
struct CameraPositionDiagram: View {
    
    /// Camera X offset in cm
    var cameraOffsetX: Float
    
    /// Camera Z offset in cm
    var cameraOffsetZ: Float
    
    /// Scale factor (cm to points)
    private let scale: CGFloat = 2.0
    
    var body: some View {
        GeometryReader { geometry in
            let center = CGPoint(x: geometry.size.width / 2, y: geometry.size.height - 30)
            
            ZStack {
                // Background
                RoundedRectangle(cornerRadius: 8)
                    .fill(Color.black.opacity(0.05))
                
                // Screen line
                Path { path in
                    path.move(to: CGPoint(x: 20, y: center.y))
                    path.addLine(to: CGPoint(x: geometry.size.width - 20, y: center.y))
                }
                .stroke(Color.gray, lineWidth: 2)
                
                Text("Screen")
                    .font(.caption2)
                    .foregroundColor(.secondary)
                    .position(x: center.x, y: center.y + 15)
                
                // Camera position
                let camX = center.x + CGFloat(cameraOffsetX) * scale
                let camY = center.y + CGFloat(cameraOffsetZ) * scale
                
                Circle()
                    .fill(Color.yellow)
                    .frame(width: 12, height: 12)
                    .position(x: camX, y: camY)
                
                Text("Camera")
                    .font(.caption2)
                    .foregroundColor(.yellow)
                    .position(x: camX, y: camY - 15)
                
                // Eye position (estimated at typical distance)
                let eyeX = center.x
                let eyeY = center.y - 60 * scale
                
                Circle()
                    .fill(Color.blue)
                    .frame(width: 10, height: 10)
                    .position(x: eyeX, y: eyeY)
                
                Text("Eyes")
                    .font(.caption2)
                    .foregroundColor(.blue)
                    .position(x: eyeX, y: eyeY - 12)
                
                // Gaze line (from eyes to camera)
                Path { path in
                    path.move(to: CGPoint(x: eyeX, y: eyeY))
                    path.addLine(to: CGPoint(x: camX, y: camY))
                }
                .stroke(Color.green.opacity(0.5), lineWidth: 1)
            }
        }
    }
}

/// Status view showing system information
struct StatusView: View {
    
    /// View model
    @ObservedObject var viewModel: SettingsViewModel
    
    var body: some View {
        VStack(spacing: 20) {
            // Connection status
            GroupBox("Backend Status") {
                VStack(spacing: 12) {
                    statusRow(
                        label: "Connection",
                        status: viewModel.backendConnected ? "Connected" : "Disconnected",
                        isGood: viewModel.backendConnected
                    )
                    
                    statusRow(
                        label: "Gaze Correction",
                        status: viewModel.gazeEnabled ? "Enabled" : "Disabled",
                        isGood: viewModel.gazeEnabled
                    )
                }
                .padding(.vertical, 8)
            }
            
            // System information
            GroupBox("System Information") {
                VStack(spacing: 8) {
                    infoRow(label: "macOS Version", value: ProcessInfo.processInfo.operatingSystemVersionString)
                    infoRow(label: "App Version", value: "1.0.0")
                    infoRow(label: "Extension Bundle", value: "com.gazecorrection.camera.extension")
                }
                .padding(.vertical, 8)
            }
            
            // Help section
            GroupBox("Help") {
                VStack(alignment: .leading, spacing: 8) {
                    helpItem(
                        title: "Backend Not Connected",
                        description: "Start the Python backend with: poetry run python bin_virtual_camera_backend.py"
                    )
                    
                    helpItem(
                        title: "Camera Not Visible",
                        description: "Ensure the extension is approved in System Settings > Privacy & Security > Extensions"
                    )
                    
                    helpItem(
                        title: "Poor Gaze Correction",
                        description: "Adjust camera offset and focal length in the Camera tab"
                    )
                }
                .padding(.vertical, 8)
            }
            
            Spacer()
        }
        .padding()
    }
    
    /// Status row helper
    private func statusRow(label: String, status: String, isGood: Bool) -> some View {
        HStack {
            Text(label)
                .font(.caption)
            
            Spacer()
            
            HStack(spacing: 6) {
                Circle()
                    .fill(isGood ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
                
                Text(status)
                    .font(.caption)
                    .foregroundColor(isGood ? .green : .red)
            }
        }
    }
    
    /// Info row helper
    private func infoRow(label: String, value: String) -> some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            
            Spacer()
            
            Text(value)
                .font(.caption)
        }
    }
    
    /// Help item helper
    private func helpItem(title: String, description: String) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title)
                .font(.caption)
                .fontWeight(.medium)
            
            Text(description)
                .font(.caption2)
                .foregroundColor(.secondary)
        }
    }
}

// MARK: - Previews

struct CalibrationView_Previews: PreviewProvider {
    static var previews: some View {
        CalibrationView(viewModel: SettingsViewModel())
    }
}

struct StatusView_Previews: PreviewProvider {
    static var previews: some View {
        StatusView(viewModel: SettingsViewModel())
    }
}
