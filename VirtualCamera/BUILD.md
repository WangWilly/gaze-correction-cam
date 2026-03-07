# Gaze Correction Virtual Camera - Build Instructions

This directory contains a macOS virtual camera implementation using CoreMediaIO Camera Extension (CMIOExtension) with MediaPipe for face detection and gaze correction.

## Overview

The virtual camera consists of two main components:

1. **VirtualCameraExtension**: A CoreMediaIO Camera Extension that creates a virtual camera device visible to applications like Zoom, FaceTime, etc.
2. **SettingsApp**: A SwiftUI application for configuring camera settings (camera offset, focal length, IPD, gaze correction toggle).

## Prerequisites

### System Requirements

- macOS 13.0 (Ventura) or later
- Xcode 15.0 or later
- Apple Developer account (for code signing)
- System Integrity Protection (SIP) considerations for development

### Dependencies

- MediaPipe Tasks Vision framework for macOS
- Python 3.12+ with the gaze correction backend (see main README.md)

## Project Structure

```
VirtualCamera/
├── Package.swift              # Swift Package definition
├── BUILD.md                   # This file
├── Sources/
│   ├── VirtualCameraExtension/
│   │   ├── main.swift                     # Extension entry point
│   │   ├── VirtualCameraProvider.swift    # CMIOExtension provider
│   │   ├── VirtualCameraDevice.swift      # Virtual camera device
│   │   ├── VirtualCameraStream.swift      # Video stream source
│   │   ├── FrameProcessor.swift           # Frame processing with MediaPipe
│   │   └── SettingsManager.swift          # User settings management
│   └── SettingsApp/
│       ├── main.swift                     # Settings app entry point
│       ├── SettingsApp.swift              # SwiftUI App definition
│       ├── ContentView.swift              # Main settings UI
│       ├── CameraOffsetView.swift         # Camera offset controls
│       └── CalibrationView.swift          # Calibration controls
└── Resources/
    ├── Info.plist                         # Extension Info.plist
    └── Entitlements.plist                 # Required entitlements
```

## Building with Xcode

### Step 1: Create Xcode Project

Since Camera Extensions require specific project structure and code signing, you'll need to create an Xcode project:

1. Open Xcode and create a new project
2. Select "Camera Extension" template under macOS
3. Configure bundle identifiers:
   - App: `com.yourcompany.GazeCorrectionCamera`
   - Extension: `com.yourcompany.GazeCorrectionCamera.Extension`

### Step 2: Import Source Files

Copy the Swift source files from this directory into your Xcode project:

```bash
# From the VirtualCamera directory
cp -r Sources/VirtualCameraExtension/* <YourXcodeProject>/CameraExtension/
cp -r Sources/SettingsApp/* <YourXcodeProject>/SettingsApp/
```

### Step 3: Configure Entitlements

The extension requires these entitlements in your Xcode project:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.developer.system-extension.install</key>
    <true/>
    <key>com.apple.security.app-sandbox</key>
    <true/>
    <key>com.apple.security.device.camera</key>
    <true/>
</dict>
</plist>
```

### Step 4: Configure Info.plist

The extension Info.plist should include:

```xml
<key>NSCameraUsageDescription</key>
<string>Gaze Correction Camera needs camera access for video processing</string>
<key>CMIOExtension</key>
<dict>
    <key>CMIOExtensionMachServiceName</key>
    <string>com.yourcompany.GazeCorrectionCamera.Extension</string>
</dict>
<key>NSSystemExtensionUsageDescription</key>
<string>This extension provides a virtual camera with gaze correction</string>
```

### Step 5: Build and Sign

1. Select your Development Team in Xcode
2. Enable "Hardened Runtime" for the extension
3. Build the project (Cmd+B)

## Building with Swift Package Manager (Development Only)

For development and testing without full extension signing:

```bash
cd VirtualCamera

# Build the package
swift build

# Build in release mode
swift build -c release
```

Note: The built extension won't work as a system extension without proper code signing.

## Installation

### Development Installation

1. Build the app in Xcode
2. Run the containing app once to trigger system extension installation
3. Approve the extension in System Settings > Privacy & Security > Extensions

### Production Distribution

1. Archive the app in Xcode
2. Notarize the app with Apple
3. Distribute via direct download or Mac App Store

## Usage with Gaze Correction Backend

The virtual camera is designed to work with the Python gaze correction backend:

### Option 1: Run the Single Window Application

```bash
# Start the gaze correction application (from main project directory)
poetry run python bin_single_window.py --backend mediapipe

# The virtual camera extension can be extended to communicate with this backend
```

### Option 2: Local Processing (MediaPipe Only)

The extension can perform basic face detection using MediaPipe directly without the Python backend:

1. Open the Settings App
2. Enable "Local Processing Mode"
3. Note: This mode only provides face detection, not full gaze correction

## Settings

The Settings App allows configuration of:

| Setting | Description | Default |
|---------|-------------|---------|
| Gaze Correction | Enable/disable gaze correction | On |
| Camera Offset X | Horizontal camera offset (cm) | 0.0 |
| Camera Offset Y | Vertical camera offset (cm) | -21.0 |
| Camera Offset Z | Depth camera offset (cm) | -1.0 |
| Focal Length | Camera focal length (pixels) | 650 |
| IPD | Inter-pupillary distance (cm) | 6.3 |

Settings are stored in:
- UserDefaults with App Group (`group.com.gazecorrection.camera`)
- Shared between extension and settings app

## Troubleshooting

### Extension Not Loading

1. Check System Settings > Privacy & Security > Extensions
2. Ensure the extension is approved
3. Check Console.app for extension logs

### Camera Not Visible in Apps

1. Restart the target application
2. Check if the extension is running: `systemextensionsctl list`
3. Try unloading and reloading: Click "Remove" in Settings App, then reinstall

### Performance Issues

1. Reduce video resolution in Settings App
2. Disable gaze correction temporarily
3. Check CPU usage in Activity Monitor

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Host Application                          │
│                  (Zoom, FaceTime, etc.)                     │
└─────────────────────┬───────────────────────────────────────┘
                      │ CMIOExtension Protocol
                      ▼
┌─────────────────────────────────────────────────────────────┐
│              VirtualCameraExtension                          │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ VirtualCameraProvider                                    ││
│  │   └── VirtualCameraDevice                               ││
│  │         └── VirtualCameraStream                         ││
│  │               └── FrameProcessor                        ││
│  │                     ├── MediaPipe (Face Detection)      ││
│  │                     └── IPC → Python Backend (Gaze)     ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
                      │ UserDefaults / IPC
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    Settings App                              │
│  ┌─────────────────────────────────────────────────────────┐│
│  │ SwiftUI Interface                                        ││
│  │   ├── Camera Offset Controls                            ││
│  │   ├── Focal Length Slider                               ││
│  │   ├── IPD Adjustment                                    ││
│  │   └── Gaze Correction Toggle                            ││
│  └─────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## License

This project is licensed under the same license as the main gaze-correction-cam project.
