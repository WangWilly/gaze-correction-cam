// swift-tools-version:5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GazeCorrectionVirtualCamera",
    platforms: [
        .macOS(.v13)
    ],
    products: [
        // Camera Extension (SystemExtension with CoreMediaIO)
        .library(
            name: "VirtualCameraExtension",
            targets: ["VirtualCameraExtension"]
        ),
        // Settings App for user configuration
        .executable(
            name: "GazeCorrectionSettings",
            targets: ["SettingsApp"]
        ),
    ],
    dependencies: [
        // MediaPipe iOS/macOS SDK (use MediaPipe Tasks Vision)
        // Note: MediaPipe is included via local framework for macOS
    ],
    targets: [
        // Virtual Camera Extension
        .target(
            name: "VirtualCameraExtension",
            dependencies: [],
            path: "Sources/VirtualCameraExtension",
            linkerSettings: [
                .linkedFramework("CoreMediaIO"),
                .linkedFramework("CoreMedia"),
                .linkedFramework("CoreVideo"),
                .linkedFramework("AVFoundation"),
                .linkedFramework("IOKit"),
            ]
        ),
        // Settings Application
        .executableTarget(
            name: "SettingsApp",
            dependencies: [],
            path: "Sources/SettingsApp",
            linkerSettings: [
                .linkedFramework("SwiftUI"),
                .linkedFramework("AppKit"),
            ]
        ),
    ]
)
