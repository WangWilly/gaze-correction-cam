//
//  main.swift
//  VirtualCameraExtension
//
//  Entry point for the CoreMediaIO Camera Extension.
//  This extension provides a virtual camera with gaze correction capabilities.
//

import Foundation
import CoreMediaIO

/// Main entry point for the Camera Extension
/// The extension is automatically launched by the system when applications
/// request camera access and this extension is approved.
let providerSource = VirtualCameraProviderSource(clientQueue: nil)
CMIOExtensionProviderSource.startService(provider: providerSource.provider)

// Keep the extension running
CFRunLoopRun()
