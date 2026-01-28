//
//  VirtualCameraProvider.swift
//  VirtualCameraExtension
//
//  Provides the CMIOExtension provider that manages virtual camera devices.
//

import Foundation
import CoreMediaIO

/// Source that creates and manages the camera extension provider
class VirtualCameraProviderSource: NSObject {
    
    /// The CMIOExtension provider instance
    let provider: CMIOExtensionProvider
    
    /// The virtual camera device
    private let device: VirtualCameraDevice
    
    /// Settings manager for user preferences
    private let settingsManager: SettingsManager
    
    /// Initialize the provider source
    /// - Parameter clientQueue: Optional dispatch queue for client callbacks
    init(clientQueue: DispatchQueue?) {
        // Initialize settings manager
        settingsManager = SettingsManager.shared
        
        // Create the virtual camera device
        device = VirtualCameraDevice(
            localizedName: "Gaze Correction Camera",
            settingsManager: settingsManager
        )
        
        // Create the provider with the device
        provider = CMIOExtensionProvider(
            source: VirtualCameraProviderSourceDelegate(),
            clientQueue: clientQueue
        )
        
        super.init()
        
        // Add the device to the provider
        do {
            try provider.addDevice(device.device)
            print("[GazeCorrection] Virtual camera device added successfully")
        } catch {
            print("[GazeCorrection] Failed to add device: \(error)")
        }
    }
}

/// Delegate for the provider source
class VirtualCameraProviderSourceDelegate: NSObject, CMIOExtensionProviderSource {
    
    /// Called when the provider is connected
    func connect(to client: CMIOExtensionClient) throws {
        print("[GazeCorrection] Provider connected to client")
    }
    
    /// Called when the provider is disconnected
    func disconnect(from client: CMIOExtensionClient) {
        print("[GazeCorrection] Provider disconnected from client")
    }
    
    /// Returns the available properties for this provider
    var availableProperties: Set<CMIOExtensionProperty> {
        return [
            CMIOExtensionProperty(rawValue: "com.apple.cmio.property.manufacturer"),
            CMIOExtensionProperty(rawValue: "com.apple.cmio.property.model")
        ]
    }
    
    /// Get the value of a property
    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionProviderProperties {
        let providerProperties = CMIOExtensionProviderProperties(dictionary: [:])
        
        for property in properties {
            switch property.rawValue {
            case "com.apple.cmio.property.manufacturer":
                providerProperties.manufacturer = "Gaze Correction"
            case "com.apple.cmio.property.model":
                providerProperties.name = "Virtual Camera v1.0"
            default:
                break
            }
        }
        
        return providerProperties
    }
    
    /// Set the value of a property
    func setProviderProperties(_ properties: CMIOExtensionProviderProperties) throws {
        // Provider properties are read-only
    }
}
