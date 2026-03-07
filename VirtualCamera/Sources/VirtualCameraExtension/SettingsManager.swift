//
//  SettingsManager.swift
//  VirtualCameraExtension
//
//  Manages user settings for the virtual camera using UserDefaults
//  and provides an interface matching the Python backend settings.
//

import Foundation

/// Camera settings matching the Python CameraUserSetting class
struct CameraSettings: Codable, Equatable {
    /// Focal length in pixels (typically 500-1000)
    var focalLength: Float = 650.0
    
    /// Inter-pupillary distance in cm
    var ipd: Float = 6.3
    
    /// Camera offset X (horizontal, positive = right) in cm
    var cameraOffsetX: Float = 0.0
    
    /// Camera offset Y (vertical, positive = down) in cm
    var cameraOffsetY: Float = -21.0
    
    /// Camera offset Z (depth, negative = behind screen) in cm
    var cameraOffsetZ: Float = -1.0
    
    /// Whether gaze correction is enabled
    var gazeEnabled: Bool = true
    
    /// Video resolution width
    var videoWidth: Int = 1280
    
    /// Video resolution height
    var videoHeight: Int = 720
    
    /// Target frame rate
    var frameRate: Int = 30
    
    /// Create default settings
    static var `default`: CameraSettings {
        return CameraSettings()
    }
    
    /// Camera offset as a tuple
    var cameraOffset: (x: Float, y: Float, z: Float) {
        return (cameraOffsetX, cameraOffsetY, cameraOffsetZ)
    }
}

/// Manages user settings with UserDefaults persistence
class SettingsManager {
    
    /// Shared instance
    static let shared = SettingsManager()
    
    /// UserDefaults suite name
    private let suiteName = "group.com.gazecorrection.camera"
    
    /// UserDefaults instance
    private let defaults: UserDefaults
    
    /// Settings change notification
    static let settingsChangedNotification = Notification.Name("GazeCorrectionSettingsChanged")
    
    // MARK: - Settings Keys
    
    private enum Keys {
        static let focalLength = "focalLength"
        static let ipd = "ipd"
        static let cameraOffsetX = "cameraOffsetX"
        static let cameraOffsetY = "cameraOffsetY"
        static let cameraOffsetZ = "cameraOffsetZ"
        static let gazeEnabled = "gazeEnabled"
        static let videoWidth = "videoWidth"
        static let videoHeight = "videoHeight"
        static let frameRate = "frameRate"
    }
    
    // MARK: - Initialization
    
    /// Initialize with optional custom UserDefaults suite
    init(suiteName: String? = nil) {
        if let suite = suiteName ?? self.suiteName {
            defaults = UserDefaults(suiteName: suite) ?? .standard
        } else {
            defaults = .standard
        }
        
        // Register defaults
        registerDefaults()
    }
    
    /// Register default values
    private func registerDefaults() {
        let defaultSettings = CameraSettings.default
        
        defaults.register(defaults: [
            Keys.focalLength: defaultSettings.focalLength,
            Keys.ipd: defaultSettings.ipd,
            Keys.cameraOffsetX: defaultSettings.cameraOffsetX,
            Keys.cameraOffsetY: defaultSettings.cameraOffsetY,
            Keys.cameraOffsetZ: defaultSettings.cameraOffsetZ,
            Keys.gazeEnabled: defaultSettings.gazeEnabled,
            Keys.videoWidth: defaultSettings.videoWidth,
            Keys.videoHeight: defaultSettings.videoHeight,
            Keys.frameRate: defaultSettings.frameRate
        ])
    }
    
    // MARK: - Settings Access
    
    /// Get current settings
    var currentSettings: CameraSettings {
        return CameraSettings(
            focalLength: focalLength,
            ipd: ipd,
            cameraOffsetX: cameraOffsetX,
            cameraOffsetY: cameraOffsetY,
            cameraOffsetZ: cameraOffsetZ,
            gazeEnabled: gazeEnabled,
            videoWidth: videoWidth,
            videoHeight: videoHeight,
            frameRate: frameRate
        )
    }
    
    /// Apply settings from a CameraSettings object
    func applySettings(_ settings: CameraSettings) {
        focalLength = settings.focalLength
        ipd = settings.ipd
        cameraOffsetX = settings.cameraOffsetX
        cameraOffsetY = settings.cameraOffsetY
        cameraOffsetZ = settings.cameraOffsetZ
        gazeEnabled = settings.gazeEnabled
        videoWidth = settings.videoWidth
        videoHeight = settings.videoHeight
        frameRate = settings.frameRate
    }
    
    /// Reset all settings to defaults
    func resetToDefaults() {
        applySettings(.default)
    }
    
    // MARK: - Individual Settings
    
    /// Focal length in pixels
    var focalLength: Float {
        get { defaults.float(forKey: Keys.focalLength) }
        set {
            defaults.set(newValue, forKey: Keys.focalLength)
            notifySettingsChanged()
        }
    }
    
    /// Inter-pupillary distance in cm
    var ipd: Float {
        get { defaults.float(forKey: Keys.ipd) }
        set {
            defaults.set(newValue, forKey: Keys.ipd)
            notifySettingsChanged()
        }
    }
    
    /// Camera offset X in cm
    var cameraOffsetX: Float {
        get { defaults.float(forKey: Keys.cameraOffsetX) }
        set {
            defaults.set(newValue, forKey: Keys.cameraOffsetX)
            notifySettingsChanged()
        }
    }
    
    /// Camera offset Y in cm
    var cameraOffsetY: Float {
        get { defaults.float(forKey: Keys.cameraOffsetY) }
        set {
            defaults.set(newValue, forKey: Keys.cameraOffsetY)
            notifySettingsChanged()
        }
    }
    
    /// Camera offset Z in cm
    var cameraOffsetZ: Float {
        get { defaults.float(forKey: Keys.cameraOffsetZ) }
        set {
            defaults.set(newValue, forKey: Keys.cameraOffsetZ)
            notifySettingsChanged()
        }
    }
    
    /// Whether gaze correction is enabled
    var gazeEnabled: Bool {
        get { defaults.bool(forKey: Keys.gazeEnabled) }
        set {
            defaults.set(newValue, forKey: Keys.gazeEnabled)
            notifySettingsChanged()
        }
    }
    
    /// Video width
    var videoWidth: Int {
        get { defaults.integer(forKey: Keys.videoWidth) }
        set {
            defaults.set(newValue, forKey: Keys.videoWidth)
            notifySettingsChanged()
        }
    }
    
    /// Video height
    var videoHeight: Int {
        get { defaults.integer(forKey: Keys.videoHeight) }
        set {
            defaults.set(newValue, forKey: Keys.videoHeight)
            notifySettingsChanged()
        }
    }
    
    /// Frame rate
    var frameRate: Int {
        get { defaults.integer(forKey: Keys.frameRate) }
        set {
            defaults.set(newValue, forKey: Keys.frameRate)
            notifySettingsChanged()
        }
    }
    
    // MARK: - Camera Offset Helpers
    
    /// Get camera offset as tuple
    var cameraOffset: (x: Float, y: Float, z: Float) {
        get { (cameraOffsetX, cameraOffsetY, cameraOffsetZ) }
        set {
            cameraOffsetX = newValue.x
            cameraOffsetY = newValue.y
            cameraOffsetZ = newValue.z
        }
    }
    
    /// Adjust camera offset by delta values
    func adjustCameraOffset(dx: Float = 0, dy: Float = 0, dz: Float = 0) {
        cameraOffsetX += dx
        cameraOffsetY += dy
        cameraOffsetZ += dz
    }
    
    /// Adjust focal length by delta
    func adjustFocalLength(delta: Float) {
        focalLength += delta
    }
    
    // MARK: - Notifications
    
    /// Post notification when settings change
    private func notifySettingsChanged() {
        NotificationCenter.default.post(
            name: SettingsManager.settingsChangedNotification,
            object: self
        )
    }
    
    // MARK: - Import/Export
    
    /// Export settings to JSON data
    func exportSettings() -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted
        return try? encoder.encode(currentSettings)
    }
    
    /// Import settings from JSON data
    func importSettings(from data: Data) -> Bool {
        guard let settings = try? JSONDecoder().decode(CameraSettings.self, from: data) else {
            return false
        }
        applySettings(settings)
        return true
    }
    
    /// Export settings to file
    func exportSettings(to url: URL) -> Bool {
        guard let data = exportSettings() else { return false }
        do {
            try data.write(to: url)
            return true
        } catch {
            print("[GazeCorrection] Failed to export settings: \(error)")
            return false
        }
    }
    
    /// Import settings from file
    func importSettings(from url: URL) -> Bool {
        do {
            let data = try Data(contentsOf: url)
            return importSettings(from: data)
        } catch {
            print("[GazeCorrection] Failed to import settings: \(error)")
            return false
        }
    }
}
