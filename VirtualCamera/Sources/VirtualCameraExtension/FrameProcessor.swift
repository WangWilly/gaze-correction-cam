//
//  FrameProcessor.swift
//  VirtualCameraExtension
//
//  Processes video frames with MediaPipe face detection and gaze correction.
//  This module uses MediaPipe for face landmark detection and communicates
//  with the Python backend for full gaze correction processing.
//

import Foundation
import CoreVideo
import CoreMedia
import Accelerate

/// Frame processor that applies gaze correction to video frames
class FrameProcessor {
    
    /// Settings manager
    private let settingsManager: SettingsManager
    
    /// Whether gaze correction is enabled
    private var gazeEnabled: Bool = true
    
    /// Processing queue
    private let processingQueue = DispatchQueue(label: "com.gazecorrection.processing")
    
    /// IPC communicator for Python backend
    private var backendCommunicator: BackendCommunicator?
    
    /// Face detection results cache
    private var lastFaceData: FaceDetectionResult?
    
    /// Frame counter for performance monitoring
    private var processedFrameCount: UInt64 = 0
    
    /// Initialize the frame processor
    /// - Parameter settingsManager: Settings manager instance
    init(settingsManager: SettingsManager) {
        self.settingsManager = settingsManager
        
        // Initialize backend communicator
        self.backendCommunicator = BackendCommunicator()
        
        // Load initial settings
        self.gazeEnabled = settingsManager.gazeEnabled
        
        // Observe settings changes
        setupSettingsObserver()
        
        print("[GazeCorrection] FrameProcessor initialized")
    }
    
    /// Set up observer for settings changes
    private func setupSettingsObserver() {
        NotificationCenter.default.addObserver(
            forName: SettingsManager.settingsChangedNotification,
            object: nil,
            queue: .main
        ) { [weak self] _ in
            self?.gazeEnabled = self?.settingsManager.gazeEnabled ?? true
        }
    }
    
    /// Process a video frame
    /// - Parameter pixelBuffer: Input pixel buffer from camera
    /// - Returns: Processed pixel buffer with gaze correction applied
    func processFrame(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        guard gazeEnabled else {
            return pixelBuffer
        }
        
        processedFrameCount += 1
        
        // Attempt to process with backend
        if let communicator = backendCommunicator,
           let processedBuffer = communicator.processFrame(pixelBuffer, settings: settingsManager.currentSettings) {
            return processedBuffer
        }
        
        // If backend is not available, apply local processing
        return applyLocalProcessing(pixelBuffer)
    }
    
    /// Apply local processing when backend is unavailable
    /// This provides basic functionality without full gaze correction
    private func applyLocalProcessing(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        // For now, just return the original buffer
        // In a full implementation, this could include:
        // - MediaPipe face detection
        // - Basic eye region highlighting
        // - Visual indicators showing backend status
        
        return pixelBuffer
    }
    
    deinit {
        NotificationCenter.default.removeObserver(self)
    }
}

// MARK: - Face Detection Result

/// Result of face detection
struct FaceDetectionResult {
    /// Left eye center position (normalized 0-1)
    var leftEyeCenter: CGPoint
    
    /// Right eye center position (normalized 0-1)
    var rightEyeCenter: CGPoint
    
    /// Face bounding box (normalized)
    var faceBounds: CGRect
    
    /// Confidence score (0-1)
    var confidence: Float
    
    /// Timestamp of detection
    var timestamp: Date
}

// MARK: - Backend Communicator

/// Communicates with the Python gaze correction backend via IPC
class BackendCommunicator {
    
    /// Named pipe path for frame data
    private let framePipePath = "/tmp/gaze_correction_frames"
    
    /// Named pipe path for results
    private let resultPipePath = "/tmp/gaze_correction_results"
    
    /// Whether the backend is connected
    private(set) var isConnected: Bool = false
    
    /// Connection check interval
    private var connectionCheckTimer: DispatchSourceTimer?
    
    /// Initialize the communicator
    init() {
        // Initial connection check
        checkBackendConnection()
        setupConnectionCheck()
    }
    
    /// Set up periodic connection checking
    private func setupConnectionCheck() {
        let timer = DispatchSource.makeTimerSource()
        timer.schedule(deadline: .now(), repeating: .seconds(5))
        timer.setEventHandler { [weak self] in
            self?.checkBackendConnection()
        }
        timer.resume()
        connectionCheckTimer = timer
    }
    
    /// Check if the backend is available
    private func checkBackendConnection() {
        let fileManager = FileManager.default
        isConnected = fileManager.fileExists(atPath: framePipePath) &&
                      fileManager.fileExists(atPath: resultPipePath)
        
        if isConnected {
            print("[GazeCorrection] Backend connection: Available")
        }
    }
    
    /// Process a frame through the backend
    /// - Parameters:
    ///   - pixelBuffer: Input pixel buffer
    ///   - settings: Current camera settings
    /// - Returns: Processed pixel buffer or nil if backend unavailable
    func processFrame(_ pixelBuffer: CVPixelBuffer, settings: CameraSettings) -> CVPixelBuffer? {
        guard isConnected else {
            return nil
        }
        
        // In a full implementation, this would:
        // 1. Convert pixelBuffer to image data
        // 2. Send to Python backend via named pipe or shared memory
        // 3. Receive processed frame
        // 4. Convert back to CVPixelBuffer
        
        // For now, return nil to indicate backend processing not available
        return nil
    }
    
    deinit {
        connectionCheckTimer?.cancel()
    }
}

// MARK: - Shared Memory Manager

/// Manages shared memory for efficient frame transfer
class SharedMemoryManager {
    
    /// Shared memory identifier
    private let shmName = "gaze_correction_shm"
    
    /// Shared memory size (enough for 1080p BGRA frame)
    private let shmSize = 1920 * 1080 * 4 + 1024 // Frame + header
    
    /// File descriptor for shared memory
    private var shmFd: Int32 = -1
    
    /// Pointer to shared memory
    private var shmPtr: UnsafeMutableRawPointer?
    
    /// Initialize shared memory
    init?() {
        guard setupSharedMemory() else {
            return nil
        }
    }
    
    /// Set up shared memory region
    private func setupSharedMemory() -> Bool {
        // Open or create shared memory
        shmFd = shm_open(shmName, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR)
        
        guard shmFd >= 0 else {
            print("[GazeCorrection] Failed to open shared memory")
            return false
        }
        
        // Set size
        guard ftruncate(shmFd, off_t(shmSize)) == 0 else {
            print("[GazeCorrection] Failed to set shared memory size")
            close(shmFd)
            return false
        }
        
        // Map memory
        shmPtr = mmap(nil, shmSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0)
        
        guard shmPtr != MAP_FAILED else {
            print("[GazeCorrection] Failed to map shared memory")
            close(shmFd)
            return false
        }
        
        print("[GazeCorrection] Shared memory initialized: \(shmSize) bytes")
        return true
    }
    
    /// Write frame data to shared memory
    func writeFrame(_ pixelBuffer: CVPixelBuffer, header: FrameHeader) -> Bool {
        guard let ptr = shmPtr else { return false }
        
        // Write header
        let headerPtr = ptr.assumingMemoryBound(to: FrameHeader.self)
        headerPtr.pointee = header
        
        // Lock pixel buffer for reading
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return false
        }
        
        // Copy frame data
        let dataPtr = ptr.advanced(by: MemoryLayout<FrameHeader>.size)
        let dataSize = CVPixelBufferGetDataSize(pixelBuffer)
        memcpy(dataPtr, baseAddress, min(dataSize, shmSize - MemoryLayout<FrameHeader>.size))
        
        return true
    }
    
    /// Read processed frame from shared memory
    func readFrame(into pixelBuffer: CVPixelBuffer) -> FrameHeader? {
        guard let ptr = shmPtr else { return nil }
        
        // Read header
        let headerPtr = ptr.assumingMemoryBound(to: FrameHeader.self)
        let header = headerPtr.pointee
        
        // Check if frame is ready
        guard header.frameReady else { return nil }
        
        // Lock pixel buffer for writing
        CVPixelBufferLockBaseAddress(pixelBuffer, [])
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else {
            return nil
        }
        
        // Copy frame data
        let dataPtr = ptr.advanced(by: MemoryLayout<FrameHeader>.size)
        let dataSize = CVPixelBufferGetDataSize(pixelBuffer)
        memcpy(baseAddress, dataPtr, min(dataSize, shmSize - MemoryLayout<FrameHeader>.size))
        
        return header
    }
    
    deinit {
        if let ptr = shmPtr {
            munmap(ptr, shmSize)
        }
        if shmFd >= 0 {
            close(shmFd)
            shm_unlink(shmName)
        }
    }
}

/// Frame header structure for shared memory communication
struct FrameHeader {
    /// Frame number
    var frameNumber: UInt64
    
    /// Frame width
    var width: Int32
    
    /// Frame height
    var height: Int32
    
    /// Bytes per row
    var bytesPerRow: Int32
    
    /// Pixel format (e.g., kCVPixelFormatType_32BGRA)
    var pixelFormat: UInt32
    
    /// Whether the frame is ready for reading
    var frameReady: Bool
    
    /// Timestamp in nanoseconds
    var timestamp: UInt64
    
    /// Camera settings
    var cameraOffsetX: Float
    var cameraOffsetY: Float
    var cameraOffsetZ: Float
    var focalLength: Float
    var ipd: Float
    var gazeEnabled: Bool
    
    /// Padding for alignment
    var reserved: (UInt64, UInt64, UInt64, UInt64)
}
