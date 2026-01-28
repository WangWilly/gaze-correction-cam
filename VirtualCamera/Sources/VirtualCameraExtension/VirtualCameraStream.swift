//
//  VirtualCameraStream.swift
//  VirtualCameraExtension
//
//  Provides the video stream with gaze-corrected frames.
//

import Foundation
import CoreMediaIO
import CoreMedia
import CoreVideo
import AVFoundation

/// Video stream that provides gaze-corrected frames
class VirtualCameraStream: NSObject {
    
    /// The underlying CMIOExtension stream
    let stream: CMIOExtensionStream
    
    /// Stream source delegate
    private let sourceDelegate: VirtualCameraStreamSource
    
    /// Frame processor for gaze correction
    private let frameProcessor: FrameProcessor
    
    /// Settings manager
    private let settingsManager: SettingsManager
    
    /// Stream UUID
    private let streamID: UUID
    
    /// Video format
    private let videoFormat: CMIOExtensionStreamFormat
    
    /// Initialize the video stream
    /// - Parameter settingsManager: Settings manager instance
    init(settingsManager: SettingsManager) {
        self.streamID = UUID()
        self.settingsManager = settingsManager
        self.frameProcessor = FrameProcessor(settingsManager: settingsManager)
        
        // Create video format description
        // Default to 1280x720 at 30fps
        let dimensions = CMVideoDimensions(width: 1280, height: 720)
        let formatDescription = VirtualCameraStream.createFormatDescription(dimensions: dimensions)
        
        self.videoFormat = CMIOExtensionStreamFormat(
            formatDescription: formatDescription,
            maxFrameDuration: CMTime(value: 1, timescale: 30),
            minFrameDuration: CMTime(value: 1, timescale: 30),
            validFrameDurations: nil
        )
        
        // Create stream source
        self.sourceDelegate = VirtualCameraStreamSource(
            streamID: streamID,
            format: videoFormat,
            frameProcessor: frameProcessor
        )
        
        // Create the stream
        self.stream = CMIOExtensionStream(
            localizedName: "Gaze Correction Video",
            streamID: streamID,
            direction: .source,
            clockType: .hostTime,
            source: sourceDelegate
        )
        
        super.init()
        
        print("[GazeCorrection] Stream initialized with format: 1280x720@30fps")
    }
    
    /// Create a format description for the specified dimensions
    private static func createFormatDescription(dimensions: CMVideoDimensions) -> CMFormatDescription {
        var formatDescription: CMFormatDescription?
        
        CMVideoFormatDescriptionCreate(
            allocator: kCFAllocatorDefault,
            codecType: kCVPixelFormatType_32BGRA,
            width: Int32(dimensions.width),
            height: Int32(dimensions.height),
            extensions: nil,
            formatDescriptionOut: &formatDescription
        )
        
        return formatDescription!
    }
    
    /// Start video capture
    func startCapture() {
        sourceDelegate.startCapture()
    }
    
    /// Stop video capture
    func stopCapture() {
        sourceDelegate.stopCapture()
    }
}

/// Stream source delegate that generates frames
class VirtualCameraStreamSource: NSObject, CMIOExtensionStreamSource {
    
    /// Stream UUID
    let streamID: UUID
    
    /// Video format
    let format: CMIOExtensionStreamFormat
    
    /// Frame processor
    let frameProcessor: FrameProcessor
    
    /// Timer for frame generation
    private var frameTimer: DispatchSourceTimer?
    
    /// Queue for frame generation
    private let frameQueue = DispatchQueue(label: "com.gazecorrection.framequeue")
    
    /// Stream clients
    private var streamingClients = Set<CMIOExtensionClient>()
    
    /// Current stream reference (set when streaming starts)
    private weak var currentStream: CMIOExtensionStream?
    
    /// Whether the stream is active
    private var isStreaming = false
    
    /// Frame counter
    private var frameCount: UInt64 = 0
    
    /// Video dimensions
    private let width: Int32 = 1280
    private let height: Int32 = 720
    
    /// Pixel buffer pool for efficient buffer allocation
    private var pixelBufferPool: CVPixelBufferPool?
    
    /// Real camera capture session
    private var captureSession: AVCaptureSession?
    private var videoOutput: AVCaptureVideoDataOutput?
    private var latestPixelBuffer: CVPixelBuffer?
    private let bufferLock = NSLock()
    
    /// Initialize the stream source
    init(streamID: UUID, format: CMIOExtensionStreamFormat, frameProcessor: FrameProcessor) {
        self.streamID = streamID
        self.format = format
        self.frameProcessor = frameProcessor
        super.init()
        
        setupPixelBufferPool()
        setupCameraCapture()
    }
    
    /// Set up the pixel buffer pool
    private func setupPixelBufferPool() {
        let poolAttributes: [String: Any] = [
            kCVPixelBufferPoolMinimumBufferCountKey as String: 3
        ]
        
        let pixelBufferAttributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any]
        ]
        
        CVPixelBufferPoolCreate(
            kCFAllocatorDefault,
            poolAttributes as CFDictionary,
            pixelBufferAttributes as CFDictionary,
            &pixelBufferPool
        )
    }
    
    /// Set up camera capture from the real camera
    private func setupCameraCapture() {
        captureSession = AVCaptureSession()
        captureSession?.sessionPreset = .hd1280x720
        
        // Find the default video device
        guard let videoDevice = AVCaptureDevice.default(for: .video) else {
            print("[GazeCorrection] No video device available")
            return
        }
        
        do {
            let input = try AVCaptureDeviceInput(device: videoDevice)
            if captureSession?.canAddInput(input) == true {
                captureSession?.addInput(input)
            }
        } catch {
            print("[GazeCorrection] Failed to create camera input: \(error)")
            return
        }
        
        // Set up video output
        videoOutput = AVCaptureVideoDataOutput()
        videoOutput?.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        videoOutput?.setSampleBufferDelegate(self, queue: frameQueue)
        
        if let output = videoOutput, captureSession?.canAddOutput(output) == true {
            captureSession?.addOutput(output)
        }
        
        print("[GazeCorrection] Camera capture configured")
    }
    
    /// Available stream properties
    var availableProperties: Set<CMIOExtensionProperty> {
        return [
            CMIOExtensionProperty(rawValue: "com.apple.cmio.property.streamActiveFormatIndex"),
            CMIOExtensionProperty(rawValue: "com.apple.cmio.property.streamFrameDuration")
        ]
    }
    
    /// Get stream properties
    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let streamProperties = CMIOExtensionStreamProperties(dictionary: [:])
        
        for property in properties {
            switch property.rawValue {
            case "com.apple.cmio.property.streamActiveFormatIndex":
                streamProperties.activeFormatIndex = 0
            case "com.apple.cmio.property.streamFrameDuration":
                streamProperties.frameDuration = CMTime(value: 1, timescale: 30)
            default:
                break
            }
        }
        
        return streamProperties
    }
    
    /// Set stream properties
    func setStreamProperties(_ properties: CMIOExtensionStreamProperties) throws {
        // Handle property changes
    }
    
    /// Return available formats
    var formats: [CMIOExtensionStreamFormat] {
        return [format]
    }
    
    /// Authorize a client to access the stream
    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        return true
    }
    
    /// Start streaming
    func startStream() throws {
        guard !isStreaming else { return }
        
        isStreaming = true
        captureSession?.startRunning()
        
        // Start frame generation timer as fallback
        startFrameTimer()
        
        print("[GazeCorrection] Stream started")
    }
    
    /// Stop streaming
    func stopStream() throws {
        guard isStreaming else { return }
        
        isStreaming = false
        captureSession?.stopRunning()
        stopFrameTimer()
        
        print("[GazeCorrection] Stream stopped")
    }
    
    /// Start frame generation timer
    private func startFrameTimer() {
        let timer = DispatchSource.makeTimerSource(queue: frameQueue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(33)) // ~30fps
        timer.setEventHandler { [weak self] in
            self?.generateFrame()
        }
        timer.resume()
        frameTimer = timer
    }
    
    /// Stop frame generation timer
    private func stopFrameTimer() {
        frameTimer?.cancel()
        frameTimer = nil
    }
    
    /// Start capture (called externally)
    func startCapture() {
        do {
            try startStream()
        } catch {
            print("[GazeCorrection] Failed to start stream: \(error)")
        }
    }
    
    /// Stop capture (called externally)
    func stopCapture() {
        do {
            try stopStream()
        } catch {
            print("[GazeCorrection] Failed to stop stream: \(error)")
        }
    }
    
    /// Generate a frame
    private func generateFrame() {
        guard isStreaming else { return }
        
        // Get the latest camera frame or create a test pattern
        var pixelBuffer: CVPixelBuffer?
        
        bufferLock.lock()
        if let latestBuffer = latestPixelBuffer {
            pixelBuffer = latestBuffer
        }
        bufferLock.unlock()
        
        // If no camera frame, create a test pattern
        if pixelBuffer == nil {
            pixelBuffer = createTestPatternBuffer()
        }
        
        guard let buffer = pixelBuffer else { return }
        
        // Process the frame (apply gaze correction)
        let processedBuffer = frameProcessor.processFrame(buffer)
        
        // Create timing info
        let now = CMClockGetTime(CMClockGetHostTimeClock())
        var timingInfo = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: 30),
            presentationTimeStamp: now,
            decodeTimeStamp: .invalid
        )
        
        // Create sample buffer
        var sampleBuffer: CMSampleBuffer?
        var formatDescription: CMFormatDescription?
        
        CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: processedBuffer,
            formatDescriptionOut: &formatDescription
        )
        
        guard let format = formatDescription else { return }
        
        CMSampleBufferCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: processedBuffer,
            dataReady: true,
            makeDataReadyCallback: nil,
            refcon: nil,
            formatDescription: format,
            sampleTiming: &timingInfo,
            sampleBufferOut: &sampleBuffer
        )
        
        guard let sample = sampleBuffer else { return }
        
        // Send to stream clients
        if let stream = currentStream {
            stream.send(
                sample,
                discontinuity: [],
                hostTimeInNanoseconds: UInt64(now.seconds * 1_000_000_000)
            )
        }
        
        frameCount += 1
    }
    
    /// Create a test pattern buffer (used when no camera is available)
    private func createTestPatternBuffer() -> CVPixelBuffer? {
        guard let pool = pixelBufferPool else { return nil }
        
        var pixelBuffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &pixelBuffer)
        
        guard let buffer = pixelBuffer else { return nil }
        
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else { return nil }
        
        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let width = CVPixelBufferGetWidth(buffer)
        
        // Create a gradient test pattern
        let pixels = baseAddress.assumingMemoryBound(to: UInt8.self)
        
        for y in 0..<height {
            for x in 0..<width {
                let offset = y * bytesPerRow + x * 4
                
                // BGRA format
                let phase = Double(frameCount) * 0.1
                let red = UInt8((sin(Double(x) * 0.02 + phase) + 1) * 127)
                let green = UInt8((sin(Double(y) * 0.02 + phase) + 1) * 127)
                let blue = UInt8((sin(Double(x + y) * 0.01 + phase) + 1) * 127)
                
                pixels[offset + 0] = blue     // B
                pixels[offset + 1] = green    // G
                pixels[offset + 2] = red      // R
                pixels[offset + 3] = 255      // A
            }
        }
        
        return buffer
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension VirtualCameraStreamSource: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        
        bufferLock.lock()
        latestPixelBuffer = pixelBuffer
        bufferLock.unlock()
    }
}
