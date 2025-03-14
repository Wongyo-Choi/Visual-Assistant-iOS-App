//  VideoCapture.swift
//  This file defines the VideoCapture class and related utilities, which manage the camera input,
//  capture video frames, and forward them to a delegate for processing.
//  All comments are written in British English for clarity and academic purposes.

import AVFoundation   // Provides the audio-visual capture functionalities.
import CoreVideo      // Provides low-level types and functionalities for video processing.
import UIKit          // Provides the user interface elements.

// MARK: - VideoCaptureDelegate Protocol

/// Protocol defining the method to handle video frame capture events.
/// Conforming types will receive video frames as CMSampleBuffer objects.
public protocol VideoCaptureDelegate: AnyObject {
    /// Called when a video frame is captured.
    ///
    /// - Parameters:
    ///   - capture: The VideoCapture instance capturing the video.
    ///   - didCaptureVideoFrame: The CMSampleBuffer containing the captured frame.
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame: CMSampleBuffer)
}

// MARK: - Best Capture Device Utility

/// Returns the best available camera device for a given position.
///
/// The function selects a device based on the specified camera position (back or front).
/// It considers user preferences (e.g. use of a telephoto camera) and available hardware.
///
/// - Parameter position: The desired camera position.
/// - Returns: The chosen AVCaptureDevice.
/// - Note: The function calls fatalError if no suitable device is found.
func bestCaptureDevice(for position: AVCaptureDevice.Position) -> AVCaptureDevice {
    if position == .back {
        // For the back camera, attempt to use the telephoto camera if enabled in user defaults.
        if UserDefaults.standard.bool(forKey: "use_telephoto"),
           let device = AVCaptureDevice.default(.builtInTelephotoCamera, for: .video, position: .back) {
            return device
        } else if let device = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back) {
            return device
        } else if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) {
            return device
        } else {
            fatalError("Expected back camera device is not available.")
        }
    } else if position == .front {
        // For the front camera, attempt to use the true depth camera first.
        if let device = AVCaptureDevice.default(.builtInTrueDepthCamera, for: .video, position: .front) {
            return device
        } else if let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .front) {
            return device
        } else {
            fatalError("Expected front camera device is not available.")
        }
    } else {
        fatalError("Unsupported camera position: \(position)")
    }
}

// MARK: - VideoCapture Class

/// Manages video capture from the device's camera, processes the video frames,
/// and provides a preview layer for displaying the camera feed.
///
/// The class configures the capture session, sets up inputs and outputs, and handles video orientation.
public class VideoCapture: NSObject {
    /// The preview layer displaying the camera feed.
    public var previewLayer: AVCaptureVideoPreviewLayer?
    
    /// Delegate to receive captured video frames.
    public weak var delegate: VideoCaptureDelegate?
    
    /// The camera device being used for capture, initialised to the best back camera.
    let captureDevice = bestCaptureDevice(for: .back)
    
    /// The capture session managing the flow of data from input to output.
    let captureSession = AVCaptureSession()
    
    /// Output for capturing video frames.
    let videoOutput = AVCaptureVideoDataOutput()
    
    /// Output for capturing still photos.
    var cameraOutput = AVCapturePhotoOutput()
    
    /// Dispatch queue for handling camera operations.
    let queue = DispatchQueue(label: "camera-queue")
    
    // MARK: - Setup Methods
    
    /// Sets up the video capture session with the specified session preset.
    ///
    /// The setup is performed asynchronously on a background queue, and the completion handler is called on the main thread.
    ///
    /// - Parameters:
    ///   - sessionPreset: The session preset defining the resolution. Defaults to hd1280x720.
    ///   - completion: A closure called with a Boolean indicating success.
    public func setUp(sessionPreset: AVCaptureSession.Preset = .hd1280x720, completion: @escaping (Bool) -> Void) {
        queue.async {
            let success = self.setUpCamera(sessionPreset: sessionPreset)
            DispatchQueue.main.async {
                completion(success)
            }
        }
    }
    
    /// Configures the capture session with inputs, outputs, and settings.
    ///
    /// - Parameter sessionPreset: The session preset to use for the capture session.
    /// - Returns: A Boolean indicating whether the camera setup was successful.
    private func setUpCamera(sessionPreset: AVCaptureSession.Preset) -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = sessionPreset
        
        // Create and add the video input.
        guard let videoInput = try? AVCaptureDeviceInput(device: captureDevice) else {
            return false
        }
        
        if captureSession.canAddInput(videoInput) {
            captureSession.addInput(videoInput)
        }
        
        // Set up the preview layer for displaying the camera feed.
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
        self.previewLayer = previewLayer
        
        // Configure the video output with appropriate settings.
        let settings: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: NSNumber(value: kCVPixelFormatType_32BGRA)
        ]
        videoOutput.videoSettings = settings
        videoOutput.alwaysDiscardsLateVideoFrames = true
        videoOutput.setSampleBufferDelegate(self, queue: queue)
        if captureSession.canAddOutput(videoOutput) {
            captureSession.addOutput(videoOutput)
        }
        
        // Add photo output if available.
        if captureSession.canAddOutput(cameraOutput) {
            captureSession.addOutput(cameraOutput)
        }
        
        // Set the video orientation based on the current device orientation.
        switch UIDevice.current.orientation {
        case .portrait:
            videoOutput.connection(with: .video)?.videoOrientation = .portrait
        case .portraitUpsideDown:
            videoOutput.connection(with: .video)?.videoOrientation = .portraitUpsideDown
        case .landscapeRight:
            videoOutput.connection(with: .video)?.videoOrientation = .landscapeLeft
        case .landscapeLeft:
            videoOutput.connection(with: .video)?.videoOrientation = .landscapeRight
        default:
            videoOutput.connection(with: .video)?.videoOrientation = .portrait
        }
        
        // Synchronise the preview layer's orientation with the video output.
        if let connection = videoOutput.connection(with: .video) {
            self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
        }
        
        // Configure the capture device for continuous auto-focus and auto-exposure.
        do {
            try captureDevice.lockForConfiguration()
            captureDevice.focusMode = .continuousAutoFocus
            captureDevice.focusPointOfInterest = CGPoint(x: 0.5, y: 0.5)
            captureDevice.exposureMode = .continuousAutoExposure
            captureDevice.unlockForConfiguration()
        } catch {
            print("Unable to configure the capture device.")
            return false
        }
        
        captureSession.commitConfiguration()
        return true
    }
    
    // MARK: - Control Methods
    
    /// Starts the capture session if it is not already running.
    public func start() {
        if !captureSession.isRunning {
            DispatchQueue.global(qos: .userInitiated).async { [weak self] in
                self?.captureSession.startRunning()
            }
        }
    }
    
    /// Stops the capture session if it is running.
    public func stop() {
        if captureSession.isRunning {
            captureSession.stopRunning()
        }
    }
    
    /// Updates the video orientation for the output connection and preview layer.
    ///
    /// Adjusts the video orientation based on the current device orientation and mirrors the video if the front camera is in use.
    func updateVideoOrientation() {
        guard let connection = videoOutput.connection(with: .video) else { return }
        switch UIDevice.current.orientation {
        case .portrait:
            connection.videoOrientation = .portrait
        case .portraitUpsideDown:
            connection.videoOrientation = .portraitUpsideDown
        case .landscapeRight:
            connection.videoOrientation = .landscapeLeft
        case .landscapeLeft:
            connection.videoOrientation = .landscapeRight
        default:
            return
        }
        
        // Determine if the video should be mirrored (for the front camera).
        let currentInput = self.captureSession.inputs.first as? AVCaptureDeviceInput
        connection.isVideoMirrored = (currentInput?.device.position == .front)
        self.previewLayer?.connection?.videoOrientation = connection.videoOrientation
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension VideoCapture: AVCaptureVideoDataOutputSampleBufferDelegate {
    /// Called when a new video frame is captured.
    ///
    /// Forwards the captured frame to the delegate.
    ///
    /// - Parameters:
    ///   - output: The AVCaptureOutput providing the video frame.
    ///   - sampleBuffer: The CMSampleBuffer containing the video frame.
    ///   - connection: The connection from which the video frame was received.
    public func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                              from connection: AVCaptureConnection) {
        delegate?.videoCapture(self, didCaptureVideoFrame: sampleBuffer)
    }
    
    /// Called when a video frame is dropped.
    ///
    /// This method can be utilised to handle or log dropped frames if needed.
    ///
    /// - Parameters:
    ///   - output: The AVCaptureOutput that dropped the frame.
    ///   - sampleBuffer: The CMSampleBuffer that was dropped.
    ///   - connection: The connection from which the frame was dropped.
    public func captureOutput(_ output: AVCaptureOutput, didDrop sampleBuffer: CMSampleBuffer,
                              from connection: AVCaptureConnection) {
        // Handle dropped frames if needed.
    }
}
