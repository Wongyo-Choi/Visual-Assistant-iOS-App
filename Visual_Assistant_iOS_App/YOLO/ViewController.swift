//  ViewController.swift
//  This file defines the main view controller for the application.
//  It handles video capture, object detection using CoreML and Vision, voice command recognition,
//  UI updates, tracking of detected objects, and various user interactions such as switching cameras,
//  taking photos, and pinch-to-zoom functionality.
//  All comments are written in British English for clarity and academic purposes.

import AVFoundation     // Provides audio-visual capture and processing functionalities.
import CoreML           // Provides support for machine learning model integration.
import CoreMedia        // Provides low-level media types.
import UIKit            // Provides UI elements and event handling.
import Vision           // Provides computer vision functionalities.
import Speech           // Provides speech recognition functionalities.

/// Adds arrowheads to a given UIBezierPath between two points.
///
/// This function calculates the geometry required to add two lines at an angle
/// to represent an arrowhead at the end point of a line.
///
/// - Parameters:
///   - path: The UIBezierPath to which the arrowhead lines will be added.
///   - start: The starting point of the main line.
///   - end: The ending point of the main line, where the arrowhead is to be drawn.
///   - length: The length of the arrowhead lines. Defaults to 10.0.
///   - angle: The angle between the main line and each arrowhead line. Defaults to π/6.
func addArrowhead(to path: UIBezierPath, from start: CGPoint, to end: CGPoint, length: CGFloat = 10.0, angle: CGFloat = .pi/6) {
    let dx = end.x - start.x
    let dy = end.y - start.y
    let lineAngle = atan2(dy, dx)
    let arrowAngle1 = lineAngle + angle
    let arrowAngle2 = lineAngle - angle
    let arrowPoint1 = CGPoint(x: end.x - length * cos(arrowAngle1), y: end.y - length * sin(arrowAngle1))
    let arrowPoint2 = CGPoint(x: end.x - length * cos(arrowAngle2), y: end.y - length * sin(arrowAngle2))
    
    // Draw first line of the arrowhead.
    path.move(to: end)
    path.addLine(to: arrowPoint1)
    
    // Draw second line of the arrowhead.
    path.move(to: end)
    path.addLine(to: arrowPoint2)
}

/// Structure representing a tracked object for object detection and tracking.
///
/// This structure stores various properties that are used to track objects over successive frames,
/// including positional information, bounding box, class, alert status, and arrow drawing data.
struct TrackedObject {
    var id: Int                        // Unique identifier for the tracked object.
    var lastBoundingBox: CGRect        // The most recent bounding box of the object.
    var lastCentroid: CGPoint          // The centroid of the bounding box in the current frame.
    var prevCentroid: CGPoint          // The centroid of the bounding box in the previous frame.
    var lastArea: CGFloat              // The area of the bounding box in the current frame.
    var lastClass: String              // The most recent class label assigned to the object.
    var lastAlertTime: Date            // The time when the last alert was issued for the object.
    var lastSeen: Date                 // The time when the object was last detected.
    var consecutiveAlertCount: Int = 0 // A counter to track consecutive alerts for the object.
    
    // Arrow-related properties for visualising movement over frames.
    var arrowFrameCount: Int = 0                        // Frame counter for arrow drawing.
    var arrowStartPoint: CGPoint? = nil                 // The starting point for drawing the arrow.
    var arrowToDraw: (start: CGPoint, end: CGPoint)? = nil // The arrow (as a start and end point) to be drawn.
    
    // Signal frame counter to manage voice alerts for traffic signals.
    var signalFrameCounter: Int = 0
}

// MARK: - Global Model Initialisation

/// Global machine learning model initialisation.
/// The default model is YOLO8m, but this may be changed later via a segmented control.
///
/// The model configuration is also set up here, including any experimental options.
var mlModel = try! yolo8m(configuration: mlmodelConfig).model
var mlmodelConfig: MLModelConfiguration = {
    let config = MLModelConfiguration()
    // Use an experimental engine if available on iOS 17 and later.
    if #available(iOS 17.0, *) {
        config.setValue(1, forKey: "experimentalMLE5EngineUsage")
    }
    return config
}()

// MARK: - ViewController Class

/// The main view controller responsible for managing video capture,
/// object detection, tracking, voice commands, and UI interactions.
class ViewController: UIViewController {
    // MARK: - IBOutlets
    
    /// The view where the video preview is displayed.
    @IBOutlet var videoPreview: UIView!
    /// An additional view, purpose defined elsewhere.
    @IBOutlet var View0: UIView!
    /// Segmented control for selecting different machine learning models.
    @IBOutlet var segmentedControl: UISegmentedControl!
    /// Button to start video capture.
    @IBOutlet var playButtonOutlet: UIBarButtonItem!
    /// Button to pause video capture.
    @IBOutlet var pauseButtonOutlet: UIBarButtonItem!
    // The following slider IBOutlet for model selection has been removed as per project changes.
    // @IBOutlet var slider: UISlider!
    /// Slider to adjust the confidence threshold.
    @IBOutlet var sliderConf: UISlider!
    /// Slider to adjust the confidence threshold in landscape mode.
    @IBOutlet weak var sliderConfLandScape: UISlider!
    /// Slider to adjust the IoU (Intersection over Union) threshold.
    @IBOutlet var sliderIoU: UISlider!
    /// Slider to adjust the IoU threshold in landscape mode.
    @IBOutlet weak var sliderIoULandScape: UISlider!
    /// Label displaying the name of the selected model.
    @IBOutlet weak var labelName: UILabel!
    /// Label displaying the current frames per second (FPS) and inference time.
    @IBOutlet weak var labelFPS: UILabel!
    /// Label displaying the current zoom level.
    @IBOutlet weak var labelZoom: UILabel!
    /// Label displaying the app version.
    @IBOutlet weak var labelVersion: UILabel!
    /// Label for the slider description.
    @IBOutlet weak var labelSlider: UILabel!
    /// Label for the confidence slider description.
    @IBOutlet weak var labelSliderConf: UILabel!
    /// Label for the confidence slider description in landscape mode.
    @IBOutlet weak var labelSliderConfLandScape: UILabel!
    /// Label for the IoU slider description.
    @IBOutlet weak var labelSliderIoU: UILabel!
    /// Label for the IoU slider description in landscape mode.
    @IBOutlet weak var labelSliderIoULandScape: UILabel!
    /// Activity indicator to show processing state.
    @IBOutlet weak var activityIndicator: UIActivityIndicatorView!
    /// Image view used as a focus indicator.
    @IBOutlet weak var focus: UIImageView!
    /// Toolbar for additional UI controls.
    @IBOutlet weak var toolBar: UIToolbar!
    
    // MARK: - Detection and Tracking Properties
    
    /// Haptic feedback generator for selection changes.
    let selection = UISelectionFeedbackGenerator()
    /// The Vision model detector initialised from the global ML model.
    var detector = try! VNCoreMLModel(for: mlModel)
    /// Capture session used for video capture.
    var session: AVCaptureSession!
    /// Custom video capture class handling the camera input.
    var videoCapture: VideoCapture!
    /// Buffer to hold the current video frame.
    var currentBuffer: CVPixelBuffer?
    /// Counter for the number of frames processed.
    var framesDone = 0
    /// Timestamp marking the start of inference.
    var t0 = 0.0
    /// Duration of the most recent inference.
    var t1 = 0.0
    /// Smoothed inference time used for averaging.
    var t2 = 0.0
    /// Timestamp for FPS measurement.
    var t3 = CACurrentMediaTime()
    /// Smoothed FPS time used for averaging.
    var t4 = 0.0
    /// Stores the longer side of the captured frame.
    var longSide: CGFloat = 3
    /// Stores the shorter side of the captured frame.
    var shortSide: CGFloat = 4
    /// Flag to indicate if the frame size has been captured.
    var frameSizeCaptured = false
    
    // Developer mode and logging flags.
    let developerMode = UserDefaults.standard.bool(forKey: "developer_mode")
    let save_detections = false
    let save_frames = false
    
    // MARK: - Bounding Boxes and Colour Management
    
    /// Maximum number of bounding box views available.
    let maxBoundingBoxViews = 100
    /// Array to hold the bounding box views used for displaying detections.
    var boundingBoxViews = [BoundingBoxView]()
    /// Dictionary mapping class labels to their assigned colours.
    var colors: [String: UIColor] = [:]
    /// Predefined array of colours used for visualising different classes.
    let ultralyticsColorsolors: [UIColor] = [
        UIColor(red: 4/255, green: 42/255, blue: 255/255, alpha: 0.6),
        UIColor(red: 11/255, green: 219/255, blue: 235/255, alpha: 0.6),
        UIColor(red: 243/255, green: 243/255, blue: 243/255, alpha: 0.6),
        UIColor(red: 0/255, green: 223/255, blue: 183/255, alpha: 0.6),
        UIColor(red: 17/255, green: 31/255, blue: 104/255, alpha: 0.6),
        UIColor(red: 255/255, green: 111/255, blue: 221/255, alpha: 0.6),
        UIColor(red: 255/255, green: 68/255, blue: 79/255, alpha: 0.6),
        UIColor(red: 204/255, green: 237/255, blue: 0/255, alpha: 0.6),
        UIColor(red: 0/255, green: 243/255, blue: 68/255, alpha: 0.6),
        UIColor(red: 189/255, green: 0/255, blue: 255/255, alpha: 0.6),
        UIColor(red: 0/255, green: 180/255, blue: 255/255, alpha: 0.6),
        UIColor(red: 221/255, green: 0/255, blue: 186/255, alpha: 0.6),
        UIColor(red: 0/255, green: 255/255, blue: 255/255, alpha: 0.6),
        UIColor(red: 38/255, green: 192/255, blue: 0/255, alpha: 0.6),
        UIColor(red: 1/255, green: 255/255, blue: 179/255, alpha: 0.6),
        UIColor(red: 125/255, green: 36/255, blue: 255/255, alpha: 0.6),
        UIColor(red: 123/255, green: 0/255, blue: 104/255, alpha: 0.6),
        UIColor(red: 255/255, green: 27/255, blue: 108/255, alpha: 0.6),
        UIColor(red: 252/255, green: 109/255, blue: 47/255, alpha: 0.6),
        UIColor(red: 162/255, green: 255/255, blue: 11/255, alpha: 0.6)
    ]
    
    // MARK: - Voice and Speech Properties
    
    /// Synthesiser used for speech output.
    let speechSynthesizer = AVSpeechSynthesizer()
    /// Timestamp of the last red alert to avoid repeated warnings.
    var lastRedAlert = Date.distantPast
    /// Timestamp of the last green alert to avoid repeated warnings.
    var lastGreenAlert = Date.distantPast
    /// Timestamp of the last approach alert.
    var lastApproachAlert = Date.distantPast
    /// Timestamp of the last traffic summary alert to prevent duplicate announcements.
    var lastTrafficSummaryTime: Date = Date.distantPast
    
    // MARK: - Object Tracking Properties
    
    /// Dictionary storing the tracking history of objects across frames.
    var trackHistory: [Int: TrackedObject] = [:]
    /// Counter to assign unique identifiers to new tracked objects.
    var nextTrackID: Int = 0
    
    // MARK: - Speech Recognition Properties
    
    /// Audio engine for capturing audio for voice commands.
    let audioEngine = AVAudioEngine()
    /// Speech recogniser for processing voice commands (set to US English).
    var speechRecognizer: SFSpeechRecognizer? = SFSpeechRecognizer(locale: Locale(identifier: "en-GB"))
    /// Request object for speech audio buffer recognition.
    var recognitionRequest: SFSpeechAudioBufferRecognitionRequest?
    /// Task representing the ongoing speech recognition.
    var recognitionTask: SFSpeechRecognitionTask?
    
    // MARK: - Vision Request for Model Prediction
    
    /// Lazy initialisation of the Vision CoreML request using the detector.
    lazy var visionRequest: VNCoreMLRequest = {
        let request = VNCoreMLRequest(model: detector, completionHandler: { [weak self] request, error in
            self?.processObservations(for: request, error: error)
        })
        // Set the image cropping and scaling option.
        request.imageCropAndScaleOption = .scaleFill
        return request
    }()
    
    // MARK: - View Lifecycle Methods
    
    /// Called after the view has been loaded.
    ///
    /// This method sets up labels, bounding box views, orientation change notifications,
    /// the machine learning model, video capture and voice command recognition.
    override func viewDidLoad() {
        super.viewDidLoad()
        // Removed legacy slider initialisation code as per project requirements.
        setLabels()
        setUpBoundingBoxViews()
        setUpOrientationChangeNotification()
        setModel() // Initialise the model.
        startVideo()
        startVoiceCommandRecognition()
    }
    
    /// Handles view size changes due to device rotation.
    ///
    /// Adjusts UI elements such as sliders and toolbar appearance based on the new orientation.
    /// Also updates the preview layer frame.
    ///
    /// - Parameters:
    ///   - size: The new size for the container view.
    ///   - coordinator: The transition coordinator object.
    override func viewWillTransition(to size: CGSize, with coordinator: UIViewControllerTransitionCoordinator) {
        super.viewWillTransition(to: size, with: coordinator)
        if size.width > size.height {
            // Landscape mode adjustments.
            labelSliderConf.isHidden = true
            sliderConf.isHidden = true
            labelSliderIoU.isHidden = true
            sliderIoU.isHidden = true
            toolBar.setBackgroundImage(UIImage(), forToolbarPosition: .any, barMetrics: .default)
            toolBar.setShadowImage(UIImage(), forToolbarPosition: .any)
            
            labelSliderConfLandScape.isHidden = false
            sliderConfLandScape.isHidden = false
            labelSliderIoULandScape.isHidden = false
            sliderIoULandScape.isHidden = false
        } else {
            // Portrait mode adjustments.
            labelSliderConf.isHidden = false
            sliderConf.isHidden = false
            labelSliderIoU.isHidden = false
            sliderIoU.isHidden = false
            toolBar.setBackgroundImage(nil, forToolbarPosition: .any, barMetrics: .default)
            toolBar.setShadowImage(nil, forToolbarPosition: .any)
            
            labelSliderConfLandScape.isHidden = true
            sliderConfLandScape.isHidden = true
            labelSliderIoULandScape.isHidden = true
            sliderIoULandScape.isHidden = true
        }
        // Update the preview layer frame to match the new size.
        self.videoCapture.previewLayer?.frame = CGRect(x: 0, y: 0, width: size.width, height: size.height)
    }
    
    /// Sets up a notification to handle device orientation changes.
    private func setUpOrientationChangeNotification() {
        NotificationCenter.default.addObserver(self, selector: #selector(orientationDidChange),
                                               name: UIDevice.orientationDidChangeNotification, object: nil)
    }
    
    /// Handler for orientation change notifications.
    ///
    /// Updates the video capture orientation accordingly.
    @objc func orientationDidChange() {
        videoCapture.updateVideoOrientation()
    }
    
    /// Provides haptic feedback when a vibration action is triggered.
    ///
    /// - Parameter sender: The sender initiating the vibration action.
    @IBAction func vibrate(_ sender: Any) {
        selection.selectionChanged()
    }
    
    // MARK: - Model Setup
    
    /// Sets up the machine learning model for object detection.
    ///
    /// Re-initialises the Vision detector using the current ML model and applies a custom ThresholdProvider.
    /// Also resets timing variables used for measuring inference performance.
    func setModel() {
        detector = try! VNCoreMLModel(for: mlModel)
        // Apply the custom ThresholdProvider for adjusting IoU and confidence thresholds.
        detector.featureProvider = ThresholdProvider()
        
        let request = VNCoreMLRequest(model: detector, completionHandler: { [weak self] request, error in
            self?.processObservations(for: request, error: error)
        })
        request.imageCropAndScaleOption = .scaleFill
        visionRequest = request
        t2 = 0.0
        t3 = CACurrentMediaTime()
        t4 = 0.0
    }
    
    // MARK: - Segmented Control Action (Model Selection)
    
    /// Handles changes in the segmented control to switch between different models.
    ///
    /// Starts an activity indicator during the model switch, updates the label, loads the selected model,
    /// and reconfigures the detection pipeline.
    ///
    /// - Parameter sender: The sender object (typically the segmented control).
    @IBAction func indexChanged(_ sender: Any) {
        selection.selectionChanged()
        activityIndicator.startAnimating()
        
        /// Switch model based on the selected segment:
        /// 0: YOLO8n, 1: YOLO8m, 2: YOLO8x.
        switch segmentedControl.selectedSegmentIndex {
        case 0:
            self.labelName.text = "YOLO8n"
            mlModel = try! yolo8n(configuration: .init()).model
        case 1:
            self.labelName.text = "YOLO8m"
            mlModel = try! yolo8m(configuration: .init()).model
        case 2:
            self.labelName.text = "YOLO8x"
            mlModel = try! yolo8x(configuration: .init()).model
        default:
            break
        }
        setModel()
        setUpBoundingBoxViews()
        activityIndicator.stopAnimating()
    }
    
    /// Action triggered when the confidence and IoU sliders are changed.
    ///
    /// Updates the labels to reflect the current thresholds and applies them to the detector's ThresholdProvider.
    ///
    /// - Parameter sender: The sender object (typically the slider).
    @IBAction func sliderChanged(_ sender: Any) {
        let conf = Double(round(100 * sliderConf.value)) / 100
        let iou = Double(round(100 * sliderIoU.value)) / 100
        self.labelSliderConf.text = "\(conf) Confidence Threshold"
        self.labelSliderIoU.text = "\(iou) IoU Threshold"
        detector.featureProvider = ThresholdProvider(iouThreshold: iou, confidenceThreshold: conf)
    }
    
    /// Captures a photo from the current video frame.
    ///
    /// Utilises AVCapturePhotoSettings and a slight delay to stabilise focus before capturing.
    /// Also logs the capture duration.
    ///
    /// - Parameter sender: The sender object initiating the photo capture.
    @IBAction func takePhoto(_ sender: Any?) {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let settings = AVCapturePhotoSettings()
        usleep(20_000)  // Brief delay to stabilise focus.
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
        print("Photo capture done in: ", Double(DispatchTime.now().uptimeNanoseconds - t0) / 1E9)
    }
    
    /// Opens the company or project website when the logo button is tapped.
    ///
    /// - Parameter sender: The sender object initiating the action.
    @IBAction func logoButton(_ sender: Any) {
        selection.selectionChanged()
        if let link = URL(string: "https://www.ultralytics.com") {
            UIApplication.shared.open(link)
        }
    }
    
    /// Sets initial text for various labels in the UI.
    func setLabels() {
        self.labelName.text = "Best Model"
        self.labelVersion.text = "Version " + (UserDefaults.standard.string(forKey: "app_version") ?? "N/A")
    }
    
    /// Starts video capture and updates UI elements accordingly.
    ///
    /// Disables the play button and enables the pause button.
    ///
    /// - Parameter sender: The sender object (typically the play button).
    @IBAction func playButton(_ sender: Any) {
        selection.selectionChanged()
        self.videoCapture.start()
        playButtonOutlet.isEnabled = false
        pauseButtonOutlet.isEnabled = true
    }
    
    /// Pauses video capture and updates UI elements accordingly.
    ///
    /// Disables the pause button and enables the play button.
    ///
    /// - Parameter sender: The sender object (typically the pause button).
    @IBAction func pauseButton(_ sender: Any?) {
        selection.selectionChanged()
        self.videoCapture.stop()
        playButtonOutlet.isEnabled = true
        pauseButtonOutlet.isEnabled = false
    }
    
    /// Switches between the front and back cameras.
    ///
    /// Removes the current camera input and adds a new input from the opposite camera.
    /// Updates the video orientation after switching.
    ///
    /// - Parameter sender: The sender object initiating the camera switch.
    @IBAction func switchCameraTapped(_ sender: Any) {
        self.videoCapture.captureSession.beginConfiguration()
        let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput
        self.videoCapture.captureSession.removeInput(currentInput!)
        guard let currentPosition = currentInput?.device.position else { return }
        
        let nextCameraPosition: AVCaptureDevice.Position = currentPosition == .back ? .front : .back
        let newCameraDevice = bestCaptureDevice(for: nextCameraPosition)
        guard let videoInput1 = try? AVCaptureDeviceInput(device: newCameraDevice) else { return }
        self.videoCapture.captureSession.addInput(videoInput1)
        self.videoCapture.updateVideoOrientation()
        self.videoCapture.captureSession.commitConfiguration()
    }
    
    /// Shares a photo captured from the video feed.
    ///
    /// Uses AVCapturePhotoSettings to capture the photo and triggers the delegate for processing.
    ///
    /// - Parameter sender: The sender object initiating the share action.
    @IBAction func shareButton(_ sender: Any) {
        selection.selectionChanged()
        let settings = AVCapturePhotoSettings()
        self.videoCapture.cameraOutput.capturePhoto(with: settings, delegate: self as AVCapturePhotoCaptureDelegate)
    }
    
    /// Placeholder for the screenshot saving functionality.
    ///
    /// Currently not in use.
    ///
    /// - Parameter shouldSave: A Boolean flag indicating whether to save the screenshot. Defaults to true.
    @IBAction func saveScreenshotButton(_ shouldSave: Bool = true) {
        // Screenshot saving functionality (currently not used).
    }
    
    // MARK: - Bounding Box Views Setup
    
    /// Initialises and configures the bounding box views for displaying object detections.
    ///
    /// Ensures that the number of bounding box views matches the maximum required,
    /// and assigns colours to class labels based on a predefined colour array.
    func setUpBoundingBoxViews() {
        while boundingBoxViews.count < maxBoundingBoxViews {
            boundingBoxViews.append(BoundingBoxView())
        }
        guard let classLabels = mlModel.modelDescription.classLabels as? [String] else {
            fatalError("Class labels are missing from the model description")
        }
        var count = 0
        for label in classLabels {
            let color = ultralyticsColorsolors[count]
            count += 1
            if count > 19 { count = 0 }
            colors[label] = color
        }
    }
    
    // MARK: - Video Capture and Prediction
    
    /// Starts the video capture session and sets up the preview layer.
    ///
    /// Adds the preview layer to the videoPreview view and initialises bounding box views.
    /// Then, the video capture session is started.
    func startVideo() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.setUp(sessionPreset: .photo) { success in
            if success, let previewLayer = self.videoCapture.previewLayer {
                self.videoPreview.layer.addSublayer(previewLayer)
                self.videoCapture.previewLayer?.frame = self.videoPreview.bounds
                
                for box in self.boundingBoxViews {
                    box.addToLayer(self.videoPreview.layer)
                }
                self.videoCapture.start()
            }
        }
    }
    
    /// Processes a captured video frame for object detection.
    ///
    /// Converts the sample buffer to a CVPixelBuffer, adjusts for frame size,
    /// sets the appropriate image orientation based on the device orientation,
    /// and performs the Vision request for detection.
    ///
    /// - Parameter sampleBuffer: The CMSampleBuffer containing the video frame.
    func predict(sampleBuffer: CMSampleBuffer) {
        if currentBuffer == nil, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
            currentBuffer = pixelBuffer
            if !frameSizeCaptured {
                let frameWidth = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let frameHeight = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                longSide = max(frameWidth, frameHeight)
                shortSide = min(frameWidth, frameHeight)
                frameSizeCaptured = true
            }
            let imageOrientation: CGImagePropertyOrientation
            switch UIDevice.current.orientation {
            case .portrait:
                imageOrientation = .up
            case .portraitUpsideDown:
                imageOrientation = .down
            case .landscapeLeft:
                imageOrientation = .up
            case .landscapeRight:
                imageOrientation = .up
            case .unknown:
                imageOrientation = .up
            default:
                imageOrientation = .up
            }
            
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: imageOrientation, options: [:])
            if UIDevice.current.orientation != .faceUp {
                t0 = CACurrentMediaTime()
                do {
                    try handler.perform([visionRequest])
                } catch {
                    print(error)
                }
                t1 = CACurrentMediaTime() - t0
            }
            currentBuffer = nil
        }
    }
    
    /// Processes the results from the Vision request.
    ///
    /// Updates tracking information, displays bounding boxes and movement vectors,
    /// and calculates performance metrics such as FPS and inference time.
    ///
    /// - Parameters:
    ///   - request: The VNRequest that completed.
    ///   - error: An optional error if the request failed.
    func processObservations(for request: VNRequest, error: Error?) {
        DispatchQueue.main.async {
            let predictions: [VNRecognizedObjectObservation]
            if let results = request.results as? [VNRecognizedObjectObservation] {
                predictions = results
            } else {
                predictions = []
            }
            // Update tracking history based on the current predictions.
            self.updateTracking(with: predictions)
            // Display the predictions using bounding boxes and movement vectors.
            self.displayPredictions(predictions)
            
            // Smooth the inference time measurements.
            if self.t1 < 10.0 {
                self.t2 = self.t1 * 0.05 + self.t2 * 0.95
            }
            self.t4 = (CACurrentMediaTime() - self.t3) * 0.05 + self.t4 * 0.95
            self.labelFPS.text = String(format: "%.1f FPS - %.1f ms", 1 / self.t4, self.t2 * 1000)
            self.t3 = CACurrentMediaTime()
        }
    }
    
    // MARK: - Tracking Logic Integration
    
    /// Updates the tracking history for each detected object.
    ///
    /// Matches new predictions with existing tracked objects based on IoU,
    /// updates movement vectors, handles alerts for object approach and traffic signals,
    /// and manages arrow drawing for visualising movement.
    ///
    /// - Parameter predictions: An array of VNRecognizedObjectObservation representing detected objects.
    func updateTracking(with predictions: [VNRecognizedObjectObservation]) {
        let now = Date()
        let previewWidth = videoPreview.bounds.width
        let previewHeight = videoPreview.bounds.height
        // Define a reference vector from the centre to the bottom centre of the screen.
        let centerPoint = CGPoint(x: previewWidth / 2, y: previewHeight / 2)
        let bottomCenter = CGPoint(x: previewWidth / 2, y: previewHeight)
        let referenceVector = CGVector(dx: bottomCenter.x - centerPoint.x, dy: bottomCenter.y - centerPoint.y)
        
        // Function to calculate the Intersection over Union (IoU) between two CGRects.
        func iou(_ rect1: CGRect, _ rect2: CGRect) -> CGFloat {
            let intersection = rect1.intersection(rect2)
            if intersection.isNull { return 0 }
            let intersectionArea = intersection.width * intersection.height
            let unionArea = rect1.width * rect1.height + rect2.width * rect2.height - intersectionArea
            return intersectionArea / unionArea
        }
        
        // Update tracking for each prediction.
        for prediction in predictions {
            guard let bestLabel = prediction.labels.first?.identifier else { continue }
            // Convert the normalised bounding box to view coordinates.
            var rect = prediction.boundingBox
            if UIDevice.current.orientation == .portrait {
                let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -1)
                rect = rect.applying(transform)
                rect = VNImageRectForNormalizedRect(rect, Int(previewWidth), Int(previewHeight))
            } else {
                rect = VNImageRectForNormalizedRect(rect, Int(previewWidth), Int(previewHeight))
            }
            let centroid = CGPoint(x: rect.midX, y: rect.midY)
            let area = rect.width * rect.height
            
            // Try to match the current prediction with an existing tracked object using IoU.
            var matchedID: Int?
            for (id, tracked) in trackHistory {
                let overlap = iou(tracked.lastBoundingBox, rect)
                if overlap > 0.5 {
                    matchedID = id
                    break
                }
            }
            
            if let id = matchedID {
                // Update an existing tracked object.
                var tracked = trackHistory[id]!
                // Update the previous centroid.
                tracked.prevCentroid = tracked.lastCentroid
                // Calculate the movement vector.
                let movement = CGVector(dx: centroid.x - tracked.lastCentroid.x, dy: centroid.y - tracked.lastCentroid.y)
                let dotProduct = movement.dx * referenceVector.dx + movement.dy * referenceVector.dy
                let movementMag = sqrt(movement.dx * movement.dx + movement.dy * movement.dy)
                let refMag = sqrt(referenceVector.dx * referenceVector.dx + referenceVector.dy * referenceVector.dy)
                let cosineSim = movementMag > 0 && refMag > 0 ? dotProduct / (movementMag * refMag) : 0
                
                // Issue an alert if the object is approaching based on movement direction and area.
                if !bestLabel.lowercased().contains("pedestrian light") &&
                    area > tracked.lastArea &&
                    cosineSim > 0.9 &&
                    now.timeIntervalSince(tracked.lastAlertTime) > 3 {
                    tracked.consecutiveAlertCount += 1
                    if tracked.consecutiveAlertCount >= 3 {
                        speak(text: "Caution! Object approaching.")
                        tracked.lastAlertTime = now
                        tracked.consecutiveAlertCount = 0
                    }
                } else {
                    tracked.consecutiveAlertCount = 0
                }
                
                // Handle alerts for traffic signals.
                if bestLabel.lowercased().contains("red pedestrian light") || bestLabel.lowercased().contains("green pedestrian light") {
                    if tracked.lastClass.lowercased().contains("red pedestrian light") || tracked.lastClass.lowercased().contains("green pedestrian light") {
                        if tracked.lastClass.lowercased() == bestLabel.lowercased() {
                            if tracked.signalFrameCounter < 100 {
                                tracked.signalFrameCounter += 1
                            } else {
                                if bestLabel.lowercased().contains("red pedestrian light") {
                                    speak(text: "Warning! Please wait, red light.")
                                } else {
                                    speak(text: "You may cross the street. Green light.")
                                }
                                tracked.signalFrameCounter = 0
                            }
                        } else {
                            if bestLabel.lowercased().contains("red pedestrian light") {
                                speak(text: "Warning! Please wait, red light.")
                            } else {
                                speak(text: "You may cross the street. Green light.")
                            }
                            tracked.signalFrameCounter = 0
                        }
                    } else {
                        if bestLabel.lowercased().contains("red pedestrian light") {
                            speak(text: "Warning! Please wait, red light.")
                        } else {
                            speak(text: "You may cross the street. Green light.")
                        }
                        tracked.signalFrameCounter = 0
                    }
                }
                
                // Handle arrow drawing for movement visualisation.
                tracked.arrowFrameCount += 1
                if tracked.arrowStartPoint == nil {
                    tracked.arrowStartPoint = tracked.lastCentroid
                }
                if tracked.arrowFrameCount >= 5 {
                    let dx = centroid.x - tracked.arrowStartPoint!.x
                    let dy = centroid.y - tracked.arrowStartPoint!.y
                    let distance = sqrt(dx * dx + dy * dy)
                    if distance > 50 {
                        tracked.arrowToDraw = (start: tracked.arrowStartPoint!, end: centroid)
                    } else {
                        tracked.arrowToDraw = nil
                    }
                    tracked.arrowStartPoint = centroid
                    tracked.arrowFrameCount = 0
                }
                
                // Update tracked object properties.
                tracked.lastBoundingBox = rect
                tracked.lastCentroid = centroid
                tracked.lastArea = area
                tracked.lastClass = bestLabel
                tracked.lastSeen = now
                trackHistory[id] = tracked
            } else {
                // Create a new tracked object if no match was found.
                let newTracked = TrackedObject(id: nextTrackID,
                                               lastBoundingBox: rect,
                                               lastCentroid: centroid,
                                               prevCentroid: centroid,
                                               lastArea: area,
                                               lastClass: bestLabel,
                                               lastAlertTime: now,
                                               lastSeen: now,
                                               arrowFrameCount: 0,
                                               arrowStartPoint: centroid,
                                               arrowToDraw: nil,
                                               signalFrameCounter: 0)
                trackHistory[nextTrackID] = newTracked
                nextTrackID += 1
            }
        }
        
        // Remove tracked objects that have not been seen for more than 3 seconds.
        for (id, tracked) in trackHistory {
            if now.timeIntervalSince(tracked.lastSeen) > 3 {
                trackHistory.removeValue(forKey: id)
            }
        }
    }
    
    /// Speaks a given text using the speech synthesiser.
    ///
    /// - Parameter text: The text to be spoken.
    func speak(text: String) {
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = AVSpeechSynthesisVoice(language: "en-GB")
        utterance.volume = 1.0
        speechSynthesizer.speak(utterance)
    }
    
    // MARK: - Voice Command Recognition (Traffic Summary)
    
    /// Initiates the speech recognition authorisation process.
    ///
    /// If authorised, starts listening for voice commands.
    func startVoiceCommandRecognition() {
        SFSpeechRecognizer.requestAuthorization { authStatus in
            if authStatus == .authorized {
                self.startListening()
            } else {
                print("Speech recognition not authorized.")
            }
        }
    }
    
    /// Starts listening for voice commands.
    ///
    /// Configures the audio engine and recognition request, and sets up a handler to process recognised speech.
    func startListening() {
        recognitionRequest = SFSpeechAudioBufferRecognitionRequest()
        guard let recognitionRequest = recognitionRequest else {
            print("Unable to create the recognition request")
            return
        }
        recognitionRequest.shouldReportPartialResults = true
        let inputNode = audioEngine.inputNode
        let recordingFormat = inputNode.outputFormat(forBus: 0)
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { (buffer, when) in
            self.recognitionRequest?.append(buffer)
        }
        audioEngine.prepare()
        do {
            try audioEngine.start()
        } catch {
            print("audioEngine couldn't start due to an error.")
        }
        recognitionTask = speechRecognizer?.recognitionTask(with: recognitionRequest, resultHandler: { (result, error) in
            if let result = result {
                let spokenText = result.bestTranscription.formattedString.lowercased()
                print("Heard: \(spokenText)")
                if spokenText.contains("traffic") && spokenText.contains("situation") {
                    self.speakTrafficSummary()
                }
            }
            if error != nil || (result?.isFinal ?? false) {
                self.audioEngine.stop()
                inputNode.removeTap(onBus: 0)
                self.recognitionRequest = nil
                self.recognitionTask = nil
                self.startListening()
            }
        })
    }
    
    /// Provides a verbal summary of the current traffic situation.
    ///
    /// Analyses the tracked objects to summarise the number of objects,
    /// average movement, density distribution, and traffic signal status.
    func speakTrafficSummary() {
        let now = Date()
        if now.timeIntervalSince(lastTrafficSummaryTime) < 5 { return }
        lastTrafficSummaryTime = now
        
        let previewWidth = videoPreview.bounds.width
        let totalObjects = trackHistory.count
        if totalObjects == 0 {
            speak(text: "No objects are currently detected on the road.")
            return
        }
        var totalMovement: CGFloat = 0
        var leftCount = 0, centerCount = 0, rightCount = 0
        
        for (_, tracked) in trackHistory {
            let dx = tracked.lastCentroid.x - tracked.prevCentroid.x
            let dy = tracked.lastCentroid.y - tracked.prevCentroid.y
            let movement = sqrt(dx * dx + dy * dy)
            totalMovement += movement
            
            let relativeX = tracked.lastCentroid.x / previewWidth
            if relativeX < 0.33 {
                leftCount += 1
            } else if relativeX < 0.66 {
                centerCount += 1
            } else {
                rightCount += 1
            }
        }
        
        let avgMovement = totalMovement / CGFloat(totalObjects)
        let flowState: String
        if avgMovement < 5 {
            flowState = "heavy or congested"
        } else if avgMovement < 15 {
            flowState = "moderate"
        } else {
            flowState = "smooth"
        }
        
        var densitySummary = ""
        if leftCount > centerCount && leftCount > rightCount {
            densitySummary = "Many objects are concentrated on the left side, indicating possible buildup in that area."
        } else if centerCount > leftCount && centerCount > rightCount {
            densitySummary = "Most objects are in the centre, which might suggest high traffic density along the main path."
        } else if rightCount > leftCount && rightCount > centerCount {
            densitySummary = "The right side shows a higher object concentration, suggesting increased activity in that area."
        } else {
            densitySummary = "The objects are fairly evenly distributed across the view."
        }
        
        var signalStatus = ""
        for (_, tracked) in trackHistory {
            if tracked.lastClass.lowercased().contains("red pedestrian light") {
                signalStatus = "Also, a red signal is detected so please wait."
                break
            } else if tracked.lastClass.lowercased().contains("green pedestrian light") {
                signalStatus = "Also, a green signal is active so you may proceed."
            }
        }
        
        let summary = "Traffic Summary: \(totalObjects) objects detected. The average movement is about \(Int(avgMovement)) points per frame, indicating \(flowState) traffic flow. \(densitySummary) \(signalStatus)"
        speak(text: summary)
    }
    
    // MARK: - Zoom Handling
    
    /// Minimum allowed zoom factor.
    let minimumZoom: CGFloat = 1.0
    /// Maximum allowed zoom factor.
    let maximumZoom: CGFloat = 10.0
    /// Stores the last applied zoom factor.
    var lastZoomFactor: CGFloat = 1.0
    
    /// Handles pinch gestures to adjust the camera zoom.
    ///
    /// Updates the zoom factor based on the gesture scale, displays the current zoom level,
    /// and adjusts the font style of the zoom label during and after the pinch gesture.
    ///
    /// - Parameter pinch: The UIPinchGestureRecognizer recognising the pinch gesture.
    @IBAction func pinch(_ pinch: UIPinchGestureRecognizer) {
        let device = videoCapture.captureDevice
        
        // Calculates the minimum and maximum allowed zoom based on device capabilities.
        func minMaxZoom(_ factor: CGFloat) -> CGFloat {
            return min(min(max(factor, minimumZoom), maximumZoom), device.activeFormat.videoMaxZoomFactor)
        }
        
        // Updates the device's zoom factor.
        func update(scale factor: CGFloat) {
            do {
                try device.lockForConfiguration()
                defer { device.unlockForConfiguration() }
                device.videoZoomFactor = factor
            } catch {
                print("\(error.localizedDescription)")
            }
        }
        
        let newScaleFactor = minMaxZoom(pinch.scale * lastZoomFactor)
        switch pinch.state {
        case .began, .changed:
            update(scale: newScaleFactor)
            self.labelZoom.text = String(format: "%.2fx", newScaleFactor)
            self.labelZoom.font = UIFont.preferredFont(forTextStyle: .title2)
        case .ended:
            lastZoomFactor = minMaxZoom(newScaleFactor)
            update(scale: lastZoomFactor)
            self.labelZoom.font = UIFont.preferredFont(forTextStyle: .body)
        default: break
        }
    }
}

// MARK: - Video Capture Delegate Extension

extension ViewController: VideoCaptureDelegate {
    /// Delegate method called when a video frame is captured.
    ///
    /// Passes the captured frame to the predict method for processing.
    ///
    /// - Parameters:
    ///   - capture: The VideoCapture instance.
    ///   - sampleBuffer: The CMSampleBuffer containing the video frame.
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame sampleBuffer: CMSampleBuffer) {
        predict(sampleBuffer: sampleBuffer)
    }
}

// MARK: - AVCapturePhotoCapture Delegate Extension

extension ViewController: AVCapturePhotoCaptureDelegate {
    /// Delegate method called after a photo is processed.
    ///
    /// Converts the captured photo data into an image, adjusts orientation based on the camera,
    /// presents a preview, and invokes the share functionality.
    ///
    /// - Parameters:
    ///   - output: The AVCapturePhotoOutput object.
    ///   - photo: The AVCapturePhoto containing the captured image data.
    ///   - error: An optional error if processing failed.
    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        if let error = error {
            print("error occurred: \(error.localizedDescription)")
        }
        if let dataImage = photo.fileDataRepresentation() {
            let dataProvider = CGDataProvider(data: dataImage as CFData)
            let cgImageRef: CGImage! = CGImage(jpegDataProviderSource: dataProvider!, decode: nil,
                                                shouldInterpolate: true, intent: .defaultIntent)
            var isCameraFront = false
            if let currentInput = self.videoCapture.captureSession.inputs.first as? AVCaptureDeviceInput,
               currentInput.device.position == .front {
                isCameraFront = true
            }
            var orientation: CGImagePropertyOrientation = isCameraFront ? .leftMirrored : .right
            switch UIDevice.current.orientation {
            case .landscapeLeft:
                orientation = isCameraFront ? .downMirrored : .up
            case .landscapeRight:
                orientation = isCameraFront ? .upMirrored : .down
            default:
                break
            }
            var image = UIImage(cgImage: cgImageRef, scale: 0.5, orientation: .right)
            if let orientedCIImage = CIImage(image: image)?.oriented(orientation),
               let cgImage = CIContext().createCGImage(orientedCIImage, from: orientedCIImage.extent) {
                image = UIImage(cgImage: cgImage)
            }
            let imageView = UIImageView(image: image)
            imageView.contentMode = .scaleAspectFill
            imageView.frame = videoPreview.frame
            let imageLayer = imageView.layer
            videoPreview.layer.insertSublayer(imageLayer, above: videoCapture.previewLayer)
            
            let bounds = UIScreen.main.bounds
            UIGraphicsBeginImageContextWithOptions(bounds.size, true, 0.0)
            self.View0.drawHierarchy(in: bounds, afterScreenUpdates: true)
            let img = UIGraphicsGetImageFromCurrentImageContext()
            UIGraphicsEndImageContext()
            imageLayer.removeFromSuperlayer()
            let activityViewController = UIActivityViewController(activityItems: [img!], applicationActivities: nil)
            activityViewController.popoverPresentationController?.sourceView = self.View0
            self.present(activityViewController, animated: true, completion: nil)
        } else {
            print("AVCapturePhotoCaptureDelegate Error")
        }
    }
}

// MARK: - Display Predictions Extension

extension ViewController {
    /// Displays the detection predictions by drawing bounding boxes and movement vectors.
    ///
    /// Processes the predictions to convert normalised coordinates to view coordinates,
    /// applies transformations based on device orientation, and displays labels with confidence values.
    /// Also draws arrow vectors to visualise object movement.
    ///
    /// - Parameter predictions: An array of VNRecognizedObjectObservation representing the detected objects.
    func displayPredictions(_ predictions: [VNRecognizedObjectObservation]) {
        var str = ""
        let date = Date()
        let calendar = Calendar.current
        let hour = calendar.component(.hour, from: date)
        let minutes = calendar.component(.minute, from: date)
        let seconds = calendar.component(.second, from: date)
        let nanoseconds = calendar.component(.nanosecond, from: date)
        let sec_day = Double(hour) * 3600.0 + Double(minutes) * 60.0 + Double(seconds) + Double(nanoseconds) / 1E9
        
        // Retrieve the dimensions of the video preview.
        let width = videoPreview.bounds.width
        let height = videoPreview.bounds.height
        
        // Remove any previously drawn movement vector layers.
        videoPreview.layer.sublayers?.removeAll(where: { $0.name == "movementVector" })
        
        if UIDevice.current.orientation == .portrait {
            var ratio: CGFloat = 1.0
            if videoCapture.captureSession.sessionPreset == .photo {
                ratio = (height / width) / (4.0 / 3.0)
            } else {
                ratio = (height / width) / (16.0 / 9.0)
            }
            
            for i in 0..<boundingBoxViews.count {
                if i < predictions.count {
                    let prediction = predictions[i]
                    var rect = prediction.boundingBox
                    
                    switch UIDevice.current.orientation {
                    case .portraitUpsideDown:
                        rect = CGRect(x: 1.0 - rect.origin.x - rect.width,
                                      y: 1.0 - rect.origin.y - rect.height,
                                      width: rect.width, height: rect.height)
                    case .landscapeLeft:
                        rect = CGRect(x: rect.origin.x, y: rect.origin.y,
                                      width: rect.width, height: rect.height)
                    case .landscapeRight:
                        rect = CGRect(x: rect.origin.x, y: rect.origin.y,
                                      width: rect.width, height: rect.height)
                    case .unknown:
                        print("Unknown orientation; predictions may be affected")
                        fallthrough
                    default: break
                    }
                    
                    if ratio >= 1 {
                        let offset = (1 - ratio) * (0.5 - rect.minX)
                        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: offset, y: -1)
                        rect = rect.applying(transform)
                        rect.size.width *= ratio
                    } else {
                        let offset = (ratio - 1) * (0.5 - rect.maxY)
                        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: offset - 1)
                        rect = rect.applying(transform)
                        ratio = (height / width) / (3.0 / 4.0)
                        rect.size.height /= ratio
                    }
                    
                    rect = VNImageRectForNormalizedRect(rect, Int(width), Int(height))
                    let bestClass = prediction.labels[0].identifier
                    let confidence = prediction.labels[0].confidence
                    let label = String(format: "%@ %.1f", bestClass, confidence * 100)
                    let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                    boundingBoxViews[i].show(frame: rect, label: label,
                                             color: colors[bestClass] ?? UIColor.white,
                                             alpha: alpha)
                    
                    if developerMode, save_detections {
                        str += String(format: "%.3f %.3f %.3f %@ %.2f %.1f %.1f %.1f %.1f\n",
                                      sec_day, 0.0, UIDevice.current.batteryLevel,
                                      bestClass, confidence,
                                      rect.origin.x, rect.origin.y,
                                      rect.size.width, rect.size.height)
                    }
                } else {
                    boundingBoxViews[i].hide()
                }
            }
        } else {
            let frameAspectRatio = longSide / shortSide
            let viewAspectRatio = width / height
            var scaleX: CGFloat = 1.0
            var scaleY: CGFloat = 1.0
            var offsetX: CGFloat = 0.0
            var offsetY: CGFloat = 0.0
            
            if frameAspectRatio > viewAspectRatio {
                scaleY = height / shortSide
                scaleX = scaleY
                offsetX = (longSide * scaleX - width) / 2
            } else {
                scaleX = width / longSide
                scaleY = scaleX
                offsetY = (shortSide * scaleY - height) / 2
            }
            
            for i in 0..<boundingBoxViews.count {
                if i < predictions.count {
                    let prediction = predictions[i]
                    var rect = prediction.boundingBox
                    rect.origin.x = rect.origin.x * longSide * scaleX - offsetX
                    rect.origin.y = height - (rect.origin.y * shortSide * scaleY - offsetY + rect.size.height * shortSide * scaleY)
                    rect.size.width *= longSide * scaleX
                    rect.size.height *= shortSide * scaleY
                    
                    let bestClass = prediction.labels[0].identifier
                    let confidence = prediction.labels[0].confidence
                    let label = String(format: "%@ %.1f", bestClass, confidence * 100)
                    let alpha = CGFloat((confidence - 0.2) / (1.0 - 0.2) * 0.9)
                    boundingBoxViews[i].show(frame: rect, label: label,
                                             color: colors[bestClass] ?? UIColor.white,
                                             alpha: alpha)
                } else {
                    boundingBoxViews[i].hide()
                }
            }
        }
        
        if developerMode, save_detections {
            // Code for saving detection logs (omitted as per project instructions).
        }
        if developerMode, save_frames {
            // Code for saving frame logs (omitted as per project instructions).
        }
        
        // --- Draw Movement Vectors (Arrow) --- //
        for (id, var tracked) in trackHistory {
            if let arrow = tracked.arrowToDraw {
                let arrowPath = UIBezierPath()
                arrowPath.move(to: arrow.start)
                arrowPath.addLine(to: arrow.end)
                addArrowhead(to: arrowPath, from: arrow.start, to: arrow.end)
                
                let arrowLayer = CAShapeLayer()
                arrowLayer.name = "movementVector"
                arrowLayer.path = arrowPath.cgPath
                arrowLayer.strokeColor = UIColor.red.cgColor
                arrowLayer.lineWidth = 2.0
                arrowLayer.fillColor = UIColor.clear.cgColor
                videoPreview.layer.addSublayer(arrowLayer)
                tracked.arrowToDraw = nil
                trackHistory[id] = tracked
            }
        }
    }
}
