# Visual Assistant App

Visual Assistant is an iOS application designed to provide real‐time visual feedback using advanced object detection, tracking, and voice command recognition. The app utilises CoreML, Vision, AVFoundation, and UIKit to deliver a seamless experience for detecting objects, drawing bounding boxes, and issuing auditory alerts.

---

## Features

- **Real‐Time Object Detection:**  
  Implements multiple YOLO8 models (YOLO8n, YOLO8m, YOLO8x) to detect objects in real time using the CoreML framework.

- **Custom Thresholds:**  
  Utilises a custom `ThresholdProvider` to adjust IoU and confidence thresholds, allowing fine-tuning of detection sensitivity.

- **Visual Feedback:**  
  - Displays bounding boxes and labels on detected objects via the `BoundingBoxView` class.  
  - Draws arrow vectors to visualise object movement and provides visual cues.

- **Video Capture & Photo Functionality:**  
  - Captures video frames using the device’s camera.  
  - Supports switching between the front and back cameras.  
  - Allows photo capture and sharing directly from the app.

- **Voice Command Recognition:**  
  Integrates speech recognition to listen for specific commands (e.g. requesting a traffic summary) and employs speech synthesis for audio alerts.

- **User Interface Enhancements:**  
  - Adjustable sliders for confidence and IoU thresholds.  
  - Dynamic UI adjustments for portrait and landscape orientations.  
  - Zoom functionality via pinch gestures.

---

## Code Structure

- **ThresholdProvider.swift**  
  Defines the `ThresholdProvider` class which conforms to `MLFeatureProvider`. It supplies custom IoU and confidence thresholds to adjust model predictions.

- **BoundingBoxView.swift**  
  Contains the `BoundingBoxView` class responsible for drawing bounding boxes and displaying labels with confidence scores on detected objects.

- **AppDelegate.swift**  
  Manages global application configurations and lifecycle events, including setting up user defaults and handling device-specific settings such as battery monitoring.

- **ViewController.swift**  
  The main view controller that:  
  - Manages video capture, object detection via Vision, and object tracking.  
  - Handles UI interactions, such as switching cameras, adjusting thresholds, zooming, and capturing photos.  
  - Integrates voice command recognition and traffic summary functionalities.

- **VideoCapture.swift**  
  Manages the camera input and video capture session. It configures the capture session, sets up video outputs, and provides a preview layer for the live camera feed.

---

## Requirements

- **Xcode:** Latest version recommended.  
- **iOS Version:** iOS 13 or later. Some experimental features may require iOS 17 or later.  
- **Programming Language:** Swift.  
- **Frameworks:** CoreML, Vision, AVFoundation, UIKit, Speech.

---

## Installation & Setup

1. **Download the Project:**  
   Download the project as a zip file from the provided source, then extract its contents to a folder on your local machine.

2. **Open in Xcode:**  
   Locate the extracted folder and open the Xcode project by double-clicking the `YOLO.xcodeproj` file or by selecting it within Xcode.

3. **Connect Your iOS Device:**  
   Connect your iPhone (or iPad) to your computer using a USB cable. Ensure that your device is recognised by Xcode. If it isn’t, verify your cable connection and unlock your device.

4. **Enable Developer Mode:**  
   On your iOS device, navigate to **Settings > Privacy & Security > Developer Mode** and enable it. Follow any on-screen instructions to complete the process, as this is necessary for installing apps directly from Xcode.

5. **Select Your Device in Xcode:**  
   In Xcode’s toolbar, choose your connected device from the device selection dropdown. This step ensures that the build targets your physical device rather than a simulator.

6. **Configure Signing & Capabilities:**  
   Open the project’s settings in Xcode and make sure your Apple ID is added for code signing. Under **Signing & Capabilities**, select your development team to satisfy the provisioning requirements.

7. **Build and Run the App:**  
   Press the **Build** button (or use Command+B) to compile the project. Then, click the **Run** button (or use Command+R) to install the app on your connected device. Monitor Xcode’s build output to confirm a successful installation.

8. **Initial App Setup on Your Device:**  
   Once the app is installed, open it on your device. The app may prompt you to grant permissions (e.g., camera, microphone, and speech recognition). Accept these permissions to ensure all features work correctly.

9. **Adjust In-App Settings:**  
   If applicable, use the app’s settings menu to fine-tune detection thresholds and voice command configurations for optimal performance.

---

## Usage

- **Real‐Time Detection:**  
  Launch the app to start the live video feed and begin detecting objects in real time.

- **Adjusting Thresholds:**  
  Use the provided sliders to adjust confidence and IoU thresholds, fine-tuning the object detection sensitivity.

- **Switching Models:**  
  Use the segmented control to switch between YOLO8n, YOLO8m, and YOLO8x models.

- **Voice Commands:**  
  Speak specific phrases (e.g. "traffic situation") to trigger voice alerts and receive a summary of the current scene.

- **Photo Capture & Sharing:**  
  Capture photos from the video feed and share them using the built-in sharing functionality.

- **Zoom Functionality:**  
  Use pinch gestures to zoom in or out on the live video feed.
