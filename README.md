# Face Ratio Analyzer

A Python application that uses your webcam to analyze facial proportions and ratios. The app guides you through capturing front, left, and right views of your face, then calculates various facial ratios used in facial aesthetics and analysis.

## Features

- **Webcam Integration**: Real-time face detection using your webcam
- **Multi-View Capture**: Guides you through front, left, and right head positions
- **Comprehensive Ratios**: Calculates 8 different facial ratios:
  - Bizygomatic Width / Facial Height
  - Facial Thirds (upper, middle, lower)
  - Intercanthal Distance / Eye Width
  - Eye Width / Eye Height
  - Nose Width / Mouth Width
  - Face Width / Lower Face Height
  - Mandible Width / Face Width

## Requirements

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

## Installation

1. Clone or download this repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

The required packages are:
- `opencv-python` - For webcam capture and image processing
- `mediapipe` - For face landmark detection
- `numpy` - For numerical calculations
- `Pillow` - For image handling

## Usage

Run the application:
```bash
python face_ratio_analyzer.py
```

### Instructions

1. **Position yourself** in front of your webcam with good lighting
2. **Follow the on-screen instructions** to capture three views:
   - **Front view**: Face the camera directly
   - **Left view**: Turn your head to the left
   - **Right view**: Turn your head to the right
3. The app will automatically detect when you're in the correct position
4. Once all three views are captured, the results will be displayed in your terminal
5. Press 'q' at any time to quit the application

## Facial Ratios Explained

### Bizygomatic Width / Facial Height
- The ratio of cheekbone width to total face height
- Ideal range: ~0.7-0.8

### Facial Thirds
- Divides the face into three horizontal sections:
  - Upper: Hairline to glabella (between eyebrows)
  - Middle: Glabella to subnasale (under nose)
  - Lower: Subnasale to chin
- Ideal: Each section should be ~0.33 of total face height for harmonious proportions

### Intercanthal Distance / Eye Width
- Ratio of the distance between inner eye corners to eye width
- Ideal: ~1.0 (intercanthal distance should equal eye width)

### Eye Width / Eye Height
- Ratio of eye width to eye height
- Ideal: ~2.0-3.0

### Nose Width / Mouth Width
- Ratio of nose width at the widest point to mouth width
- Ideal: ~0.5-0.6

### Face Width / Lower Face Height
- Ratio of face width to lower face height (subnasale to chin)
- Ideal: ~1.2-1.4

### Mandible Width / Face Width
- Ratio of jaw width to overall face width
- Ideal: ~0.8-0.9

## Troubleshooting

- **No face detected**: Ensure you're in a well-lit area and your face is clearly visible in the frame
- **Webcam not opening**: Check that your webcam is properly connected and not in use by another application
- **Application crashes**: Ensure all dependencies are installed correctly with the correct versions

## Technical Details

- Uses MediaPipe Face Mesh for 468-point facial landmark detection
- OpenCV for webcam capture and display
- Ratios calculated from 3D facial landmarks projected to 2D
- Head position detection based on landmark visibility and nose tip position

## License

See LICENSE file for details.
