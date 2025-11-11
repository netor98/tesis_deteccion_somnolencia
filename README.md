# Driver Drowsiness Detection System

A comprehensive real-time driver drowsiness detection system using MediaPipe FaceMesh. The system detects multiple indicators of drowsiness including eye closure, yawning, and head pose (tilt).

## Features

- **Eye Closure Detection**: Monitors Eye Aspect Ratio (EAR) to detect when eyes are closed
- **Yawning Detection**: Tracks Mouth Aspect Ratio (MAR) to detect yawning
- **Head Pose Detection**: Monitors head tilt in three dimensions (roll, pitch, yaw)
- **Real-time Processing**: Uses WebRTC for real-time video and audio streaming
- **Configurable Thresholds**: Adjustable detection thresholds via Streamlit UI
- **Audio Alerts**: Plays alarm sound when drowsiness is detected

## Project Structure

```
drowsiness/
├── drowsiness_detector/          # Main package
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration constants and defaults
│   ├── detection.py             # Detection algorithms and VideoFrameHandler
│   ├── audio_handler.py        # Audio processing and alarm playback
│   └── utils.py                # Utility functions
├── audio/                       # Audio assets
│   └── wake_up.wav             # Alarm sound file
├── streamlit_app.py            # Main Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. **Clone or download the repository**

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Start the Streamlit application**:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Configure thresholds** in the web interface:
   - **Wait Time**: Time (in seconds) before alarm triggers
   - **Eye Separation (EAR)**: Threshold for eye closure detection (ideal: 0.15-0.2)
   - **Mouth Separation (MAR)**: Threshold for yawning detection (ideal: 0.5-0.7)
   - **Head Tilt Thresholds**: Roll, Pitch, and Yaw angles in degrees

3. **Allow camera and microphone access** when prompted

4. **Monitor the video feed** for real-time detection metrics

## Detection Metrics

### Eye Aspect Ratio (EAR)
- Measures the ratio of eye width to height
- Lower values indicate closed eyes
- Default threshold: 0.18

### Mouth Aspect Ratio (MAR)
- Measures mouth opening
- Higher values indicate yawning
- Default threshold: 0.6

### Head Pose Angles
- **Roll**: Left/right head tilt (default: 20°)
- **Pitch**: Up/down nodding (default: 15°)
- **Yaw**: Left/right head turning (default: 15°)

## Configuration

Default thresholds can be modified in `drowsiness_detector/config.py`:

```python
DEFAULT_THRESHOLDS = {
    "EAR_THRESH": 0.18,
    "MAR_THRESH": 0.6,
    "WAIT_TIME": 1.0,
    "ROLL_THRESH": 20.0,
    "PITCH_THRESH": 15.0,
    "YAW_THRESH": 15.0,
}
```

## Dependencies

- `streamlit` - Web application framework
- `streamlit-webrtc` - WebRTC integration for real-time streaming
- `mediapipe` - Face detection and landmark extraction
- `opencv-python` - Image processing
- `numpy` - Numerical operations
- `pydub` - Audio processing
- `av` - Audio/video frame handling

## Technical Details

### Detection Algorithms

1. **EAR Calculation**: Uses 6 facial landmarks per eye to calculate the eye aspect ratio
2. **MAR Calculation**: Uses 8 mouth landmarks to calculate mouth opening ratio
3. **Head Pose**: Calculates roll, pitch, and yaw angles using key facial landmarks

### Architecture

- **Modular Design**: Separated into detection, audio handling, and configuration modules
- **State Tracking**: Maintains temporal state for each detection metric
- **Thread-Safe**: Uses locks for shared state between video and audio callbacks

## Troubleshooting

- **Camera not working**: Ensure camera permissions are granted in your browser
- **Audio not playing**: Check that the `audio/wake_up.wav` file exists
- **Poor detection**: Adjust thresholds in the UI or ensure good lighting conditions

## License

This project is provided as-is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure code follows the existing structure and style.

