This project is a real-time drowsiness detection system designed for drivers. It uses computer vision and possibly audio cues to monitor the
driver’s alertness and trigger alarms if signs of drowsiness are detected. 

The goal: keep drivers awake, alive, and on the road—not in a ditch.

## Project Structure

• drowsiness_detector/ – Core detection logic (video/audio analysis, config, utils)

• audio/ – Alarm sounds (e.g., wake_up.wav)

• prolog_kb/ – Prolog knowledge base (for advanced logic/rules, maybe for demo or research)

• standalone_detector.py – Main script for running detection standalone

• streamlit_app.py – Web UI for demo/testing (Streamlit)

• auto_start.py, drowsiness-detector.service – Scripts/services for auto-start (e.g., on Raspberry Pi)

• requirements.txt – Python dependencies

• README.md – (You’re reading it)

## Features

• Real-time drowsiness detection using camera (and possibly audio)

• Alarm triggers (audio alert) when drowsiness is detected

• Can run as a standalone script or as a web app (Streamlit)

• Designed for Raspberry Pi or similar embedded systems

• Knowledge base integration (Prolog) for advanced rule-based logic (optional/experimental)

## Installation

### 1. Clone the repo

git clone <YOUR_REPO_URL>
cd drowsiness

### 2. Install dependencies

pip install -r requirements.txt

### Standalone Detector (CLI)

python standalone_detector.py

• This will start the detection loop using your default camera.

• When drowsiness is detected, an alarm sound will play.

### Streamlit Web App

streamlit run streamlit_app.py

• Launches a web interface for testing/demo purposes.

### (Optional) Prolog Knowledge Base

• See prolog_kb/ and prolog_engine.py for integrating rule-based logic.

## Hardware Requirements

• Camera (USB or PiCam)

• (Optional) Speakers for audio alarm

• (Recommended) Raspberry Pi 4 or better for embedded use

## Configuration

• Edit drowsiness_detector/config.py to tweak detection thresholds, alarm settings, etc.

## How it Works (High-Level)

1. Captures video frames from the camera.
2. Analyzes facial features (eyes, mouth) to detect signs of drowsiness (e.g., eye closure, yawning).
3. If drowsiness is detected, triggers an alarm sound.
4. (Optional) Uses Prolog rules for more complex detection logic.
