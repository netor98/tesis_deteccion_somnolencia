# Quick Start Guide - Drowsiness Detection System

## Running the Application

### Option 1: Using the Existing Virtual Environment (Recommended)

```bash
# Navigate to the project directory
cd /home/napo/Downloads/drowsiness

# Activate the virtual environment
source env/bin/activate

# Install/update dependencies (if needed)
pip install -r requirements.txt

# Run the Streamlit app
streamlit run streamlit_app.py
```

### Option 2: Fresh Installation

```bash
# Navigate to the project directory
cd /home/napo/Downloads/drowsiness

# Create a new virtual environment
python3 -m venv env

# Activate it
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## First Time Setup

1. **Start the backend API** (if you want to store data):

   ```bash
   # In a separate terminal, navigate to the backend
   cd /home/napo/risk-advisor-backend

   # Activate backend environment and run
   source env/bin/activate
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

2. **Configure the API URL** in the Streamlit app:

   -  Open the app in your browser (usually http://localhost:8501)
   -  In the sidebar, set the API URL (e.g., `http://localhost:8000`)
   -  Click "Reintentar Conexión" to verify connection

3. **Set up a trip**:

   -  Use the backend admin panel to create a conductor (driver)
   -  Start an active trip for that conductor
   -  In the Streamlit app, click "Buscar Viaje Activo" to connect

4. **Adjust detection thresholds**:
   -  Use the "Umbrales de Detección" sliders in the sidebar
   -  PERCLOS threshold: 10-70% (lower = more sensitive)
   -  Adjust based on your face and lighting conditions

## Usage

1. **Allow camera/microphone access** when prompted by your browser

2. **Position yourself** in front of the camera

   -  Face should be clearly visible
   -  Good lighting is important
   -  Camera should be at eye level or slightly above

3. **Monitor the metrics** displayed on screen:

   -  **EAR**: Eye Aspect Ratio (normal: 0.2-0.3, closed: <0.18)
   -  **MAR**: Mouth Aspect Ratio (normal: 0.2-0.4, yawn: >0.6)
   -  **PERCLOS**: % of time eyes are closed (alarm if >15%)
   -  **Head Pose**: Roll, Pitch, Yaw angles

4. **Alarm triggers when**:
   -  PERCLOS exceeds threshold (default 15%)
   -  Yawning detected for >1 second
   -  Head tilt detected for >3 seconds

## Troubleshooting

### Camera not working

```bash
# Check available cameras
ls /dev/video*

# Test camera with
python check_camera.py
```

### API connection issues

-  Verify backend is running: `curl http://localhost:8000/health`
-  Check firewall settings
-  Ensure correct IP address if on different machines

### Performance issues on Raspberry Pi

-  The app automatically detects Raspberry Pi and uses optimized settings
-  Close other applications to free up resources
-  Consider lowering video resolution

### Audio not playing

-  Check that `audio/wake_up.wav` exists
-  Verify browser allows audio autoplay
-  Check system audio settings

## Stopping the Application

1. In the terminal where Streamlit is running, press `Ctrl+C`
2. Deactivate the virtual environment: `deactivate`

## Running as a Service (Linux)

To run the detector automatically on boot:

```bash
# Copy the service file
sudo cp drowsiness-detector.service /etc/systemd/system/

# Edit paths in the service file if needed
sudo nano /etc/systemd/system/drowsiness-detector.service

# Enable and start
sudo systemctl enable drowsiness-detector
sudo systemctl start drowsiness-detector

# Check status
sudo systemctl status drowsiness-detector
```

## Testing Without Backend

You can run the detector without the backend API:

-  It will show "No conectado a ningún viaje" warning
-  Detection and alarms will still work
-  Data won't be saved to database

## Performance Tips

1. **For better accuracy**:

   -  Ensure good lighting (face clearly visible)
   -  Position camera at eye level
   -  Keep face centered in frame
   -  Calibrate thresholds for your face

2. **For better performance**:

   -  Close unnecessary browser tabs
   -  Use Chrome/Edge for better WebRTC support
   -  On Raspberry Pi: use wired connection, close other apps

3. **For production use**:
   -  Run backend API on reliable server
   -  Use fixed IP addresses
   -  Set up automatic restart on failure
   -  Monitor logs regularly

## Need Help?

Check the following files for more information:

-  `README.md` - Full documentation
-  `FIX_INSTANCE_RECREATION.md` - Performance optimization details
-  `STRUCTURE.md` - Project architecture
-  `notes.md` - Development notes
