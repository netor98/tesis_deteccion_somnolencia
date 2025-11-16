"""
Streamlit application for driver drowsiness detection.

This is the main entry point for the drowsiness detection web application.
"""

import os
import av
import threading
import streamlit as st
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

from drowsiness_detector import VideoFrameHandler, AudioFrameHandler, DEFAULT_THRESHOLDS, get_api_client


# Define the audio file to use.
alarm_file_path = os.path.join("audio", "wake_up.wav")

# Verify audio file exists
if not os.path.exists(alarm_file_path):
    st.error(f"Audio file not found: {alarm_file_path}")
    st.stop()

# Streamlit Components web page config
st.set_page_config(
    page_title="Detección de Somnolencia",
    page_icon="",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
    menu_items={
    },
)

# API Configuration
API_BASE_URL = os.getenv("RISK_ADVISOR_API_URL", "http://localhost:8000")
api_client = get_api_client()
if hasattr(api_client, 'base_url'):
    api_client.base_url = API_BASE_URL

# Sidebar for configuration
with st.sidebar:
    st.header("Configuración")

    # Driver selection
    st.subheader("Seleccionar Conductor")
    conductor_id = st.number_input(
        "ID del Conductor",
        min_value=1,
        value=st.session_state.get("conductor_id", 1),
        step=1,
        help="Ingrese el ID del conductor para el viaje"
    )

    # Get active trip
    viaje_id = None
    if st.button("Conectar a Viaje Activo"):
        try:
            active_trip = api_client.get_active_trip_by_driver(conductor_id)
            if active_trip:
                viaje_id = active_trip.get("id_viaje")
                st.session_state["viaje_id"] = viaje_id
                st.session_state["conductor_id"] = conductor_id
                st.success(f"✅ Conectado al viaje #{viaje_id}")
            else:
                st.warning("⚠️ No hay un viaje activo para este conductor. El administrador debe iniciar un viaje primero.")
                st.session_state["viaje_id"] = None
        except Exception as e:
            st.error(f"❌ Error al conectar: {str(e)}")
            st.session_state["viaje_id"] = None

    # Show current connection status
    if st.session_state.get("viaje_id"):
        st.info(f"🔗 Conectado al viaje #{st.session_state['viaje_id']}")
        if st.button("Desconectar"):
            st.session_state["viaje_id"] = None
            st.rerun()
    else:
        st.warning("⚠️ No conectado a ningún viaje")

# Use default thresholds from config
thresholds = DEFAULT_THRESHOLDS.copy()

# Get viaje_id from session state
viaje_id = st.session_state.get("viaje_id")

# Initialize handlers (video_handler will be recreated if viaje_id changes)
if "video_handler" not in st.session_state or st.session_state.get("last_viaje_id") != viaje_id:
    st.session_state["video_handler"] = VideoFrameHandler(viaje_id=viaje_id)
    st.session_state["last_viaje_id"] = viaje_id

video_handler = st.session_state["video_handler"]
audio_handler = AudioFrameHandler(sound_file_path=alarm_file_path)

lock = threading.Lock()  # For thread-safe access & to prevent race-condition.
shared_state = {"play_alarm": False}


def video_frame_callback(frame: av.VideoFrame):
    """Callback function to process video frames."""
    frame = frame.to_ndarray(format="bgr24")  # Decode and convert frame to RGB
    # print(frame)

    frame, play_alarm = video_handler.process(frame, thresholds)  # Process frame
    with lock:
        shared_state["play_alarm"] = play_alarm  # Update shared state

    return av.VideoFrame.from_ndarray(frame, format="bgr24")  # Encode and return BGR frame


def audio_frame_callback(frame: av.AudioFrame):
    with lock:  # access the current “play_alarm” state
        play_alarm = shared_state["play_alarm"]

    new_frame: av.AudioFrame = audio_handler.process(frame, play_sound=play_alarm)
    return new_frame

# WebRTC streamer component (protocol that handles video and audio streaming)
ctx = webrtc_streamer(
    key="drowsiness-detection",
    video_frame_callback=video_frame_callback,
    audio_frame_callback=audio_frame_callback,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this to config for cloud deployment.
    media_stream_constraints={"video": {"height": {"ideal": 480}}, "audio": True},
    video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
)

