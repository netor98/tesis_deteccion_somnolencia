import os
import av
import threading
import streamlit as st
import streamlit_nested_layout
from streamlit_webrtc import VideoHTMLAttributes, webrtc_streamer

from audio_handling import AudioFrameHandler
from drowsy_detection import VideoFrameHandler
# from ads import css_string


# Define the audio file to use.
alarm_file_path = os.path.join("audio", "wake_up.wav")
print(alarm_file_path)

# Streamlit Components web page config
st.set_page_config(
    page_title="",
    page_icon="",
    layout="wide",  # centered, wide
    initial_sidebar_state="expanded",
    menu_items={
    },
)


col1, col2 = st.columns(spec=[6, 2], gap="medium")

# Sidebar
with col1:
    st.title("")
    with st.container():
        c1, c2 = st.columns(spec=[1, 1])
        with c1:
            # The amount of time (in seconds) to wait before sounding the alarm.
            WAIT_TIME = st.slider("Segundos antes de que suene la alarma:", 0.0, 5.0, 1.0, 0.25)

        with c2:
            # Lowest valid value of Eye Aspect Ratio. Ideal values [0.15, 0.2].
            EAR_THRESH = st.slider("Separación de ojo:", 0.0, 0.4, 0.18, 0.01)

        with st.container():
            c3, c4 = st.columns(spec=[1, 1])
            with c3:
                # Lowest valid value of Mouth Aspect Ratio. Ideal values [0.5, 0.7].
                MAR_THRESH = st.slider("Separación de boca:", 0.0, 1.0, 0.6, 0.01)

# Store the thresholds in a dictionary to pass to the callback function.
thresholds = {
    "EAR_THRESH": EAR_THRESH,
    "MAR_THRESH": MAR_THRESH,
    "WAIT_TIME": WAIT_TIME,
}

# For streamlit-webrtc
video_handler = VideoFrameHandler() # Initialize the video frame handler (drowsiness detection)
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
with col1:
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Add this to config for cloud deployment.
        media_stream_constraints={"video": {"height": {"ideal": 480}}, "audio": True},
        video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=False, muted=False),
    )

