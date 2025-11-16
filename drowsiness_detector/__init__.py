"""
Drowsiness Detection System

A comprehensive driver drowsiness detection system using MediaPipe
for real-time face analysis including eye closure, yawning, and head pose detection.
"""

__version__ = "1.0.0"

from .detection import VideoFrameHandler
from .audio_handler import AudioFrameHandler
from .config import DEFAULT_THRESHOLDS, LANDMARK_INDICES
from .api_client import APIClient, get_api_client

__all__ = [
    "VideoFrameHandler",
    "AudioFrameHandler",
    "DEFAULT_THRESHOLDS",
    "LANDMARK_INDICES",
    "APIClient",
    "get_api_client",
]

