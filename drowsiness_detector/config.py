"""
Configuration module for drowsiness detection system.

Contains default thresholds, landmark indices, and other configuration constants.
"""

# Default detection thresholds
DEFAULT_THRESHOLDS = {
    "EAR_THRESH": 0.18,      # Eye Aspect Ratio threshold (ideal: 0.15-0.2)
    "MAR_THRESH": 0.6,       # Mouth Aspect Ratio threshold (ideal: 0.5-0.7)
    "WAIT_TIME": 1.0,        # Time in seconds before alarm triggers (for yawn)
    "HEAD_TILT_WAIT_TIME": 4.0,  # Time in seconds before head tilt alarm triggers (increased for less false positives)
    "PERCLOS_WINDOW": 30.0,  # PERCLOS calculation window in seconds (typical: 30-60)
    "PERCLOS_THRESH": 35.0,  # PERCLOS threshold percentage (typical: 15-20%)
    "ROLL_THRESH": 20.0,     # Head roll (tilt left/right) threshold in degrees
    "PITCH_THRESH": 12.0,    # Head pitch (nodding up/down) threshold in degrees (improved sensitivity)
    "YAW_THRESH": 15.0,      # Head yaw (turning left/right) threshold in degrees
}

# MediaPipe FaceMesh landmark indices
LANDMARK_INDICES = {
    "eye": {
        "left": [362, 385, 387, 263, 373, 380],
        "right": [33, 160, 158, 133, 153, 144],
    },
    "mouth": {
        "A": 61,   # left mouth corner
        "B": 291,  # right mouth corner
        "C": 78,   # upper left lip
        "D": 308,  # upper right lip
        "E": 13,   # upper center lip
        "F": 14,   # lower center lip
        "G": 82,   # lower left lip
        "H": 312,  # lower right lip
    },
    "head_pose": {
        "left_eye": 33,      # Left eye outer corner
        "right_eye": 263,    # Right eye outer corner
        "nose_tip": 4,       # Nose tip
        "forehead": 10,      # Forehead center
        "chin": 152,         # Chin center
        "face_left": 234,    # Left face edge
        "face_right": 454,   # Right face edge
    },
}

# MediaPipe FaceMesh configuration
MEDIAPIPE_CONFIG = {
    "max_num_faces": 1,
    "refine_landmarks": True,
    "min_detection_confidence": 0.5,
    "min_tracking_confidence": 0.5,
}

# Color constants (BGR format for OpenCV)
COLORS = {
    "RED": (0, 0, 255),
    "GREEN": (0, 255, 0),
    "BLUE": (255, 0, 0),
}

# Text display positions (relative positions, will be adjusted based on frame size)
TEXT_POSITIONS = {
    "EAR": (10, 30),
    "MAR": (10, 60),
    "HEAD_POSE": (10, 90),
}

