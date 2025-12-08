"""
Detection algorithms for drowsiness, yawning, and head pose.

Contains functions and classes for calculating EAR, MAR, head pose angles,
and processing video frames.
"""

import time
import numpy as np
import mediapipe as mp
import cv2

from .utils import distance, get_landmark_coordinates
from .config import LANDMARK_INDICES, MEDIAPIPE_CONFIG, MEDIAPIPE_CONFIG_RASPBERRY_PI, COLORS, TEXT_POSITIONS


def get_mediapipe_app(use_raspberry_pi_optimization=False, **kwargs):
    """Initialize and return MediaPipe FaceMesh Solution Graph object.

    Args:
        use_raspberry_pi_optimization: If True, uses optimized config for Raspberry Pi
        **kwargs: MediaPipe configuration parameters (override defaults)

    Returns:
        FaceMesh: Initialized MediaPipe FaceMesh object
    """
    # Use Raspberry Pi optimized config if requested
    base_config = MEDIAPIPE_CONFIG_RASPBERRY_PI if use_raspberry_pi_optimization else MEDIAPIPE_CONFIG
    config = {**base_config, **kwargs}
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=config["max_num_faces"],
        refine_landmarks=config["refine_landmarks"],
        min_detection_confidence=config["min_detection_confidence"],
        min_tracking_confidence=config["min_tracking_confidence"],
    )
    return face_mesh


def calculate_ear(landmarks, eye_indices, frame_width, frame_height):
    """Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: Detected landmarks list
        eye_indices: Index positions of the chosen landmarks [P1, P2, P3, P4, P5, P6]
        frame_width: Width of captured frame
        frame_height: Height of captured frame

    Returns:
        tuple: (ear, coords_points) where ear is float and coords_points is list or None
    """
    try:
        coords_points = get_landmark_coordinates(landmarks, eye_indices, frame_width, frame_height)

        if None in coords_points:
            return 0.0, None

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        if P1_P4 > 0:
            ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)
        else:
            ear = 0.0

    except Exception:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    """Calculate average Eye Aspect Ratio for both eyes.

    Args:
        landmarks: Detected landmarks list
        left_eye_idxs: Left eye landmark indices
        right_eye_idxs: Right eye landmark indices
        image_w: Width of captured frame
        image_h: Height of captured frame

    Returns:
        tuple: (avg_ear, (left_coords, right_coords))
    """
    left_ear, left_lm_coordinates = calculate_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = calculate_ear(landmarks, right_eye_idxs, image_w, image_h)
    avg_ear = (left_ear + right_ear) / 2.0

    return avg_ear, (left_lm_coordinates, right_lm_coordinates)


def calculate_mar(landmarks, mouth_idxs, image_w, image_h):
    """Calculate Mouth Aspect Ratio.

    Args:
        landmarks: Detected landmarks list
        mouth_idxs: Dictionary of mouth landmark indices
        image_w: Width of captured frame
        image_h: Height of captured frame

    Returns:
        tuple: (mar, coords_points) where mar is float and coords_points is list or None
    """
    try:
        coords_points = get_landmark_coordinates(landmarks, mouth_idxs, image_w, image_h)

        # Check if any coordinate is None
        if None in coords_points.values():
            return 0.0, None

        # Calculate the distances
        ab_dist = distance(coords_points["A"], coords_points["B"])
        cd_dist = distance(coords_points["C"], coords_points["D"])
        ef_dist = distance(coords_points["E"], coords_points["F"])
        gh_dist = distance(coords_points["G"], coords_points["H"])

        # Calculate the Mouth Aspect Ratio (MAR)
        if ab_dist > 0:
            mar = (cd_dist + ef_dist + gh_dist) / (3.0 * ab_dist)
        else:
            mar = 0.0

    except Exception:
        mar = 0.0
        coords_points = None

    return mar, coords_points


def calculate_head_pose(landmarks, head_pose_idxs, image_w, image_h):
    """Calculate head pose angles (roll, pitch, yaw) using facial landmarks.

    Improved calculation using multiple reference points for better accuracy,
    especially for pitch (forward/backward head tilt).

    Args:
        landmarks: Detected landmarks list
        head_pose_idxs: Dictionary of head pose landmark indices
        image_w: Width of captured frame
        image_h: Height of captured frame

    Returns:
        tuple: (angles_dict, coords_points_dict) or (None, None) on error
    """
    try:
        coords_points = get_landmark_coordinates(landmarks, head_pose_idxs, image_w, image_h)

        # Check if any coordinate is None
        if None in coords_points.values():
            return {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}, None

        left_eye = coords_points["left_eye"]
        right_eye = coords_points["right_eye"]
        nose_tip = coords_points["nose_tip"]
        forehead = coords_points["forehead"]
        chin = coords_points["chin"]
        face_left = coords_points["face_left"]
        face_right = coords_points["face_right"]

        # Calculate Roll (tilt left/right) using eye corners
        # This is accurate and doesn't need improvement
        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # Calculate Yaw (turning left/right) using face width and nose position
        # Improved: Use both horizontal position and distance ratios
        face_center_x = (face_left[0] + face_right[0]) / 2.0
        face_width = distance(face_left, face_right)

        # Calculate nose offset from face center
        nose_offset_x = nose_tip[0] - face_center_x

        if face_width > 0:
            # Normalize offset by face width and calculate angle
            normalized_offset = nose_offset_x / (face_width / 2.0)
            # Clamp to [-1, 1] to avoid arcsin domain errors
            normalized_offset = np.clip(normalized_offset, -1.0, 1.0)
            yaw = np.degrees(np.arcsin(normalized_offset))
        else:
            yaw = 0.0

        # Calculate Pitch (nodding up/down) - ADAPTED FOR OVERHEAD CAMERA (REARVIEW MIRROR POSITION)
        # Camera is positioned above driver (like rearview mirror), looking down
        # This changes the perspective: nose appears lower in the image naturally
        # We need to detect changes from this baseline position

        # Calculate eye center (midpoint between left and right eye)
        eye_center_y = (left_eye[1] + right_eye[1]) / 2.0

        # Calculate face height (forehead to chin)
        face_height = abs(chin[1] - forehead[1])

        # Calculate where nose is positioned vertically in the face
        # In overhead camera view, nose is naturally lower in the image
        nose_position_in_face = (nose_tip[1] - forehead[1]) / face_height if face_height > 0 else 0.5

        # In overhead camera perspective:
        # - Neutral position: nose is typically at 0.35-0.45 of face height from forehead
        # - Head tilts forward: nose moves down more (ratio increases toward 0.5+)
        # - Head tilts backward: nose moves up (ratio decreases toward 0.3-)

        if face_height > 20:
            # Neutral position for overhead camera (nose appears lower naturally)
            # Adjusted for your specific setup: if pitch is 15-25 in neutral,
            # we need to increase the neutral position significantly
            # With scale factor 2.5: 15° ≈ 0.11 deviation, 25° ≈ 0.19 deviation
            # So nose is at ~0.51-0.59 of face height, neutral should be ~0.52-0.54
            neutral_nose_position = 0.52  # Increased to compensate for 15-25° pitch in neutral

            # Calculate deviation from neutral
            position_deviation = nose_position_in_face - neutral_nose_position

            # Convert deviation to pitch angle
            # Scale factor: 0.1 position change ≈ 12-15 degrees of head tilt
            # Use moderate scaling for overhead camera perspective
            pitch = np.degrees(np.arctan(position_deviation * 2.5))

            # Clamp to reasonable range
            pitch = np.clip(pitch, -45.0, 45.0)
        else:
            pitch = 0.0

        angles = {
            "roll": round(roll, 2),
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2)
        }

    except Exception as e:
        # Log error for debugging but return default values
        angles = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        coords_points = None

    return angles, coords_points


class StateTracker:
    """Tracks the state of drowsiness detection metrics."""

    def __init__(self):
        """Initialize state tracker with default values."""
        self.yawn_start_time = time.perf_counter()
        self.head_tilt_start_time = 0.0  # Initialize to 0, will be set when condition starts
        self.yawn_time = 0.0
        self.head_tilt_time = 0.0
        self.color = COLORS["GREEN"]
        self.play_alarm = False

        # PERCLOS tracking: store eye closure states with timestamps
        self.eye_closure_history = []  # List of (timestamp, is_closed) tuples
        self.perclos_window = 30.0  # Window size in seconds (30 seconds for real-time)
        self.perclos_threshold = 0.15  # 15% threshold for drowsiness (typical: 15-20%)
        self.current_perclos = 0.0

    def reset_all(self):
        """Reset all timers and state."""
        self.yawn_start_time = time.perf_counter()
        self.head_tilt_start_time = 0.0  # Initialize to 0, will be set when condition starts
        self.yawn_time = 0.0
        self.head_tilt_time = 0.0
        self.color = COLORS["GREEN"]
        self.play_alarm = False
        self.eye_closure_history.clear()
        self.current_perclos = 0.0

    def update_perclos(self, eyes_closed: bool, current_time: float):
        """Update PERCLOS (Percentage of Eyelid Closure) calculation.

        PERCLOS is the percentage of time eyes are closed during a time window.
        This is a standard metric for drowsiness detection.

        Args:
            eyes_closed: Boolean indicating if eyes are currently closed (EAR < threshold)
            current_time: Current timestamp
        """
        # Add current state to history
        self.eye_closure_history.append((current_time, eyes_closed))

        # Remove old entries outside the time window
        cutoff_time = current_time - self.perclos_window
        self.eye_closure_history = [(t, state) for t, state in self.eye_closure_history
                                     if t >= cutoff_time]

        # Calculate PERCLOS: percentage of time eyes were closed in the window
        if len(self.eye_closure_history) < 2:
            self.current_perclos = 0.0
            return

        # Calculate total time with eyes closed
        total_closed_time = 0.0
        window_start = self.eye_closure_history[0][0]
        window_end = current_time
        window_duration = window_end - window_start

        if window_duration <= 0:
            self.current_perclos = 0.0
            return

        # Calculate closed time by summing intervals where eyes were closed
        prev_time = window_start
        prev_closed = False

        for timestamp, is_closed in self.eye_closure_history:
            if prev_closed:
                total_closed_time += timestamp - prev_time
            prev_time = timestamp
            prev_closed = is_closed

        # Add time from last entry to current time if eyes are currently closed
        if eyes_closed:
            total_closed_time += current_time - prev_time

        # Calculate PERCLOS percentage
        self.current_perclos = (total_closed_time / window_duration) * 100.0

        # Update alarm state based on PERCLOS threshold
        if self.current_perclos >= (self.perclos_threshold * 100.0):
            self.color = COLORS["RED"]
            self.play_alarm = True
        else:
            # Only reset alarm if no other conditions are met
            if not (self.yawn_time > 0 or self.head_tilt_time > 0):
                self.color = COLORS["GREEN"]
                self.play_alarm = False

    def update_yawn(self, condition_met, wait_time):
        """Update yawn state based on condition.

        Args:
            condition_met: Boolean indicating if yawn condition is met
            wait_time: Time threshold before alarm triggers
        """
        if condition_met:
            end_time = time.perf_counter()
            self.yawn_time += end_time - self.yawn_start_time
            self.yawn_start_time = end_time
            self.color = COLORS["RED"]
            if self.yawn_time >= wait_time:
                self.play_alarm = True
        else:
            self.yawn_start_time = time.perf_counter()
            self.yawn_time = 0.0

    def update_head_tilt(self, condition_met, wait_time):
        """Update head tilt state based on condition.

        Args:
            condition_met: Boolean indicating if head tilt condition is met
            wait_time: Time threshold before alarm triggers
        """
        current_time = time.perf_counter()

        if condition_met:
            # If condition is met, calculate elapsed time since condition started
            if self.head_tilt_start_time == 0:
                # Condition just started, initialize start time
                self.head_tilt_start_time = current_time
                self.head_tilt_time = 0.0
            else:
                # Condition continues, accumulate time
                self.head_tilt_time = current_time - self.head_tilt_start_time

            self.color = COLORS["RED"]
            if self.head_tilt_time >= wait_time:
                self.play_alarm = True
        else:
            # Condition not met, reset timer
            self.head_tilt_start_time = 0.0
            self.head_tilt_time = 0.0


class VideoFrameHandler:
    """Main handler for processing video frames and detecting drowsiness."""

    def __init__(self, viaje_id: int = None, use_raspberry_pi_optimization=False):
        """Initialize the video frame handler with MediaPipe and state tracking.

        Args:
            viaje_id: ID of the active trip to send detections to backend
            use_raspberry_pi_optimization: If True, uses optimized MediaPipe config for Raspberry Pi
        """
        self.eye_idxs = LANDMARK_INDICES["eye"]
        self.mouth_idxs = LANDMARK_INDICES["mouth"]
        self.head_pose_idxs = LANDMARK_INDICES["head_pose"]

        self.facemesh_model = get_mediapipe_app(use_raspberry_pi_optimization=use_raspberry_pi_optimization)
        self.state_tracker = StateTracker()

        self.text_positions = TEXT_POSITIONS.copy()
        self.viaje_id = viaje_id
        self.last_reading_time = 0.0
        self.reading_interval = 2.0  # Send reading every 2 seconds
        self.last_alarm_state = False
        self.perclos_window_size = 30.0  # PERCLOS window in seconds
        self.perclos_threshold_pct = 15.0  # PERCLOS threshold percentage

    def reset_perclos(self):
        """Reset PERCLOS value and all detection state to initial values."""
        self.state_tracker.reset_all()
        self.last_alarm_state = False

    def process(self, frame: np.ndarray, thresholds: dict):
        """Process a video frame and detect drowsiness indicators.

        Args:
            frame: Input frame as numpy array (BGR format)
            thresholds: Dictionary containing detection thresholds

        Returns:
            tuple: (processed_frame, play_alarm_boolean)
        """
        import time
        from .api_client import send_reading_async, send_alert_async

        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape
        current_time = time.perf_counter()

        # Calculate text positions based on frame size
        drowsy_time_pos = (10, int(frame_h // 2 * 1.7))
        yawn_time_pos = (10, int(frame_h // 2 * 1.6))
        head_tilt_time_pos = (10, int(frame_h // 2 * 1.5))
        alarm_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate metrics
            ear, eye_coordinates = calculate_avg_ear(
                landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w, frame_h
            )
            mar, mouth_coordinates = calculate_mar(
                landmarks, self.mouth_idxs, frame_w, frame_h
            )
            head_pose_angles, head_pose_coords = calculate_head_pose(
                landmarks, self.head_pose_idxs, frame_w, frame_h
            )

            # Draw landmarks
            frame = self._plot_eye_landmarks(frame, eye_coordinates[0], eye_coordinates[1])
            frame = self._plot_mouth_landmarks(frame, mouth_coordinates)

            # Check drowsiness using PERCLOS (Percentage of Eyelid Closure)
            ear_threshold = thresholds.get("EAR_THRESH", 0.18)
            eyes_closed = ear < ear_threshold

            # Update PERCLOS calculation
            perclos_window = thresholds.get("PERCLOS_WINDOW", self.perclos_window_size)
            perclos_threshold = thresholds.get("PERCLOS_THRESH", self.perclos_threshold_pct) / 100.0

            self.state_tracker.perclos_window = perclos_window
            self.state_tracker.perclos_threshold = perclos_threshold
            self.state_tracker.update_perclos(eyes_closed, current_time)

            # Show alert if PERCLOS exceeds threshold
            alert_text = None
            if self.state_tracker.current_perclos >= (perclos_threshold * 100.0):
                alert_text = "ALERTA!!!"

            # Check yawning (MAR)
            mar_threshold = thresholds.get("MAR_THRESH", 0.6)
            self.state_tracker.update_yawn(mar > mar_threshold, thresholds.get("WAIT_TIME", 1.0))

            if mar > mar_threshold and self.state_tracker.yawn_time >= thresholds.get("WAIT_TIME", 1.0):
                alert_text = "BOSTEZO!!!"

            # Check head tilt - Improved detection with separate thresholds
            tilt_detected = False
            tilt_type = None
            if head_pose_angles:
                roll_thresh = thresholds.get("ROLL_THRESH", 20.0)
                pitch_thresh = thresholds.get("PITCH_THRESH", 12.0)  # Lowered for better forward/backward detection
                yaw_thresh = thresholds.get("YAW_THRESH", 15.0)

                roll_abs = abs(head_pose_angles["roll"])
                pitch_abs = abs(head_pose_angles["pitch"])
                yaw_abs = abs(head_pose_angles["yaw"])

                # Detect specific types of head tilt
                roll_tilt = roll_abs > roll_thresh
                pitch_tilt = pitch_abs > pitch_thresh
                yaw_tilt = yaw_abs > yaw_thresh

                # Determine tilt type for better feedback
                if pitch_tilt:
                    if head_pose_angles["pitch"] > 0:
                        tilt_type = "CABEZA HACIA ADELANTE"
                    else:
                        tilt_type = "CABEZA HACIA ATRÁS"
                elif roll_tilt:
                    if head_pose_angles["roll"] > 0:
                        tilt_type = "CABEZA INCLINADA DERECHA"
                    else:
                        tilt_type = "CABEZA INCLINADA IZQUIERDA"
                elif yaw_tilt:
                    if head_pose_angles["yaw"] > 0:
                        tilt_type = "CABEZA GIRADA DERECHA"
                    else:
                        tilt_type = "CABEZA GIRADA IZQUIERDA"

                tilt_detected = roll_tilt or pitch_tilt or yaw_tilt

                # Use specific wait time for head tilt (longer than yawn to reduce false positives)
                head_tilt_wait_time = thresholds.get("HEAD_TILT_WAIT_TIME", 3.0)
                self.state_tracker.update_head_tilt(tilt_detected, head_tilt_wait_time)

                if tilt_detected and self.state_tracker.head_tilt_time >= head_tilt_wait_time:
                    if tilt_type:
                        alert_text = f"{tilt_type}!!!"
                    else:
                        alert_text = "CABEZA INCLINADA!!!"

            # Update color if no alerts
            # PERCLOS is handled in update_perclos, so we only check yawn and head tilt here
            if not (mar > mar_threshold or tilt_detected):
                if not any([self.state_tracker.yawn_time > 0,
                           self.state_tracker.head_tilt_time > 0]):
                    # PERCLOS alarm is handled separately in update_perclos
                    if self.state_tracker.current_perclos < (perclos_threshold * 100.0):
                        self.state_tracker.color = COLORS["GREEN"]
                        self.state_tracker.play_alarm = False

            # Build metrics dictionary to return
            metrics = {
                "ear": round(ear, 3),
                "mar": round(mar, 3),
                "perclos": round(self.state_tracker.current_perclos, 1),
                "yawn_time": round(self.state_tracker.yawn_time, 2),
                "head_tilt_time": round(self.state_tracker.head_tilt_time, 2),
                "head_pose": head_pose_angles if head_pose_angles else {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "alert_text": alert_text,
                "tilt_type": tilt_type,
                "is_alarm": self.state_tracker.play_alarm,
            }

            # Send periodic readings to backend
            if self.viaje_id and (current_time - self.last_reading_time) >= self.reading_interval:
                reading_data = {
                    "id_viaje": self.viaje_id,
                    "percios": self.state_tracker.current_perclos,  # Using PERCLOS as percios value
                    "conteo_cabeceos": 1 if eyes_closed else 0,  # Current frame eye closure state
                    "conteo_bostezos": 1 if mar > thresholds.get("MAR_THRESH", 0.6) else 0,
                }
                send_reading_async(reading_data)
                self.last_reading_time = current_time

            # Send alert when alarm state changes from False to True
            if self.viaje_id and self.state_tracker.play_alarm and not self.last_alarm_state:
                # Determine alert type based on what triggered it
                alert_type = "SOMNOLENCIA_PERCLOS"
                if self.state_tracker.yawn_time > 0 and self.state_tracker.yawn_time >= thresholds.get("WAIT_TIME", 1.0):
                    alert_type = "SOMNOLENCIA_BOSTEZOS"
                elif tilt_detected and self.state_tracker.head_tilt_time >= thresholds.get("HEAD_TILT_WAIT_TIME", 3.0):
                    alert_type = "SOMNOLENCIA_CABECEOS"

                alert_data = {
                    "id_viaje": self.viaje_id,
                    "tipo_alerta": alert_type,
                    "nivel_somnolencia": "CRITICO" if self.state_tracker.current_perclos > 20 else "ALTO",
                }
                print(f"🚨 Enviando alerta: {alert_type} - PERCLOS: {self.state_tracker.current_perclos:.1f}%")
                send_alert_async(alert_data)

            self.last_alarm_state = self.state_tracker.play_alarm
        else:
            self.state_tracker.reset_all()
            frame = cv2.flip(frame, 1)
            metrics = {
                "ear": 0.0,
                "mar": 0.0,
                "perclos": 0.0,
                "yawn_time": 0.0,
                "head_tilt_time": 0.0,
                "head_pose": {"roll": 0.0, "pitch": 0.0, "yaw": 0.0},
                "alert_text": None,
                "tilt_type": None,
                "is_alarm": False,
            }

        return frame, self.state_tracker.play_alarm, metrics

    def _plot_eye_landmarks(self, frame, left_lm_coordinates, right_lm_coordinates):
        """Plot eye landmarks on the frame."""
        if not frame.flags.writeable:
            frame = frame.copy()
        for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
            if lm_coordinates:
                for coord in lm_coordinates:
                    if coord:
                        cv2.circle(frame, coord, 2, self.state_tracker.color, -1)
        return frame

    def _plot_mouth_landmarks(self, frame, lm_coordinates):
        """Plot mouth landmarks on the frame."""
        if not frame.flags.writeable:
            frame = frame.copy()
        if lm_coordinates:
            for coord in lm_coordinates.values():
                if coord:
                    cv2.circle(frame, coord, 2, self.state_tracker.color, -1)
        frame = cv2.flip(frame, 1)
        return frame

    def _plot_text(self, image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX,
                   fnt_scale=0.8, thickness=2):
        """Plot text on the image."""
        image = cv2.putText(image, text, origin, font, fnt_scale, color, thickness)
        return image

    def _display_metrics(self, frame, ear, mar, head_pose_angles,
                        drowsy_time_pos, yawn_time_pos, head_tilt_time_pos):
        """Display all detection metrics on the frame."""
        ear_txt = f"EAR: {round(ear, 2)}"
        mar_txt = f"MAR: {round(mar, 2)}"

        if head_pose_angles:
            # Enhanced display with better formatting for pitch
            pitch_value = head_pose_angles['pitch']
            pitch_direction = ""
            if abs(pitch_value) > 5:  # Show direction for significant pitch
                if pitch_value > 0:
                    pitch_direction = " (↓ Adelante)"
                else:
                    pitch_direction = " (↑ Atrás)"

            head_pose_txt = (f"Roll: {head_pose_angles['roll']:.1f}° | "
                           f"Pitch: {head_pose_angles['pitch']:.1f}°{pitch_direction} | "
                           f"Yaw: {head_pose_angles['yaw']:.1f}°")
        else:
            head_pose_txt = "Head Pose: N/A"

        # Display PERCLOS instead of drowsy time
        perclos_txt = f"PERCLOS: {self.state_tracker.current_perclos:.1f}%"
        yawn_time_txt = f"BOSTEZO: {round(self.state_tracker.yawn_time, 3)} Secs"
        head_tilt_time_txt = f"INCLINACION: {round(self.state_tracker.head_tilt_time, 3)} Secs"

        self._plot_text(frame, ear_txt, self.text_positions["EAR"], self.state_tracker.color)
        self._plot_text(frame, mar_txt, self.text_positions["MAR"], self.state_tracker.color)
        self._plot_text(frame, head_pose_txt, self.text_positions["HEAD_POSE"], self.state_tracker.color)
        self._plot_text(frame, perclos_txt, drowsy_time_pos, self.state_tracker.color)
        self._plot_text(frame, yawn_time_txt, yawn_time_pos, self.state_tracker.color)
        self._plot_text(frame, head_tilt_time_txt, head_tilt_time_pos, self.state_tracker.color)

