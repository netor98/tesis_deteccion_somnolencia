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
from .config import LANDMARK_INDICES, MEDIAPIPE_CONFIG, COLORS, TEXT_POSITIONS


def get_mediapipe_app(**kwargs):
    """Initialize and return MediaPipe FaceMesh Solution Graph object.

    Args:
        **kwargs: MediaPipe configuration parameters

    Returns:
        FaceMesh: Initialized MediaPipe FaceMesh object
    """
    config = {**MEDIAPIPE_CONFIG, **kwargs}
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

        # Calculate Roll (tilt left/right) using eye corners
        left_eye = coords_points["left_eye"]
        right_eye = coords_points["right_eye"]
        roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))

        # Calculate Yaw (turning left/right) using face width
        face_left = coords_points["face_left"]
        face_right = coords_points["face_right"]
        nose_tip = coords_points["nose_tip"]

        # Calculate distances from nose to left and right face edges
        dist_left = distance(nose_tip, face_left)
        dist_right = distance(nose_tip, face_right)
        face_width = distance(face_left, face_right)

        # Yaw is positive when turning right (more of right side visible)
        # Negative when turning left (more of left side visible)
        if face_width > 0:
            yaw = np.degrees(np.arcsin((dist_right - dist_left) / face_width))
        else:
            yaw = 0.0

        # Calculate Pitch (nodding up/down) using forehead and chin
        forehead = coords_points["forehead"]
        chin = coords_points["chin"]

        # Calculate the angle between forehead-chin line and horizontal
        vertical_dist = chin[1] - forehead[1]
        horizontal_dist = abs(chin[0] - forehead[0])

        if horizontal_dist > 0:
            pitch = np.degrees(np.arctan2(vertical_dist, horizontal_dist)) - 90
        else:
            pitch = 0.0

        angles = {
            "roll": round(roll, 2),
            "pitch": round(pitch, 2),
            "yaw": round(yaw, 2)
        }

    except Exception:
        angles = {"roll": 0.0, "pitch": 0.0, "yaw": 0.0}
        coords_points = None

    return angles, coords_points


class StateTracker:
    """Tracks the state of drowsiness detection metrics."""

    def __init__(self):
        """Initialize state tracker with default values."""
        self.drowsy_start_time = time.perf_counter()
        self.yawn_start_time = time.perf_counter()
        self.head_tilt_start_time = time.perf_counter()
        self.drowsy_time = 0.0
        self.yawn_time = 0.0
        self.head_tilt_time = 0.0
        self.color = COLORS["GREEN"]
        self.play_alarm = False

    def reset_all(self):
        """Reset all timers and state."""
        self.drowsy_start_time = time.perf_counter()
        self.yawn_start_time = time.perf_counter()
        self.head_tilt_start_time = time.perf_counter()
        self.drowsy_time = 0.0
        self.yawn_time = 0.0
        self.head_tilt_time = 0.0
        self.color = COLORS["GREEN"]
        self.play_alarm = False

    def update_drowsy(self, condition_met, wait_time):
        """Update drowsy state based on condition.

        Args:
            condition_met: Boolean indicating if drowsy condition is met
            wait_time: Time threshold before alarm triggers
        """
        if condition_met:
            end_time = time.perf_counter()
            self.drowsy_time += end_time - self.drowsy_start_time
            self.drowsy_start_time = end_time
            self.color = COLORS["RED"]
            if self.drowsy_time >= wait_time:
                self.play_alarm = True
        else:
            self.drowsy_start_time = time.perf_counter()
            self.drowsy_time = 0.0

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
        if condition_met:
            end_time = time.perf_counter()
            self.head_tilt_time += end_time - self.head_tilt_start_time
            self.head_tilt_start_time = end_time
            self.color = COLORS["RED"]
            if self.head_tilt_time >= wait_time:
                self.play_alarm = True
        else:
            self.head_tilt_start_time = time.perf_counter()
            self.head_tilt_time = 0.0


class VideoFrameHandler:
    """Main handler for processing video frames and detecting drowsiness."""

    def __init__(self):
        """Initialize the video frame handler with MediaPipe and state tracking."""
        self.eye_idxs = LANDMARK_INDICES["eye"]
        self.mouth_idxs = LANDMARK_INDICES["mouth"]
        self.head_pose_idxs = LANDMARK_INDICES["head_pose"]

        self.facemesh_model = get_mediapipe_app()
        self.state_tracker = StateTracker()

        self.text_positions = TEXT_POSITIONS.copy()

    def process(self, frame: np.ndarray, thresholds: dict):
        """Process a video frame and detect drowsiness indicators.

        Args:
            frame: Input frame as numpy array (BGR format)
            thresholds: Dictionary containing detection thresholds

        Returns:
            tuple: (processed_frame, play_alarm_boolean)
        """
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

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

            # Check drowsiness (EAR)
            ear_threshold = thresholds.get("EAR_THRESH", 0.18)
            self.state_tracker.update_drowsy(ear < ear_threshold, thresholds.get("WAIT_TIME", 1.0))

            if ear < ear_threshold and self.state_tracker.drowsy_time >= thresholds.get("WAIT_TIME", 1.0):
                self._plot_text(frame, "ALERTA!!!", alarm_pos, self.state_tracker.color)

            # Check yawning (MAR)
            mar_threshold = thresholds.get("MAR_THRESH", 0.6)
            self.state_tracker.update_yawn(mar > mar_threshold, thresholds.get("WAIT_TIME", 1.0))

            if mar > mar_threshold and self.state_tracker.yawn_time >= thresholds.get("WAIT_TIME", 1.0):
                self._plot_text(frame, "BOSTEZO!!!", alarm_pos, self.state_tracker.color)

            # Check head tilt
            tilt_detected = False
            if head_pose_angles:
                roll_thresh = thresholds.get("ROLL_THRESH", 20.0)
                pitch_thresh = thresholds.get("PITCH_THRESH", 15.0)
                yaw_thresh = thresholds.get("YAW_THRESH", 15.0)

                roll_abs = abs(head_pose_angles["roll"])
                pitch_abs = abs(head_pose_angles["pitch"])
                yaw_abs = abs(head_pose_angles["yaw"])

                tilt_detected = (roll_abs > roll_thresh or
                               pitch_abs > pitch_thresh or
                               yaw_abs > yaw_thresh)

                self.state_tracker.update_head_tilt(tilt_detected, thresholds.get("WAIT_TIME", 1.0))

                if tilt_detected and self.state_tracker.head_tilt_time >= thresholds.get("WAIT_TIME", 1.0):
                    self._plot_text(frame, "CABEZA INCLINADA!!!", alarm_pos, self.state_tracker.color)

            # Update color if no alerts
            if not (ear < ear_threshold or mar > mar_threshold or tilt_detected):
                if not any([self.state_tracker.drowsy_time > 0,
                           self.state_tracker.yawn_time > 0,
                           self.state_tracker.head_tilt_time > 0]):
                    self.state_tracker.color = COLORS["GREEN"]
                    self.state_tracker.play_alarm = False

            # Display metrics
            self._display_metrics(frame, ear, mar, head_pose_angles,
                                drowsy_time_pos, yawn_time_pos, head_tilt_time_pos)
        else:
            self.state_tracker.reset_all()
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker.play_alarm

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
            head_pose_txt = (f"Roll: {head_pose_angles['roll']}° | "
                           f"Pitch: {head_pose_angles['pitch']}° | "
                           f"Yaw: {head_pose_angles['yaw']}°")
        else:
            head_pose_txt = "Head Pose: N/A"

        drowsy_time_txt = f"TIEMPO: {round(self.state_tracker.drowsy_time, 3)} Secs"
        yawn_time_txt = f"BOSTEZO: {round(self.state_tracker.yawn_time, 3)} Secs"
        head_tilt_time_txt = f"INCLINACION: {round(self.state_tracker.head_tilt_time, 3)} Secs"

        self._plot_text(frame, ear_txt, self.text_positions["EAR"], self.state_tracker.color)
        self._plot_text(frame, mar_txt, self.text_positions["MAR"], self.state_tracker.color)
        self._plot_text(frame, head_pose_txt, self.text_positions["HEAD_POSE"], self.state_tracker.color)
        self._plot_text(frame, drowsy_time_txt, drowsy_time_pos, self.state_tracker.color)
        self._plot_text(frame, yawn_time_txt, yawn_time_pos, self.state_tracker.color)
        self._plot_text(frame, head_tilt_time_txt, head_tilt_time_pos, self.state_tracker.color)

