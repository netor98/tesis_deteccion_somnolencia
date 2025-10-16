import cv2
import time
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def get_mediapipe_app(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
):
    """Initialize and return Mediapipe FaceMesh Solution Graph object"""
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=max_num_faces,
        refine_landmarks=refine_landmarks,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    return face_mesh


def distance(point_1, point_2):
    """Calculate l2-norm between two points"""
    dist = sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5
    return dist


def get_ear(landmarks, refer_idxs, frame_width, frame_height):
    """
    Calculate Eye Aspect Ratio for one eye.

    Args:
        landmarks: (list) Detected landmarks list
        refer_idxs: (list) Index positions of the chosen landmarks
                            in order P1, P2, P3, P4, P5, P6
        frame_width: (int) Width of captured frame
        frame_height: (int) Height of captured frame

    Returns:
        ear: (float) Eye aspect ratio
    """
    try:
        # Compute the euclidean distance between the horizontal
        coords_points = []
        for i in refer_idxs:
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords_points.append(coord)

        # Eye landmark (x, y)-coordinates
        P2_P6 = distance(coords_points[1], coords_points[5])
        P3_P5 = distance(coords_points[2], coords_points[4])
        P1_P4 = distance(coords_points[0], coords_points[3])

        # Compute the eye aspect ratio
        ear = (P2_P6 + P3_P5) / (2.0 * P1_P4)

    except:
        ear = 0.0
        coords_points = None

    return ear, coords_points


def calculate_avg_ear(landmarks, left_eye_idxs, right_eye_idxs, image_w, image_h):
    # Calculate Eye aspect ratio

    left_ear, left_lm_coordinates = get_ear(landmarks, left_eye_idxs, image_w, image_h)
    right_ear, right_lm_coordinates = get_ear(landmarks, right_eye_idxs, image_w, image_h)
    Avg_EAR = (left_ear + right_ear) / 2.0

    return Avg_EAR, (left_lm_coordinates, right_lm_coordinates)


def calculate_mar(landmarks, mouth_idxs, image_w, image_h):
    """
    Calculate Mouth Aspect Ratio.

    Args:
        landmarks: (list) Detected landmarks list
        mouth_idxs: (dict) Index positions of the chosen landmarks for the mouth
        image_w: (int) Width of captured frame
        image_h: (int) Height of captured frame

    Returns:
        mar: (float) Mouth aspect ratio
        coords_points: (list) Coordinates of the mouth landmarks
    """
    try:
        # Get the coordinates of the 8 mouth landmarks
        coords_points = []
        for i in mouth_idxs.values():
            lm = landmarks[i]
            coord = denormalize_coordinates(lm.x, lm.y, image_w, image_h)
            coords_points.append(coord)

        # Calculate the distances
        ab_dist = distance(coords_points[0], coords_points[1])
        cd_dist = distance(coords_points[2], coords_points[3])
        ef_dist = distance(coords_points[4], coords_points[5])
        gh_dist = distance(coords_points[6], coords_points[7])

        # Calculate the Mouth Aspect Ratio (MAR)
        mar = (cd_dist + ef_dist + gh_dist) / (3.0 * ab_dist)

    except:
        mar = 0.0
        coords_points = None

    return mar, coords_points


def plot_eye_landmarks(frame, left_lm_coordinates, right_lm_coordinates, color):
    if not frame.flags.writeable:
        frame = frame.copy()
    for lm_coordinates in [left_lm_coordinates, right_lm_coordinates]:
        if lm_coordinates:
            for coord in lm_coordinates:
                cv2.circle(frame, coord, 2, color, -1)

    return frame


def plot_mouth_landmarks(frame, lm_coordinates, color):
    if not frame.flags.writeable:
        frame = frame.copy()
    if lm_coordinates:
        for coord in lm_coordinates:
            cv2.circle(frame, coord, 2, color, -1)

    frame = cv2.flip(frame, 1)
    return frame


def plot_text(image, text, origin, color, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, thickness=2):
    image = cv2.putText(image, text, origin, font, fntScale, color, thickness)
    return image


class VideoFrameHandler:
    def __init__(self):
        """
        Initialize the necessary constants, mediapipe app
        and tracker variables
        """
        # Left and right eye chosen landmarks. LANDMARKS EYES
        self.eye_idxs = {
            "left": [362, 385, 387, 263, 373, 380],
            "right": [33, 160, 158, 133, 153, 144],
        }

        self.mouth_idxs = {
        "A": 61,   # left mouth corner
    "B": 291,  # right mouth corner
    "C": 78,   # upper left lip
    "D": 308,  # upper right lip
    "E": 13,   # upper center lip
    "F": 14,   # lower center lip
    "G": 82,   # lower left lip
    "H": 312,  # lower right lip
}

        # Used for coloring landmark points.
        # Its value depends on the current EAR value.
        self.RED = (0, 0, 255)  # BGR
        self.GREEN = (0, 255, 0)  # BGR

        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = get_mediapipe_app()

        # For tracking counters and sharing states in and out of callbacks.
        self.state_tracker = {
            "drowsy_start_time": time.perf_counter(),
            "yawn_start_time": time.perf_counter(),
            "DROWSY_TIME": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "YAWN_TIME": 0.0,  # Holds the amount of time passed with MAR > MAR_THRESH
            "COLOR": self.GREEN,
            "play_alarm": False,
        }

        self.EAR_txt_pos = (10, 30)
        self.MAR_txt_pos = (10, 60)

    def process(self, frame: np.array, thresholds: dict):
        """
        This function is used to implement our Drowsy detection algorithm

        Args:
            frame: (np.array) Input frame matrix.
            thresholds: (dict) Contains the two threshold values
                               WAIT_TIME and EAR_THRESH.

        Returns:
            The processed frame and a boolean flag to
            indicate if the alarm should be played or not.
        """

        # To improve performance,
        # mark the frame as not writeable to pass by reference.
        frame.flags.writeable = False
        frame_h, frame_w, _ = frame.shape

        DROWSY_TIME_txt_pos = (10, int(frame_h // 2 * 1.7))
        YAWN_TIME_txt_pos = (10, int(frame_h // 2 * 1.6))
        ALM_txt_pos = (10, int(frame_h // 2 * 1.85))

        results = self.facemesh_model.process(frame)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            EAR, coordinates = calculate_avg_ear(landmarks, self.eye_idxs["left"], self.eye_idxs["right"], frame_w,
                                                 frame_h)
            MAR, mouth_coordinates = calculate_mar(landmarks, self.mouth_idxs, frame_w, frame_h)
            frame = plot_eye_landmarks(frame, coordinates[0], coordinates[1], self.state_tracker["COLOR"])
            frame = plot_mouth_landmarks(frame, mouth_coordinates, self.state_tracker["COLOR"])

            if EAR < thresholds["EAR_THRESH"]:
                # Increase DROWSY_TIME to track the time period with EAR less than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["DROWSY_TIME"] += end_time - self.state_tracker["drowsy_start_time"]
                self.state_tracker["drowsy_start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["DROWSY_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "ALERTA!!!", ALM_txt_pos, self.state_tracker["COLOR"])

            else:
                self.state_tracker["drowsy_start_time"] = time.perf_counter()
                self.state_tracker["DROWSY_TIME"] = 0.0
                self.state_tracker["COLOR"] = self.GREEN
                self.state_tracker["play_alarm"] = False

            if MAR > thresholds["MAR_THRESH"]:
                # Increase YAWN_TIME to track the time period with MAR greater than the threshold
                # and reset the start_time for the next iteration.
                end_time = time.perf_counter()

                self.state_tracker["YAWN_TIME"] += end_time - self.state_tracker["yawn_start_time"]
                self.state_tracker["yawn_start_time"] = end_time
                self.state_tracker["COLOR"] = self.RED

                if self.state_tracker["YAWN_TIME"] >= thresholds["WAIT_TIME"]:
                    self.state_tracker["play_alarm"] = True
                    plot_text(frame, "BOSTEZO!!!", ALM_txt_pos, self.state_tracker["COLOR"])
            else:
                self.state_tracker["yawn_start_time"] = time.perf_counter()
                self.state_tracker["YAWN_TIME"] = 0.0
                self.state_tracker["play_alarm"] = False

            EAR_txt = f"EAR: {round(EAR, 2)}"
            MAR_txt = f"MAR: {round(MAR, 2)}"
            DROWSY_TIME_txt = f"TIEMPO: {round(self.state_tracker['DROWSY_TIME'], 3)} Secs"
            YAWN_TIME_txt = f"BOSTEZO: {round(self.state_tracker['YAWN_TIME'], 3)} Secs"
            plot_text(frame, EAR_txt, self.EAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, MAR_txt, self.MAR_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, DROWSY_TIME_txt, DROWSY_TIME_txt_pos, self.state_tracker["COLOR"])
            plot_text(frame, YAWN_TIME_txt, YAWN_TIME_txt_pos, self.state_tracker["COLOR"])

        else:
            self.state_tracker["drowsy_start_time"] = time.perf_counter()
            self.state_tracker["yawn_start_time"] = time.perf_counter()
            self.state_tracker["DROWSY_TIME"] = 0.0
            self.state_tracker["YAWN_TIME"] = 0.0
            self.state_tracker["COLOR"] = self.GREEN
            self.state_tracker["play_alarm"] = False

            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)

        return frame, self.state_tracker["play_alarm"]
