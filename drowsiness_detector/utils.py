"""
Utility functions for drowsiness detection.

Contains helper functions for calculations, coordinate transformations, and distance measurements.
"""

import numpy as np
from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates


def distance(point_1, point_2):
    """Calculate l2-norm (Euclidean distance) between two points.

    Args:
        point_1: Tuple or list of (x, y) coordinates
        point_2: Tuple or list of (x, y) coordinates

    Returns:
        float: Euclidean distance between the two points
    """
    if point_1 is None or point_2 is None:
        return 0.0
    return sum([(i - j) ** 2 for i, j in zip(point_1, point_2)]) ** 0.5


def get_landmark_coordinates(landmarks, indices, frame_width, frame_height):
    """Extract and denormalize landmark coordinates.

    Args:
        landmarks: MediaPipe landmarks list
        indices: List or dict of landmark indices to extract
        frame_width: Width of the frame
        frame_height: Height of the frame

    Returns:
        list or dict: Denormalized coordinates
    """
    if isinstance(indices, dict):
        coords = {}
        for key, idx in indices.items():
            lm = landmarks[idx]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords[key] = coord
        return coords
    else:
        coords = []
        for idx in indices:
            lm = landmarks[idx]
            coord = denormalize_coordinates(lm.x, lm.y, frame_width, frame_height)
            coords.append(coord)
        return coords

