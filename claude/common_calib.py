"""
common_calib.py
================
Shared ChArUco board setup, detection, and transform-averaging helpers used
by both capture_realsense.py (live capture) and calibrate_from_captures.py
(offline calibration + point cloud). No pyrealsense2 dependency here on
purpose, so calibrate_from_captures.py can run on a machine with no cameras
attached, purely from saved images.

*** EDIT THE CONFIG SECTION BELOW TO MATCH YOUR ACTUAL PRINTED BOARD ***
"""

import numpy as np
import cv2

# =============================================================================
# CONFIG -- EDIT THIS SECTION FOR YOUR SETUP
# =============================================================================

CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 5
SQUARE_LENGTH_M = 0.04     # <-- ASSUMPTION: 4cm squares. Measure your real board and correct.
MARKER_LENGTH_M = 0.03     # <-- ASSUMPTION: 3cm markers. Measure your real board and correct.
ARUCO_DICT_NAME = "DICT_5X5_100"

MIN_CHARUCO_CORNERS = 6    # minimum corners detected to accept a pose


# =============================================================================
# CHARUCO SETUP (handles both old and new OpenCV aruco APIs)
# =============================================================================

def build_charuco_board():
    aruco_dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

    if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard, "create") is False:
        try:
            board = cv2.aruco.CharucoBoard(
                (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
                SQUARE_LENGTH_M,
                MARKER_LENGTH_M,
                dictionary,
            )
            return board, dictionary, "new"
        except Exception:
            pass

    board = cv2.aruco.CharucoBoard_create(
        CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
        SQUARE_LENGTH_M, MARKER_LENGTH_M,
        dictionary,
    )
    return board, dictionary, "legacy"


def detect_charuco(gray_image, board, dictionary, api_kind):
    """Returns (charuco_corners, charuco_ids) or (None, None) if not enough found.
    Guards against the new-API edge case where corners/ids come back mismatched
    in length (partial/noisy detections)."""
    if api_kind == "new":
        detector_params = cv2.aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(dictionary, detector_params)
        corners, ids, rejected = aruco_detector.detectMarkers(gray_image)
        if ids is None or len(ids) == 0:
            return None, None
        charuco_detector = cv2.aruco.CharucoDetector(board)
        ch_corners, ch_ids, _, _ = charuco_detector.detectBoard(gray_image)
        if ch_corners is None or ch_ids is None:
            return None, None
        if len(ch_corners) != len(ch_ids) or len(ch_corners) < MIN_CHARUCO_CORNERS:
            return None, None
        return ch_corners, ch_ids
    else:
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(gray_image, dictionary, parameters=params)
        if ids is None or len(ids) == 0:
            return None, None
        ret, ch_corners, ch_ids = cv2.aruco.interpolateCornersCharuco(
            corners, ids, gray_image, board
        )
        if ch_corners is None or ch_ids is None or ret < MIN_CHARUCO_CORNERS:
            return None, None
        return ch_corners, ch_ids


def solve_board_pose(ch_corners, ch_ids, board, camera_matrix, dist_coeffs):
    """Returns (rvec, tvec) mapping board-frame points -> this camera's frame, or None."""
    try:
        obj_points, img_points = board.matchImagePoints(ch_corners, ch_ids)
        if obj_points is None or len(obj_points) < 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    except AttributeError:
        ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            ch_corners, ch_ids, board, camera_matrix, dist_coeffs, None, None
        )
    if not ok:
        return None
    return rvec, tvec


# =============================================================================
# TRANSFORM HELPERS
# =============================================================================

def rvec_tvec_to_T(rvec, tvec):
    Rm, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = Rm
    T[:3, 3] = tvec.flatten()
    return T


def invert_T(T):
    Rm = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = Rm.T
    T_inv[:3, 3] = -Rm.T @ t
    return T_inv


def average_transforms(T_list):
    """Average a list of 4x4 transforms: quaternion averaging for rotation,
    simple mean for translation. Imports scipy locally so this module only
    requires scipy when this function is actually called."""
    from scipy.spatial.transform import Rotation as R

    quats = []
    trans = []
    ref_quat = None
    for T in T_list:
        q = R.from_matrix(T[:3, :3]).as_quat()  # x,y,z,w
        if ref_quat is None:
            ref_quat = q
        elif np.dot(q, ref_quat) < 0:
            q = -q  # handle quaternion double-cover sign flip
        quats.append(q)
        trans.append(T[:3, 3])
    mean_quat = np.mean(np.array(quats), axis=0)
    mean_quat /= np.linalg.norm(mean_quat)
    mean_R = R.from_quat(mean_quat).as_matrix()
    mean_t = np.mean(np.array(trans), axis=0)
    T_avg = np.eye(4)
    T_avg[:3, :3] = mean_R
    T_avg[:3, 3] = mean_t
    return T_avg