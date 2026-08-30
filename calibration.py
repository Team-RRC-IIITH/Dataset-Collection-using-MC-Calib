"""
Multi-RealSense (3x D455) ChArUco extrinsic calibration + merged point cloud
==============================================================================

WHAT THIS DOES
---------------
1. CALIBRATE mode: opens all 3 D455 cameras live, shows ChArUco detection
   overlays, lets you capture synchronized "poses" (press 'c') where the
   board is visible to all 3 cameras. After enough captures, it computes
   the rigid transform of each camera relative to a chosen reference
   camera (cam0), using the board as a per-frame common reference (the
   board itself does NOT need to stay static between captures -- move it
   around to get better angular coverage, that's actually recommended).
   Saves the result to extrinsics.json.

2. POINTCLOUD mode: opens all 3 cameras live again, builds an Open3D point
   cloud per camera from color+depth, transforms each into the shared
   "world" frame (= cam0's frame) using the saved extrinsics, merges them
   into one point cloud, and shows it live. Press 's' to save a .ply
   snapshot, 'q' to quit.

Default mode is "full": calibrate, then immediately go to point cloud view.

ASSUMPTIONS I MADE (CHANGE THESE IF WRONG) -- see CONFIG section below:
  - Board is 5x5 squares, DICT_5X5_100 dictionary (since you said "5x5
    charuco markers")
  - SQUARE_LENGTH_M = 0.04 (4 cm) and MARKER_LENGTH_M = 0.03 (3 cm)
    *** MEASURE YOUR ACTUAL PRINTED BOARD WITH CALIPERS AND FIX THESE ***
    Wrong physical measurements here will silently scale your whole
    calibration and point cloud. This is the single most common source
    of error in ChArUco calibration.
  - Camera serials / role assignment taken from your product box photo.
  - Reference camera (world origin) = the first serial in CAMERA_SERIALS.

INSTALL (on your machine, not this sandbox):
    pip install pyrealsense2 opencv-contrib-python open3d numpy scipy

USAGE:
    python multi_realsense_calibration.py --mode full
    python multi_realsense_calibration.py --mode calibrate
    python multi_realsense_calibration.py --mode pointcloud --extrinsics extrinsics.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed. pip install pyrealsense2")
    sys.exit(1)

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d not installed. pip install open3d")
    sys.exit(1)

from scipy.spatial.transform import Rotation as R


# =============================================================================
# CONFIG -- EDIT THIS SECTION FOR YOUR SETUP
# =============================================================================

# Serials read from your product box photo. Order matters: index 0 becomes
# the reference/world-frame camera. Reorder if you want a different one as
# the reference (e.g. the most central camera is usually the best choice).
# CAMERA_SERIALS = [
#     "234322307090",  # cam0 -- becomes world origin
#     "234322306310",  # cam1
#     "234222302215",  # cam2
# ]
CAMERA_SERIALS = [
    "234322307090",
    "203522252036",
]

# --- ChArUco board geometry --- MEASURE YOUR ACTUAL BOARD AND FIX THESE ---
CHARUCO_SQUARES_X = 5
CHARUCO_SQUARES_Y = 5
SQUARE_LENGTH_M = 0.04     # <-- ASSUMPTION: 4cm squares. Measure and correct.
MARKER_LENGTH_M = 0.03     # <-- ASSUMPTION: 3cm markers. Measure and correct.
ARUCO_DICT_NAME = "DICT_5X5_100"

# Stream settings
STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 30
DEPTH_TRUNC_M = 3.0        # ignore depth beyond this range (meters)
MIN_CHARUCO_CORNERS = 6    # minimum corners detected to accept a pose

DEFAULT_EXTRINSICS_PATH = "extrinsics.json"


# =============================================================================
# CHARUCO SETUP (handles both old and new OpenCV aruco APIs)
# =============================================================================

def build_charuco_board():
    aruco_dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_id)

    if hasattr(cv2.aruco, "CharucoBoard") and hasattr(cv2.aruco.CharucoBoard, "create") is False:
        # New API (OpenCV >= 4.7)
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

    # Legacy API fallback
    board = cv2.aruco.CharucoBoard_create(
        CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y,
        SQUARE_LENGTH_M, MARKER_LENGTH_M,
        dictionary,
    )
    return board, dictionary, "legacy"


def detect_charuco(gray_image, board, dictionary, api_kind):
    """Returns (charuco_corners, charuco_ids) or (None, None) if not enough found."""
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
        if ch_corners is None or ret < MIN_CHARUCO_CORNERS:
            return None, None
        return ch_corners, ch_ids


def solve_board_pose(ch_corners, ch_ids, board, camera_matrix, dist_coeffs):
    """Returns (rvec, tvec) mapping board-frame points -> this camera's frame, or None."""
    try:
        # New API: board.matchImagePoints
        obj_points, img_points = board.matchImagePoints(ch_corners, ch_ids)
        if obj_points is None or len(obj_points) < 4:
            return None
        ok, rvec, tvec = cv2.solvePnP(obj_points, img_points, camera_matrix, dist_coeffs)
    except AttributeError:
        # Legacy API
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
    simple mean for translation."""
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


# =============================================================================
# REALSENSE CAMERA WRAPPER
# =============================================================================

class RealSenseCam:
    def __init__(self, serial):
        self.serial = serial
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, STREAM_WIDTH, STREAM_HEIGHT, rs.format.bgr8, STREAM_FPS)
        cfg.enable_stream(rs.stream.depth, STREAM_WIDTH, STREAM_HEIGHT, rs.format.z16, STREAM_FPS)
        profile = self.pipeline.start(cfg)

        self.align = rs.align(rs.stream.color)

        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.camera_matrix = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1],
        ])
        self.dist_coeffs = np.array(intr.coeffs, dtype=np.float64)
        self.width = intr.width
        self.height = intr.height

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return color_image, depth_image

    def o3d_intrinsic(self):
        return o3d.camera.PinholeCameraIntrinsic(
            self.width, self.height,
            self.camera_matrix[0, 0], self.camera_matrix[1, 1],
            self.camera_matrix[0, 2], self.camera_matrix[1, 2],
        )

    def stop(self):
        self.pipeline.stop()


# =============================================================================
# CALIBRATION MODE
# =============================================================================

def run_calibration(cams, board, dictionary, api_kind, min_captures=8):
    print("\n=== CALIBRATION MODE ===")
    print("Move the ChArUco board so it's visible to ALL 3 cameras at once.")
    print("Press 'c' to capture a pose when all 3 show green corner overlays.")
    print(f"Capture at least {min_captures} poses at varied angles/positions.")
    print("Press 'q' when done to compute calibration.\n")

    n_cams = len(cams)
    # relative_samples[i] holds list of T_cam_i_to_cam0 (i = 1..n-1)
    relative_samples = {i: [] for i in range(1, n_cams)}
    capture_count = 0

    while True:
        color_imgs = []
        gray_imgs = []
        detections = []  # (ch_corners, ch_ids) per cam, or (None, None)

        for cam in cams:
            color, _ = cam.get_frames()
            if color is None:
                color = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids = detect_charuco(gray, board, dictionary, api_kind)
            vis = color.copy()
            if ch_corners is not None:
                cv2.aruco.drawDetectedCornersCharuco(vis, ch_corners, ch_ids, (0, 255, 0))
            color_imgs.append(vis)
            gray_imgs.append(gray)
            detections.append((ch_corners, ch_ids))

        all_detected = all(d[0] is not None for d in detections)
        status_color = (0, 255, 0) if all_detected else (0, 0, 255)
        status_text = "ALL 3 DETECTED - press 'c' to capture" if all_detected else "waiting for all 3 cameras..."

        display_imgs = []
        for i, img in enumerate(color_imgs):
            small = cv2.resize(img, (STREAM_WIDTH // 2, STREAM_HEIGHT // 2))
            cv2.putText(small, f"cam{i} ({cams[i].serial})", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            display_imgs.append(small)
        combined = np.hstack(display_imgs)
        cv2.putText(combined, status_text, (10, combined.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(combined, f"Captures: {capture_count} (need >= {min_captures})",
                    (10, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Multi-camera ChArUco calibration", combined)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            if capture_count < min_captures:
                print(f"Only {capture_count} captures -- recommend at least {min_captures}. "
                      f"Press 'q' again to proceed anyway, or keep capturing.")
                key2 = cv2.waitKey(0) & 0xFF
                if key2 != ord('q'):
                    continue
            break
        elif key == ord('c') and all_detected:
            poses = []
            ok = True
            for i, cam in enumerate(cams):
                ch_corners, ch_ids = detections[i]
                result = solve_board_pose(ch_corners, ch_ids, board, cam.camera_matrix, cam.dist_coeffs)
                if result is None:
                    ok = False
                    break
                poses.append(rvec_tvec_to_T(*result))  # T_board_to_cam_i
            if not ok:
                print("solvePnP failed on one camera, skipping this capture.")
                continue

            T_board_to_cam0 = poses[0]
            T_cam0_to_board = invert_T(T_board_to_cam0)
            for i in range(1, n_cams):
                T_board_to_cam_i = poses[i]
                T_cam_i_to_board = invert_T(T_board_to_cam_i)
                # p_cam0 = T_board_to_cam0 @ T_cam_i_to_board @ p_cam_i
                T_cam_i_to_cam0 = T_board_to_cam0 @ T_cam_i_to_board
                relative_samples[i].append(T_cam_i_to_cam0)

            capture_count += 1
            print(f"Captured pose #{capture_count}")

    cv2.destroyAllWindows()

    if capture_count == 0:
        print("No captures made. Aborting calibration.")
        return None

    extrinsics = {cams[0].serial: np.eye(4).tolist()}
    for i in range(1, n_cams):
        if len(relative_samples[i]) == 0:
            print(f"WARNING: no valid samples for cam{i} ({cams[i].serial}). Skipping.")
            continue
        T_avg = average_transforms(relative_samples[i])
        extrinsics[cams[i].serial] = T_avg.tolist()
        # Report spread as a sanity check
        translations = np.array([T[:3, 3] for T in relative_samples[i]])
        spread_mm = np.std(translations, axis=0) * 1000
        print(f"cam{i} ({cams[i].serial}): {len(relative_samples[i])} samples, "
              f"translation std dev [mm]: {spread_mm.round(2)} "
              f"(large values e.g. >5mm suggest board size is wrong, or shaky captures)")

    return extrinsics


# =============================================================================
# POINT CLOUD MODE
# =============================================================================

def run_pointcloud(cams, extrinsics):
    print("\n=== POINT CLOUD MODE ===")
    print("Press 's' in the OpenCV window to save a snapshot .ply, 'q' to quit.\n")

    T_by_serial = {serial: np.array(T) for serial, T in extrinsics.items()}
    for cam in cams:
        if cam.serial not in T_by_serial:
            print(f"WARNING: no extrinsics for {cam.serial}, skipping this camera in point cloud.")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Merged point cloud", width=1024, height=768)
    merged_pcd = o3d.geometry.PointCloud()
    added = False

    frame_idx = 0
    try:
        while True:
            all_points = []
            all_colors = []

            for cam in cams:
                if cam.serial not in T_by_serial:
                    continue
                color, depth = cam.get_frames()
                if color is None or depth is None:
                    continue

                color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
                o3d_color = o3d.geometry.Image(color_rgb)
                o3d_depth = o3d.geometry.Image(depth)

                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    o3d_color, o3d_depth,
                    depth_scale=1.0 / cam.depth_scale,
                    depth_trunc=DEPTH_TRUNC_M,
                    convert_rgb_to_intensity=False,
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, cam.o3d_intrinsic())
                pcd.transform(T_by_serial[cam.serial])
                all_points.append(np.asarray(pcd.points))
                all_colors.append(np.asarray(pcd.colors))

            if all_points:
                pts = np.vstack(all_points)
                cols = np.vstack(all_colors)
                merged_pcd.points = o3d.utility.Vector3dVector(pts)
                merged_pcd.colors = o3d.utility.Vector3dVector(cols)
                merged_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)

                if not added:
                    vis.add_geometry(merged_pcd)
                    added = True
                else:
                    vis.update_geometry(merged_pcd)

            vis.poll_events()
            vis.update_renderer()

            # small cv2 window purely to capture keypresses reliably
            key_img = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(key_img, "Click here: 's'=save  'q'=quit", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("controls", key_img)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fname = f"merged_pointcloud_{int(time.time())}.ply"
                o3d.io.write_point_cloud(fname, merged_pcd)
                print(f"Saved {fname} ({len(merged_pcd.points)} points)")

            frame_idx += 1
    finally:
        vis.destroy_window()
        cv2.destroyAllWindows()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["calibrate", "pointcloud", "full"], default="full")
    parser.add_argument("--extrinsics", default=DEFAULT_EXTRINSICS_PATH)
    parser.add_argument("--min-captures", type=int, default=8)
    args = parser.parse_args()

    print("Connecting to cameras:", CAMERA_SERIALS)
    cams = [RealSenseCam(s) for s in CAMERA_SERIALS]

    try:
        extrinsics = None

        if args.mode in ("calibrate", "full"):
            board, dictionary, api_kind = build_charuco_board()
            extrinsics = run_calibration(cams, board, dictionary, api_kind, args.min_captures)
            if extrinsics is None:
                print("Calibration failed / aborted.")
                return
            with open(args.extrinsics, "w") as f:
                json.dump(extrinsics, f, indent=2)
            print(f"Saved extrinsics to {args.extrinsics}")

        if args.mode == "pointcloud":
            with open(args.extrinsics) as f:
                extrinsics = json.load(f)

        if args.mode in ("pointcloud", "full"):
            run_pointcloud(cams, extrinsics)

    finally:
        for cam in cams:
            cam.stop()


if __name__ == "__main__":
    main()