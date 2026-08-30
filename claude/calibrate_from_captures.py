"""
calibrate_from_captures.py
============================
Fully offline. Reads the folder structure written by capture_realsense.py
and:

  --mode calibrate   Uses all pose_*/ folders (board captures) to compute
                      each camera's extrinsic transform relative to a
                      reference camera. Saves extrinsics.json.

  --mode pointcloud   Uses all scene_*/ folders (or pose_*/ if no scene
                      folders exist) plus a saved extrinsics.json to build
                      a merged point cloud per scene folder, save it as a
                      .ply, and optionally display it.

  --mode full         Does calibrate, then pointcloud, in one run.

No pyrealsense2 or camera hardware required -- this only touches files
under --captures_dir. You can copy a captures/ folder to another machine
and run this there.

USAGE:
    python calibrate_from_captures.py --mode calibrate
    python calibrate_from_captures.py --mode pointcloud --show
    python calibrate_from_captures.py --mode full --show
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    import open3d as o3d
except ImportError:
    o3d = None  # only required for --mode pointcloud/full

from common_calib import (
    build_charuco_board, detect_charuco, solve_board_pose,
    rvec_tvec_to_T, invert_T, average_transforms,
)


def load_intrinsics(captures_dir):
    intr_dir = captures_dir / "intrinsics"
    if not intr_dir.exists():
        print(f"ERROR: {intr_dir} not found. Run capture_realsense.py first.")
        sys.exit(1)
    intrinsics = {}
    for f in sorted(intr_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)
        intrinsics[data["serial"]] = data
    if not intrinsics:
        print(f"ERROR: no intrinsics json files found in {intr_dir}")
        sys.exit(1)
    return intrinsics


def camera_matrix_from_intrinsics(intr):
    return np.array([
        [intr["fx"], 0, intr["ppx"]],
        [0, intr["fy"], intr["ppy"]],
        [0, 0, 1],
    ])


def find_pose_folders(captures_dir, prefix="pose"):
    return sorted(captures_dir.glob(f"{prefix}_*"))


# =============================================================================
# CALIBRATE MODE
# =============================================================================

def run_calibrate(captures_dir, intrinsics, reference_serial=None):
    board, dictionary, api_kind = build_charuco_board()

    serials = list(intrinsics.keys())
    if reference_serial is None:
        reference_serial = serials[0]
    if reference_serial not in serials:
        print(f"ERROR: --reference-serial {reference_serial} not found in intrinsics "
              f"({serials})")
        sys.exit(1)

    other_serials = [s for s in serials if s != reference_serial]
    print(f"Reference camera: {reference_serial}")
    print(f"Other cameras: {other_serials}")

    pose_folders = find_pose_folders(captures_dir, "pose")
    if not pose_folders:
        print(f"ERROR: no pose_* folders found in {captures_dir}. "
              f"Capture some with capture_realsense.py first (press 'c').")
        sys.exit(1)
    print(f"Found {len(pose_folders)} pose folder(s)")

    relative_samples = {s: [] for s in other_serials}
    used_count = 0

    for folder in pose_folders:
        poses = {}
        ok = True
        for serial in serials:
            color_path = folder / f"{serial}_color.png"
            if not color_path.exists():
                ok = False
                break
            color = cv2.imread(str(color_path))
            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
            ch_corners, ch_ids = detect_charuco(gray, board, dictionary, api_kind)
            if ch_corners is None:
                ok = False
                break
            cam_matrix = camera_matrix_from_intrinsics(intrinsics[serial])
            dist = np.array(intrinsics[serial]["dist_coeffs"], dtype=np.float64)
            result = solve_board_pose(ch_corners, ch_ids, board, cam_matrix, dist)
            if result is None:
                ok = False
                break
            poses[serial] = rvec_tvec_to_T(*result)  # T_board_to_cam

        if not ok:
            print(f"  {folder.name}: skipped (board not detected/solved in all cameras)")
            continue

        T_board_to_ref = poses[reference_serial]
        for serial in other_serials:
            T_board_to_cam = poses[serial]
            T_cam_to_board = invert_T(T_board_to_cam)
            # p_ref = T_board_to_ref @ T_cam_to_board @ p_cam
            T_cam_to_ref = T_board_to_ref @ T_cam_to_board
            relative_samples[serial].append(T_cam_to_ref)

        used_count += 1
        print(f"  {folder.name}: OK")

    if used_count == 0:
        print("No usable pose folders -- calibration failed.")
        return None

    print(f"\nUsed {used_count}/{len(pose_folders)} pose folders")

    extrinsics = {reference_serial: np.eye(4).tolist()}
    for serial in other_serials:
        if not relative_samples[serial]:
            print(f"WARNING: no valid samples for {serial}, skipping.")
            continue
        T_avg = average_transforms(relative_samples[serial])
        extrinsics[serial] = T_avg.tolist()
        translations = np.array([T[:3, 3] for T in relative_samples[serial]])
        spread_mm = np.std(translations, axis=0) * 1000
        print(f"{serial}: {len(relative_samples[serial])} samples, "
              f"translation std dev [mm]: {spread_mm.round(2)} "
              f"(>5mm suggests wrong board dimensions or shaky captures)")

    return extrinsics


# =============================================================================
# POINT CLOUD MODE
# =============================================================================

def run_pointcloud(captures_dir, intrinsics, extrinsics, show=False, voxel_size=0.005):
    if o3d is None:
        print("ERROR: open3d not installed. pip install open3d")
        sys.exit(1)

    T_by_serial = {serial: np.array(T) for serial, T in extrinsics.items()}

    scene_folders = find_pose_folders(captures_dir, "scene")
    used_prefix = "scene"
    if not scene_folders:
        print("No scene_* folders found -- falling back to pose_* folders for point clouds.")
        scene_folders = find_pose_folders(captures_dir, "pose")
        used_prefix = "pose"
    if not scene_folders:
        print(f"ERROR: no {used_prefix}_* folders found in {captures_dir} at all.")
        sys.exit(1)

    out_dir = captures_dir / "pointclouds"
    out_dir.mkdir(exist_ok=True)

    for folder in scene_folders:
        merged_points = []
        merged_colors = []

        for serial, intr in intrinsics.items():
            if serial not in T_by_serial:
                continue
            color_path = folder / f"{serial}_color.png"
            depth_path = folder / f"{serial}_depth.png"
            if not color_path.exists() or not depth_path.exists():
                continue

            color = cv2.imread(str(color_path))
            depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)  # uint16, raw depth units
            color_rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

            o3d_color = o3d.geometry.Image(color_rgb)
            o3d_depth = o3d.geometry.Image(depth)

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d_color, o3d_depth,
                depth_scale=1.0 / intr["depth_scale"],
                depth_trunc=3.0,
                convert_rgb_to_intensity=False,
            )
            intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
                intr["width"], intr["height"], intr["fx"], intr["fy"], intr["ppx"], intr["ppy"]
            )
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic_o3d)
            pcd.transform(T_by_serial[serial])
            merged_points.append(np.asarray(pcd.points))
            merged_colors.append(np.asarray(pcd.colors))

        if not merged_points:
            print(f"{folder.name}: no usable camera data, skipping")
            continue

        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(merged_points))
        merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(merged_colors))
        merged_pcd = merged_pcd.voxel_down_sample(voxel_size)

        out_path = out_dir / f"{folder.name}.ply"
        o3d.io.write_point_cloud(str(out_path), merged_pcd)
        print(f"{folder.name}: {len(merged_pcd.points)} points -> {out_path}")

        if show:
            o3d.visualization.draw_geometries([merged_pcd], window_name=folder.name)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captures_dir", default="captures")
    parser.add_argument("--mode", choices=["calibrate", "pointcloud", "full"], default="full")
    parser.add_argument("--extrinsics", default=None,
                         help="Defaults to <captures_dir>/extrinsics.json")
    parser.add_argument("--reference-serial", default=None,
                         help="Which camera serial becomes the world origin. "
                              "Defaults to the first one found in intrinsics/.")
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--show", action="store_true",
                         help="Open an Open3D window per scene point cloud")
    args = parser.parse_args()

    captures_dir = Path(args.captures_dir)
    if not captures_dir.exists():
        print(f"ERROR: {captures_dir} does not exist.")
        sys.exit(1)

    extrinsics_path = Path(args.extrinsics) if args.extrinsics else captures_dir / "extrinsics.json"
    intrinsics = load_intrinsics(captures_dir)

    extrinsics = None
    if args.mode in ("calibrate", "full"):
        extrinsics = run_calibrate(captures_dir, intrinsics, args.reference_serial)
        if extrinsics is None:
            print("Calibration failed.")
            sys.exit(1)
        with open(extrinsics_path, "w") as f:
            json.dump(extrinsics, f, indent=2)
        print(f"\nSaved extrinsics to {extrinsics_path}")

    if args.mode == "pointcloud":
        if not extrinsics_path.exists():
            print(f"ERROR: {extrinsics_path} not found. Run --mode calibrate first.")
            sys.exit(1)
        with open(extrinsics_path) as f:
            extrinsics = json.load(f)

    if args.mode in ("pointcloud", "full"):
        run_pointcloud(captures_dir, intrinsics, extrinsics, show=args.show, voxel_size=args.voxel_size)


if __name__ == "__main__":
    main()