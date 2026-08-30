"""
merge_pointcloud.py
=====================
Loads camera intrinsics AND extrinsics directly from an MC-Calib
`calibrated_cameras_data.yml` (or your `realsense_3cam_calibrated.yml`) using
OpenCV's FileStorage -- no hand-copied matrix values, which is what caused
the previous misalignment (a single-digit typo in a manually copied
translation component).

Works for however many cameras are in the yml (nb_camera), not hardcoded to 2.

TRANSFORM DIRECTION: MC-Calib's camera_pose_matrix follows the standard
OpenCV extrinsic convention (world/reference -> camera), matching
solvePnP/stereoCalibrate. So to bring a point cloud captured in camera i's
frame into the reference camera's frame, we apply inv(camera_pose_matrix_i).
Camera 0 (the reference) is always identity, so it needs no transform.

If your merged cloud still looks flipped/rotated wrongly after fixing the
typo, that's the one assumption in this script worth challenging -- see
the --invert flag below to quickly test the opposite convention.

USAGE:
    python merge_pointcloud.py \
        --calib realsense_3cam_calibrated.yml \
        --images cam0_color.png,cam0_depth.png cam1_color.png,cam1_depth.png \
        --out merged_scene.ply --show

    # if the cloud looks wrong, try the opposite transform convention:
    python merge_pointcloud.py ... --invert

    # add ICP refinement to clean up residual mm-level misalignment
    # left over from reprojection error:
    python merge_pointcloud.py ... --icp
"""

import argparse
import os
import sys

import numpy as np
import cv2
import open3d as o3d


def load_calibration(yml_path):
    """Returns list of dicts: [{camera_matrix, dist_coeffs, pose_matrix, width, height}, ...]
    ordered camera_0, camera_1, ..."""
    fs = cv2.FileStorage(yml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        print(f"ERROR: could not open {yml_path}")
        sys.exit(1)

    nb_camera_node = fs.getNode("nb_camera")
    nb_camera = int(nb_camera_node.real()) if not nb_camera_node.empty() else None

    cams = []
    i = 0
    while True:
        node = fs.getNode(f"camera_{i}")
        if node.empty():
            break
        camera_matrix = node.getNode("camera_matrix").mat()
        dist_coeffs = node.getNode("distortion_vector").mat()
        pose_matrix = node.getNode("camera_pose_matrix").mat()
        width = int(node.getNode("img_width").real())
        height = int(node.getNode("img_height").real())
        cams.append({
            "camera_matrix": camera_matrix,
            "dist_coeffs": dist_coeffs,
            "pose_matrix": pose_matrix,
            "width": width,
            "height": height,
        })
        i += 1

    fs.release()

    if nb_camera is not None and nb_camera != len(cams):
        print(f"WARNING: yml says nb_camera={nb_camera} but found {len(cams)} "
              f"camera_N blocks. Using the {len(cams)} found.")
    if not cams:
        print(f"ERROR: no camera_N blocks found in {yml_path}")
        sys.exit(1)

    print(f"Loaded {len(cams)} camera(s) from {yml_path}")
    for idx, c in enumerate(cams):
        t = c["pose_matrix"][:3, 3]
        print(f"  camera_{idx}: {c['width']}x{c['height']}, "
              f"pose translation = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
    return cams


def build_pointcloud(color_path, depth_path, cam, depth_scale, depth_trunc=3.0):
    if not os.path.isfile(color_path):
        print(f"ERROR: color image not found: {color_path}")
        sys.exit(1)
    if not os.path.isfile(depth_path):
        print(f"ERROR: depth image not found: {depth_path}")
        sys.exit(1)
    color = o3d.io.read_image(color_path)
    depth = o3d.io.read_image(depth_path)
    if len(np.asarray(color)) == 0 or len(np.asarray(depth)) == 0:
        print(f"ERROR: failed to read image data from {color_path} / {depth_path} "
              f"(file exists but appears empty/corrupt)")
        sys.exit(1)
    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=depth_scale, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False,
    )
    cm = cam["camera_matrix"]
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        cam["width"], cam["height"], cm[0, 0], cm[1, 1], cm[0, 2], cm[1, 2]
    )
    return o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calib", required=True, help="Path to MC-Calib yml")
    parser.add_argument("--images", nargs="+", required=True,
                         help="One 'color.png,depth.png' pair per camera, in camera_0, "
                              "camera_1, ... order")
    parser.add_argument("--depth-scale", type=float, default=1000.0,
                         help="RealSense-style depth scale: metric_meters = raw_uint16_value / "
                              "depth_scale. 1000.0 is standard (raw values in millimeters). "
                              "Check your actual device's depth_scale if results look "
                              "systematically scaled wrong.")
    parser.add_argument("--depth-trunc", type=float, default=3.0)
    parser.add_argument("--voxel-size", type=float, default=0.005)
    parser.add_argument("--invert", action="store_true",
                         help="Try the opposite transform direction if the default looks wrong")
    parser.add_argument("--icp", action="store_true",
                         help="Run pairwise point-to-plane ICP refinement after the coarse "
                              "extrinsic alignment, to clean up residual mm-level error")
    parser.add_argument("--out", default="merged_scene.ply")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    cams = load_calibration(args.calib)

    if len(args.images) != len(cams):
        print(f"ERROR: yml has {len(cams)} camera(s) but you gave "
              f"{len(args.images)} --images pair(s). These must match 1:1, in order.")
        sys.exit(1)

    depth_scale = args.depth_scale

    pcds = []
    for i, (cam, pair) in enumerate(zip(cams, args.images)):
        color_path, depth_path = pair.split(",")
        print(f"camera_{i}: loading {color_path} / {depth_path}")
        pcd = build_pointcloud(color_path, depth_path, cam, depth_scale, args.depth_trunc)

        pose = cam["pose_matrix"]
        if i > 0:  # camera_0 is the reference, no transform needed
            T = pose if args.invert else np.linalg.inv(pose)
            pcd.transform(T)
        pcds.append(pcd)

    if args.icp:
        print("Running ICP refinement (aligning each cloud to the reference, camera_0)...")
        ref = pcds[0]
        ref.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
        for i in range(1, len(pcds)):
            pcds[i].estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.02, max_nn=30))
            result = o3d.pipelines.registration.registration_icp(
                pcds[i], ref, max_correspondence_distance=0.,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            )
            print(f"  camera_{i}: ICP fitness={result.fitness:.4f}, "
                  f"rmse={result.inlier_rmse:.5f} m")
            pcds[i].transform(result.transformation)

    merged = pcds[0]
    for pcd in pcds[1:]:
        merged += pcd
    merged = merged.voxel_down_sample(args.voxel_size)

    o3d.io.write_point_cloud(args.out, merged)
    print(f"Saved {len(merged.points)} points -> {args.out}")

    if args.show:
        o3d.visualization.draw_geometries([merged], window_name="Merged point cloud")


if __name__ == "__main__":
    main()