"""
capture_realsense.py
=====================
Live tool that:
  1. Auto-detects whatever RealSense cameras are currently connected (or
     use --serials to specify an explicit list/order).
  2. Shows a live per-camera view with a ChArUco detection overlay and a
     clear DETECTED / NOT DETECTED status -- this alone is your "are the
     markers detecting" check, just run the script and don't press any
     capture key.
  3. Lets you save synchronized captures to disk:
       - press 'c'  -> save a CALIBRATION pose (requires the board to be
                        detected in ALL connected cameras simultaneously)
       - press 's'  -> save a SCENE snapshot (no board required -- use
                        this after calibration to capture whatever object/
                        scene you want the merged point cloud of)
       - press 'q'  -> quit

OUTPUT LAYOUT (default --output captures/):
    captures/
      intrinsics/
        <serial>.json          (written once per camera at startup)
      pose_000/
        <serial>_color.png
        <serial>_depth.png     (16-bit PNG, raw depth units -- see intrinsics
                                 json for depth_scale to convert to meters)
      pose_001/
        ...
      scene_000/
        <serial>_color.png
        <serial>_depth.png
      ...

Nothing here does any calibration math -- that all happens offline in
calibrate_from_captures.py, which reads this exact folder structure.

USAGE:
    python capture_realsense.py                          # auto-detect all connected cams
    python capture_realsense.py --serials 234322307090,203522252036
    python capture_realsense.py --output my_session
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    print("ERROR: pyrealsense2 not installed. pip install pyrealsense2")
    sys.exit(1)

from common_calib import build_charuco_board, detect_charuco

STREAM_WIDTH = 1280
STREAM_HEIGHT = 720
STREAM_FPS = 28


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

    def intrinsics_dict(self):
        return {
            "serial": self.serial,
            "width": self.width,
            "height": self.height,
            "fx": float(self.camera_matrix[0, 0]),
            "fy": float(self.camera_matrix[1, 1]),
            "ppx": float(self.camera_matrix[0, 2]),
            "ppy": float(self.camera_matrix[1, 2]),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "depth_scale": float(self.depth_scale),
        }

    def stop(self):
        self.pipeline.stop()


def get_connected_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    return [d.get_info(rs.camera_info.serial_number) for d in devices]


def next_index(output_dir, prefix):
    existing = sorted(output_dir.glob(f"{prefix}_*"))
    if not existing:
        return 0
    last = existing[-1].name
    try:
        return int(last.split("_")[-1]) + 1
    except ValueError:
        return len(existing)


def save_capture(output_dir, prefix, cams, images):
    idx = next_index(output_dir, prefix)
    folder = output_dir / f"{prefix}_{idx:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    for cam, (color, depth) in zip(cams, images):
        cv2.imwrite(str(folder / f"{cam.serial}_color.png"), color)
        cv2.imwrite(str(folder / f"{cam.serial}_depth.png"), depth)  # 16-bit PNG, lossless
    print(f"Saved {folder}")
    return folder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serials", default="",
                         help="Comma-separated serials, in the order you want them treated "
                              "(first = reference camera later). Leave empty to auto-detect "
                              "all connected cameras.")
    parser.add_argument("--output", default="captures")
    args = parser.parse_args()

    if args.serials.strip():
        serials = [s.strip() for s in args.serials.split(",") if s.strip()]
    else:
        serials = get_connected_serials()
        if not serials:
            print("No RealSense devices detected. Check connections (see USB3 troubleshooting "
                  "if this keeps happening) and try again.")
            sys.exit(1)
        print(f"Auto-detected {len(serials)} camera(s): {serials}")

    output_dir = Path(args.output)
    intrinsics_dir = output_dir / "intrinsics"
    intrinsics_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to cameras:", serials)
    cams = [RealSenseCam(s) for s in serials]

    for cam in cams:
        with open(intrinsics_dir / f"{cam.serial}.json", "w") as f:
            json.dump(cam.intrinsics_dict(), f, indent=2)
    print(f"Wrote intrinsics for {len(cams)} camera(s) to {intrinsics_dir}")

    board, dictionary, api_kind = build_charuco_board()

    print("\nControls:")
    print("  'c' -> save CALIBRATION pose (needs board detected in ALL cameras)")
    print("  's' -> save SCENE snapshot (no board needed)")
    print("  'q' -> quit\n")

    try:
        while True:
            color_imgs = []
            depth_imgs = []
            detections = []

            for cam in cams:
                color, depth = cam.get_frames()
                if color is None:
                    color = np.zeros((STREAM_HEIGHT, STREAM_WIDTH, 3), dtype=np.uint8)
                    depth = np.zeros((STREAM_HEIGHT, STREAM_WIDTH), dtype=np.uint16)
                gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
                ch_corners, ch_ids = detect_charuco(gray, board, dictionary, api_kind)
                color_imgs.append(color)
                depth_imgs.append(depth)
                detections.append((ch_corners, ch_ids))

            all_detected = all(d[0] is not None for d in detections)

            display_imgs = []
            for i, (cam, color, (ch_corners, ch_ids)) in enumerate(zip(cams, color_imgs, detections)):
                vis = color.copy()
                if ch_corners is not None:
                    # Draw corners manually with cv2.circle instead of
                    # cv2.aruco.drawDetectedCornersCharuco -- that function has
                    # a flaky internal assertion on some OpenCV 5.x builds even
                    # when corners/ids are valid and equal length.
                    pts = np.asarray(ch_corners).reshape(-1, 2)
                    for pt in pts:
                        cv2.circle(vis, (int(pt[0]), int(pt[1])), 5, (0, 255, 0), -1)
                    n = len(pts)
                    status, color_txt = f"DETECTED ({n} corners)", (0, 255, 0)
                else:
                    status, color_txt = "NOT DETECTED", (0, 0, 255)
                small = cv2.resize(vis, (STREAM_WIDTH // 2, STREAM_HEIGHT // 2))
                cv2.putText(small, f"cam {i}: {cam.serial}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
                cv2.putText(small, status, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color_txt, 2)
                display_imgs.append(small)

            # arrange in a grid: 2 columns
            rows = []
            for i in range(0, len(display_imgs), 2):
                pair = display_imgs[i:i + 2]
                if len(pair) == 1:
                    pair.append(np.zeros_like(pair[0]))
                rows.append(np.hstack(pair))
            combined = np.vstack(rows) if rows else np.zeros((100, 100, 3), dtype=np.uint8)

            footer = "ALL DETECTED - 'c' available" if all_detected else "waiting for all cameras to detect board..."
            cv2.putText(combined, footer, (10, combined.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0) if all_detected else (0, 0, 255), 2)
            cv2.imshow("RealSense capture", combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                if all_detected:
                    save_capture(output_dir, "pose", cams, zip(color_imgs, depth_imgs))
                else:
                    print("Board not detected in all cameras -- not saving a calibration pose.")
            elif key == ord('s'):
                save_capture(output_dir, "scene", cams, zip(color_imgs, depth_imgs))
    finally:
        cv2.destroyAllWindows()
        for cam in cams:
            cam.stop()


if __name__ == "__main__":
    main()