import pyrealsense2 as rs
import numpy as np
import cv2
import os

# 1. Setup Dataset Folders
# This matches the folder we mounted into Docker
base_dir = os.path.expanduser("~/realsense_dataset")
cam1_dir = os.path.join(base_dir, "Cam_001")
cam2_dir = os.path.join(base_dir, "Cam_002")

os.makedirs(cam1_dir, exist_ok=True)
os.makedirs(cam2_dir, exist_ok=True)

# 2. Detect Connected Cameras
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    print(f"Error: Found only {len(devices)} RealSense devices. Connect at least 2.")
    exit()

print(f"Found {len(devices)} cameras. Initializing the first two...")

# 3. Start Pipelines for Both Cameras
pipelines = []
for i, dev in enumerate(devices[:2]):
    serial = dev.get_info(rs.camera_info.serial_number)
    print(f"Camera {i+1} Serial: {serial} -> Saving to Cam_00{i+1}")
    
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    
    # 1280x720 at 30 FPS is standard high-res for D455 RGB
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    
    pipeline.start(config)
    pipelines.append(pipeline)

# 4. Capture Loop
img_count = 1
print("\n--- INSTRUCTIONS ---")
print("[SPACE] Capture synchronized frames")
print("[ Q ] Quit script")
print("--------------------\n")

try:
    while True:
        # Grab frames from both cameras
        framesets = [pipe.wait_for_frames() for pipe in pipelines]
        color_frames = [fs.get_color_frame() for fs in framesets]

        if not all(color_frames):
            continue

        # Convert to numpy arrays for OpenCV
        images = [np.asanyarray(f.get_data()) for f in color_frames]

        # Combine images side-by-side for the live preview
        combined_img = np.hstack(images)
        display_img = cv2.resize(combined_img, (1280, 480)) # Scale down for viewing
        
        cv2.imshow("Calibration Capture (Cam 1 | Cam 2)", display_img)

        # Handle Keyboard Inputs
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
        elif key & 0xFF == 32: # Spacebar
            # MC-Calib likes 3-digit zero-padded numbers
            filename = f"{img_count:03d}.png" 
            
            cv2.imwrite(os.path.join(cam1_dir, filename), images[0])
            cv2.imwrite(os.path.join(cam2_dir, filename), images[1])
            
            print(f"Captured: {filename} in both folders.")
            img_count += 1

finally:
    print("Stopping cameras...")
    for pipe in pipelines:
        pipe.stop()
    cv2.destroyAllWindows()