import cv2
import numpy as np
import pyrealsense2 as rs
import sys

# ==========================================
# 1. CHARUCO CONFIGURATION
# ==========================================
# ==========================================
# 1. CHARUCO CONFIGURATION
# ==========================================
ARUCO_DICT = cv2.aruco.DICT_5X5_100  
SQUARES_X = 5                        
SQUARES_Y = 5                        
SQUARE_LENGTH = 0.04                 # 4 cm in meters
MARKER_LENGTH = 0.032                # 3.2 cm in meters

dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)
charuco_detector = cv2.aruco.CharucoDetector(board)

# ==========================================
# 2. DETECTION HELPER FUNCTION
# ==========================================
def process_frame(frame, detector):
    """Runs Charuco detection on a frame and draws the overlays."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    status_text = "NOT DETECTED"
    status_color = (0, 0, 255) # Red

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
        
        if charuco_corners is not None and charuco_ids is not None and len(charuco_ids) > 0:
            try:
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids, (0, 255, 0))
                status_text = f"DETECTED: {len(charuco_ids)} corners"
                status_color = (0, 255, 0)
            except cv2.error:
                status_text = "Frame glitch, ignoring..."
                status_color = (0, 165, 255)
        else:
            status_text = "Markers found, resolving corners..."
            status_color = (0, 165, 255)

    # Draw background box and text
    cv2.rectangle(frame, (10, 15), (600, 70), (0,0,0), -1)
    cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
    
    return frame

# ==========================================
# 3. REALSENSE DUAL CAMERA SETUP
# ==========================================
print("Searching for Intel RealSense devices...")
ctx = rs.context()
devices = ctx.query_devices()

if len(devices) < 2:
    print(f"Error: Found {len(devices)} camera(s). Please connect at least 2 RealSense cameras.")
    sys.exit()

# Extract serial numbers for the first two cameras
serial1 = devices[0].get_info(rs.camera_info.serial_number)
serial2 = devices[1].get_info(rs.camera_info.serial_number)
print(f"Found Camera 1 (Serial: {serial1})")
print(f"Found Camera 2 (Serial: {serial2})")

# Initialize pipelines
pipelines = []
for serial in [serial1, serial2]:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial) # Bind this config to a specific camera
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)
    pipelines.append(pipeline)

print("Both cameras started. Press 'q' to quit.")

# ==========================================
# 4. MAIN LOOP
# ==========================================
try:
    while True:
        # Wait for frames from both cameras
        frames1 = pipelines[0].wait_for_frames()
        frames2 = pipelines[1].wait_for_frames()
        
        color_frame1 = frames1.get_color_frame()
        color_frame2 = frames2.get_color_frame()

        if not color_frame1 or not color_frame2:
            continue

        # Convert to numpy arrays
        img1 = np.asanyarray(color_frame1.get_data())
        img2 = np.asanyarray(color_frame2.get_data())

        # Process both frames using the helper function
        img1_processed = process_frame(img1, charuco_detector)
        img2_processed = process_frame(img2, charuco_detector)

        # Two 1280x720 windows side-by-side might exceed your monitor width.
        # Resize them by 50% for display purposes. (Detection still happens at 720p)
        img1_resized = cv2.resize(img1_processed, (640, 360))
        img2_resized = cv2.resize(img2_processed, (640, 360))

        # Stitch them horizontally side-by-side
        combined_image = np.hstack((img1_resized, img2_resized))

        cv2.imshow('Dual D455 ChArUco Detection', combined_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    print("Stopping pipelines...")
    for pipeline in pipelines:
        pipeline.stop()
    cv2.destroyAllWindows()