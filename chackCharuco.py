import cv2
import os

# 1. Paths and parameters (Matching your YAML configuration)
image_path = os.path.expanduser("~/realsense_dataset/Cam_001/001.png")
squaresX = 5          # number_x_square
squaresY = 5          # number_y_square
squareLength = 0.04   # length_square
markerLength = 0.032  # length_marker

# Load the image
img = cv2.imread(image_path)
if img is None:
    print(f"Error: Could not load image at {image_path}. Did you capture images yet?")
    exit()

# 2. Setup ChArUco Detector (MC-Calib uses DICT_6X6_250)
print("Detecting ChArUco board...")
try:
    # OpenCV 4.7+ API
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((squaresX, squaresY), squareLength, markerLength, dictionary)
    charuco_detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(img)

except AttributeError:
    # OpenCV 4.6 and older API (Fallback)
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard_create(squaresX, squaresY, squareLength, markerLength, dictionary)
    marker_corners, marker_ids, rejected = cv2.aruco.detectMarkers(img, dictionary)
    if marker_corners and len(marker_corners) > 0:
        ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(marker_corners, marker_ids, img, board)
    else:
        charuco_ids = None

# 3. Draw and Display Results
if charuco_ids is not None and len(charuco_ids) > 0:
    print(f"Success! Detected {len(charuco_ids)} inner corners.")
    # Draw green boxes/dots on the detected corners
    cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids, (0, 255, 0))
else:
    print("Failed to detect ChArUco corners. Check focus, lighting, or the printed board.")

# Resize the window so it fits on your screen
display_img = cv2.resize(img, (1280, 720))
cv2.imshow("Detection Verification (Press any key to close)", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
