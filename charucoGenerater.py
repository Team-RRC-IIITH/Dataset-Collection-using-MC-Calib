import cv2

# Use the exact same configuration
ARUCO_DICT = cv2.aruco.DICT_5X5_100  
SQUARES_X = 5                        
SQUARES_Y = 5                        
SQUARE_LENGTH = 0.04                 
MARKER_LENGTH = 0.032                

# Initialize the dictionary and board
dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT)
board = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_LENGTH, MARKER_LENGTH, dictionary)

# Generate a high-resolution image (2000x2000 pixels for crisp printing)
# The physical measurements in meters don't affect this pixel generation, 
# it just strictly follows the layout rules.
img = board.generateImage((2000, 2000))

# Save the image to your current folder
filename = "OpenCV_Charuco_5x5_Perfect.png"
cv2.imwrite(filename, img)

print(f"Saved {filename} successfully! Print this image without scaling.")