"""
This script allows users to manually select four corner points on a live camera feed to
transform the perspective of the image and correct any distortion.
 
**Features:**
* Captures video from a specified camera ID.
* Allows selecting four points using mouse clicks.
* Saves the selected points to a pickle file for future use.
* Warps the image based on the selected points.
* Displays the original and transformed images side-by-side.
 
**Usage:**
1. Run the script.
2. Click on the four corners of the desired area in the "Original Image" window.
3. The script will automatically update the transformed image and save the points to a file.
4. You can then use the saved points to warp other images without manually selecting them.
 
**Requirements:**
* OpenCV library
* Numpy library
* pickle library
 
"""
import pickle
import cv2
import numpy as np
 
#########################
# Camera settings
cam_id = 0  # Change this to your desired camera ID
width, height = 1920, 1080  # Change this to desired image resolution
#########################
 
# Initialize variables
cap = cv2.VideoCapture(cam_id)  # For Webcam
cap.set(3, width)  # Set width
cap.set(4, height)  # Set height
points = np.zeros((4, 2), int)  # Array to store clicked points
counter = 0  # Counter to track the number of clicked points
 
 
def mousePoints(event, x, y, flags, params):
    """
    Callback function for mouse clicks.
 
    Args:
        event: OpenCV event type (e.g., cv2.EVENT_LBUTTONDOWN)
        x: X-coordinate of the mouse click
        y: Y-coordinate of the mouse click
        flags: Additional flags associated with the event
        params: User-defined parameters passed to the callback function
 
    Returns:
        None
    """
    global counter
    if event == cv2.EVENT_LBUTTONDOWN:
        points[counter] = x, y  # Store the clicked point
        counter += 1  # Increment counter
        print(f"Clicked points: {points}")
 
def warp_image(img, points, size=[1920, 1080]):
    """
    Warps an image based on the selected points.
 
    Args:
        img: The image to be warped
        points: Array containing four clicked points
        size: Desired size of the warped image
 
    Returns:
        imgOutput: The warped image
        matrix: The perspective transformation matrix
    """
    pts1 = np.float32(points)  # Convert points to float32
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Calculate perspective transformation matrix
    imgOutput = cv2.warpPerspective(img, matrix, (size[0], size[1]))  # Warp the image
    return imgOutput, matrix
 
 
while True:
    success, img = cap.read()
 
    if counter == 4:
        # Save selected points to file
        fileObj = open("Hamburg_map.p", "wb")
        pickle.dump(points, fileObj)
        fileObj.close()
        print("Points saved to file: Hamburg_map.p")
 
        # Warp the image
        imgOutput, matrix = warp_image(img, points)
        # Display warped image
        cv2.imshow("Output Image ", imgOutput)
 
    # Draw circles at clicked points
    for x in range(0, 4):
        cv2.circle(img, (points[x][0], points[x][1]), 10, (0, 255, 0), cv2.FILLED)
 
    cv2.imshow("Original Image ", img)
    cv2.setMouseCallback("Original Image ", mousePoints)
    cv2.waitKey(1)  # Wait for a key press
 
# Release resources
cap.release()
cv2.destroyAllWindows()