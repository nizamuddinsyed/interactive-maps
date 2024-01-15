"""
Script for marking and saving Regions of Interest (ROIs) on a video stream.
 
Libraries Used:
- cv2: OpenCV library for computer vision tasks
- np: NumPy library for numerical operations
- pickle: Library for serializing Python objects
 
Parameters:
- map_file_path: Path to the file containing map coordinates (serialized using pickle).
- districts_file_path: Path to the file containing previously marked countries (serialized using pickle).
- cam_id: Camera ID for capturing video (default is set to 3).
- width, height: Width and height of the webcam frame.
 
Usage:
1. Run the script.
2. Mark four points for each country polygon by clicking with the left mouse button.
3. Press 's' to save the marked polygon along with the country name.
4. Press 'd' to delete the last marked polygon.
5. Press 'q' to save the marked polygons to the file and exit the script.
 
"""
 
import pickle  # Pickle library for serializing Python objects
import cv2  # OpenCV library for computer vision tasks
import numpy as np  # NumPy library for numerical operations
 
#############################
map_file_path = "../Step1-getCornerPoints/Hamburg_map.p"
districts_file_path = "../Step2-getDistrictsPolygon/Hamburg_districts.p"
cam_id = 0 
width, height = 1920, 1080
#############################
 
 
# Open a connection to the webcam
cap = cv2.VideoCapture(cam_id)  # For Webcam
cap.set(3, width)
cap.set(4, height)

file_obj = open(map_file_path, 'rb')
map_points = pickle.load(file_obj)
file_obj.close()
print(f"Loaded Hamburg map coordinates.\n", map_points)

# Temporary list to store the four points of the current polygon being marked
current_polygon = []
 
# Counter to keep track of how many polygons have been created
counter = 0
polygons = []
# Load previously defined Regions of Interest (ROIs) polygons from a file
# BUG : HANDLE FILE EXCEPTION, if the file is not present its not loading warp image and not able to create polygons
# try:

if districts_file_path:
    file_obj = open(districts_file_path, 'rb')
    polygons = pickle.load(file_obj)
    file_obj.close()
    print(f"Loaded {len(polygons)} Hamburg Districts.")
else:
    polygons = []


def warp_image(img, points, size=[1920, 1080]):
    pts1 = np.float32(points)  # Convert points to float32
    pts2 = np.float32([[0, 0], [size[0], 0], [0, size[1]], [size[0], size[1]]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # Calculate perspective transformation matrix
    imgOutput = cv2.warpPerspective(img, matrix, (size[0], size[1]))  # Warp the image
    return imgOutput, matrix


# Function to handle mouse events (used to mark points for polygons)
def mousePoints(event, x, y, flags, params):
    """
    Handle mouse events to mark points for polygons.
 
    Parameters:
    - event: The type of mouse event.
    - x, y: Coordinates of the mouse click.
    - flags: Additional information about the mouse event.
    - params: Additional parameters passed to the callback.
 
    Returns:
    None
    """
    global counter, current_polygon
 
    # If left mouse button is clicked
    if event == cv2.EVENT_LBUTTONDOWN:
        # Append the clicked point (x, y) to the current_polygon list
        current_polygon.append((x, y))

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    imgWarped, _ = warp_image(img, map_points)

    # print(current_polygon)
    key = cv2.waitKey(1)

    # If the "s" key is pressed, save the polygon
    if key == ord("s") and len(current_polygon) > 2:
        country_name = input("Enter the District name: ")
        polygons.append([current_polygon, country_name])  # Add the polygon to the list
        current_polygon = []  # Reset for the next polygon
        counter += 1  # Increment the counter
        print("Number of Hamburg Districts saved: ", len(polygons))  # Print the collected polygons

    # If the "q" key is pressed, save the polygons and exit the loop
    if key == ord("q"):
        fileObj = open(districts_file_path, 'wb')
        pickle.dump(polygons, fileObj)  # Save the polygons to a file
        fileObj.close()
        print(f"Saved {len(polygons)} Districts")
        break
 
    if key == ord("d"):
        polygons.pop()

    if current_polygon:
        cv2.polylines(imgWarped, [np.array(current_polygon)], isClosed=True, color=(0, 0, 255), thickness=4)

    overlay = imgWarped.copy()
    # Draw the collected polygons on the image
    for polygon, name in polygons:
        cv2.polylines(imgWarped, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.fillPoly(overlay, [np.array(polygon)], (0, 255, 0))

    cv2.addWeighted(overlay, 0.35, imgWarped, 0.65, 0, imgWarped)

    # Display the image with marked polygons
    cv2.imshow("Original Image", img)
    cv2.imshow("Warped Image", imgWarped)

    # Set the mouse callback function for marking points
    cv2.setMouseCallback("Warped Image", mousePoints)

    cv2.waitKey(1)
 