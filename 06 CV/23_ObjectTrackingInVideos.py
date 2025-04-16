"""
Project 223. Object tracking in videos
Description:
Object tracking is the task of detecting an object in a video and continuously tracking its location across frames. It's used in video surveillance, autonomous vehicles, drone navigation, and sports analytics. In this project, we'll use OpenCV's built-in tracking API, specifically the CSRT (Discriminative Correlation Filter) tracker, for accurate object tracking.

About:
What It Does:
This project allows you to interactively select an object in a video and tracks it in real-time using the CSRT algorithm, known for its accuracy. It can be enhanced with multiple object tracking, YOLO-based detection, or Deep SORT for robust real-time tracking across complex scenes.
"""
# Install OpenCV if not already:
# pip install opencv-python
 
import cv2
 
# Load a video file
video_path = 'object_tracking_sample.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_path)
 
# Read the first frame
success, frame = cap.read()
if not success:
    print("Failed to read video.")
    exit()
 
# Let the user draw a bounding box around the object to track
bbox = cv2.selectROI("Select Object to Track", frame, fromCenter=False)
cv2.destroyAllWindows()
 
# Initialize CSRT tracker (good for accuracy, even with occlusion or rotation)
tracker = cv2.TrackerCSRT_create()
tracker.init(frame, bbox)
 
# Loop through video frames
while True:
    success, frame = cap.read()
    if not success:
        break
 
    # Update the tracker with the new frame
    success, bbox = tracker.update(frame)
 
    # Draw the bounding box if tracking was successful
    if success:
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Tracking", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Lost Tracking", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
 
    # Show the frame
    cv2.imshow("Object Tracking", frame)
 
    # Break loop with ESC key
    if cv2.waitKey(30) & 0xFF == 27:
        break
 
# Release resources
cap.release()
cv2.destroyAllWindows()