"""
Project 651: Visual Servoing Implementation
Description:
Visual servoing is a technique in robotics that uses visual feedback (from cameras) to control the motion of a robot. The robot adjusts its movements based on the position and orientation of objects in its visual field. In this project, we will implement a simple image-based visual servoing system that controls the position of a robot's end effector (e.g., a robotic arm) based on the image coordinates of an object in the camera's field of view.
"""

import numpy as np
import cv2
 
# 1. Define the visual servoing system
class VisualServoing:
    def __init__(self, camera_matrix, dist_coeffs):
        self.camera_matrix = camera_matrix  # Camera matrix for calibration
        self.dist_coeffs = dist_coeffs  # Distortion coefficients
        self.target_position = np.array([320, 240])  # Target position (center of the image)
        self.kp = 0.01  # Proportional gain for visual servoing
 
    def calculate_error(self, current_position):
        """
        Calculate the error between the current position and the target position.
        :param current_position: Current position of the object in the image (x, y)
        :return: Error vector (dx, dy)
        """
        return self.target_position - current_position
 
    def control(self, error):
        """
        Apply proportional control to generate velocity command.
        :param error: Error vector (dx, dy)
        :return: Velocity command for the robot (e.g., linear velocity, angular velocity)
        """
        velocity_command = self.kp * error  # Simple proportional control
        return velocity_command
 
# 2. Initialize the visual servoing system
camera_matrix = np.array([[500, 0, 320],
                          [0, 500, 240],
                          [0, 0, 1]])  # Simplified camera intrinsic matrix (focal length, principal point)
dist_coeffs = np.zeros(4)  # Assuming no lens distortion
 
servoing_system = VisualServoing(camera_matrix, dist_coeffs)
 
# 3. Start video capture and perform visual servoing
cap = cv2.VideoCapture(0)  # Use the webcam
 
while True:
    ret, frame = cap.read()
    if not ret:
        break
 
    # 4. Detect an object in the image (for simplicity, using a simple color-based detection)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_color = np.array([30, 50, 50])  # Example color range for object (e.g., green)
    upper_color = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_color, upper_color)
    moments = cv2.moments(mask)
 
    # 5. Calculate the center of the object
    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        object_position = np.array([cx, cy])
 
        # 6. Calculate the error in position and apply visual servoing control
        error = servoing_system.calculate_error(object_position)
        velocity_command = servoing_system.control(error)
 
        # Display the velocity command
        print(f"Velocity Command (dx, dy): {velocity_command}")
 
        # Visualize the object and target position
        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), -1)  # Object center in green
        cv2.circle(frame, (servoing_system.target_position[0], servoing_system.target_position[1]), 10, (0, 0, 255), -1)  # Target center in red
 
    # 7. Display the frame with object detection and visualization
    cv2.imshow("Visual Servoing - Object Detection", frame)
 
    # 8. Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
 
# 9. Release resources and close the window
cap.release()
cv2.destroyAllWindows()