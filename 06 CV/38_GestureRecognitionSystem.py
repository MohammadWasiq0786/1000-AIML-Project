"""
Project 238. Gesture recognition system
Description:
Gesture recognition enables machines to understand human hand gestures as input commands. It's widely used in touchless interfaces, sign language translation, gaming, and robotics. In this project, we'll build a real-time gesture recognition system using MediaPipe Hands from Google, which detects hand landmarks and classifies gestures.

About:
What It Does:
This project captures your hand in real-time, analyzes the finger positions, and classifies gestures like "Fist" vs. "Open Palm". You can expand it to support more gestures like Thumbs Up, Peace, or Rock sign, and integrate it into smart TVs, VR/AR apps, or even IoT devices for gesture-based control.
"""
# Install required packages:
# pip install mediapipe opencv-python
 
import cv2
import mediapipe as mp
 
# Initialize MediaPipe Hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
 
# Define a simple gesture classifier (e.g., Open Palm vs. Fist)
def classify_gesture(landmarks):
    # Use distance between landmarks to detect an open palm
    # Example: if finger tip (index) is above its base, it's extended
    tips = [8, 12, 16, 20]  # Index to pinky
    count = 0
    for tip in tips:
        if landmarks[tip].y < landmarks[tip - 2].y:
            count += 1
    return "Open Palm" if count >= 4 else "Fist"
 
# Start webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
 
    # Flip and convert the image to RGB
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Process frame to detect hand landmarks
    result = hands.process(rgb)
 
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
 
            # Classify the gesture
            gesture = classify_gesture(hand_landmarks.landmark)
 
            # Display gesture label
            cv2.putText(frame, f'Gesture: {gesture}', (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
 
    # Show the result
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to quit
        break
 
cap.release()
cv2.destroyAllWindows()