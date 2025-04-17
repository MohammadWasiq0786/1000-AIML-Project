"""
Project 663: Human-Robot Interaction
Description:
Human-Robot Interaction (HRI) is an essential aspect of robotics where robots are designed to communicate, understand, and collaborate with humans. In this project, we will implement a simple human-robot interaction system where a robot can respond to basic voice commands using speech recognition and perform simple tasks like moving to a specific location or picking up an object. We will use speech recognition for command input and robot motion simulation for task execution.
"""

import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt
import time
 
# 1. Define the robot class with basic motion capabilities
class Robot:
    def __init__(self, position=(0, 0)):
        self.position = np.array(position)  # Initial position (x, y)
 
    def move_to(self, target_position):
        """
        Move the robot towards the target position (simulate movement).
        """
        direction = target_position - self.position
        distance = np.linalg.norm(direction)
        
        if distance > 0:
            direction = direction / distance  # Normalize direction
            self.position += direction * 0.1  # Move the robot step by step
 
    def get_position(self):
        return self.position
 
# 2. Define the speech recognition system
def listen_for_command():
    """
    Listen for a speech command from the user.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening for command...")
        audio = recognizer.listen(source)
        try:
            command = recognizer.recognize_google(audio)
            print(f"Command received: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I did not understand the command.")
            return None
        except sr.RequestError:
            print("Sorry, there was a problem with the speech recognition service.")
            return None
 
# 3. Define the task handler for human-robot interaction
class HumanRobotInteraction:
    def __init__(self, robot):
        self.robot = robot
 
    def process_command(self, command):
        """
        Process the command and perform the corresponding task.
        :param command: Command received from the user
        """
        if 'move to' in command:
            # Parse the target position from the command (assume "move to x, y")
            parts = command.split('move to')[-1].strip()
            target_position = tuple(map(float, parts.split(',')))
            print(f"Moving to position: {target_position}")
            self.robot.move_to(np.array(target_position))
        else:
            print("Command not recognized.")
 
    def interact(self):
        """
        Start the interaction loop with the user.
        """
        while True:
            command = listen_for_command()  # Listen for command from the user
            if command:
                self.process_command(command)  # Process the command
 
            # Simulate robot movement and update position
            position = self.robot.get_position()
            print(f"Robot position: {position}")
            time.sleep(1)
 
            # Break the loop after a certain condition (e.g., a certain position is reached)
            if np.linalg.norm(position - np.array([5, 5])) < 0.2:
                print("Goal reached!")
                break
 
# 4. Initialize the robot and human-robot interaction system
robot = Robot(position=(0, 0))
hri_system = HumanRobotInteraction(robot)
 
# 5. Start the human-robot interaction
hri_system.interact()