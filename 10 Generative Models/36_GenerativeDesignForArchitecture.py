"""
Project 396. Generative design for architecture
Description:
Generative design in architecture refers to using generative algorithms to create optimized building designs, urban layouts, or other architectural elements based on given inputs like environmental factors, structural constraints, and aesthetic preferences. These designs can be highly efficient and unique, enabling architects to explore a wide range of possibilities that might not be intuitive.

In this project, we will explore generative design techniques that use optimization algorithms to generate architectural layouts. We'll create a basic model that generates architectural floor plans or building layouts using an evolutionary algorithm or neural networks.

About:
✅ What It Does:
Generates a random floor plan with rooms placed randomly on a grid.

The number of rooms and their sizes are randomly chosen within predefined limits.

Visualizes the generated floor plan using matplotlib, where the rooms are marked in white, and the empty space is in black.

Key features:
Random design generation: This basic implementation uses a random approach to place rooms on a grid.

Scalability: The design can be expanded to handle more complex layouts, such as adding walls, windows, doors, or specific room types.

Visualization: The floor plan is visualized as a simple 2D layout, making it easy to see how rooms are distributed in the space.
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Define a simple function to generate a random floor plan
def generate_floor_plan(width=10, height=10):
    # Randomly place rooms in a grid
    floor_plan = np.zeros((height, width))
    
    num_rooms = np.random.randint(3, 6)  # Random number of rooms
    for _ in range(num_rooms):
        room_width = np.random.randint(2, width // 2)
        room_height = np.random.randint(2, height // 2)
        
        x_pos = np.random.randint(0, width - room_width)
        y_pos = np.random.randint(0, height - room_height)
        
        # Place room in the grid
        floor_plan[y_pos:y_pos + room_height, x_pos:x_pos + room_width] = 1
    
    return floor_plan
 
# 2. Visualization function to plot the floor plan
def plot_floor_plan(floor_plan):
    plt.imshow(floor_plan, cmap='gray', origin='upper')
    plt.title("Generated Floor Plan")
    plt.axis('off')
    plt.show()
 
# 3. Generate and display a random floor plan
floor_plan = generate_floor_plan(width=10, height=10)
plot_floor_plan(floor_plan)