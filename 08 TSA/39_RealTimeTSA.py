"""
Project 319. Real-time time series analysis
Description:
Real-time time series analysis involves processing and analyzing data as it arrives — essential for:

IoT sensor monitoring

Stock market feeds

Network traffic analysis

In this project, we simulate a real-time data stream, perform rolling window analytics, and plot live updates using matplotlib animation.

About:
✅ What It Does:
Simulates a live data stream from a noisy sine wave

Maintains a sliding window buffer for analytics

Computes and plots a rolling mean

Uses matplotlib animation for real-time visual updates

This is the foundation for alerting systems, dashboarding, or streaming ML models.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
 
# 1. Simulate a real-time data source
np.random.seed(42)
stream_length = 200
data_stream = np.sin(np.linspace(0, 10*np.pi, stream_length)) + 0.2 * np.random.randn(stream_length)
 
# 2. Create buffers for rolling window
window_size = 30
data_window = deque([0]*window_size, maxlen=window_size)
ma_window = deque([0]*window_size, maxlen=window_size)
 
# 3. Set up real-time plot
fig, ax = plt.subplots()
line_data, = ax.plot([], [], label="Signal")
line_ma, = ax.plot([], [], label="Rolling Mean", linestyle='--')
ax.set_xlim(0, window_size)
ax.set_ylim(-2, 2)
ax.set_title("Real-Time Time Series Analysis")
ax.legend()
 
# 4. Update function for animation
def update(frame):
    data_point = data_stream[frame]
    data_window.append(data_point)
    ma_window.append(np.mean(data_window))
 
    line_data.set_data(range(window_size), list(data_window))
    line_ma.set_data(range(window_size), list(ma_window))
    return line_data, line_ma
 
ani = animation.FuncAnimation(fig, update, frames=len(data_stream), interval=100, blit=True)
plt.show()