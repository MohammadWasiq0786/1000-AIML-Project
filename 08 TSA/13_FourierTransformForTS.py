"""
Project 293. Fourier transform for time series
Description:
The Fourier Transform breaks down a time series into its sine and cosine components — revealing dominant frequencies, cycles, and periodicities in the data. It's widely used in:

Signal processing

Seasonality analysis

Noise filtering

We’ll apply Fast Fourier Transform (FFT) to a time series to extract and visualize its frequency spectrum.

About:
✅ What It Does:
Simulates a signal with known frequencies (5 Hz and 10 Hz)

Applies FFT to reveal hidden periodicities

Visualizes the frequency domain, highlighting signal components

Great for identifying cyclical behavior or filtering noise
"""

import numpy as np
import matplotlib.pyplot as plt
 
# 1. Generate synthetic signal with known frequencies
t = np.linspace(0, 2, 500)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.sin(2 * np.pi * 10 * t) + np.random.normal(0, 0.2, len(t))
 
# 2. Apply Fast Fourier Transform (FFT)
fft_result = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(t), d=(t[1] - t[0]))  # frequency bins
magnitude = np.abs(fft_result)
 
# 3. Only keep positive half (symmetric output)
half = len(t) // 2
frequencies = frequencies[:half]
magnitude = magnitude[:half]
 
# 4. Plot original signal
plt.figure(figsize=(10, 3))
plt.plot(t, signal)
plt.title("Original Time Series Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
 
# 5. Plot frequency spectrum
plt.figure(figsize=(10, 3))
plt.plot(frequencies, magnitude)
plt.title("Frequency Spectrum (FFT)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)
plt.show()