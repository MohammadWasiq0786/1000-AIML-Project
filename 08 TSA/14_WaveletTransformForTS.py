"""
Project 294. Wavelet transform for time series
Description:
Unlike Fourier Transform, which gives global frequency information, the Wavelet Transform provides time-localized frequency analysis — meaning it can capture when certain frequencies occur. This is crucial for analyzing non-stationary signals, like sudden spikes or changing patterns over time.

We’ll use the Continuous Wavelet Transform (CWT) with a Morlet wavelet to visualize both time and frequency components.

About:
✅ What It Does:
Creates a chirp signal with increasing frequency

Applies CWT to decompose the signal into time-scale space

Displays a scalogram showing when certain frequencies appear

Great for detecting transients, bursts, and non-stationary patterns
"""


import numpy as np
import matplotlib.pyplot as plt
import pywt
 
# 1. Generate a signal with time-varying frequencies
t = np.linspace(0, 1, 400)
signal = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 20 * t * t)  # chirp: increasing freq
 
# 2. Perform Continuous Wavelet Transform (CWT)
scales = np.arange(1, 100)
coeffs, freqs = pywt.cwt(signal, scales, wavelet='morl', sampling_period=t[1] - t[0])
 
# 3. Plot original signal
plt.figure(figsize=(10, 3))
plt.plot(t, signal)
plt.title("Time Series Signal with Varying Frequency")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
 
# 4. Plot scalogram (time-frequency representation)
plt.figure(figsize=(10, 4))
plt.imshow(np.abs(coeffs), extent=[0, 1, 1, 100], cmap='jet', aspect='auto', origin='lower')
plt.title("Wavelet Transform (Scalogram)")
plt.xlabel("Time")
plt.ylabel("Scale (Inverse Frequency)")
plt.colorbar(label="Magnitude")
plt.show()