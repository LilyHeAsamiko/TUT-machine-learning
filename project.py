#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 10:32:30 2018

@author: He Qin
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io import wavfile
from scipy import signal
import winsound
import sounddevice as sd
import math

plt.figure(1)
# Plot time domain visualization 
fs, x = wavfile.read('rhythm_birdland.wav')  # fs is the sampling frequency of the signal

fs = float(fs)
f, t, F = signal.stft(x, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(F), vmin=0)
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
gamma = 0.3
alpha = 0.3
W = (np.abs(F)) ** (2 * gamma)
P = (1 / 2) * W
H = (1 / 2) * W
P1 = P
H1 = H
kmax = 20
for k in range(0, (kmax - 1)):
    for h in range(1, (np.shape(H)[0]) - 1):
        for i in range(1, (np.shape(H)[1]) - 1):
            delta = alpha * ((H[h, int(i - 1)] - 2.0 * H[h, i] + H[h, int(i + 1)]) / 4.0) - (1 - alpha) * ((P[int(h - 1), i] - 2 * P[h, i] + P[int(h + 1), i]) / 4.0)
            H[h, i] = np.minimum(np.maximum((float(H[h, i]) + float(delta)), 0.0), float(W[h, i]))
            P1[h, i] = W[h, i] - H1[h, i]
    H = H1
    P = P1
# At the end of the for loop we will have H kmax-1 and P kmax-1
print("end of iteration for K")
for h in range(0, np.shape(H)[0]):
    for i in range(0, np.shape(H)[1]):
        if H[h, i] < P[h, i]:
            H[h, i] = 0
            P[h, i] = W[h, i]
        else :
            H[h, i] = W[h, i]
            P[h, i] = 0
print("Matrix set")
t, h = signal.istft(H, fs, nperseg=1000)
SNRH = 10 * np.log10(sum(x**2)/sum((x-h)**2))
print('SNR for percussive part %f' % SNRH)
plt.figure(2)
plt.plot(t, h)
plt.title('Time domain visualization of Harmonic part')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()
plt.figure(3)
t, p = signal.istft(P, fs, nperseg=1000)
SNRP = 10 * np.log10(sum(x**2)/sum((x-p)**2))
print('SNR for percussive part %f' % SNRP)
plt.plot(t, p)
plt.title('Time domain visualization of Percussive part')
plt.grid('on')
plt.ylabel('Amplitude')
plt.xlabel('Time (s)')
plt.show()
plt.figure(4)
f, t, HSpec = signal.stft(h, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(HSpec), vmin=0)
plt.title('Harmonic spectrum')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
plt.figure(5)
f, t, PSpec = signal.stft(p, fs, nperseg=1000)
plt.pcolormesh(t, f, np.abs(PSpec), vmin=0)
plt.title('Percussive spectrum')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()
winsound.PlaySound('rhythm_birdland.wav', winsound.SND_FILENAME)
scipy.io.wavfile.write('Percussive.wav', int(fs), np.int16(p))
winsound.PlaySound('Percussive.wav', winsound.SND_FILENAME)
scipy.io.wavfile.write('Harmonic.wav', int(fs), np.int16(h))
winsound.PlaySound('Harmonic.wav', winsound.SND_FILENAME)