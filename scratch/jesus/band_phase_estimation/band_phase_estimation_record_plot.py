#!/usr/bin/env python3

import math
import matplotlib.pyplot as plt

# Definitions
data_file='/data/jesus/band_estimation.dat'

# Extract the data from the data file
t = []
tau = []
theta = []
phase0_raw = []
phase1_raw = []
with open(data_file, 'r') as f:
    for line in f:
        x0, x1, x2, x3, x4 = map(float, line.replace('[', '').replace(']', '').split())
        t.append(x0)
        tau.append(x1)
        theta.append(x2)
        phase0_raw.append(x3)
        phase1_raw.append(x4)

phase0 = []
phase1 = []
# Convert raw phase to rad
for p in phase0_raw:
    phase0.append(p*math.pi/2**15)

for p in phase1_raw:
    phase1.append(p*math.pi/2**15)

# Plot the phase signals
plt.figure('Raw phase (channel = 0)')
plt.plot(phase0)
plt.grid()
plt.xlabel('Sample number')
plt.ylabel('Phase [rad]')

plt.figure('Raw phase (channel = 1)')
plt.plot(phase1)
plt.grid()
plt.xlabel('Sample number')
plt.ylabel('Phase [rad]')

plt.figure('Phase time delay (tau) estimation')
plt.plot(tau)
plt.grid()
plt.xlabel('Sample number')
plt.ylabel('Phase time delay (tau) [s]')

plt.figure('Phase offset (theta) estimation')
plt.plot(theta)
plt.grid()
plt.xlabel('Sample number')
plt.ylabel('Phase offset (theta) [rad]')

plt.show()
