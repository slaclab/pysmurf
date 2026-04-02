#!/usr/bin/env python3
import sys 
import numpy as np
import serial 
from time import sleep 
import numpy as np

import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
import matplotlib.pylab as plt


def create_waveform(pulse_durations_us, amplitudes, wvfm_samp_rate_Hz, wvfm_duration_s, dac_max=2047, plot=True):
    """
    pulse_durations_us and amplitudes must be iterable, with same length = num_pulses
    amplitudes are each multiplied by dac_max 
    returns comma separated waveform string. 
    """
    
    num_pulses = len(pulse_durations_us)
    if num_pulses != len(amplitudes):
        print(num_pulses)
        print(len(amplitudes))
        return "pulse_durations_us and amplitudes must have same length"
    
    segment_duration_s = (wvfm_duration_s / num_pulses) 
    segment_samples = int(wvfm_samp_rate_Hz * segment_duration_s)  # Samples per segment
    #print(f"segment_samples = {segment_samples}")
    #print(f"segment duration (s)  = {segment_duration_s}")
    # Create the arbitrary waveform data
    waveform_data = []

    for idx in range(num_pulses):
        amplitude = amplitudes[idx]
        t_us = pulse_durations_us[idx]
        # Calculate the number of points for each square wave (high and low) given t_us
        t_samples = int(wvfm_samp_rate_Hz * (t_us * 1e-6))  # Convert t in microseconds to number of samples
        #print(f't_samples ={t_samples}')
        dac_amplitude = int(dac_max * amplitude)
        # Create one segment with alternating high and low values
        high_segment = [dac_amplitude] * t_samples  # High segment for duration t
        low_segment = [0] * (segment_samples-t_samples)  # Low segment for duration t

        # Combine to form a complete segment
        segment = high_segment + low_segment
        segment = segment[:segment_samples]  # Ensure the segment is the correct length
        # Add the segment to the overall waveform
        waveform_data.extend(segment)

    # Ensure total waveform length 
    #waveform_data = waveform_data[:wvfm_samples] + [0] * (wvfm_samples - len(waveform_data))

    # Convert the waveform data to a comma-separated string
    waveform_string = ','.join(f'{point:.5f}' for point in waveform_data)
    if plot == True: 
        print(np.shape(t_samples))
    plt.plot(np.arange(len(waveform_data))/wvfm_samp_rate_Hz, waveform_data, 'C0.-')
    plt.xlabel("time (s)")
    plt.ylabel("DAC")
    plt.show()

    return waveform_string
