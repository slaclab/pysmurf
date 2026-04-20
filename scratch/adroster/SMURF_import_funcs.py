import os
import sys
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as cbar

import pyximport
pyximport.install()

from smurf_data_loader2 import *

def read_file(filename):
    '''
    Input: filename (str)
    
    Returns:
    d: data array (np.array)
    h: header array (dict)
    m: metadata (dict)
    '''
    
    d, h, m = read_smurf_data_array(filename)
    return d, h, m

def get_sample_rate(header):
    '''
    The timestamps in a smurf data header are not really accurate since it's not on a clock, but they're approximately accurate
    '''
    times_ns = header['timestamp']
    fs = 1/(np.mean(np.gradient(times_ns)) * 1e-9)
    fs = round(fs, 1) #sample rate in Hz
    return fs

def match_times(data, header):
    ## Find indices that best match times
    ## returns time in seconds
    times_ns = header['timestamp']
    
    start_time_s = 0
    stop_time_s = (times_ns[-1] - times_ns[0]) * 1e9
    
    fs = get_sample_rate(header)
    times = np.arange(data.shape[0]) / fs
    
    #start_idx = np.argmin(np.abs(times - start_time_s))
    #stop_idx = np.argmin(np.abs(times - stop_time_s))
    return times


def reformat_data(data):
    '''
    data comes out of read_smurf_data in a numpy array with format (ch1_I, ch1_Q, ch2_I, ch2_Q, ...).
    We want it in the form of amp*angle. mag*phase

    Returns: data_reformated (dict). Keys are channel_num, values are data_dict (dict). data_dict (dict) has keys 'amps' with values amplitudes
    (np.array) and key 'phases' with values phases (np.array). So like

    {channel1: {'amps': np.array([...]), 'phases': np.array([...])}, channel2: {'amps': np.array([...]), 'phases': np.array([...])}, channel3: ...}
    '''
    n_rows = data.shape[0]
    n_cols = data.shape[1]

    data_reformated = {}
    data_keys = ['amps', 'phases']
    
    for i in range(int(n_cols/2)):
        channel_num = i+1 #indexing channel numbers starting at 1, bc that makes sense to me rn
        channel_key = 'channel'+str(channel_num)
                
        I = data[:, 2*i]
        Q = data[:, 2*i+1]
        z = I + Q*1j   

        #smurf outputs raw adc data, so amp >> 1 (often like 10,000). This messes up data processing. Let's rescale using
        #a typical SQUAT attenuation of 70 dB.
        rescale_factor_dB = -70
        rescale_factor = 10**(rescale_factor_dB/10)
        
        amp = np.abs(z) * rescale_factor
        phase = np.angle(z)

        data_dict = dict(zip(data_keys, [amp, phase]))
        data_reformated[channel_key] = data_dict

    return data_reformated
    
    
