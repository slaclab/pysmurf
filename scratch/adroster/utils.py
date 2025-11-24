#!/usr/bin/env python
# coding: utf-8

# In[4]:


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import time
import pickle as pkl
import os
import sys
import numpy as np
from matplotlib import cm
from scipy.signal import butter, welch, filtfilt, periodogram, savgol_filter
import pandas as pd
import glob
from scipy.optimize import curve_fit
from scipy.optimize import fminbound
import scipy.linalg as linalg
import scipy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import butter, welch, filtfilt, periodogram, savgol_filter


# In[2]:


def welch_IQ(iq, fs,  plot=True, welch_nperseg=2**18, title='', show_plot=True):
    i = iq.real
    q = iq.imag
    ffi, pxxi = welch(i,fs=fs, nperseg=welch_nperseg)
    ffq, pxxq = welch(q,fs=fs, nperseg=welch_nperseg)

    # scale to dBc/Hz by the voltage magnitude
    magfac = np.mean(q)**2 + np.mean(i)**2
    pxxi_dbc = 10. * np.log10(pxxi / magfac)
    pxxq_dbc = 10. * np.log10(pxxq / magfac)


    if plot:
        plt.gca().semilogx(ffi,pxxi_dbc,linestyle='-',label=f'i {title}')
        plt.gca().semilogx(ffq,pxxq_dbc, linestyle='--',label=f'q {title}')
        plt.ylabel('dBc/Hz')
        plt.xlabel('Frequency (Hz)')
        plt.title(title)
        plt.legend(loc='lower left')
        if show_plot: plt.show()
    return


# In[3]:


def freqSweep(S, band, start_freq=-250, stop_freq=250, subband=None, tone_power=None, n_avg=1, makeplot=False):
    """
    Inputs:
    S (obj): PySMuRF control instance
    start_freq (int): start frequency of sweep in the IF [MHz]
    stop_freq (int): stop frequency of sweep in the IF [MHz]
    subband:
    tone_power (int): int between 0 and 15. Each integer is 0.5 dBm of power. Not much dynamic range here. Defaults to 12 due to config file.
    n_avg (int): nubmer of times to average. this is a sweep-to-sweep average, analogous to averaging on a VNA
    makeplot (bool): 

    Outputs:
    freqs (ndarray): real-valued 1D array of frequenies (RF, not IF)
    resp_avg (ndarray): complex-valued 1D array of response
    """
    #adapted from S.find_freq()
    band_center = S.get_band_center_mhz(band)
    if subband is None:
        start_subband = S.freq_to_subband(band, band_center + start_freq)[0]
        stop_subband = S.freq_to_subband(band, band_center + stop_freq)[0]
        step = 1
        if stop_subband < start_subband:
            step = -1
        subband = np.arange(start_subband, stop_subband+1, step)
    else:
        sb, sbc = S.get_subband_centers(band)
        start_freq = sbc[subband[0]]
        stop_freq  = sbc[subband[-1]]

    # Turn off all tones in this band first.  May want to make
    # this only turn off tones in each sub-band before sweeping,
    # instead?
    S.band_off(band)

    if tone_power is None:
        tone_power = S._amplitude_scale[band]
        S.log('No tone_power given. Using value in config ' +
                 f'file: {tone_power}')

    #the following for loop does an amplitude sweep n_avg times, reads the response, and addes the response array to resp_tot
    #resp_tot has shape (n_avg, 31*n_channels). n_channels is the number of processed channels + 1; there are 31 frequencies per channel.
    S.log(f'Sweeping across frequencies {start_freq + band_center}MHz to {stop_freq + band_center}MHz. Averaging {n_avg} traces.')
    BW_MHz = 500
    freqs_per_channel = 31
    n_channels = int(np.ceil((S.get_number_processed_channels(band))*(stop_freq - start_freq)/BW_MHz))
    resp_shape = (n_avg, freqs_per_channel*n_channels)
    resp_arr = np.zeros(resp_shape, dtype = np.complex_)
    for n in range(n_avg):
        f, resp = S.full_band_ampl_sweep(band, subband, tone_power, n_read=2) #n_read doesn't actually do anything; must be legacy code from previous SMuRF implementations
        
        #below is adapted from S.find_all_peak()
        # Stack all the frequency and response data into a
        # Frequency array is the same on every iteration, so only need to find & sort frequencies once
        if n == 0:
            sb, _ = np.where(f !=0)
            idx = np.unique(sb)
            f_stack = np.ravel(f[idx])
            # Frequency is interleaved, so sort it
            s = np.argsort(f_stack)
            f_stack = f_stack[s]
            freqs = f_stack

            # convert frequencies from IF to the RF
            if band is not None:
                bandCenterMHz = S.get_band_center_mhz(band)
                scale = 1
                if np.max(f_stack) > 1.0E8:
                    self.log('Frequency is probably in Hz. Converting to MHz')
                    scale = 1.0E-6
                freq_rf = bandCenterMHz - freqs*scale
            else:
                freq_rf = bandCenterMHz - freqs
            
        r_stack = np.ravel(resp[idx])
    
        # response is also interleaved, so sort it
        r_stack = r_stack[s]

        #append the sorted & stacked response array to resp_arr
        resp_arr[n,:] = r_stack

    #Frequencies are just f_stack; Average the response

    resp_avg = np.mean(resp_arr, axis=0)
    
    # Break apart the data to find phase, deriv of phase
    angle = np.unwrap(np.angle(resp_avg))
    x = np.arange(len(angle))
    p1 = np.poly1d(np.polyfit(x, angle, 1))
    angle -= p1(x)
    grad_kernel_width = 8
    grad = np.convolve(angle, np.repeat([1,-1], grad_kernel_width), mode='same')
    #grad_kernel_width is number of samples to take after a point to calculate gradient; default is 8

    #find the amplitude
    amp = np.abs(resp_avg)
    
    #calculating rolling median. Adapted from S.find_peaks()   
    # Calculate the rolling median. This uses pandas. This is analogous to smoothing on a VNA.
    import pandas as pd
    med_amp = pd.Series(amp).rolling(window=50, center=True, min_periods=1).median()

    # convert frequencies from IF to the RF
    if band is not None:
        bandCenterMHz = S.get_band_center_mhz(band)
        scale = 1
        if np.max(f_stack) > 1.0E8:
            self.log('Frequency is probably in Hz. Converting to MHz')
            scale = 1.0E-6
        freq_rf = bandCenterMHz - freqs*scale
    else:
        freq_rf = bandCenterMHz - freqs
    
    #plotting
    if makeplot:
        plt.ion()

        fig, ax = plt.subplots(2, figsize=(8,6), sharex=True)

        # Plot response
        ax[0].plot(freq_rf, amp)
        ax[0].plot(freq_rf, med_amp)

        ax[1].plot(freq_rf, grad)
        ax[1].set_ylim(-2, 20)

        ax[0].set_ylabel('Amp.')
        ax[1].set_ylabel('Deriv Phase')
        ax[1].set_xlabel('Freq. [MHz]')

        # Text label
        text = ''
        if band is not None:
            text += f'Band: {band}' + '\n'
            text += f'Center Freq: {bandCenterMHz} MHz' + '\n'
        ax[0].text(.025, .975, text, transform=ax[0].transAxes, ha='left',
            va='top')

    
    return freq_rf, resp_avg


# In[4]:


# find the frequency between parity bands
def lorentzian(f, f0, Gc, Gr, g, Omega):
    Delta = 2*np.pi*(f - f0)
    S21 = 1 - (Gc/2/g)*(1-Delta*1j/g)/(1+(Delta/g)**2+(2*np.pi*Omega)**2/g/Gr)
    return S21
    
def invertedDoubleLorentzian(f, f0_1, f0_2, Gc_1, GC_2, Gr_1, Gr_2, g_1, g_2, Omega_1, Omega_2, baseline, scale):
    '''
    Eq. E22 in the design paper
    '''
    lorentzian1 = lorentzian(f, f0_1, Gc_1, Gr_1, g_1, Omega_1)
    lorentzian2 = lorentzian(f, f0_2, Gc_2, Gr_2, g_2, Omega_2)
    S21 = baseline + scale*(lorentzian1 + lorentzian2)
    return np.abs(S21)

def negInvertedDoubleLorentzian(f, f0_1, f0_2, Gc_1, GC_2, Gr_1, Gr_2, g_1, g_2, Omega_1, Omega_2, baseline, scale):
    '''
    Eq. E22 in the design paper
    '''
    lorentzian1 = lorentzian(f, f0_1, Gc_1, Gr_1, g_1, Omega_1)
    lorentzian2 = lorentzian(f, f0_2, Gc_2, Gr_2, g_2, Omega_2)
    S21 = baseline + scale*(lorentzian1 + lorentzian2)
    return -np.abs(S21)


# In[5]:


def takeDebugData(S, band, channel, nsamp, plot=True, welch_nperseg=2**18, show_plot=True, channel_mode=0):
    timestamp = S.get_timestamp() 
    filename = f'{timestamp}_single_channel_b{band}ch{channel:03}'
    i,q,sync = S.take_debug_data(band=band,channel=channel,rf_iq=True,nsamp=nsamp,filename=filename, single_channel_readout=channel_mode) 
    i = i / (1.2)
    q = q / (-1.2)
    iq = i + 1j * q
    if plot: 
        fs = S.get_channel_frequency_mhz(band) * 1.0E6
        welch_IQ(iq, fs, welch_nperseg, title=filename, show_plot=show_plot)
        ### save plots 
        fig = plt.gcf()
        #plt.savefig(f'/data/smurf_data/mkid_1tone_streaming_metadata/_Figs/{filename}_psd', fmt='tiff')
        plt.show()
    return iq, sync, filename


# In[6]:


def butterworth_filter(data, fs, cutoff_hz=200, order=1):
    b, a = butter(N=order, Wn=cutoff_hz, btype='low', fs=fs)
    zi = sp.signal.lfilter_zi(b, a)
    filt_data, _ = sp.signal.lfilter(b, a, data, zi=zi*data[0])
    return filt_data


# In[ ]:




