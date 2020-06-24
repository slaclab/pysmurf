import pysmurf.client
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import scipy.signal as signal
from scipy.optimize import curve_fit

epics_prefix="smurf_server_s5"
config_file = '/data/pysmurf_cfg/experiment_fp30_cc02-03_lbOnlyBay0.cfg' 
S = pysmurf.client.SmurfControl(epics_root=epics_prefix,
                                cfg_file=config_file, setup=False,
                                make_logfile=False, shelf_manager="shm-smrf-sp01")

# Parameters
band = 2
bias_group = 2
bias_high = .005
freq_fit_min = 1
freq_fit_max = 400

_, dac_pos, dac_neg = np.ravel(S.bias_group_to_pair[np.where(S.bias_group_to_pair[:,0]==bias_group)])

# Set to high current to bypass TES filter
S.set_tes_bias_high_current(bias_group)
S.set_tes_bias_bipolar(bias_group, 0)

# Set the downsample factor to 1
S.set_downsample_factor(1)
fs = S.get_flux_ramp_freq()*1.0E3
S.set_downsample_filter(4, fs/4)
S.set_filter_disable(True)

filename = S.stream_data_on(make_freq_mask=False)
#for i in np.arange(2):
#    time.sleep(1)
#    S.set_rtm_slow_dac_volt(dac_pos, bias_high)
#    time.sleep(1)
#    S.set_rtm_slow_dac_volt(dac_pos, 0)
#time.sleep(1)
time.sleep(2)
S.set_rtm_slow_dac_volt(dac_pos, bias_high)
time.sleep(2)
S.stream_data_off(register_file=True)
S.set_rtm_slow_dac_volt(dac_pos, 0)

# Load data
t, d, m, h = S.read_stream_data(filename,
                                return_tes_bias=True)

d *= S.pA_per_phi0 / 2 / np.pi
h *= 2 * S._rtm_slow_dac_bit_to_volt / S.bias_line_resistance * 1.0E12 * S.high_low_current_ratio

nperseg = 2**15

t = np.arange(len(h[bias_group]))/fs
freq, ph = signal.welch(h[bias_group], fs=fs, nperseg=nperseg)
freq_fit_idx = np.where(np.logical_and(freq>freq_fit_min,
                                        freq<freq_fit_max))
freq_fit = freq[freq_fit_idx]

def transfer_model(freq, cutoff, gain):
    order = 1
    b, a = signal.butter(order, cutoff, 
                         btype='lowpass', fs=fs)
    _, hh = signal.freqz(b, a, worN=freq, fs=fs)
    hh *= gain
    return np.abs(hh)
    
channel = S.which_on(band)
for ch in channel:
    fig, ax = plt.subplots(3, figsize=(5,10))
    idx = m[band, ch]
    
    ax[0].plot(t, d[idx]-np.median(d[idx]), label='TES signal')
    ax[0].plot(t, h[bias_group], label='bias')
    ax[0].set_xlabel('Time [s]')
    ax[0].set_ylabel('Resp [pA]')

    ax[0].legend(loc='upper right')

    # Cross spectral density
    _, pd = signal.csd(d[idx], h[bias_group], nperseg=nperseg, fs=fs)
    
    H = pd/ph
    H_fit = H[freq_fit_idx]
    
    popt, pcov = curve_fit(transfer_model, freq_fit, np.abs(H_fit), p0=[30, 1],
                           bounds=([0,0],[fs/2, 10]))
    print(popt)

    H_model = transfer_model(freq, *popt)
    
    ax[1].loglog(freq, H)
    ax[1].plot(freq, H_model)
    ax[1].set_ylim((.01, 1.1))
    ax[2].semilogx(freq, np.unwrap(np.angle(H)))

    fig.suptitle(f'ch {ch:03}')
    
    plt.tight_layout()
