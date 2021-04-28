#!/usr/bin/env python

#from contextlib import contextmanager
#import glob
import os
#import threading
#import time
#import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pysmurf.client.base import SmurfBase
#from pysmurf.client.command.sync_group import SyncGroup as SyncGroup
#from pysmurf.client.util.SmurfFileReader import SmurfStreamReader
#from pysmurf.client.util.pub import set_action

class SmurfTkidMixin(SmurfBase):

    def heater_swipe_NEP_tkid(self, band, bias_group, n_detectors,
                              tone_amplitude_array=[8,10,12,14], grad_cut=5):
        '''
        Function for exracting the NEP of the TKIDs.

        Args
        -----
        band : int
            The band to get the NEP of.
        bias_group : int
            The bias group.
        n_detectors : int
            The number of detectors.
        tone_amplitude_array : int array
            Specifies the tone powers to loop through.
        grad_cut: float
            The value of the gradient of phase to look for resonances.
        '''
        # Estimate phase delay (in seconds)
        delay = self.estimate_phase_delay(band, show_plot=False)[0] * 1e-6
        print('delay (in seconds): ')
        print(delay)

        # Loop through different gain values
        for gain in tone_amplitude_array:
            # Loop through the power on the heaters
            for dc in np.arange(15)/9.:
                # Set DC heater power, no square wave
                self.set_tes_bias_bipolar(bias_group, dc)

                # Find frequencies and plot output
                freq, resp = self.find_freq(band, start_freq=-250, stop_freq=250,
                                            grad_cut=grad_cut,
                                            tone_power=gain,
                                            make_plot=True, save_plot=True)

                # Get higher signal-to-noise of the resonator dip
                self.setup_notches(band, tone_power=gain)

                # TODO: Should we make sure we found the same number of peaks (len(keys)) as n_detectors?

                # TODO: Confirm that we don't need the vna_fit stuff; find_freqs does everything?

                # TODO: Guard tone stuff
                # TODO: Eventually want to use self.take_stream_data() and use phase info
                keys = self.freq_resp[band]['resonances'].keys()
                for k in keys:
                    r = self.freq_resp[band]['resonances'][k]
                    channel = r['channel']
                    res_freq = r['freq'] * 1e6
                    I_debug, Q_debug, sync = self.take_debug_data(band=band, channel=channel,
                                                                  rf_iq=True)
                    # Filter to consider only 0.1 MHz around each noise acquisition
                    channel_freq = self.get_channel_frequency_mhz(band) * 1.0e6  # Sampling frequency
                    nyq = 0.5 * channel_freq
                    low = (res_freq-0.05e6) / nyq
                    high = (res_freq+0.05e6) / nyq
                    b, a = signal.butter(N=4, Wn=[low,high], btype='bandpass', fs=channel_freq)
                    I_debug = signal.filtfilt(b, a, I_debug, method='gust')
                    Q_debug = signal.filtfilt(b, a, Q_debug, method='gust')
                    # Scale by eta
                    eta = r['eta']
                    eta_mag = r['eta_mag']
                    eta_phase_deg = r['eta_phase']
                    resp_debug = I_debug + 1j*Q_debug
                    resp_debug_eta = resp_debug * eta_mag * np.exp(1j * eta_phase_deg * 2*np.pi/360)
                    I_debug = np.real(resp_debug_eta)
                    Q_debug = np.imag(resp_debug_eta)
                    # Plot output of take_debug_data
                    plt.figure(figsize=(9,4.5))
                    plt.axhline(0, color='k', linestyle=':', alpha=.5)
                    plt.axvline(0, color='k', linestyle=':', alpha=.5)
                    plt.scatter(I_debug, Q_debug, color='b', s=3, label='Noise')
                    # Then plot I/Q circle from setup_notches
                    freq_k = r['freq_eta_scan']
                    resp_k = r['resp_eta_scan']
                    res_num = k
                    timestamp = self.get_timestamp()
                    #plt.ioff() # TODO: Uncomment after testing
                    I = np.real(resp_k)
                    Q = np.imag(resp_k)
                    plt.scatter(I, Q, color='r', s=3, label='IQ from setup_notches (before multiplying by eta)')
                    label = r'$\eta/\eta_{mag}$' + \
                            f': {np.real(eta/eta_mag):4.3f}' + \
                            f'+{np.imag(eta/eta_mag):4.3f}\n'
                    label = label + r'$\eta_{mag}$' + f': {eta_mag:1.3e}' + '\n'
                    label = label + r'$\eta_{ang}$' + f': {eta_phase_deg:3.2f}' + '\n'
                    bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
                    ax = plt.gca()
                    plt.text(.03, .81, label, transform=ax.transAxes, fontsize=10,
                             bbox=bbox)
                    eta = eta/eta_mag
                    respp = eta * resp_k
                    Ip = np.real(respp)
                    Qp = np.imag(respp)
                    plt.scatter(Ip, Qp, c='g', s=3, label='IQ from setup_notches (after multiplying by eta)')
                    plt.xlabel('I')
                    plt.ylabel('Q')
                    plt.title(f'Channel {channel}\n(considering only 0.1 MHz around each noise acquisition)')
                    plt.legend()

                    # Save the diagnostic plot
                    save_name = f'{timestamp}_eta_b{band}_res{res_num:03}_dc.png'
                    path = os.path.join(self.plot_dir, save_name)
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()

                    # Set square wave
                    self.play_square_tes(bias_group, tone_amp=0.07, tone_freq=1, dc_amp=dc)

                    # Take noise data # TODO: Basically the same thing as before, if it works
