#!/usr/bin/env python

#from contextlib import contextmanager
#import glob
import os
#import threading
#import time
#import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pysmurf.client.base import SmurfBase
#from pysmurf.client.command.sync_group import SyncGroup as SyncGroup
#from pysmurf.client.util.SmurfFileReader import SmurfStreamReader
#from pysmurf.client.util.pub import set_action

class SmurfTkidMixin(SmurfBase):

    def heater_swipe_NEP_tkid(self, band, bias_group,
                              tone_amplitude_array=[8,10,12],
                              dc_power_array=np.arange(15)/9.,
                              ac_tone_amp=0.07):
        '''
        Function for exracting the NEP of the TKIDs.

        Args
        ----------
        band : int
            The band to get the NEP of.
        bias_group : int
            The bias group.
        tone_amplitude_array : int array
            Specifies the tone powers to loop through.
        grad_cut: float
            The value of the gradient of phase to look for resonances.

        Returns
        ----------
        NEP: dict
            Nested dictionary with keys being the resonator numbers, tone
            amplitudes, and heater powers, and
            the values being the NEP in pW.
        '''
        # Estimate phase delay (in seconds)
        delay = self.estimate_phase_delay(band, show_plot=False)[0] * 1e-6
        print('delay (in seconds): ')
        print(delay)

        NEP = defaultdict(lambda: defaultdict(dict))
        # Loop through different gain values
        for gain in tone_amplitude_array:
            # Loop through the power on the heaters
            for dc in dc_power_array:
                # Set DC heater power, no square wave
                self.set_tes_bias_bipolar(bias_group, dc)

                # Find frequencies and plot output
                freq, resp = self.find_freq(band, start_freq=-250, stop_freq=250,
                                            tone_power=gain,
                                            make_plot=True, save_plot=True)

                # Get higher signal-to-noise of the resonator dip
                self.setup_notches(band, tone_power=gain, new_master_assignment=True)

                self.plot_tune_summary(band, eta_scan=True)
                self.run_serial_gradient_descent(band)
                self.run_serial_eta_scan(band)
                # Turn on slow tracking
                self.set_feedback_enable(band=band, val=1)

                # TODO: Confirm that we don't need the vna_fit stuff; find_freqs does everything?

                # TODO: Guard tone stuff
                # TODO: Eventually want to use self.take_stream_data() and use phase info
                keys = self.freq_resp[band]['resonances'].keys()
                f_dc = {}
                for k in keys:
                    r = self.freq_resp[band]['resonances'][k]
                    channel = r['channel']
                    res_freq = r['freq'] * 1.0e6
                    #freq_k = r['freq_eta_scan']
                    resp_k = r['resp_eta_scan']
                    res_num = k
                    eta = r['eta']
                    eta_mag = r['eta_mag']
                    eta_phase_deg = r['eta_phase']
                    timestamp = self.get_timestamp()

                    # Plot IQ circle from setup_notches
                    I = np.real(resp_k)
                    Q = np.imag(resp_k)
                    plt.ioff()
                    plt.figure(figsize=(9,4.5))
                    plt.axhline(0, color='k', linestyle=':', alpha=.5)
                    plt.axvline(0, color='k', linestyle=':', alpha=.5)
                    plt.plot(I, Q, 'ro', markersize=3, label='I/Q from setup_notches (before eta scaling)')
                    label = r'$\eta/\eta_{mag}$' + \
                            f': {np.real(eta/eta_mag):4.3f}' + \
                            f'+{np.imag(eta/eta_mag):4.3f}\n'
                    label = label + r'$\eta_{mag}$' + f': {eta_mag:1.3e}' + '\n'
                    label = label + r'$\eta_{ang}$' + f': {eta_phase_deg:3.2f}' + '\n'
                    bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
                    ax = plt.gca()
                    plt.text(.03, .75, label, transform=ax.transAxes, fontsize=10,
                             bbox=bbox)
                    respp = eta/eta_mag * resp_k
                    Ip = np.real(respp)
                    Qp = np.imag(respp)
                    plt.plot(Ip, Qp, 'bo', markersize=3, label='I/Q from setup_notches (after eta scaling)')

                    # Take noise
                    self.band_off(band)
                    self.set_fixed_tone(freq_mhz=res_freq/1.0e6, tone_power=gain)
                    I_debug, Q_debug, sync = self.take_debug_data(band=band, channel=channel,
                                                                  single_channel_readout=2, rf_iq=True)

                    # Filter to consider only 0.1 MHz around each noise acquisition
                    channel_freq = self.get_channel_frequency_mhz(band) * 1.0e6  # Sampling frequency
                    sos = signal.butter(N=4, Wn=0.1e6, btype='lowpass', fs=channel_freq, output='sos')
                    I_debug = signal.sosfiltfilt(sos, I_debug)
                    Q_debug = signal.sosfiltfilt(sos, Q_debug)

                    # Scale by eta
                    # TODO: This 0.6 scaling on the I, Q from take_debug_data is a bug in take_debug_data; remove when fixed
                    #resp_debug = I_debug + 1j*Q_debug
                    resp_debug = I_debug/0.6 - 1j*Q_debug/0.6
                    resp_debug_eta = resp_debug * eta/eta_mag
                    I_debug_eta = np.real(resp_debug_eta)
                    Q_debug_eta = np.imag(resp_debug_eta)

                    # Plot output of take_debug_data
                    # TODO: The 0.6 scaling is here too
                    plt.plot(I_debug/0.6, -1*Q_debug/0.6, 'yx', markersize=3, label='Noise (before eta scaling)')
                    plt.plot(I_debug_eta, Q_debug_eta, 'gx', markersize=3, label='Noise (after eta scaling)')
                    plt.xlabel('I')
                    plt.ylabel('Q')
                    plt.title(f'Channel {channel}\n(considering only 0.1 MHz around each noise acquisition)')
                    plt.legend(loc='upper right')

                    # Save the diagnostic plot
                    save_name = f'{timestamp}_eta_b{band}_res{res_num:03}_dc{dc}_tone{gain}.png'
                    path = os.path.join(self.plot_dir, save_name)
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()

                    # We need this later for the NEP calculation at the end
                    self.set_feedback_enable_channel(band, channel, 1) # The noise taking turns off feedback, so turn back on
                    f, df, sync = self.take_debug_data(band, channel, IQstream=0, single_channel_readout=2)
                    f_dc[k] = f

                # Set square wave
                self.play_square_tes(bias_group, tone_amp=ac_tone_amp, tone_freq=1, dc_amp=dc)

                # Take noise data, same as above
                # TODO: Guard tone stuff
                # TODO: Eventually want to use self.take_stream_data() and use phase info
                for k in keys:
                    r = self.freq_resp[band]['resonances'][k]
                    channel = r['channel']
                    res_freq = r['freq'] * 1.0e6
                    #freq_k = r['freq_eta_scan']
                    resp_k = r['resp_eta_scan']
                    res_num = k
                    eta = r['eta']
                    eta_mag = r['eta_mag']
                    eta_phase_deg = r['eta_phase']
                    timestamp = self.get_timestamp()

                    # Plot IQ circle from setup_notches
                    I = np.real(resp_k)
                    Q = np.imag(resp_k)
                    plt.ioff()
                    plt.figure(figsize=(9,4.5))
                    plt.axhline(0, color='k', linestyle=':', alpha=.5)
                    plt.axvline(0, color='k', linestyle=':', alpha=.5)
                    plt.plot(I, Q, 'ro', markersize=3, label='I/Q from setup_notches (before eta scaling)')
                    label = r'$\eta/\eta_{mag}$' + \
                            f': {np.real(eta/eta_mag):4.3f}' + \
                            f'+{np.imag(eta/eta_mag):4.3f}\n'
                    label = label + r'$\eta_{mag}$' + f': {eta_mag:1.3e}' + '\n'
                    label = label + r'$\eta_{ang}$' + f': {eta_phase_deg:3.2f}' + '\n'
                    bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
                    ax = plt.gca()
                    plt.text(.03, .75, label, transform=ax.transAxes, fontsize=10,
                             bbox=bbox)
                    respp = eta/eta_mag * resp_k
                    Ip = np.real(respp)
                    Qp = np.imag(respp)
                    plt.plot(Ip, Qp, 'bo', markersize=3, label='I/Q from setup_notches (after eta scaling)')

                    # Take noise
                    self.band_off(band)
                    self.set_fixed_tone(freq_mhz=res_freq/1.0e6, tone_power=gain)
                    I_debug, Q_debug, sync = self.take_debug_data(band=band, channel=channel,
                                                                  single_channel_readout=2, rf_iq=True)

                    # Filter to consider only 0.1 MHz around each noise acquisition
                    channel_freq = self.get_channel_frequency_mhz(band) * 1.0e6  # Sampling frequency
                    sos = signal.butter(N=4, Wn=0.1e6, btype='lowpass', fs=channel_freq, output='sos')
                    I_debug = signal.sosfiltfilt(sos, I_debug)
                    Q_debug = signal.sosfiltfilt(sos, Q_debug)

                    # Scale by eta
                    # TODO: This 0.6 scaling on the I, Q from take_debug_data is a bug in take_debug_data; remove when fixed
                    #resp_debug = I_debug + 1j*Q_debug
                    resp_debug = I_debug/0.6 - 1j*Q_debug/0.6
                    resp_debug_eta = resp_debug * eta/eta_mag
                    I_debug_eta = np.real(resp_debug_eta)
                    Q_debug_eta = np.imag(resp_debug_eta)

                    # Plot output of take_debug_data
                    # TODO: The 0.6 scaling is here too
                    plt.plot(I_debug/0.6, -1*Q_debug/0.6, 'yx', markersize=3, label='Noise (before eta scaling)')
                    plt.plot(I_debug_eta, Q_debug_eta, 'gx', markersize=3, label='Noise (after eta scaling)')
                    plt.xlabel('I')
                    plt.ylabel('Q')
                    plt.title(f'Channel {channel}\n(considering only 0.1 MHz around each noise acquisition)')
                    plt.legend(loc='upper right')

                    # Save the diagnostic plot
                    save_name = f'{timestamp}_eta_b{band}_res{res_num:03}_ac{dc}_tone{gain}.png'
                    path = os.path.join(self.plot_dir, save_name)
                    plt.savefig(path, bbox_inches='tight')
                    plt.close()

                    # Responsivity calculation
                    heater_dP = ((dc+ac_tone_amp)**2/0.35 - (dc-ac_tone_amp)**2/0.35) * 1e12 # In pW; gold resistors on-chip are usually ~0.35 Ohms
                    self.set_feedback_enable_channel(band, channel, 1)
                    f, df, sync = self.take_debug_data(band, channel, IQstream=0, single_channel_readout=2) # single_channel_readout specifies data rate
                    # Frequency response difference for the top of the square wave vs bottom
                    diff = np.mean(f[f > np.mean(f)]) - np.mean(f[f < np.mean(f)])
                    responsivity = diff/heater_dP # Hz/pW
                    # Get NEP
                    NEP[k][gain][dc] = np.std(f_dc[k])/responsivity # np.std(f_dc[k]) is the noise, responsivity converts to NEP

        # Convert from defaultdict object to dict
        NEP = {k:dict(v) for k, v in NEP.items()}
        return NEP
