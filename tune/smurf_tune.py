import numpy as np
import os
import time
from pysmurf.base import SmurfBase
from scipy import optimize
import scipy.signal as signal
from collections import Counter
from ..util import tools
from pysmurf.command.sync_group import SyncGroup as SyncGroup

class SmurfTuneMixin(SmurfBase):
    """
    This contains all the tuning scripts
    """

    def tune_band(self, band, freq=None, resp=None, n_samples=2**19, 
        make_plot=False, plot_chans = [], save_plot=True, save_data=True, 
        make_subband_plot=False, subband=None, n_scan=5,
        subband_plot_with_slow=False,
        grad_cut=.05, freq_min=-2.5E8, freq_max=2.5E8, amp_cut=1,
        use_slow_eta=False):
        """
        This does the full_band_resp, which takes the raw resonance data.
        It then finds the where the reseonances are. Using the resonance
        locations, it calculates the eta parameters.

        Args:
        -----
        band (int): The band to tune

        Opt Args:
        ---------
        freq (float array): The frequency information. If both freq and resp
            are not None, it will skip full_band_resp.
        resp (float array): The response information. If both freq and resp
            are not None, it will skip full_band_resp.
        n_samples (int): The number of samples to take in full_band_resp.
            Default is 2^19.
        make_plot (bool): Whether to make plots. This is slow, so if you want
            to tune quickly, set to False. Default True.
        plot_chans (list): if making plots, which channels to plot. If empty,
	       will just plot all of them
        save_plot (bool): Whether to save the plot. If True, it will close the
            plots before they are shown. If False, plots will be brought to the
            screen.
        save_data (bool): If True, saves the data to disk.
        grad_cut (float): The value of the gradient of phase to look for 
            resonances. Default is .05
        amp_cut (float): The distance from the median value to decide whether
            there is a resonance. Default is .25.
        freq_min (float): The minimum frequency relative to the center of
            the band to look for resonances. Units of Hz. Defaults is -2.5E8
        freq_max (float): The maximum frequency relative to the center of
            the band to look for resonances. Units of Hz. Defaults is 2.5E8

        Returns:
        --------
        res (dict): A dictionary with resonance frequency, eta, eta_phase,
            R^2, and amplitude.

        """
        timestamp = self.get_timestamp()

        if make_plot and save_plot:
            import matplotlib.pyplot as plt
            plt.ioff()

        if freq is None or resp is None:
            self.band_off(band)
            self.flux_ramp_off()
            self.log('Running full band resp')
            # Inject high amplitude noise with known waveform, measure it, and 
            # then find resonators and etaParameters from cross-correlation.
            freq, resp = self.full_band_resp(band, n_samples=n_samples,
                make_plot=make_plot, save_data=save_data, timestamp=timestamp,
                n_scan=n_scan)

            # Now let's scale/shift phase/mag to match what DSP sees

            # fit phase, calculate delay +/- 250MHz
            idx = np.where( (freq > freq_min) & (freq < freq_max) )

            p     = np.polyfit(freq[idx], np.unwrap(np.angle(resp[idx])), 1)
            delay = 1e6*np.abs(p[0]/(2*np.pi))

            # delay from signal being sent out, coming through the system, and then 
            # being read as data.  
            # empirical, may need to iterate on this **must be right** for tracking
            processing_delay  = 1.842391045639787 

            # DSP sees cable delay + processing delay 
            #   - refPhaseDelay/2.4 (2.4 MHz ticks) + ref_phase_delay_fine/307.2
            # calculate refPhaseDelay and refPhaseDelayFine
            ref_phase_delay = np.ceil( (delay + processing_delay) * 2.4 )
            ref_phase_delay_fine = np.floor( np.abs(delay + processing_delay - 
                ref_phase_delay/2.4) * 307.2 )

            comp_delay = (delay + processing_delay - ref_phase_delay/2.4 + 
                ref_phase_delay_fine/307.2)
            mag_scale = 5.*0.04232/0.1904    # empirical

            add_phase_slope = (2*np.pi*1e-6)*(delay - comp_delay)

            # scale magnitude
            mag_resp = np.abs(resp)
            comp_mag_resp = mag_scale*mag_resp

            # adjust slope of phase response
            # finally there may also be some overall phase shift (DC)

            #FIXME - want to match phase at a frequency where there is no resonator
            match_freq_offset = -0.8 # match phase at -0.8 MHz

            phase_resp = np.angle(resp)

            idx0  = np.abs(freq - match_freq_offset*1e6).argmin()
            tf_phase  = phase_resp[idx0] + freq[idx0]*add_phase_slope
            self.set_ref_phase_delay(band, int(ref_phase_delay))
            self.set_lms_delay(band, int(ref_phase_delay))
            self.set_ref_phase_delay_fine(band, int(ref_phase_delay_fine))
            
            self.set_eta_mag_scaled_channel(band, 0, 1)
            self.set_center_frequency_mhz_channel(band, 0, match_freq_offset)
            self.set_amplitude_scale_channel(band, 0, 10)
            self.set_eta_phase_degree_channel(band, 0, 0)
            dsp_I = [self.get_frequency_error_mhz(band, 0) for i in range(20)]
            self.set_eta_phase_degree_channel(band, 0, -90)
            dsp_Q = [self.get_frequency_error_mhz(band, 0) for i in range(20)]
            self.set_amplitude_scale_channel(band, 0, 0)
            dsp_phase = np.arctan2(np.mean(dsp_Q), np.mean(dsp_I)) 
            phase_shift = dsp_phase - tf_phase

            comp_phase_resp = phase_resp + freq*add_phase_slope + phase_shift

            # overall compensated response
            comp_resp = comp_mag_resp*(np.cos(comp_phase_resp) + \
                1j*np.sin(comp_phase_resp))

            resp = comp_resp

        # Find peaks
        peaks = self.find_peak(freq, resp, rolling_med=True, band=band, make_plot=make_plot, 
            save_plot=save_plot, grad_cut=grad_cut, freq_min=freq_min,
            freq_max=freq_max, amp_cut=amp_cut, 
            make_subband_plot=make_subband_plot, timestamp=timestamp,
            subband_plot_with_slow=subband_plot_with_slow, pad=50, min_gap=50)

        # Eta scans
        resonances = {}
        for i, p in enumerate(peaks):
            eta, eta_scaled, eta_phase_deg, r2, eta_mag, latency, Q= \
                self.eta_fit(freq, resp, p, 50E3, make_plot=make_plot, 
                plot_chans=plot_chans, save_plot=save_plot, res_num=i, 
                band=band, timestamp=timestamp, use_slow_eta=use_slow_eta)

            resonances[i] = {
                'freq': p,
                'eta': eta,
                'eta_scaled': eta_scaled,
                'eta_phase': eta_phase_deg,
                'r2': r2,
                'eta_mag': eta_mag,
                'latency': latency,
                'Q': Q
            }

        if save_data:
            self.log('Saving resonances to {}'.format(self.output_dir))
            np.save(os.path.join(self.output_dir, 
                '{}_b{}_resonances'.format(timestamp, band)), resonances)

        # Assign resonances to channels
        self.log('Assigning channels')
        f = [resonances[k]['freq']*1.0E-6 for k in resonances.keys()]
        subbands, channels, offsets = self.assign_channels(f, band=band)

        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})

        self.freq_resp[band] = resonances
        np.save(os.path.join(self.output_dir, 
            '{}_freq_resp'.format(timestamp)), self.freq_resp)

        self.relock(band)
        self.log('Done')
        return resonances

    def tune_band_quad(self, band, del_f=.01, n_samples=2**19,
        make_plot=False, plot_chans=[], save_plot=True, save_data=True,
        subband=None, n_scan=1, grad_cut=.05, freq_min=-2.5E8, 
        freq_max=2.5E8, amp_cut=1,  n_pts=20):
        """
        """
        timestamp = self.get_timestamp()

        if make_plot and save_plot:
            import matplotlib.pyplot as plt
            plt.ioff()

        self.band_off(band)
        self.flux_ramp_off()
        self.log('Running full band resp')


        # Find resonators with noise blast
        freq, resp = self.full_band_resp(band, n_samples=n_samples,
                                         make_plot=make_plot, save_data=save_data, 
                                         timestamp=timestamp, n_scan=n_scan)
        
        peaks = self.find_peak(freq, resp, band=band, make_plot=make_plot,
            save_plot=save_plot, grad_cut=grad_cut, freq_min=freq_min,
            freq_max=freq_max, amp_cut=amp_cut,
            make_subband_plot=make_subband_plot, timestamp=timestamp,
            subband_plot_with_slow=subband_plot_with_slow)

        # Assign resonances to channels                                                       
        resonances = {}
        self.log('Assigning channels')
        for i, p in enumerate(peaks):
            resonances[i] = {'peak' : p}

        subbands, channels, offsets = self.assign_channels(peaks*1.0E-6, band=band)

        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})

            # Fill with dummy values so relock does not fail
            resonances[k].update({'eta': 0})
            resonances[k].update({'eta_scaled': 0})
            resonances[k].update({'eta_phase': 0})
            resonances[k].update({'r2': 0})
            resonances[k].update({'eta_mag': 0})
            resonances[k].update({'latency': 0})
            resonances[k].update({'Q': 0})

        self.freq_resp[band] = resonances
        self.relock(band, check_vals=False)

        channels = self.which_on(band)

        feedback_array = np.zeros_like(self.get_feedback_enable_array(band))
        self.set_feedback_enable_array(band, feedback_array)

        eta_mag_array = np.zeros_like(self.get_eta_mag_array(band))
        eta_mag_array[channels] = 1
        self.set_eta_mag_array(band, eta_mag_array)

        self.log('Measuring eta_phase = 0')
        eta_phase_array = np.zeros_like(self.get_eta_phase_array(band))
        self.set_eta_phase_array(band, eta_phase_array)

        adc0 = np.zeros((n_pts, len(channels)))
        for i in np.arange(n_pts):
            adc0[i] = self.get_frequency_error_array(band)[channels]

        self.log('Measuring eta phase = 90')
        eta_phase_array[channels] = 90.
        self.set_eta_phase_array(band, eta_phase_array)
        
        adc90 = np.zeros_like(adc0)
        for i in np.arange(n_pts):
            adc90[i] = self.get_frequency_error_array(band)[channels]

        adc0_est = np.median(adc0, axis=0)
        adc90_est = np.median(adc90, axis=0)

        in_phase_rad = np.arctan2(adc90_est, adc0_est)
        quad_phase_rad = in_phase_rad + np.pi/2.
        in_phase_deg = np.rad2deg(in_phase_rad)
        quad_phase_deg = np.rad2deg(quad_phase_rad)

        self.log('In phase: {:4.2f}'.format(in_phase_deg) + \
            ' Quad phase: {:4.2f}'.format(quad_phase_deg))

        center_freq_array = self.get_center_frequency_array(band)

        eta_phase_array[channels] = [tools.limit_phase_deg(qpd) for qpd in quad_phase_deg]
        self.set_eta_phase_array(band, eta_phase_array)

        self.log('Measuring eta_mag')
        adc_plus = np.zeros((n_pts, len(channels)))
        adc_minus = np.zeros((n_pts, len(channels)))

        self.set_center_frequency_array(band, center_freq_array+del_f)
        for i in np.arange(n_pts):
            adc_plus[i] = self.get_frequency_error_array(band)[channels]

        self.set_center_frequency_array(band, center_freq_array-del_f)
        for i in np.arange(n_pts):
            adc_minus[i] = self.get_frequency_error_array(band)[channels]
        
        self.set_center_frequency_array(band, center_freq_array)
            
        adc_plus_est = np.median(adc_plus, axis=0)
        adc_minus_est = np.median(adc_minus, axis=0)
        
        dig_freq = self.get_digitizer_frequency_mhz(band)
        n_subband = self.get_number_sub_bands(band)
        sb_halfwidth = dig_freq / n_subband  # MHz

        eta_est = (2*del_f/(adc_plus_est - adc_minus_est))
        eta_mag = np.abs(eta_est)
        eta_scaled = eta_mag / sb_halfwidth

        eta_phase_array = self.get_eta_phase_array(band)
        eta_phase_array[channels]=[tools.limit_phase_deg(eP+180) if eE<0 \
            else eP for (eP,eE) in zip(eta_phase_array[channels], etaEst)]
        self.set_eta_phase_array(band, eta_phase_array)
        
                        
    def plot_tune_summary(self, band, eta_scan=False, show_plot=False,
                          save_plot=True):
        """
        Plots summary of tuning. Requires self.freq_resp to be filled.
        In other words, you must run find_freq and setup_notches
        before calling this function.
        
        Args:
        -----
        band (int): The band number to plot
        
        Opt Args:
        ---------
        eta_scan (bool) : Whether to also plot individual eta scans.
           Warning this is slow. Default is False.
        show_plot (bool) : Whether to display the plot. Default is False.
        save_plot (bool) : Whether to save the plot. Default is True.
        """
        import matplotlib.pyplot as plt
        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        resonances = self.freq_resp[band]['resonances']
        timestamp = self.freq_resp[band]['find_freq']['timestamp'][0]

        keys = resonances.keys()

        fig, ax = plt.subplots(2,2, figsize=(10,6))

        # Subband
        sb = np.array([resonances[k]['subband'] for k in keys])
        c = Counter(sb)
        y = np.array([c[i] for i in np.arange(128)])
        ax[0,0].plot(np.arange(128), y, '.', color='k')
        for i in np.arange(0, 128, 16):
            ax[0,0].axvspan(i-.5, i+7.5, color='k', alpha=.2)
        ax[0,0].set_ylim((-.2, np.max(y)+1.2))
        ax[0,0].set_yticks(np.arange(0,np.max(y)+.1))
        ax[0,0].set_xlim((0, 128))
        ax[0,0].set_xlabel('Subband')
        ax[0,0].set_ylabel('# Res')
        ax[0,0].text(.02, .92, 'Total: {}'.format(len(sb)),
                      fontsize=10, transform=ax[0,0].transAxes)


        eta = np.array([resonances[k]['eta'] for k in keys])
        f = np.array([resonances[k]['freq'] for k in keys])
        ax[0,1].plot(f, np.real(eta), '.', label='Real')
        ax[0,1].plot(f, np.imag(eta), '.', label='Imag')
        ax[0,1].plot(f, np.abs(eta), '.', label='Abs', color='k')
        ax[0,1].legend(loc='lower right')
        bc = self.get_band_center_mhz(band)
        ax[0,1].set_xlim((bc-250, bc+250))
        ax[0,1].set_xlabel('Freq [MHz]')
        ax[0,1].set_ylabel('Eta')
    

        phase = np.array([resonances[k]['eta_phase'] for k in keys])
        ax[1,1].plot(f, phase, color='k')
        ax[1,1].set_xlim((bc-250, bc+250))
        ax[1,1].set_ylim((-180,180))
        ax[1,1].set_yticks(np.arange(-180, 180.1, 90))
        ax[1,1].set_xlabel('Freq [MHz]')
        ax[1,1].set_ylabel('Eta phase')

        fig.suptitle('Band {} - {}'.format(band, timestamp))
        plt.subplots_adjust(left=.08, right=.95, top=.92, bottom=.08, 
                            wspace=.21, hspace=.21)

        if save_plot:
            save_name = '{}_tune_summary.png'.format(timestamp)
            plt.savefig(os.path.join(self.plot_dir, save_name),
                        bbox_inches='tight')
            if not show_plot:
                plt.close()

        if eta_scan:
            keys = self.freq_resp[band]['resonances'].keys()
            n_keys = len(keys)
            for k in keys:
                self.log('Eta plot {} of {}'.format(k+1, n_keys))
                r = self.freq_resp[band]['resonances'][k]
                self.plot_eta_fit(r['freq_eta_scan'], r['resp_eta_scan'],
                                  eta=r['eta'], eta_mag=r['eta_mag'],
                                  eta_phase_deg=r['eta_phase'],
                                  band=band, res_num=k,
                                  timestamp=timestamp, save_plot=save_plot,
                                  show_plot=show_plot, peak_freq=r['freq'])


    def full_band_resp(self, band, n_scan=1, n_samples=2**19, make_plot=False, 
        save_plot=True, save_data=False, timestamp=None, save_raw_data=False,
        correct_att=True, swap=False, hw_trigger=True):
        """
        Injects high amplitude noise with known waveform. The ADC measures it.
        The cross correlation contains the information about the resonances.

        Args:
        -----
        band (int): The band to sweep.

        Opt Args:
        ---------
        n_scan (int): The number of scans to take and average
        n_samples (int): The number of samples to take. Default 2^18.
        make_plot (bool): Whether the make plots. Default is False.
        save_data (bool): Whether to save the data.
        timestamp (str): The timestamp as a string.
        correct_att (bool): Correct the response for the attenuators. Default
            is True.
        Returns:
        --------
        f (float array): The frequency information. Length n_samples/2
        resp (complex array): The response information. Length n_samples/2
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        resp = np.zeros((int(n_scan), int(n_samples/2)), dtype=complex)
        for n in np.arange(n_scan):
            self.set_trigger_hw_arm(0, write_log=True)  # Default setup sets to 1

            self.set_noise_select(band, 1, wait_done=True, write_log=True)
            try:
                adc = self.read_adc_data(band, n_samples, hw_trigger=hw_trigger)
            except Exception:
                self.log('ADC read failed. Trying one more time', self.LOG_ERROR)
                adc = self.read_adc_data(band, n_samples, hw_trigger=hw_trigger)
            time.sleep(.05)  # Need to wait, otherwise dac call interferes with adc

            try:
                dac = self.read_dac_data(band, n_samples, hw_trigger=hw_trigger)
            except:
                self.log('ADC read failed. Trying one more time', self.LOG_ERROR)
                dac = self.read_dac_data(band, n_samples, hw_trigger=hw_trigger)
            time.sleep(.05)

            self.set_noise_select(band, 0, wait_done=True, write_log=True)



            if correct_att:
                att_uc = self.get_att_uc(band)
                att_dc = self.get_att_dc(band)
                self.log('UC (DAC) att: {}'.format(att_uc))
                self.log('DC (ADC) att: {}'.format(att_dc))
                if att_uc > 0:
                    scale = (10**(-att_uc/2/20))
                    self.log('UC attenuator > 0. Scaling by {:4.3f}'.format(scale))
                    dac *= scale
                if att_dc > 0:
                    scale = (10**(att_dc/2/20))
                    self.log('UC attenuator > 0. Scaling by {:4.3f}'.format(scale))
                    adc *= scale

            if save_raw_data:
                self.log('Saving raw data...', self.LOG_USER)
                np.save(os.path.join(self.output_dir, 
                    '{}_adc'.format(timestamp)), adc)
                np.save(os.path.join(self.output_dir,
                    '{}_dac'.format(timestamp)), dac)

            # To do : Implement cross correlation to get shift
            
            if swap:
                adc = adc[::-1]

            f, p_dac = signal.welch(dac, fs=614.4E6, nperseg=n_samples/2)
            f, p_adc = signal.welch(adc, fs=614.4E6, nperseg=n_samples/2)
            f, p_cross = signal.csd(dac, adc, fs=614.4E6, nperseg=n_samples/2)

            idx = np.argsort(f)
            f = f[idx]
            p_dac = p_dac[idx]
            p_adc = p_adc[idx]
            p_cross = p_cross[idx]

            resp[n] = p_cross / p_dac

        resp = np.mean(resp, axis=0)

        if make_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(3, figsize=(5,8), sharex=True)
            f_plot = f / 1.0E6

            plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))

            ax[0].semilogy(f_plot, p_dac)
            ax[0].set_ylabel('DAC')
            ax[1].semilogy(f_plot, p_adc)
            ax[1].set_ylabel('ADC')
            ax[2].semilogy(f_plot, np.abs(p_cross))
            ax[2].set_ylabel('Cross')
            ax[2].set_xlabel('Frequency [MHz]')
            ax[0].set_title(timestamp)

            # plt.tight_layout()

            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, 
                    '{}_b{}_full_band_resp_raw.png'.format(timestamp, band)),
                    bbox_inches='tight')
                plt.close()

            fig, ax = plt.subplots(1)

            ax.plot(f_plot[plot_idx], np.log10(np.abs(resp[plot_idx])))
            ax.set_xlabel('Freq [MHz]')
            ax.set_ylabel('Response')
            ax.set_title(timestamp)
            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, 
                    '{}_b{}_full_band_resp.png'.format(timestamp, band)),
                    bbox_inches='tight')
                plt.close()

        if save_data:
            save_name = timestamp + '_{}_full_band_resp.txt'
            np.savetxt(os.path.join(self.output_dir, save_name.format('freq')), 
                f)
            np.savetxt(os.path.join(self.output_dir, save_name.format('real')), 
                np.real(resp))
            np.savetxt(os.path.join(self.output_dir, save_name.format('imag')), 
                np.imag(resp))
            
        return f, resp


    def find_peak(self, freq, resp, rolling_med=False, window=500,
	grad_cut=.05, amp_cut=.25, freq_min=-2.5E8, freq_max=2.5E8, make_plot=False, 
	save_plot=True, band=None,subband=None, make_subband_plot=False, 
	subband_plot_with_slow=False, timestamp=None, pad=2, min_gap=2,plot_title=None):
        """find the peaks within a given subband

        Args:
        -----
        freq (float array): should be a single row of the broader freq array
        resp (complex array): complex response for just this subband

        Opt Args:
        ---------
        rolling_med (bool): whether to use a rolling median for the background
        window (int): number of samples to window together for rolling med
        grad_cut (float): The value of the gradient of phase to look for 
            resonances. Default is .05
        amp_cut (float): The fractional distance from the median value to decide
            whether there is a resonance. Default is .25.
        freq_min (float): The minimum frequency relative to the center of
            the band to look for resonances. Units of Hz. Defaults is -2.5E8
        freq_max (float): The maximum frequency relative to the center of
            the band to look for resonances. Units of Hz. Defaults is 2.5E8
        make_plot (bool): Whether to make a plot. Default is False.
        make_subband_plot (bool): Whether to make a plot per subband. This is
            very slow. Default is False.
        save_plot (bool): Whether to save the plot to self.plot_dir. Default
            is True.
        band (int): The band to take find the peaks in. Mainly for saving
            and plotting.
        timestamp (str): The timestamp. Mainly for saving and plotting
        pad (int): number of samples to pad on either side of a resonance search 
            window
        min_gap (int): minimum number of samples between resonances


        Returns:
        -------_
        resonances (float array): The frequency of the resonances in the band
            in Hz.
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        angle = np.unwrap(np.angle(resp))
        grad = np.ediff1d(angle, to_end=[np.nan])
        amp = np.abs(resp)

        grad_loc = np.array(grad > grad_cut)

        if rolling_med:
            import pandas as pd

            med_amp = pd.Series(amp).rolling(window=window, center=True).median()
        else:
            med_amp = np.median(amp) * np.ones(len(amp))

        starts, ends = self.find_flag_blocks(self.pad_flags(grad_loc, 
            before_pad=pad, after_pad=pad, min_gap=min_gap))

        peak = np.array([], dtype=int)
        for s, e in zip(starts, ends):
            if freq[s] > freq_min and freq[e] < freq_max:
                idx = np.ravel(np.where(amp[s:e] == np.min(amp[s:e])))[0]
                idx += s
                if 1-amp[idx]/med_amp[idx] > amp_cut:
                    peak = np.append(peak, idx)

        # Make summary plot
        if make_plot:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, figsize=(8,4))

            plot_freq = freq

            ax.plot(plot_freq,amp)
            ax.plot(plot_freq, med_amp)
            ax.plot(plot_freq[peak], amp[peak], 'kx')

            for s, e in zip(starts, ends):
                ax.axvspan(plot_freq[s], plot_freq[e], color='k', alpha=.1)

            ax.set_ylabel('Amp')
            ax.set_xlabel('Freq [MHz]')
            title = timestamp
            if band is not None:
                title = title + ' band {}'.format(band)
            if subband is not None:
                title = title + ' subband {}'.format(subband)
            ax.set_title(title)

            if save_plot:
                save_name = timestamp
                if band is not None:
                    save_name = save_name + '_b{}'.format(int(band))
                if subband is not None:
                    save_name = save_name + '_sb{}'.format(int(subband))
                save_name = save_name + '_find_freq.png'
                plt.savefig(os.path.join(self.plot_dir, save_name),
                            bbox_inches='tight', dpi=300)
                plt.close()

        # Make plot per subband
        if make_subband_plot:
            import matplotlib.pyplot as plt
            subbands, subband_freq = self.get_subband_centers(band, 
                hardcode=True)  # remove hardcode mode
            plot_freq = freq
            plot_width = 5.5  # width of plotting in MHz
            width = (subband_freq[1] - subband_freq[0])

            for sb, sbf in zip(subbands, subband_freq):
                self.log('Making plot for subband {}'.format(sb))
                idx = np.logical_and(plot_freq > sbf - plot_width/2.,
                    plot_freq < sbf + plot_width/2.)
                if np.sum(idx) > 1:
                    f = plot_freq[idx]
                    p = angle[idx]
                    x = np.arange(len(p))
                    fp = np.polyfit(x, p, 1)
                    p = p - x*fp[0] - fp[1]

                    g = grad[idx]
                    a = amp[idx]
                    ma = med_amp[idx]

                    fig, ax = plt.subplots(2, sharex=True)
                    ax[0].plot(f, p, label='Phase')
                    ax[0].plot(f, g, label=r'$\Delta$ phase')
                    ax[1].plot(f, a, label='Amp')
                    ax[1].plot(f, ma, label='Median Amp')
                    for s, e in zip(starts, ends):
                        if (plot_freq[s] in f) or (plot_freq[e] in f):
                            ax[0].axvspan(plot_freq[s], plot_freq[e], color='k', 
                                          alpha=.1)
                            ax[1].axvspan(plot_freq[s], plot_freq[e], color='k', 
                                          alpha=.1)

                    for pp in peak:
                        if plot_freq[pp] > sbf - plot_width/2. and \
                                plot_freq[pp] < sbf + plot_width/2.:
                            ax[1].plot(plot_freq[pp], amp[pp], 'xk')

                    ax[0].legend(loc='upper right')
                    ax[1].legend(loc='upper right')

                    ax[0].axvline(sbf, color='k' ,linestyle=':', alpha=.4)
                    ax[1].axvline(sbf, color='k' ,linestyle=':', alpha=.4)
                    ax[0].axvline(sbf - width/2., color='k' ,linestyle='--', 
                                  alpha=.4)
                    ax[0].axvline(sbf + width/2., color='k' ,linestyle='--', 
                                  alpha=.4)
                    ax[1].axvline(sbf - width/2., color='k' ,linestyle='--', 
                                  alpha=.4)
                    ax[1].axvline(sbf + width/2., color='k' ,linestyle='--', 
                                  alpha=.4)

                    ax[1].set_xlim((sbf-plot_width/2., sbf+plot_width/2.))

                    ax[0].set_ylabel('[Rad]')
                    ax[1].set_xlabel('Freq [MHz]')
                    ax[1].set_ylabel('Amp')

                    ax[0].set_title('Band {} Subband {}'.format(band,
                                                                sb, sbf))

                    if subband_plot_with_slow:
                        ff = np.arange(-3, 3.1, .05)
                        rr, ii = self.eta_scan(band, sb, ff, 10, write_log=False)
                        dd = rr + 1.j*ii
                        sbc = self.get_subband_centers(band)
                        ax[1].plot(ff+sbc[1][sb], np.abs(dd)/2.5E6)

                    if save_plot:
                        save_name = '{}_find_freq_b{}_sb{:03}.png'.format(timestamp, band, sb)
                        plt.savefig(os.path.join(self.plot_dir, save_name),
                                    bbox_inches='tight')
                        plt.close()
                else:
                    self.log('No data for subband {}'.format(sb))

        return freq[peak]

    def find_flag_blocks(self, flag, minimum=None, min_gap=None):
        """ 
        Find blocks of adjacent points in a boolean array with the same value. 

        Args:
        -----
        flag : bool, array_like 
            The array in which to find blocks 

        Opt Args:
        ---------
        minimum : int (optional)
            The minimum length of block to return. Discards shorter blocks 
        min_gap : int (optional)
            The minimum gap between flag blocks. Fills in gaps smaller.

        Returns
        ------- 
        starts, ends : int arrays
            The start and end indices for each block.
            NOTE: the end index is the last index in the block. Add 1 for 
            slicing, where the upper limit should be after the block 
        """
        if min_gap is not None:
            _flag = self.pad_flags(np.asarray(flag, dtype=bool),
                min_gap=min_gap).astype(np.int8)
        else:
            _flag = np.asarray(flag).astype(int)

        marks = np.diff(_flag)
        start = np.where(marks == 1)[0]+1
        if _flag[0]:
            start = np.concatenate([[0],start])
        end = np.where(marks == -1)[0]
        if _flag[-1]:
            end = np.concatenate([end,[len(_flag)-1]])

        if minimum is not None:
            inds = np.where(end - start + 1 > minimum)[0]
            return start[inds],end[inds]
        else:
            return start,end

    def pad_flags(self, f, before_pad=0, after_pad=0, min_gap=0, min_length=0):
        """
        Args:
        -----
        f (bool array): The flag array to pad

        Opt Args:
        ---------
        before_pad (int): The number of samples to pad before a flag
        after_pad (int); The number of samples to pad after a flag
        min_gap (int): The smallest allowable gap. If bigger, it combines.
        min_length (int): The smallest length a pad can be.

        Ret:
        ----
        pad_flag (bool array): The padded boolean array
        """
        before, after = self.find_flag_blocks(f)
        after += 1 

        inds = np.where(np.subtract(before[1:],after[:-1]) < min_gap)[0]
        after[inds] = before[inds+1]

        before -= before_pad
        after += after_pad

        padded = np.zeros_like(f)

        for b, a in zip(before, after):
            if (a-after_pad)-(b+before_pad) > min_length:
                padded[np.max([0,b]):a] = True

        return padded

    def plot_find_peak(self, freq, resp, peak_ind, save_plot=True, 
        save_name=None):
        """
        Plots the output of find_Freq

        Args:
        -----
        freq (float array): The frequency data
        resp (float array): The response to full_band_resp
        peak_ind (int array): The indicies of peaks found

        Opt Args:
        ---------
        save_plot (bool): Whether to save the plot
        save_name (str): THe name of the plot
        """
        import matplotlib.pyplot as plt

        if save_plot:
            plt.ioff()
        else:
            plt.ion()

        Idat = np.real(resp)
        Qdat = np.imag(resp)
        phase = np.unwrap(np.arctan2(Qdat, Idat))
        
        fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))
        ax[0].plot(freq, np.abs(resp), label='amp', color='b')
        ax[0].plot(freq, Idat, label='I', color='r', linestyle=':', alpha=.5)
        ax[0].plot(freq, Qdat, label='Q', color='g', linestyle=':', alpha=.5)
        ax[0].legend(loc='lower right')
        ax[1].plot(freq, phase, color='b')
        ax[1].set_ylim((-np.pi, np.pi))

        if len(peak_ind):  # empty array returns False
            ax[0].plot(freq[peak_ind], np.abs(resp[peak_ind]), 'x', color='k')
            ax[1].plot(freq[peak_ind], phase[peak_ind], 'x', color='k')
        else:
            self.log('No peak_ind values.', self.LOG_USER)

        fig.suptitle("Peak Finding")
        ax[1].set_xlabel("Frequency offset from Subband Center (MHz)")
        ax[0].set_ylabel("Response")
        ax[1].set_ylabel("Phase [rad]")

        if save_plot:
            if save_name is None:
                self.log('Using default name for saving: find_peak.png \n' +
                    'Highly recommended that you input a non-default name')
                save_name = 'find_peak.png'
            else:
                self.log('Plotting saved to {}'.format(save_name))
            plt.savefig(os.path.join(self.plot_dir, save_name),
                bbox_inches='tight')
            plt.close()

    def eta_fit(self, freq, resp, peak_freq, delta_freq, 
        subband_half_width=614.4/128, make_plot=False, plot_chans=[], 
        save_plot=True, band=None, timestamp=None, res_num=None,
        use_slow_eta=False):
        """
        Cyndia's eta finding code

        Args:
        -----
        freq (float array): The frequency data
        resp (float array): The response data
        peak_freq (float): The frequency of the resonance peak
        delta_freq (float): The width of frequency to calculate values

        Opt Args:
        ---------
        subband_half_width (float): The width of a subband in MHz. Default
            is 614.4/128
        make_plot (bool): Whether to make plots. Default is False.
        save_plot (bool): Whether to save plots. Default is True.
        plot_chans (int array): The channels to plot. If an empty array, it
            will make plots for all channels.
        band (int): Only used for plotting - the band number of the resontaor
        timestamp (str): The timestamp of the data.
        res_num (int): The resonator number
        

        Rets:
        -----
        eta (complex): The eta parameter
        eta_scaled (complex): The eta parameter divided by subband_half_Width
        eta_phase_deg (float): The angle to rotate IQ circle
        r2 (float): The R^2 value copared to the resonator fit
        eta_mag (float): The amplitude of eta
        latency (float): THe delay
        Q (float): THe resonator quality factor
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        amp = np.abs(resp)
        
        fit = np.polyfit(freq, np.unwrap(np.angle(resp)), 1)
        fitted_line = np.poly1d(fit)  
        phase = np.unwrap(np.angle(resp) - fitted_line(freq))
        
        # Find minimum
        min_idx = np.ravel(np.where(freq == peak_freq))[0]
        
        try:
            left = np.where(freq < peak_freq - delta_freq)[0][-1]
        except IndexError:
            left = 0
        try:
            left_plot = np.where(freq < peak_freq - 5*delta_freq)[0][-1]
        except IndexError:
            left = 0

        right = np.where(freq > peak_freq + delta_freq)[0][0]
        right_plot = np.where(freq > peak_freq + 5*delta_freq)[0][0]
        
        eta = (freq[right] - freq[left]) / (resp[right] - resp[left])
        
        if use_slow_eta:
            band_center = self.get_band_center_mhz(band)
            f_slow, resp_slow, eta_slow = self.eta_estimator(band, 
                peak_freq*1.0E-6+band_center)

        # Get eta parameters
        # w = 5
        # eta = (np.mean(freq[right-w:right+w]) - np.mean(freq[left-w:left+w]))/ \
        #     (np.mean(resp[right-w:right+w]) - np.mean(resp[left-w:left+w]))
        latency = (np.unwrap(np.angle(resp))[-1] - \
            np.unwrap(np.angle(resp))[0]) / (freq[-1] - freq[0])/2/np.pi
        eta_mag = np.abs(eta)
        # eta_mag /= (10*np.log10( uc_att / 2 ) )
        eta_angle = np.angle(eta)
        eta_scaled = eta_mag * 1e-6/ subband_half_width # convert to MHz
        eta_phase_deg = eta_angle * 180 / np.pi


        if left != right:
            sk_fit = tools.fit_skewed_lorentzian(freq[left_plot:right_plot], 
                amp[left_plot:right_plot])
            r2 = np.sum((amp[left_plot:right_plot] - 
                tools.skewed_lorentzian(freq[left_plot:right_plot], 
                *sk_fit))**2)
            Q = sk_fit[5]
        else:
            r2 = np.nan
            Q = np.nan

        if make_plot:
            if len(plot_chans) == 0:
                self.log('Making plot for band' + 
                    ' {} res {:03}'.format(band, res_num))
                self.plot_eta_fit(freq[left_plot:right_plot], 
                    resp[left_plot:right_plot], 
                    eta=eta, eta_mag=eta_mag, r2=r2,
                    save_plot=save_plot, timestamp=timestamp, band=band,
                    res_num=res_num, sk_fit=sk_fit, f_slow=f_slow, resp_slow=resp_slow)
            else:
                if res_num in plot_chans:
                    self.log('Making plot for band ' + 
                        '{} res {:03}'.format(band, res_num))
                    self.plot_eta_fit(freq[left_plot:right_plot], 
                        resp[left_plot:right_plot], 
                        eta=eta, eta_mag=eta_mag, eta_phase_deg=eta_phase_deg, 
                        r2=r2, save_plot=save_plot, timestamp=timestamp, 
                        band=band, res_num=res_num, sk_fit=sk_fit, 
                        f_slow=f_slow, resp_slow=resp_slow)

        return eta, eta_scaled, eta_phase_deg, r2, eta_mag, latency, Q


    def plot_eta_fit(self, freq, resp, eta=None, eta_mag=None, peak_freq=None,
        eta_phase_deg=None, r2=None, save_plot=True, show_plot=False, timestamp=None, 
        res_num=None, band=None, sk_fit=None, f_slow=None, resp_slow=None):
        """
        Plots the eta parameter fits

        Args:
        -----
        freq (float array): The frequency data
        resp (complex array): THe response data

        Opt Args:
        ---------
        eta (complex): The eta parameter
        eta_mag (complex): The amplitude of the eta parameter
        eta_phase_deg (float): The angle of the eta parameter in degrees
        r2 (float): The R^2 value
        save_plot (bool): Whether to save the plot. Default True.
        timestamp (str): The timestamp to name the file
        res_num (int): The resonator number to label the plot
        band (int): The band number to label the plot
        sk_fit (flot array): The fit parameters for the skewe lorentzian
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        if timestamp is None:
            timestamp = self.get_timestamp()

        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        

        I = np.real(resp)
        Q = np.imag(resp)
        amp = np.sqrt(I**2 + Q**2)
        phase = np.unwrap(np.arctan2(Q, I))  # radians

        #if peak_freq is not None:
        #    freq = freq - peak_freq
        if peak_freq is not None:
            plot_freq = freq - peak_freq
        else:
            plot_freq = freq

        plot_freq = plot_freq * 1.0E3

        center_idx = np.ravel(np.where(amp==np.min(amp)))[0]

        fig = plt.figure(figsize=(9,4.5))
        gs=GridSpec(2,3)
        ax0 = fig.add_subplot(gs[0,0])
        ax1 = fig.add_subplot(gs[1,0], sharex=ax0)
        ax2 = fig.add_subplot(gs[:,1:])
        ax0.plot(plot_freq, I, label='I', linestyle=':', color='k')
        ax0.plot(plot_freq, Q, label='Q', linestyle='--', color='k')
        ax0.scatter(plot_freq, amp, c=np.arange(len(freq)), s=3,
            label='amp')
        if sk_fit is not None:
            ax0.plot(plot_freq, tools.skewed_lorentzian(plot_freq*1.0E6, 
                *sk_fit), color='r', linestyle=':')
        ax0.legend(fontsize=10, loc='lower right')
        ax0.set_ylabel('Resp')

        ax1.scatter(plot_freq, np.rad2deg(phase), c=np.arange(len(freq)), s=3)
        ax1.set_ylabel('Phase [deg]')
        ax1.set_xlabel('Freq [kHz]')

        # IQ circle
        ax2.axhline(0, color='k', linestyle=':', alpha=.5)
        ax2.axvline(0, color='k', linestyle=':', alpha=.5)

        ax2.scatter(I, Q, c=np.arange(len(freq)), s=3)
        ax2.set_xlabel('I')
        ax2.set_ylabel('Q')

        if peak_freq is not None:
            ax0.text(.03, .9, '{:5.2f} MHz'.format(peak_freq),
                      transform=ax0.transAxes, fontsize=10)

        lab = ''
        if eta is not None:
            if eta_mag is not None:
                lab = r'$\eta/\eta_{mag}$' + \
                ': {:4.3f}+{:4.3f}'.format(np.real(eta/eta_mag), 
                    np.imag(eta/eta_mag)) + '\n'
            else:
                lab = lab + r'$\eta$' + ': {}'.format(eta) + '\n'
        if eta_mag is not None:
            lab = lab + r'$\eta_{mag}$' + ': {:1.3e}'.format(eta_mag) + '\n'
        if eta_phase_deg is not None:
            lab = lab + r'$\eta_{ang}$' + \
                ': {:3.2f}'.format(eta_phase_deg) + '\n'
        if r2 is not None:
            lab = lab + r'$R^2$' + ' :{:4.3f}'.format(r2)

        ax2.text(.03, .80, lab, transform=ax2.transAxes, fontsize=10)

        if eta is not None:
            if eta_mag is not None:
                eta = eta/eta_mag
            respp = eta*resp
            Ip = np.real(respp)
            Qp = np.imag(respp)
            ax2.scatter(Ip, Qp, c=np.arange(len(freq)), cmap='inferno', s=3)

        if f_slow is not None and resp_slow is not None:
            self.log('Adding slow eta scan')
            mag_scale = 5E5
            band_center = self.get_band_center_mhz(band)

            resp_slow /= mag_scale
            I_slow = np.real(resp_slow)
            Q_slow = np.imag(resp_slow)
            phase_slow = np.unwrap(np.arctan2(Q_slow, I_slow))  # radians

            ax0.scatter(f_slow-band_center, np.abs(resp_slow), 
                c=np.arange(len(f_slow)), cmap='Greys', s=3)
            ax1.scatter(f_slow-band_center, np.rad2deg(phase_slow),
                c=np.arange(len(f_slow)), cmap='Greys', s=3)
            ax2.scatter(I_slow, Q_slow, c=np.arange(len(f_slow)), cmap='Greys', 
                s=3)

        plt.tight_layout()

        if save_plot:
            if res_num is not None and band is not None:
                save_name = '{}_eta_b{}_res{:03}.png'.format(timestamp, band, 
                    res_num)
            else:
                save_name = '{}_eta.png'.format(timestamp)
            plt.savefig(os.path.join(self.plot_dir, save_name), 
                bbox_inches='tight')

        if not show_plot:
            plt.close()

    def get_closest_subband(self, f, band, as_offset=True):
        """
        Gives the closest subband number for a given input frequency.

        Args:
        -----
        f (float): The frequency to search for a subband
        band (int): The band to identify

        Ret:
        ----
        subband (int): The subband that contains the frequency
        """
        # get subband centers:
        subbands, centers = self.get_subband_centers(band, as_offset=as_offset)
        if self.check_freq_scale(f, centers[0]):
            pass
        else:
            raise ValueError('{} and {}'.format(f, centers[0]))
            
        idx = np.argmin([abs(x - f) for x in centers])
        return idx

    def check_freq_scale(self, f1, f2):
        """
        Makes sure that items are the same frequency scale (ie MHz, kHZ, etc.)

        Args:
        -----
        f1 (float): The first frequency
        f2 (float): The second frequency

        Ret:
        ----
        same_scale (bool): Whether the frequency scales are the same
        """
        if abs(f1/f2) > 1e3:
            return False
        else:
            return True

    def assign_channels(self, freq, band=None, bandcenter=None, 
        channel_per_subband=4, as_offset=True, min_offset=0.5):
        """
        Figures out the subbands and channels to assign to resonators

        Args:
        -----
        freq (flot array): The frequency of the resonators. This is not the
            same as the frequency output from full_band_resp. This is only
            where the resonators are.

        Opt Args:
        ---------
        band (int): The band to assign channels
        band_center (float array): The frequency center of the band. Must supply
            band or subband center.
        channel_per_subband (int): The number of channels to assign per
            subband. Default is 4.
        min_offset (float): The minimum offset between two resonators in MHz.
            If closer, then both are ignored.

        Ret:
        ----
        subbands (int array): An array of subbands to assign resonators
        channels (int array): An array of channel numbers to assign resonators
        offsets (float array): The frequency offset from the subband center
        """
        freq = np.sort(freq)  # Just making sure its in sequential order
        d_freq = np.diff(freq)
        close_idx = d_freq > min_offset
        close_idx = np.logical_and(np.hstack((close_idx, True)), 
            np.hstack((True, close_idx)))

        if band is None and bandcenter is None:
            self.log('Must have band or bandcenter', self.LOG_ERROR)
            raise ValueError('Must have band or bandcenter')

        subbands = np.zeros(len(freq), dtype=int)
        channels = -1 * np.ones(len(freq), dtype=int)
        offsets = np.zeros(len(freq))
        
        # Assign all frequencies to a subband
        for idx in range(len(freq)):
            subbands[idx] = self.get_closest_subband(freq[idx], band, 
                                                     as_offset=as_offset)
            subband_center = self.get_subband_centers(band, 
                as_offset=as_offset)[1][subbands[idx]]

            offsets[idx] = freq[idx] - subband_center
        
        # Assign unique channel numbers
        for unique_subband in set(subbands):
            chans = self.get_channels_in_subband(band, int(unique_subband))
            mask = np.where(subbands == unique_subband)[0]
            if len(mask) > channel_per_subband:
                concat_mask = mask[:channel_per_subband]
            else:
                concat_mask = mask[:]
            
            chans = chans[:len(list(concat_mask))] #I am so sorry
            
            channels[mask[:len(chans)]] = chans

        # Prune channels that are too close
        channels[~close_idx] = -1
        
        return subbands, channels, offsets


    def compare_tuning(self, tune, ref_tune, make_plot=False):
        """
        Compares tuning file to a reference tuning file. Does not work yet.

        """

        # Load data
        tune, freq, resp = self.load_tuning(tune)
        tune_ref, freq_ref, resp_ref = self.load_tuning(ref_tune)

        if make_plot:
            import matplotlib.pyplot as plt

            plt_freq = freq * 1.0E-6
            fig, ax = plt.subplots(2, sharex=True, figsize=(6,5))
            ax[0].plot(plt_freq, np.abs(resp))
            ax[0].plot(plt_freq, np.abs(resp_ref))

            for k in tune.keys():
                ax[0].axvline(tune[k]['freq']*1.0E-6, color='b', linestyle=':')
            for k in tune_ref.keys():
                ax[0].axvline(tune_ref[k]['freq']*1.0E-6, color='r', 
                    linestyle=':')


            ax[1].plot(plt_freq, np.abs(resp) - np.abs(resp_ref))

            plt.tight_layout()

    def load_tuning(self, tune, load_raw=True):
        """
        Loads tuning files from disk.

        Args:
        -----
        tune (str): The full path to the freq_resp.npy file.

        Opt Args:
        ---------
        load_raw (bool): Whether to load the freq and response data. Default
            is True.

        Ret:
        ----
        tune (dict): The tuning file
        freq (float array): The frequency information. Returns if load_raw is
            True.
        resp (complex array): The full band response information. Returns if
            load_raw is True.
        """
        self.log('Loading {}'.format(tune), self.LOG_INFO)
        if load_raw:
            dirname = os.path.dirname(tune)
            basename = os.path.basename(tune).split('_')[0]
            freq = np.loadtxt(os.path.join(dirname, 
                basename+'_freq_full_band_resp.txt'))
            resp = np.loadtxt(os.path.join(dirname, 
                basename+'_real_full_band_resp.txt')) + \
                1.j * np.loadtxt(os.path.join(dirname, 
                basename+'_imag_full_band_resp.txt'))
        tune = np.load(tune).item()

        if load_raw:
            return tune, freq, resp
        else:
            return tune

    def relock(self, band, res_num=None, drive=None, r2_max=.08, 
        q_max=100000, q_min=0, check_vals=False, min_gap=None):
        """
        Turns on the tones. Also cuts bad resonators.

        Args:
        -----
        band (int): The band to relock

        Opt args:
        ---------
        res_num (int array): The resonators to lock. If None, tries all the
            resonators.
        drive (int): The tone amplitudes to set
        r2_max (float): The highest allowable R^2 value
        q_max (float): The maximum resonator Q factor
        q_min (float): The minimum resonator Q factor
        """
        if res_num is None:
            res_num = np.arange(512)
        else:
            res_num = np.array(res_num)

        if drive is None:
            drive = self.freq_resp[band]['drive']

        digitzer_freq = self.get_digitizer_frequency_mhz(band)
        n_subband = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)

        subband = digitzer_freq/(n_subband/2.)  # Oversample by 2

        amplitude_scale = np.zeros(n_channels)
        center_freq = np.zeros(n_channels)
        feedback_enable = np.zeros(n_channels)
        eta_phase = np.zeros(n_channels)
        eta_mag = np.zeros(n_channels)

        f = [self.freq_resp[band]['resonances'][k]['freq'] \
                     for k in self.freq_resp[band]['resonances'].keys()]
            

        # Populate arrays
        counter = 0
        for k in self.freq_resp[band]['resonances'].keys():
            ch = self.freq_resp[band]['resonances'][k]['channel']
            idx = np.where(f == self.freq_resp[band]['resonances'][k]['freq'])[0][0]
            f_gap = np.min(np.abs(np.append(f[:idx], f[idx+1:])-f[idx]))

            self.log('Res {:03} - Channel {}'.format(k, ch))
            if ch < 0: 
                self.log('No channel assigned: res {:03}'.format(k))
            elif min_gap is not None and f_gap < min_gap:
                self.log('Closest resonator is {:3.3f} MHz away'.format(f_gap))
            elif self.freq_resp[band]['resonances'][k]['r2'] > r2_max and check_vals:
                self.log('R2 too high: res {:03}'.format(k))
            elif self.freq_resp[band]['resonances'][k]['Q'] < q_min and check_vals:
                self.log('Q too low: res {:03}'.format(k))
            elif self.freq_resp[band]['resonances'][k]['Q'] > q_max and check_vals:
                self.log('Q too high: res {:03}'.format(k))
            elif k not in res_num:
                self.log('Not in resonator list')
            else:
                center_freq[ch] = self.freq_resp[band]['resonances'][k]['offset']
                amplitude_scale[ch] = drive
                feedback_enable[ch] = 1
                eta_phase[ch] = self.freq_resp[band]['resonances'][k]['eta_phase']
                eta_mag[ch] = self.freq_resp[band]['resonances'][k]['eta_scaled']
                counter += 1

        # Set the actual variables
        self.set_center_frequency_array(band, center_freq, write_log=True,
            log_level=self.LOG_INFO)
        self.set_amplitude_scale_array(band, amplitude_scale.astype(int),
            write_log=True, log_level=self.LOG_INFO)
        self.set_feedback_enable_array(band, feedback_enable.astype(int),
            write_log=True, log_level=self.LOG_INFO)
        self.set_eta_phase_array(band, eta_phase, write_log=True,
            log_level=self.LOG_INFO)
        self.set_eta_mag_array(band, eta_mag, write_log=True, 
            log_level=self.LOG_INFO)

        self.log('Setting on {} channels on band {}'.format(counter, band),
            self.LOG_USER)

        
    def eta_estimator(self, band, freq, drive=10, f_sweep_half=.3, 
                      df_sweep=.002, delta_freq=.05):
        """
        Estimates eta parameters using the slow eta_scan
        """
        subband, offset = self.freq_to_subband(freq, band)
        f_sweep = np.arange(offset-f_sweep_half, offset+f_sweep_half, df_sweep)
        f, resp = self.fast_eta_scan(band, subband, f_sweep, 2, drive)
        # resp = rr + 1.j*ii

        a_resp = np.abs(resp)
        idx = np.ravel(np.where(a_resp == np.min(a_resp)))[0]
        f0 = f_sweep[idx]

        try:
            left = np.where(f_sweep < f0 - delta_freq)[0][-1]
        except IndexError:
            left = 0

        try:
            right = np.where(f_sweep > f0 + delta_freq)[0][0]
        except:
            right = len(f_sweep)-1

        subband_half_width = self.get_digitizer_frequency_mhz(band)/\
            self.get_number_sub_bands(band)

        eta = (f_sweep[right]-f_sweep[left])/(resp[right]-resp[left])
        eta_mag = np.abs(eta)
        eta_phase = np.angle(eta)
        eta_phase_deg = np.rad2deg(eta_phase)
        eta_scaled = eta_mag/subband_half_width
        
        sb, sbc = self.get_subband_centers(band, as_offset=False)

        return f_sweep+sbc[subband], resp, eta

    def eta_scan(self, band, subband, freq, drive, write_log=False,
                 sync_group=True):
        """
        Same as slow eta scans
        """
        if len(self.which_on(band)):
            self.band_off(band, write_log=False)

        n_subband = self.get_number_sub_bands(band)
        n_channel = self.get_number_channels(band)
        channel_order = self.get_channel_order()
        first_channel = channel_order[::n_channel//n_subband]

        self.set_eta_scan_channel(band, first_channel[subband], 
                                  write_log=write_log)
        self.set_eta_scan_amplitude(band, drive, write_log=write_log)
        self.set_eta_scan_freq(band, freq, write_log=write_log)
        self.set_eta_scan_dwell(band, 0, write_log=write_log)

        self.set_run_eta_scan(band, 1, wait_done=False, write_log=write_log)
        pvs = [self._cryo_root(band) + self._eta_scan_results_real,
               self._cryo_root(band) + self._eta_scan_results_imag]

        if sync_group:
            sg = SyncGroup(pvs, skip_first=False)

            sg.wait()
            vals = sg.get_values()
            rr = vals[pvs[0]]
            ii = vals[pvs[1]]
        else:
            rr = self.get_eta_scan_results_real(2, len(freq))
            ii = self.get_eta_scan_results_imag(2, len(freq))

        self.set_amplitude_scale_channel(band, first_channel[subband], 0)

        return rr, ii

    def flux_ramp_check(self, band, reset_rate_khz=4, 
                        fraction_full_scale=.495, flux_ramp=True,
                        save_plot=True, show_plot=False):
        """
        Tries to measure the V-phi curve in feedback disable mode. 
        You can also run this with flux ramp off to see the intrinsic
        noise on the readout channel.

        Args:
        -----
        band (int) : The band to check.

        Opt Args:
        ---------
        reset_rate_khz (float) : The flux ramp rate in kHz.
        fraction_full_scale (float) : The amplitude of the flux ramp from
           zero to one.
        flux_ramp (bool) : Whether to flux ramp. Default is True.
        save_plot (bool) : Whether to save the plot. Default True.
        show_plot (bool) : Whether to show the plot. Default False.
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        old_fb = self.get_feedback_enable_array(band)

        # Turn off feedback
        self.set_feedback_enable_array(band, np.zeros_like(old_fb))
        d, df, sync = self.tracking_setup(band,0, reset_rate_khz=reset_rate_khz,
                                          fraction_full_scale=fraction_full_scale,
                                          make_plot=False,
                                          save_plot=False, show_plot=False,
                                          lms_enable1=False, lms_enable2=False,
                                          lms_enable3=False, flux_ramp=flux_ramp)


        n_samp, n_chan = np.shape(df)

        dd = np.ravel(np.where(np.diff(sync[:,0]) !=0))
        first_idx = dd[0]//512
        second_idx = dd[4]//512
        dt = int(second_idx-first_idx)  # In slow samples
        n_fr = int(len(sync[:,0])/512/dt)
        reset_idx = np.arange(first_idx, n_fr*dt + first_idx+1, dt)

        # Reset to the previous FB state
        self.set_feedback_enable_array(band, old_fb)

        fs = self.get_digitizer_frequency_mhz(band) * 1.0E6 /2/512

        # Only plot channels that are on - group by subband
        chan = self.which_on(band)
        freq = np.zeros(len(chan), dtype=float)
        subband = np.zeros(len(chan), dtype=int)
        for i, c in enumerate(chan):
            freq[i] = self.channel_to_freq(band, c)
            (subband[i], _) = self.freq_to_subband(freq[i], band)

        unique_subband = np.unique(subband)

        cm = plt.get_cmap('viridis')
        
        timestamp = self.get_timestamp()

        self.log('Making plots...')
        scale = 1.0E3
        
        n_high = 3
        highs = np.zeros((n_high, len(chan)))

        for sb in unique_subband:
            idx = np.ravel(np.where(subband == sb))
            chs = chan[idx]
            # fig, ax = plt.subplots(1, 2, figsize=(8,4), sharey=True)
            fig = plt.figure(figsize=(8,6))
            gs = GridSpec(2,2)
            ax0 = fig.add_subplot(gs[0,:])
            ax1 = fig.add_subplot(gs[1,0])
            ax2 = fig.add_subplot(gs[1,1])

            for i, c in enumerate(chs):
                color = cm(i/len(chs))
                ax0.plot(np.arange(n_samp)/fs*scale,
                         df[:,c], label='ch {}'.format(c),
                         color=color)
                holder = np.zeros((n_fr-1, dt))
                for i in np.arange(n_fr-1):
                    holder[i] = df[first_idx+dt*i:first_idx+dt*(i+1),c]
                ds = np.mean(holder, axis=0)
                ax1.plot(np.arange(len(ds))/fs*scale, ds, color=color)
                ff, pp = signal.welch(df[:,c], fs=fs)
                ax2.semilogy(ff/1.0E3, pp, color=color)
                
                sort_idx = np.argsort(pp)[::-1]
                #print(sort_idx)
                #highs[:,i] = ff[sort_idx[:n_high]]
                #print(ff[sort_idx[:n_high]])

            for k in reset_idx:
                ax0.axvline(k/fs*scale, color='k', alpha=.6, linestyle=':')

            ax0.legend(loc='upper left')
            ax1.set_xlabel('Time [ms]')
            ax2.set_xlabel('Freq [kHz]')
            fig.suptitle('Band {} Subband {}'.format(band, sb))

            if save_plot:
                save_name = timestamp
                if not flux_ramp:
                    save_name = save_name + '_no_FR'
                save_name = save_name+ \
                    '_b{}_sb{:03}_flux_ramp_check.png'.format(band, sb)
                plt.savefig(os.path.join(self.plot_dir, save_name),
                            bbox_inches='tight')
                if not show_plot:
                    plt.close()

        # Make summary plot
        #highs = highs * 1.0E-3
        #step_size = np.max(ff)*1.0E-3/20
        #bins = np.arange(0, 20*step_size, step_size)
        #fig, ax = plt.subplots(n_high, sharex=True, figsize=(5,8))
        #for h in np.arange(n_high):
        #    ax[h].hist(highs[h], bins=bins)
        #    ax[h].set_ylabel('{} count'.format(h))
        #ax[n_high-1].set_ylabel('freq [kHz]')

        #if save_plot:
        #    save_name = timestamp
        #    if not flux_ramp:
        #        save_name = save_name + '_no_FR'
        #    save_name = save_name + 'flux_ramp_check_max.png'
        #    plt.savefig(os.path.join(self.plot_dir, save_name),
        #                bbox_inches='tight')
        #    if not show_plot:
        #        plt.close()
        return d, df, sync

    def tracking_setup(self, band, channel, reset_rate_khz=4., write_log=False, 
        make_plot=False, save_plot=True, show_plot=True,
        lms_freq_hz=4000., flux_ramp=True, fraction_full_scale=.4950,
        lms_enable1=True, lms_enable2=True, lms_enable3=True, lms_gain=7):
        """
        Args:
        -----
        band (int) : The band number
        channel (int) : The channel to check

        Opt Args:
        ---------
        reset_rate_khz (float) : The flux ramp frequency
        write_log (bool) : Whether to write output to the log
        make_plot (bool) : Whether to make plots. Default False.
        save_plot (bool) : Whether to save plots. Default True.
        show_plot (bool) : Whether to display the plot. Default True.
        lms_freq_hz (float) : The frequency of the tracking algorithm.
           Default is 4000
        flux_ramp (bool) : Whether to turn on flux ramp. Default True.
        fraction_full_scale (float) : The flux ramp amplitude, as a
           fraction of the maximum. Default is .4950.
        lms_enable1 (bool) : Whether to use the first harmonic for tracking.
           Default True.
        lms_enable2 (bool) : Whether to use the second harmonic for tracking.
           Default True.
        lms_enable3 (bool) : Whether to use the third harmonic for tracking.
           Default True.
        lms_gain (int) : The tracking gain parameters. Default 7.
        """
        if not flux_ramp:
            self.log('WARNING: THIS WILL NOT TURN ON FLUX RAMP!')
            
        if make_plot:
            import matplotlib.pyplot as plt
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

        # To do: Move to experiment config
        flux_ramp_full_scale_to_phi0 = 2.825/0.75

        lms_delay = 6  # nominally match refPhaseDelay
        if not flux_ramp:
            lms_enable1 = 0
            lms_enable2 = 0
            lms_enable3 = 0
                

        lms_rst_dly = 31  # disable error term for 31 2.4MHz ticks after reset

        self.log("Using lmsFreqHz = {}".format(lms_freq_hz), self.LOG_USER)
        lms_delay2    = 255  # delay DDS counter resets, 307.2MHz ticks
        lms_delay_fine = 0
        iq_stream_enable = 0  # stream IQ data from tracking loop

        self.set_lms_delay(band, lms_delay, write_log=write_log)
        self.set_lms_dly_fine(band, lms_delay_fine, write_log=write_log)
        self.set_lms_gain(band, lms_gain, write_log=write_log)
        self.set_lms_enable1(band, lms_enable1, write_log=write_log)
        self.set_lms_enable2(band, lms_enable2, write_log=write_log)
        self.set_lms_enable3(band, lms_enable3, write_log=write_log)
        self.set_lms_rst_dly(band, lms_rst_dly, write_log=write_log)
        self.set_lms_freq_hz(band, lms_freq_hz, write_log=write_log)
        self.set_lms_delay2(band, lms_delay2, write_log=write_log)
        self.set_iq_stream_enable(band, iq_stream_enable, write_log=write_log)

        self.flux_ramp_setup(reset_rate_khz, fraction_full_scale) # write_log?

        if flux_ramp:
            self.flux_ramp_on(write_log=write_log)

        # take one dataset with all channels
        f, df, sync = self.take_debug_data(band, IQstream = iq_stream_enable, 
            single_channel_readout=0)
        df_std = np.std(df, 0)
        channels_on = list(set(np.where(df_std > 0)[0]) & set(self.which_on(band)))
        self.log("Number of channels on = {}".format(len(channels_on)), 
            self.LOG_USER)
        self.log("Flux ramp demod. mean error std = "+ 
            "{} kHz".format(np.mean(df_std[channels_on]) * 1e3), self.LOG_USER)
        self.log("Flux ramp demod. median error std = "+
            "{} kHz".format(np.median(df_std[channels_on]) * 1e3), self.LOG_USER)
        f_span = np.max(f,0) - np.min(f,0)
        self.log("Flux ramp demod. mean p2p swing = "+
            "{} kHz".format(np.mean(f_span[channels_on]) * 1e3), self.LOG_USER)
        self.log("Flux ramp demod. median p2p swing = "+
            "{} kHz".format(np.median(f_span[channels_on]) * 1e3), self.LOG_USER)

        for c in channels_on:
            self.log('ch {} - f_span {}'.format(c, f_span[c]))

        if make_plot:
            timestamp = self.get_timestamp()

            plt.figure()
            plt.hist(df_std[channels_on] * 1e3,bins = 20,edgecolor = 'k')            
            plt.xlabel('Flux ramp demod error std (kHz)')
            plt.ylabel('number of channels')
            plt.title('LMS freq = {}, n_channels = {}'.format(lms_freq_hz, 
                len(channels_on)))

            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, timestamp + 
                                     '_FRtrackingErrorHist.png'))
            if not show_plot:
                plt.close()

            plt.figure()
            plt.hist(f_span[channels_on] * 1e3,bins = 20,edgecolor='k')
            plt.xlabel('Flux ramp amplitude (kHz)')
            plt.ylabel('number of channels')
            plt.title('LMS freq = {}, n_channels = {}'.format(lms_freq_hz, 
                len(channels_on)))
            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, timestamp + 
                            '_FRtrackingAmplitudeHist.png'))
            if not show_plot:
                plt.close()


            fig, ax = plt.subplots(1)
            ax.plot(f_span[channels_on], df_std[channels_on], '.')
            ax.set_xlabel('FR Amp')
            ax.set_ylabel('RF demod error')
            x = np.array([0, np.max(f_span[channels_on])])
            y = x/10.
            ax.plot(x,y, color='k', linestyle=':')
            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, timestamp + 
                            '_FR_amp_v_err.png'))
            if not show_plot:
                plt.close()


            self.log("Taking data on single channel number {}".format(channel), 
                self.LOG_USER)

            n_els = 2500
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(f[:n_els, channel])
            ax[0].set_ylabel('Tracked Freq [MHz]')
            ax[0].text(.025, .9, 'LMS Freq {}'.format(lms_freq_hz), fontsize=10,
                        transform=ax[0].transAxes)

            ax[0].text(.9, .9, 'Band {} Ch {:03}'.format(band, channel), fontsize=10,
                        transform=ax[0].transAxes, horizontalalignment='right')

            ax[1].plot(df[:n_els, channel])
            ax[1].set_ylabel('Freq Error [MHz]')
            ax[1].set_xlabel('Samp Num')
            ax[1].text(.025, .9, 'RMS error: {:5.4f} \n'.format(df_std[channel]) +
                        'FR amp: {:3.2f}'.format(fraction_full_scale),
                        fontsize=10, transform=ax[1].transAxes)

            plt.tight_layout()

            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, timestamp + 
                    '_FRtracking_band{}_ch{:03}.png'.format(band,channel)),
                    bbox_inches='tight')
            if not show_plot:
                plt.close()

        self.set_iq_stream_enable(band, 1, write_log=write_log)

        return f, df, sync

    
    def eta_phase_check(self, band, rot_step_size=30, rot_max=360,
                        reset_rate_khz=4., 
                        fraction_full_scale=.495,
                        flux_ramp=True):
        """
        """
        ret = {}

        eta_phase0 = self.get_eta_phase_array(band)
        ret['eta_phase0'] = eta_phase0
        ret['band'] = band

        old_fb = self.get_feedback_enable_array(band)
        self.set_feedback_enable_array(band, np.zeros_like(old_fb))

        rot_ang = np.arange(0, rot_max, rot_step_size)
        ret['rot_ang'] = rot_ang
        ret['data'] = {}

        for i, r in enumerate(rot_ang):
            self.log('Rotating {:3.1f} deg'.format(r))
            eta_phase = np.zeros_like(eta_phase0)
            for c in np.arange(512):
                eta_phase[c] = tools.limit_phase_deg(eta_phase0[c] + r)
            self.set_eta_phase_array(band, eta_phase)
                         
            d, df, sync = self.tracking_setup(band,0, reset_rate_khz=reset_rate_khz,
                                          fraction_full_scale=fraction_full_scale,
                                          make_plot=False,
                                          save_plot=False, show_plot=False,
                                          lms_enable1=False, lms_enable2=False,
                                          lms_enable3=False, flux_ramp=flux_ramp)
            ret['data'][r] = {}
            ret['data'][r]['df'] = df
            ret['data'][r]['sync'] = sync

        self.set_feedback_enable_array(band, old_fb)
        self.set_eta_phase_array(2, eta_phase0)

        return ret

    def analyze_eta_phase_check(self, dat, channel):
        """
        """
        import matplotlib.pyplot as plt
        keys = dat['data'].keys()
        band = dat['band']
        n_keys = len(keys)

        fs = self.get_digitizer_frequency_mhz(band) * 1.0E6 /2/512
        scale = 1.0E3

        fig, ax = plt.subplots(1)
        cm = plt.get_cmap('viridis')
        for j, k in enumerate(keys):
            sync = dat['data'][k]['sync']
            df = dat['data'][k]['df']
            dd = np.ravel(np.where(np.diff(sync[:,0]) !=0))
            first_idx = dd[0]//512
            second_idx = dd[4]//512
            dt = int(second_idx-first_idx)  # In slow samples                                             
            n_fr = int(len(sync[:,0])/512/dt)
            reset_idx = np.arange(first_idx, n_fr*dt + first_idx+1, dt)

            holder = np.zeros((n_fr-1, dt))
            for i in np.arange(n_fr-1):
                holder[i] = df[first_idx+dt*i:first_idx+dt*(i+1), channel]
            ds = np.mean(holder, axis=0)

            color = cm(j/n_keys)
            ax.plot(np.arange(len(ds))/fs*scale, ds, color=color,
                    label='{:3.1f}'.format(k))

        ax.legend()
        ax.set_title('Band {} Ch {:03}'.format(band, channel))
        ax.set_xlabel('Time [ms]')




    _num_flux_ramp_dac_bits = 16
    _cryo_card_flux_ramp_relay_bit = 16
    _cryo_card_relay_wait = 0.25 #sec
    def unset_fixed_flux_ramp_bias(self,acCouple=True):
        """
        Alias for setting ModeControl=0
        """

        # make sure flux ramp is configured off before switching back into mode=1
        self.flux_ramp_off() 

        self.log("Setting flux ramp ModeControl to 0.",self.LOG_USER)        
        self.set_mode_control(0)

        ## Don't want to flip relays more than we have to.  Check if it's in the correct
        ## position ; only explicitly flip to DC if we have to.
        if acCouple and (self.get_cryo_card_relays() >> 
            self._cryo_card_flux_ramp_relay_bit & 1):
            self.log("Flux ramp set to DC mode (rly=0).",
                     self.LOG_USER)
            self.set_cryo_card_relay_bit(self._cryo_card_flux_ramp_relay_bit,0)

            # make sure it gets picked up by cryo card before handing back
            while (self.get_cryo_card_relays() >> 
                self._cryo_card_flux_ramp_relay_bit & 1):
                self.log("Waiting for cryo card to update",
                         self.LOG_USER)
                time.sleep(self._cryo_card_relay_wait)


    def set_fixed_flux_ramp_bias(self,fractionFullScale):
        """
        ???

        Args:
        -----
        fractionFullScale (float) : Fraction of full flux ramp scale to output 
        from [-1,1]
        """

        # fractionFullScale must be between [0,1]
        if abs(np.abs(fractionFullScale))>1:
            raise ValueError("fractionFullScale = {} not in [-1,1]".format(fractionFullScale))

        ## Disable flux ramp if it was on
        ## Doesn't seem to effect the fixed DC value being output
        ## if already in fixed flux ramp mode ModeControl=1
        self.flux_ramp_off() 

        ## Don't want to flip relays more than we have to.  Check if it's in the correct
        ## position ; only explicitly flip to DC if we have to.
        if not (self.get_cryo_card_relays() >> self._cryo_card_flux_ramp_relay_bit & 1):
            self.log("Flux ramp relay is either in AC mode or we haven't set it yet - explicitly setting to DC mode (=1).",
                     self.LOG_USER)
            self.set_cryo_card_relay_bit(self._cryo_card_flux_ramp_relay_bit,1)

            while not (self.get_cryo_card_relays() >> self._cryo_card_flux_ramp_relay_bit & 1):
                self.log("Waiting for cryo card to update",
                         self.LOG_USER)
                time.sleep(self._cryo_card_relay_wait)
                
        ## ModeControl must be 1
        mode_control=self.get_mode_control()
        if not mode_control==1:

            #before switching to ModeControl=1, make sure DAC is set to output zero V
            LTC1668RawDacData0=np.floor(0.5*(2**self._num_flux_ramp_dac_bits))
            self.log("Before switching to fixed DC flux ramp output, " + 
                " explicitly setting flux ramp DAC to zero "+
                "(LTC1668RawDacData0={})".format(mode_control,LTC1668RawDacData0), 
                     self.LOG_USER)
            self.set_flux_ramp_dac(LTC1668RawDacData0)

            self.log("Flux ramp ModeControl is {}".format(mode_control) +
                " - changing to 1 for fixed DC output.", 
                     self.LOG_USER)
            self.set_mode_control(1)

        ## Compute and set flux ramp DAC to requested value
        LTC1668RawDacData = np.floor((2**self._num_flux_ramp_dac_bits)*
            (1-np.abs(fractionFullScale))/2);
        ## 2s complement
        if fractionFullScale<0:
            LTC1668RawDacData = 2**self._num_flux_ramp_dac_bits-LTC1668RawDacData-1
        self.log("Setting flux ramp to {}".format(100 * fractionFullScale, 
            int(LTC1668RawDacData)) + "% of full scale (LTC1668RawDacData={})", 
                 self.LOG_USER)
        self.set_flux_ramp_dac(LTC1668RawDacData)        

    def flux_ramp_setup(self, reset_rate_khz, fraction_full_scale, df_range=.1, 
        do_read=False, band=2, write_log=False):
        """
        Set flux ramp sawtooth rate and amplitude. If there are errors, check 
        that you are using an allowed reset rate! Not all rates are allowed.

        Allowed rates: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15 kHz
        """

        # Disable flux ramp
        self.flux_ramp_off(write_log=write_log) # no write log?
        #self.set_cfg_reg_ena_bit(0) # let us switch this to flux ramp on/off

        digitizerFrequencyMHz = self.get_digitizer_frequency_mhz(band)
        dspClockFrequencyMHz=digitizerFrequencyMHz/2

        desiredRampMaxCnt = ((dspClockFrequencyMHz*1e3)/
            (reset_rate_khz)) - 1
        rampMaxCnt = np.floor(desiredRampMaxCnt)

        resetRate = (dspClockFrequencyMHz * 1e6) / (rampMaxCnt + 1)

        HighCycle = 5 # not sure why these are hardcoded
        LowCycle = 5
        rtmClock = (dspClockFrequencyMHz * 1e6) / (HighCycle + LowCycle + 2)
        trialRTMClock = rtmClock

        fullScaleRate = fraction_full_scale * resetRate
        desFastSlowStepSize = (fullScaleRate * 2**self.num_flux_ramp_counter_bits) / rtmClock
        trialFastSlowStepSize = round(desFastSlowStepSize)
        FastSlowStepSize = trialFastSlowStepSize

        trialFullScaleRate = trialFastSlowStepSize * trialRTMClock / (2**self.num_flux_ramp_counter_bits)

        trialResetRate = (dspClockFrequencyMHz * 1e6) / (rampMaxCnt + 1)
        trialFractionFullScale = trialFullScaleRate / trialResetRate
        fractionFullScale = trialFractionFullScale
        diffDesiredFractionFullScale = np.abs(trialFractionFullScale - 
            fraction_full_scale)

        self.log("Percent full scale = {}%".format(100 * fractionFullScale), 
            self.LOG_USER)

        if diffDesiredFractionFullScale > df_range:
            raise ValueError("Difference from desired fraction of full scale " +
                "exceeded! {}".format(diffDesiredFractionFullScale) +
                " vs acceptable {}".format(df_range))
            self.log("Difference from desired fraction of full scale exceeded!" +
                " P{} vs acceptable {}".format(diffDesiredFractionFullScale, 
                    df_range), 
                self.LOG_USER)

        if rtmClock < 2e6:
            raise ValueError("RTM clock rate = "+
                "{} is too low (SPI clock runs at 1MHz)".format(rtmClock*1e-6))
            self.log("RTM clock rate = "+
                "{} is too low (SPI clock runs at 1MHz)".format(rtmClock * 1e-6), 
                self.LOG_USER)
            return


        FastSlowRstValue = np.floor((2**self.num_flux_ramp_counter_bits) * 
            (1 - fractionFullScale)/2)


        KRelay = 3 #where do these values come from
        SelectRamp = self.get_select_ramp() # from config file
        RampStartMode = self.get_ramp_start_mode() # from config file
        PulseWidth = 400
        DebounceWidth = 255
        RampSlope = 0
        ModeControl = 0
        EnableRampTrigger = 1

        self.set_low_cycle(LowCycle, write_log=write_log) 
        self.set_high_cycle(HighCycle, write_log=write_log)
        self.set_k_relay(KRelay, write_log=write_log)
        self.set_ramp_max_cnt(rampMaxCnt, write_log=write_log)
        self.set_select_ramp(SelectRamp, write_log=write_log)
        self.set_ramp_start_mode(RampStartMode, write_log=write_log)
        self.set_pulse_width(PulseWidth, write_log=write_log)
        self.set_debounce_width(DebounceWidth, write_log=write_log)
        self.set_ramp_slope(RampSlope, write_log=write_log)
        self.set_mode_control(ModeControl, write_log=write_log)
        self.set_fast_slow_step_size(FastSlowStepSize, write_log=write_log)
        self.set_fast_slow_rst_value(FastSlowRstValue, write_log=write_log)
        self.set_enable_ramp_trigger(EnableRampTrigger, write_log=write_log)
        self.set_ramp_rate(reset_rate_khz, write_log=write_log)

    def check_lock(self, band, f_min=.05, f_max=.2, df_max=.03,
        make_plot=False, flux_ramp=True, fraction_full_scale=.99,
        lms_freq_hz=4000., reset_rate_khz=4., **kwargs):
        """
        Checks the bad resonators
        
        Args:
        -----
        band (int) : The band the check

        Opt Args:
        ---------
        f_min (float) : The maximum frequency swing.
        f_max (float) : The minimium frequency swing
        df_max (float) : The maximum value of the stddev of df
        make_plot (bool) : Whether to make plots. Default False
        flux_ramp (bool) : Whether to flux ramp or not. Default True
        faction_full_scale (float): Number between 0 and 1. The amplitude
           of the flux ramp.
        """
        self.log('Checking lock on band {}'.format(band))
        
        channels = self.which_on(band)
        n_chan = len(channels)
        
        self.log('Currently {} channels on'.format(n_chan))

        # Tracking setup returns information on all channels in a band
        f, df, sync = self.tracking_setup(band, 0, make_plot=False,
            flux_ramp=flux_ramp, fraction_full_scale=fraction_full_scale,
            lms_freq_hz=lms_freq_hz, reset_rate_khz=reset_rate_khz)

        high_cut = np.array([])
        low_cut = np.array([])
        df_cut = np.array([])

        if make_plot:
            import matplotlib.pyplot as plt

        for ch in channels:
            f_chan = f[:,ch]
            f_span = np.max(f_chan) - np.min(f_chan)
            df_rms = np.std(df[:,ch])

            self.log('Ch {} f_span {} df_rms {}'.format(ch, f_span, df_rms))

            if make_plot:
                plt.figure()
                plt.plot(f_chan)
                plt.title(ch)

            if f_span > f_max:
                self.set_amplitude_scale_channel(band, ch, 0, **kwargs)
                high_cut = np.append(high_cut, ch)
            elif f_span < f_min:
                self.set_amplitude_scale_channel(band, ch, 0, **kwargs)
                low_cut = np.append(low_cut, ch)
            elif df_rms > df_max:
                self.set_amplitude_scale_channel(band, ch, 0, **kwargs)
                df_cut = np.append(df_cut, ch)
            #else:
            #    self.log('Ch {:03} acceptable: {:4.3f}'.format(ch, f_span))

        chan_after = self.which_on(band)
        
        self.log('High cut channels {}'.format(high_cut))
        self.log('Low cut channels {}'.format(low_cut))
        self.log('df cut channels {}'.format(df_cut))
        self.log('Good channels {}'.format(chan_after))
        self.log('High cut count: {}'.format(len(high_cut)))
        self.log('Low cut count: {}'.format(len(low_cut)))
        self.log('df cut count: {}'.format(len(df_cut)))
        self.log('Started with {}. Now {}'.format(n_chan, len(chan_after)))

        timestamp = self.get_timestamp(as_int=True)
        self.freq_resp[band]['lock_status'][timestamp] = {
            'action' : 'check_lock',
            'flux_ramp': flux_ramp,
            'f_min' : f_min,
            'f_max' : f_max,
            'df_max' : df_max,
            'high_cut' : high_cut,
            'low_cut' : low_cut,
            'channels_before' : channels,
            'channels_after' : chan_after
        }


    def check_lock_flux_ramp_off(self, band,df_max=.03,
                   make_plot=False, **kwargs):
        """
        Simple wrapper function for check_lock with the flux ramp off
        """
        self.check_lock(band, f_min=0., f_max=np.inf, df_max=df_max, 
            make_plot=make_plot, flux_ramp=False, **kwargs)


    def find_freq(self, band, subband=np.arange(13,115), drive_power=10,
        n_read=2, make_plot=False, save_plot=True, window=50, rolling_med=True):
        '''
        Finds the resonances in a band (and specified subbands)

        Args:
        -----
        band (int) : The band to search

        Optional Args:
        --------------
        subband (int) : An int array for the subbands
        drive_power (int) : The drive amplitude
        n_read (int) : The number sweeps to do per subband
        make_plot (bool) : make the plot frequency sweep. Default False.
        save_plot (bool) : save the plot. Default True.
        save_name (string) : What to name the plot. default find_freq.png
        rolling_med (bool) : Whether to iterate on a rolling median or just
           the median of the whole sample.
        window (int) : The width of the rolling median window
        '''
        self.log('Sweeping across frequencies')
        f, resp = self.full_band_ampl_sweep(band, subband, drive_power, n_read)

        timestamp = self.get_timestamp()

        # Save data
        save_name = '{}_amp_sweep_{}.txt'
        np.savetxt(os.path.join(self.output_dir, 
            save_name.format(timestamp, 'freq')), f)
        np.savetxt(os.path.join(self.output_dir, 
            save_name.format(timestamp, 'resp')), resp)

        # Place in dictionary - dictionary declared in smurf_control
        self.freq_resp[band]['find_freq'] = {}
        self.freq_resp[band]['find_freq']['subband'] = subband
        self.freq_resp[band]['find_freq']['f'] = f
        self.freq_resp[band]['find_freq']['resp'] = resp
        if 'timestamp' in self.freq_resp[band]['find_freq']:
            self.freq_resp[band]['timestamp'] = \
                np.append(self.freq_resp[band]['find_freq']['timestamp'], timestamp)
        else:
            self.freq_resp[band]['find_freq']['timestamp'] = np.array([timestamp])

        # Find resonances
        res_freq = self.find_all_peak(self.freq_resp[band]['find_freq']['f'],
            self.freq_resp[band]['find_freq']['resp'], subband, make_plot=make_plot,
            band=band, rolling_med=rolling_med, window=window)
        self.freq_resp[band]['find_freq']['resonance'] = res_freq

        # Save resonances
        np.savetxt(os.path.join(self.output_dir,
            save_name.format(timestamp, 'resonance')), 
            self.freq_resp[band]['find_freq']['resonance'])

        # Call plotting
        if make_plot:
            self.plot_find_freq(self.freq_resp[band]['find_freq']['f'], 
                self.freq_resp[band]['find_freq']['resp'], save_plot=save_plot, 
                save_name=save_name.replace('.txt', '.png').format(timestamp,
                    band))

        return f, resp

    def plot_find_freq(self, f=None, resp=None, subband=None, filename=None, 
        save_plot=True, save_name='amp_sweep.png'):
        '''
        Plots the response of the frequency sweep. Must input f and resp, or
        give a path to a text file containing the data for offline plotting.

        To do:
        Add ability to use timestamp and multiple plots

        Optional Args:
        --------------
        save_plot (bool) : save the plot. Default True.
        save_name (string) : What to name the plot. default find_freq.png
        '''
        if subband is None:
            subband = np.arange(128)
        subband = np.asarray(subband)

        if (f is None or resp is None) and filename is None:
            self.log('No input data or file given. Nothing to plot.')
            return
        else:
            if filename is not None:
                f, resp = np.load(filename)

            import matplotlib.pyplot as plt
            cm = plt.cm.get_cmap('viridis')
            fig = plt.figure(figsize=(10,4))

            for i, sb in enumerate(subband):
                color = cm(float(i)/len(subband)/2. + .5*(i%2))
                plt.plot(f[sb,:], np.abs(resp[sb,:]), '.', markersize=4, 
                    color=color)
            plt.title("findfreq response")
            plt.xlabel("Frequency offset (MHz)")
            plt.ylabel("Normalized Amplitude")

            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, save_name),
                    bbox_inches='tight')


    def full_band_ampl_sweep(self, band, subband, drive, N_read):
        """sweep a full band in amplitude, for finding frequencies

        args:
        -----
            band (int) = bandNo (500MHz band)
            subband (int) = which subbands to sweep
            drive (int) = drive power (defaults to 10)
            n_read (int) = numbers of times to sweep, defaults to 2

        returns:
        --------
            freq (list, n_freq x 1) = frequencies swept
            resp (array, n_freq x 2) = complex response
        """

        digitizer_freq = self.get_digitizer_frequency_mhz(band)  # in MHz
        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        band_center = self.get_band_center_mhz(band)  # in MHz

        subband_width = 2 * digitizer_freq / n_subbands

        scan_freq = np.arange(-3, 3.05, 0.05)  # take out this hardcode

        resp = np.zeros((n_subbands, np.shape(scan_freq)[0]), dtype=complex)
        freq = np.zeros((n_subbands, np.shape(scan_freq)[0]))

        subband_nos, subband_centers = self.get_subband_centers(band)

        self.log('Working on band {:d}'.format(band))
        for sb in subband:
            self.log('Sweeping subband no: {}'.format(sb))
            f, r = self.fast_eta_scan(band, sb, scan_freq, N_read, 
                drive)
            resp[sb,:] = r
            freq[sb,:] = f
            freq[sb,:] = scan_freq + \
                subband_centers[subband_nos.index(sb)]
        return freq, resp


    def peak_finder(self, x, y, threshold):
        """finds peaks in x,y data with some threshhold

        Not currently being used
        """
        in_peak = 0

        peakstruct_max = []
        peakstruct_nabove = []
        peakstruct_freq = []

        for idx in range(len(y)):
            freq = x[idx]
            amp = y[idx]

            if in_peak == 0:
                pk_max = 0
                pk_freq = 0
                pk_nabove = 0

            if amp > threshold:
                if in_peak == 0: # start a new peak
                    n_peaks = n_peaks + 1

                in_peak = 1
                pk_nabove = pk_nabove + 1

                if amp > pk_max: # keep moving until find the top
                    pk_max = amp
                    pk_freq = freq

                if idx == len(y) or y[idx + 1] < threshhold:
                    peakstruct_max.append(pk_max)
                    peakstruct_nabove.append(pk_nabove)
                    peakstruct_freq.append(pk_freq)
                    in_peak = 0
        return peakstruct_max, peakstruct_nabove, peakstruct_freq

    def find_peak_2(self, freq, resp, normalize=False, 
        n_samp_drop=1, threshold=.5, margin_factor=1., phase_min_cut=1, 
        phase_max_cut=1, make_plot=False, save_plot=True, save_name=None):
        """find the peaks within a given subband

        Args:
        -----
        freq (vector): should be a single row of the broader freq array
        response (complex vector): complex response for just this subband

        Optional Args:
        --------------
        normalize (bool) : 
        n_samp_drop (int) :
        threshold (float) :
        margin_factor (float):
        phase_min_cut (int) :
        phase_max_cut (int) :

        Returns:
        -------_
        resonances (list of floats) found in this subband
        """
        resp_input = np.copy(resp)
        if np.isnan(resp).any():
            if np.isnan(resp).all():
                self.log("Warning - All values are NAN. Skipping", 
                    self.LOG_ERROR)                
                return
            self.log("Warning - at least one NAN. Interpolating...", 
                self.LOG_ERROR)
            idx = ~np.isnan(resp)
            resp = np.interp(freq, freq[idx], resp[idx])

        # This was what was in Cyndia and Shawns code. Im deprecating this
        # for now.
        # df = freq[1] - freq[0]
        # Idat = np.real(resp)
        # Qdat = np.imag(resp)
        # phase = np.unwrap(np.arctan2(Qdat, Idat))

        # diff_phase = np.diff(phase)
        # diff_freq = np.add(freq[:-1], df / 2)  # lose an index from diff

        # if normalize==True:
        #     norm_min = min(diff_phase[nsampdrop:-nsampdrop])
        #     norm_max = max(diff_phase[nsampdrop:-nsampdrop])

        #     diff_phase = (diff_phase - norm_min) / (norm_max - norm_min)
        #
        # peakstruct_max, peakstruct_nabove, peakstruct_freq = \
        #     self.peak_finder(diff_freq, diff_phase, threshold)

        # return peakstruct_max, peakstruct_nabove, peakstruct_freq

        # For now use scipy - needs scipy
        # hardcoded values should be exposed in some meaningful way

        import scipy.signal as signal
        peak_ind, props = signal.find_peaks(-np.abs(resp), distance=10, 
            prominence=.025)

        if make_plot:
            self.plot_find_peak(freq, resp_input, peak_ind, save_plot=save_plot,
                save_name=save_name)

        return freq[peak_ind]

    def plot_find_peak(self, freq, resp, peak_ind, save_plot=True, 
        save_name=None):
        """
        """
        import matplotlib.pyplot as plt
        if save_plot:
            plt.ioff()
        else:
            plt.ion()

        Idat = np.real(resp)
        Qdat = np.imag(resp)
        phase = np.unwrap(np.arctan2(Qdat, Idat))
        
        fig, ax = plt.subplots(2, sharex=True, figsize=(6,4))
        ax[0].plot(freq, np.abs(resp), label='amp', color='b')
        ax[0].plot(freq, Idat, label='I', color='r', linestyle=':', alpha=.5)
        ax[0].plot(freq, Qdat, label='Q', color='g', linestyle=':', alpha=.5)
        ax[0].legend(loc='lower right')
        ax[1].plot(freq, phase, color='b')
        ax[1].set_ylim((-np.pi, np.pi))

        if len(peak_ind):  # empty array returns False
            ax[0].plot(freq[peak_ind], np.abs(resp[peak_ind]), 'x', color='k')
            ax[1].plot(freq[peak_ind], phase[peak_ind], 'x', color='k')
        else:
            self.log('No peak_ind values.', self.LOG_USER)

        fig.suptitle("Peak Finding")
        ax[1].set_xlabel("Frequency offset from Subband Center (MHz)")
        ax[0].set_ylabel("Response")
        ax[1].set_ylabel("Phase [rad]")

        if save_plot:
            if save_name is None:
                self.log('Using default name for saving: find_peak.png \n' +
                    'Highly recommended that you input a non-default name')
                save_name = 'find_peak.png'
            else:
                self.log('Plotting saved to {}'.format(save_name))
            plt.savefig(os.path.join(self.plot_dir, save_name),
                bbox_inches='tight')
            plt.close()

    def find_all_peak(self, freq, resp, subband=None, rolling_med=False, 
        window=500, grad_cut=0.05, amp_cut=0.25, freq_min=-2.5E8, freq_max=2.5E8, 
        make_plot=False, save_plot=True, band=None, make_subband_plot=False, 
        subband_plot_with_slow=False, timestamp=None, pad=2, min_gap=2):
        """
        find the peaks within each subband requested from a fullbandamplsweep

        Args:
        -----
        freq (array):  (n_subbands x n_freq_swept) array of frequencies swept
        response (complex array): n_subbands x n_freq_swept array of complex 
            response
        subbands (list of ints): subbands that we care to search in

        Optional Args:
        --------------
	see find_peak for optional arguments. Used the same defaults here.
        """
        peaks = np.array([])
        subbands = np.array([])
        timestamp = self.get_timestamp()

        #for sb in subband:
        #    peak = self.find_peak(freq[sb,:], resp[sb,:], 
        #        rolling_med=rolling_med, window=window, grad_cut=grad_cut,
        #        amp_cut=amp_cut, freq_min=freq_min, freq_max=freq_max, 
        #        make_plot=make_plot, save_plot=save_plot, band=band, 
        #        make_subband_plot=make_subband_plot, 
        #        subband_plot_with_slow=subband_plot_with_slow, 
        #        timestamp=timestamp, pad=pad, min_gap=min_gap,
        #        subband=sb)

        #    if peak is not None:
        #        peaks = np.append(peaks, peak)
        #        subbands = np.append(subbands, 
        #            np.ones_like(peak, dtype=int)*sb)

        # Stack all the frequency and response data into a 
        sb, _ = np.where(freq !=0)
        idx = np.unique(sb)
        f_stack = np.ravel(freq[idx])
        r_stack = np.ravel(resp[idx])
        
        # Frequency is interleaved, so sort it
        s = np.argsort(f_stack)
        f_stack = f_stack[s]
        r_stack = r_stack[s]

        peaks = self.find_peak(f_stack, r_stack,
                rolling_med=rolling_med, window=window, grad_cut=grad_cut,
                amp_cut=amp_cut, freq_min=freq_min, freq_max=freq_max,
                make_plot=make_plot, save_plot=save_plot, band=band,
                make_subband_plot=make_subband_plot,
                subband_plot_with_slow=subband_plot_with_slow,
                timestamp=timestamp, pad=pad, min_gap=min_gap)


        #res = np.vstack((peaks, subbands))
        #return res
        return peaks

    def fast_eta_scan(self, band, subband, freq, n_read, drive, 
        make_plot=False):
        """copy of fastEtaScan.m from Matlab. Sweeps quickly across a range of
        freq and gets I, Q response

        Args:
         band (int): which 500MHz band to scan
         subband (int): which subband to scan
         freq (n_freq x 1 array): frequencies to scan relative to subband 
            center
         n_read (int): number of times to scan
         drive (int): tone power

        Optional Args:
        make_plot (bool): Make eta plots

        Outputs:
         resp (n_freq x 2 array): real, imag response as a function of 
            frequency
         freq (n_freq x n_read array): frequencies scanned, relative to 
            subband center
        """
        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)

        channel_order = self.get_channel_order(None) # fix this later

        channels_per_subband = int(n_channels / n_subbands)
        first_channel_per_subband = channel_order[0::channels_per_subband]
        subchan = first_channel_per_subband[subband]

        self.set_eta_scan_freq(band, freq)
        self.set_eta_scan_amplitude(band, drive)
        self.set_eta_scan_channel(band, subchan)
        self.set_eta_scan_dwell(band, 0)

        self.set_run_eta_scan(band, 1)

        I = self.get_eta_scan_results_real(band, count=len(freq))
        Q = self.get_eta_scan_results_imag(band, count=len(freq))

        self.band_off(band)

        response = np.zeros((len(freq), ), dtype=complex)

        for index in range(len(freq)):
            Ielem = I[index]
            Qelem = Q[index]
            if Ielem > 2**23:
                Ielem = Ielem - 2**24
            if Qelem > 2**23:
                Qelem = Qelem - 2**24
            
            Ielem = Ielem / 2**23
            Qelem = Qelem / 2**23

            response[index] = Ielem + 1j*Qelem

        if make_plot:
            import matplotlib.pyplot as plt
            # To do : make plotting

        return freq, response

    def setup_notches(self, band, resonance=None, drive=10, sweep_width=.3, 
        df_sweep=.005, subband_half_width=614.4/128,min_offset = 0.1):
        """

        Args:
        -----
        band (int) : The 500 MHz band to setup.

        Optional Args:
        --------------
        resonance (float array) : A 2 dimensional array with resonance 
            frequencies and the subband they are in. If given, this will take 
            precedent over the one in self.freq_resp.
        drive (int) : The power to drive the resonators. Default 10.
        sweep_width (float) : The range to scan around the input resonance in
            units of MHz. Default .3
        sweep_df (float) : The sweep step size in MHz. Default .005
        min_offset (float): minimum distance in MHz between two resonators for assigning channels

        Returns:
        --------

        """

        # Check if any resonances are stored
        if 'resonance' not in self.freq_resp[band]['find_freq'] and resonance is None:
            self.log('No resonances stored in band {}'.format(band) +
                '. Run find_freq first.', self.LOG_ERROR)
            return

        if resonance is not None:
            input_res = resonance
        else:
            input_res = self.freq_resp[band]['find_freq']['resonance']

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        n_subchannels = n_channels / n_subbands

        sweep = np.arange(-sweep_width, sweep_width+df_sweep, df_sweep)

        self.freq_resp[band]['drive'] = drive

        # Loop over inputs and do eta scans
        resonances = {}
        band_center = self.get_band_center_mhz(band)
        input_res = input_res + band_center
        # for i, (f, sb) in enumerate(zip(input_res, input_subband)):
        n_res = len(input_res)
        for i, f in enumerate(input_res):
            # self.log('freq {:5.4f} sb {}'.format(f, sb))
            self.log('freq {:5.4f} - {} of {}'.format(f, i+1, n_res))
            freq, resp, eta = self.eta_estimator(band, f, drive, 
                                                 f_sweep_half=sweep_width,
                                                 df_sweep=df_sweep)
            eta_phase_deg = np.angle(eta)*180/np.pi
            eta_mag = np.abs(eta)
            eta_scaled = eta_mag / subband_half_width
            
            abs_resp = np.abs(resp)
            idx = np.ravel(np.where(abs_resp == np.min(abs_resp)))[0]
            # _, sbc = self.get_subband_centers(band, as_offset=True)

            f_min = freq[idx]

            resonances[i] = {
                'freq' : f_min,
                'eta' : eta,
                'eta_scaled' : eta_scaled,
                'eta_phase' : eta_phase_deg,
                'r2' : 1,  # This is BS
                'eta_mag' : eta_mag,
                'latency': 0,  # This is also BS
                'Q' : 1,  # This is also also BS
                'freq_eta_scan' : freq,
                'resp_eta_scan' : resp
            }

        self.freq_resp[band]['resonances'] = resonances

        # Assign resonances to channels                                                       
        self.log('Assigning channels')
        f = [resonances[k]['freq'] for k in resonances.keys()]
        subbands, channels, offsets = self.assign_channels(f, band=band, 
                                        as_offset=False,min_offset=min_offset)

        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})

        self.save_tune()

        self.relock(band)
    
    def save_tune(self, update_last_tune=True):
        """
        Saves the tuning information (self.freq_resp) to tuning directory
        """
        timestamp = self.get_timestamp()
        savedir = os.path.join(self.tune_dir, timestamp+"_tune")
        self.log('Saving to : {}.npy'.format(savedir))
        np.save(savedir, self.freq_resp)

        return savedir + ".npy"

    def load_tune(self, filename):
        """
        Loads the tuning information (self.freq_resp) from tuning directory

        Args:
        -----
        filename (str) : The name of the tuning.
        """
        fs = np.load(filename).item()
        self.log('Loaded...')
        self.log('{}'.format(fs))

        self.freq_resp = fs
