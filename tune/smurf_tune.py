import numpy as np
import os
import glob
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

    def tune(self, load_tune=True, tune_file=None, last_tune=False,
             retune=False, f_min=.02, f_max=.3, df_max=.03,
             fraction_full_scale=None, make_plot=False,
             save_plot=True, show_plot=False,
             new_master_assignment=False, track_and_check=True):
        """
        This runs a tuning, does tracking setup, and prunes bad
        channels using check lock. When this is done, we should
        be ready to take data.

        Args:
        -----
        load_tune (bool): Whether to load in a tuning file. If False, will
            do a full tuning. This will be very slow (~ 1 hour)
        tune_file (str): The tuning file to load in. Default is None. If
            tune_file is None and last_tune is False, this will load the
            default tune file defined in exp.cfg.
        last_tune (bool): Whether to load the most recent tuning file. Default
            is False.
        retune (bool): Whether to re-run tune_band_serial to refind peaks and
            eta params. This will take about 5 minutes. Default is False.
        f_min (float): The minimum frequency swing allowable for check_lock.
        f_max (float): The maximum frequency swing allowable for check_lock.
        df_max (float): The maximum df stddev allowable for check_lock.
        make_plot (bool): Whether to make a plot. Default is False.
        save_plot (bool): If making plots, whether to save them. Default is True.
        show_plot (bool): Whether to display the plots to screen. Default is False.
        track_and_check (bool): Whether or not after tuning to run
            track and check.  Default is True.
        """
        bands = self.config.get('init').get('bands')
        tune_cfg = self.config.get('tune_band')
        
        # Load fraction_full_scale from file if not given
        if fraction_full_scale is None:
            fraction_full_scale = tune_cfg.get('fraction_full_scale')
        
        if load_tune:
            if last_tune:
                tune_file = self.last_tune()
                self.log('Last tune is : {}'.format(tune_file))
            elif tune_file is None:
                tune_file = tune_cfg.get('default_tune')
                self.log('Loading default tune file: {}'.format(tune_file))
            self.load_tune(tune_file)

        # Runs find_freq and setup_notches. This takes forever.
        else:
            cfg = self.config.get('init')
            for b in bands:
                drive = cfg.get('band_{}'.format(b)).get('amplitude_scale')
                self.find_freq(b, 
                    drive_power=drive)
                self.setup_notches(b, drive=drive, 
                                   new_master_assignment=new_master_assignment)

        # Runs tune_band_serial to re-estimate eta params
        if retune:
            for b in bands:
                self.log('Running tune band serial on band {}'.format(b))
                self.tune_band_serial(b, from_old_tune=load_tune,
                                      old_tune=tune_file, make_plot=make_plot,
                                      show_plot=show_plot, save_plot=save_plot,
                                      new_master_assignment=new_master_assignment)

        # Starts tracking and runs check_lock to prune bad resonators
        if track_and_check:
            for b in bands:
                self.log('Tracking and checking band {}'.format(b))
                self.track_and_check(b, fraction_full_scale=fraction_full_scale, 
                                     f_min=f_min,
                                     f_max=f_max, df_max=df_max, make_plot=make_plot,
                                     save_plot=save_plot, show_plot=show_plot)
        

    def tune_band(self, band, freq=None, resp=None, n_samples=2**19, 
        make_plot=False, show_plot=False, plot_chans=[], save_plot=True, save_data=True, 
        make_subband_plot=False, subband=None, n_scan=5,
        subband_plot_with_slow=False, drive=None,
        grad_cut=.05, freq_min=-2.5E8, freq_max=2.5E8, amp_cut=.5,
        use_slow_eta=False):
        """
        This does the full_band_resp, which takes the raw resonance data.
        It then finds the where the resonances are. Using the resonance
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
                n_scan=n_scan, show_plot=show_plot)


        # Find peaks
        peaks = self.find_peak(freq, resp, rolling_med=True, band=band, 
            make_plot=make_plot, show_plot=show_plot, window=5000,
            save_plot=save_plot, grad_cut=grad_cut, freq_min=freq_min,
            freq_max=freq_max, amp_cut=amp_cut, 
            make_subband_plot=make_subband_plot, timestamp=timestamp,
            subband_plot_with_slow=subband_plot_with_slow, pad=50, min_gap=50)

        # Eta scans
        band_center_mhz = self.get_band_center_mhz(band)
        resonances = {}
        for i, p in enumerate(peaks):
            eta, eta_scaled, eta_phase_deg, r2, eta_mag, latency, Q= \
                self.eta_fit(freq, resp, p, 50E3, make_plot=False, 
                plot_chans=plot_chans, save_plot=save_plot, res_num=i, 
                band=band, timestamp=timestamp, use_slow_eta=use_slow_eta)

            resonances[i] = {
                'freq': p*1.0E-6 + band_center_mhz,
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
        f = [resonances[k]['freq'] for k in resonances.keys()]
        subbands, channels, offsets = self.assign_channels(f, band=band)

        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})

        self.freq_resp[band]['resonances'] = resonances
        if drive is None:
            drive = self.config.get('init').get('band_{}'.format(band)).get('amplitude_scale')
        self.freq_resp[band]['drive'] = drive

        self.save_tune()
#        np.save(os.path.join(self.output_dir, 
#            '{}_freq_resp'.format(timestamp)), self.freq_resp)

        self.relock(band)
        self.log('Done')
        return resonances

    def tune_band_serial(self, band, n_samples=2**19,
        make_plot=False, save_plot=True, save_data=True, show_plot=False,
        make_subband_plot=False, subband=None, n_scan=5,
        subband_plot_with_slow=False, window=5000, rolling_med=True,
        grad_cut=.03, freq_min=-2.5E8, freq_max=2.5E8, amp_cut=.25,
        del_f=.005, drive=None, new_master_assignment=False, from_old_tune=False,
        old_tune=None, pad=50, min_gap=50):
        """
        Tunes band using serial_gradient_descent and then
        serial_eta_scan. This takes about 3 minutes per band if there
        are about 150 resonators.

        Args:
        -----
        band (int): The band the tune

        Opt Args:
        ---------
        from_old_tune (bool): Whether to use an old tuning file. This
            will load a tuning file and use its peak frequencies as 
            a starting point for seria_gradient_descent.
        old_tune (str): The full path to the tuning file.
        new_master_assignment (bool): Whether to overwrite the previous
            master_assignment list. Default is False.
        make_plot (bool): Whether to make plots. Default is False.
        save_plot (bool): If make_make plot is True, whether to save
            the plots. Default is True.
        show_plot (bool): If make_plot is True, whether to display
            the plots to screen.
        """
        timestamp = self.get_timestamp()
        center_freq = self.get_band_center_mhz(band)

        self.flux_ramp_off()  # flux ramping messes up eta params

        freq=None
        resp=None
        if from_old_tune:
            if old_tune is None:
                self.log('Using default tuning file')
                old_tune = self.config.get('tune_band').get('default_tune')
            self.load_tune(old_tune,band=band)

            resonances = np.copy(self.freq_resp[band]['resonances']).item()

            if new_master_assignment:
                f = np.array([resonances[k]['freq'] for k in resonances.keys()])
                # f += self.get_band_center_mhz(band)
                subbands, channels, offsets = self.assign_channels(f, band=band,
                    as_offset=False, new_master_assignment=new_master_assignment)

                for i, k in enumerate(resonances.keys()):
                    resonances[k].update({'subband': subbands[i]})
                    resonances[k].update({'channel': channels[i]})
                    resonances[k].update({'offset': offsets[i]})
                self.freq_resp[band]['resonances'] = resonances

        else:
            # Inject high amplitude noise with known waveform, measure it, and
            # then find resonators and etaParameters from cross-correlation.
            old_att = self.get_att_uc(band)
            self.set_att_uc(band, 0, wait_after=.5, write_log=True)
            self.get_att_uc(band, write_log=True)
            freq, resp = self.full_band_resp(band, n_samples=n_samples,
                                         make_plot=make_plot, save_data=save_data, 
                                         show_plot=False, timestamp=timestamp,
                                         n_scan=n_scan)
            self.set_att_uc(band, old_att, write_log=True)

            # Find peaks
            peaks = self.find_peak(freq, resp, rolling_med=rolling_med, window=window,
                               band=band, make_plot=make_plot, save_plot=save_plot, 
                               show_plot=show_plot, grad_cut=grad_cut, freq_min=freq_min,
                               freq_max=freq_max, amp_cut=amp_cut,
                               make_subband_plot=make_subband_plot, timestamp=timestamp,
                               subband_plot_with_slow=subband_plot_with_slow, pad=pad, 
                               min_gap=min_gap)

            resonances = {}
            for i, p in enumerate(peaks):
                resonances[i] = {
                    'freq': p*1.0E-6 + center_freq,  # in MHz
                    'r2' : 0,
                    'Q' : 1,
                    'eta_phase' : 1 , # Fill etas with arbitrary values for now
                    'eta_scaled' : 1,
                    'eta_mag' : 0,
                    'eta' : 0 + 0.j
                }

            # Assign resonances to channels
            self.log('Assigning channels')
            f = np.array([resonances[k]['freq'] for k in resonances.keys()])
            subbands, channels, offsets = self.assign_channels(f, band=band, 
                as_offset=False, new_master_assignment=new_master_assignment)

            for i, k in enumerate(resonances.keys()):
                resonances[k].update({'subband': subbands[i]})
                resonances[k].update({'channel': channels[i]})
                resonances[k].update({'offset': offsets[i]})
                self.freq_resp[band]['resonances'] = resonances

        if drive is None:
            drive = \
                self.config.get('init')['band_{}'.format(band)]['amplitude_scale']
        self.freq_resp[band]['drive'] = drive
        self.freq_resp[band]['full_band_resp'] = {}
        if freq is not None:
            self.freq_resp[band]['full_band_resp']['freq'] = freq * 1.0E-6 + center_freq
        if resp is not None:
            self.freq_resp[band]['full_band_resp']['resp'] = resp
        self.freq_resp[band]['timestamp'] = timestamp


        # Set the resonator frequencies without eta params 
        self.relock(band, drive=drive)

        # Find the resonator minima
        self.log('Finding resonator minima...')
        self.run_serial_gradient_descent(band, timeout=1200)
        #self.run_serial_min_search(band)
        
        # Calculate the eta params
        self.log('Calculating eta parameters...')
        self.run_serial_eta_scan(band, timeout=1200)

        # Read back new eta parameters and populate freq_resp
        subband_half_width = self.get_digitizer_frequency_mhz(band)/\
            self.get_number_sub_bands(band)
        eta_phase = self.get_eta_phase_array(band)
        eta_scaled = self.get_eta_mag_array(band)
        eta_mag = eta_scaled * subband_half_width
        eta = eta_mag * np.cos(np.deg2rad(eta_phase)) + \
            1.j * np.sin(np.deg2rad(eta_phase))
        
        chs = self.get_eta_scan_result_channel(band)

        chs = self.get_eta_scan_result_channel(band)

        for i, ch in enumerate(chs):
            if ch != -1:
                resonances[i]['eta_phase'] = eta_phase[ch]
                resonances[i]['eta_scaled'] = eta_scaled[ch]
                resonances[i]['eta_mag'] = eta_mag[ch]
                resonances[i]['eta'] = eta[ch]

        self.freq_resp[band]['resonances'] = resonances

        self.save_tune()

        self.log('Done with serial tuning')

    def tune_band_parallel(self, band, n_samples=2**19,
        make_plot=False, save_plot=True, save_data=True, show_plot=False,
        make_subband_plot=False, subband=None, n_scan=2,
        subband_plot_with_slow=False, window=5000, rolling_med=True,
        grad_cut=.025, freq_min=-2.5E8, freq_max=2.5E8, amp_cut=.25,
        load_peaks_from_file=True, del_f=.005):
        """
        Uses parallel eta scans to tune the band. This does not work 
        very well. Use tune_band_serial instead.
        """
        timestamp = self.get_timestamp()

        if make_plot:
            import matplotlib.pyplot as plt
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

        self.band_off(band)
        self.flux_ramp_off()
        self.log('Running full band resp')

        if load_peaks_from_file:
            peaks = np.loadtxt('/home/cryo/ey/all_resonators.txt', delimiter=',')
            if band == 2:
                peaks = peaks[peaks < 5.500] - 5.25
            elif band == 3:
                peaks = peaks[peaks > 5.500] - 5.75
            peaks *= 1.0E9

        else:
            # Inject high amplitude noise with known waveform, measure it, and 
            # then find resonators and etaParameters from cross-correlation.
            att_uc = self.get_att_uc(band)
            drive = self.config.get('init')['band_{}'.format(band)]['amplitude_scale']
            self.set_att_uc(band, (15-drive)*6+att_uc, write_log=True)
            freq, resp = self.full_band_resp(band, n_samples=n_samples,
                make_plot=make_plot, save_data=save_data, show_plot=False,
                timestamp=timestamp,
                n_scan=n_scan)

            self.set_att_uc(band, att_uc)

            # Find peaks
            peaks = self.find_peak(freq, resp, rolling_med=rolling_med, window=window, 
                band=band, make_plot=make_plot,
                save_plot=save_plot, show_plot=show_plot, grad_cut=grad_cut, freq_min=freq_min,
                freq_max=freq_max, amp_cut=amp_cut,
                make_subband_plot=make_subband_plot, timestamp=timestamp,
                subband_plot_with_slow=subband_plot_with_slow, pad=50, min_gap=50)

        # Assign resonances
        resonances = {}
        for i, p in enumerate(peaks):
            resonances[i] = {
                'freq': p*1.0E-6,  # in MHz
                'r2' : 0,
                'Q' : 1,
                'eta_phase' : 1 , # Fill etas with arbitrary values for now
                'eta_scaled' : 1
            }
        
        # Assign resonances to channels                                              
        self.log('Assigning channels')
        f = [resonances[k]['freq'] for k in resonances.keys()]
        subbands, channels, offsets = self.assign_channels(f, band=band)
        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})
        self.freq_resp[band]['resonances'] = resonances
        self.freq_resp[band]['drive'] = \
            self.config.get('init')['band_{}'.format(band)]['amplitude_scale']

        # Set the resonator frequencies without eta params
        self.relock(band)
        
        # Run parallel eta scans
        self.set_eta_scan_del_f(band, del_f)
        self.run_parallel_eta_scan(band)

        eta_phase = self.get_eta_phase_array(band)
        eta_scaled = self.get_eta_mag_array(band)
        chs = self.get_eta_scan_result_channel(band)
        for i, ch in enumerate(chs):
            if ch != -1:
                resonances[i]['eta_phase'] = eta_phase[ch]
                resonances[i]['eta_scaled'] = eta_scaled[ch]

        self.freq_resp[band]['resonances'] = resonances

        self.save_tune()


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

        timestamp = self.get_timestamp()            

        fig, ax = plt.subplots(2,2, figsize=(10,6))

        # Subband
        sb = self.get_eta_scan_result_subband(band)
        ch = self.get_eta_scan_result_channel(band)
        idx = np.where(ch!=-1)  # ignore unassigned channels
        sb = sb[idx]
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

        # Eta stuff
        eta = self.get_eta_scan_result_eta(band)
        eta = eta[idx]
        f = self.get_eta_scan_result_freq(band)
        f = f[idx]

        ax[0,1].plot(f, np.real(eta), '.', label='Real')
        ax[0,1].plot(f, np.imag(eta), '.', label='Imag')
        ax[0,1].plot(f, np.abs(eta), '.', label='Abs', color='k')
        ax[0,1].legend(loc='lower right')
        bc = self.get_band_center_mhz(band)
        ax[0,1].set_xlim((bc-250, bc+250))
        ax[0,1].set_xlabel('Freq [MHz]')
        ax[0,1].set_ylabel('Eta')

        phase = np.rad2deg(np.angle(eta))
        ax[1,1].plot(f, phase, color='k')
        ax[1,1].set_xlim((bc-250, bc+250))
        ax[1,1].set_ylim((-180,180))
        ax[1,1].set_yticks(np.arange(-180, 180.1, 90))
        ax[1,1].set_xlabel('Freq [MHz]')
        ax[1,1].set_ylabel('Eta phase')

        fig.suptitle('Band {} {}'.format(band, timestamp))
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
            if 'full_band_resp' in self.freq_resp[band]:
                freq = self.freq_resp[band]['full_band_resp']['freq']
                resp = self.freq_resp[band]['full_band_resp']['resp']
                for k in keys:
                    r = self.freq_resp[band]['resonances'][k]
                    width = .300 # 300 kHz
                    center_freq = r['freq']
                    idx = np.logical_and(freq > center_freq - width,
                                         freq < center_freq + width)

                    self.plot_eta_fit(freq[idx], resp[idx], 
                                      eta_mag=r['eta_mag'], eta_phase_deg=r['eta_phase'],
                                      band=band, res_num=k,
                                      timestamp=timestamp, save_plot=save_plot,
                                      show_plot=show_plot, peak_freq=center_freq,
                                      channel=r['channel'])
            else:
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
        save_plot=True, show_plot=False, save_data=False, timestamp=None, 
        save_raw_data=False, correct_att=True, swap=False, hw_trigger=True):
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
            bay=self.band_to_bay(band)
            self.set_trigger_hw_arm(bay, 0, write_log=True)  # Default setup sets to 1

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
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

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
            if show_plot:
                plt.show()
            else:
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


    def find_peak(self, freq, resp, rolling_med=True, window=5000,
	grad_cut=.5, amp_cut=.25, freq_min=-2.5E8, freq_max=2.5E8, make_plot=False, 
	save_plot=True, show_plot=False, band=None,subband=None, make_subband_plot=False, 
	subband_plot_with_slow=False, timestamp=None, pad=50, min_gap=100, plot_title=None):
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
        x = np.arange(len(angle))
        p1 = np.poly1d(np.polyfit(x, angle, 1))
        angle -= p1(x)
        grad = np.convolve(angle, np.repeat([1,-1], 8),
                           mode='same')
        #grad = np.ediff1d(angle, to_end=[np.nan])
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
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

            fig, ax = plt.subplots(2, figsize=(8,6), sharex=True)

            if band is not None:
                bandCenterMHz = self.get_band_center_mhz(band)
                plot_freq = freq*1.0E-6 + bandCenterMHz
            else:
                plot_freq = freq

            ax[0].plot(plot_freq, amp)
            ax[0].plot(plot_freq, med_amp)
            ax[0].plot(plot_freq[peak], amp[peak], 'kx')
            ax[1].plot(plot_freq, grad)

            ax[1].set_ylim(-2, 20)
            for s, e in zip(starts, ends):
                ax[0].axvspan(plot_freq[s], plot_freq[e], color='k', alpha=.1)
                ax[1].axvspan(plot_freq[s], plot_freq[e], color='k', alpha=.1)
            

            ax[0].set_ylabel('Amp.')
            ax[1].set_xlabel('Freq. [MHz]')

            title = timestamp
            if band is not None:
                title = title + ': band {}, center = {:.1f} MHz'.format(band,bandCenterMHz)
            if subband is not None:
                title = title + ' subband {}'.format(subband)
            fig.suptitle(title)

            if save_plot:
                save_name = timestamp
                if band is not None:
                    save_name = save_name + '_b{}'.format(int(band))
                if subband is not None:
                    save_name = save_name + '_sb{}'.format(int(subband))
                save_name = save_name + '_find_freq.png'
                plt.savefig(os.path.join(self.plot_dir, save_name),
                            bbox_inches='tight', dpi=300)
            if show_plot:
                plt.show()
            else:
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
        eta_scaled = eta_mag / subband_half_width
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
        res_num=None, band=None, sk_fit=None, f_slow=None, resp_slow=None,
        channel=None):
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

        bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
        if peak_freq is not None:
            ax0.text(.03, .9, '{:5.2f} MHz'.format(peak_freq),
                      transform=ax0.transAxes, fontsize=10,
                      bbox=bbox)

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
        ax2.text(.03, .81, lab, transform=ax2.transAxes, fontsize=10,
                  bbox=bbox)

        if channel is not None:
            ax2.text(.85, .92, 'Ch {:03}'.format(channel),
                      transform=ax2.transAxes, fontsize=10,
                      bbox=bbox)

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

    def load_master_assignment(self, band, filename):
        """
        By default, pysmurf loads the most recent master assignment.
        Use this function to overwrite the default one.

        Args:
        -----
        band (int): The band for the master assignment file
        filename (str): The full path to the new master assignment
            file. Should be in self.tune_dir.
        """
        if 'band_{}'.format(band) in self.channel_assignment_files.keys():
            self.log('Old master assignment file:'+
                     ' {}'.format(self.channel_assignment_files['band_{}'.format(band)]))
        self.channel_assignment_files['band_{}'.format(band)] = filename
        self.log('New master assignment file:'+
                 ' {}'.format(self.channel_assignment_files['band_{}'.format(band)]))
        

    def get_master_assignment(self, band):
        """
        Returns the master assignment list.

        Ret:
        ----
        freqs (float array): The frequency of the resonators
        subbands (int array): The subbands the channels are assigned to
        channels (int array): The channels the resonators are assigned to
        groups (int array): The bias group the channel is in
        """
        fn = self.channel_assignment_files['band_%i' % (band)]
        self.log('Drawing channel assignments from {}'.format(fn))
        d = np.loadtxt(fn,delimiter=',')
        freqs = d[:,0]
        subbands = d[:,1].astype(int)
        channels = d[:,2].astype(int)
        groups = d[:,3].astype(int)
        return freqs, subbands, channels, groups


    def assign_channels(self, freq, band=None, bandcenter=None, 
        channel_per_subband=4, as_offset=True, min_offset=0.1,
        new_master_assignment=False):
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

        if band is None and bandcenter is None:
            self.log('Must have band or bandcenter', self.LOG_ERROR)
            raise ValueError('Must have band or bandcenter')

        subbands = np.zeros(len(freq), dtype=int)
        channels = -1 * np.ones(len(freq), dtype=int)
        offsets = np.zeros(len(freq))
        
        if not new_master_assignment:
            freq_master,subbands_master,channels_master,groups_master = \
                self.get_master_assignment(band)
            n_freqs = len(freq)
            n_unmatched = 0
            for idx in range(n_freqs):
                f = freq[idx]
                found_match = False
                for i in range(len(freq_master)):
                    f_master = freq_master[i]
                    if np.absolute(f-f_master) < min_offset:
                        ch = channels_master[i]
                        channels[idx] = ch
                        sb =  subbands_master[i]
                        subbands[idx] = sb
                        g = groups_master[i]
                        sb_center = self.get_subband_centers(band,as_offset=as_offset)[1][sb]
                        offsets[idx] = f-sb_center
                        self.log('Matching {:.2f} MHz to {:.2f} MHz in master channel list: assigning to subband {}, ch. {}, group {}'.format(f,f_master,\
                                                                     sb,ch,g))
                        found_match = True
                        break
                if not found_match:
                    n_unmatched += 1
                    self.log('No match found for {:.2f} MHz'.format(f))
            self.log('No channel assignment for {} of {} resonances.'.format(n_unmatched,n_freqs))
        else:
            d_freq = np.diff(freq)
            close_idx = d_freq > min_offset
            close_idx = np.logical_and(np.hstack((close_idx, True)), 
                                       np.hstack((True, close_idx)))
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

            # write the channel assignments to file
            self.write_master_assignment(band, freq, subbands, channels)
        
        return subbands, channels, offsets


    def write_master_assignment(self, band, freqs, subbands, channels, groups=None):
        '''
        writes a comma-separated list in the form band, freq (MHz), subband, channel, group.
        Group number defaults to -1.
        '''
        timestamp = self.get_timestamp()
        if groups is None:
            groups = -np.ones(len(freqs),dtype=int)

        #fn = self.channel_assignment_files['band_%i' % (band)]
        fn = os.path.join(self.tune_dir, 
                          '{}_channel_assignment_b{}.txt'.format(timestamp, int(band)))
        self.log('Writing new channel assignment to {}'.format(fn))
        f = open(fn,'w')
        for i in range(len(channels)):   
            f.write('%.4f,%i,%i,%i\n' % (freqs[i],subbands[i],channels[i],groups[i]))
        f.close()

        #self.channel_assignment_files['band_{}'.format(band)] = fn
        self.load_master_assignment(band, fn)

    def make_master_assignment_from_file(self, band, tuning_filename):
        self.log('Drawing band-{} tuning data from {}'.format(band,tuning_filename))
        d = np.load(tuning_filename).item()[band]['resonances']
        freqs = []
        subbands = []
        channels = []
        for i in range(len(d)):
            freqs.append(d[i]['freq'])
            subbands.append(d[i]['subband'])
            channels.append(d[i]['channel'])
        self.write_master_assignment(band, freqs, subbands, channels)

    
    def get_group_list(self,band,group):
        _,_,channels,groups = self.get_master_assignment(band)
        chs_in_group = []
        for i in range(len(channels)):
            if groups[i] == group:
                chs_in_group.append(channels[i])
        return chs_in_group

    def get_group_number(self,band,ch):
        _,_,channels,groups = self.get_master_assignment(band)
        for i in range(len(channels)):
            if channels[i] == ch:
                return groups[i]
        return None

    def write_group_assignment(self,band,group,ch_list):
        '''
        Combs master channel assignment and assigns group number to all channels in ch_list. Does not affect other channels in the master file.
        '''
        freqs_master,subbands_master,channels_master,groups_master = self.get_master_assignment(band)
        for i in range(len(ch_list)):
            for j in range(len(channels_master)):
                if ch_list[i] == channels_master[j]:
                    groups_master[j] = group
                    break
        self.write_master_assignment(band,freqs_master,subbands_master,channels_master,groups=groups_master)

    def compare_tune(self, tune, ref_tune=None, make_plot=False):
        """
        Compares tuning file to a reference tuning file. Does not work yet.

        """

        # Load data
        res1 = self.load_tune(tune)
        if ref_tune is None:
            res2 = self.freq_resp

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


    def relock(self, band, res_num=None, drive=None, r2_max=.08, 
        q_max=100000, q_min=0, check_vals=False, min_gap=None,
        write_log=False):
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
        check_vals (bool) : Whether to check r2 and Q values. Default is False.
        r2_max (float): The highest allowable R^2 value
        q_max (float): The maximum resonator Q factor
        q_min (float): The minimum resonator Q factor
        min_gap (float) : Thee minimum distance between resonators.
        """

        digitizer_freq = self.get_digitizer_frequency_mhz(band)
        n_subband = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        
        self.log('Relocking...')
        if res_num is None:
            res_num = np.arange(n_channels)
        else:
            res_num = np.array(res_num)

        if drive is None:
            drive = self.freq_resp[band]['drive']

        subband = digitizer_freq/(n_subband/2.)  # Oversample by 2

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
            f_gap=None
            if len(f)>1:
                f_gap = np.min(np.abs(np.append(f[:idx], f[idx+1:])-f[idx]))
            if write_log:
                self.log('Res {:03} - Channel {}'.format(k, ch))
            for ll, hh in self.bad_mask:
                if f[idx] > ll and f[idx] < hh:
                    self.log('{:4.3f} in bad list.'.format(f[idx]))
                    ch = -1
            if ch < 0: 
                if write_log:
                    self.log('No channel assigned: res {:03}'.format(k))
            elif min_gap is not None and f_gap is not None and f_gap < min_gap:
                if write_log:
                    self.log('Closest resonator is {:3.3f} MHz away'.format(f_gap))
            elif self.freq_resp[band]['resonances'][k]['r2'] > r2_max and check_vals:
                if write_log:
                    self.log('R2 too high: res {:03}'.format(k))
            #elif self.freq_resp[band]['resonances'][k]['Q'] < q_min and check_vals:
            #    if write_log:
            #        self.log('Q too low: res {:03}'.format(k))
            #elif self.freq_resp[band]['resonances'][k]['Q'] > q_max and check_vals:
            #    if write_log:
            #        self.log('Q too high: res {:03}'.format(k))
            elif k not in res_num:
                if write_log:
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

    def fast_relock(self, band):
        """
        """
        self.log('Fast relocking with: {}'.format(self.tune_file))
        self.set_tune_file_path(self.tune_file)
        self.set_load_tune_file(band, 1)
        self.log('Done fast relocking')

    def _get_eta_scan_result_from_key(self, band, key):
        """
        """
        if 'resonances' not in self.freq_resp[band].keys():
            self.log('No tuning. Run setup_notches() or load_tune()')
            return None

        return np.array([self.freq_resp[band]['resonances'][k][key]
                         for k in self.freq_resp[band]['resonances'].keys()])


    def get_eta_scan_result_freq(self, band):
        """
        Convenience function that gets the frequency results from
        eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        freq (float array) : The frequency in MHz of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'freq')


    def get_eta_scan_result_eta(self, band):
        """
        Convenience function that gets thee eta values from
        eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        eta (complex array) : The eta of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'eta')

    def get_eta_scan_result_eta_mag(self, band):
        """
        Convenience function that gets thee eta mags from
        eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        eta_mag (float array) : The eta of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'eta_mag')

    def get_eta_scan_result_eta_scaled(self, band):
        """
        Convenience function that gets the eta scaled from
        eta scans. eta_scaled is eta_mag/digitizer_freq_mhz/n_subbands

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        eta_mag (float array) : The eta_scaled of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'eta_scaled')


    def get_eta_scan_result_eta_phase(self, band):
        """
        Convenience function that gets the eta phase values from
        eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        eta_phase (float array) : The eta_phase of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'eta_phase')


    def get_eta_scan_result_channel(self, band):
        """
        Convenience function that gets the channel assignments from
        eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        channels (int array) : The channels of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'channel')


    def get_eta_scan_result_subband(self, band):
        """
        Convenience function that gets the subband from
        eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        subband (float array) : The subband of the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'subband')


    def get_eta_scan_result_offset(self, band):
        """
        Convenience function that gets the offset from center frequency 
        from eta scans.

        Args:
        -----
        band (int) : The band

        Ret:
        ----
        offset (float array) : The offset from the subband centers  of 
           the resonators.
        """
        return self._get_eta_scan_result_from_key(band, 'offset')

        
    def eta_reestimator(self, band, f0, drive, delta_freq=.01):
        """
        """
        subband, offset = self.freq_to_subband(f0, band)
        
        #left = f0 - delta_freq
        #right = f0 + delta_freq

        f_sweep = np.array([offset-delta_freq, offset+delta_freq])
        f, resp = self.fast_eta_scan(band, subband, f_sweep, 2, drive)


        eta = (f_sweep[1]-f_sweep[0])/(resp[1]-resp[0])

        sb, sbc = self.get_subband_centers(band, as_offset=False)

        return f_sweep+sbc[subband], resp, eta

    def eta_estimator(self, band, freq, drive=10, f_sweep_half=.3, 
                      df_sweep=.002, delta_freq=.01):
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
        channel_order = self.get_channel_order(band)
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
                        fraction_full_scale=None, flux_ramp=True,
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

        n_channels = self.get_number_channels(band)            
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
        first_idx = dd[0]//n_channels
        second_idx = dd[4]//n_channels
        dt = int(second_idx-first_idx)  # In slow samples
        n_fr = int(len(sync[:,0])/n_channels/dt)
        reset_idx = np.arange(first_idx, n_fr*dt + first_idx+1, dt)

        # Reset to the previous FB state
        self.set_feedback_enable_array(band, old_fb)

        fs = self.get_digitizer_frequency_mhz(band) * 1.0E6 /2/n_channels

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

        return d, df, sync

    def tracking_setup(self, band, channel=None, reset_rate_khz=4., write_log=False, 
        make_plot=False, save_plot=True, show_plot=True, nsamp=2**19,
        lms_freq_hz=None, meas_lms_freq=False, flux_ramp=True, fraction_full_scale=None,
        lms_enable1=True, lms_enable2=True, lms_enable3=True, lms_gain=7,
        return_data=True, new_epics_root=None,
        feedback_start_frac=None, feedback_end_frac=None):
        """
        Args:
        -----
        band (int) : The band number
        channel (int) : The channel to check

        Opt Args:
        ---------
        reset_rate_khz (float) : The flux ramp frequency
        write_log (bool) : Whether to write output to the log.  Default False.
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
        feedback_start_frac (float) : The fraction of the full flux
           ramp at which to stop applying feedback in each flux ramp
           cycle.  Must be in [0,1).  Defaults to whatever's in the cfg
           file.
        feedback_end_frac (float) : The fraction of the full flux ramp
           at which to stop applying feedback in each flux ramp cycle.
           Must be >0.  Defaults to whatever's in the cfg file.
        meas_lms_freq (bool) : Whether or not to try to estimate the
           carrier rate using the flux_mod2 function.  Default false.
           lms_freq_hz must be None.
        """

        ##
        ## Load unprovided optional args from cfg
        if feedback_start_frac is None:
            feedback_start_frac = self.config.get('tune_band').get('feedback_start_frac')[str(band)]
        if feedback_end_frac is None:
            feedback_end_frac = self.config.get('tune_band').get('feedback_end_frac')[str(band)]
        ## End loading unprovided optional args from cfg
        ##

        ##
        ## Argument validation

        # Validate feedback_start_frac and feedback_end_frac
        if (feedback_start_frac < 0) or (feedback_start_frac >= 1):
            raise ValueError("feedback_start_frac = {} not in [0,1)".format(feedback_start_frac))        
        if (feedback_end_frac < 0):
            raise ValueError("feedback_end_frac = {} not > 0".format(feedback_end_frac))
        # If feedback_start_frac exceeds feedback_end_frac, then
        # there's no range of the flux ramp cycle over which we're
        # applying feedback.
        if (feedback_end_frac < feedback_start_frac):
            raise ValueError("feedback_end_frac = {} is not less than feedback_start_frac = {}".format(feedback_end_frac, feedback_start_frac))
        # Done validating feedbackStart and feedbackEnd
        
        ## End argument validation
        ##
        
        if not flux_ramp:
            self.log('WARNING: THIS WILL NOT TURN ON FLUX RAMP!')
            
        if make_plot:
            import matplotlib.pyplot as plt
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

        if fraction_full_scale is None:
            fraction_full_scale = self.fraction_full_scale
        else:
            self.fraction_full_scale = fraction_full_scale
        
        # Switched to a more stable estimator
        if lms_freq_hz is None:
            if meas_lms_freq:
                lms_freq_hz = self.estimate_lms_freq(band,fraction_full_scale=fraction_full_scale,channel=channel,make_plot=False)
            else:
                lms_freq_hz = self.config.get('tune_band').get('lms_freq')[str(band)]
            self.lms_freq_hz[band] = lms_freq_hz
            self.log('Using lms_freq_estimator : {:.0f} Hz'.format(lms_freq_hz))

        if not flux_ramp:
            lms_enable1 = 0
            lms_enable2 = 0
            lms_enable3 = 0

        self.log("Using lmsFreqHz = {:.0f} Hz".format(lms_freq_hz), self.LOG_USER)

        self.set_lms_gain(band, lms_gain, write_log=write_log)
        self.set_lms_enable1(band, lms_enable1, write_log=write_log)
        self.set_lms_enable2(band, lms_enable2, write_log=write_log)
        self.set_lms_enable3(band, lms_enable3, write_log=write_log)
        self.set_lms_freq_hz(band, lms_freq_hz, write_log=write_log)

        iq_stream_enable = 0  # must be zero to access f,df stream        
        self.set_iq_stream_enable(band, iq_stream_enable, write_log=write_log)

        self.flux_ramp_setup(reset_rate_khz, fraction_full_scale,
                             write_log=write_log, new_epics_root=new_epics_root)

        # Doing this after flux_ramp_setup so that if needed we can
        # set feedback_end based on the flux ramp settings.

        # Compute feedback_start/feedback_end from
        # feedback_start_frac/feedback_end_frac.
        channel_frequency_mhz = self.get_channel_frequency_mhz(band)
        digitizer_frequency_mhz = self.get_digitizer_frequency_mhz(band)
        feedback_start = int(
            feedback_start_frac*(self.get_ramp_max_cnt()+1)/(
                digitizer_frequency_mhz/channel_frequency_mhz/2. ) )
        feedback_end = int(
            feedback_end_frac*(self.get_ramp_max_cnt()+1)/(
                digitizer_frequency_mhz/channel_frequency_mhz/2. ) )

        # Set feedbackStart and feedbackEnd
        self.set_feedback_start(band, feedback_start, write_log=write_log)
        self.set_feedback_end(band, feedback_end, write_log=write_log)

        self.log("Applying feedback over {:.1f}% of each flux ramp cycle (with feedbackStart={} and feedbackEnd={})".format(
                                         (feedback_end_frac-feedback_start_frac)*100.,
                                         feedback_start,
                                         feedback_end),
                 self.LOG_USER)
        
        if flux_ramp:
            self.flux_ramp_on(write_log=write_log, new_epics_root=new_epics_root)

        # take one dataset with all channels
        if return_data or make_plot:
            f, df, sync = self.take_debug_data(band, IQstream = iq_stream_enable, 
                                           single_channel_readout=0, nsamp=nsamp)
            
            df_std = np.std(df, 0)

            #downselect_channels = self.get_downselect_channels()
            #print(np.where(df_std>0))
            df_channels = np.ravel(np.where(df_std >0))

            channels_on = list(set(df_channels) & set(self.which_on(band)))
            self.log("Number of channels on = {}".format(len(channels_on)), 
                self.LOG_USER)

            f_span = np.max(f,0) - np.min(f,0)

        if make_plot:
            timestamp = self.get_timestamp()

            fig,ax = plt.subplots(1,3,figsize = (12,5))
            fig.suptitle('LMS freq = {:.0f} Hz, n_channels = {}'.format(lms_freq_hz,len(channels_on)))
            
            ax[0].hist(df_std[channels_on] * 1e3,bins = 20,edgecolor = 'k')            
            ax[0].set_xlabel('Flux ramp demod error std (kHz)')
            ax[0].set_ylabel('number of channels')

            ax[1].hist(f_span[channels_on] * 1e3,bins = 20,edgecolor='k')
            ax[1].set_xlabel('Flux ramp amplitude (kHz)')
            ax[1].set_ylabel('number of channels')

            ax[2].plot(f_span[channels_on]*1e3, df_std[channels_on]*1e3, '.')
            ax[2].set_xlabel('FR Amp (kHz)')
            ax[2].set_ylabel('RF demod error (kHz)')
            x = np.array([0, np.max(f_span[channels_on])*1.0E3])
            y_factor = 10
            y = x/y_factor
            ax[2].plot(x,y, color='k', linestyle=':',label='1:%i' % (y_factor))
            ax[2].legend(loc='best')
            
            if save_plot:
                plt.savefig(os.path.join(self.plot_dir, timestamp + 
                            '_FR_amp_v_err.png'),bbox_inches='tight')
            if not show_plot:
                plt.close()

            if channel is not None:
                channel = np.ravel(np.array(channel))
                self.log("Taking data on single channel number {}".format(channel), 
                         self.LOG_USER)
                sync_idx = self.make_sync_flag(sync)

                bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)

                for ch in channel:
                    fig, ax = plt.subplots(2, sharex=True)
                    ax[0].plot(f[:, ch]*1e3)
                    ax[0].set_ylabel('Tracked Freq [kHz]')
                    ax[0].text(.025, .9, 'LMS Freq {:.0f} Hz'.format(lms_freq_hz), fontsize=10,
                        transform=ax[0].transAxes, bbox=bbox)

                    ax[0].text(.95, .9, 'Band {} Ch {:03}'.format(band, ch), fontsize=10,
                        transform=ax[0].transAxes, horizontalalignment='right', bbox=bbox)

                    ax[1].plot(df[:, ch]*1e3)
                    ax[1].set_ylabel('Freq Error [kHz]')
                    ax[1].set_xlabel('Samp Num')
                    ax[1].text(.025, .8, 'RMS error = {:.2f} kHz\n'.format(df_std[ch]*1e3) +
                        'FR frac. full scale = {:.2f}'.format(fraction_full_scale),
                        fontsize=10, transform=ax[1].transAxes, bbox=bbox)

                    for s in sync_idx:
                        ax[0].axvline(s, color='k', linestyle=':', alpha=.5)
                        ax[1].axvline(s, color='k', linestyle=':', alpha=.5)

                    plt.tight_layout()

                    if save_plot:
                        plt.savefig(os.path.join(self.plot_dir, timestamp + 
                                                 '_FRtracking_band{}_ch{:03}.png'.format(band,ch)),
                                    bbox_inches='tight')
                    if not show_plot:
                        plt.close()

        self.set_iq_stream_enable(band, 1, write_log=write_log)

        if return_data:
            return f, df, sync

    def track_and_check(self, band, channel=None, reset_rate_khz=4., 
        make_plot=False, save_plot=True, show_plot=True,
        lms_freq_hz=None, flux_ramp=True, fraction_full_scale=None,
        lms_enable1=True, lms_enable2=True, lms_enable3=True, lms_gain=7,
        f_min=.015, f_max=.2, df_max=.03, toggle_feedback=True,
        relock=True,tracking_setup=True):
        """
        This runs tracking setup and check_lock to prune bad channels.
        
        Args:
        -----
        band (int): The band to track and check
        
        Opt Args:
        ---------
        reset_rate_khz (float); The flux ramp reset rate.
        channel (int or int array): List of channels to plot.
        toggle_feedback (bool): Whether or not to reset feedback (both
                                the global band feedbackEnable and the 
                                lmsEnables between tracking_setup and 
                                check_lock.
        relock (bool): Whether or not to relock at the start.  
                       Default True.
        tracking_setup (bool): Whether or not to run tracking_setup.  
                               Default True.
        """
        if relock:
            self.relock(band)

        if tracking_setup:
            self.tracking_setup(band, channel=channel, 
                                reset_rate_khz=reset_rate_khz,
                                make_plot=make_plot, save_plot=save_plot, 
                                show_plot=show_plot,
                                lms_freq_hz=lms_freq_hz, flux_ramp=flux_ramp, 
                                fraction_full_scale=fraction_full_scale,
                                lms_enable1=lms_enable1, lms_enable2=lms_enable2, 
                                lms_enable3=lms_enable3, 
                                lms_gain=lms_gain, return_data=False)

        if toggle_feedback:
            self.toggle_feedback(band)

        self.check_lock(band, f_min=f_min, f_max=f_max, df_max=df_max,
                        make_plot=make_plot, flux_ramp=flux_ramp, 
                        fraction_full_scale=fraction_full_scale,
                        lms_freq_hz=lms_freq_hz, reset_rate_khz=4.)
    
    def eta_phase_check(self, band, rot_step_size=30, rot_max=360,
                        reset_rate_khz=4., 
                        fraction_full_scale=None,
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
            for c in np.arange(n_channels):
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
        
        n_channels = self.get_number_channels(band)
        fs = self.get_digitizer_frequency_mhz(band) * 1.0E6 /2/n_channels
        scale = 1.0E3

        fig, ax = plt.subplots(1)
        cm = plt.get_cmap('viridis')
        for j, k in enumerate(keys):
            sync = dat['data'][k]['sync']
            df = dat['data'][k]['df']
            dd = np.ravel(np.where(np.diff(sync[:,0]) !=0))
            first_idx = dd[0]//n_channels
            second_idx = dd[4]//n_channels
            dt = int(second_idx-first_idx)  # In slow samples                                             
            n_fr = int(len(sync[:,0])/n_channels/dt)
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


    def set_fixed_flux_ramp_bias(self,fractionFullScale,debug=True, do_config=True):
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

        if do_config:
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
        if debug:
            self.log("Setting flux ramp to {}".format(100 * fractionFullScale, 
                     int(LTC1668RawDacData)) + "% of full scale (LTC1668RawDacData={})", 
                     self.LOG_USER)
        self.set_flux_ramp_dac(LTC1668RawDacData)        

    def flux_ramp_setup(self, reset_rate_khz, fraction_full_scale, df_range=.1, 
                        do_read=False, band=2, write_log=False, new_epics_root=None):
        """
        Set flux ramp sawtooth rate and amplitude. If there are errors, check 
        that you are using an allowed reset rate! Not all rates are allowed.

        Allowed rates: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15 kHz
        """

        # Disable flux ramp
        self.flux_ramp_off(new_epics_root=new_epics_root,
                           write_log=write_log)

        digitizerFrequencyMHz = self.get_digitizer_frequency_mhz(band,
            new_epics_root=new_epics_root)
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

        self.log("Percent full scale = {:0.3f}%".format(100 * fractionFullScale), 
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
        SelectRamp = self.get_select_ramp(new_epics_root=new_epics_root) # from config file
        RampStartMode = self.get_ramp_start_mode(new_epics_root=new_epics_root) # from config file
        PulseWidth = 64
        DebounceWidth = 255
        RampSlope = 0
        ModeControl = 0
        EnableRampTrigger = 1

        self.set_low_cycle(LowCycle, new_epics_root=new_epics_root,
                           write_log=write_log) 
        self.set_high_cycle(HighCycle, new_epics_root=new_epics_root,
                           write_log=write_log)
        self.set_k_relay(KRelay, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_ramp_max_cnt(rampMaxCnt, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_select_ramp(SelectRamp, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_ramp_start_mode(RampStartMode, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_pulse_width(PulseWidth, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_debounce_width(DebounceWidth, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_ramp_slope(RampSlope, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_mode_control(ModeControl, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_fast_slow_step_size(FastSlowStepSize, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_fast_slow_rst_value(FastSlowRstValue, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_enable_ramp_trigger(EnableRampTrigger, new_epics_root=new_epics_root,
                            write_log=write_log)
        self.set_ramp_rate(reset_rate_khz, new_epics_root=new_epics_root,
                            write_log=write_log)



    def get_fraction_full_scale(self, new_epics_root=None):
        """
        Returns the fraction_full_scale
        """
        return 1-2*(self.get_fast_slow_rst_value(new_epics_root=new_epics_root)/
                    2**self.num_flux_ramp_counter_bits)

    def check_lock(self, band, f_min=.015, f_max=.2, df_max=.03,
        make_plot=False, flux_ramp=True, fraction_full_scale=None,
        lms_freq_hz=None, reset_rate_khz=4., **kwargs):
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

        if fraction_full_scale is None:
            fraction_full_scale = self.fraction_full_scale

        if lms_freq_hz is None:
            lms_freq_hz = self.lms_freq_hz[band]

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

            # self.log('Ch {} f_span {} df_rms {}'.format(ch, f_span, df_rms))

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


    def find_freq(self, band, subband=np.arange(13,115), drive_power=None,
        n_read=2, make_plot=False, save_plot=True, window=50, rolling_med=True,
                  make_subband_plot=False):
        '''
        Finds the resonances in a band (and specified subbands)

        Args:
        -----
        band (int) : The band to search

        Optional Args:
        --------------
        subband (int) : An int array for the subbands
        drive_power (int) : The drive amplitude.  If none given, takes from cfg.
        n_read (int) : The number sweeps to do per subband
        make_plot (bool) : make the plot frequency sweep. Default False.
        save_plot (bool) : save the plot. Default True.
        save_name (string) : What to name the plot. default find_freq.png
        rolling_med (bool) : Whether to iterate on a rolling median or just
           the median of the whole sample.
        window (int) : The width of the rolling median window
        '''

        if drive_power is None:
            drive_power = self.config.get('init')['band_{}'.format(band)].get('amplitude_scale')
            self.log('No drive_power given. Using value in config file: {}'.format(drive_power))

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
            band=band, rolling_med=rolling_med, window=window, make_subband_plot=make_subband_plot)
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


    def full_band_ampl_sweep(self, band, subband, drive, n_read, n_step=121):
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

        # scan_freq = np.arange(-3, 3.05, 0.05)  # take out this hardcode
        step_size = 2/n_step
        scan_freq = np.ceil(digitizer_freq/n_subbands/2)*np.arange(-1, 1, step_size)
        print(scan_freq)
        
        resp = np.zeros((n_subbands, np.shape(scan_freq)[0]), dtype=complex)
        freq = np.zeros((n_subbands, np.shape(scan_freq)[0]))

        subband_nos, subband_centers = self.get_subband_centers(band)

        self.log('Working on band {:d}'.format(band))
        for sb in subband:
            self.log('Sweeping subband no: {}'.format(sb))
            f, r = self.fast_eta_scan(band, sb, scan_freq, n_read, 
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

        channel_order = self.get_channel_order(band)

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

    def setup_notches(self, band, resonance=None, drive=None,
                      sweep_width=.3, df_sweep=.002,
                      subband_half_width=614.4/128, min_offset=0.1,
                      delta_freq=0.01, new_master_assignment=False):
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
        min_offset (float): Minimum distance in MHz between two resonators for assigning channels.
        delta_freq (float): The frequency offset at which to measure
            the complex transmission to compute the eta parameters.
            Passed to eta_estimator.  Units are MHz.  Default is 0.01
            (10kHz).

        Returns:
        --------

        """

        # Turn off all tones in this band first
        self.band_off(band)
        
        # Check if any resonances are stored
        if 'resonance' not in self.freq_resp[band]['find_freq'] and resonance is None:
            self.log('No resonances stored in band {}'.format(band) +
                '. Run find_freq first.', self.LOG_ERROR)
            return

        if drive is None:
            drive = self.config.get('init')['band_{}'.format(band)].get('amplitude_scale')
            self.log('No drive given. Using value in config file: {}'.format(drive))

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

        n_res = len(input_res)
        for i, f in enumerate(input_res):
            self.log('freq {:5.4f} - {} of {}'.format(f, i+1, n_res))
            freq, resp, eta = self.eta_estimator(band, f, drive,
                                                 f_sweep_half=sweep_width,
                                                 df_sweep=df_sweep,delta_freq=delta_freq)
            eta_phase_deg = np.angle(eta)*180/np.pi
            eta_mag = np.abs(eta)
            eta_scaled = eta_mag / subband_half_width
            
            abs_resp = np.abs(resp)
            idx = np.ravel(np.where(abs_resp == np.min(abs_resp)))[0]

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


        # Assign resonances to channels                                                       
        self.log('Assigning channels')
        f = [resonances[k]['freq'] for k in resonances.keys()]
        subbands, channels, offsets = self.assign_channels(f, band=band, 
                                        as_offset=False,min_offset=min_offset,
                                        new_master_assignment=new_master_assignment)

        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})

        self.freq_resp[band]['resonances'] = resonances

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
        self.tune_file = savedir+'.npy'

        return savedir + ".npy"

    def load_tune(self, filename=None, override=True, last_tune=True, band=None):
        """
        Loads the tuning information (self.freq_resp) from tuning directory


        Opt Args:
        ---------
        filename (str) : The name of the tuning.
        last_tune (bool): Whether to use the most recent tuning
            file. Default is True.
        override (bool) : Whether to replace self.freq_resp. Default
            is True.
        band (int, int array) : if None, loads entire tune.  If band
            number is provided, only loads the tune for that band.
            Not used at all unless override=True.
        """
        if filename is None and last_tune:
            filename = self.last_tune()
            self.log('Defaulting to last tuning: {}'.format(filename))
        elif filename is not None and last_tune:
            self.log('filename explicitly given. Overriding last_tune bool in load_tune.')

        fs = np.load(filename).item()
        self.log('Done loading tuning')

        if override:
            if band is None:
                bands_in_file=list(fs.keys())
                self.log('Loading tune data for all bands={}.'.format(str(bands_in_file)))
                self.freq_resp = fs
                # Load all tune data for all bands in file.  only
                # update tune_file if both band are loaded from the
                # same file right now.  May want to handle this
                # differently to allow loading different tunes for
                # different bands, etc.
                self.tune_file = filename
            else:
                # Load only the tune data for the requested band(s).
                band=np.ravel(np.array(band))
                self.log('Only loading tune data for bands={}.'.format(str(band)))
                for b in band:
                    self.freq_resp[b] = fs[b]                    
        else:
            # Right now, returns tune data for all bands in file;
            # doesn't know about the band arg.
            return fs

    def last_tune(self):
        """
        Returns the full path to the most recent tuning file.
        """
        return np.sort(glob.glob(os.path.join(self.tune_dir, 
                                              '*_tune.npy')))[-1]


    def parallel_scan(self, band, channels, drive,
                      scan_freq=np.arange(-3, 3, .1)):
        """
        Does all the eta scans at once. The center frequency
        array must already be populated.

        Args:
        -----
        band (int) : The band to eta scan
        channels (int array): The list of channels to
           eta scan.
        drive (int) : The drive amplitude
        
        Opt Args:
        ---------
        scan_freq (float array) : The frequencies to 
           scan. 
        """
        self.flux_ramp_off()

        n_channels = self.get_number_channels(band)        
        ch_idx = np.zeros(n_channels, dtype=int)
        for c in channels:
            ch_idx[c] = 1
        
        self.set_eta_mag_array(band, np.ones(n_channels, dtype=int))
        self.set_feedback_enable_array(band, np.zeros(n_channels, dtype=int))
        self.set_amplitude_scale_array(band, ch_idx*drive)

        freq_error = np.zeros((len(scan_freq), n_channels), dtype='complex')
        real_imag = np.array([1, 1.j])
        eta_phase = np.array([0., 90.])

        self.log('Starting parallel scan')
        
        for j in np.arange(2):
            self.set_eta_phase_array(band, eta_phase[j] * np.ones(n_channels, dtype=int))
            for i in np.arange(len(scan_freq)):
                self.log('scan {}'.format(i))
                self.set_center_frequency_array(band, scan_freq[i]*np.ones(n_channels, dtype=int))
                freq_error[i] = freq_error[i] + real_imag[j] * self.get_frequency_error_array(band)

        return scan_freq, freq_error


    def estimate_lms_freq(self, band, fraction_full_scale=None,
                          reset_rate_khz=4., new_epics_root=None,
                          channel=None, make_plot=False):
        """
        
        Ret:
        ----
        The estimated lms frequency in Hz
        """
        if fraction_full_scale is None:
            fraction_full_scale = \
                self.config.get('tune_band').get('fraction_full_scale')

        old_feedback = self.get_feedback_enable(band)

        self.set_feedback_enable(band, 0)
        f, df, sync = self.tracking_setup(band, 0, make_plot=False,
            flux_ramp=True, fraction_full_scale=fraction_full_scale,
            reset_rate_khz=reset_rate_khz, lms_freq_hz=0,
            new_epics_root=new_epics_root)

        s = self.flux_mod2(band, df, sync, make_plot=make_plot, channel=channel)

        self.set_feedback_enable(band, old_feedback)
        return reset_rate_khz * s * 1000  # convert to Hz

    def flux_mod2(self, band, df, sync, min_scale=.002, make_plot=False, 
                  channel=None, threshold=.5):
        """
        Attempts to find the number of phi0s in a tracking_setup.
        Takes df and sync from a tracking_setup with feedback off.

        Args:
        -----
        band (int) : which band
        df (float array): The df term from tracking setup with
            feedback off.
        sync (float array): The sync term from tracking setup.

        Opt Args:
        ---------
        min_scale (float): The minimum df amplitude used in analysis.
            This is used to cut channels that are not responding
            to flux ramp. Default is .002
        threshold (float): The minimum convolution amplitude to
            consider a peak. Default is .01
        make_plot (bool): Whether to make a plot. If True, you must
            also supply the channels to plot using the channel opt
            arg.
        channel (int or int array): The channels to plot. Default 
            is None.

        Ret:
        ----
        n (float): The number of phi0 swept out per sync. To get
           lms_freq_hz, multiply by the flux ramp frequency.
        """
        sync_flag = self.make_sync_flag(sync)

        # The longest time between resets
        max_len = np.max(np.diff(sync_flag)) 
        n_sync = len(sync_flag) - 1
        n_samp, n_chan = np.shape(df)

        # Only for plotting
        channel = np.ravel(np.array(channel))

        if make_plot:
            import matplotlib.pyplot as plt

        peaks = np.zeros(n_chan)*np.nan

        band_chans=list(self.which_on(band))
        for ch in np.arange(n_chan):
            if ch in band_chans and np.std(df[:,ch]) > min_scale:
                # Holds the data for all flux ramps
                flux_resp = np.zeros((n_sync, max_len)) * np.nan
                for i in np.arange(n_sync):
                    flux_resp[i] = df[sync_flag[i]:sync_flag[i+1],ch]

                # Average over all the flux ramp sweeps to generate template
                template = np.nanmean(flux_resp, axis=0)
                template_mean = np.mean(template)
                template -= template_mean

                # Normalize template so thresholding can be
                # independent of the units of this crap
                if np.std(template)!=0:
                    template = template/np.std(template)

                # Multiply the matrix with the first element, then array
                # of first two elements, then array of first three...etc...
                # The array that maximizes this tells you the frequency
                corr_amp = np.zeros(max_len//2)
                for i in np.arange(1, max_len//2):
                    x = np.tile(template[:i], max_len//i+1)
                    # Normalize by elements in sum so that the
                    # correlation comes out something like unity at
                    # max
                    corr_amp[i] = np.sum(x[:max_len]*template)/max_len

                s, e = self.find_flag_blocks(corr_amp > threshold, min_gap=4)
                if len(s) == 0:
                    peaks[ch] = np.nan
                elif s[0] == e[0]:
                    peaks[ch] = s[0]
                else:
                    #peaks[ch] = np.ravel(np.where(corr_amp[s[0]:e[0]] == 
                    #                          np.max(corr_amp[s[0]:e[0]])))[0] + s[0]
                    peaks[ch] = np.argmax(corr_amp[s[0]:e[0]]) + s[0]

                    #polyfit
                    Xf = [-1, 0, 1]
                    Yf = [corr_amp[int(peaks[ch]-1)],corr_amp[int(peaks[ch])],corr_amp[int(peaks[ch]+1)]]
                    V = np.polyfit(Xf, Yf, 2)
                    offset = -V[1]/(2.0 * V[0])
                    peak = offset + peaks[ch]

                    #kill shawn
                    peaks[ch]=peak
                    
                if make_plot and ch in channel:
                    fig, ax = plt.subplots(2)
                    for i in np.arange(n_sync):
                        ax[0].plot(df[sync_flag[i]:sync_flag[i+1],ch]-
                                   template_mean)
                    ax[0].plot(template, color='k')
                    ax[1].plot(corr_amp)
                    ax[1].plot(peaks[ch], corr_amp[int(peaks[ch])], 'x' ,color='k')

        return max_len/np.nanmedian(peaks)


    def make_sync_flag(self, sync):
        """
        Takes the sync from tracking setup and makes a flag for when the sync
        is True.

        Args:
        -----
        sync (float array): The sync term from tracking_setup

        Ret:
        ----
        start (int array): The start index of the sync
        end (int array) The end index of the sync
        """
        s, e = self.find_flag_blocks(sync[:,0], min_gap=1000)
        n_proc=self.get_number_processed_channels()
        return s//n_proc


    def flux_mod(self, df, sync, threshold=.4, minscale=.02,
                 min_spectrum=.9, make_plot=False):
        """
        Joe made this
        """
        mkr_ratio = 512  # ratio of rates on marker to rates in data
        num_channels = len(df[0,:])

        result = np.zeros(num_channels)
        peak_to_peak = np.zeros(num_channels)
        lp_power = np.zeros(num_channels)

        # Flux ramp markers
        n_sync = len(sync[:,0])
        mkrgap = 0
        mkr1 = 0
        mkr2 = 0
        totmkr = 0
        lastmkr = 0
        for n in np.arange(n_sync):
            mkrgap = mkrgap + 1
            if (sync[n,0] > 0) and (mkrgap > 1000):

                mkrgap = 0
                totmkr = totmkr + 1
                last_mkr = n

                if mkr1 == 0:
                    mkr1 = n
                elif mkr2 == 0:
                    mkr2 = n

        dn = int(np.round((mkr2 - mkr1)/mkr_ratio))
        sn = int(np.round(mkr1/mkr_ratio))

        for ch in np.arange(num_channels):
            peak_to_peak[ch] = np.max(df[:,ch]) - np.min(df[:,ch])
            if peak_to_peak[ch] > 0:
                lp_power[ch] = np.std(df[:,ch])/np.std(np.diff(df[:,ch]))
            if ((peak_to_peak[ch] > minscale) and (lp_power[ch] > min_spectrum)):
                flux = np.zeros(dn)
                for mkr in np.arange(totmkr-1):
                    for n in np.arange(dn):
                        flux[n] = flux[n] + df[sn + mkr * dn + n, ch]
                flux = flux - np.mean(flux)

                sxarray = np.array([0])
                pts = len(flux)

                for rlen in np.arange(1, np.round(pts/2), dtype=int):
                    refsig = flux[:rlen]
                    sx = 0
                    for pt in np.arange(pts):
                        pr = pt % rlen
                        sx = sx + refsig[pr] * flux[pt]
                    sxarray = np.append(sxarray, sx)
                
                ac = 0
                for n in np.arange(pts):
                    ac = ac + flux[n]**2

                scaled_array = sxarray / ac

                pk = 0
                for n in np.arange(np.round(pts/2)-1, dtype=int):
                    if scaled_array[n] > threshold:
                        if scaled_array[n] > scaled_array[pk]:
                            pk = n
                        else:
                            break;

                Xf = [-1, 0, 1]
                Yf = [scaled_array[pk-1], scaled_array[pk], scaled_array[pk+1]]
                V = np.polyfit(Xf, Yf, 2)
                offset = -V[1]/(2 * V[0]);
                peak = offset + pk

                result[ch] = dn /  peak

                if make_plot:   # plotting routine to show sin fit
                    import matplotlib.pyplot as plt
                    rs = 0
                    rc = 0
                    r = pts * [0]
                    s = pts * [0]
                    c = pts * [0]
                    scl = np.max(flux) - np.min(flux)
                    for n in range(0, pts):
                        s[n] = np.sin(n * 2 * np.pi / (dn/result[ch]));
                        c[n] = np.cos(n * 2 * np.pi / (dn/result[ch]));
                        rs = rs + s[n] * flux[n]
                        rc = rc + c[n] * flux[n]
                    
                    theta = np.arctan2(rc, rs)
                    for n in range(0, pts):
                        r[n] = 0.5 * scl *  np.sin(theta + n * 2 * np.pi / (dn/result[ch]));

                    plt.figure()
                    plt.plot(r)
                    plt.plot(flux)
                    plt.show()

        fmod_array = []
        for n in range(0, num_channels):
            if(result[n] > 0):
                fmod_array.append(result[n])
        mod_median = np.median(fmod_array)
        return mod_median


    def find_bad_pairs(self, band, reset_rate_khz=4., write_log=False,
        make_plot=False, save_plot=True, show_plot=True,
        lms_freq_hz=None, flux_ramp=True, fraction_full_scale=.4950,
        lms_enable1=True, lms_enable2=True, lms_enable3=True, lms_gain=7):
        """
        """
        resonators = self.freq_resp[band]['resonances']
        keys = resonators.keys()

        freqs = np.array([resonators[k]['freq'] for k in keys])
        channels = np.array([resonators[k]['channel'] for k in keys])

        idx = np.argsort(freqs)
        freqs = freqs[idx]
        channels = channels[idx]
        res_nums = np.arange(len(keys))

        df_err = np.zeros((len(keys), 2))
        f_span = np.zeros((len(keys), 2))

        for i in np.arange(len(keys)-1):
            f1 = freqs[i]
            ch1 = channels[i]
            rn1 = res_nums[i]
            f2 = freqs[i+1]
            ch2 = channels[i+1]
            rn2 = res_nums[i+1]

            self.log('Freq {} {}'.format(f1, f2))

            self.band_off(band)
            self.relock(band, res_num=np.array([rn1, rn2]))

            d, df, sync = self.tracking_setup(band, 0, reset_rate_khz=reset_rate_khz,
                                              lms_freq_hz=lms_freq_hz, flux_ramp=flux_ramp,
                                              lms_enable1=lms_enable1, lms_enable2=lms_enable2,
                                              lms_enable3=lms_enable3, lms_gain=lms_gain,
                                              fraction_full_scale=fraction_full_scale,
                                              make_plot=False)
            df_err[i,0] = np.std(df[:,ch1])
            df_err[i,1] = np.std(df[:,ch2])
            f_span[i,0] = np.max(d[:,ch1]) - np.min(d[:,ch1])
            f_span[i,1] = np.max(d[:,ch2]) - np.min(d[:,ch2])

        return f_span, df_err

    def dump_state(self, output_file=None, return_screen=False):
        """
        Dump the current tuning info to config file and write to disk

        Args:
        -----
        output_file (str): path to output file location. Defaults to the config file status dir and timestamp
        return_screen (bool): whether to also return the contents of the config file in addition to writing to file. Defaults False. 
        """

        # get the HEMT info because why not
        self.add_output('hemt_status', self.get_amplifier_biases())

        # get jesd status for good measure
        self.add_output('jesd_status', self.check_jesd())

        # get TES bias info
        self.add_output('tes_biases', list(self.get_tes_bias_bipolar_array()))

        # get flux ramp info
        self.add_output('flux_ramp_rate', self.get_flux_ramp_freq())
        self.add_output('flux_ramp_amplitude', self.get_fraction_full_scale()) # this doesn't exist yet

        # initialize band outputs
        self.add_output('band_outputs', {})

        # there is probably a better way to do this
        for band in self.config.get('init')['bands']:
            band_outputs = {} # Python copying is weird so this is easier
            band_outputs[band] = {} 
            band_outputs[band]['amplitudes'] = list(self.get_amplitude_scale_array(band).astype(float))
            band_outputs[band]['freqs'] = list(self.get_center_frequency_array(band))
            band_outputs[band]['eta_mag'] = list(self.get_eta_mag_array(band))
            band_outputs[band]['eta_phase'] = list(self.get_eta_phase_array(band))
        self.config.update_subkey('outputs', 'band_outputs', band_outputs)

        # dump to file
        if output_file is None:
            filename = self.get_timestamp() + '_status.cfg'
            status_dir = self.status_dir
            output_file = os.path.join(status_dir, filename)

        self.log('Dumping status to file:{}'.format(output_file))
        self.write_output(output_file)

        if return_screen:
            return self.config.config
