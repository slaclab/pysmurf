# frozen functions list:

# SETUP FUNCTIONS

# setup

# set_amplifier_bias
def set_amplifier_bias(self, bias_hemt = None, bias_50k = None, **kwargs):
        self.log('set_amplifier_bias: Deprecated. Calling set_amp_gate_voltage')
        if bias_hemt is not None:
            self.set_amp_gate_voltage('hemt', bias_hemt, **kwargs)

        if bias_50k is not None:
            self.set_amp_gate_voltage('50k', bias_50k, **kwargs)

# set_cryo_card_ps_en
def set_cryo_card_ps_en(self, enable=3, write_log=False):
        """
        Write the cryo card power supply enables. Can use this to set both
        power supplies at once rather than setting them individually

        Args
        ----
        enables : int, optional, default 3
            2-bit number with status of the power supplies enables
            Bit 0 for HEMT supply
            Bit 1 for 50K supply
            Bit == 1 means enabled
            Bit == 0 means disabled

            therefore:
            0 = all off
            1 = 50K on, HEMT off
            2 = HEMT on, 50K off
            3 = both on

            Default (enable=3) turns on both power supplies.
        """
        if write_log:
            self.log('Writing Cryocard PS enable using cryo_card ' +
                f'object to {enable}')
        self.C.write_ps_en(enable)

# which_on
def which_on(self, band):
        """
        Finds all detectors that are on.

        Args
        ----
        band : int
            The band to search.

        Returns
        --------
        int array
            The channels that are on.
        """
        amps = self.get_amplitude_scale_array(band)
        return np.ravel(np.where(amps != 0))

# band_off
def band_off(self, band, **kwargs):
        """
        Turns off all tones in a band

        Args
        ----
        band : int
            The band that is to be turned off.
        """
        # Warning ; you might think using the
        # set_amplitude_scale_array function would be fast than this
        # but it is apparently not!
        self.set_amplitude_scales(band, 0, **kwargs)
        n_channels = self.get_number_channels(band)
        self.set_feedback_enable_array(
            band, np.zeros(n_channels, dtype=int), **kwargs)
        self.set_cfg_reg_ena_bit(0, wait_after=.2, **kwargs)

# channel_off
def channel_off(self, band, channel, **kwargs):
        """
        Turns off the tone for a single channel by setting the amplitude to
        zero and disabling feedback.

        Args
        ----
        band : int
            The band that is to be turned off.
        channel : int
            The channel to turn off.
        """
        self.log(f'Turning off band {band} channel {channel}',
                 self.LOG_USER)
        self.set_amplitude_scale_channel(band, channel, 0, **kwargs)
        self.set_feedback_enable_channel(band, channel, 0, **kwargs)

# TUNING FUNCTIONS

# full_band_resp ***
def full_band_resp(self, band, n_scan=1, nsamp=2**19, make_plot=False,
            save_plot=True, show_plot=False, save_data=False, timestamp=None,
            save_raw_data=False, correct_att=True, swap=False, hw_trigger=True,
            write_log=False, return_plot_path=False,
            check_if_adc_is_saturated=True):
        """
        Injects high amplitude noise with known waveform. The ADC measures it.
        The cross correlation contains the information about the resonances.

        Args
        ----
        band : int
            The band to sweep.

        n_scan : int, optional, default 1
            The number of scans to take and average.
        nsamp : int, optional, default 2**19
            The number of samples to take.
        make_plot : bool, optional, default False
            Whether the make plots.
        save_plot : bool, optional, default True
            If making plots, whether to save them.
        show_plot : bool, optional, default False
            Whether to show plots.
        save_data : bool, optional, default False
            Whether to save the data.
        timestamp : str or None, optional, default None
            The timestamp as a string. If None, loads the current
            timestamp.
        save_raw_data : bool, optional, default False
            Whether to save the raw ADC/DAC data.
        correct_att : bool, optional, default True
            Correct the response for the attenuators.
        swap : bool, optional, default False
            Whether to reverse the data order of the ADC relative to
            the DAC. This solved a legacy problem.
        hw_trigger : bool, optional, default True
            Whether to start the broadband noise file using the
            hardware trigger.
        write_log : bool, optional, default False
            Whether to write output to the log.
        return_plot_path : bool, optional, default False
            Whether to return the full path to the summary plot.
        check_if_adc_is_saturated : bool, optional, default True
            Right after playing the noise file, checks if ADC for the
            requested band is saturated.  If it is saturated, gives up
            with an error.

        Returns
        -------
        f : float array
            The frequency information. Length nsamp/2.
        resp : complex array
            The response information. Length nsamp/2.
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        resp = np.zeros((int(n_scan), int(nsamp/2)), dtype=complex)
        for n in np.arange(n_scan):
            bay = self.band_to_bay(band)
            # Default setup sets to 1
            self.set_trigger_hw_arm(bay, 0, write_log=write_log)

            self.set_noise_select(band, 1, wait_done=True, write_log=write_log)
            # if true, checks whether or not playing noise file saturates the ADC.
            #If ADC is saturated, throws an exception.
            if check_if_adc_is_saturated:
                adc_is_saturated = self.check_adc_saturation(band)
                if adc_is_saturated:
                    raise ValueError('Playing the noise file saturates the '+
                        f'ADC for band {band}.  Try increasing the DC '+
                        'attenuation for this band.')

            # Take read the ADC data
            try:
                adc = self.read_adc_data(band, nsamp, hw_trigger=hw_trigger,
                    save_data=False)
            except Exception:
                self.log('ADC read failed. Trying one more time', self.LOG_ERROR)
                adc = self.read_adc_data(band, nsamp, hw_trigger=hw_trigger,
                    save_data=False)
            time.sleep(.05)  # Need to wait, otherwise dac call interferes with adc

            try:
                dac = self.read_dac_data(
                    band, nsamp, hw_trigger=hw_trigger,
                    save_data=False)
            except BaseException:
                self.log('ADC read failed. Trying one more time', self.LOG_ERROR)
                dac = self.read_dac_data(
                    band, nsamp, hw_trigger=hw_trigger,
                    save_data=False)
            time.sleep(.05)

            self.set_noise_select(band, 0, wait_done=True, write_log=write_log)

            # Account for the up and down converter attenuators
            if correct_att:
                att_uc = self.get_att_uc(band)
                att_dc = self.get_att_dc(band)
                self.log(f'UC (DAC) att: {att_uc}')
                self.log(f'DC (ADC) att: {att_dc}')
                if att_uc > 0:
                    scale = (10**(-att_uc/2/20))
                    self.log(f'UC attenuator > 0. Scaling by {scale:4.3f}')
                    dac *= scale
                if att_dc > 0:
                    scale = (10**(att_dc/2/20))
                    self.log(f'DC attenuator > 0. Scaling by {scale:4.3f}')
                    adc *= scale

            if save_raw_data:
                self.log('Saving raw data...', self.LOG_USER)

                path = os.path.join(self.output_dir, f'{timestamp}_adc')
                np.save(path, adc)
                self.pub.register_file(path, 'adc', format='npy')

                path = os.path.join(self.output_dir,f'{timestamp}_dac')
                np.save(path, dac)
                self.pub.register_file(path, 'dac', format='npy')

            # Swap frequency ordering of data of ADC relative to DAC
            if swap:
                adc = adc[::-1]

            # Take PSDs of ADC, DAC, and cross
            fs = self.get_digitizer_frequency_mhz() * 1.0E6
            f, p_dac = signal.welch(dac, fs=fs, nperseg=nsamp/2,
                                    return_onesided=False)
            f, p_adc = signal.welch(adc, fs=fs, nperseg=nsamp/2,
                                    return_onesided=False)
            f, p_cross = signal.csd(dac, adc, fs=fs, nperseg=nsamp/2,
                                    return_onesided=False)

            # Sort frequencies
            idx = np.argsort(f)
            f = f[idx]
            p_dac = p_dac[idx]
            p_adc = p_adc[idx]
            p_cross = p_cross[idx]

            resp[n] = p_cross / p_dac

        # Average over the multiple scans
        resp = np.mean(resp, axis=0)

        plot_path = None
        if make_plot:
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

            if save_plot:
                path = os.path.join(
                    self.plot_dir,
                    f'{timestamp}_b{band}_full_band_resp_raw.png')
                plt.savefig(path, bbox_inches='tight')
                self.pub.register_file(path, 'response', plot=True)
                plt.close()

            fig, ax = plt.subplots(1, figsize=(5.5, 3))

            # Log y-scale plot
            ax.plot(f_plot[plot_idx], np.log10(np.abs(resp[plot_idx])))
            ax.set_xlabel('Freq [MHz]')
            ax.set_ylabel('Response')
            ax.set_title(f'full_band_resp {timestamp}')
            plt.tight_layout()
            if save_plot:
                plot_path = (
                    os.path.join(
                        self.plot_dir,
                        f'{timestamp}_b{band}_full_band_resp.png'))

                plt.savefig(plot_path, bbox_inches='tight')
                self.pub.register_file(plot_path, 'response', plot=True)

            # Show/Close plots
            if show_plot:
                plt.show()
            else:
                plt.close()

        if save_data:
            save_name = timestamp + '_{}_full_band_resp.txt'

            path = os.path.join(self.output_dir, save_name.format('freq'))
            np.savetxt(path, f)
            self.pub.register_file(path, 'full_band_resp', format='txt')

            path = os.path.join(self.output_dir, save_name.format('real'))
            np.savetxt(path, np.real(resp))
            self.pub.register_file(path, 'full_band_resp', format='txt')

            path = os.path.join(self.output_dir, save_name.format('imag'))
            np.savetxt(path, np.imag(resp))
            self.pub.register_file(path, 'full_band_resp', format='txt')

        if return_plot_path:
            return f, resp, plot_path
        else:
            return f, resp

# find_freq ***
def find_freq(self, band, start_freq=-250, stop_freq=250, subband=None,
            tone_power=None, n_read=2, make_plot=False, save_plot=True,
            plotname_append='', window=50, rolling_med=True,
            make_subband_plot=False, show_plot=False, grad_cut=.05,
            flip_phase=False, grad_kernel_width=8,
            amp_cut=.25, pad=2, min_gap=2):
        '''
        Finds the resonances in a band (and specified subbands)

        Args
        ----
        band : int
            The band to search.
        start_freq : float, optional, default -250
            The scan start frequency in MHz (from band center)
        stop_freq : float, optional, default 250
            The scan stop frequency in MHz (from band center)
        subband : deprecated, use start_freq/stop_freq.
            numpy.ndarray of int or None, optional, default None
            An int array for the subbands.  If None, set to all
            processed subbands =numpy.arange(13,115).
            Takes precedent over start_freq/stop_freq.
        tone_power : int or None, optional, default None
            The drive amplitude.  If None, takes from cfg.
        n_read : int, optional, default 2
            The number sweeps to do per subband.
        make_plot : bool, optional, default False
            Make the plot frequency sweep.
        save_plot : bool, optional, default True
            Save the plot.
        plotname_append : str, optional, default ''
            Appended to the default plot filename.
        window : int, optional, default 50
            The width of the rolling median window.
        rolling_med : bool, optional, default True
            Whether to iterate on a rolling median or just the median
            of the whole sample.
        grad_cut : float, optional, default 0.05
            The value of the gradient of phase to look for
            resonances.
        flip_phase : bool, optional, default False
            Whether to flip the sign of phase before
            evaluating the gradient cut.
        amp_cut : float, optional, default 0.25
            The fractional distance from the median value to decide
            whether there is a resonance.
        pad : int, optional, default 2
            Number of samples to pad on either side of a resonance
            search window
        min_gap : int, optional, default 2
            Minimum number of samples between resonances.
        '''
        band_center = self.get_band_center_mhz(band)
        if subband is None:
            start_subband = self.freq_to_subband(band, band_center + start_freq)[0]
            stop_subband = self.freq_to_subband(band, band_center + stop_freq)[0]
            step = 1
            if stop_subband < start_subband:
                step = -1
            subband = np.arange(start_subband, stop_subband+1, step)
        else:
            sb, sbc = self.get_subband_centers(band)
            start_freq = sbc[subband[0]]
            stop_freq  = sbc[subband[-1]]

        # Turn off all tones in this band first.  May want to make
        # this only turn off tones in each sub-band before sweeping,
        # instead?
        self.band_off(band)

        if tone_power is None:
            tone_power = self._amplitude_scale[band]
            self.log('No tone_power given. Using value in config ' +
                     f'file: {tone_power}')

        self.log(f'Sweeping across frequencies {start_freq + band_center}MHz to {stop_freq + band_center}MHz')
        f, resp = self.full_band_ampl_sweep(band, subband, tone_power, n_read)

        timestamp = self.get_timestamp()

        # Save data
        save_name = '{}_amp_sweep_{}.txt'

        path = os.path.join(self.output_dir, save_name.format(timestamp, 'freq'))
        np.savetxt(path, f)
        self.pub.register_file(path, 'sweep_response', format='txt')

        path = os.path.join(self.output_dir, save_name.format(timestamp, 'resp'))
        np.savetxt(path, resp)
        self.pub.register_file(path, 'sweep_response', format='txt')

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

        # Find resonator peaks
        res_freq = self.find_all_peak(self.freq_resp[band]['find_freq']['f'],
            self.freq_resp[band]['find_freq']['resp'], subband,
            make_plot=make_plot, save_plot=save_plot, show_plot=show_plot,
            plotname_append=plotname_append, band=band,
            rolling_med=rolling_med, window=window,
            make_subband_plot=make_subband_plot, grad_cut=grad_cut,
            flip_phase=flip_phase, grad_kernel_width=grad_kernel_width,
            amp_cut=amp_cut, pad=pad, min_gap=min_gap)
        self.freq_resp[band]['find_freq']['resonance'] = res_freq

        # Save resonances
        path = os.path.join(self.output_dir,
            save_name.format(timestamp, 'resonance'))
        np.savetxt(path, self.freq_resp[band]['find_freq']['resonance'])
        self.pub.register_file(path, 'resonances', format='txt')

        # Call plotting
        if make_plot:
            self.plot_find_freq(self.freq_resp[band]['find_freq']['f'],
                self.freq_resp[band]['find_freq']['resp'],
                subband=np.arange(self.get_number_sub_bands(band)),
                save_plot=save_plot,
                show_plot=show_plot,
                save_name=save_name.replace('.txt', plotname_append +
                                            '.png').format(timestamp, band))


        return f, resp

# setup_notches ***
def setup_notches(self, band, resonance=None, tone_power=None,
                      sweep_width=.3, df_sweep=.002, min_offset=0.1,
                      delta_freq=None, new_master_assignment=False,
                      lock_max_derivative=False,
                      scan_unassigned=False):
        """
        Does a fine sweep over the resonances found in find_freq. This
        information is used for placing tones onto resonators. It is
        recommended that you follow this up with run_serial_gradient_descent()
        afterwards.

        Args
        ----
        band : int
            The 500 MHz band to setup.
        resonance : float array or None, optional, default None
            A 2 dimensional array with resonance frequencies and the
            subband they are in. If given, this will take precedent
            over the one in self.freq_resp.
        tone_power : int or None, optional, default None
            The power to drive the resonators. Default is defined in cfg file.
        sweep_width : float, optional, default 0.3
            The range to scan around the input resonance in units of
            MHz.
        df_sweep : float, optional, default 0.002
            The sweep step size in MHz.
        min_offset : float, optional, default 0.1
            Minimum distance in MHz between two resonators for assigning channels.
        delta_freq : float or None, optional, default None
            The frequency offset at which to measure the complex
            transmission to compute the eta parameters.  Passed to
            eta_estimator.  Units are MHz.  If None, takes value in
            config file.
        new_master_assignment : bool, optional, default False
            Whether to create a new master assignment file. This file
            defines the mapping between resonator frequency and
            channel number.
        scan_unassigned : bool, optional, default False
            Whether or not to scan unassigned channels.  Unassigned
            channels are scanned serially after the much faster
            assigned channel scans.
        """
        # Turn off all tones in this band first
        self.band_off(band)

        # Check if any resonances are stored
        if 'find_freq' not in self.freq_resp[band]:
            self.log(f'No find_freq in freq_resp dictionary for band {band}. ' +
                     'Run find_freq first.', self.LOG_ERROR)
            return
        elif 'resonance' not in self.freq_resp[band]['find_freq'] and resonance is None:
            self.log(
                f'No resonances stored in band {band}' +
                '. Run find_freq first.', self.LOG_ERROR)
            return

        if tone_power is None:
            tone_power = self._amplitude_scale[band]
            self.log(
                f'No tone_power given. Using value in config file: {tone_power}')

        if delta_freq is None:
            delta_freq = self._delta_freq[band]

        if resonance is not None:
            input_res = resonance
        else:
            input_res = self.freq_resp[band]['find_freq']['resonance']

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        digitizer_frequency_mhz = self.get_digitizer_frequency_mhz(band)
        subband_half_width = digitizer_frequency_mhz/\
            n_subbands

        self.freq_resp[band]['tone_power'] = tone_power

        # Loop over inputs and do eta scans
        resonances = {}
        band_center = self.get_band_center_mhz(band)
        input_res = input_res + band_center

        n_res = len(input_res)
        for i, f in enumerate(input_res):
            self.log(f'freq {f:5.4f} - {i+1} of {n_res}')
            # fillers for now
            f_min = f
            freq  = None
            resp  = None
            eta   = 1
            eta_scaled = 1
            eta_phase_deg = 0
            eta_mag = 1

            resonances[i] = {
                'freq' : f_min,
                'eta' : eta,
                'eta_scaled' : eta_scaled,
                'eta_phase' : eta_phase_deg,
                'r2': 1, # This is BS
                'eta_mag' : eta_mag,
                'latency': 0,  # This is also BS
                'Q' : 1,  # This is also also BS
                'freq_eta_scan' : freq,
                'resp_eta_scan' : resp
            }

        subbands, channels, offsets = self.assign_channels(input_res, band=band,
            as_offset=False, min_offset=min_offset,
            new_master_assignment=new_master_assignment)

        f_sweep = np.arange(-sweep_width, sweep_width, df_sweep)
        n_step  = len(f_sweep)

        resp = np.zeros((n_channels, n_step), dtype=complex)
        freq = np.zeros((n_channels, n_step))

        for i, ch in enumerate(channels):
            freq[ch, :] = offsets[i] + f_sweep

        self.set_eta_scan_freq(band, freq.flatten())
        self.set_eta_scan_amplitude(band, tone_power)
        self.set_run_serial_find_freq(band, 1)

        I = self.get_eta_scan_results_real(band, count=n_step*n_channels)
        I = np.asarray(I)
        idx = np.where( I > 2**23 )
        I[idx] = I[idx] - 2**24
        I /= 2**23
        I = I.reshape(n_channels, n_step)

        Q = self.get_eta_scan_results_imag(band, count=n_step*n_channels)
        Q = np.asarray(Q)
        idx = np.where( Q > 2**23 )
        Q[idx] = Q[idx] - 2**24
        Q /= 2**23
        Q = Q.reshape(n_channels, n_step)

        resp = I + 1j*Q

        for i, channel in enumerate(channels):
            # the fast eta scan sweep only returns valid data for
            # assigned channels.
            if channel!=-1:
                freq_s, resp_s, eta = self.eta_estimator(band,
                                                         subbands[i],
                                                         freq[channel, :],
                                                         resp[channel, :],
                                                         delta_freq=delta_freq,
                                                         lock_max_derivative=lock_max_derivative)
                eta_phase_deg = np.angle(eta)*180/np.pi
                eta_mag = np.abs(eta)
                eta_scaled = eta_mag / subband_half_width

                abs_resp = np.abs(resp_s)
                idx = np.ravel(np.where(abs_resp == np.min(abs_resp)))[0]

                f_min = freq_s[idx]

                resonances[i] = {
                    'freq' : f_min,
                    'eta' : eta,
                    'eta_scaled' : eta_scaled,
                    'eta_phase' : eta_phase_deg,
                    'r2' : 1,  # This is BS
                    'eta_mag' : eta_mag,
                    'latency': 0,  # This is also BS
                    'Q' : 1,  # This is also also BS
                    'freq_eta_scan' : freq_s,
                    'resp_eta_scan' : resp_s
                }

            elif channel==-1 and scan_unassigned:
                self.log(
                    f'scan_unassiged=True : Scanning unassigned frequency at {input_res[i]:.3f} MHz.',self.LOG_USER)

                # Unassigned channels aren't scanned in the time
                # optimized assigned channel scans above.  This could
                # be sped up.  For now, just inefficiently running
                # same function used for assigned channels, just to be
                # sure we're computing everything the same way.

                # Run single eta scan on just this channel.
                # "u" for unassigned
                frequ = offsets[i] + f_sweep
                Iu,Qu = self.eta_scan(band,
                                      subbands[i],
                                      frequ,
                                      tone_power)

                idx = np.where( Iu > 2**23 )
                Iu[idx] = Iu[idx] - 2**24
                Iu /= 2**23

                idx = np.where( Qu > 2**23 )
                Qu[idx] = Qu[idx] - 2**24
                Qu /= 2**23

                # The eta_scan has a different convention for
                # inphase/quadrature.  This remaps it to match the
                # convention used by serialFindFreq, which we use for
                # assigned channels.
                respu = Qu - 1j*Iu

                # estimate eta from response
                frequ_s, respu_s, etau = self.eta_estimator(band,
                                                            subbands[i],
                                                            frequ,
                                                            respu,
                                                            delta_freq=delta_freq,
                                                            lock_max_derivative=lock_max_derivative)

                eta_phase_degu = np.angle(etau)*180/np.pi
                eta_magu = np.abs(etau)
                eta_scaledu = eta_magu / subband_half_width

                abs_respu = np.abs(respu_s)
                idx = np.ravel(np.where(abs_respu == np.min(abs_respu)))[0]

                f_minu = frequ_s[idx]

                resonances[i] = {
                    'freq' : f_minu,
                    'eta' : etau,
                    'eta_scaled' : eta_scaledu,
                    'eta_phase' : eta_phase_degu,
                    'r2' : 1,  # This is BS
                    'eta_mag' : eta_magu,
                    'latency': 0,  # This is also BS
                    'Q' : 1,  # This is also also BS
                    'freq_eta_scan' : frequ_s,
                    'resp_eta_scan' : respu_s
                }

        # Assign resonances to channels
        self.log('Assigning channels')
        f = [resonances[k]['freq'] for k in resonances.keys()]

        subbands, channels, offsets = self.assign_channels(f, band=band,
            as_offset=False, min_offset=min_offset,
            new_master_assignment=new_master_assignment)

        for i, k in enumerate(resonances.keys()):
            resonances[k].update({'subband': subbands[i]})
            resonances[k].update({'channel': channels[i]})
            resonances[k].update({'offset': offsets[i]})

        self.freq_resp[band]['resonances'] = resonances

        self.save_tune()

        self.relock(band)

# run_serial_gradient_descent
def run_serial_gradient_descent(self, band, sync_group=True,
                                    timeout=240, **kwargs):
        """
        Does a gradient descent search for the minimum.

        Args
        ----
        band : int
            The band to run serial gradient descent on.
        sync_group : bool, optional, default True
            Whether to use the sync group to monitor the PV.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_gradient_descent_reg
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            sg.get_values()


    #_sel_ext_ref_reg = "SelExtRef"

# run_serial_eta_scan
def run_serial_eta_scan(self, band, sync_group=True, timeout=240,
                            **kwargs):
        """
        Does an eta scan serially across the entire band. You must
        already be tuned close to the resontor dip. Use
        run_serial_gradient_descent to get it.

        Args
        ----
        band  : int
            The band to eta scan.
        sync_group : bool, optional, default True
            Whether to use the sync group to monitor the PV.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_eta_scan_reg
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            sg.get_values()


    #_run_serial_min_search_reg = 'runSerialMinSearch'

# plot_tune_summary ***
def plot_tune_summary(self, band, eta_scan=False, show_plot=False,
            save_plot=True, eta_width=.3, channels=None,
            plot_summary=True, plotname_append=''):
        """
        Plots summary of tuning. Requires self.freq_resp to be filled.
        In other words, you must run find_freq and setup_notches
        before calling this function. Saves the plot to plot_dir.
        This will also make individual eta plots as well if {eta_scan}
        is True.  The eta scan plots are slow because there are many
        of them.

        Args
        ----
        band : int
            The band number to plot.
        eta_scan : bool, optional, default False
           Whether to also plot individual eta scans.  Warning this is
           slow.
        show_plot : bool, optional, default False
            Whether to display the plot.
        save_plot : bool, optional, default True
            Whether to save the plot.
        eta_width : float, optional, default 0.3
            The width to plot in MHz.
        channels : list of int or None, optional, default None
            Which channels to plot.  If None, plots all available
            channels.
        plot_summary : bool, optional, default True
            Plot summary.
        plotname_append : str, optional, default ''
            Appended to the default plot filename.
        """
        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        timestamp = self.get_timestamp()

        if plot_summary:
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
            ax[0,0].text(.02, .92, f'Total: {len(sb)}',
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

            fig.suptitle(f'Band {band} {timestamp}')
            plt.subplots_adjust(left=.08, right=.95, top=.92, bottom=.08,
                                wspace=.21, hspace=.21)

            if save_plot:
                save_name = (
                    f'{timestamp}_tune_summary{plotname_append}.png')
                path = os.path.join(self.plot_dir, save_name)
                plt.savefig(path, bbox_inches='tight')
                self.pub.register_file(path, 'tune', plot=True)
                if not show_plot:
                    plt.close()

        # Plot individual eta scan
        if eta_scan:
            keys = self.freq_resp[band]['resonances'].keys()
            # If using full band response as input
            if 'full_band_resp' in self.freq_resp[band]:
                freq = self.freq_resp[band]['full_band_resp']['freq']
                resp = self.freq_resp[band]['full_band_resp']['resp']
                for k in keys:
                    r = self.freq_resp[band]['resonances'][k]
                    channel=r['channel']
                    # If user provides a channel restriction list, only
                    # plot channels in that list.
                    if channel is not None and channel not in channels:
                        continue
                    center_freq = r['freq']
                    idx = np.logical_and(freq > center_freq - eta_width,
                        freq < center_freq + eta_width)

                    # Actually plot the data
                    self.plot_eta_fit(freq[idx], resp[idx],
                        eta_mag=r['eta_mag'], eta_phase_deg=r['eta_phase'],
                        band=band, res_num=k, timestamp=timestamp,
                        save_plot=save_plot, show_plot=show_plot,
                        peak_freq=center_freq, channel=channel, plotname_append=plotname_append)
            # This is for data from find_freq/setup_notches
            else:
                # For setup_notches plotting, only count & plot existing data;
                # e.g. unassigned channels may not be scanned.
                scanned_keys=np.array([k for k in keys if self.freq_resp[band]['resonances'][k]['resp_eta_scan'] is not None])
                n_scanned_keys=len(scanned_keys)
                for skidx,k in enumerate(scanned_keys):
                    r = self.freq_resp[band]['resonances'][k]
                    channel=r['channel']
                    # If user provides a channel restriction list, only
                    # plot channels in that list.
                    if channels is not None:
                        if channel not in channels:
                            continue
                        else:
                            self.log(f'Eta plot for channel {channel}')
                    else:
                        self.log(f'Eta plot {skidx+1} of {n_scanned_keys}')
                    self.plot_eta_fit(r['freq_eta_scan'], r['resp_eta_scan'],
                        eta=r['eta'], eta_mag=r['eta_mag'],
                        eta_phase_deg=r['eta_phase'], band=band, res_num=k,
                        timestamp=timestamp, save_plot=save_plot,
                        show_plot=show_plot, peak_freq=r['freq'],
                        channel=channel, plotname_append=plotname_append)

# tracking_setup ***
def tracking_setup(self, band, channel=None, reset_rate_khz=None,
            write_log=False, make_plot=False, save_plot=True, show_plot=True,
            nsamp=2**19, lms_freq_hz=None, meas_lms_freq=False,
            meas_flux_ramp_amp=False, n_phi0=4, flux_ramp=True,
            fraction_full_scale=None, lms_enable1=True, lms_enable2=True,
            lms_enable3=True, feedback_gain=None, lms_gain=None, return_data=True,
            new_epics_root=None, feedback_start_frac=None,
            feedback_end_frac=None, setup_flux_ramp=True, plotname_append=''):
        """
        The function to start tracking. Starts the flux ramp and if requested
        attempts to measure the lms (demodulation) frequency. Otherwise this
        just tracks at the input lms frequency. This will also make plots for
        the channels listed in {channel} input.

        Args
        ----
        band : int
            The band number.
        channel : int or int array or None, optional, default None
            The channels to plot.
        reset_rate_khz : float or None, optional, default None
            The flux ramp frequency.
        write_log : bool, optional, default False
            Whether to write output to the log.
        make_plot : bool, optional, default False
            Whether to make plots.
        save_plot : bool, optional, default True
            Whether to save plots.
        show_plot : bool, optional, default True
            Whether to display the plot.
        nsamp : int, optional, default 2**19
            The number of samples to take of the flux ramp.
        lms_freq_hz : float or None, optional, default None
            The frequency of the tracking algorithm.
        meas_lms_freq : bool, optional, default False
            Whether or not to try to estimate the carrier rate using
            the flux_mod2 function.  lms_freq_hz must be None.
        meas_flux_ramp_amp : bool, optional, default False
            Whether or not to adjust fraction_full_scale to get the number of
            phi0 defined by n_phi0. lms_freq_hz must be None for this to work.
        n_phi0 : float, optional, default 4
            The number of phi0 to match using meas_flux_ramp_amp.
        flux_ramp : bool, optional, default True
            Whether to turn on flux ramp.
        fraction_full_scale : float or None, optional, default None
            The flux ramp amplitude, as a fraction of the maximum.
        lms_enable1 : bool, optional, default True
            Whether to use the first harmonic for tracking.
        lms_enable2 : bool, optional, default True
            Whether to use the second harmonic for tracking.
        lms_enable3 : bool, optional, default True
            Whether to use the third harmonic for tracking.
        feedback_gain : Int16, optional, default None.
            The tracking feedback gain parameter.
            This is applied to all channels within a band.
            1024 corresponds to approx 2kHz bandwidth.
            2048 corresponds to approx 4kHz bandwidth.
            Tune this parameter to track the demodulated band
            of interest (0...2kHz for 4kHz flux ramp).
            High gains may affect noise performance and will
            eventually cause the tracking loop to go unstable.
        lms_gain : int or None, optional, default None
            ** Internal register dynamic range adjustment **
            ** Use with caution - you probably want feedback_gain**
            Select which bits to slice from accumulation over
            a flux ramp period.
            Tracking feedback parameters are integrated over a flux
            ramp period at 2.4MHz.  The internal register allows for up
            to 9 bits of growth (from full scale).
            lms_gain = 0 : select upper bits from accumulation register (9 bits growth)
            lms_gain = 1 : select upper bits from accumulation register (8 bits growth)
            lms_gain = 2 : select upper bits from accumulation register (7 bits growth)
            lms_gain = 3 : select upper bits from accumulation register (6 bits growth)
            lms_gain = 4 : select upper bits from accumulation register (5 bits growth)
            lms_gain = 5 : select upper bits from accumulation register (4 bits growth)
            lms_gain = 6 : select upper bits from accumulation register (3 bits growth)
            lms_gain = 7 : select upper bits from accumulation register (2 bits growth)

            The max bit gain is given by ceil(log2(2.4e6/FR_rate)).
            For example a 4kHz FR can  accumulate ceil(log2(2.4e6/4e3)) = 10 bits
            if the I/Q tracking parameters are at full scale (+/- 2.4MHz)
            Typical SQUID frequency throws of 100kHz have a bit growth of
            ceil(log2( (100e3/2.4e6)*(2.4e6/FR_rate) ))
            So 100kHz SQUID throw at 4kHz has bit growth ceil(log2(100e3/4e3)) = 5 bits.
            Try lms_gain = 4.

            This should be approx 9 - ceil(log2(100/reset_rate_khz)) for CMB applications.

            Too low of lms_gain will use only a small dynamic range of the streaming
            registers and contribute to incrased noise.
            Too high of lms_gain will overflow the register and greatly incrase noise.
        return_data : bool, optional, default True
            Whether or not to return f, df, sync.
        new_epics_root : str or None, optional, default None
            Override the original epics root.
        feedback_start_frac : float or None, optional, default None
            The fraction of the full flux ramp at which to stop
            applying feedback in each flux ramp cycle.  Must be in
            [0,1).  Defaults to whatever's in the cfg file.
        feedback_end_frac : float or None, optional, default None
            The fraction of the full flux ramp at which to stop
            applying feedback in each flux ramp cycle.  Must be >0.
            Defaults to whatever's in the cfg file.
        setup_flux_ramp : bool, optional, default True
            Whether to setup the flux ramp.
        plotname_append : str, optional, default ''
            Optional string to append plots with.
        """
        if reset_rate_khz is None:
            reset_rate_khz = self._reset_rate_khz
        if lms_gain is None:
            lms_gain = int(9 - np.ceil(np.log2(100/reset_rate_khz)))
            if lms_gain > 7:
                lms_gain = 7
        else:
            self.log("Using LMS gain is now an advanced feature.")
            self.log("Unless you are an expert, you probably want feedback_gain.")
            self.log("See tracking_setup docstring.")
        if feedback_gain is None:
            feedback_gain = self._feedback_gain[band]

        ##
        ## Load unprovided optional args from cfg
        if feedback_start_frac is None:
            feedback_start_frac = self._feedback_start_frac[band]
        if feedback_end_frac is None:
            feedback_end_frac = self._feedback_end_frac[band]
        ## End loading unprovided optional args from cfg
        ##

        ##
        ## Argument validation

        # Validate feedback_start_frac and feedback_end_frac
        if (feedback_start_frac < 0) or (feedback_start_frac >= 1):
            raise ValueError(
                f"feedback_start_frac = {feedback_start_frac} " +
                "not in [0,1)")
        if (feedback_end_frac < 0):
            raise ValueError(
                f"feedback_end_frac = {feedback_end_frac} not > 0")
        # If feedback_start_frac exceeds feedback_end_frac, then
        # there's no range of the flux ramp cycle over which we're
        # applying feedback.
        if (feedback_end_frac < feedback_start_frac):
            raise ValueError(
                f"feedback_end_frac = {feedback_end_frac} " +
                "is not less than " +
                f"feedback_start_frac = {feedback_start_frac}")
        # Done validating feedbackStart and feedbackEnd

        ## End argument validation
        ##

        if not flux_ramp:
            self.log('WARNING: THIS WILL NOT TURN ON FLUX RAMP!')

        if make_plot:
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

        if fraction_full_scale is None:
            fraction_full_scale = self._fraction_full_scale
        else:
            self.fraction_full_scale = fraction_full_scale

        # Measure either LMS freq or the flux ramp amplitude
        if lms_freq_hz is None:
            if meas_lms_freq and meas_flux_ramp_amp:
                self.log('Requested measurement of both LMS freq '+
                         'and flux ramp amplitude. Cannot do both.',
                         self.LOG_ERROR)
                return None, None, None
            elif meas_lms_freq:
                # attempts to measure the flux ramp frequency and leave the
                # flux ramp amplitude the same
                lms_freq_hz = self.estimate_lms_freq(
                    band, reset_rate_khz,
                    fraction_full_scale=fraction_full_scale,
                    channel=channel)
            elif meas_flux_ramp_amp:
                # attempts to measure the the number of phi0 and adjust
                # the ampltidue of the flux ramp to achieve the desired number
                # of phi0 per flux ramp
                fraction_full_scale = self.estimate_flux_ramp_amp(band,
                    n_phi0,reset_rate_khz=reset_rate_khz, channel=channel)
                lms_freq_hz = reset_rate_khz * n_phi0 * 1.0E3
            else:
                # Load from config
                lms_freq_hz = self._lms_freq_hz[band]
            self._lms_freq_hz[band] = lms_freq_hz
            if write_log:
                self.log('Using lms_freq_estimator : ' +
                    f'{lms_freq_hz:.0f} Hz')

        if not flux_ramp:
            lms_enable1 = 0
            lms_enable2 = 0
            lms_enable3 = 0

        if write_log:
            self.log("Using lmsFreqHz = " +
                     f"{lms_freq_hz:.0f} Hz",
                     self.LOG_USER)

        self.set_feedback_gain(band, feedback_gain, write_log=write_log)
        self.set_lms_gain(band, lms_gain, write_log=write_log)
        self.set_lms_enable1(band, lms_enable1, write_log=write_log)
        self.set_lms_enable2(band, lms_enable2, write_log=write_log)
        self.set_lms_enable3(band, lms_enable3, write_log=write_log)
        self.set_lms_freq_hz(band, lms_freq_hz, write_log=write_log)

        iq_stream_enable = 0  # must be zero to access f,df stream
        self.set_iq_stream_enable(band, iq_stream_enable, write_log=write_log)

        if setup_flux_ramp:
            self.flux_ramp_setup(reset_rate_khz, fraction_full_scale,
                             write_log=write_log, new_epics_root=new_epics_root)
        else:
            self.log("Not changing flux ramp status. Use setup_flux_ramp " +
                     "boolean to run flux_ramp_setup")

        # Doing this after flux_ramp_setup so that if needed we can
        # set feedback_end based on the flux ramp settings.
        feedback_start = self._feedback_frac_to_feedback(band, feedback_start_frac)
        feedback_end = self._feedback_frac_to_feedback(band, feedback_end_frac)

        # Set feedbackStart and feedbackEnd
        self.set_feedback_start(band, feedback_start, write_log=write_log)
        self.set_feedback_end(band, feedback_end, write_log=write_log)

        if write_log:
            self.log("Applying feedback over "+
                f"{(feedback_end_frac-feedback_start_frac)*100.:.1f}% of each "+
                f"flux ramp cycle (with feedbackStart={feedback_start} and " +
                f"feedbackEnd={feedback_end})", self.LOG_USER)

        if flux_ramp:
            self.flux_ramp_on(write_log=write_log, new_epics_root=new_epics_root)

        # take one dataset with all channels
        if return_data or make_plot:
            f, df, sync = self.take_debug_data(band, IQstream=iq_stream_enable,
                single_channel_readout=0, nsamp=nsamp)

            df_std = np.std(df, 0)
            df_channels = np.ravel(np.where(df_std >0))

            # Intersection of channels that are on and have some flux ramp resp
            channels_on = list(set(df_channels) & set(self.which_on(band)))

            self.log(f"Number of channels on : {len(self.which_on(band))}",
                     self.LOG_USER)
            self.log("Number of channels on with flux ramp "+
                f"response : {len(channels_on)}", self.LOG_USER)

            f_span = np.max(f,0) - np.min(f,0)

        if make_plot:
            timestamp = self.get_timestamp()

            fig,ax = plt.subplots(1,3, figsize=(12,5))
            fig.suptitle(f'{timestamp} Band {band}')

            # Histogram the stddev
            ax[0].hist(df_std[channels_on]*1e3, bins=20, edgecolor = 'k')
            ax[0].set_xlabel('Flux ramp demod error std (kHz)')
            ax[0].set_ylabel('number of channels')

            # Histogram the max-min flux ramp amplitude response
            ax[1].hist(f_span[channels_on]*1e3, bins=20, edgecolor='k')
            ax[1].set_xlabel('Flux ramp amplitude (kHz)')
            ax[1].set_ylabel('number of channels')

            # Plot df vs resp amplitude
            ax[2].plot(f_span[channels_on]*1e3, df_std[channels_on]*1e3, '.')
            ax[2].set_xlabel('FR Amp (kHz)')
            ax[2].set_ylabel('RF demod error (kHz)')
            x = np.array([0, np.max(f_span[channels_on])*1.0E3])

            # useful line to guide the eye
            y_factor = 100
            y = x/y_factor
            ax[2].plot(x, y, color='k', linestyle=':',label=f'1:{y_factor}')
            ax[2].legend(loc='upper right')

            bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)

            text = f"Reset rate: {reset_rate_khz} kHz" + "\n" + \
                f"LMS freq: {lms_freq_hz:.0f} Hz" + "\n" + \
                f"LMS gain: {lms_gain}" + "\n" + \
                f"FR amp: {self.get_fraction_full_scale():1.3f}" + "\n" + \
                f"FB start: {feedback_start_frac}" + "\n" + \
                f"FB end: {feedback_end_frac}" + "\n" + \
                f"FB enable 1/2/3 : {lms_enable1}/{lms_enable2}/{lms_enable3}" + \
                "\n" + \
                r"$n_{chan}$:" + f" {len(channels_on)}"
            ax[2].text(.05, .97, text, transform=ax[2].transAxes, va='top',
                ha='left', fontsize=10, bbox=bbox)

            fig.tight_layout(rect=[0, 0.03, 1, 0.95])

            if save_plot:
                path = os.path.join(self.plot_dir,
                    timestamp + '_FR_amp_v_err' + plotname_append + '.png')
                plt.savefig(path, bbox_inches='tight')
                self.pub.register_file(path, 'amp_vs_err', plot=True)

            if not show_plot:
                # If we don't want a live view, close the plot
                plt.close()

            if channel is not None:
                channel = np.ravel(np.array(channel))
                sync_idx = self.make_sync_flag(sync)

                for ch in channel:
                    # Setup plotting
                    fig, ax = plt.subplots(2, sharex=True, figsize=(9, 4.75))

                    # Plot tracked component
                    ax[0].plot(f[:, ch]*1e3)
                    ax[0].set_ylabel('Tracked Freq [kHz]')
                    ax[0].text(.025, .93,
                        f'LMS Freq {lms_freq_hz:.0f} Hz', fontsize=10,
                        transform=ax[0].transAxes, bbox=bbox, ha='left',
                        va='top')

                    ax[0].text(.95, .93, f'Band {band} Ch {ch:03}',
                        fontsize=10, transform=ax[0].transAxes, ha='right',
                        va='top', bbox=bbox)

                    # Plot the untracking part
                    ax[1].plot(df[:, ch]*1e3)
                    ax[1].set_ylabel('Freq Error [kHz]')
                    ax[1].set_xlabel('Samp Num')
                    ax[1].text(.025, .93,
                        f'RMS error = {df_std[ch]*1e3:.2f} kHz\n' +
                        f'FR frac. full scale = {fraction_full_scale:.2f}',
                        fontsize=10, transform=ax[1].transAxes, bbox=bbox,
                        ha='left', va='top')

                    n_sync_idx = len(sync_idx)
                    for i, s in enumerate(sync_idx):
                        # Lines for reset
                        ax[0].axvline(s, color='k', linestyle=':', alpha=.5)
                        ax[1].axvline(s, color='k', linestyle=':', alpha=.5)

                        # highlight used regions
                        if i < n_sync_idx-1:
                            nsamp = sync_idx[i+1]-sync_idx[i]
                            start = s + feedback_start_frac*nsamp
                            end = s + feedback_end_frac*nsamp

                            ax[0].axvspan(start, end, color='k', alpha=.15)
                            ax[1].axvspan(start, end, color='k', alpha=.15)
                    plt.tight_layout()

                    if save_plot:
                        path = os.path.join(self.plot_dir, timestamp +
                            f'_FRtracking_b{band}_ch{ch:03}{plotname_append}.png')
                        plt.savefig(path, bbox_inches='tight')
                        self.pub.register_file(path, 'tracking', plot=True)

                    if not show_plot:
                        plt.close()

        self.set_iq_stream_enable(band, 1, write_log=write_log)

        if return_data:
            return f, df, sync

# set_amplitude_scale_array
def set_amplitude_scale_array(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._amplitude_scale_array_reg,
            val, **kwargs)

# TES/FLUX RAMP FUNCTIONS

# set_tes_bias_bipolar_array
def set_tes_bias_bipolar_array(self, bias_group_volt_array, do_enable=False,
                                   **kwargs):
        """
        Set TES bipolar values for all DACs at once.  Set using a
        pyrogue array write, so should be much more efficient than
        setting each TES bias one at a time (a single register
        transaction vs. many).  Only DACs assigned to TES bias groups
        are touched by this function.  The enable status and output
        voltage of all DACs not assigned to a TES bias group are
        maintained.

        Args
        ----
        bias_group_volt_array : float array
            The TES bias to command in voltage for each bipolar TES
            bias group. Should be (n_bias_groups,).
        do_enable : bool, optional, default True
            Set the enable bit for both DACs for every TES bias group.
        """

        n_bias_groups = self._n_bias_groups

        # in this function we're only touching the DACs defined in TES
        # bias groups.  Need to make sure we carry along the setting
        # and enable of any DACs that are being used for something
        # else.
        dac_enable_array = self.get_rtm_slow_dac_enable_array()
        dac_volt_array = self.get_rtm_slow_dac_volt_array()

        if len(bias_group_volt_array) != n_bias_groups:
            self.log("Received the wrong number of biases. Expected " +
                     f"an array of n_bias_groups={n_bias_groups} voltages",
                     self.LOG_ERROR)
        else:
            for bg in np.arange(n_bias_groups):
                bias_order = self.bias_group_to_pair[:,0]
                dac_positives = self.bias_group_to_pair[:,1]
                dac_negatives = self.bias_group_to_pair[:,2]

                bias_group_idx = np.ravel(np.where(bias_order == bg))

                dac_positive = dac_positives[bias_group_idx][0] - 1 # freakin Mitch
                dac_negative = dac_negatives[bias_group_idx][0] - 1 # 1 vs 0 indexing

                volts_pos = bias_group_volt_array[bg] / 2
                volts_neg = - bias_group_volt_array[bg] / 2

                if do_enable:
                    dac_enable_array[dac_positive] = 2
                    dac_enable_array[dac_negative] = 2

                dac_volt_array[dac_positive] = volts_pos
                dac_volt_array[dac_negative] = volts_neg

            if do_enable:
                self.set_rtm_slow_dac_enable_array(dac_enable_array, **kwargs)

            self.set_rtm_slow_dac_volt_array(dac_volt_array, **kwargs)

# set_tes_bias_high_current
def set_tes_bias_high_current(self, bias_group, write_log=False):
        """
        Sets all bias groups to high current mode. Note that the bias group
        number is not the same as the relay number. It also does not matter,
        because Joe's code secretly flips all the relays when you flip one.

        Args
        ----
        bias_group : int
            The bias group(s) to set to high current mode.
        """
        old_relay = self.get_cryo_card_relays()
        old_relay = self.get_cryo_card_relays()  # querey twice to ensure update
        new_relay = np.copy(old_relay)
        if write_log:
            self.log(f'Old relay {bin(old_relay)}')

        n_bias_groups = self._n_bias_groups
        bias_group = np.ravel(np.array(bias_group))
        for bg in bias_group:
            if bg < n_bias_groups:
                r = np.ravel(self._pic_to_bias_group[
                    np.where(self._pic_to_bias_group[:,1]==bg)])[0]
            else:
                r = bg
            new_relay = (1 << r) | new_relay
        if write_log:
            self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay, write_log=write_log)
        self.get_cryo_card_relays()

# set_tes_bias_low_current
def set_tes_bias_low_current(self, bias_group, write_log=False):
        """
        Sets all bias groups to low current mode. Note that the bias group
        number is not the same as the relay number. It also does not matter,
        because Joe's code secretly flips all the relays when you flip one

        Args
        ----
        bias_group : int
            The bias group to set to low current mode.
        """
        old_relay = self.get_cryo_card_relays()
        old_relay = self.get_cryo_card_relays()  # querey twice to ensure update
        new_relay = np.copy(old_relay)

        n_bias_groups = self._n_bias_groups
        bias_group = np.ravel(np.array(bias_group))
        if write_log:
            self.log(f'Old relay {bin(old_relay)}')
        for bg in bias_group:
            if bg < n_bias_groups:
                r = np.ravel(self._pic_to_bias_group[np.where(
                    self._pic_to_bias_group[:,1]==bg)])[0]
            else:
                r = bg
            if old_relay & 1 << r != 0:
                new_relay = new_relay & ~(1 << r)
        if write_log:
            self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay, write_log=write_log)
        self.get_cryo_card_relays()

# set_mode_dc
def set_mode_dc(self, write_log=False):
        """
        Sets flux ramp to DC coupling

        Args
        ----
        write_log : bool, optional, default False
            Whether to write outputs to log.
        """
        # The 16th bit (0 indexed) is the AC/DC coupling
        # self.set_tes_bias_high_current(16)
        r = 16

        old_relay = self.get_cryo_card_relays()
        # query twice to ensure update
        old_relay = self.get_cryo_card_relays()
        self.log(f'Old relay {bin(old_relay)}')

        new_relay = np.copy(old_relay)
        new_relay = (1 << r) | new_relay
        self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay, write_log=write_log)
        self.get_cryo_card_relays()

# set_mode_ac
def set_mode_ac(self, write_log=False):
        """
        Sets flux ramp to AC coupling

        Args
        ----
        write_log : bool, optional, default False
            Whether to write outputs to log.
        """
        # The 16th bit (0 indexed) is the AC/DC coupling
        # self.set_tes_bias_low_current(16)
        old_relay = self.get_cryo_card_relays()
        old_relay = self.get_cryo_card_relays()  # querey twice to ensure update
        new_relay = np.copy(old_relay)

        r = 16
        if old_relay & 1 << r != 0:
            new_relay = new_relay & ~(1 << r)

        self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay)
        self.get_cryo_card_relays()

# flux_ramp_setup ***
def flux_ramp_setup(self, reset_rate_khz, fraction_full_scale, df_range=.1,
            band=2, write_log=False, new_epics_root=None):
        """
        Set flux ramp sawtooth rate and amplitude.

        Flux ramp reset rate must integer divide 2.4MHz. E.g. you
        can't run with a 7kHz flux ramp rate.  If you ask for a flux
        ramp reset rate which doesn't integer divide 2.4MHz, you'll
        get the closest reset rate to your requested rate that integer
        divides 2.4MHz.

        If you are not using the timing system, you can use any flux
        ramp rate which integer divides 2.4MHz.

        If you are using a timing system (i.e. if
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_ramp_start_mode`
        returns 0x1), you may only select from a handful of
        pre-programmed reset rates.  See
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_ramp_rate`
        for more details.

        Args
        ----
        reset_rate_khz : float
           The flux ramp rate to set in kHz.
        fraction_full_scale : float
           The amplitude of the flux ramp as a fraction of the maximum
           possible value.
        df_range : float, optional, default 0.1
           If the difference between the desired fraction full scale
           and the closest achievable fraction full scale exceeds
           this will turn off the flux ramp and raise an exception.
        band : int, optional, default 2
           The band to setup the flux ramp on.
        write_log : bool, optional, default False
           Whether to write output to the log.
        new_epics_root : str or None, optional, default None
           Override the original epics root.  If None, does nothing.

        Raises
        ------
        ValueError
           Raised if either 1) the requested RTM clock rate is too low
           (<2MHz) or 2) the difference between the desired fraction
           full scale and the closest achievable fraction full scale
           exceeds the `df_range` argument.

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
        desFastSlowStepSize = (fullScaleRate * 2**self._num_flux_ramp_counter_bits) / rtmClock
        trialFastSlowStepSize = round(desFastSlowStepSize)
        FastSlowStepSize = trialFastSlowStepSize

        trialFullScaleRate = trialFastSlowStepSize * trialRTMClock / (2**self._num_flux_ramp_counter_bits)

        trialResetRate = (dspClockFrequencyMHz * 1e6) / (rampMaxCnt + 1)
        trialFractionFullScale = trialFullScaleRate / trialResetRate
        fractionFullScale = trialFractionFullScale
        diffDesiredFractionFullScale = np.abs(trialFractionFullScale -
            fraction_full_scale)

        self.log(
            f"Percent full scale = {100 * fractionFullScale:0.3f}%",
            self.LOG_USER)

        if diffDesiredFractionFullScale > df_range:
            raise ValueError(
                "Difference from desired fraction of full scale " +
                f"exceeded! {diffDesiredFractionFullScale} " +
                f"vs acceptable {df_range}.")
            self.log(
                "Difference from desired fraction of full scale " +
                f"exceeded!  P{diffDesiredFractionFullScale} vs " +
                f"acceptable {df_range}.",
                self.LOG_USER)

        if rtmClock < 2e6:
            raise ValueError(
                "RTM clock rate = " +
                f"{rtmClock*1e-6} is too low " +
                "(SPI clock runs at 1MHz)")
            self.log(
                "RTM clock rate = " +
                f"{rtmClock * 1e-6} is too low " +
                "(SPI clock runs at 1MHz)",
                self.LOG_USER)
            return

        FastSlowRstValue = np.floor((2**self._num_flux_ramp_counter_bits) *
            (1 - fractionFullScale)/2)

        KRelay = 3 #where do these values come from
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
        self.set_pulse_width(PulseWidth, new_epics_root=new_epics_root,
            write_log=write_log)
        self.set_debounce_width(DebounceWidth, new_epics_root=new_epics_root,
            write_log=write_log)
        self.set_ramp_slope(RampSlope, new_epics_root=new_epics_root,
            write_log=write_log)
        self.set_mode_control(ModeControl, new_epics_root=new_epics_root,
            write_log=write_log)
        self.set_fast_slow_step_size(FastSlowStepSize,
            new_epics_root=new_epics_root,
            write_log=write_log)
        self.set_fast_slow_rst_value(FastSlowRstValue,
            new_epics_root=new_epics_root,
            write_log=write_log)
        self.set_enable_ramp_trigger(EnableRampTrigger,
            new_epics_root=new_epics_root,
            write_log=write_log)
        # If RampStartMode is 0x1, using timing system, which
        # overrides internal triggering.  Must select one of the
        # available ramp rates programmed into the timing system using
        # the set_ramp_rate routine.
        if self.get_ramp_start_mode() == 1:
            self.set_ramp_rate(
                reset_rate_khz, new_epics_root=new_epics_root,
                write_log=write_log)

# DATA ACQUISITION FUNCTIONS

# set_stream_enable
def set_stream_enable(self, val, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        self._caput(self.app_core + self._stream_enable_reg, val, **kwargs)

# take_stream_data
def take_stream_data(self, meas_time, downsample_factor=None,
                         write_log=True, update_payload_size=True,
                         reset_unwrapper=True, reset_filter=True,
                         return_data=False, make_freq_mask=True,
                         register_file=False):
        """
        Takes streaming data for a given amount of time

        To do: move downsample_factor to config table

        Args
        ----
        meas_time : float
            The amount of time to observe for in seconds.
        downsample_factor : int or None, optional, default None
            The number of fast sample (the flux ramp reset rate -
            typically 4kHz) to skip between reporting. If None, does
            not update.
        write_log : bool, optional, default True
            Whether to write to the log file.
        update_payload_size : bool, optional, default True
            Whether to update the payload size (the number of channels
            written to disk). If the number of channels on is greater
            than the payload size, then only the first N channels are
            written. This bool will update the payload size to be the
            same as the number of channels on across all bands)
        reset_unwrapper : bool, optional, default True
            Whether to reset the unwrapper before taking data.
        reset_filter : bool, optional, default True
            Whether to reset the filter before taking data.
        return_data : bool, optional, default False
            Whether to return the data. If False, returns the full
            path to the data.
        make_freq_mask : bool, optional, default True
            Whether to write a text file with resonator frequencies.
        register_file : bool, optional, default False
            Whether to register the data file with the pysmurf
            publisher.


        Returns
        -------
        data_filename : str
            The fullpath to where the data is stored.
        """
        if write_log:
            self.log('Starting to take data.', self.LOG_USER)
        data_filename = self.stream_data_on(downsample_factor=downsample_factor,
            update_payload_size=update_payload_size, write_log=write_log,
            reset_unwrapper=reset_unwrapper, reset_filter=reset_filter,
            make_freq_mask=make_freq_mask)

        # Sleep for the full measurement time
        time.sleep(meas_time)

        # Stop acq
        self.stream_data_off(write_log=write_log, register_file=register_file)

        if write_log:
            self.log('Done taking data.', self.LOG_USER)

        if return_data:
            t, d, m = self.read_stream_data(data_filename)
            return t, d, m
        else:
            return data_filename

# take_noise_psd
def take_noise_psd(self, meas_time,
                       channel=None, nperseg=2**12,
                       detrend='constant', fs=None,
                       low_freq=None,
                       high_freq=None,
                       make_channel_plot=True,
                       make_summary_plot=True, save_data=False,
                       show_plot=False,
                       grid_on=False, datafile=None,
                       downsample_factor=None,
                       write_log=True, reset_filter=True,
                       reset_unwrapper=True,
                       return_noise_params=False,
                       plotname_append=''):
        """
        Takes a timestream of noise and calculates its PSD. It also
        attempts to fit a white noise and 1/f component to the data.
        It takes into account the sampling frequency and the downsampling
        filter and downsampler.

        Args
        ----
        meas_time : float
            The amount of time to observe in seconds.
        channel : int array or None, optional, default None
            The channels to plot. Note that this script always takes
            data on all the channels. This only sets the ones to plot.
            If None, plots all channels that are on.
        nperseg : int, optional, default 2**12
            The number of elements per segment in the PSD.
        detrend : str, optional, default 'constant'
            Extends the scipy.signal.welch detrend.
        fs : float or None, optional, default None
            Sample frequency. If None, reads it in.
        make_channel_plot : bool, optional, default True
            Whether to make the individual channel plots.
        make_summary_plot : bool, optional, default True
            Whether to make the summary plots.
        save_data : bool, optional, default False
            Whether to save the band averaged data as a text file.
        show_plot : bool, optional, default False
            Show the plot on the screen.
        datafile : str or None, optional, default None
            If data has already been taken, can point to a file to
            bypass data taking and just analyze.
        downsample_factor : int or None, optional, default None
            The datarate is the flux ramp rate divided by the
            downsample_factor.
        write_log : bool, optional, default True
            Whether to write to the log file (or the screen if the
            logfile is not defined).
        reset_filter : bool, optional, default True
            Whether to reset the filter before taking data.
        reset_unwrapper : bool, optional, default True
            Whether to reset the unwrapper before taking data.
        plotname_append : str, optional, default ''
            Appended to the default plot filename.

        Returns
        -------
        datafile : str
             The full path to the raw data.
        """
        if low_freq is None:
            low_freq=np.array([.1, 1.])
        if high_freq is None:
            high_freq=np.array([1., 10.])
        if datafile is None:
            # Take the data if not given as input
            datafile = self.take_stream_data(meas_time,
                downsample_factor=downsample_factor,
                write_log=write_log, reset_unwrapper=reset_unwrapper,
                reset_filter=reset_filter)
        else:
            self.log(f'Reading data from {datafile}')

        basename, _ = os.path.splitext(os.path.basename(datafile))

        # Get downsample filter params
        filter_b = self.get_filter_b()
        filter_a = self.get_filter_a()

        timestamp, phase, mask = self.read_stream_data(datafile)
        bands, channels = np.where(mask!=-1)

        phase *= self._pA_per_phi0/(2.*np.pi) # phase converted to pA

        flux_ramp_freq = self.get_flux_ramp_freq() * 1.0E3  # convert to Hz

        if fs is None:
            if downsample_factor is None:
                downsample_factor = self.get_downsample_factor()
            fs = flux_ramp_freq/downsample_factor

        # Generate downsample transfer function - downsampling is at
        # flux ramp freq
        downsample_freq, downsample_transfer = signal.freqz(filter_b,
            filter_a, worN=np.arange(.01, fs/2, .01), fs=flux_ramp_freq)
        downsample_transfer = np.abs(downsample_transfer)

        if write_log:
            self.log(f'Plotting {bands}, {channels}', self.LOG_USER)

        n_channel = len(channels)

        if make_summary_plot or make_channel_plot:
            plt.rcParams["patch.force_edgecolor"] = True

        noise_floors = np.full((len(low_freq), n_channel), np.nan)
        f_knees = np.full(n_channel,np.nan)
        res_freqs = np.full(n_channel,np.nan)

        wl_list = []
        f_knee_list = []
        n_list = []

        # Live plotting or not
        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        for c, (b, ch) in enumerate(zip(bands, channels)):
            if ch < 0:
                continue
            ch_idx = mask[b, ch]

            # Calculate PSD
            f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg,
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)

            good_fit = False
            try:
                # Fit the PSD
                popt, pcov, f_fit, Pxx_fit = self.analyze_psd(f, Pxx,
                    fs=fs, flux_ramp_freq=flux_ramp_freq)
                wl, n, f_knee = popt
                if f_knee != 0.:
                    wl_list.append(wl)
                    f_knee_list.append(f_knee)
                    f_knees[c]=f_knee
                    n_list.append(n)
                    good_fit = True
                if write_log:
                    self.log(f'{c+1}. b{b}ch{ch:03}:' +
                         f' white-noise level = {wl:.2f}' +
                         f' pA/rtHz, n = {n:.2f}' +
                         f', f_knee = {f_knee:.2f} Hz')
            except Exception as e:
                if write_log:
                    self.log(f'{c+1} b{b}ch{ch:03}: bad fit to noise model')
                    self.log(e)

            # Calculate noise in various frequency bins
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                idx = np.logical_and(f>l, f<h)
                noise_floors[i, c] = np.mean(Pxx[idx])

            if make_channel_plot:
                fig, ax = plt.subplots(2, figsize=(8, 6))

                sampleNums = np.arange(len(phase[ch_idx]))
                t_array = sampleNums / fs

                # Plot the data
                ax[0].plot(t_array,phase[ch_idx] - np.mean(phase[ch_idx]))
                ax[0].set_xlabel('Time [s]')
                ax[0].set_ylabel('Phase [pA]')

                if grid_on:
                    ax[0].grid()

                ax[1].loglog(f, Pxx)
                ylim = ax[1].get_ylim()

                # Plot the fit
                if good_fit:
                    ax[1].plot(f_fit, Pxx_fit, linestyle='--', label=f'n={n:3.2f}')

                    # plot f_knee
                    ax[1].plot(f_knee, 2.*wl, linestyle='none', marker='o',
                        label=r'$f_\mathrm{knee} = ' + f'{f_knee:0.2f},' +
                        r'\mathrm{Hz}$')
                    ax[1].plot(f_fit,wl + np.zeros(len(f_fit)), linestyle=':',
                        label=r'$\mathrm{wl} = $'+ f'{wl:0.2f},' +
                        r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}$')
                    ax[1].plot(downsample_freq, wl*downsample_transfer,
                               color='k', linestyle='dashdot',
                               alpha=.5, label='Lowpass')
                    ax[1].legend(loc='best')
                    ax[1].set_ylim(ylim)

                ax[1].set_xlabel('Frequency [Hz]')
                ax[1].set_xlim(f[1],f[-1])
                ax[1].set_ylabel('Amp [pA/rtHz]')

                if grid_on:
                    ax[1].grid()

                if write_log:
                    self.log(noise_floors[-1, c])

                res_freq = self.channel_to_freq(b, ch)
                res_freqs[c]=res_freq

                ax[0].set_title(f'Band {b} Ch {ch:03} - {res_freq:.2f} MHz')

                fig.tight_layout()

                plot_name = basename + \
                    f'_noise_timestream_b{b}_ch{ch:03}{plotname_append}.png'
                fig.savefig(os.path.join(self.plot_dir, plot_name),
                    bbox_inches='tight')
                self.pub.register_file(os.path.join(self.plot_dir, plot_name),
                                       'noise_timestream', plot=True)

                # Close the individual channel plots - otherwise too many
                # plots are brought to screen
                plt.close(fig)

        if save_data:
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                save_name = basename+f'_{l:3.2f}_{h:3.2f}.txt'
                # outfn = os.path.join(self.plot_dir, save_name)
                outfn = os.path.join(self.output_dir, save_name)

                np.savetxt(outfn, np.c_[res_freqs, noise_floors[i], f_knees])
                # Publish the data
                self.pub.register_file(outfn, 'noise_timestream', format='txt')

                np.save(os.path.join(self.output_dir,
                    f'{basename}_wl_list'), wl_list)
                np.save(os.path.join(self.output_dir,
                    f'{basename}_f_knee_list'), f_knee_list)
                np.save(os.path.join(self.output_dir,
                    f'{basename}_n_list'), n_list)

        if make_summary_plot:
            bins = np.arange(0,351,20)
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                fig, ax = plt.subplots(1, figsize=(10,6))
                ax.hist(noise_floors[i,~np.isnan(noise_floors[i])], bins=bins)
                ax.text(0.03, 0.95, f'{l:3.2f}' + '-' + f'{h:3.2f} Hz',
                        transform=ax.transAxes, fontsize=10)
                ax.set_xlabel(r'Mean noise [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')

                plot_name = (
                    basename +
                    f'{l}_{h}_noise_hist{plotname_append}.png')
                plt.savefig(os.path.join(self.plot_dir, plot_name),
                    bbox_inches='tight')
                self.pub.register_file(
                    os.path.join(self.plot_dir, plot_name),
                    'noise_hist', plot=True
                )
                if show_plot:
                    plt.show()
                else:
                    plt.close()

            if len(wl_list) > 0:
                wl_median = np.median(wl_list)
                n_median = np.median(n_list)
                f_knee_median = np.median(f_knee_list)

                n_fit = len(wl_list)
                n_attempt = len(channels)

                fig,ax = plt.subplots(1,3, figsize=(10,6))
                fig.suptitle(
                    f'{basename} noise parameters' +
                    f' ({n_fit} fit of {n_attempt} attempted)')
                ax[0].hist(wl_list,
                    bins=np.logspace(np.floor(np.log10(np.min(wl_list))),
                        np.ceil(np.log10(np.max(wl_list))), 10))

                ax[0].set_xlabel('White-noise level (pA/rtHz)')
                ax[0].set_xscale('log')
                ax[0].set_title(f'median = {wl_median:.3e} pA/rtHz')
                ax[1].hist(n_list)
                ax[1].set_xlabel('Noise index')
                ax[1].set_title(f'median = {n_median:.3e}')
                ax[2].hist(f_knee_list,
                    bins=np.logspace(np.floor(np.log10(np.min(f_knee_list))),
                        np.ceil(np.log10(np.max(f_knee_list))), 10))
                ax[2].set_xlabel('Knee frequency')
                ax[2].set_xscale('log')
                ax[2].set_title(f'median = {f_knee_median:.3e} Hz')
                plt.tight_layout()
                fig.subplots_adjust(top = 0.9)
                noise_params_hist_fname = basename + \
                    f'_noise_params{plotname_append}.png'
                plt.savefig(os.path.join(self.plot_dir,
                    noise_params_hist_fname),
                    bbox_inches='tight')
                self.pub.register_file(
                    os.path.join(self.plot_dir, noise_params_hist_fname),
                    'noise_params', plot=True
                )

                if show_plot:
                    plt.show()
                else:
                    plt.close()

        if return_noise_params:
            return datafile, (res_freqs, noise_floors, f_knees, wl_list, f_knee_list, n_list)

        else:
            return datafile

# stream_data_on
def stream_data_on(self, write_config=False, data_filename=None,
                       downsample_factor=None, write_log=True,
                       update_payload_size=True, reset_filter=True,
                       reset_unwrapper=True, make_freq_mask=True,
                       channel_mask=None, make_datafile=True,
                       filter_wait_time=0.1):
        """
        Turns on streaming data.

        Args
        ----
        write_config : bool, optional, default False
            Whether to dump the entire config. Warning this can be
            slow.
        data_filename : str or None, optional, default None
            The full path to store the data. If None, it uses the
            timestamp.
        downsample_factor : int or None, optional, default None
            The number of fast samples to skip between sending.
        write_log : bool, optional, default True
            Whether to write to the log file.
        update_payload_size : bool, optional, default True
            Whether to update the payload size (the number of channels
            written to disk). If this is True, will set the payload size to
            0, which tells rogue to automatically adjust it based on the
            channel count.
        reset_filter : bool, optional, default True
            Whether to reset the filter before taking data.
        reset_unwrapper : bool, optional, default True
            Whether to reset the unwrapper before taking data.
        make_freq_mask : bool, optional, default True
            Whether to write a text file with resonator frequencies.
        channel_mask : list or None, optional, default None
            Channel mask to set before streamig data. This should be an array
            of absolute smurf channels between 0 and
            ``nbands * chans_per_band``. If None will create the channel mask
            containing all channels with a non-zero tone amplitude.
        make_datafile : bool, optional, default True
            Whether to create a datafile.
        filter_wait_time : float, optional, default 0.1
            Time in seconds to wait after filter reset.

        Returns
        -------
        data_filename : str
            The fullpath to where the data is stored.
        """
        bands = self._bands

        if downsample_factor is not None:
            self.set_downsample_factor(downsample_factor)
        else:
            downsample_factor = self.get_downsample_factor()
            if write_log:
                self.log('Input downsample factor is None. Using '+
                     'value already in pyrogue:'+
                     f' {downsample_factor}')

        if update_payload_size:
            self.set_payload_size(0)

        # Check if flux ramp is non-zero
        ramp_max_cnt = self.get_ramp_max_cnt()
        if ramp_max_cnt == 0:
            self.log('Flux ramp frequency is zero. Cannot take data.',
                self.LOG_ERROR)
        else:
            # check which flux ramp relay state we're in
            # read_ac_dc_relay_status() should be 0 in DC mode, 3 in
            # AC mode.  this check is only possible if you're using
            # one of the newer C02 cryostat cards.
            flux_ramp_ac_dc_relay_status = self.C.read_ac_dc_relay_status()
            if flux_ramp_ac_dc_relay_status == 0:
                if write_log:
                    self.log("FLUX RAMP IS DC COUPLED.", self.LOG_USER)
            elif flux_ramp_ac_dc_relay_status == 3:
                if write_log:
                    self.log("Flux ramp is AC-coupled.", self.LOG_USER)
            else:
                self.log("flux_ramp_ac_dc_relay_status = " +
                         f"{flux_ramp_ac_dc_relay_status} " +
                         "- NOT A VALID STATE.", self.LOG_ERROR)

            if channel_mask is None:
                # Creates a channel mask with all channels that have enabled
                # tones
                smurf_chans = {}
                for b in bands:
                    smurf_chans[b] = self.which_on(b)

                channel_mask = self.make_channel_mask(bands, smurf_chans)
                self.set_channel_mask(channel_mask)
            else:
                channel_mask = np.atleast_1d(channel_mask)
                self.set_channel_mask(channel_mask)

            time.sleep(0.5)

            # start streaming before opening file
            # to avoid transient filter step
            self.set_stream_enable(1, write_log=False, wait_done=True)

            if reset_unwrapper:
                self.set_unwrapper_reset(write_log=write_log)
            if reset_filter:
                self.set_filter_reset(write_log=write_log)
            if reset_unwrapper or reset_filter:
                time.sleep(filter_wait_time)

            # Make the data file
            timestamp = self.get_timestamp()
            if data_filename is None:
                data_filename = os.path.join(self.output_dir, timestamp+'.dat')

            if make_datafile:
                self.set_data_file_name(data_filename)

            # Optionally write PyRogue configuration
            if write_config:
                config_filename=os.path.join(self.output_dir, timestamp+'.yml')
                if write_log:
                    self.log('Writing PyRogue configuration to file : ' +
                        f'{config_filename}', self.LOG_USER)
                self.write_config(config_filename)

                # short wait
                time.sleep(5.)
            if write_log:
                self.log(f'Writing to file : {data_filename}',
                         self.LOG_USER)


            # Save mask file as text file. Eventually this will be in the
            # raw data output
            mask_fname = os.path.join(data_filename.replace('.dat',
                '_mask.txt'))
            np.savetxt(mask_fname, channel_mask, fmt='%i')
            self.pub.register_file(mask_fname, 'mask')
            self.log(mask_fname)

            if make_freq_mask:
                if write_log:
                    self.log("Writing frequency mask.")
                freq_mask = self.make_freq_mask(channel_mask)
                np.savetxt(os.path.join(data_filename.replace('.dat',
                    '_freq.txt')), freq_mask, fmt='%4.4f')
                self.pub.register_file(
                    os.path.join(data_filename.replace('.dat', '_freq.txt')),
                    'mask', format='txt')

            if make_datafile:
                self.open_data_file(write_log=write_log)

            return data_filename

# stream_data_off
def stream_data_off(self, write_log=True, register_file=False):
        """
        Turns off streaming data.

        Args
        ----
        write_log : bool, optional, default True
            Whether to log the CA commands or not.
        register_file : bool, optional, default False
            If true, the stream data file will be registered through
            the publisher.
        """
        self.close_data_file(write_log=write_log)

        if register_file:
            datafile = self.get_data_file_name().tostring().decode()
            if datafile:
                self.log(f"Registering File {datafile}")
                self.pub.register_file(datafile, 'data', format='dat')

        self.set_stream_enable(0, write_log=write_log, wait_after=.15)

# read_stream_data
def read_stream_data(self, datafile, channel=None,
                         nsamp=None, array_size=None,
                         return_header=False,
                         return_tes_bias=False, write_log=True,
                         n_max=2048, make_freq_mask=False,
                         gcp_mode=False):
        """
        Loads data taken with the function stream_data_on.
        Gives back the resonator data in units of phase. Also
        can optionally return the header (which has things
        like the TES bias).

        Args
        ----
        datafile : str
            The full path to the data to read.
        channel : int or int array or None, optional, default None
            Channels to load.
        nsamp : int or None, optional, default None
            The number of samples to read.
        array_size : int or None, optional, default None
            The size of the output arrays. If 0, then the size will be
            the number of channels in the data file.
        return_header : bool, optional, default False
            Whether to also read in the header and return the header
            data. Returning the full header is slow for large
            files. This overrides return_tes_bias.
        return_tes_bias : bool, optional, default False
            Whether to return the TES bias.
        write_log : bool, optional, default True
            Whether to write outputs to the log file.
        n_max : int, optional, default 2048
            The number of elements to read in before appending the
            datafile. This is just for speed.
        make_freq_mask : bool, optional, default False
            Whether to write a text file with resonator frequencies.
        gcp_mode (bool) : Indicates that the data was written in GCP mode. This
            is the legacy data mode which was depracatetd in Rogue 4.

        Ret:
        ----
        t (float array): The timestamp data
        d (float array): The resonator data in units of radians
        m (int array): The maskfile that maps smurf num to gcp num
        h (dict) : A dictionary with the header information.
        """
        if gcp_mode:
            self.log('Data is in GCP mode.')
            return self.read_stream_data_gcp_save(datafile, channel=channel,
                unwrap=True, downsample=1, nsamp=nsamp)

        # Why were we globbing here?
        #try:
        #    datafile = glob.glob(datafile+'*')[-1]
        #except BaseException:
        #    self.log(f'datafile={datafile}')
        if not os.path.isfile(datafile):
            raise FileNotFoundError(f'datafile={datafile}')

        if write_log:
            self.log(f'Reading {datafile}')

        if channel is not None:
            self.log(f'Only reading channel {channel}')

        # Flag to indicate we are about the read the fist frame from the disk
        # The number of channel will be extracted from the first frame and the
        # data structures will be build based on that
        first_read = True
        with SmurfStreamReader(datafile,
                isRogue=True, metaEnable=True) as file:
            for header, data in file.records():
                if first_read:
                    # Update flag, so that we don't do this code again
                    first_read = False

                    # Read in all used channels by default
                    if channel is None:
                        channel = np.arange(header.number_of_channels)

                    channel = np.ravel(np.asarray(channel))
                    n_chan = len(channel)

                    # Indexes for input channels
                    channel_mask = np.zeros(n_chan, dtype=int)
                    for i, c in enumerate(channel):
                        channel_mask[i] = c

                    #initialize data structure
                    phase=list()
                    for _,_ in enumerate(channel):
                        phase.append(list())
                    for i,_ in enumerate(channel):
                        phase[i].append(data[i])
                    t = [header.timestamp]
                    if return_header or return_tes_bias:
                        tmp_tes_bias = np.array(header.tesBias)
                        tes_bias = np.zeros((0,16))

                    # Get header values if requested
                    if return_header or return_tes_bias:
                        tmp_header_dict = {}
                        header_dict = {}
                        for i, h in enumerate(header._fields):
                            tmp_header_dict[h] = np.array(header[i])
                            header_dict[h] = np.array([],
                                                      dtype=type(header[i]))
                        tmp_header_dict['tes_bias'] = np.array([header.tesBias])


                    # Already loaded 1 element
                    counter = 1
                else:
                    for i in range(n_chan):
                        phase[i].append(data[i])
                    t.append(header.timestamp)

                    if return_header or return_tes_bias:
                        for i, h in enumerate(header._fields):
                            tmp_header_dict[h] = np.append(tmp_header_dict[h],
                                                       header[i])
                        tmp_tes_bias = np.vstack((tmp_tes_bias, header.tesBias))

                    if counter % n_max == n_max - 1:
                        if write_log:
                            self.log(f'{counter+1} elements loaded')

                        if return_header:
                            for k in header_dict.keys():
                                header_dict[k] = np.append(header_dict[k],
                                                           tmp_header_dict[k])
                                tmp_header_dict[k] = \
                                    np.array([],
                                             dtype=type(header_dict[k][0]))
                            print(np.shape(tes_bias), np.shape(tmp_tes_bias))
                            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
                            tmp_tes_bias = np.zeros((0, 16))

                        elif return_tes_bias:
                            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
                            tmp_tes_bias = np.zeros((0, 16))

                    counter += 1

        phase=np.array(phase)
        t=np.array(t)

        if return_header:
            for k in header_dict.keys():
                header_dict[k] = np.append(header_dict[k],
                    tmp_header_dict[k])
            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
            tes_bias = np.transpose(tes_bias)

        elif return_tes_bias:
            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
            tes_bias = np.transpose(tes_bias)

        # rotate and transform to phase
        phase = phase.astype(float) / 2**15 * np.pi

        if np.size(phase) == 0:
            self.log("Only 1 element in datafile. This is often an indication" +
                "that the data was taken in GCP mode. Try running this"+
                " function again with gcp_mode=True")

        # make a mask from mask file
        #  regexp pattern to match any filename which ends in a
        #  . followed by a number, as occurs when MaxFileSize is
        #  nonzero and rogue rolls over files by appending an
        #  increasing number at the end after a .
        extpattern=re.compile('(.+?).dat.([0-9]|[1-9][0-9]+)$')
        extmatch=extpattern.match(datafile)
        if ".dat.part" in datafile:
            mask = self.make_mask_lookup(datafile.split(".dat.part")[0] +
                "_mask.txt")
        elif extmatch is not None:
            mask = self.make_mask_lookup(extmatch[1]+"_mask.txt")
        else:
            mask = self.make_mask_lookup(datafile.replace('.dat', '_mask.txt'),
                                         make_freq_mask=make_freq_mask)

        # If an array_size was defined, resize the phase array
        if array_size is not None:
            phase.resize(array_size, phase.shape[1])

        if return_header:
            header_dict['tes_bias'] = tes_bias
            return t, phase, mask, header_dict
        elif return_tes_bias:
            return t, phase, mask, tes_bias
        else:
            return t, phase, mask

# set_downsample_filter
def set_downsample_filter(self, filter_order, cutoff_freq, write_log=False):
        """
        Sets the downsample filter. This is anti-alias filter
        that filters data at the flux_ramp reset rate, which is
        before the downsampler.

        Args
        ----
        filter_order : int
            The number of poles in the filter.
        cutoff_freq : float
            The filter cutoff frequency.
        """
        # Get flux ramp frequency
        flux_ramp_freq = self.get_flux_ramp_freq()*1.0E3

        # Get filter parameters
        b, a = signal.butter(filter_order,
                             2*cutoff_freq/flux_ramp_freq)

        # Set filter parameters
        self.set_filter_order(filter_order, write_log=write_log)
        self.set_filter_a(a, write_log=write_log)
        self.set_filter_b(b, write_log=write_log, wait_done=True)

        self.set_filter_reset(wait_after=.1, write_log=write_log)

# IV FUNCTIONS

# run_iv
def run_iv(self, bias_groups=None, wait_time=.1, bias=None,
               bias_high=1.5, bias_low=0, bias_step=.005,
               show_plot=False, overbias_wait=2., cool_wait=30,
               make_plot=True, save_plot=True, plotname_append='',
               channels=None, band=None, high_current_mode=True,
               overbias_voltage=8., grid_on=True,
               phase_excursion_min=3., bias_line_resistance=None,
               do_analysis=True):
        """Takes a slow IV

        Steps the TES bias down slowly. Starts at bias_high to
        bias_low with step size bias_step. Waits wait_time between
        changing steps. If this analyzes the data, the outputs are
        stored to output_dir.

        Args
        ----
        bias_groups : numpy.ndarray or None, optional, default None
            Which bias groups to take the IV on. If None, defaults to
            the groups in the config file.
        wait_time : float, optional, default 0.1
            The amount of time between changing TES biases in seconds.
        bias : float array or None, optional, default None
            A float array of bias values. Must go high to low.
        bias_high : float, optional, default 1.5
            The maximum TES bias in volts.
        bias_low : float, optional, default 0
            The minimum TES bias in volts.
        bias_step : float, optional, default 0.005
            The step size in volts.
        show_plot : bool, optional, default False
            Whether to show plots.
        overbias_wait : float, optional, default 2.0
            The time to stay in the overbiased state in seconds.
        cool_wait : float, optional, default 30.0
            The time to stay in the low current state after
            overbiasing before taking the IV.
        make_plot : bool, optional, default True
            Whether to make plots.
        save_plot : bool, optional, default True
            Whether to save the plot.
        plotname_append : str, optional, default ''
            Appended to the default plot filename.
        channels : int array or None, optional, default None
            A list of channels to make plots.
        band : int array or None, optional, default None
            The bands to analyze.
        high_current_mode : bool, optional, default True
            The current mode to take the IV in.
        overbias_voltage : float, optional, default 8.0
            The voltage to set the TES bias in the overbias stage.
        grid_on : bool, optional, default True
            Grids on plotting. This is Ari's fault.
        phase_excursion_min : float, optional, default 3.0
            The minimum phase excursion required for making plots.
        bias_line_resistance : float or None, optional, default None
            The resistance of the bias lines in Ohms. If None, loads value
            in config file
        do_analysis: bool, optional, default True
            Whether to do the pysmurf IV analysis

        Returns
        -------
        output_path : str
            Full path to IV analyzed file.
        """

        n_bias_groups = self._n_bias_groups

        if bias_groups is None:
            bias_groups = self._all_groups
        bias_groups = np.array(bias_groups)

        if overbias_voltage != 0.:
            overbias = True
        else:
            overbias = False

        if bias is None:
            # Set actual bias levels
            bias = np.arange(bias_high, bias_low-bias_step, -bias_step)

        # Overbias the TESs to drive them normal
        if overbias:
            self.overbias_tes_all(bias_groups=bias_groups,
                overbias_wait=overbias_wait, tes_bias=np.max(bias),
                cool_wait=cool_wait, high_current_mode=high_current_mode,
                overbias_voltage=overbias_voltage)

        self.log('Starting to take IV.', self.LOG_USER)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        bias_group_bool = np.zeros((n_bias_groups,))
        bias_group_bool[bias_groups] = 1 # only set things on the bias groups that are on

        self.set_tes_bias_bipolar_array(bias[0] * bias_group_bool)
        time.sleep(wait_time) # loops are in pyrogue now, which are faster?

        datafile = self.stream_data_on()
        self.log(f'writing to {datafile}')

        for b in bias:
            self.log(f'Bias at {b:4.3f}')
            self.set_tes_bias_bipolar_array(b * bias_group_bool)
            time.sleep(wait_time) # loops are now in pyrogue, so no division

        self.stream_data_off(register_file=True)
        self.log('Done with TES bias ramp', self.LOG_USER)


        basename, _ = os.path.splitext(os.path.basename(datafile))
        path = os.path.join(self.output_dir, basename + '_iv_bias_all')
        np.save(path, bias)

        # publisher announcement
        self.pub.register_file(path, 'iv_bias', format='npy')

        iv_raw_data = {}
        iv_raw_data['bias'] = bias
        iv_raw_data['high_current_mode'] = high_current_mode
        iv_raw_data['bias group'] = bias_groups
        iv_raw_data['datafile'] = datafile
        iv_raw_data['basename'] = basename
        iv_raw_data['output_dir'] = self.output_dir
        iv_raw_data['plot_dir'] = self.plot_dir
        fn_iv_raw_data = os.path.join(self.output_dir, basename +
            '_iv_raw_data.npy')
        self.log(f'Writing IV metadata to {fn_iv_raw_data}.')

        path = os.path.join(self.output_dir, fn_iv_raw_data)
        np.save(path, iv_raw_data)
        self.pub.register_file(path, 'iv_raw', format='npy')

        R_sh=self._R_sh

        if do_analysis:
            self.log(f'Analyzing IV (do_analysis={do_analysis}).')

            self.analyze_iv_from_file(fn_iv_raw_data, make_plot=make_plot,
                show_plot=show_plot, save_plot=save_plot,
                plotname_append=plotname_append, R_sh=R_sh, grid_on=grid_on,
                phase_excursion_min=phase_excursion_min, channel=channels,
                band=band, bias_line_resistance=bias_line_resistance)

        return path

# analyze_iv
def analyze_iv(self, v_bias, resp, make_plot=True, show_plot=False,
            save_plot=True, basename=None, band=None, channel=None, R_sh=None,
            plot_dir=None, high_current_mode=False, bias_group=None,
            grid_on=False, R_op_target=0.007, pA_per_phi0=None,
            bias_line_resistance=None, plotname_append='', **kwargs):
        """
        Analyzes the IV curve taken with run_iv()

        Args
        ----
        v_bias : float array
            The commanded bias in voltage. Length n_steps.
        resp : float array
            The TES phase response in radians. Of length n_pts (not
            the same as n_steps).
        make_plot : bool, optional, default True
            Whether to make the plot.
        show_plot : bool, optional, default False
            Whether to show the plot.
        save_plot : bool, optional, default True
            Whether to save the plot.
        basename : str or None, optional, default None
            The basename of the IV plot. If None, uses the current
            timestamp.
        band : int or None, optional, default None
            The 500 MHz band the data was taken in. This is only for
            plotting.
        channel : int or None, optional, default None
            The SMuRF channel. Only used for plotting.
        R_sh : float or None, optional, default None
            The shunt resistance in ohms. If not supplied, will try to
            read from config file.
        plot_dir : str or None, optional, default None
            Path to the plot directory where plots are to be saved.
            If None, uses self.plot_dir.
        high_current_mode : bool, optional, default False
            Whether the data was taken in high current mode. This is
            important for knowing what current actually enters the
            cryostat.
        grid_on : bool, optional, default False
            Whether to plot with grids on.
        pA_per_phi0 : float or None, optional, default None
            The conversion for phi0 to pA. If None, attempts to read
            it from the config file.
        bias_line_resistance : float or None, optional, default None
            The resistance of the bias lines in Ohms. If None, reads
            from config.
        plotname_append : str, optional, default ''
            An optional string to append the plot names.

        Returns
        -------
        R_sh : float
            Shunt resistance.
        """
        v_bias = np.abs(v_bias)

        if R_sh is None:
            R_sh = self._R_sh

        if pA_per_phi0 is None:
            pA_per_phi0 = self._pA_per_phi0
        resp *= pA_per_phi0/(2.*np.pi*1e6) # convert phase to uA

        step_loc = np.where(np.diff(v_bias))[0]

        if step_loc[0] != 0:
            step_loc = np.append([0], step_loc)  # starts from zero
        n_step = len(step_loc) - 1

        # arrays for holding response, I, and V
        resp_bin = np.zeros(n_step)
        v_bias_bin = np.zeros(n_step)
        i_bias_bin = np.zeros(n_step)

        # load bias line resistance for voltage/current conversion
        if bias_line_resistance is None:
            r_inline = self._bias_line_resistance
        else:
            r_inline = bias_line_resistance

        if high_current_mode:
            # high-current mode generates higher current by decreases the
            # in-line resistance
            r_inline /= self._high_low_current_ratio
        i_bias = 1.0E6 * v_bias / r_inline

        if make_plot:
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

        # Find steps and then calculate the TES values in bins
        for i in np.arange(n_step):
            s = step_loc[i]
            e = step_loc[i+1]

            st = e - s
            sb = int(s + np.floor(st/2))
            eb = int(e - np.floor(st/10))

            resp_bin[i] = np.mean(resp[sb:eb])
            v_bias_bin[i] = v_bias[sb]
            i_bias_bin[i] = i_bias[sb]

        d_resp = np.diff(resp_bin)
        d_resp = d_resp[::-1]
        dd_resp = np.diff(d_resp)
        v_bias_bin = v_bias_bin[::-1]
        i_bias_bin = i_bias_bin[::-1]
        resp_bin = resp_bin[::-1]

        # index of the end of the superconducting branch
        dd_resp_abs = np.abs(dd_resp)
        sc_idx = np.ravel(np.where(dd_resp_abs == np.max(dd_resp_abs)))[0] + 1
        if sc_idx == 0:
            sc_idx = 1

        # index of the start of the normal branch
        nb_idx_default = int(0.8*n_step) # default to partway from beginning of IV curve
        nb_idx = nb_idx_default
        for i in np.arange(nb_idx_default, sc_idx, -1):
            # look for minimum of IV curve outside of superconducting region
            # but get the sign right by looking at the sc branch
            if d_resp[i]*np.mean(d_resp[:sc_idx]) < 0.:
                nb_idx = i+1
                break

        nb_fit_idx = int(np.mean((n_step,nb_idx)))
        norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:], resp_bin[nb_fit_idx:], 1)
        if norm_fit[0] < 0:  # Check for flipped polarity
            resp_bin = -1 * resp_bin
            norm_fit = np.polyfit(i_bias_bin[nb_fit_idx:], resp_bin[nb_fit_idx:], 1)

        resp_bin -= norm_fit[1]  # now in real current units

        sc_fit = np.polyfit(i_bias_bin[:sc_idx], resp_bin[:sc_idx], 1)

        # subtract off unphysical y-offset in superconducting branch; this is
        # probably due to an undetected phase wrap at the kink between the
        # superconducting branch and the transition, so it is *probably*
        # legitimate to remove it by hand. We don't use the offset of the
        # superconducting branch for anything meaningful anyway. This will just
        # make our plots look nicer.
        resp_bin[:sc_idx] -= sc_fit[1]
        sc_fit[1] = 0 # now change s.c. fit offset to 0 for plotting

        R = R_sh * (i_bias_bin/(resp_bin) - 1)
        R_n = np.mean(R[nb_fit_idx:])
        R_L = np.mean(R[1:sc_idx])

        v_tes = i_bias_bin*R_sh*R/(R+R_sh) # voltage over TES
        p_tes = (v_tes**2)/R # electrical power on TES

        R_trans_min = R[sc_idx]
        R_trans_max = R[nb_idx]
        R_frac_min = R_trans_min/R_n
        R_frac_max = R_trans_max/R_n

        i_R_op = 0
        for i in range(len(R)-1,-1,-1):
            if R[i] < R_op_target:
                i_R_op = i
                break
        i_op_target = i_bias_bin[i_R_op]
        v_bias_target = v_bias_bin[i_R_op]
        v_tes_target = v_tes[i_R_op]
        p_trans_median = np.median(p_tes[sc_idx:nb_idx])

        i_tes = resp_bin
        smooth_dist = 5
        w_len = 2*smooth_dist + 1

        # Running average
        w = (1./float(w_len))*np.ones(w_len) # window
        i_tes_smooth = np.convolve(i_tes, w, mode='same')
        v_tes_smooth = np.convolve(v_tes, w, mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth

        # Take derivatives
        di_tes = np.diff(i_tes_smooth)
        dv_tes = np.diff(v_tes_smooth)
        R_L_smooth = np.ones(len(r_tes_smooth))*R_L
        R_L_smooth[:sc_idx] = dv_tes[:sc_idx]/di_tes[:sc_idx]
        r_tes_smooth_noStray = r_tes_smooth - R_L_smooth
        i0 = i_tes_smooth[:-1]
        r0 = r_tes_smooth_noStray[:-1]
        rL = R_L_smooth[:-1]
        si_etf = -1./(i0*r0)
        beta = 0.

        # Responsivity estimate
        si = -(1./i0)*( dv_tes/di_tes - (r0+rL+beta*r0) ) / \
            ( (2.*r0-rL+beta*r0)*dv_tes/di_tes - 3.*rL*r0 - rL**2 )

        if i_R_op == len(si):
            i_R_op -= 1
        si_target = si[i_R_op]

        if make_plot:
            colors = []
            tableau = Colors.TABLEAU_COLORS
            for c in tableau:
                colors.append(tableau[c])

            fig = plt.figure(figsize = (10,6))
            gs = GridSpec(3,3)
            ax_ii = fig.add_subplot(gs[0,:2])
            ax_ri = fig.add_subplot(gs[1,:2])
            ax_pr = fig.add_subplot(gs[1,2])
            ax_si = fig.add_subplot(gs[2,:2])
            ax_i = [ax_ii,ax_ri,ax_si] # axes with I_b as x-axis

            # Construct title
            title = ""
            plot_name = "IV_curve"
            if band is not None:
                title = title + f'Band {band}'
                plot_name = plot_name + f'_b{band}'
            if bias_group is not None:
                title = title + f' BG {bias_group}'
                plot_name = plot_name + f'_bg{bias_group}'
            if channel is not None:
                title = title + f' Ch {channel:03}'
                plot_name = plot_name + f'_ch{channel:03}'
            if basename is None:
                basename = self.get_timestamp()
            if band is not None and channel is not None:
                if self.offline:
                    self.log(
                        "Offline mode does not know resonator " +
                        "frequency.  Not adding to title.")
                else:
                    channel_freq=self.channel_to_freq(band, channel)
                    title += f', {channel_freq:.2f} MHz'
            title += (r', $R_\mathrm{sh}$ = ' + f'${R_sh*1.0E3:.2f}$ ' +
                      r'$\mathrm{m}\Omega$')
            plot_name = basename + '_' + plot_name
            title = basename + ' ' + title
            plot_name += plotname_append + '.png'

            fig.suptitle(title)

            color_meas = colors[0]
            color_norm = colors[1]
            color_target = colors[2]
            color_sc = colors[6]
            color_etf = colors[3]

            ax_ii.axhline(0.,color='grey',linestyle=':')
            ax_ii.plot(i_bias_bin, resp_bin, color=color_meas)
            ax_ii.set_ylabel(r'$I_\mathrm{TES}$ $[\mu A]$')

            # Plot normal branch fit
            ax_ii.plot(i_bias_bin, norm_fit[0] * i_bias_bin ,
                       linestyle='--',
                       color=color_norm, label=r'$R_N$' +
                       f'  = ${R_n*1000.:.0f}$' +
                       r' $\mathrm{m}\Omega$')
            # Plot superconducting branch fit
            ax_ii.plot(
                i_bias_bin[:sc_idx],
                sc_fit[0] * i_bias_bin[:sc_idx] + sc_fit[1],
                linestyle='--',
                color=color_sc,
                label=(
                    r'$R_L$' +
                    f' = ${R_L/1e-6:.0f}$' +
                    r' $\mu\mathrm{\Omega}$'))

            label_target = (
                r'$R = ' + f'{R_op_target/1e-3:.0f}' + r'$ ' +
                r'$\mathrm{m}\Omega$')
            label_rfrac = (f'{R_frac_min:.2f}-{R_frac_max:.2f}' +
                           r'$R_N$')

            for i in range(len(ax_i)):
                if ax_i[i] == ax_ri:
                    label_vline = label_target
                    label_vspan = label_rfrac
                else:
                    label_vline = None
                    label_vspan = None
                ax_i[i].axvline(i_op_target, color='g', linestyle='--',
                    label=label_vline)
                ax_i[i].axvspan(i_bias_bin[sc_idx], i_bias_bin[nb_idx],
                    color=color_etf, alpha=.15,label=label_vspan)
                if grid_on:
                    ax_i[i].grid()
                ax_i[i].set_xlim(min(i_bias_bin), max(i_bias_bin))
                if i != len(ax_i)-1:
                    ax_i[i].set_xticklabels([])
            ax_si.axhline(0., color=color_norm, linestyle='--')

            ax_ii.legend(loc='best')
            ax_ri.legend(loc='best')
            ax_ri.plot(i_bias_bin, R/R_n, color=color_meas)
            ax_pr.plot(p_tes,R/R_n, color=color_meas)
            for ax in [ax_ri, ax_pr]:
                ax.axhline(1, color=color_norm, linestyle='--')
            ax_ri.set_ylabel(r'$R/R_N$')
            ax_i[-1].set_xlabel(r'$I_{b}$ [$\mu\mathrm{A}$]')

            r_min = 0.
            r_max = 1.1
            ax_ri.set_ylim(r_min,r_max)
            ax_pr.set_ylim(r_min,r_max)
            ax_pr.set_yticklabels([])

            # Make top label in volts
            axt = ax_i[0].twiny()
            axt.set_xlim(ax_i[0].get_xlim())
            ib_max = np.max(i_bias_bin)
            ib_min = np.min(i_bias_bin)
            n_ticks = 5
            delta = float(ib_max - ib_min)/n_ticks
            vb_max = np.max(v_bias)
            vb_min = np.min(v_bias)
            delta_v = float(vb_max - vb_min)/n_ticks
            xticks = np.arange(ib_min, ib_max+delta, delta)[:n_ticks+1]
            xticklabels = np.arange(vb_min, vb_max+delta_v, delta_v)[:n_ticks+1]

            axt.set_xticks(xticks)
            axt.set_xticklabels(
                [f'{x:.2f}' for x in xticklabels])
            axt.set_xlabel(r'Commanded $V_b$ [V]')

            ax_si.plot(i_bias_bin[:-1],si,color=color_meas)
            ax_si.plot(i_bias_bin[:-1],si_etf,linestyle = '--',
                label=r'$-1/V_\mathrm{TES}$',color=color_etf)
            ax_si.set_ylabel(r'$S_I$ [$\mu\mathrm{V}^{-1}$]')
            ax_si.set_ylim(-2./v_tes_target,2./v_tes_target)
            ax_si.legend(loc='upper right')

            ax_pr.set_xlabel(r'$P_\mathrm{TES}$ [pW]')
            ax_pr.set_xscale('log')
            ax_pr.axhspan(R_trans_min/R_n,R_trans_max/R_n,color=color_etf,
                alpha=.15)
            label_pr = f'{p_trans_median:.1f} pW'
            ax_pr.axvline(p_trans_median, linestyle='--', label=label_pr,
                color=color_etf)
            ax_pr.plot(p_tes[i_R_op],R[i_R_op]/R_n,'o',color=color_target,
                label=label_target)
            ax_pr.legend(loc='best')
            if grid_on:
                ax_pr.grid()

            fig.subplots_adjust(top=0.875)

            if save_plot:
                if plot_dir is None:
                    plot_dir = self.plot_dir
                plot_filename = os.path.join(plot_dir, plot_name)
                self.log(f'Saving IV plot to {plot_filename}')
                plt.savefig(plot_filename,bbox_inches='tight')
                self.pub.register_file(plot_filename, 'iv', plot=True)

            if show_plot:
                plt.show()
            else:
                plt.close()

        iv_dict = {}
        iv_dict['R'] = R
        iv_dict['R_n'] = R_n
        iv_dict['trans idxs'] = np.array([sc_idx,nb_idx])
        iv_dict['p_tes'] = p_tes
        iv_dict['p_trans'] = p_trans_median
        iv_dict['v_bias_target'] = v_bias_target
        iv_dict['si'] = si
        iv_dict['v_bias'] = v_bias_bin
        iv_dict['si_target'] = si_target
        iv_dict['v_tes_target'] = v_tes_target
        iv_dict['v_tes'] = v_tes

        return iv_dict

# DATA OUTPUTS TO DISK

# tune files generated when new resonators are found
# channel mapping file format
# .dat noise files - generated by take_stream_data
# iv_files - generated by run_iv"