import numpy as np
from scipy import signal
from scipy import optimize
from pysmurf.base import SmurfBase
import os
import time

class SmurfNoiseMixin(SmurfBase):

    def take_noise_psd(self, band, meas_time, channel=None, nperseg=2**12, 
        detrend='constant', fs=None, low_freq=np.array([.1, 1.]), 
        high_freq=np.array([1., 10.]), make_channel_plot=True,
        make_summary_plot=True, save_data=False, show_plot=False,
        grid_on=False, gcp_mode=True, datafile=None):
        """
        Takes a timestream of noise and calculates its PSD.

        Args:
        -----
        band (int): The band to take noise data on
        meas_time (float): The amount of time to observe in seconds.

        Opt Args:
        ---------
        channel (int array): The channels to plot. Note that this script always
            takes data on all the channels. This only sets the ones to plot.
            If None, plots all channels that are on. Default is None.
        nperseg (int): The number of elements per segment in the PSD. Default
            2**12.
        detrend (str): Extends the scipy.signal.welch detrend. Default is 
            'constant'
        fs (float): Sample frequency. If None, reads it in. Default is None.
        low_freq (float array):
        high_freq (float array):
        make_channel_plot (bool): Whether to make the individual channel
            plots. Default is True.
        make_summary_plot (bool): Whether to make the summary plots. Default
            is True.
        save_data (bool): Whether to save the band averaged data as a text file.
            Default is False.
        show_plot (bool): Show the plot on the screen. Default False.
        datefile (str): if data has already been taken, can point to a file to 
            bypass data taking and just analyze.
        """
        if channel is None:
            channel = self.which_on(band)
        n_channel = self.get_number_channels(band)

        if datafile == None:
            datafile = self.take_stream_data(meas_time, gcp_mode=gcp_mode)
        else:
            self.log('Reading data from %s' % (datafile))

        basename, _ = os.path.splitext(os.path.basename(datafile))

        # timestamp, I, Q = self.read_stream_data(datafile)
        timestamp, phase, mask = self.read_stream_data_gcp_save(datafile)
        phase *= self.pA_per_phi0/(2.*np.pi) # phase converted to pA

        if fs is None:
            self.log('No flux ramp freq given. Loading current flux ramp'+
                'frequency', self.LOG_USER)
            fs = self.get_flux_ramp_freq()*1.0E3

        self.log('Plotting channels {}'.format(channel), self.LOG_USER)

        if make_summary_plot or make_channel_plot:
            import matplotlib.pyplot as plt
            plt.rcParams["patch.force_edgecolor"] = True

        noise_floors = np.zeros((len(low_freq), n_channel))*np.nan
        wl_list = []
        f_knee_list = []
        n_list = []

        plt.ion()
        if not show_plot:
            plt.ioff()
        for c, ch in enumerate(channel):
            # phase = self.iq_to_phase(I[ch], Q[ch])

            # Calculate to power spectrum
            # ch_idx = 512*band + ch
            # ch_idx = np.where(mask == 512*band + ch)[0][0]
            ch_idx = mask[band, ch]
            f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg, 
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)

            good_fit = False
            try:
                popt,pcov,f_fit,Pxx_fit = self.analyze_psd(f,Pxx)
                wl,n,f_knee = popt
                if f_knee != 0.:
                    wl_list.append(wl)
                    f_knee_list.append(f_knee)
                    n_list.append(n)
                    good_fit = True    
                self.log('%i. Band %i, ch. %i:' % (c+1,band,ch) + ' white-noise level = {:.2f}'.format(wl) +
                        ' pA/rtHz, n = {:.2f}'.format(n) + 
                        ', f_knee = {:.2f} Hz'.format(f_knee))
            except:
                self.log('Band %i, ch. %i: bad fit to noise model' % (band,ch))

            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                idx = np.logical_and(f>l, f<h)
                noise_floors[i, ch] = np.mean(Pxx[idx])

            if make_channel_plot:
                fig, ax = plt.subplots(2, figsize=(8,6))

                sampleNums = np.arange(len(phase[ch_idx]))
                t_array = sampleNums/fs

                ax[0].plot(t_array,phase[ch_idx] - np.mean(phase[ch_idx]))
                ax[0].set_xlabel('Time [s]')
                ax[0].set_ylabel('Phase [pA]')
                if grid_on:
                    ax[0].grid()

                ax[1].plot(f, Pxx)
                if good_fit:
                    ax[1].plot(f_fit,Pxx_fit,linestyle = '--',label=r'$n=%.2f$' % (n))
                    ax[1].plot(f_knee,2.*wl,linestyle = 'none',marker = 'o',label=r'$f_\mathrm{knee} = %.2f\,\mathrm{Hz}$' % (f_knee))
                    ax[1].plot(f_fit,wl + np.zeros(len(f_fit)),linestyle = ':',label=r'$\mathrm{wl} = %.0f\,\mathrm{pA}/\sqrt{\mathrm{Hz}}$' % (wl))
                    ax[1].legend(loc='best')
                ax[1].set_xlabel('Freq [Hz]')
                ax[1].set_xlim(f[1],f[-1])
                ax[1].set_ylabel('Amp [pA/rtHz]')
                ax[1].set_yscale('log')
                ax[1].set_xscale('log')
                if grid_on:
                    ax[1].grid()

                self.log(noise_floors[-1, ch])
                
                res_freq = self.channel_to_freq(band, ch)
                ax[0].set_title('Band {} Ch {:03} - {:.2f} MHz'.format(band, ch, res_freq))

                plt.tight_layout()

                plot_name = basename+'_b{}_ch{:03}.png'.format(band, ch)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
                if show_plot:
                    plt.show()
                else:
                    plt.close()

        if save_data:
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                save_name = basename+'_{:3.2f}_{:3.2f}.txt'.format(l, h)
                np.savetxt(os.path.join(self.plot_dir, save_name), 
                    noise_floors[i])

        if make_summary_plot:
            bins = np.arange(0,351,20)
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                fig, ax = plt.subplots(1, figsize=(10,6))
                ax.hist(noise_floors[i,~np.isnan(noise_floors[i])], bins=bins)
                ax.text(0.03, 0.95, '{:3.2f}'.format(l) + '-' +
                    '{:3.2f} Hz'.format(h),
                    transform=ax.transAxes, fontsize=10)
                ax.set_xlabel(r'Mean noise [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')

                plot_name = basename + \
                    '_b{}_{}_{}_noise_hist.png'.format(band, l, h)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
                if show_plot:
                    plt.show()
                else:
                    plt.close()

            if len(wl_list) > 0:
                wl_median = np.median(wl_list)
                n_median = np.median(n_list)
                f_knee_median = np.median(f_knee_list)

                n_fit = len(wl_list)
                n_attempt = len(channel)

                fig,ax = plt.subplots(1,3,figsize=(10,6))
                fig.suptitle('{}: band {} noise parameters'.format(basename, band) + 
                    ' ({} fit of {} attempted)'.format(n_fit, n_attempt))
                ax[0].hist(wl_list,bins=np.logspace(np.floor(np.log10(np.min(wl_list))),
                        np.ceil(np.log10(np.max(wl_list))), 10))

                ax[0].set_xlabel('White-noise level (pA/rtHz)')
                ax[0].set_xscale('log')
                ax[0].set_title('median = %.3e pA/rtHz' % (wl_median))
                ax[1].hist(n_list)
                ax[1].set_xlabel('Noise index')
                ax[1].set_title('median = %.3e' % (n_median))
                ax[2].hist(f_knee_list,
                    bins=np.logspace(np.floor(np.log10(np.min(f_knee_list))),
                        np.ceil(np.log10(np.max(f_knee_list))), 10))
                ax[2].set_xlabel('Knee frequency')
                ax[2].set_xscale('log')
                ax[2].set_title('median = %.3e Hz' % (f_knee_median))
                plt.tight_layout()
                fig.subplots_adjust(top = 0.9)
                noise_params_hist_fname = basename + \
                    '_b{}_noise_params.png'.format(band)
                plt.savefig(os.path.join(self.plot_dir,noise_params_hist_fname),
                    bbox_inches='tight')
                plt.show()

        return noise_floors

    def turn_off_noisy_channels(self, band, noise, cutoff=150):
        """
        Args:
        -----
        band (int): The band to search
        noise (float array): The noise floors. Length 512. Presumably calculated
            using take_noise_psd

        Optional Args:
        --------------
        cutoff (float) : The value to cut at in the same units as noise.
        """
        n_channel = self.get_number_channels(band)
        for ch in np.arange(n_channel):
            if noise[ch] > cutoff:
                self.channel_off(band, ch)


    def noise_vs_bias(self, band, bias_group,bias_high=6, bias_low=3, step_size=.1,
        bias=None, high_current_mode=False,
        meas_time=30., analyze=False, channel=None, nperseg=2**13,
        detrend='constant', fs=None,show_plot = False,cool_wait = 30.,gcp_mode = True,
        psd_ylim = (10.,1000.)):
        """
        This ramps the TES voltage from bias_high to bias_low and takes noise
        measurements. You can make it analyze the data and make plots with the
        optional argument analyze=True. Note that the analysis is a little
        slow.

        Args:
        -----
        band (int): The band to take noise vs bias data on
        bias_group (int or int array): which bias group(s) to bias/read back.

        Opt Args:
        ---------
        bias (float array): The array of bias values to step through. If None,
            uses values in defined by bias_high, bias_low, and step_size.
        bias_high (float): The bias voltage to start at
        bias_low (float): The bias votlage to end at
        step_size (float): The step in voltage.
        meas_time (float): The amount of time to take data at each TES bias.
        analyze (bool): Whether to analyze the data
        channel (int): The channel to run analysis on. Note that data is taken
            on all channels. This only affects what is analyzed. You can always
            run the analyze script later.
        nperseg (int): The number of samples per segment in the PSD.
        detrend (str): Whether to detrend the data before taking the PSD.
            Default is to remove a constant.
        fs (float): The sample frequency.
        show_plot: Whether to show analysis plots. Defaults to False.
        """
        if bias is None:
            if step_size > 0:
                step_size *= -1
            bias = np.arange(bias_high, bias_low-np.absolute(step_size), step_size)

        self.noise_vs(band=band,bias_group=bias_group,var='bias',var_range=bias,
                 meas_time=meas_time, analyze=analyze, channel=channel, nperseg=nperseg,
                 detrend=detrend, fs=fs, show_plot=show_plot,
                 gcp_mode=gcp_mode, psd_ylim=psd_ylim,
                 cool_wait=cool_wait, high_current_mode=high_current_mode)

    def noise_vs_amplitude(self, band, amplitude_high=11, amplitude_low=9, step_size=1,
                           amplitudes=None,
                           meas_time=30., analyze=False, channel=None, nperseg=2**13,
                           detrend='constant', fs=None, show_plot = False,
                           gcp_mode = True,
                           psd_ylim = None):
        """
        Args:
        -----
        band (int): The band to take noise vs bias data on
        """
        if amplitudes is None:
            if step_size > 0:
                step_size *= -1
            amplitudes = np.arange(amplitude_high, amplitude_low-np.absolute(step_size), step_size)

        self.noise_vs(band=band,var='amplitude',var_range=amplitudes,
                 meas_time=meas_time, analyze=analyze, channel=channel, nperseg=nperseg,
                 detrend=detrend, fs=fs, show_plot=show_plot,
                 gcp_mode=gcp_mode, psd_ylim=psd_ylim)

    def noise_vs(self, band, var, var_range, 
                 meas_time=30, analyze=False, channel=None, nperseg=2**13,
                 detrend='constant', fs=None, show_plot=False,
                 gcp_mode=True, psd_ylim=None,
                 **kwargs):

        # aliases
        biasaliases=['bias']
        amplitudealiases=['amplitude']

        # vs TES bias
        if var in biasaliases:  
            # requirement
            assert ('bias_group' in kwargs.keys()),'Must specify bias_group.'
            # defaults
            if 'high_current_mode' not in kwargs.keys():
                kwargs['high_current_mode']=False
            if 'cool_wait' not in kwargs.keys():
                kwargs['cool_wait']=30.

        if var in amplitudealiases:  
            # no parameters (yet) but need to null this until we rework the analysis
            kwargs['bias_group']=-1
            pass
                
        psd_dir = os.path.join(self.output_dir, 'psd')
        self.make_dir(psd_dir)

        timestamp = self.get_timestamp()
        np.savetxt(os.path.join(psd_dir, '{}_{}.txt'.format(timestamp,var)),
            var_range)
        datafiles = np.array([], dtype=str)

        for v in var_range:

            if var in biasaliases:
                self.log('Bias {}'.format(v))
                if type(kwargs['bias_group']) is int: # only received one group
                    self.overbias_tes(kwargs['bias_group'], tes_bias=v, 
                                  high_current_mode=kwargs['high_current_mode'],
                                  cool_wait=kwargs['cool_wait'])
                else:
                    self.overbias_tes_all(kwargs['bias_group'], tes_bias=v,
                                  high_current_mode=kwargs['high_current_mode'],
                                  cool_wait=kwargs['cool_wait'])

            if var in amplitudealiases:
                self.log('Tone amplitude {}'.format(v))
                
                ## turn off flux ramp
                self.log('Turning flux ramp off.')
                self.flux_ramp_off()

                ## which channels are configured?
                channels=self.which_on(band)

                ## change amplitude for configured channels
                '''
                self.log('Setting amplitude of all configured channels to {}.'.format(var))
                feedbackEnableArray0=self.get_feedback_enable_array(band)
                self.set_feedback_enable_array(band,np.zeros(feedbackEnableArray0.shape,dtype='int'))
                amplitudeScaleArray=self.get_amplitude_scale_array(band)
                amplitudeScaleArray[channels]=v
                self.set_amplitude_scale_array(band,amplitudeScaleArray)

                ## reLock
                self.log('Re-locking channels at new tone amplitude.')
                self.run_parallel_eta_scan(band)
                '''
                self.log('Tuning for amplitude = {}'.format(v))
                self.setup_notches(band,drive=v)

                ## turn on flux ramp and track
                lms_freq_hz=self.get_lms_freq_hz(band)
                self.log('Turning flux ramp back on and setting up tracking (lms_freq_hz={}).'.format(lms_freq_hz))
                self.tracking_setup(band,channels[0],lms_freq_hz=lms_freq_hz)

            self.log('Taking data')
            datafile = self.take_stream_data(meas_time,gcp_mode=gcp_mode)
            datafiles = np.append(datafiles, datafile)
            self.log('datafile {}'.format(datafile))
            
        self.log('Done with noise vs %s'%(var))

        np.savetxt(os.path.join(psd_dir, '{}_datafiles.txt'.format(timestamp)),
            datafiles, fmt='%s')

        if analyze:
            self.analyze_noise_vs_bias(var_range, datafiles, channel=channel, 
                band=band, bias_group = kwargs['bias_group'], nperseg=nperseg, detrend=detrend, fs=fs, 
                save_plot=True, show_plot=show_plot, data_timestamp=timestamp,
                gcp_mode=gcp_mode,psd_ylim=psd_ylim)

    def get_datafiles_from_file(self,fn_datafiles):
        '''
        For, e.g., noise_vs_bias, the list of datafiles is recorded in a txt file. This function
        simply extracts those filenames and returns them as a list.
        fn_datafiles (str): full path to txt containing names of data files
        Returns: datafiles (list): strings of data-file names.
        '''
        datafiles = []
        f_datafiles = open(fn_datafiles,'r')
        for line in f_datafiles:
            datafiles.append(line.split()[0])
        return datafiles

    def get_biases_from_file(self,fn_biases):
        '''
        For, e.g., noise_vs_bias, the list of commanded bias voltages is recorded in a txt file.
        This function simply extracts those values and returns them as a list.
        fn_biases (str): full path to txt containing list of bias voltages
        Returns biases (list): floats of commanded bias voltages
        '''
        biases = []
        f_biases = open(fn_biases,'r')
        for line in f_biases:
            biases.append(float(line.split()[0]))
        return biases

    def analyze_noise_vs_bias(self, bias, datafile, channel=None, band=None,
        nperseg=2**13, detrend='constant', fs=None, save_plot=True, 
        show_plot=False, make_timestream_plot=False, data_timestamp=None,
        psd_ylim = None,gcp_mode = True,bias_group=None,smooth_len=11,
        show_legend=True,freq_range_summary=None):
        """
        Analysis script associated with noise_vs_bias.

        Args:
        -----
        bias (float array): The bias in voltage. Can also pass an absolute 
            path to a txt containing the bias points.
        datafile (str array): The paths to the datafiles. Must be same length 
            as bias array. Can also pass an absolute path to a txt containing 
            the names of the datafiles.

        Opt Args:
        ---------
        channel (int array): The channels to analyze.
        band (int): The band where the data is taken.
        nperseg (int): Passed to scipy.signal.welch. Number of elements per 
            segment of the PSD.
        detrend (str): Passed to scipy.signal.welch.
        fs (float): Passed to scipy.signal.welch. The sample rate.
        save_plot (bool): Whether to save the plot. Default is True.
        show_plot (bool): Whether to how the plot. Default is False.
        data_timestamp (str): The string used as a save name. Default is None.
        bias_group (int or int array): which bias groups were used. Default is None. 
        smooth_len (int): length of window over which to smooth PSDs for plotting
        freq_range_summary (tup): frequencies between which to take mean noise 
            for summary plot of noise vs. bias; if None, then plot white-noise 
            level from model fit
        """
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        if not show_plot:
            plt.ioff()

        if band is None and channel is None:
            channel = np.arange(512)
        elif band is not None and channel is None:
            channel = self.which_on(band)

        if fs is None:
            self.log('No flux ramp freq given. Loading current flux ramp' +
                'frequency', self.LOG_USER)
            fs = self.get_flux_ramp_freq()*1.0E3

        if isinstance(bias,str):
            self.log('Biases being read from file: %s' % (bias))
            bias = self.get_biases_from_file(bias)
        
        if isinstance(datafile,str):
            self.log('Noise data files being read from file: %s' % (datafile))
            datafile = self.get_datafiles_from_file(datafile)

        mask = np.loadtxt(self.smurf_to_mce_mask_file)

        # Analyze data and save
        for i, (b, d) in enumerate(zip(bias, datafile)):
            # timestamp, I, Q = self.read_stream_data(d)
            timestamp, phase, mask = self.read_stream_data_gcp_save(d)
            phase *= self.pA_per_phi0/(2.*np.pi) # phase converted to pA

            basename, _ = os.path.splitext(os.path.basename(d))
            dirname = os.path.dirname(d)
            psd_dir = os.path.join(dirname, 'psd')
            self.make_dir(psd_dir)

            for ch in channel:
                # ch_idx = np.where(mask == 512*band + ch)[0][0]
                # phase = self.iq_to_phase(I[ch], Q[ch]) * 1.334  # convert to uA
                ch_idx = mask[band, ch]
                f, Pxx = signal.welch(phase[ch_idx], nperseg=nperseg, 
                    fs=fs, detrend=detrend)
                Pxx = np.sqrt(Pxx)  # pA
                np.savetxt(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)), np.array([f, Pxx]))

                if make_timestream_plot:
                    fig,ax = plt.subplots(1)
                    ax.plot(phase[ch])
                    res_freq = self.channel_to_freq(band, ch)
                    ax.set_title('Channel {:03} - {:5.4f} MHz'.format(ch, res_freq))
                    ax.set_xlabel(r'Time index')
                    ax.set_ylabel(r'Phase (pA)')
                
                    if show_plot:
                        plt.show()
                    if save_plot:
                        plt.savefig(os.path.join(self.plot_dir, basename + \
                                    '_timestream_ch{:03}.png'.format(ch)),\
                                    bbox_inches='tight')
                        plt.close()

            # Explicitly remove objects from memory
            del timestamp
            del phase

        # Make plot
        cm = plt.get_cmap('plasma')
        noise_est_data = []
        for ch in channel:
            fig = plt.figure(figsize = (8,5))
            gs = GridSpec(1,3)
            ax0 = fig.add_subplot(gs[:2])
            ax1 = fig.add_subplot(gs[2])
            noise_est_list = []
            for i, (b, d) in enumerate(zip(bias, datafile)):
                basename, _ = os.path.splitext(os.path.basename(d))
                dirname = os.path.dirname(d)

                self.log(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)))

                f, Pxx =  np.loadtxt(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)))
                # smooth Pxx for plotting
                if smooth_len >= 3:
                    window_len = smooth_len
                    self.log('Smoothing PSDs for plotting with window of length %i' % (window_len))
                    s = np.r_[Pxx[window_len-1:0:-1],Pxx,Pxx[-2:-window_len-1:-1]]
                    w = np.hanning(window_len)
                    Pxx_smooth_ext = np.convolve(w/w.sum(),s,mode='valid')
                    ndx_add = window_len % 2
                    Pxx_smooth = Pxx_smooth_ext[(window_len//2)-1+ndx_add:-(window_len//2)]
                else:
                    self.log('No smoothing of PSDs for plotting.')
                    Pxx_smooth = Pxx

                color = cm(float(i)/len(bias))
                ax0.plot(f, Pxx_smooth, color=color, label='{:.2f} V'.format(b))
                ax0.set_xlim(min(f[1:]),max(f[1:]))
                ax0.set_ylim(psd_ylim)

                # fit to noise model; catch error if fit is bad
                popt,pcov,f_fit,Pxx_fit = self.analyze_psd(f,Pxx)
                wl,n,f_knee = popt
                self.log('ch. {}, bias = {:.2f}'.format(ch,b) +
                         ', white-noise level = {:.2f}'.format(wl) +
                         ' pA/rtHz, n = {:.2f}'.format(n) + 
                         ', f_knee = {:.2f} Hz'.format(f_knee))

                # get noise estimate to summarize PSD for given bias
                if freq_range_summary is not None:
                    freq_min,freq_max = freq_range_summary
                    noise_est = np.mean(Pxx[np.logical_and(f>=freq_min,f<=freq_max)])
                    self.log('ch. {}, bias = {:.2f}'.format(ch,b) + 
                             ', mean noise between {:.3e} and {:.3e} Hz = {:.2f} pA/rtHz'.format(freq_min,freq_max,noise_est))
                else:
                    noise_est = wl
                noise_est_list.append(noise_est)

                ax0.plot(f_fit, Pxx_fit, color=color, linestyle='--')
                ax0.plot(f, wl + np.zeros(len(f)), color=color,
                        linestyle=':')
                ax0.plot(f_knee,2.*wl,marker = 'o',linestyle = 'none',
                        color=color)
                
                ax1.plot(b,wl,color=color,marker='s',linestyle='none')

            ax0.set_xlabel(r'Freq [Hz]')
            ax0.set_ylabel(r'PSD [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]')
            ax0.set_xscale('log')
            ax0.set_yscale('log')
            if show_legend:
                ax0.legend(loc = 'upper right')
            res_freq = self.channel_to_freq(band, ch)

            ax1.set_xlabel(r'Commanded bias voltage [V]')
            if freq_range_summary is not None:
                ylabel_summary = r'Mean noise %.2f-%.2f Hz' % (freq_min,freq_max)
            else:
                ylabel_summary = r'White-noise level'
            ax1.set_ylabel(r'%s [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]' % (ylabel_summary))
            bottom = max(0.95*min(noise_est_list),0.)
            top_desired = 1.05*max(noise_est_list)
            if psd_ylim is not None:
                top = min(psd_ylim[1],top_desired)
            else:
                top = top_desired
            ax1.set_ylim(bottom=bottom, top=top)
            ax1.grid()

            if type(bias_group) is not int: # ie if there were more than one
                fig_title_string = ''
                file_name_string = ''
                for i in range(len(bias_group)):
                    g = bias_group[i]
                    fig_title_string += str(g) + ',' # I'm sorry but the satellite was down
                    file_name_string += str(g) + '_'
            else:
                fig_title_string = str(bias_group) + ','
                file_name_string = str(bias_group) + '_'

            fig.suptitle(basename + ' Band {}, Group {} Channel {:03} - {:.2f} MHz'.format(band,fig_title_string,ch, res_freq))
            plt.tight_layout(rect=[0.,0.03,1.,0.95])

            if show_plot:
                plt.show()

            if save_plot:
                plot_name = 'noise_vs_bias_band{}_g{}ch{:03}.png'.format(band,file_name_string,
                    ch)
                if data_timestamp is not None:
                    plot_name = '{}_'.format(data_timestamp) + plot_name
                else:
                    plot_name = '{}_'.format(self.get_timestamp()) + plot_name
                plot_fn = os.path.join(self.plot_dir, plot_name)
                self.log('Saving plot to %s' % (plot_fn))
                plt.savefig(plot_fn,
                    bbox_inches='tight')
                plt.close()

            del f
            del Pxx

            noise_est_data.append({'ch':ch,'noise_est_list':noise_est_list})
        
        # make summary histogram of noise vs. bias over all analyzed channels
        noise_est_data_bias = []
        for i in range(len(bias)):
            b = bias[i]
            noise_est_bias = []
            for j in range(len(noise_est_data)):
                noise_est_bias.append(noise_est_data[j]['noise_est_list'][i])
            noise_est_data_bias.append(np.array(noise_est_bias))
        
        if psd_ylim is not None:
            bin_min = np.log10(psd_ylim[0])
            bin_max = np.log10(psd_ylim[1])
        else:
            bin_min = np.floor(np.log10(np.min(noise_est_data_bias)))
            bin_max = np.ceil(np.log10(np.max(noise_est_data_bias)))

        plt.figure()
        bins_hist = np.logspace(bin_min,bin_max,20)
        hist_mat = np.zeros((len(bias),len(bins_hist)-1))
        noise_est_median_list = []
        for i in range(len(bias)):
            hist_mat[i,:],_ = np.histogram(noise_est_data_bias[i],bins=bins_hist)
            noise_est_median_list.append(np.median(noise_est_data_bias[i]))
        X_hist,Y_hist = np.meshgrid(bins_hist,np.arange(len(bias)+1))
        plt.pcolor(X_hist,Y_hist,hist_mat)
        cbar = plt.colorbar()
        cbar.set_label('Number of channels')
        plt.xscale('log')
        plt.xlabel(r'%s [$\mathrm{pA}/\sqrt{\mathrm{Hz}}$]' % (ylabel_summary))
        plt.xlim(10**bin_min,10**bin_max)
        plt.title(basename + ': Band {}, Group {}'.format(band,fig_title_string.strip(',')))
        ytick_labels = []
        for b in bias:
            ytick_labels.append('{}'.format(b))
        ytick_locs = np.arange(len(bias)) + 0.5
        plt.yticks(ticks=ytick_locs,labels=ytick_labels)
        plt.ylabel('Commanded bias voltage [V]')
        plt.plot(noise_est_median_list,ytick_locs,linestyle='--',marker='o',
                 color='r',label='Median')
        plt.legend(loc='center left')
        if show_plot:
            plt.show()
        if save_plot:
            plot_name = 'noise_vs_bias_band{}_g{}hist.png'.format(band,file_name_string)
            if data_timestamp is not None:
                plot_name = '{}_'.format(data_timestamp) + plot_name
            else:
                plot_name = '{}_'.format(self.get_timestamp()) + plot_name
            plot_fn = os.path.join(self.plot_dir, plot_name)
            self.log('\nSaving histogram to %s' % (plot_fn))
            plt.savefig(plot_fn,bbox_inches='tight')
            plt.close()

    def analyze_psd(self, f, Pxx, p0=[100.,0.5,0.01]):
        def noise_model(freq, wl, n, f_knee):
            '''
            Crude model for noise modeling.
            wl (float): white-noise level
            n (float): exponent of 1/f^n component
            f_knee (float): frequency at which white noise = 1/f^n component
            '''
            A = wl*(f_knee**n)
            return A/(freq**n) + wl
        bounds_low = [0.,0.,0.] # constrain 1/f^n to be red spectrum
        bounds_high = [np.inf,np.inf,np.inf]
        bounds = (bounds_low,bounds_high)

        freq_max = 0.5*self.config.get('smurf_to_mce').get('filter_freq')  # maximum frequency for noise-model; roll-off of low-pass filter in Hz
        n_pts = len(f)
        i_max = n_pts - 1 # index of max. frequency; highest index if freq array never gets above freq_max
        for i in range(1,n_pts):
            if f[i] > freq_max:
                i_max = i
                break

        try:
            popt, pcov = optimize.curve_fit(noise_model, f[1:i_max+1], Pxx[1:i_max+1], p0=p0, 
                bounds=bounds)
        except Exception as e:
            wl = np.mean(Pxx[1:i_max+1])
            print('Unable to fit noise model. Reporting mean noise: %.2f pA/rtHz' % (wl))
            popt = [wl,1.,0.]
            pcov = None
        df = f[1] - f[0]
        f_fit = np.arange(f[1],f[-1] + df,df/10.)
        Pxx_fit = noise_model(f_fit,*popt)

        return popt,pcov,f_fit,Pxx_fit
