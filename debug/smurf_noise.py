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
        grid_on=False,gcp_mode=True,datafile = None):
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
        datefile: if data has already been taken, can point to a file to bypass data taking and just analyze
        """
        if channel is None:
            channel = self.which_on(band)
        n_channel = self.get_number_channels(band)

        if datafile == None:
            datafile = self.take_stream_data(band, meas_time, gcp_mode=gcp_mode)
        else:
            self.log('Reading data from %s' % (datafile))
        basename, _ = os.path.splitext(os.path.basename(datafile))

        # timestamp, I, Q = self.read_stream_data(datafile)
        timestamp, phase = self.read_stream_data(datafile,gcp_mode = gcp_mode)
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
            ch_idx = ch
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
                # ax[0].plot(I[ch], label='I')
                # ax[0].plot(Q[ch], label='Q')
                # ax[0].legend()
                # ax[0].set_ylabel('I/Q')
                # ax[0].set_xlabel('Sample Num')
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
                ax[0].set_title('Band {} Ch {:03} - {:.1f} MHz'.format(band, ch, res_freq))

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
            bins = np.arange(0,351,10)
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                fig, ax = plt.subplots(1, figsize=(10,6))
                ax.hist(noise_floors[i,~np.isnan(noise_floors[i])], bins=bins)
                ax.text(0.03, 0.95, '{:3.2f}'.format(l) + '-' +
                    '{:3.2f} Hz'.format(h),
                    transform=ax.transAxes, fontsize=10)

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

                fig,ax = plt.subplots(1,3)
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
        psd_ylim = None):
        """
        This ramps the TES voltage from bias_high to bias_low and takes noise
        measurements. You can make it analyze the data and make plots with the
        optional argument analyze=True. Note that the analysis is a little
        slow.

        Args:
        -----
        band (int): The band to take noise vs bias data on

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

        psd_dir = os.path.join(self.output_dir, 'psd')
        self.make_dir(psd_dir)


        timestamp = self.get_timestamp()
        np.savetxt(os.path.join(psd_dir, '{}_bias.txt'.format(timestamp)),
            bias)
        datafiles = np.array([], dtype=str)

        for b in bias:
            self.log('Bias {}'.format(b))
            self.overbias_tes(bias_group, tes_bias=b, 
                              high_current_mode=high_current_mode,
                              cool_wait=cool_wait)

            self.log('Taking data')
            datafile = self.take_stream_data(band, meas_time,gcp_mode = gcp_mode)
            datafiles = np.append(datafiles, datafile)
            self.log('datafile {}'.format(datafile))

        self.log('Done with noise vs bias')
        np.savetxt(os.path.join(psd_dir, '{}_datafiles.txt'.format(timestamp)),
            datafiles, fmt='%s')

        if analyze:
            self.analyze_noise_vs_bias(bias, datafiles, channel=channel, 
                band=band, bias_group = bias_group,nperseg=nperseg, detrend=detrend, fs=fs, 
                save_plot=True, show_plot=show_plot, data_timestamp=timestamp,
                gcp_mode=gcp_mode,psd_ylim=psd_ylim)

    def analyze_noise_vs_bias(self, bias, datafile, channel=None, band=None,
        nperseg=2**13, detrend='constant', fs=None, save_plot=True, 
        show_plot=False, make_timestream_plot=False, data_timestamp=None,
        psd_ylim = None,gcp_mode = True,bias_group=None):
        """
        Analysis script associated with noise_vs_bias.

        Args:
        -----
        bias (float array): The bias in voltage.
        datafile (str array): The paths to the datafiles. Must be same length 
            as bias array.

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
        """
        import matplotlib.pyplot as plt

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

        # Analyze data and save
        for i, (b, d) in enumerate(zip(bias, datafile)):
            # timestamp, I, Q = self.read_stream_data(d)
            timestamp, phase = self.read_stream_data(d,gcp_mode=gcp_mode)
            phase *= self.pA_per_phi0/(2.*np.pi) # phase converted to pA

            basename, _ = os.path.splitext(os.path.basename(d))
            dirname = os.path.dirname(d)
            psd_dir = os.path.join(dirname, 'psd')
            self.make_dir(psd_dir)

            for ch in channel:
                # phase = self.iq_to_phase(I[ch], Q[ch]) * 1.334  # convert to uA
                f, Pxx = signal.welch(phase[ch], nperseg=nperseg, 
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
        for ch in channel:
            fig, ax = plt.subplots(1)
            for i, (b, d) in enumerate(zip(bias, datafile)):
                basename, _ = os.path.splitext(os.path.basename(d))
                dirname = os.path.dirname(d)

                self.log(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)))

                f, Pxx =  np.loadtxt(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)))

                color = cm(float(i)/len(bias))
                ax.plot(f, Pxx, color=color, label='{:.2f} V'.format(b))
                ax.set_xlim(min(f[1:]),max(f[1:]))
                ax.set_ylim(psd_ylim)
                # fit to noise model; catch error if fit is bad
                try:
                    popt,pcov,f_fit,Pxx_fit = self.analyze_psd(f,Pxx)
                    wl,n,f_knee = popt
                    self.log('ch. {}, bias = {:.2f}'.format(ch,b) +
                        ', white-noise level = {:.2f}'.format(wl) +
                        ' pA/rtHz, n = {:.2f}'.format(n) + 
                        ', f_knee = {:.2f} Hz'.format(f_knee))

                    ax.plot(f_fit, Pxx_fit, color=color, linestyle='--')
                    ax.plot(f, wl + np.zeros(len(f)), color=color,
                        linestyle=':')
                    ax.plot(f_knee,2.*wl,marker = 'o',linestyle = 'none',
                        color=color)
                except Exception as e: 
                    print(e)
                    self.log('%s, bias = %.2f: bad fit to noise model' % (d,b))

            ax.set_xlabel(r'Freq [Hz]')
            ax.set_ylabel(r'$\mathrm{pA}/\sqrt{\mathrm{Hz}}]$')
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.legend()
            res_freq = self.channel_to_freq(band, ch)
            ax.set_title(basename + ' Band {}, Group {}, Channel {:03} - {:.1f} MHz'.format(band,bias_group,ch, res_freq))

            if show_plot:
                plt.show()

            if save_plot:
                plot_name = 'noise_vs_bias_band{}_g{}_ch{:03}.png'.format(band,bias_group,
                    ch)
                if data_timestamp is not None:
                    plot_name = '{}_'.format(data_timestamp) + plot_name
                else:
                    plot_name = '{}_'.format(self.get_timestamp()) + plot_name
                plt.savefig(os.path.join(self.plot_dir, plot_name),
                    bbox_inches='tight')
                plt.close()
            
            del f
            del Pxx


    def analyze_psd(self, f, Pxx, p0=[100.,0.5,0.001]):
        def noise_model(freq, wl, n, f_knee):
            '''
            Crude model for noise modeling.
            wl (float): white-noise level
            n (float): exponent of 1/f^n component
            f_knee (float): frequency at which white noise = 1/f^n component
            '''
            A = wl*(f_knee**n)
            return A/(freq**n) + wl
        bounds_low = [0.,-np.inf,0.]
        bounds_high = [np.inf,np.inf,np.inf]
        bounds = (bounds_low,bounds_high)

        try:
            popt, pcov = optimize.curve_fit(noise_model, f[1:], Pxx[1:], p0=p0, bounds=bounds)
        except Exception as e:
            print(e)
            wl = np.mean(Pxx[1:])
            print('Unable to fit noise model. Reporting mean noise: %.2f pA/rtHz' % (wl))
            popt = [wl,1.,0.]
            pcov = None
        df = f[1] - f[0]
        f_fit = np.arange(f[1],f[-1] + df,df/10.)
        Pxx_fit = noise_model(f_fit,*popt)

        return popt,pcov,f_fit,Pxx_fit
