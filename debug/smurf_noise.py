import numpy as np
from scipy import signal
from pysmurf.base import SmurfBase
import os
import time

class SmurfNoiseMixin(SmurfBase):

    def take_noise_psd(self, band, meas_time, channel=None, nperseg=2**12, 
        detrend='constant', fs=None, low_freq=np.array([.1, 1.]), 
        high_freq=np.array([1., 5.]), make_channel_plot=True,
        make_summary_plot=True, save_data=False, show_plot=False):
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
        """
        if channel is None:
            channel = self.which_on(band)
        n_channel = self.get_number_channels(band)

        datafile = self.take_stream_data(band, meas_time)
        basename, _ = os.path.splitext(os.path.basename(datafile))

        timestamp, I, Q = self.read_stream_data(datafile)

        if fs is None:
            self.log('No flux ramp freq given. Loading current flux ramp'+
                'frequency', self.LOG_USER)
            fs = self.get_flux_ramp_freq()*1.0E3

        self.log('Plotting channels {}'.format(channel), self.LOG_USER)

        import matplotlib.pyplot as plt

        noise_floors = np.zeros((len(low_freq), n_channel))*np.nan

        if not show_plot:
            plt.ioff()
        for c, ch in enumerate(channel):
            phase = self.iq_to_phase(I[ch], Q[ch])

            # Calculate to power spectrum and convert to pA
            f, Pxx = signal.welch(phase, nperseg=nperseg, 
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)/(2*np.pi)*self.pA_per_phi0*1e12
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                idx = np.logical_and(f>l, f<h)
                noise_floors[i, ch] = np.mean(Pxx[idx])

            if make_channel_plot:
                fig, ax = plt.subplots(3, figsize=(5,7))
                ax[0].plot(I[ch], label='I')
                ax[0].plot(Q[ch], label='Q')
                ax[0].legend()
                ax[0].set_ylabel('I/Q')
                ax[0].set_xlabel('Sample Num')

                ax[1].plot(self.pA_per_phi0 * phase / (2*np.pi))
                ax[1].set_xlabel('Sample Num')
                ax[1].set_ylabel('Phase [pA]')

                ax[2].plot(f, Pxx)
                ax[2].set_xlabel('Freq [Hz]')
                ax[2].set_ylabel('Amp [pA/rtHz]')
                ax[2].set_yscale('log')
                ax[2].set_xscale('log')

                ax[2].axhline(noise_floors[-1,ch], color='k', linestyle='--')
                print(noise_floors[-1, ch])

                ax[0].set_title('Band {} Ch {:03}'.format(band, ch))

                plt.tight_layout()

                plot_name = basename+'_b{}_ch{:03}.png'.format(band, ch)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
                if not show_plot:
                    plt.close()

        if save_data:
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                save_name = basename+'_{:3.2f}_{:3.2f}.txt'.format(l, h)
                np.savetxt(os.path.join(self.plot_dir, save_name), 
                    noise_floors[i])

        if make_summary_plot:
            bins = np.arange(0,351,10)
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                fig, ax = plt.subplots(1, figsize=(4,3))
                ax.hist(noise_floors[i,~np.isnan(noise_floors[i])], bins=bins)
                ax.text(0.03, 0.95, '{:3.2f}'.format(l) + '-' +
                    '{:3.2f} Hz'.format(h),
                    transform=ax.transAxes, fontsize=10)

                plot_name = basename + \
                    '_b{}_{}_{}_noise_hist.png'.format(band, l, h)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
                if not show_plot:
                    plt.close()

        plt.ion()

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

    def noise_vs_bias(self, band, bias_high=6, bias_low=3, step_size=.1,
        meas_time=30., analyze=False, channel=None, nperseg=2**13,
        detrend='constant', fs=None):
        """
        This ramps the TES voltage from bias_high to bias_low and takes noise
        measurements. You can make it analyze the data and make plots with the
        optional argument analyze=True. Note that the analysis is a little
        slow.

        Args:
        -----
        band (int): The band to take noise vs bias data on

        Opt Args:
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
        """
        bias = np.arange(bias_high, bias_low-step_size, -1*step_size)

        psd_dir = os.path.join(self.output_dir, 'psd')
        self.make_dir(psd_dir)


        timestamp = self.get_timestamp()
        np.savetxt(os.path.join(psd_dir, '{}_bias.txt'.format(timestamp)),
            bias)
        datafiles = np.array([], dtype=str)

        for b in bias:
            self.log('Bias {}'.format(b))
            self.overbias_tes(4, tes_bias=b)

            self.log('Taking data')
            datafile = self.take_stream_data(band, meas_time)
            datafiles = np.append(datafiles, datafile)
            self.log('datafile {}'.format(datafile))

        self.log('Done with noise vs bias')
        np.savetxt(os.path.join(psd_dir, '{}_datafiles.txt'.format(timestamp)),
            datafiles, fmt='%s')

        if analyze:
            self.analyze_noise_vs_bias(bias, datafiles, channel=channel, 
                band=band, nperseg=nperseg, detrend=detrend, fs=fs, 
                save_plot=True, show_plot=False, data_timestamp=timestamp)

    def analyze_noise_vs_bias(self, bias, datafile, channel=None, band=None,
        nperseg=2**13, detrend='constant', fs=None, save_plot=True, 
        show_plot=False, data_timestamp=None):
        """

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
            timestamp, I, Q = self.read_stream_data(d)
            basename, _ = os.path.splitext(os.path.basename(d))
            dirname = os.path.dirname(d)
            psd_dir = os.path.join(dirname, 'psd')
            self.make_dir(psd_dir)

            for ch in channel:
                phase = self.iq_to_phase(I[ch], Q[ch]) * 1.334  # convert to uA
                f, Pxx = signal.welch(phase, nperseg=nperseg, 
                    fs=fs, detrend=detrend)
                Pxx = np.sqrt(Pxx) * 1.0E6  # pA
                np.savetxt(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)), np.array([f, Pxx]))

            # Explicitly remove objects from memory
            del timestamp
            del I
            del Q

        # Make plot
        cm = plt.get_cmap('viridis')
        for ch in channel:
            fig, ax = plt.subplots(1)
            for i, (b, d) in enumerate(zip(bias, datafile)):
                basename, _ = os.path.splitext(os.path.basename(d))
                dirname = os.path.dirname(d)

                print(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)))

                f, Pxx =  np.loadtxt(os.path.join(psd_dir, basename + 
                    '_psd_ch{:03}.txt'.format(ch)))

                color = cm(float(i)/len(bias))
                ax.plot(f, Pxx, color=color, label='{:3.2f}'.format(b))        
                ax.set_xlabel(r'Freq [Hz]')
                ax.set_ylabel(r'$pA/\sqrt{Hz}]$')
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.legend()
                ax.set_title('Channel {:03}'.format(ch))

            if show_plot:
                plt.show()

            if save_plot:
                plot_name = 'noise_vs_bias_band{}_ch{:03}.png'.format(band,
                    ch)
                if data_timestamp is not None:
                    plot_name = '{}_'.format(data_timestamp) + plot_name
                else:
                    plot_name = '{}_'.format(self.get_timestamp) + plot_name
                plt.savefig(os.path.join(self.plot_dir, plot_name))
