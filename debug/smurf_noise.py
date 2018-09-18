import numpy as np
from scipy import signal
from pysmurf.base import SmurfBase
import os

class SmurfNoiseMixin(SmurfBase):

    def take_noise_psd(self, band, meas_time, channel=None, nperseg=2**12, 
        detrend='constant', fs=None, low_freq=np.array([.1, 1.]), 
        high_freq=np.array([1., 5.]), make_channel_plot=True,
        make_summary_plot=True):
        """
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

        pA_per_phi0 = 9e-6 # pA/Phi0

        noise_floors = np.zeros((len(low_freq), n_channel))*np.nan

        plt.ioff()
        for c, ch in enumerate(channel):
            phase = self.iq_to_phase(I[ch], Q[ch])
            phase -= np.mean(phase)

            # Calculate to power spectrum and convert to pA
            f, Pxx = signal.welch(phase, nperseg=nperseg, 
                fs=fs, detrend=detrend)
            Pxx = np.sqrt(Pxx)/(2*np.pi)*pA_per_phi0*1e12
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                idx = np.logical_and(f>l, f<h)
                noise_floors[i, ch] = np.mean(Pxx[idx])

            if make_channel_plot:
                fig, ax = plt.subplots(3, figsize=(5,7))
                ax[0].plot(I[ch], label='I')
                ax[0].plot(Q[ch], label='Q')
                ax[0].legend()
                ax[0].set_ylabel('I/Q')
                ax[0].set_xlabel('Time')

                ax[1].plot(phase)
                ax[1].set_xlabel('Time')
                ax[1].set_ylabel('Phase')

                ax[2].plot(f, Pxx)
                ax[2].set_xlabel('Freq [Hz]')
                ax[2].set_ylabel('Amp [pA/rtHz]')
                ax[2].set_yscale('log')
                ax[2].set_xscale('log')

                ax[2].axhline(noise_floors[i,-1], color='k', linestyle='--')

                ax[0].set_title('Band {} Ch {:03}'.format(band, ch))

                plt.tight_layout()

                plot_name = basename+'_b{}_ch{:03}.png'.format(band, ch)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
                plt.close()

        if make_summary_plot:
            bins = np.arange(0,351,10)
            for i, (l, h) in enumerate(zip(low_freq, high_freq)):
                fig, ax = plt.subplots(1, figsize=(4,3))
                ax.hist(noise_floors[i,~np.isnan(noise_floors[i])], bins=bins)
                ax.text(0.03, 0.95, '{:3.2f}'.format(l) + '-' +'{:3.2f} Hz'.format(h),
                    transform=ax.transAxes, fontsize=10)

                plot_name = basename+'_b{}_{}_{}_noise_hist.png'.format(band, l, h)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
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
