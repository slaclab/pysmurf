import numpy as np
from pysmurf.base import SmurfBase
import time
import os

class SmurfIVMixin(SmurfBase):

    def slow_iv(self, band, wait_time=.1, bias_high=19.9, bias_low=0, 
        bias_step=.1, show_plot=False, high_current_wait=.25, make_plot=True):
        """
        Steps the TES bias down slowly. Starts at bias_high to bias_low with
        step size bias_step. Waits wait_time between changing steps.

        Args:
        -----
        daq (int) : The DAQ number

        Opt Args:
        ---------
        wait_time (float): The amount of time between changing TES biases in 
            seconds. Default .1 sec.
        bias_high (int): The maximum TES bias in volts. Default 19.9
        bias_low (int): The minimum TES bias in volts. Default 0
        bias_step (int): The step size in volts. Default .1
        """
        # Look for good channels
        channel = self.which_on(band)
        n_channel = self.get_number_channels(band)

        # drive high current through the TES to attempt to drive nomral
        self.set_tes_bias_bipolar(4, bias_high)
        time.sleep(.1)
        self.log('Driving high current through TES. ' + \
            'Waiting {}'.format(high_current_wait))
        self.set_cryo_card_relays(0x10004)
        time.sleep(high_current_wait)
        self.set_cryo_card_relays(0x10000)
        time.sleep(.1)

        self.log('Staring to take IV.', self.LOG_USER)
        datafile = self.stream_data_on(band)

        bias = np.arange(bias_high, bias_low, -bias_step)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        for b in bias:
            self.set_tes_bias_bipolar(4, b)
            time.sleep(wait_time)

        self.log('Done with TES bias ramp', self.LOG_USER)

        self.stream_data_off(band)

        basename, _ = os.path.splitext(os.path.basename(datafile))
        np.save(os.path.join(basename + '_iv_bias.txt'), bias)
        if make_plot:
            import matplotlib.pyplot as plt
            timestamp, I, Q = self.read_stream_data(datafile)

            if not show_plot:
                plt.ioff()
            for c, ch in enumerate(channel):
                phase = self.iq_to_phase(I[ch], Q[ch])
                fig, ax = plt.subplots(2, sharex=True)

                ax[0].plot(I[ch], label='I')
                ax[0].plot(Q[ch], label='Q')
                ax[0].legend()
                ax[0].set_ylabel('I/Q')
                ax[0].set_xlabel('Sample Num')

                ax[1].plot(self.pA_per_phi0 * phase / (2*np.pi))
                ax[1].set_xlabel('Sample Num')
                ax[1].set_ylabel('Phase [pA]')

                ax[0].set_title('Band {} Ch {:03}'.format(band, ch))
                plt.tight_layout()

                plot_name = basename+'_IV_b{}_ch{:03}.png'.format(band, ch)
                plt.savefig(os.path.join(self.plot_dir, plot_name), 
                    bbox_inches='tight')
                if not show_plot:
                    plt.close()

    def analyze_slow_iv(self, v_bias, resp, make_plot=True, show_plot=False,
        **kwargs):
        """
        Analyzes the IV curve taken with slow_iv()

        Args:
        v_bias (float array): The commanded bias in voltage. Length n_steps
        resp (float array): The TES response. Of length n_pts (not the same 
            as n_steps)
        """
        n_pts = len(resp)
        n_step = len(v_bias)

        step_size = float(n_pts)/n_step  # The number of samples per V_bias step

        resp_bin = np.zeros(n_step)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1)
        ax.plot(resp)
        for i in np.arange(n_step):
            s = i*step_size
            e = (i+1) * step_size
            sb = int(s + np.ceil(step_size * 4. / 5))
            eb = int(s + np.ceil(step_size * 9. / 10))
            if i%2 == 0:
                ax.axvspan(s, e, color='k', alpha=.1)
            resp_bin[i] = np.mean(resp[sb:eb])
            # print('{} - {} : {}'.format(sb, eb, resp_bin[i]))
            ax.axhline(resp_bin[i], xmin=sb, xmax=eb, color='r')

        fig, ax = plt.subplots(1, sharex=True)
        ax.plot(v_bias, resp_bin, '.')
        ax.set_ylabel(r'I_{TES}')
        ax.set_xlabel(r'$V_{b}$')
        d_resp = np.diff(resp_bin)

        d_resp = d_resp[::-1]
        d_v = (v_bias[:-1] + v_bias[1:])/2.
        d_v = d_v[::-1]
        v_bias = v_bias[::-1]
        resp_bin = resp_bin[::-1]


        # index of the end of the superconducting branch
        sc_idx = np.ravel(np.where(d_resp == np.max(d_resp)))[0]

        # index of the start of the normal branch
        nb_idx = n_step-2
        for i in np.arange(n_step-2, sc_idx, -1):
            if d_resp[i] > 0:
                nb_idx = i
                break

        ax.axvline(d_v[sc_idx], color='k')
        ax.axvline(d_v[nb_idx], color='k')

        norm_fit = np.polyfit(v_bias[nb_idx:], resp_bin[nb_idx:], 1)
        ax.plot(v_bias[nb_idx:], 
            norm_fit[0] * v_bias[nb_idx:] + norm_fit[1], linestyle='--', 
            color='k')

        sc_fit = np.polyfit(v_bias[:sc_idx], resp_bin[:sc_idx], 1)
        ax.plot(v_bias[:sc_idx], 
            sc_fit[0] * v_bias[:sc_idx] + sc_fit[1], linestyle='--', 
            color='k')

