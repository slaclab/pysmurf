import numpy as np
from pysmurf.base import SmurfBase
import time
import os

class SmurfIVMixin(SmurfBase):

    def slow_iv(self, band, wait_time=.1, bias=None, bias_high=19.9, bias_low=0, 
        bias_step=.1, show_plot=False, high_current_wait=.25, make_plot=True,
        save_plot=True):
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
        bias (float array): A float array of bias values. Must go high to low.
        bias_high (int): The maximum TES bias in volts. Default 19.9
        bias_low (int): The minimum TES bias in volts. Default 0
        bias_step (int): The step size in volts. Default .1
        """
        # Look for good channels
        channel = self.which_on(band)
        n_channel = self.get_number_channels(band)

        # drive high current through the TES to attempt to drive nomral
        self.set_tes_bias_bipolar(4, 19.9)
        time.sleep(.1)
        self.log('Driving high current through TES. ' + \
            'Waiting {}'.format(high_current_wait))
        self.set_cryo_card_relays(0x10004)
        time.sleep(high_current_wait)
        self.set_cryo_card_relays(0x10000)
        time.sleep(.1)

        self.log('Staring to take IV.', self.LOG_USER)

        if bias is None:
            bias = np.arange(bias_high, bias_low, -bias_step)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        self.set_tes_bias_bipolar(4, bias[0])
        time.sleep(1)

        datafile = self.stream_data_on(band)

        for b in bias:
            self.log('Bias at {:4.3f}'.format(b))
            self.set_tes_bias_bipolar(4, b)  # 4 is for band 3
            time.sleep(wait_time)

        self.log('Done with TES bias ramp', self.LOG_USER)

        self.stream_data_off(band)

        basename, _ = os.path.splitext(os.path.basename(datafile))
        np.save(os.path.join(basename + '_iv_bias.txt'), bias)

        ivs = {}
        ivs['bias'] = bias


        for c, ch in enumerate(channel):
            timestamp, I, Q = self.read_stream_data(datafile)
            phase = self.iq_to_phase(I[ch], Q[ch]) * 1.443
            if make_plot:
                import matplotlib.pyplot as plt
                
                if not show_plot:
                    plt.ioff()

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

                plot_name = basename + \
                    '_IV_stream_b{}_ch{:03}.png'.format(band, ch)
                if save_plot:
                    plt.savefig(os.path.join(self.plot_dir, plot_name), 
                        bbox_inches='tight', dpi=300)
                if not show_plot:
                    plt.close()

            r, rn = self.analyze_slow_iv(bias, phase, basename=basename, 
                band=band, channel=ch, make_plot=make_plot, show_plot=show_plot,
                save_plot=save_plot)
            ivs[ch] = {
                'R' : r,
                'Rn' : rn
            }

        np.save(os.path.join(self.output_dir, basename + '_iv'), ivs)

    def analyze_slow_iv(self, v_bias, resp, make_plot=True, show_plot=False,
        save_plot=True, basename=None, band=None, channel=None, **kwargs):
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

        i_bias = 1.0E6 * v_bias / 8.038E3  # Total line impedance and uA

        if make_plot:
            import matplotlib.pyplot as plt
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

        for i in np.arange(n_step):
            s = i*step_size
            e = (i+1) * step_size
            sb = int(s + np.ceil(step_size * 3. / 5))
            eb = int(s + np.ceil(step_size * 9. / 10))

            resp_bin[i] = np.mean(resp[sb:eb])

        d_resp = np.diff(resp_bin)
        d_resp = d_resp[::-1]
        d_v = (v_bias[:-1] + v_bias[1:])/2.
        d_v = d_v[::-1]
        d_i = (i_bias[:-1] + i_bias[1:])/2.
        d_i = d_i[::-1]
        v_bias = v_bias[::-1]
        i_bias = i_bias[::-1]
        resp_bin = resp_bin[::-1]

        # index of the end of the superconducting branch
        sc_idx = np.ravel(np.where(d_resp == np.max(d_resp)))[0]

        if sc_idx == 0:
            return None, None

        # index of the start of the normal branch
        nb_idx = n_step-2
        for i in np.arange(n_step-2, sc_idx, -1):
            if d_resp[i] > 0:
                nb_idx = i
                break

        norm_fit = np.polyfit(i_bias[nb_idx:], resp_bin[nb_idx:], 1)
        if norm_fit[0] < 0:  # Check for flipped polarity
            resp_bin = -1 * resp_bin
            norm_fit = np.polyfit(i_bias[nb_idx:], resp_bin[nb_idx:], 1)

        resp_bin -= norm_fit[1]  # now in real current units
        
        sc_fit = np.polyfit(i_bias[:sc_idx], resp_bin[:sc_idx], 1)

        R_sh = .003
        R = R_sh * (i_bias/(resp_bin) - 1)
        R_n = np.mean(R[nb_idx:])

        if make_plot:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(i_bias, resp_bin, '.')
            ax[0].set_ylabel(r'$I_{TES}$ $[\mu A]$')

            ax[0].plot(i_bias, norm_fit[0] * i_bias , linestyle='--', color='k')  

        

            ax[0].plot(i_bias[:sc_idx], 
                sc_fit[0] * i_bias[:sc_idx] + sc_fit[1], linestyle='--', 
                color='k')

            # Highlight the transition
            ax[0].axvspan(d_i[sc_idx], d_i[nb_idx], color='k', alpha=.15)
            ax[1].axvspan(d_i[sc_idx], d_i[nb_idx], color='k', alpha=.15)

            ax[0].text(.95, .04, 'SC slope: {:3.2f}'.format(sc_fit[0]), 
                transform=ax[0].transAxes, fontsize=12, 
                horizontalalignment='right')


            ax[1].plot(i_bias, R/R_n, '.')
            # ax[1].axhline(R_n, color='k', linestyle=':')
            ax[1].axhline(1, color='k', linestyle='--')
            ax[1].set_ylabel(r'$R/R_N$')
            ax[1].set_xlabel(r'$I_{b}$ ' + '$[\mu A]$')
            ax[1].set_ylim(0, 1.1)

            ax[1].text(.95, .18, r'$R_{sh}$: ' + '{}'.format(R_sh*1.0E3) + 
                r' $m\Omega$' , transform=ax[1].transAxes, fontsize=12,
                horizontalalignment='right')
            ax[1].text(.95, .04, r'$R_{N}$: ' + '{:3.2f}'.format(R_n*1.0E3) + 
                r' $m\Omega$' , transform=ax[1].transAxes, fontsize=12,
                horizontalalignment='right')

            # axt = ax[0].twiny()
            # axt.set_xlim(ax[0].get_xlim())
            # axt.set_xticks()

            if band is not None and channel is not None:
                ax[0].set_title('Band {} Ch {:03}'.format(band, channel))

            plt.tight_layout()

            if save_plot:
                if basename is None:
                    basename = self.get_timestamp()
                plot_name = basename + \
                        '_IV_curve_b{}_ch{:03}.png'.format(band, channel)
                plt.savefig(os.path.join(self.plot_dir, plot_name),
                    bbox_inches='tight', dpi=300)
            if show_plot:
                plt.show()
            else:
                plt.close()

        return R, R_n


