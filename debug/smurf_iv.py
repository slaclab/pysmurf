import numpy as np
from pysmurf.base import SmurfBase
import time
import os

class SmurfIVMixin(SmurfBase):

    def slow_iv(self, band, bias_group, wait_time=.1, bias=None, bias_high=19.9, 
        bias_low=0, bias_step=.1, show_plot=False, high_current_wait=.25, 
        make_plot=True, save_plot=True, channels=None, high_current_mode=False,
        rn_accept_min=1e-3, rn_accept_max=1., overbias_voltage=19.9):
        """
        Steps the TES bias down slowly. Starts at bias_high to bias_low with
        step size bias_step. Waits wait_time between changing steps.

        Args:
        -----
        band (int) : The frequency band to take the data in
        bias_group (int) : The bias group to take data on.

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
        if channels is None:
            channels = self.which_on(band)

        if overbias_voltage != 0.:
            overbias = True
        else:
            overbias = False

        # drive high current through the TES to attempt to drive nomral
        #self.set_tes_bias_bipolar(bias_num, 19.9)
        #time.sleep(.1)
        #self.log('Driving high current through TES. ' + \
        #    'Waiting {}'.format(high_current_wait))
        #self.set_cryo_card_relays(0x10004)
        #time.sleep(high_current_wait)
        #self.set_cryo_card_relays(0x10000)
        #time.sleep(.1)
        if bias is None:
            bias = np.arange(bias_high, bias_low, -bias_step)
            
        overbias_wait = 5.
        cool_wait = 10.
        if overbias:
            self.overbias_tes(bias_group, overbias_wait=overbias_wait, 
                tes_bias=np.max(bias), cool_wait=cool_wait,
                high_current_mode=high_current_mode,
                overbias_voltage=overbias_voltage)

        self.log('Staring to take IV.', self.LOG_USER)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        self.set_tes_bias_bipolar(bias_group, bias[0])
        time.sleep(1)

        datafile = self.stream_data_on(band)
        self.log('writing to {}'.format(datafile))

        for b in bias:
            self.log('Bias at {:4.3f}'.format(b))
            self.set_tes_bias_bipolar(bias_group, b)  
            time.sleep(wait_time)

        self.log('Done with TES bias ramp', self.LOG_USER)
        self.set_cryo_card_relays(2**16)

        self.stream_data_off(band)

        basename, _ = os.path.splitext(os.path.basename(datafile))
        np.save(os.path.join(basename + '_iv_bias.txt'), bias)

        iv_raw_data = {}
        iv_raw_data['bias'] = bias
        iv_raw_data['band'] = band
        iv_raw_data['bias group'] = bias_group
        iv_raw_data['channels'] = channels
        iv_raw_data['datafile'] = datafile
        iv_raw_data['basename'] = basename
        iv_raw_data['output_dir'] = self.output_dir
        iv_raw_data['plot_dir'] = self.plot_dir
        fn_iv_raw_data = os.path.join(self.output_dir, basename + 
            '_iv_raw_data.npy')
        np.save(os.path.join(self.output_dir, fn_iv_raw_data), iv_raw_data)

        self.analyze_slow_iv_from_file(fn_iv_raw_data, make_plot=make_plot,
            show_plot=show_plot, save_plot=save_plot, R_sh=325e-6, 
            high_current_mode=high_current_mode, rn_accept_min=rn_accept_min,
            rn_accept_max=rn_accept_max)

    def analyze_slow_iv_from_file(self, fn_iv_raw_data, make_plot=True,
        show_plot=False, save_plot=True, R_sh=.0029, high_current_mode=False,
        rn_accept_min = 1e-3, rn_accept_max = 1., phase_excursion_min=3.):
        """
        """
        self.log('Analyzing from file: {}'.format(fn_iv_raw_data))

        iv_raw_data = np.load(fn_iv_raw_data).item()
        bias = iv_raw_data['bias']
        band = iv_raw_data['band']
        bias_group = iv_raw_data['bias group']
        channels = iv_raw_data['channels']
        datafile = iv_raw_data['datafile']
        basename = iv_raw_data['basename']
        output_dir = iv_raw_data['output_dir']
        plot_dir = iv_raw_data['plot_dir']

        ivs = {}
        ivs['bias'] = bias

        # timestamp, phase = self.read_stream_data(datafile)
        # phase *= 1.443
        
        rn_list = []
        phase_excursion_list = []
        for c, ch in enumerate(channels):
            self.log('Analyzing channel {}'.format(ch))
            # timestamp, I, Q = self.read_stream_data(datafile)
            # phase = self.iq_to_phase(I[ch], Q[ch]) * 1.443
            timestamp, phase = self.read_stream_data_gcp_save(datafile, ch)
            ch_idx = ch
            # phase_ch = phase[ch_idx]
         
            phase_excursion = max(phase) - min(phase)
            # don't analyze channels with a small phase excursion; these are probably just noise
            if phase_excursion < phase_excursion_min:
                continue
            phase_excursion_list.append(phase_excursion)

            if make_plot:
                import matplotlib.pyplot as plt
                plt.rcParams["patch.force_edgecolor"] = True
                
                if not show_plot:
                    plt.ioff()

                fig, ax = plt.subplots(1, sharex=True)

                ax.plot(phase)
                ax.set_xlabel('Sample Num')
                ax.set_ylabel('Phase [pA]')

                ax.set_title('Band {}, Group {}, Ch {:03}'.format(band,
                    bias_group, ch))
                plt.tight_layout()

                plot_name = basename + \
                    '_IV_stream_b{}_g{}_ch{:03}.png'.format(band,bias_group,ch)
                if save_plot:
                    # self.log('Saving IV plot to {}'.format(os.path.join(plot_dir, plot_name)))
                    plt.savefig(os.path.join(plot_dir, plot_name), 
                        bbox_inches='tight', dpi=300)
                if not show_plot:
                    plt.close()

            r, rn, idx = self.analyze_slow_iv(bias, phase, 
                basename=basename, band=band, channel=ch, make_plot=make_plot, 
                show_plot=show_plot, save_plot=save_plot, plot_dir=plot_dir,
                R_sh = R_sh, high_current_mode = high_current_mode,
                bias_group=bias_group)
            try:
                if rn <= rn_accept_max and rn >= rn_accept_min:
                    rn_list.append(rn)
            except:
                self.log('fitted rn is not float')
            ivs[ch] = {
                'R' : r,
                'Rn' : rn,
                'idx': idx
            }

        np.save(os.path.join(output_dir, basename + '_iv'), ivs)

        if make_plot:
            import matplotlib.pyplot as plt
            if not show_plot:
                plt.ioff()
            plt.figure()
            plt.hist(rn_list)
            plt.xlabel('r_n')
            plt.title('%s, band %i, group %i: %i btwn. %.3e and %.3e Ohm' % \
                          (basename,band,bias_group,len(rn_list),\
                               rn_accept_min,rn_accept_max))
            plot_filename = os.path.join(plot_dir,'%s_IV_rn_hist.png' % (basename))
            plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
            if not show_plot:
                plt.close()

            plt.figure()
            plt.hist(phase_excursion_list, 
                bins=np.logspace(np.floor(np.log10(
                    np.min(phase_excursion_list))),
                np.ceil(np.log10(np.max(phase_excursion_list))),20))
            plt.xlabel('phase excursion')
            plt.ylabel('number of channels')
            plt.title('%s, band %i, group %i: %i with phase excursion > %.3e' % (basename,band,bias_group,len(phase_excursion_list),phase_excursion_min))
            plt.xscale('log')
            phase_hist_filename = os.path.join(plot_dir,'%s_IV_phase_excursion_hist.png' % (basename))
            plt.savefig(phase_hist_filename,bbox_inches='tight',dpi=300)
            if not show_plot:
                plt.close()

    def analyze_slow_iv(self, v_bias, resp, make_plot=True, show_plot=False,
        save_plot=True, basename=None, band=None, channel=None, R_sh=.003,
        plot_dir = None,high_current_mode = False,bias_group = None,**kwargs):
        """
        Analyzes the IV curve taken with slow_iv()

        Args:
        -----
        v_bias (float array): The commanded bias in voltage. Length n_steps
        resp (float array): The TES response. Of length n_pts (not the same 
            as n_steps

        Returns:
        --------
        R (float array): 
        R_n (float): 
        idx (int array): 
        R_sh (float): Shunt resistance
        """
        n_pts = len(resp)
        n_step = len(v_bias)

        step_size = float(n_pts)/n_step  # The number of samples per V_bias step

        resp_bin = np.zeros(n_step)

        #r_inline = 26.66e3 # for K7; estimate from Dan (Oct. 10, 2018)
        # r_inline = 8.e3 # for FP Run 28; estimate from SWH (Oct. 19, 2018)
        r_inline = self.bias_line_resistance
        if high_current_mode:
            # high-current mode generates higher current by decreases the in-line resistance
            r_inline /= self.high_low_current_ratio
        i_bias = 1.0E6 * v_bias / r_inline  # Total line impedance and uA

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
            #return None, None, None
            sc_idx = 5
        # index of the start of the normal branch
        nb_idx = n_step-5
        for i in np.arange(n_step-50, sc_idx, -1):
            if d_resp[i] > 0:
                nb_idx = i
                break

        norm_fit = np.polyfit(i_bias[nb_idx:], resp_bin[nb_idx:], 1)
        if norm_fit[0] < 0:  # Check for flipped polarity
            resp_bin = -1 * resp_bin
            norm_fit = np.polyfit(i_bias[nb_idx:], resp_bin[nb_idx:], 1)

        resp_bin -= norm_fit[1]  # now in real current units
        print(i_bias)
        print(resp_bin)

        sc_fit = np.polyfit(i_bias[:sc_idx], resp_bin[:sc_idx], 1)

        R = R_sh * (i_bias/(resp_bin) - 1)
        R_n = np.mean(R[nb_idx:])

        self.log('make_plot {}'.format(make_plot))
        if make_plot:
            fig, ax = plt.subplots(2, sharex=True)
            ax[0].plot(i_bias, resp_bin, '.')
            #ax[0].plot((i_bias[1:]+i_bias[:-1])/2, d_resp, 'r.')
            ax[0].set_ylabel(r'$I_{TES}$ $[\mu A]$')

            ax[0].plot(i_bias, norm_fit[0] * i_bias , linestyle='--', color='k')  

            ax[0].plot(i_bias[:sc_idx], 
                sc_fit[0] * i_bias[:sc_idx] + sc_fit[1], linestyle='--', 
                color='r')

            # Highlight the transition
            ax[0].axvspan(d_i[sc_idx], d_i[nb_idx], color='k', alpha=.15)
            ax[1].axvspan(d_i[sc_idx], d_i[nb_idx], color='k', alpha=.15)

            ax[0].text(.95, .04, 'SC slope: {:3.2f}'.format(sc_fit[0]), 
                transform=ax[0].transAxes, fontsize=12, 
                horizontalalignment='right')

            ax[1].plot(i_bias, R/R_n, '.')
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

            # Make top label in volts
            axt = ax[0].twiny()
            axt.set_xlim(ax[0].get_xlim())
            ib_max = np.max(i_bias)
            ib_min = np.min(i_bias)
            delta = float(ib_max - ib_min)/5
            vb_max = np.max(v_bias)
            vb_min = np.min(v_bias)
            delta_v = float(vb_max - vb_min)/5
            axt.set_xticks(np.arange(ib_min, ib_max+delta, delta))
            axt.set_xticklabels(['{:3.2f}'.format(x) for x in 
                np.arange(vb_min, vb_max+delta_v, delta_v)])
            axt.set_xlabel(r'$Commanded V_{b}$ [V]')

            if band is not None and channel is not None and bias_group is not None:
                fig.suptitle('Band {}, Group {}, Ch {:03}'.format(band, 
                    bias_group, channel))

            if basename is not None:
                ax[0].text(.95, .88, basename , transform=ax[0].transAxes, 
                    fontsize=12, horizontalalignment='right')

            if save_plot:
                if basename is None:
                    basename = self.get_timestamp()
                plot_name = basename + \
                    '_IV_curve_b{}_g{}_ch{:03}.png'.format(band, bias_group, channel)
                if plot_dir == None:
                    plot_dir = self.plot_dir
                plot_filename = os.path.join(plot_dir, plot_name)
                self.log('Saving IV plot to:{}'.format(plot_filename))
                plt.savefig(plot_filename,bbox_inches='tight', dpi=300)
            if show_plot:
                plt.show()
            else:
                plt.close()

        return R, R_n, np.array([sc_idx, nb_idx])

        
