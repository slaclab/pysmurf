#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf debug module - SmurfIVMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/debug/smurf_iv.py
# Created    : 2018-09-17
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import numpy as np
from pysmurf.client.base import SmurfBase
import time
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as Colors

class SmurfIVMixin(SmurfBase):

    def slow_iv_all(self, bias_groups=None, wait_time=.1, bias=None,
                    bias_high=1.5, bias_low=0, bias_step=.005,
                    show_plot=False, overbias_wait=2., cool_wait=30,
                    make_plot=True, save_plot=True, plotname_append='',
                    channels=None, band=None,
                    high_current_mode=True, overbias_voltage=8.,
                    grid_on=True, phase_excursion_min=3.):
        """
        Steps the TES bias down slowly. Starts at bias_high to bias_low with
        step size bias_step. Waits wait_time between changing steps.
        If this analyzes the data, the outputs are stored to output_dir.
        Opt Args:
        ---------
        bias_groups (np array): which bias groups to take the IV on. defaults
            to the groups in the config file
        wait_time (float): The amount of time between changing TES biases in
            seconds. Default .1 sec.
        bias (float array): A float array of bias values. Must go high to low.
        bias_high (int): The maximum TES bias in volts. Default 19.9
        bias_low (int): The minimum TES bias in volts. Default 0
        bias_step (int): The step size in volts. Default .1
        overbias_wait (float) : The time to stay in the overbiased state in
            seconds. The default is 2 sec.
        cool_wait (float) : The time to stay in the low current state after
            overbiasing before taking the IV.
        make_plot (bool) : Whether to make plots. Default True
        save_plot (bool) : Whether to save the plot. Default True.
        plotname_append (string): Appended to the default plot filename. Default ''.
        channels (int array) : A list of channels to make plots
        band (int array) : The bands to analyze
        high_current_mode (bool) : The current mode to take the IV in.
        overbias_voltage (float) : The voltage to set the TES bias in the
            overbias stage.
        grid_on (bool) : Grids on plotting. This is Aris fault.
        phase_excursion_min (float) : The minimum phase excursion required for
            making plots.
        """

        n_bias_groups = self._n_bias_groups

        if bias_groups is None:
            bias_groups = self.all_groups
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

        bias_group_bool = np.zeros((n_bias_groups,)) # hard coded to have 8 bias groups
        bias_group_bool[bias_groups] = 1 # only set things on the bias groups that are on

        self.set_tes_bias_bipolar_array(bias[0] * bias_group_bool)
        time.sleep(wait_time) # loops are in pyrogue now, which are faster?

        datafile = self.stream_data_on()
        self.log('writing to {}'.format(datafile))

        for b in bias:
            self.log('Bias at {:4.3f}'.format(b))
            self.set_tes_bias_bipolar_array(b * bias_group_bool)
            time.sleep(wait_time) # loops are now in pyrogue, so no division

        self.stream_data_off()
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
        self.log('Writing IV metadata to {}.'.format(fn_iv_raw_data))

        path = os.path.join(self.output_dir, fn_iv_raw_data)
        np.save(path, iv_raw_data)
        self.pub.register_file(path, 'iv_raw', format='npy')

        R_sh=self.R_sh
        self.analyze_slow_iv_from_file(fn_iv_raw_data, make_plot=make_plot,
            show_plot=show_plot, save_plot=save_plot,
            plotname_append=plotname_append, R_sh=R_sh, grid_on=grid_on,
            phase_excursion_min=phase_excursion_min,chs=channels,band=band)


    def partial_load_curve_all(self, bias_high_array, bias_low_array=None,
            wait_time=0.1, bias_step=0.1, show_plot=False, analyze=True,
            make_plot=True, save_plot=True, channels=None, overbias_voltage=None,
            overbias_wait=1.0, phase_excursion_min=1.):
        """
        Take a partial load curve on all bias groups. Function will step
        up to bias_high value, then step down. Will NOT change TES bias
        relay (ie, will stay in high current mode if already there). Will
        hold at the bias_low point at the end, so different bias groups
        may have different length ramps.

        Args:
        -----
        bias_high_array (float array): (n_bias_groups,) array of voltage biases, in
          bias group order
        Opt Args:
        -----
        bias_low_array (float array): (n_bias_groups,) array of voltage biases, in
          bias group order. Defaults to whatever is currently set
        wait_time (float): Time to wait at each commanded bias value.
            Default 0.1
        bias_step (float): Interval size to step the commanded voltage bias.
          Default 0.1
        show_plot (bool): whether to show plots. Default False.
        make_plot (bool): whether to generate plots. Default True.
        analyze (bool): whether to analyze the data. Default True.
        save_plot (bool): whether to save generated plots. Default True.
        channels (int array): which channels to analyze. Default to anything
          that exceeds phase_excursion_min
        overbias_voltage (float): value in V at which to overbias. If None,
          won't overbias. If value is set, uses high current mode.
        overbias_wait (float): if overbiasing, time to hold the overbiased
          voltage in high current mode
        phase_excursion_min (float): minimum change in phase to be analyzed,
          defined as abs(phase_max - phase_min). Default 1, units radians.
        """

        original_biases = self.get_tes_bias_bipolar_array()

        if bias_low_array is None: # default to current values
            bias_low_array = original_biases

        if overbias_voltage is not None: # only overbias if this is set
            if self.high_current_mode_bool:
                tes_bias = 2. # Probably should actually move this over to
            else: #overbias_tes_all
                tes_bias = 19.9
            self.overbias_tes_all(overbias_voltage=overbias_voltage,
            overbias_wait=overbias_wait, tes_bias=tes_bias,
            high_current_mode=self.high_current_mode_bool)

        ### make arrays of voltages to set ###
        # first, figure out the length of the longest sweep
        bias_diff = bias_high_array - bias_low_array
        max_spread_idx = np.ravel(np.where(bias_diff == max(bias_diff)))[0]
        max_length = int(np.ceil(bias_diff[max_spread_idx]/bias_step))

        # second, initialize bias array of size n_bias_groups by n_steps_max
        bias_sweep_array = np.zeros((len(bias_high_array), max_length + 1))
        # to be honest not sure if I need the +1 but adding it just in case

        for idx in np.arange(len(bias_high_array)): # for each bias group
            # third, re-initialize each bias group to be the low point
            bias_sweep_array[idx,:] = bias_low_array[idx]

            # fourth, override the first j entries of each with the sweep
            sweep = np.arange(bias_high_array[idx], bias_low_array[idx]-bias_step,
                -bias_step)
            bias_sweep_array[idx, 0:len(sweep)] = sweep

        self.log('Starting to take partial load curve.', self.LOG_USER)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        datafile = self.stream_data_on()
        self.log('writing to {}'.format(datafile))

        # actually set the arrays
        for step in np.arange(np.shape(bias_sweep_array)[1]):
            self.set_tes_bias_bipolar_array(bias_sweep_array[:,step])
            time.sleep(wait_time) # divide by something here? unclear.

        # explicitly set back to the original biases
        self.set_tes_bias_bipolar_array(original_biases)

        self.stream_data_off()
        self.log('Done with TES bias ramp', self.LOG_USER)

        # should I be messing with lmsGain?

        # save and analyze
        basename, _ = os.path.splitext(os.path.basename(datafile))

        path = os.path.join(self.output_dir, basename + '_plc_bias_all')
        np.save(path, bias_sweep_array)
        self.pub.register_file(path, 'plc_bias', format='npy')

        plc_raw_data = {}
        plc_raw_data['bias'] = bias_sweep_array
        plc_raw_data['bias group'] = np.where(original_biases != 0)
        plc_raw_data['datafile'] = datafile
        plc_raw_data['basename'] = basename
        plc_raw_data['output_dir'] = self.output_dir
        plc_raw_data['plot_dir'] = self.plot_dir
        fn_plc_raw_data = os.path.join(self.output_dir, basename +
            '_plc_raw_data.npy')

        path = os.path.join(self.output_dir, fn_plc_raw_data)
        np.save(path, plc_raw_data)
        self.pub.register_file(path, 'plc_raw', format='npy')

        if analyze:
            self.analyze_plc_from_file(fn_plc_raw_data, make_plot=make_plot,
                show_plot=show_plot, save_plot=save_plot, R_sh=self.R_sh,
                high_current_mode=self.high_current_mode_bool,
                phase_excursion_min=phase_excursion_min, channels=channels)


    def analyze_slow_iv_from_file(self, fn_iv_raw_data, make_plot=True,
                                  show_plot=False, save_plot=True,
                                  plotname_append='', R_sh=None,
                                  phase_excursion_min=3., grid_on=False,
                                  R_op_target=0.007, pA_per_phi0=None,
                                  chs=None, band=None, datafile=None,
                                  plot_dir=None, bias_line_resistance=None):
        """
        Function to analyze a load curve from its raw file. Can be used to
          analyze IV's/generate plots separately from issuing commands.
        Args:
        -----
        fn_iv_raw_data (str): *_iv_raw_data.npy file to analyze
        Opt Args:
        ---------
        make_plot (bool): Defaults True. Usually this is the slowest part.
        show_plot (bool): Defaults False.
        save_plot (bool): Defaults True.
        plotname_append (string): Appended to the default plot filename. Default ''.
        phase_excursion_min (float): abs(max - min) of phase in radians. Analysis
          ignores any channels without this phase excursion. Default 3.
        grid_on (bool): Whether to draw the grid on the PR plot. Defaults False
          Defaults true.
        R_op_target (float): Target operating resistance. Function will
          generate a histogram indicating bias voltage needed to achieve
          this value.
        chs (int array): Which channels to analyze. Defaults to all
          the channels that are on and exceed phase_excursion_min
        data_path (str) : The full path to the data. This is used for offline mode
          where the data was copied to a new directory. The directory is usually
          loaded from the .npy file, and this overwrites it.
        plot_dir (str) : The full path to the plot directory. This is usually
            loaded with the input numpy dictionary. This overwrites this.
        bias_line_resistance (float) : The resistance of the bias lines in Ohms.
            Default is None.
        """
        self.log('Analyzing from file: {}'.format(fn_iv_raw_data))

        # Extract data from dict
        iv_raw_data = np.load(fn_iv_raw_data, allow_pickle=True).item()
        # bias = iv_raw_data['bias']
        high_current_mode = iv_raw_data['high_current_mode']
        bias_group = iv_raw_data['bias group']

        # This overwrites the datafile path, which is usually loaded
        # from the .npy file
        if datafile is None:
            datafile = iv_raw_data['datafile']
        else:
            self.log(f"Using input datafile {datafile}")

        mask = self.make_mask_lookup(datafile.replace('.dat','_mask.txt'))
        bands, chans = np.where(mask != -1)

        basename = iv_raw_data['basename']
        output_dir = iv_raw_data['output_dir']

        if plot_dir is None:
            plot_dir = iv_raw_data['plot_dir']

        # Load raw data
        timestamp, phase_all, mask, tes_bias = self.read_stream_data(datafile,
            return_tes_bias=True)
        bands, chans = np.where(mask != -1)

        # IV output dictionary
        ivs = {}
        ivs['high_current_mode'] = high_current_mode
        for b in np.unique(bands):
            ivs[b] = {}

        # Extract V_bias
        v_bias = 2 * tes_bias[bias_group[0]] * self._rtm_slow_dac_bit_to_volt

        rn_list = []
        phase_excursion_list = []
        v_bias_target_list = []
        p_trans_list = []
        si_target_list = []
        v_tes_target_list = []
        for c, (b, ch) in enumerate(zip(bands,chans)):
            if (chs is not None) and (ch not in chs):
                self.log(f'Not in desired channel list: skipping band {b} ch {ch}')
                continue
            elif (band is not None) and (b != band):
                self.log(f'Not in desired band: skipping band {b} ch. {ch}')
                continue

            self.log('Analyzing band {} channel {}'.format(b,ch))

            ch_idx = mask[b, ch]
            phase = phase_all[ch_idx]

            phase_excursion = max(phase) - min(phase)

            if phase_excursion < phase_excursion_min:
                self.log('Skipping channel {}:  phase excursion < min'.format(ch))
                continue
            phase_excursion_list.append(phase_excursion)

            if make_plot:
                plt.rcParams["patch.force_edgecolor"] = True

                if not show_plot:
                    plt.ioff()

                fig, ax = plt.subplots(1)

                ax.plot(phase)
                ax.set_xlabel('Sample Num')
                ax.set_ylabel('Phase [rad.]')
                if grid_on:
                    ax.grid()

                ax.set_title('Band {}, Group {}, Ch {:03}'.format(np.unique(band),
                    bias_group, ch))
                plt.tight_layout()

                bg_str = ""
                for bg in np.unique(bias_group):
                    bg_str = bg_str + str(bg)

                # Define plot name
                plot_name = basename + \
                    '_IV_stream_b{}ch{:03}_g{}.png'.format(b, ch, bg_str)
                # Optional append
                if len(plotname_append) > 0:
                    plot_name.replace('.png', f'_{plotname_append}.png')

                if save_plot:
                    plot_fn = os.path.join(plot_dir, plot_name)
                    plt.savefig(plot_fn, bbox_inches='tight', dpi=300)
                    self.pub.register_file(plot_fn, 'iv_stream', plot=True)
                if not show_plot:
                    plt.close()

            iv_dict = self.analyze_slow_iv(v_bias, phase,
                basename=basename, band=b, channel=ch, make_plot=make_plot,
                show_plot=show_plot, save_plot=save_plot, plot_dir=plot_dir,
                R_sh=R_sh, high_current_mode=high_current_mode,
                grid_on=grid_on, R_op_target=R_op_target,
                pA_per_phi0=pA_per_phi0,
                bias_line_resistance=bias_line_resistance)
            iv_dict['R']

            rn = iv_dict['R_n']
            p_trans = iv_dict['p_trans']
            v_bias_target = iv_dict['v_bias_target']
            si_target = iv_dict['si_target']
            v_tes_target = iv_dict['v_tes_target']

            if p_trans is not None and not np.isnan(p_trans):
                p_trans_list.append(p_trans)
            else:
                self.log('p_trans is not float')
                continue
            try:
                rn_list.append(rn)
            except BaseException:
                self.log('fitted rn is not float')
                continue
            v_bias_target_list.append(v_bias_target)
            si_target_list.append(si_target)
            v_tes_target_list.append(v_tes_target)
            ivs[b][ch] = iv_dict

        fn_iv_analyzed = basename + '_iv'

        path = os.path.join(output_dir, fn_iv_analyzed)
        self.log(f'Writing analyzed IV data to {path}')
        np.save(path, ivs)
        self.pub.register_file(path, 'iv', format='npy')

        v_bias_target_median = np.median(v_bias_target_list)
        rn_median = np.median(rn_list)
        ptrans_median = np.median(p_trans_list)
        si_target_median = np.median(si_target_list)
        v_tes_target_median = np.median(v_tes_target_list)
        si_goal = -1./v_tes_target_median

        if len(phase_excursion_list) == 0:
            self.log('phase excursion list length 0')
        elif make_plot:
            colors = []
            tableau = Colors.TABLEAU_COLORS
            for c in tableau:
                colors.append(tableau[c])
            if not show_plot:
                plt.ioff()
            color_hist = colors[0]
            color_median = colors[1]
            color_goal = colors[2]

            fig = plt.figure(figsize=(8,5))
            gs = GridSpec(2,2)
            ax_rn = fig.add_subplot(gs[0,0])
            ax_ptrans = fig.add_subplot(gs[0,1])
            ax_vbias = fig.add_subplot(gs[1,0])
            ax_si = fig.add_subplot(gs[1,1])

            rn_array_mOhm = np.array(rn_list)/1e-3
            ax_rn.hist(rn_array_mOhm,bins=20,color=color_hist)
            ax_rn.set_xlabel(r'$R_N$ [$\mathrm{m}\Omega$]')
            ax_rn.axvline(rn_median/1e-3,linestyle='--',color=color_median,
                label=r'Median = {:.0f}'.format(rn_median/1e-3) +
                r' $\mathrm{m}\Omega$')
            ax_rn.legend(loc='best')

            ax_vbias.hist(v_bias_target_list,bins=20,color=color_hist)
            ax_vbias.axvline(v_bias_target_median,linestyle = '--',
                        color=color_median,
                        label='Median = {:.2f} V'.format(v_bias_target_median))
            ax_vbias.set_xlabel(r'Commanded voltage bias [V] for $R = $' +
                '{:.0f}'.format(R_op_target/1e-3) +
                r' $\mathrm{m}\Omega$')
            ax_vbias.legend(loc='best')

            ax_ptrans.hist(p_trans_list,bins=20,color=color_hist)
            ax_ptrans.axvline(ptrans_median,linestyle='--',color=color_median,
                label=r'Median = {:.1f} pW'.format(ptrans_median))
            ax_ptrans.set_xlabel('In-transition electrical power [pW]')
            ax_ptrans.legend(loc='best')

            ax_si.hist(si_target_list,bins=20)
            ax_si.axvline(si_target_median,linestyle='--',color=color_median,
                label='Median = {:.2f}'.format(si_target_median) +
                r' $\mu\mathrm{V}^{-1}$')
            ax_si.axvline(si_goal,linestyle='--',color=color_goal,
                label=r'$-\mathrm{med}(V_\mathrm{TES})^{-1} = $' +
                '{:.2f}'.format(si_goal) +
                r' $\mu\mathrm{V}^{-1}$')
            ax_si.set_xlabel(r'Responsivity [$\mu\mathrm{V}^{-1}$] at $R = $' +
                '{:.0f}'.format(R_op_target/1e-3) +
                r' $\mathrm{m}\Omega$')
            plt.legend(loc='best')

            plt.tight_layout()
            fig.subplots_adjust(top=0.925)

            # Title
            plt.suptitle('{}, band {}, group{}'.format(basename,
                np.unique(band),bias_group))
            iv_hist_filename = os.path.join(plot_dir,
                f'{basename}_IV_hist{plotname_append}.png')

            # Save the figure
            plt.savefig(iv_hist_filename,bbox_inches='tight')
            self.pub.register_file(iv_hist_filename, 'iv_hist', plot=True)

            self.log('Saved IV histogram to {}'.format(iv_hist_filename))
            if not show_plot:
                plt.close()


    def analyze_slow_iv(self, v_bias, resp, make_plot=True, show_plot=False,
            save_plot=True, basename=None, band=None, channel=None, R_sh=None,
            plot_dir=None, high_current_mode=False, bias_group=None,
            grid_on=False, R_op_target=0.007, pA_per_phi0=None,
            bias_line_resistance=None, plotname_append='', **kwargs):
        """
        Analyzes the IV curve taken with slow_iv()

        Args:
        -----
        v_bias (float array): The commanded bias in voltage. Length n_steps
        resp (float array): The TES phase response in radians. Of length
            n_pts (not the same as n_steps).
        make_plot (bool) : Whether to make the plot. Default is True.
        show_plot (bool) : Whether to show the plot. Default is False.
        save_plot (bool) : Whether to save the plot. Default is True.
        basename (str) : The basename of the IV plot. If None, uses the current
            timestamp. Default is None.
        band (int) : The 500 MHz band the data was taken in. This is only for
            plotting . Default None.
        channel (int) : The SMuRF channel. Only used for plotting. Default
            is None.
        R_sh (int) : The shunt resistance in ohms. If not supplied, will try
            to read from config file.
        plot_dir (str) : Path to the plot directory where plots are to be saved.
            If None, uses self.plot_dir. Default is None.
        high_current_mode (bool) : Whether the data was taken in high current
            mode. This is important for knowing what current actually enters
            the cryostat.
        grid_on (bool) : Whether to plot with grids on. Default is False.
        pA_per_phi0 (float) : The conversion for phi0 to pA. If None, attempts
            to read it from the config file. Default is None.
        bias_line_resistance (float) : The resistance of the bias lines in
            Ohms. If None, reads from config. Default is None.
        plotname_append (str) : An optional string to append the plot names.

        Returns:
        --------
        R (float array):
        R_n (float):
        idx (int array):
        R_sh (float): Shunt resistance
        """
        v_bias = np.abs(v_bias)

        if R_sh is None:
            R_sh=self.R_sh

        if pA_per_phi0 is None:
            pA_per_phi0 = self.pA_per_phi0
        resp *= pA_per_phi0/(2.*np.pi*1e6) # convert phase to uA

        # n_pts = len(resp)
        step_loc = np.where(np.diff(v_bias))[0]

        if step_loc[0] != 0:
            step_loc = np.append([0], step_loc)  # starts from zero
        # step_size = np.diff(v_bias)[step_loc]
        n_step = len(step_loc) - 1

        resp_bin = np.zeros(n_step)
        v_bias_bin = np.zeros(n_step)
        i_bias_bin = np.zeros(n_step)

        if bias_line_resistance is None:
            r_inline = self.bias_line_resistance
        else:
            r_inline = bias_line_resistance

        if high_current_mode:
            # high-current mode generates higher current by decreases the
            # in-line resistance
            r_inline /= self.high_low_current_ratio
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
        v_bias_target = v_bias[i_R_op]
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

            title = ""
            plot_name = "IV_curve"
            if band is not None:
                title = title + 'Band {}'.format(band)
                plot_name = plot_name + '_b{}'.format(band)
            if bias_group is not None:
                title = title + ' BG {}'.format(bias_group)
                plot_name = plot_name + '_bg{}'.format(bias_group)
            if channel is not None:
                title = title + ' Ch {:03}'.format(channel)
                plot_name = plot_name + '_ch{:03}'.format(channel)
            if basename is None:
                basename = self.get_timestamp()
            if band is not None and channel is not None:
                if self.offline:
                    self.log("Offline mode does not know resonator frequency." +
                        " Not adding to title.")
                else:
                    title += ', {:.2f} MHz'.format(self.channel_to_freq(band, channel))
            title += r', $R_\mathrm{sh}$ = ' + '${:.2f}$ '.format(R_sh*1.0E3) + \
                r'$\mathrm{m}\Omega$'
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

            ax_ii.plot(i_bias_bin, norm_fit[0] * i_bias_bin , linestyle='--',
                       color=color_norm, label=r'$R_N$' +
                       '  = ${:.0f}$'.format(R_n/1e-3) +
                       r' $\mathrm{m}\Omega$')
            ax_ii.plot(i_bias_bin[:sc_idx],
                sc_fit[0] * i_bias_bin[:sc_idx] + sc_fit[1], linestyle='--',
                color=color_sc, label=r'$R_L$' +
                    ' = ${:.0f}$'.format(R_L/1e-6) +
                    r' $\mu\mathrm{\Omega}$')

            label_target = r'$R = {:.0f}$ '.format(R_op_target/1e-3) + \
                r'$\mathrm{m}\Omega$'
            label_rfrac = '{:.2f}-{:.2f}'.format(R_frac_min,
                R_frac_max) + r'$R_N$'

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
            axt.set_xticks(np.arange(ib_min, ib_max+delta, delta))
            axt.set_xticklabels(['{:.2f}'.format(x) for x in
                np.arange(vb_min, vb_max+delta_v, delta_v)])
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
                self.log('Saving IV plot to {}'.format(plot_filename))
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

    def analyze_plc_from_file(self, fn_plc_raw_data, make_plot=True, show_plot=False,
            save_plot=True, R_sh=None, high_current_mode=None, phase_excursion_min=1.,
            channels=None):
        """
        Function to analyze a partial load curve from its raw file. Basically
        the same as the slow_iv analysis but without fitting the superconducting
        branch.
        Args:
        -----
        fn_plc_raw_data (str): *_plc_raw_data.npy file to analyze
        Opt Args:
        -----
        make_plot (bool): Defaults True. This is slow.
        show_plot (bool): Defaults False.
        save_plot (bool): Defaults True.
        R_sh (float): shunt resistance; defaults to the value in the config file
        high_current_mode (bool): Whether to perform analysis assuming that commanded
          voltages were in high current mode. Defaults to the value in the config file.
        phase_excursion_min (float): abs(max-min) of phase in radian. Analysis will
          ignore any channels that do not meet this criterion. Default 1.
        channels (int array): which channels to analyze. Defaults to all channels that
          are on and exceed phase_excursion_min
        """

        self.log('Analyzing plc from file: {}'.format(fn_plc_raw_data))

        plc_raw_data = np.load(fn_plc_raw_data).item()
        #bias_sweep_array = plc_raw_data['bias']
        bias_group = plc_raw_data['band']
        datafile = plc_raw_data['datafile']
        basename = plc_raw_data['basename']
        #output_dir = plc_raw_data['output_dir']
        plot_dir = plc_raw_data['plot_dir']

        timestamp, phase_all, mask = self.read_stream_data(datafile)


        band, chans = np.where(mask != -1) #are these masks secretly the same?

        #rn_list = []
        phase_excursion_list = []
        #v_bias_target_list = []
        #p_trans_list = []

        for c, (b, ch) in enumerate(zip(band, chans)):
            if (channels is not None) and (ch not in channels):
                continue
            self.log('Analyzing band {} channel {}'.format(b,ch))
            ch_idx = mask[b,ch]
            phase = phase_all[ch_idx]

            phase_excursion = max(phase) - min(phase)
            if phase_excursion < phase_excursion_min:
                self.log('Skipping ch {}: phase excursion < min'.format(ch))
                continue
            phase_excursion_list.append(phase_excursion)

            # need to figure out how to do analysis b/c channels have different biases :(
            # would work once we had a permanent lookup table of ch to bias group...

            if make_plot: # make the timestream plot
                plt.rcParams["patch.force_edgecolor"] = True

                if not show_plot:
                    plt.ioff()

                fig, ax = plt.subplots(1, sharex=True)
                ax.plot(phase)
                ax.set_xlabel('Sample Num')
                ax.set_ylabel('Phase [rad.]')
                # no grid on for you, Ari
                ax.set_title('Band {}, Group {}, Ch {:03}'.format(np.unique(band),
                    bias_group, ch)) # this is not going to be very useful...
                plt.tight_layout()

                if save_plot:
                    bg_str = ""
                    for bg in np.unique(bias_group):
                        bg_str = bg_str + str(bg)

                    plot_name = basename + \
                        'plc_stream_b{}_g{}_ch{:03}.png'.format(b, bg_str, ch)
                    path = os.path.join(plot_dir, plot_name)
                    plt.savefig(path, bbox_inches='tight', dpi=300)
                    self.pub.register_file(path, 'plc_stream', plot=True)

                if not show_plot:
                    plt.close()

    def estimate_opt_eff(self, iv_fn_hot, iv_fn_cold,t_hot=293.,t_cold=77.,
            channels = None, dPdT_lim=(0.,0.5)):
        """
        Estimate optical efficiency between two sets of load curves. Returns
          per-channel plots and a histogram.
        Args:
        iv_fn_hot (str): timestamp/filename of load curve taken at higher temp
        iv_fn_cold (str): timestamp/filename of load curve taken at cooler temp
        Opt Args:
        t_hot (float): temperature in K of hotter load curve. Defaults to 293.
        t_cold (float): temperature in K of cooler load curve. Defaults to 77.
        channels (int array): which channels to analyze. Defaults to the ones
          populated in the colder load curve
        dPdT_lim(len 2 tuple): min, max allowed values for dPdT. If calculated
          val is outside this range, channel is excluded from histogram and
          added to outliers. Defaults to min=0pW/K, max=0.5pW/K.
        """
        ivs_hot = np.load(iv_fn_hot).item()
        ivs_cold = np.load(iv_fn_cold).item()

        iv_fn_raw_hot = iv_fn_hot.split('.')[0] + '_raw_data.npy'
        iv_fn_raw_cold = iv_fn_cold.split('.')[0] + '_raw_data.npy'

        ivs_raw_hot = np.load(iv_fn_raw_hot).item()
        ivs_raw_cold = np.load(iv_fn_raw_cold).item()

        basename_hot = ivs_raw_hot['basename']
        basename_cold = ivs_raw_cold['basename']

        #band = ivs_raw_hot['band']
        #assert ivs_raw_cold['band'] == band, \
        #    'Files must contain IVs from the same band'
        group = ivs_raw_hot['bias group']
        assert ivs_raw_cold['bias group'], \
            'Files must contain IVs from the same bias group'

        plot_dir = self.plot_dir

        dT = t_hot - t_cold
        dPdT_list = []
        n_outliers = 0
        for ch in ivs_hot:
            if channels is not None:
                if ch not in channels:
                    continue
            elif not isinstance(ch,np.int64):
                continue

            if ch not in ivs_cold:
                continue

            P_hot = ivs_hot[ch]['P']
            P_cold = ivs_cold[ch]['P']
            R_hot = ivs_hot[ch]['R']
            R_cold = ivs_cold[ch]['R']
            Ptrans_hot = ivs_hot[ch]['Ptrans']
            Ptrans_cold = ivs_cold[ch]['Ptrans']
            if Ptrans_hot is None or Ptrans_cold is None:
                print(f'Missing in-transition electrical powers for Ch. {ch}')
                continue
            dPdT = (Ptrans_cold - Ptrans_hot)/dT
            self.log(f'Group {group}, Ch {ch:03}: dP/dT = {dPdT:.3f} pW/K')
            if dPdT_lim is not None:
                if dPdT >= dPdT_lim[0] and dPdT <= dPdT_lim[1]:
                    dPdT_list.append(dPdT)
                else:
                    n_outliers += 1
            else:
                dPdT_list.append(dPdT)

            # Make figure
            fig_pr,ax_pr = plt.subplots(1, sharex=True)

            # Labels
            ax_pr.set_xlabel(r'$R_\mathrm{TES}$ [$\Omega$]')
            ax_pr.set_ylabel(r'$P_\mathrm{TES}$ [pW]')
            label_hot = f'{basename_hot}: {t_hot:.0f} K'
            label_cold = f'{basename_cold}: {t_cold:.0f} K'
            ax_pr.axhline(y=Ptrans_hot, linestyle='--', color='b')
            ax_pr.axhline(y=Ptrans_cold, linestyle='--', color='r')

            # Plot data
            ax_pr.plot(R_hot,P_hot,label=label_hot,color='b')
            ax_pr.plot(R_cold,P_cold,label=label_cold,color='r')
            ax_pr.legend(loc='best')
            fig_pr.suptitle(f'Group {group}, Ch {ch:03}: dP/dT = {dPdT:.3f} pW/K')
            ax_pr.grid()

            # Plot name
            plot_name = basename_hot + '_' + basename_cold + f'_optEff_g{group}_ch{ch:03}.png'
            plot_filename = os.path.join(plot_dir, plot_name)
            self.log('Saving optical-efficiency plot to {}'.format(plot_filename))
            plt.savefig(plot_filename, bbox_inches='tight', dpi=300)

            # Publish
            self.pub.register_file(plot_filename, 'opt_efficiency', plot=True)
            plt.close()

        plt.figure()
        plt.hist(dPdT_list,edgecolor='k')
        plt.xlabel('dP/dT [pW/K]')
        plt.grid()
        dPdT_median = np.median(dPdT_list)
        plt.title(f'Group {group}, median = {dPdT_median:.3f} pW/K '+
            f'({n_outliers} outliers not plotted)')
        plot_name = basename_hot + '_' + basename_cold + '_dPdT_hist_g{}.png'.format(group)
        hist_filename = os.path.join(plot_dir,plot_name)
        self.log('Saving optical-efficiency histogram to {}'.format(hist_filename))
        plt.savefig(hist_filename, bbox_inches='tight', dpi=300)
        self.pub.register_file(hist_filename, 'opt_efficiency', plot=True)
        plt.close()


    def estimate_bias_voltage(self, iv_file, target_r_frac=.5,
                              normal_resistance=None,
                              normal_resistance_frac=.25,
                              show_plot=False, save_plot=True,
                              make_plot=True):
        """
        Attempts to estimate the bias point per bias group.
        """
        # Load IV summary file and raw dat file
        iv = np.load(iv_file).item()
        iv_raw_dat = np.load(iv_file.replace('_iv.npy',
                                             '_iv_raw_data.npy')).item()

        # Assume all ints in keys are bands
        band = np.array([k for k in iv.keys()
                          if np.issubdtype(type(k), np.integer)],
                         dtype=int)
        v_max = 0
        
        # Get bias groups - first index bg, second index band
        bias_group = iv_raw_dat['bias group']
        bg_list = {}
        target_bias_voltage = np.zeros(len(bias_group))
        
        for bg in bias_group:
            bg_list[bg] = {}
            for b in band:
                bg_list[bg][b] = self.get_group_list(b, bg)
                
        for bg in bias_group:
            v_bias = np.array([])
            R_n = np.array([])
            for b in band:
                channel = np.intersect1d(list(iv[b].keys()),
                                         bg_list[bg][b])
                for ch in channel:
                    R = iv[b][ch]['R_n']
                    if np.logical_and(R > normal_resistance * (1-normal_resistance_frac),
                                      R < normal_resistance * (1+normal_resistance_frac)):
                        R_frac = iv[b][ch]['R']/iv[b][ch]['R_n']
                        dR_frac = np.abs(R_frac - target_r_frac)
                        idx = np.where(dR_frac == np.min(dR_frac))
                        v_bias = np.append(v_bias, iv[b][ch]['v_bias'][idx])
                        R_n = np.append(R_n, iv[b][ch]['R_n'])
                    v_max = np.max(iv[b][ch]['v_bias'])

            if len(v_bias) > 0:
                target_bias_voltage[bg] = np.median(v_bias)
                    
            # Make summary plot
            if make_plot:
                fig, ax = plt.subplots(2, figsize=(4,5), sharex=True)
                for b in band:
                    channel = np.intersect1d(list(iv[b].keys()),
                                         bg_list[bg][b])
                    for ch in channel:
                        ax[0].plot(iv[b][ch]['v_bias'],
                                iv[b][ch]['R']*1.0E3,
                                color='k',alpha=.2)

                    # ax[0].set_ylim((0, ax[0].get_ylim()[1]))
                if len(R_n) > 0:
                    ax[0].set_ylim((0, np.max(R_n)*1.1E3))
                ax[0].set_ylabel('R [mOhm]')
                if len(v_bias) > 0:
                    ax[1].hist(v_bias,
                                 bins=np.arange(0, v_max, .1),
                               alpha=.5)

                    ax[0].axvline(target_bias_voltage[bg], color='r',
                                  linestyle='--')
                    ax[1].axvline(target_bias_voltage[bg], color='r',
                                  linestyle='--')
                    
                ax[1].set_xlabel(r'$V_{bias}$ [V]')
                ax[0].set_title(f'Bias Group {bg}', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.plot_dir,
                                         iv_raw_dat['basename'] +
                                         f'_estimate_bias_bg{bg}.png'),
                            bbox_inches='tight')

            if not show_plot:
                plt.close(fig)
        return target_bias_voltage
