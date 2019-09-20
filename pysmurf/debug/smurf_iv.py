import numpy as np
from pysmurf.base import SmurfBase
import time
import os,sys

class SmurfIVMixin(SmurfBase):

    def slow_iv(self, band, bias_group, wait_time=.25, bias=None, bias_high=19.9, 
        bias_low=0, bias_step=.1, show_plot=False, overbias_wait=5., cool_wait=30.,
        make_plot=True, save_plot=True, channels=None, high_current_mode=False,
        rn_accept_min=1e-3, rn_accept_max=1., overbias_voltage=19.9,
        gcp_mode=True, grid_on=True, phase_excursion_min=3.):
        """
        >>>NOTE: DEPRECATED. USE SLOW_IV_ALL WITH A SINGLE-ELEMENT ARRAY INSTEAD.<<<

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
        phase_excursion_min (int): The minimum phase excursion allowable
        """
        self.log("WARNING: I AM NOW DEPRICATED. USE slow_iv_all")
        # Look for good channels
        if channels is None:
            channels = self.which_on(band)

        if overbias_voltage != 0.:
            overbias = True
        else:
            overbias = False

        if bias is None:
            bias = np.arange(bias_high, bias_low-bias_step, -bias_step)
            
        if overbias:
            self.overbias_tes(bias_group, overbias_wait=overbias_wait, 
                tes_bias=np.max(bias), cool_wait=cool_wait,
                high_current_mode=high_current_mode,
                overbias_voltage=overbias_voltage)

        self.log('Turning lmsGain to 0.', self.LOG_USER)
        lms_gain = self.get_lms_gain(band)
        self.set_lms_gain(band, 0)

        self.log('Starting to take IV.', self.LOG_USER)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        self.set_tes_bias_bipolar(bias_group, bias[0])
        time.sleep(1)

        datafile = self.stream_data_on(gcp_mode=gcp_mode)
        self.log('writing to {}'.format(datafile))

        for b in bias:
            self.log('Bias at {:4.3f}'.format(b))
            #sys.stdout.write('\rBias at {:4.3f} V\033[K'.format(b))
            #sys.stdout.flush()
            self.set_tes_bias_bipolar(bias_group, b)  
            time.sleep(wait_time)
        #sys.stdout.write('\n')

        self.stream_data_off(gcp_mode=gcp_mode)

        self.log('Done with TES bias ramp', self.LOG_USER)

        self.log('Returning lmsGain to ' + str(lms_gain), self.LOG_USER)
        self.set_lms_gain(band, lms_gain)

        #self.set_cryo_card_relays(2**16)



        basename, _ = os.path.splitext(os.path.basename(datafile))
        outfn = os.path.join(self.output_dir, basename + '_iv_bias')

        np.save(outfn, bias)
        self.pub.register_file(outfn, 'iv_bias', format='npy')

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

        path = os.path.join(self.output_dir, fn_iv_raw_data)
        np.save(path, iv_raw_data)
        self.pub.register_file(path, 'iv_raw', format='npy')

        R_sh=self.R_sh
        self.analyze_slow_iv_from_file(fn_iv_raw_data, make_plot=make_plot,
            show_plot=show_plot, save_plot=save_plot, R_sh=R_sh, 
            high_current_mode=high_current_mode, rn_accept_min=rn_accept_min,
            rn_accept_max=rn_accept_max, gcp_mode=gcp_mode,grid_on=grid_on,
            phase_excursion_min=phase_excursion_min)

    def slow_iv_all(self, bias_groups=None, wait_time=.1, bias=None, 
                    bias_high=1.5, gcp_mode=True, bias_low=0, bias_step=.005, 
                    show_plot=False, overbias_wait=2., cool_wait=30,
                    make_plot=True, save_plot=True, channels=None, band=None,
                    high_current_mode=True, overbias_voltage=8., 
                    grid_on=True, phase_excursion_min=3.):
        """
        Steps the TES bias down slowly. Starts at bias_high to bias_low with
        step size bias_step. Waits wait_time between changing steps.


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
        """
        if bias_groups is None:
            bias_groups = self.all_groups

        if overbias_voltage != 0.:
            overbias = True
        else:
            overbias = False

        if bias is None:
            bias = np.arange(bias_high, bias_low-bias_step, -bias_step)

        if overbias:
            self.overbias_tes_all(bias_groups=bias_groups, 
                overbias_wait=overbias_wait, tes_bias=np.max(bias), 
                cool_wait=cool_wait, high_current_mode=high_current_mode,
                overbias_voltage=overbias_voltage)

        self.log('Turning lmsGain to 0.', self.LOG_USER)
        lms_gain2 = self.get_lms_gain(2) # just do this on both bands
        lms_gain3 = self.get_lms_gain(3) # should fix the hardcoding though -CY
        self.set_lms_gain(2, 0)
        self.set_lms_gain(3, 0)

        self.log('Starting to take IV.', self.LOG_USER)
        self.log('Starting TES bias ramp.', self.LOG_USER)


        self.log('Starting to take IV.', self.LOG_USER)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        bias_group_bool = np.zeros((8,)) # hard coded to have 8 bias groups
        bias_group_bool[bias_groups] = 1 # only set things on the bias groups that are on

        self.set_tes_bias_bipolar_array(bias[0] * bias_group_bool)
        time.sleep(wait_time) # loops are in pyrogue now, which are faster?

        datafile = self.stream_data_on(gcp_mode=gcp_mode)
        self.log('writing to {}'.format(datafile))

        for b in bias:
            self.log('Bias at {:4.3f}'.format(b))
            self.set_tes_bias_bipolar_array(b * bias_group_bool)
            time.sleep(wait_time) # loops are now in pyrogue, so no division

        self.stream_data_off(gcp_mode=gcp_mode)
        self.log('Done with TES bias ramp', self.LOG_USER)

        self.log('Returning lmsGain to original values', self.LOG_USER)
        self.set_lms_gain(2, lms_gain2)
        self.set_lms_gain(3, lms_gain3)

        basename, _ = os.path.splitext(os.path.basename(datafile))
        path = os.path.join(self.output_dir, basename + '_iv_bias_all')
        np.save(path, bias)
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
            show_plot=show_plot, save_plot=save_plot, R_sh=R_sh,
            gcp_mode=gcp_mode, grid_on=grid_on,
            phase_excursion_min=phase_excursion_min,chs=channels,band=band)

    def partial_load_curve_all(self, bias_high_array, bias_low_array=None, 
        wait_time=0.1, bias_step=0.1, gcp_mode=True, show_plot=False, analyze=True,  
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
        bias_high_array (float array): (8,) array of voltage biases, in
          bias group order

        Opt Args:
        -----
        bias_low_array (float array): (8,) array of voltage biases, in 
          bias group order. Defaults to whatever is currently set
        wait_time (float): Time to wait at each commanded bias value. Default 0.1
        bias_step (float): Interval size to step the commanded voltage bias.
          Default 0.1
        gcp_mode (bool): whether to stream data in GCP mode. Default True.
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

        datafile = self.stream_data_on(gcp_mode=gcp_mode)
        self.log('writing to {}'.format(datafile))

        # actually set the arrays
        for step in np.arange(np.shape(bias_sweep_array)[1]):
            self.set_tes_bias_bipolar_array(bias_sweep_array[:,step])
            time.sleep(wait_time) # divide by something here? unclear.

        # explicitly set back to the original biases
        self.set_tes_bias_bipolar_array(original_biases)

        self.stream_data_off(gcp_mode=gcp_mode)
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
                high_current_mode=self.high_current_mode_bool, gcp_mode=gcp_mode,
                phase_excursion_min=phase_excursion_min, channels=channels)

    def analyze_slow_iv_from_file(self, fn_iv_raw_data, make_plot=True,
                                  show_plot=False, save_plot=True, R_sh=None, 
                                  phase_excursion_min=3., grid_on=False, 
                                  gcp_mode=True, R_op_target=0.007,
                                  chs=None, band=None):
        """
        Function to analyze a load curve from its raw file. Can be used to 
          analyze IV's/generate plots separately from issuing commands.

        Args:
        fn_iv_raw_data (str): *_iv_raw_data.npy file to analyze

        Opt Args:
        make_plot (bool): Defaults True. Usually this is the slowest part.
        show_plot (bool): Defaults False.
        save_plot (bool): Defaults True.
        phase_excursion_min (float): abs(max - min) of phase in radians. Analysis 
          ignores any channels without this phase excursion. Default 3.
        grid_on (bool): Whether to draw the grid on the PR plot. Defaults False.
        gcp_mode (bool): Whether data was streamed to file in GCP mode. 
          Defaults true.
        R_op_target (float): Target operating resistance. Function will 
          generate a histogram indicating bias voltage needed to achieve 
          this value. 
        chs (int array): Which channels to analyze. Defaults to all 
          the channels that are on and exceed phase_excursion_min
        """
        self.log('Analyzing from file: {}'.format(fn_iv_raw_data))

        iv_raw_data = np.load(fn_iv_raw_data).item()
        bias = iv_raw_data['bias']
        high_current_mode = iv_raw_data['high_current_mode']
        bias_group = iv_raw_data['bias group']
        datafile = iv_raw_data['datafile']
        
        mask = self.make_mask_lookup(datafile.replace('.dat','_mask.txt'))
        bands, chans = np.where(mask != -1)

        basename = iv_raw_data['basename']
        output_dir = iv_raw_data['output_dir']
        plot_dir = iv_raw_data['plot_dir']

        # IV output dictionary
        ivs = {}
        ivs['high_current_mode'] = high_current_mode
        for b in np.unique(bands):
            ivs[b] = {}

        if gcp_mode:
            timestamp, phase_all, mask = self.read_stream_data_gcp_save(datafile)
        else:
            timestamp, phase_all = self.read_stream_data(datafile)
        
        rn_list = []
        phase_excursion_list = []
        v_bias_target_list = []
        p_trans_list = []
        si_target_list = []
        v_tes_target_list = []
        for c, (b, ch) in enumerate(zip(bands,chans)):
            if (chs is not None) and (ch not in chs):
                self.log('Not in desired channel list: skipping band {} ch. {}'.format(b,ch))
                continue
            elif (band is not None) and (b != band):
                self.log('Not in desired band: skipping band {} ch. {}'.format(b,ch))
                continue

            self.log('Analyzing band {} channel {}'.format(b,ch))
        
            # ch_idx = np.where(mask == 512*band + ch)[0][0]
            ch_idx = mask[b, ch]
            phase = phase_all[ch_idx]
         
            phase_excursion = max(phase) - min(phase)

            if phase_excursion < phase_excursion_min:
                self.log('Skipping channel {}:  phase excursion < min'.format(ch))
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
                ax.set_ylabel('Phase [rad.]')
                if grid_on:
                    ax.grid()

                ax.set_title('Band {}, Group {}, Ch {:03}'.format(np.unique(band),
                    bias_group, ch))
                plt.tight_layout()

                bg_str = ""
                for bg in np.unique(bias_group):
                    bg_str = bg_str + str(bg)


                plot_name = basename + \
                    '_IV_stream_b{}_g{}_ch{:03}.png'.format(b, bg_str, ch)
                if save_plot:
                    plt.savefig(os.path.join(plot_dir, plot_name), 
                        bbox_inches='tight', dpi=300)
                if not show_plot:
                    plt.close()

            iv_dict = self.analyze_slow_iv(bias, phase, 
                basename=basename, band=b, channel=ch, make_plot=make_plot, 
                show_plot=show_plot, save_plot=save_plot, plot_dir=plot_dir,
                R_sh = R_sh, high_current_mode = high_current_mode,
                grid_on=grid_on,R_op_target=R_op_target)
            r = iv_dict['R']
            rn = iv_dict['R_n']
            idx = iv_dict['trans idxs']
            p_tes = iv_dict['p_tes']
            p_trans = iv_dict['p_trans']
            v_bias_target = iv_dict['v_bias_target']
            si = iv_dict['si']
            si_target = iv_dict['si_target']
            v_tes_target = iv_dict['v_tes_target']

            if p_trans is not None and not np.isnan(p_trans):
                p_trans_list.append(p_trans)
            else:
                self.log('p_trans is not float')
                continue
            try:
                rn_list.append(rn)
            except:
                self.log('fitted rn is not float')
                continue
            v_bias_target_list.append(v_bias_target)
            si_target_list.append(si_target)
            v_tes_target_list.append(v_tes_target)
            ivs[b][ch] = iv_dict

        fn_iv_analyzed = basename + '_iv'
        self.log('Writing analyzed IV data to {}.'.format(fn_iv_analyzed))

        path = os.path.join(output_dir, fn_iv_analyzed)
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
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            import matplotlib.colors as Colors
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
            ax_rn.axvline(rn_median/1e-3,linestyle='--',color=color_median,\
                              label=r'Median = {:.0f}'.format(rn_median/1e-3)+\
                              r' $\mathrm{m}\Omega$')
            ax_rn.legend(loc='best')
                      
            ax_vbias.hist(v_bias_target_list,bins=20,color=color_hist)
            ax_vbias.axvline(v_bias_target_median,linestyle = '--',\
                        color=color_median,\
                        label='Median = {:.2f} V'.format(v_bias_target_median))
            ax_vbias.set_xlabel(r'Commanded voltage bias [V] for $R = $' + \
                                    '{:.0f}'.format(R_op_target/1e-3) + \
                                    r' $\mathrm{m}\Omega$')
            ax_vbias.legend(loc='best')            

            ax_ptrans.hist(p_trans_list,bins=20,color=color_hist)
            ax_ptrans.axvline(ptrans_median,linestyle='--',color=color_median,\
                           label=r'Median = {:.1f} pW'.format(ptrans_median))
            ax_ptrans.set_xlabel('In-transition electrical power [pW]')
            ax_ptrans.legend(loc='best')

            ax_si.hist(si_target_list,bins=20)
            ax_si.axvline(si_target_median,linestyle='--',color=color_median,\
                              label='Median = {:.2f}'.format(si_target_median)\
                              + ' $\mu\mathrm{V}^{-1}$')
            ax_si.axvline(si_goal,linestyle='--',color=color_goal,\
                              label=r'$-\mathrm{med}(V_\mathrm{TES})^{-1} = $'+\
                              '{:.2f}'.format(si_goal) + \
                              ' $\mu\mathrm{V}^{-1}$')
            ax_si.set_xlabel('Responsivity [$\mu\mathrm{V}^{-1}$] at $R = $'+\
                                 '{:.0f}'.format(R_op_target/1e-3) + \
                                 ' $\mathrm{m}\Omega$')
            plt.legend(loc='best')

            plt.tight_layout()
            fig.subplots_adjust(top=0.925)
            plt.suptitle('{}, band {}, group{}'.format(basename,\
                                             np.unique(band),bias_group))
            iv_hist_filename = os.path.join(plot_dir,\
                                                '%s_IV_hist.png' % (basename))
            plt.savefig(iv_hist_filename,bbox_inches='tight')
            self.log('Saved IV histogram to {}'.format(iv_hist_filename))
            if not show_plot:
                plt.close()

    def analyze_slow_iv(self, v_bias, resp, make_plot=True, show_plot=False,
        save_plot=True, basename=None, band=None, channel=None, R_sh=None,
        plot_dir=None, high_current_mode=False, bias_group = None,
        grid_on=False,R_op_target=0.007, **kwargs):
        """
        Analyzes the IV curve taken with slow_iv()

        Args:
        -----
        v_bias (float array): The commanded bias in voltage. Length n_steps
        resp (float array): The TES phase response in radians. Of length 
                            n_pts (not the same as n_steps

        Returns:
        --------
        R (float array): 
        R_n (float): 
        idx (int array): 
        R_sh (float): Shunt resistance
        """
        if R_sh is None:
            R_sh=self.R_sh

        resp *= self.pA_per_phi0/(2.*np.pi*1e6) # convert phase to uA

        n_pts = len(resp)
        n_step = len(v_bias)

        step_size = float(n_pts)/n_step  # The number of samples per V_bias step

        resp_bin = np.zeros(n_step)

        r_inline = self.bias_line_resistance
        if high_current_mode:
            # high-current mode generates higher current by decreases the in-line resistance
            r_inline /= self.high_low_current_ratio
        i_bias = 1.0E6 * v_bias / r_inline 

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
        dd_resp = np.diff(d_resp)
        v_bias = v_bias[::-1]
        i_bias = i_bias[::-1]
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
        norm_fit = np.polyfit(i_bias[nb_fit_idx:], resp_bin[nb_fit_idx:], 1)
        if norm_fit[0] < 0:  # Check for flipped polarity
            resp_bin = -1 * resp_bin
            norm_fit = np.polyfit(i_bias[nb_fit_idx:], resp_bin[nb_fit_idx:], 1)

        resp_bin -= norm_fit[1]  # now in real current units

        sc_fit = np.polyfit(i_bias[:sc_idx], resp_bin[:sc_idx], 1)
        resp_bin[:sc_idx] -= sc_fit[1] # subtract off unphysical y-offset in superconducting branch; this is probably due to an undetected phase wrap at the kink between the superconducting branch and the transition, so it is *probably* legitimate to remove it by hand. We don't use the offset of the superconducting branch for anything meaningful anyway. This will just make our plots look nicer.
        sc_fit[1] = 0 # now change s.c. fit offset to 0 for plotting

        R = R_sh * (i_bias/(resp_bin) - 1)
        R_n = np.mean(R[nb_fit_idx:])
        R_L = np.mean(R[1:sc_idx])

        v_tes = i_bias*R_sh*R/(R+R_sh) # voltage over TES
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
        i_op_target = i_bias[i_R_op]
        v_bias_target = v_bias[i_R_op]
        v_tes_target = v_tes[i_R_op]
        p_trans_median = np.median(p_tes[sc_idx:nb_idx])

        i_tes = resp_bin        
        smooth_dist = 5
        w_len = 2*smooth_dist + 1
        w = (1./float(w_len))*np.ones(w_len) # window
        i_tes_smooth = np.convolve(i_tes,w,mode='same')
        v_tes_smooth = np.convolve(v_tes,w,mode='same')
        r_tes_smooth = v_tes_smooth/i_tes_smooth
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
        si = -(1./i0)*( dv_tes/di_tes - (r0+rL+beta*r0) ) / \
            ( (2.*r0-rL+beta*r0)*dv_tes/di_tes - 3.*rL*r0 - rL**2 )
        '''
        plt.figure()
        plt.plot(i_bias[:-1],rL)
        plt.plot(i_bias[:-1],R_L*np.ones(len(rL)))
        plt.plot(i_bias[:-1],r0)
        plt.show()
        '''
        if i_R_op == len(si):
            i_R_op -= 1
        si_target = si[i_R_op]

        if make_plot:
            from matplotlib.gridspec import GridSpec
            import matplotlib.colors as Colors
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
                title += ', {:.2f} MHz'.format(self.channel_to_freq(band, channel))
            title += r', $R_\mathrm{sh}$ = ' + '${:.2f}$ '.format(R_sh*1.0E3) + \
                '$\mathrm{m}\Omega$'
            plot_name = basename + '_' + plot_name
            title = basename + ' ' + title
            plot_name += '.png'

            fig.suptitle(title)

            color_meas = colors[0]
            color_norm = colors[1]
            color_target = colors[2]
            color_sc = colors[6]
            color_etf = colors[3]

            ax_ii.axhline(0.,color='grey',linestyle=':')
            ax_ii.plot(i_bias, resp_bin, color=color_meas)
            ax_ii.set_ylabel(r'$I_\mathrm{TES}$ $[\mu A]$')

            ax_ii.plot(i_bias, norm_fit[0] * i_bias , linestyle='--', 
                       color=color_norm, label=r'$R_N$' + \
                           '  = ${:.0f}$'.format(R_n/1e-3) + \
                           r' $\mathrm{m}\Omega$')  
            ax_ii.plot(i_bias[:sc_idx], 
                sc_fit[0] * i_bias[:sc_idx] + sc_fit[1], linestyle='--', 
                color=color_sc,label=r'$R_L$' + \
                           ' = ${:.0f}$'.format(R_L/1e-6) + \
                           r' $\mu\mathrm{\Omega}$')

            label_target = r'$R = {:.0f}$ '.format(R_op_target/1e-3)+\
                        r'$\mathrm{m}\Omega$'
            label_rfrac = '{:.2f}-{:.2f}'.format(R_frac_min,\
                                             R_frac_max) + r'$R_N$'

            for i in range(len(ax_i)):
                if ax_i[i] == ax_ri:
                    label_vline = label_target
                    label_vspan = label_rfrac
                else:
                    label_vline = None
                    label_vspan = None
                ax_i[i].axvline(i_op_target,color='g',linestyle='--',
                                label=label_vline)
                ax_i[i].axvspan(i_bias[sc_idx], i_bias[nb_idx], 
                                color=color_etf, alpha=.15,label=label_vspan)
                if grid_on:
                    ax_i[i].grid()
                ax_i[i].set_xlim(min(i_bias),max(i_bias))
                if i != len(ax_i)-1:
                    ax_i[i].set_xticklabels([])
            ax_si.axhline(0.,color=color_norm,linestyle='--')

            ax_ii.legend(loc='best')
            ax_ri.legend(loc='best')
            ax_ri.plot(i_bias, R/R_n, color=color_meas)
            ax_pr.plot(p_tes,R/R_n, color=color_meas)
            for ax in [ax_ri,ax_pr]:
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
            ib_max = np.max(i_bias)
            ib_min = np.min(i_bias)
            n_ticks = 5
            delta = float(ib_max - ib_min)/n_ticks
            vb_max = np.max(v_bias)
            vb_min = np.min(v_bias)
            delta_v = float(vb_max - vb_min)/n_ticks
            axt.set_xticks(np.arange(ib_min, ib_max+delta, delta))
            axt.set_xticklabels(['{:.2f}'.format(x) for x in 
                np.arange(vb_min, vb_max+delta_v, delta_v)])
            axt.set_xlabel(r'Commanded $V_b$ [V]')

            ax_si.plot(i_bias[:-1],si,color=color_meas)
            ax_si.plot(i_bias[:-1],si_etf,linestyle = '--',
                       label=r'$-1/V_\mathrm{TES}$',color=color_etf)
            ax_si.set_ylabel(r'$S_I$ [$\mu\mathrm{V}^{-1}$]')
            ax_si.set_ylim(-2./v_tes_target,2./v_tes_target)
            ax_si.legend(loc='upper right')

            ax_pr.set_xlabel(r'$P_\mathrm{TES}$ [pW]')
            ax_pr.set_xscale('log')
            ax_pr.axhspan(R_trans_min/R_n,R_trans_max/R_n,color=color_etf, 
                              alpha=.15)
            label_pr = r'%.1f pW' % (p_trans_median)
            ax_pr.axvline(p_trans_median, linestyle='--', label=label_pr,
                              color=color_etf)
            ax_pr.plot(p_tes[i_R_op],R[i_R_op]/R_n,'o',color=color_target,
                       label=label_target)
            ax_pr.legend(loc='best')
            if grid_on:
                ax_pr.grid()

            fig.subplots_adjust(top=0.875)

            if save_plot:
                if plot_dir == None:
                    plot_dir = self.plot_dir
                plot_filename = os.path.join(plot_dir, plot_name)
                self.log('Saving IV plot to {}'.format(plot_filename))
                plt.savefig(plot_filename,bbox_inches='tight')
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
        iv_dict['v_bias'] = v_bias
        iv_dict['si_target'] = si_target
        iv_dict['v_tes_target'] = v_tes_target
        iv_dict['v_tes'] = v_tes

        return iv_dict

    def analyze_plc_from_file(self, fn_plc_raw_data, make_plot=True, show_plot=False, 
        save_plot=True, R_sh=None, high_current_mode=None, phase_excursion_min=1., 
        gcp_mode=True, channels=None):
        """
        Function to analyze a partial load curve from its raw file. Basically the same 
        as the slow_iv analysis but without fitting the superconducting branch.

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
        gcp_mode (bool): whether data was streamed to file in GCP mode. Default True.
        channels (int array): which channels to analyze. Defaults to all channels that
          are on and exceed phase_excursion_min 
        """

        self.log('Analyzing plc from file: {}'.format(fn_plc_raw_data))

        plc_raw_data = np.load(fn_plc_raw_data).item()
        bias_sweep_array = plc_raw_data['bias']
        bias_group = plc_raw_data['band']
        datafile = plc_raw_data['datafile']
        basename = plc_raw_data['basename']
        output_dir = plc_raw_data['output_dir']
        plot_dir = plc_raw_data['plot_dir']

        if gcp_mode:
            timestamp, phase_all, mask = self.read_stream_data_gcp_save(datafile)
        else:
            timestamp, phase_all = self.read_stream_data(datafile)
            mask = self.make_mask_lookup(datafile.replace('.dat', '_mask.txt'))
            # ask Ed about this later

        band, chans = np.where(mask != -1) #are these masks secretly the same?

        rn_list = []
        phase_excursion_list = []
        v_bias_target_list = []
        p_trans_list = []

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
                import matplotlib.pyplot as plt
                plt.rcParams["patch.force_edgecolor"] = True

                if not show_plot:
                    plot.ioff()

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
                    plt.savefig(of.path.join(plot_dir, plt_name), bbox_inches='tight', 
                        dpi=300)

                if not show_plot:
                    plt.close()


    def find_bias_groups(self, make_plot=False, show_plot=False, 
                         save_plot=True, min_gap=.5):
        """
        Loops through all the bias groups and ramps the TES bias.
        It takes full_band resp at each TES bias and looks for
        frequency swings. Using this data, it attempts to assign
        channels to bias groups. 

        Opt Args:
        --------
        make_plot (bool): Whether to make plots.
        show_plot (bool): If make_plot is True, whether to show
            the plot.
        save_plot (bool): If make_plot is True, whether to save
            the plot.
        min_gap (float): The minimum allowable gap.
        """
        self.log('This is specific for the Keck K2 umux FPU.')
        self.log('Working on band 2 first')
        tes_freq = {}
        for bg in np.arange(4):
            # The frequency of TESs in MHz
            tes_freq[bg] = self.find_tes(2, bg, make_plot=make_plot) + \
                self.get_band_center_mhz(2)
            
        good_tes = {}

        # Anything we see in BG 0 is noise.
        bad_res = tes_freq[0]

        for bg in np.arange(1,4):
            good_tes[bg] = np.array([])
            
            # Find resonators too close to known bad resonators
            for r in tes_freq[bg]:
                if np.min(np.abs(bad_res - r)) > min_gap:
                    good_tes[bg] = np.append(good_tes[bg], r)
                    
        ca_freq, ca_sb, ca_ch, ca_bg = self.get_master_assignment(2)

        for bg in np.arange(1,4):
            for tes in good_tes[bg]:
                nearest = np.min(np.abs(ca_freq - tes))
                print(nearest)
                if nearest < min_gap:
                    idx = np.where(np.abs(ca_freq-tes) == nearest)
                    ca_bg[idx] = bg
            
        self.write_master_assignment(2, ca_freq, ca_sb, ca_ch,
                                     groups=ca_bg)

        self.log('Working on band 3')
        for bg in np.arange(4,8):
            # The frequency of TESs in MHz
            tes_freq[bg] = self.find_tes(3, bg, make_plot=make_plot) + \
                self.get_band_center_mhz(3)

        # Anything we see in BG 6 is noise.
        bad_res = tes_freq[6]

        for bg in np.array([4,5,7]):
            good_tes[bg] = np.array([])
            
            # Find resonators too close to known bad resonators
            for r in tes_freq[bg]:
                if np.min(np.abs(bad_res - r)) > min_gap:
                    good_tes[bg] = np.append(good_tes[bg], r)
                    
        ca_freq, ca_sb, ca_ch, ca_bg = self.get_master_assignment(3)

        for bg in np.array([4,5,7]):
            for tes in good_tes[bg]:
                nearest = np.min(np.abs(ca_freq - tes))
                if nearest < min_gap:
                    idx = np.where(np.abs(ca_freq-tes) == nearest)
                    ca_bg[idx] = bg
            
        self.write_master_assignment(3, ca_freq, ca_sb, ca_ch,
                                     groups=ca_bg)

        #for k in good_tes.keys():
        #    self.log('{} TESs in BG {}'.format(len(good_tes[k], k)))

        return good_tes


    def find_tes(self, band, bias_group, bias=np.arange(0,2.5,.4),
                 make_plot=False, show_plot=False, save_plot=True,
                 make_debug_plot=False, delta_peak_cutoff=.2):
        """
        This changes the bias on the bias groups and attempts to find
        resonators. 

        Args:
        -----
        band (int) : The band to search.
        bias_group (int): The bias group to search
        
        Opt Args:
        ---------
        bias (float array) : The TES biases in volts to set to look
            for TESs.
        make_plot (bool) : Whether to make a summary plot. Default True.
        make_debug_plot (bool) : Whether to make debugging plots. 
            Default is False.
        delta_peak_cutoff (float) : The minimum a TES must move in MHz.
            Default is 0.2. 


        Ret:
        ----
        res_freq (float array) : The frequency of the resonators that
           have TESs.
        """
        if make_plot:
            import matplotlib.pyplot as plt

        self.flux_ramp_off()

        f, d = self.full_band_resp(band)
        f *= 1.0E-6  # convert freq to MHz

        ds = np.zeros((len(bias), len(d)), dtype=complex)

        # Find resonators at different TES biases
        for i, b in enumerate(bias):
            self.set_tes_bias_bipolar(bias_group, b, wait_after=.1)
            _, ds[i] = self.full_band_resp(band)

        # Find resonator peaks
        peaks = self.find_peak(f, ds[0], rolling_med=True, window=2500, pad=50,
                               min_gap=50)

        # Difference from zero bias
        delta_ds = ds[1:] - ds[0]
        
        # The delta_ds at resonances
        delta_ds_peaks = np.zeros((len(peaks), len(bias)-1))
        for i, p in enumerate(peaks):
            idx = np.where(f == p)[0][0]
            delta_ds_peaks[i] = np.abs(delta_ds[:,idx])
            if make_debug_plot:
                n_lines = 8
                if i % n_lines == 0:
                    plt.figure()
                plt.plot(bias[1:], delta_ds_peaks[i], 
                         label='{:6.5f}'.format(f[idx]*1.0E-6))
                if i % n_lines == n_lines-1:
                    plt.legend()
                    plt.xlabel('Bias [V]')
                    plt.ylabel('Res Amp')

        peak_span = np.max(delta_ds_peaks, axis=1) - \
            np.min(delta_ds_peaks, axis=1)


        if make_plot:
            if show_plot:
                plt.ion()
            else:
                plt.ioff()
            fig, ax = plt.subplots(2, sharex=True)
            cm = plt.get_cmap('viridis')

            for i, b in enumerate(bias):
                color = cm(i/len(bias))
                ax[0].plot(f, np.abs(ds[i]), color=color, 
                         label='{:3.2f}'.format(b))
            ax[0].legend()
            
            ax[1].plot(peaks, peak_span, '.')

            ax[1].axhline(delta_peak_cutoff, color='k', linestyle=':')
            fig.suptitle('band {} BG {}'.format(band, bias_group))
            ax[1].set_xlabel('Freq [MHz]')

            ax[0].axvspan(-250, -125, color='k', alpha=.1)
            ax[1].axvspan(-250, -125, color='k', alpha=.1)
            ax[0].axvspan(0, 125, color='k', alpha=.1)
            ax[1].axvspan(0, 125, color='k', alpha=.1)


            if save_plot:
                timestamp = self.get_timestamp()
                plt.savefig(os.path.join(self.plot_dir, 
                                         '{}_find_tes.png'.format(timestamp)),
                            bbox_inches='tight')
            if show_plot:
                plt.show()
            else:
                plt.close()

        idx = np.ravel(np.where(peak_span > delta_peak_cutoff))
        return peaks[idx]


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
    
        import matplotlib.pyplot as plt
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
                print('Missing in-transition electrical powers for Ch. %i' % (ch))
                continue
            dPdT = (Ptrans_cold - Ptrans_hot)/dT
            self.log('Group {}, Ch {:03}: dP/dT = {:.3f} pW/K'.format(group, ch, dPdT))
            if dPdT_lim is not None:
                if dPdT >= dPdT_lim[0] and dPdT <= dPdT_lim[1]:
                    dPdT_list.append(dPdT)
                else:
                    n_outliers += 1
            else:
                dPdT_list.append(dPdT)

            fig_pr,ax_pr = plt.subplots(1,sharex=True)
            ax_pr.set_xlabel(r'$R_\mathrm{TES}$ [$\Omega$]')
            ax_pr.set_ylabel(r'$P_\mathrm{TES}$ [pW]')
            label_hot = '%s: %.0f K' % (basename_hot,t_hot)
            label_cold = '%s: %.0f K' % (basename_cold,t_cold)
            ax_pr.axhline(y=Ptrans_hot,linestyle = '--',color = 'b')
            ax_pr.axhline(y=Ptrans_cold,linestyle = '--',color = 'r')
            ax_pr.plot(R_hot,P_hot,label=label_hot,color='b')
            ax_pr.plot(R_cold,P_cold,label=label_cold,color='r')
            ax_pr.legend(loc='best')
            fig_pr.suptitle('Group {}, Ch {:03}: dP/dT = {:.3f} pW/K'.format(group, ch, dPdT))
            ax_pr.grid()
            
            plot_name = basename_hot + '_' + basename_cold + '_optEff_g{}_ch{:03}.png'.format(group, ch)
            plot_filename = os.path.join(plot_dir, plot_name)
            self.log('Saving optical-efficiency plot to {}'.format(plot_filename))
            plt.savefig(plot_filename,bbox_inches='tight', dpi=300)
            plt.close()

        plt.figure()
        plt.hist(dPdT_list,edgecolor='k')
        plt.xlabel('dP/dT [pW/K]')
        plt.grid()
        dPdT_median = np.median(dPdT_list)
        plt.title('Group {}, median = {:.3f} pW/K ({} outliers not plotted)'.format(group,dPdT_median,n_outliers))
        plot_name = basename_hot + '_' + basename_cold + '_dPdT_hist_g{}.png'.format(group)
        hist_filename = os.path.join(plot_dir,plot_name)
        self.log('Saving optical-efficiency histogram to {}'.format(hist_filename))
        plt.savefig(hist_filename,bbox_inches='tight',dpi=300)
        plt.close()


