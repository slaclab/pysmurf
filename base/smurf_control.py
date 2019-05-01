import numpy as np
import os
import sys
import time
import glob
from pysmurf.command.smurf_command import SmurfCommandMixin as SmurfCommandMixin
from pysmurf.util.smurf_util import SmurfUtilMixin as SmurfUtilMixin
from pysmurf.tune.smurf_tune import SmurfTuneMixin as SmurfTuneMixin
from pysmurf.debug.smurf_noise import SmurfNoiseMixin as SmurfNoiseMixin
from pysmurf.debug.smurf_iv import SmurfIVMixin as SmurfIVMixin
from pysmurf.base.smurf_config import SmurfConfig as SmurfConfig

class SmurfControl(SmurfCommandMixin, SmurfUtilMixin, SmurfTuneMixin, 
    SmurfNoiseMixin, SmurfIVMixin):
    '''
    Base class for controlling Smurf. Loads all the mixins.
    '''
    def __init__(self, epics_root=None, 
        cfg_file='/home/cryo/pysmurf/cfg_files/experiment_k2umux.cfg', 
        data_dir=None, name=None, make_logfile=True, 
        setup=False, offline=False, smurf_cmd_mode=False, no_dir=False,
        **kwargs):
        '''
        Args:
        -----
        epics_root (string) : The epics root to be used. Default mitch_epics
        cfg_file (string) : Path the config file
        data_dir (string) : Path to the data dir
        '''
        self.config = SmurfConfig(cfg_file)
        if epics_root is None:
            epics_root = self.config.get('epics_root')

        super().__init__(epics_root=epics_root, offline=offline, **kwargs)

        if cfg_file is not None or data_dir is not None:
            self.initialize(cfg_file=cfg_file, data_dir=data_dir, name=name,
                make_logfile=make_logfile,
                setup=setup, smurf_cmd_mode=smurf_cmd_mode, 
                no_dir=no_dir, **kwargs)

    def initialize(self, cfg_file, data_dir=None, name=None, 
        make_logfile=True, setup=False, smurf_cmd_mode=False, 
        no_dir=False, **kwargs):
        '''
        Initizializes SMuRF with desired parameters set in experiment.cfg.
        Largely stolen from a Cyndia/Shawns SmurfTune script
        '''

        if no_dir:
            print('Warning! Not making output directories!'+ \
                'This will break may things!')
        elif smurf_cmd_mode:
            # Get data dir
            self.data_dir = self.config.get('smurf_cmd_dir')
            self.start_time = self.get_timestamp()

            # Define output and plot dirs
            self.base_dir = os.path.abspath(self.data_dir)
            self.output_dir = os.path.join(self.base_dir, 'outputs')
            self.tune_dir = self.config.get('tune_dir')
            self.plot_dir = os.path.join(self.base_dir, 'plots')
            self.status_dir = self.config.get('status_dir')
            self.make_dir(self.output_dir)
            self.make_dir(self.tune_dir)
            self.make_dir(self.plot_dir)
            self.make_dir(self.status_dir)

            # Set logfile
            datestr = time.strftime('%y%m%d_', time.gmtime())
            self.log_file = os.path.join(self.output_dir, 'logs', datestr + 'smurf_cmd.log')
            self.log.set_logfile(self.log_file)
        else:
            # define data dir
            if data_dir is not None:
                self.data_dir = data_dir
            else:
                self.data_dir = self.config.get('default_data_dir')

            self.date = time.strftime("%Y%m%d")

            # name
            self.start_time = self.get_timestamp()
            if name is None:
                name = self.start_time
            self.name = name

            self.base_dir = os.path.abspath(self.data_dir)

            # create output and plot directories
            self.output_dir = os.path.join(self.base_dir, self.date, name, 
                'outputs')
            self.tune_dir = self.config.get('tune_dir')
            self.plot_dir = os.path.join(self.base_dir, self.date, name, 'plots')
            self.status_dir = self.config.get('status_dir')
            self.make_dir(self.output_dir)
            self.make_dir(self.tune_dir)
            self.make_dir(self.plot_dir)
            self.make_dir(self.status_dir)

            # name the logfile and create flags for it
            if make_logfile:
                self.log_file = os.path.join(self.output_dir, name + '.log')
                self.log.set_logfile(self.log_file)
            else:
                self.log.set_logfile(None)

        # Useful constants
        constant_cfg = self.config.get('constant')
        self.pA_per_phi0 = constant_cfg.get('pA_per_phi0')

        # Mapping from attenuator numbers to bands
        att_cfg = self.config.get('attenuator')
        keys = att_cfg.keys()
        self.att_to_band = {}
        self.att_to_band['band'] = np.zeros(len(keys))
        self.att_to_band['att'] = np.zeros(len(keys))
        for i, k in enumerate(keys):
            self.att_to_band['band'][i] = att_cfg[k]
            self.att_to_band['att'][i] = int(k[-1])

        # Cold amplifier biases
        amp_cfg = self.config.get('amplifier')
        keys = amp_cfg.keys()
        if 'hemt_Vg' in keys:
            self.hemt_Vg=amp_cfg['hemt_Vg']
        if 'LNA_Vg' in keys:
            self.LNA_Vg=amp_cfg['LNA_Vg']
        if 'dac_num_50k' in keys:
            self._dac_num_50k=amp_cfg['dac_num_50k']
        if 'bit_to_V_50k' in keys:
            self._bit_to_V_50k=amp_cfg['bit_to_V_50k']
        if 'bit_to_V_hemt' in keys:
            self._bit_to_V_hemt=amp_cfg['bit_to_V_hemt']
        if 'hemt_Id_offset' in keys:
            self._hemt_Id_offset=amp_cfg['hemt_Id_offset']
        if 'hemt_gate_min_voltage' in keys:
            self._hemt_gate_min_voltage=amp_cfg['hemt_gate_min_voltage']
        if 'hemt_gate_max_voltage' in keys:
            self._hemt_gate_max_voltage=amp_cfg['hemt_gate_max_voltage']

            
        # Flux ramp hardware detail
        flux_ramp_cfg = self.config.get('flux_ramp')
        keys = flux_ramp_cfg.keys()
        self.num_flux_ramp_counter_bits=20
        if 'num_flux_ramp_counter_bits' in keys:
            self.num_flux_ramp_counter_bits=flux_ramp_cfg['num_flux_ramp_counter_bits']

        # Mapping from chip number to frequency in GHz
        chip_cfg = self.config.get('chip_to_freq')
        keys = chip_cfg.keys()
        self.chip_to_freq = np.zeros((len(keys), 3))
        for i, k in enumerate(chip_cfg.keys()):
            val = chip_cfg[k]
            self.chip_to_freq[i] = [k, val[0], val[1]]

        # Mapping from band to chip number
        band_cfg = self.config.get('band_to_chip')
        keys = band_cfg.keys()
        self.band_to_chip = np.zeros((len(keys), 5))
        for i, k in enumerate(keys):
            val = band_cfg[k]
            self.band_to_chip[i] = np.append([i], val)
            
        # channel assignment file
        #self.channel_assignment_files = self.config.get('channel_assignment')
        self.channel_assignment_files = {}
        if not no_dir:
            for b in self.config.get('init').get('bands'):
                self.channel_assignment_files['band_{}'.format(b)] = \
                    np.sort(glob.glob(os.path.join(self.tune_dir, 
                            '*channel_assignment_b{}.txt'.format(b))))[-1]

        # bias groups available
        self.all_groups = self.config.get('all_bias_groups')

        # bias group to pair
        bias_group_cfg = self.config.get('bias_group_to_pair')
        keys = bias_group_cfg.keys()
        self.bias_group_to_pair = np.zeros((len(keys), 3), dtype=int)
        for i, k in enumerate(keys):
            val = bias_group_cfg[k]
            self.bias_group_to_pair[i] = np.append([k], val)

        # Mapping from peripheral interface controller (PIC) to bias group
        pic_cfg = self.config.get('pic_to_bias_group')
        keys = pic_cfg.keys()
        self.pic_to_bias_group = np.zeros((len(keys), 2), dtype=int)
        for i, k in enumerate(keys):
            val = pic_cfg[k]
            self.pic_to_bias_group[i] = [k, val]

        # The resistance in line with the TES bias
        self.bias_line_resistance = self.config.get('bias_line_resistance')

        # The TES shunt resistance
        self.R_sh = self.config.get('R_sh')

        # The ratio of current for high-current mode to low-current mode;
        # also the inverse of the in-line resistance for the bias lines.
        self.high_low_current_ratio = self.config.get('high_low_current_ratio')

        # whether we are running in high vs low current mode
        self.high_current_mode_bool = self.config.get('high_current_mode_bool')

        # Sampling frequency in gcp mode in Hz
        self.fs = self.config.get('fs')

        # The smurf to mce config data
        smurf_to_mce_cfg = self.config.get('smurf_to_mce')
        self.smurf_to_mce_file = smurf_to_mce_cfg.get('smurf_to_mce_file')
        self.smurf_to_mce_ip = smurf_to_mce_cfg.get('receiver_ip')
        self.smurf_to_mce_port = smurf_to_mce_cfg.get('port_number')
        self.smurf_to_mce_mask_file = smurf_to_mce_cfg.get('mask_file')

        # Bad resonator mask
        bm_config = self.config.get('bad_mask')
        bm_keys = bm_config.keys()
        self.bad_mask = np.zeros((len(bm_keys), 2))
        for i, k in enumerate(bm_keys):
            self.bad_mask[i] = bm_config[k]

        # Which MicrowaveMuxCore[#] blocks are being used?
        self.bays=None

        # Dictionary for frequency response
        self.freq_resp = {}
        self.lms_freq_hz = {}
        self.fraction_full_scale = self.config.get('tune_band').get('fraction_full_scale')
        smurf_init_config = self.config.get('init')
        bands = smurf_init_config['bands']
        for b in bands:
            # Make band dictionaries
            self.freq_resp[b] = {}
            self.freq_resp[b]['lock_status'] = {}
            self.lms_freq_hz[b] = 4000

        # Load in tuning parameters, if present
        tune_band_cfg=self.config.get('tune_band')
        tune_band_keys=tune_band_cfg.keys()
        for cfg_var in ['gradient_descent_gain', 'gradient_descent_averages', 'eta_scan_averages']:
            if cfg_var in tune_band_keys:
                setattr(self, cfg_var, {})
                for b in  bands:
                    getattr(self,cfg_var)[b]=tune_band_cfg[cfg_var][str(b)]

        if setup:
            self.setup(**kwargs)

        # initialize outputs cfg
        self.config.update('outputs', {})

    def setup(self, write_log=True, **kwargs):
        """
        Sets the PVs to the default values from the experiment.cfg file
        """
        self.log('Setting up...', (self.LOG_USER))

        # Which bands are we configuring?
        smurf_init_config = self.config.get('init')
        bands = smurf_init_config['bands']

        # determine which bays to configure from the 
        # bands requested and the band-to-bay 
        # correspondence
        self.bays=np.unique([self.band_to_bay(band) for band in bands])
        # Right now, resetting both DACs in both MicrowaveMuxCore blocks,
        # but may want to determine at runtime which are actually needed and
        # only reset the DAC in those.
        self.log('Toggling DACs')
        dacs=[0,1]
        for val in [1,0]:
            for bay in self.bays:
                for dac in dacs:
                    self.set_dac_reset(bay, dac, val, write_log=write_log)

        self.set_read_all(write_log=write_log)
        self.set_defaults_pv(write_log=write_log)

        # The per band configs. May want to make available per-band values.
        for b in bands:
            band_str = 'band_{}'.format(b)
            self.set_iq_swap_in(b, smurf_init_config[band_str]['iq_swap_in'], 
                write_log=write_log, **kwargs)
            self.set_iq_swap_out(b, smurf_init_config[band_str]['iq_swap_out'], 
                write_log=write_log, **kwargs)
            self.set_ref_phase_delay(b, 
                smurf_init_config[band_str]['refPhaseDelay'], 
                write_log=write_log, **kwargs)
            self.set_ref_phase_delay_fine(b, 
                smurf_init_config[band_str]['refPhaseDelayFine'], 
                write_log=write_log, **kwargs)
            self.set_tone_scale(b, smurf_init_config[band_str]['toneScale'], 
                write_log=write_log, **kwargs)
            self.set_analysis_scale(b, 
                smurf_init_config[band_str]['analysisScale'], 
                write_log=write_log, **kwargs)
            self.set_feedback_enable(b, 
                smurf_init_config[band_str]['feedbackEnable'],
                write_log=write_log, **kwargs)
            self.set_feedback_gain(b, 
                smurf_init_config[band_str]['feedbackGain'], 
                write_log=write_log, **kwargs)
            self.set_lms_gain(b, smurf_init_config[band_str]['lmsGain'], 
                write_log=write_log, **kwargs)

            self.set_feedback_limit_khz(b, 225)  # why 225?

            self.set_feedback_polarity(b, 
                smurf_init_config[band_str]['feedbackPolarity'], 
                write_log=write_log, **kwargs)
            self.set_synthesis_scale(b, 
                smurf_init_config[band_str]['synthesisScale'],
                write_log=write_log, **kwargs)

            for dmx in np.array(smurf_init_config[band_str]["data_out_mux"]):
                self.set_data_out_mux(int(self.band_to_bay(b)), int(dmx), "UserData", write_log=write_log,
                    **kwargs)

            self.set_att_uc(b, smurf_init_config[band_str]['att_uc'],
                write_log=write_log)
            self.set_att_dc(b, smurf_init_config[band_str]['att_dc'],
                write_log=write_log)

            self.set_dsp_enable(b, smurf_init_config['dspEnable'], 
                write_log=write_log, **kwargs)
            
            # Tuning defaults - only set if present in cfg
            if hasattr(self,'gradient_descent_gain') and b in self.gradient_descent_gain.keys():
                self.set_gradient_descent_gain(b, self.gradient_descent_gain[b], write_log=write_log, **kwargs)
            if hasattr(self,'gradient_descent_averages') and b in self.gradient_descent_averages.keys():
                self.set_gradient_descent_averages(b, self.gradient_descent_averages[b], write_log=write_log, **kwargs)
            if hasattr(self,'eta_scan_averages') and b in self.eta_scan_averages.keys():
                self.set_eta_scan_averages(b, self.eta_scan_averages[b], write_log=write_log, **kwargs)


        # Things that have to be done for both AMC bays, regardless of whether or not an AMC
        # is plugged in there.
        for bay in [0,1]:
            self.set_trigger_hw_arm(bay, 0, write_log=write_log)
        
        self.set_trigger_width(0, 10, write_log=write_log)  # mystery bit that makes triggering work
        self.set_trigger_enable(0, 1, write_log=write_log)
        self.set_evr_channel_reg_enable(0, True, write_log=write_log)
        self.set_evr_trigger_reg_enable(0, True, write_log=write_log)
        self.set_evr_trigger_channel_reg_dest_sel(0, 0x20000, write_log=write_log)

        self.set_enable_ramp_trigger(1, write_log=True)

        flux_ramp_cfg = self.config.get('flux_ramp')
        self.set_select_ramp(flux_ramp_cfg['select_ramp'], write_log=write_log)
        self.set_ramp_start_mode(flux_ramp_cfg['ramp_start_mode'], 
                                 write_log=write_log)

        self.set_cpld_reset(0, write_log=write_log)
        self.cpld_toggle(write_log=write_log)

        # Setup SMuRF to MCE converter
        self.make_smurf_to_gcp_config()
        time.sleep(.1)
        self.read_smurf_to_gcp_config()  # Only for IP address

        # Make sure flux ramp starts off
        self.flux_ramp_off(write_log=write_log)
        self.flux_ramp_setup(4, .5, write_log=write_log) 

        # Turn off GCP streaming
        self.set_smurf_to_gcp_stream(False, write_log=write_log)
        self.set_smurf_to_gcp_writer(False, write_log=write_log)

        # Turn on stream enable for all bands
        self.set_stream_enable(1, write_log=write_log)

        self.set_smurf_to_gcp_clear(1, write_log=write_log, wait_after=1)
        self.set_smurf_to_gcp_clear(0, write_log=write_log)

        self.set_amplifier_bias(write_log=write_log)
        _ = self.get_amplifier_bias()
        self.log("Cryocard temperature = "+ str(self.C.read_temperature())) # also read the temperature of the CC

        self.log('Done with setup')
        for bay in self.bays:
            self.log('Select external reference for bay %i' % (bay))
            self.sel_ext_ref(bay)

    def make_dir(self, directory):
        """check if a directory exists; if not, make it

           Args:
            directory (str): path of directory to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_timestamp(self, as_int=False):
        """
        Returns:
        timestampe (str): Timestamp as a string
        """
        t = '{:10}'.format(int(time.time()))

        if as_int:
            return int(t)
        else:
            return t

    def add_output(self, key, val):
        """
        Add a key to the output config.

        Args:
          key (any): the name of the key to update
          val (any): value to assign to the key
        """

        self.config.update_subkey('outputs', key, val)

    def write_output(self, filename=None):
        """
        Dump the current configuration to a file. This wraps around the config
        file writing in the config object. Files are timestamped and dumped to
        the S.output_dir by default.

        Opt Args:
        -----
        filename (str): full path to output file
        """

        timestamp = self.get_timestamp()
        if filename is not None:
            output_file = filename 
        else:
            output_file = timestamp + '.cfg'

        full_path = os.path.join(self.output_dir, output_file)
        self.config.write(full_path)

