#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf base module - SmurfControl class
#-----------------------------------------------------------------------------
# File       : pysmurf/base/smurf_control.py
# Created    : 2018-08-29
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
import os
import time
import glob
from pysmurf.client.command.smurf_command import SmurfCommandMixin as SmurfCommandMixin
from pysmurf.client.command.smurf_atca_monitor import SmurfAtcaMonitorMixin as SmurfAtcaMonitorMixin
from pysmurf.client.util.smurf_util import SmurfUtilMixin as SmurfUtilMixin
from pysmurf.client.tune.smurf_tune import SmurfTuneMixin as SmurfTuneMixin
from pysmurf.client.debug.smurf_noise import SmurfNoiseMixin as SmurfNoiseMixin
from pysmurf.client.debug.smurf_iv import SmurfIVMixin as SmurfIVMixin
from pysmurf.client.base.smurf_config import SmurfConfig as SmurfConfig



class SmurfControl(SmurfCommandMixin, SmurfAtcaMonitorMixin, SmurfUtilMixin, SmurfTuneMixin,
        SmurfNoiseMixin, SmurfIVMixin):
    '''
    Base class for controlling Smurf. Loads all the mixins.
    '''

    def __init__(self, epics_root=None,
                 cfg_file=None,
                 data_dir=None, name=None, make_logfile=True,
                 setup=False, offline=False, smurf_cmd_mode=False,
                 no_dir=False, shelf_manager='shm-smrf-sp01',
                 validate_config=True, **kwargs):
        '''
        Initializer for SmurfControl.

        Args:
        -----
        epics_root (string) : The epics root to be used. Default mitch_epics
        cfg_file (string) : Config file path. Default is None. Must be provided
            if not on offline mode.
        data_dir (string) : Path to the data dir

        Opt Args:
        ----------
        data_dir (str) : Path to the data directory
        name (str) : The name of the output directory. If None, it will use
            the current timestamp as the output directory name. Default is None.
        make_logfile (bool) : Whether to make a log file. If False, outputs will
            go to the screen.
        setup (bool) : Whether to run the setup step. Default is False.
        smurf_cmd_mode (bool) : This mode tells the system that the input is
            coming in from the command line (rather than a python session). 
            Everything implemented here are in smurf_cmd.py. Default is False.
        no_dir (bool) : Whether to make a skip making a directory. Default is 
            False.
        validate_config (bool) : Whether to check if the input config file is
            correct. Default is True.
        shelf_manager (str):
            Shelf manager ip or network name.  Usually each SMuRF server is
            connected one-to-one with a SMuRF crate, in which case we by
            default give the shelf manager the network name 'shm-smrf-sp01'.

        '''
        if not offline and cfg_file is None:
            raise ValueError('Must provide config file.')

            self.shelf_manager=shelf_manager
            if epics_root is None:
                epics_root = self.config.get('epics_root')

        # In offline mode, epics root is not needed.
        if offline and epics_root is None:
            epics_root = ''

        super().__init__(epics_root=epics_root, offline=offline, **kwargs)

        if cfg_file is not None or data_dir is not None:
            self.initialize(cfg_file=cfg_file, data_dir=data_dir,
                name=name, make_logfile=make_logfile, setup=setup,
                smurf_cmd_mode=smurf_cmd_mode, no_dir=no_dir, **kwargs)

    def initialize(self, cfg_file, data_dir=None, name=None,
                   make_logfile=True, setup=False,
                   smurf_cmd_mode=False, no_dir=False, publish=False,
                   **kwargs):
        '''
        Initizializes SMuRF with desired parameters set in experiment.cfg.
        Largely stolen from a Cyndia/Shawns SmurfTune script
        '''

        if no_dir:
            print('Warning! Not making output directories!' +
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
            self.log_file = os.path.join(self.output_dir, 'logs', datestr +
                'smurf_cmd.log')
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

        # Crate/carrier configuration details that won't change.
        self.crate_id=self.get_crate_id()
        self.slot_number=self.get_slot_number()

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
        if '50k_Id_offset' in keys:
            self._50k_Id_offset=amp_cfg['50k_Id_offset']
        if 'hemt_gate_min_voltage' in keys:
            self._hemt_gate_min_voltage=amp_cfg['hemt_gate_min_voltage']
        if 'hemt_gate_max_voltage' in keys:
            self._hemt_gate_max_voltage=amp_cfg['hemt_gate_max_voltage']


        # Flux ramp hardware detail
        flux_ramp_cfg = self.config.get('flux_ramp')
        keys = flux_ramp_cfg.keys()
        self.num_flux_ramp_counter_bits=flux_ramp_cfg['num_flux_ramp_counter_bits']

        # Mapping from chip number to frequency in GHz
        chip_cfg = self.config.get('chip_to_freq')
        keys = chip_cfg.keys()
        self.chip_to_freq = np.zeros((len(keys), 3))
        for i, k in enumerate(chip_cfg.keys()):
            val = chip_cfg[k]
            self.chip_to_freq[i] = [k, val[0], val[1]]

        # channel assignment file
        #self.channel_assignment_files = self.config.get('channel_assignment')
        self.channel_assignment_files = {}
        if not no_dir:
            for b in self.config.get('init').get('bands'):
                all_channel_assignment_files=glob.glob(os.path.join(self.tune_dir,
                    '*channel_assignment_b{}.txt'.format(b)))
                if len(all_channel_assignment_files):
                    self.channel_assignment_files['band_{}'.format(b)] = \
                        np.sort(all_channel_assignment_files)[-1]

        # bias groups available
        self.all_groups = self.config.get('all_bias_groups')

        # bias group to pair
        bias_group_cfg = self.config.get('bias_group_to_pair')
        # how many bias groups are there?
        self._n_bias_groups=len(bias_group_cfg)
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
        self.reset_rate_khz = self.config.get('tune_band').get('reset_rate_khz')

        smurf_init_config = self.config.get('init')
        bands = smurf_init_config['bands']
        # Load in tuning parameters, if present
        tune_band_cfg=self.config.get('tune_band')
        tune_band_keys=tune_band_cfg.keys()
        self.lms_gain = {}
        for b in bands:
            # Make band dictionaries
            self.freq_resp[b] = {}
            self.freq_resp[b]['lock_status'] = {}
            self.lms_freq_hz[b] = tune_band_cfg['lms_freq'][str(b)]
            band_str = f'band_{b}'
            self.lms_gain[b] = smurf_init_config[band_str]['lmsGain']

        # Load in tuning parameters, if present
        tune_band_cfg=self.config.get('tune_band')
        tune_band_keys=tune_band_cfg.keys()
        for cfg_var in ['gradient_descent_gain', 'gradient_descent_averages',
                        'gradient_descent_converge_hz', 'gradient_descent_step_hz',
                        'gradient_descent_momentum', 'gradient_descent_beta',
                        'eta_scan_del_f', 'eta_scan_amplitude',
                        'eta_scan_averages','delta_freq']:
            if cfg_var in tune_band_keys:
                setattr(self, cfg_var, {})
                for b in bands:
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

        # If active, disable hardware logging while doing setup.
        if self._hardware_logging_thread is not None:
            self.log('Hardware logging is enabled.  Pausing for setup.',
                (self.LOG_USER))
            self.pause_hardware_logging()

        # Thermal OT protection
        ultrascale_temperature_limit_degC = self.config.get('ultrascale_temperature_limit_degC')
        if ultrascale_temperature_limit_degC is not None:
            self.log('Setting ultrascale OT protection limit to {}C'.format(ultrascale_temperature_limit_degC),
                (self.LOG_USER))
            # OT threshold in degrees C
            self.set_ultrascale_ot_threshold(
                self.config.get('ultrascale_temperature_limit_degC'),
                write_log=write_log)

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
            # in DSPv3, lmsDelay should be 4*refPhaseDelay (says
            # Mitch).  If none provided in cfg, enforce that
            # constraint.  If provided in cfg, override with provided
            # value.
            if smurf_init_config[band_str]['lmsDelay'] is None:
                self.set_lms_delay(b, int(4*smurf_init_config[band_str]['refPhaseDelay']),
                                   write_log=write_log, **kwargs)
            else:
                self.set_lms_delay(b, smurf_init_config[band_str]['lmsDelay'],
                                   write_log=write_log, **kwargs)

            self.set_feedback_enable(b,
                smurf_init_config[band_str]['feedbackEnable'],
                write_log=write_log, **kwargs)
            self.set_feedback_gain(b,
                smurf_init_config[band_str]['feedbackGain'],
                write_log=write_log, **kwargs)
            self.set_lms_gain(b, smurf_init_config[band_str]['lmsGain'],
                write_log=write_log, **kwargs)
            self.set_trigger_reset_delay(b, smurf_init_config[band_str]['trigRstDly'],
                write_log=write_log, **kwargs)

            self.set_feedback_limit_khz(b, smurf_init_config[band_str]['feedbackLimitkHz'],
                write_log=write_log, **kwargs)

            self.set_feedback_polarity(b,
                smurf_init_config[band_str]['feedbackPolarity'],
                write_log=write_log, **kwargs)

            for dmx in np.array(smurf_init_config[band_str]["data_out_mux"]):
                self.set_data_out_mux(int(self.band_to_bay(b)), int(dmx), "UserData", write_log=write_log,
                    **kwargs)

            self.set_dsp_enable(b, smurf_init_config['dspEnable'],
                write_log=write_log, **kwargs)

            # Tuning defaults - only set if present in cfg
            if hasattr(self,'gradient_descent_gain') and b in self.gradient_descent_gain.keys():
                self.set_gradient_descent_gain(b, self.gradient_descent_gain[b], write_log=write_log, **kwargs)
            if hasattr(self,'gradient_descent_averages') and b in self.gradient_descent_averages.keys():
                self.set_gradient_descent_averages(b, self.gradient_descent_averages[b], write_log=write_log, **kwargs)
            if hasattr(self,'gradient_descent_converge_hz') and b in self.gradient_descent_converge_hz.keys():
                self.set_gradient_descent_converge_hz(b, self.gradient_descent_converge_hz[b], write_log=write_log, **kwargs)
            if hasattr(self,'gradient_descent_step_hz') and b in self.gradient_descent_step_hz.keys():
                self.set_gradient_descent_step_hz(b, self.gradient_descent_step_hz[b], write_log=write_log, **kwargs)
            if hasattr(self,'gradient_descent_momentum') and b in self.gradient_descent_momentum.keys():
                self.set_gradient_descent_momentum(b, self.gradient_descent_momentum[b], write_log=write_log, **kwargs)
            if hasattr(self,'gradient_descent_beta') and b in self.gradient_descent_beta.keys():
                self.set_gradient_descent_beta(b, self.gradient_descent_beta[b], write_log=write_log, **kwargs)
            if hasattr(self,'eta_scan_averages') and b in self.eta_scan_averages.keys():
                self.set_eta_scan_averages(b, self.eta_scan_averages[b], write_log=write_log, **kwargs)
            if hasattr(self,'eta_scan_amplitude') and b in self.eta_scan_amplitude.keys():
                self.set_eta_scan_amplitude(b, self.eta_scan_amplitude[b], write_log=write_log, **kwargs)
            if hasattr(self,'eta_scan_del_f') and b in self.eta_scan_del_f.keys():
                self.set_eta_scan_del_f(b, self.eta_scan_del_f[b], write_log=write_log, **kwargs)

        # To work around issue where setting the UC attenuators is for
        # some reason also setting the UC attenuators for other bands
        # with the new C03 AMCs, first set all the UC attenuators
        # (which don't seem to be affected by setting the DC
        # attenuators, at least for LB bands 2 and 3), then set all
        # the UC attenuators.
        for b in bands:
            self.set_att_uc(b, smurf_init_config[band_str]['att_uc'],
                            write_log=write_log)
        for b in bands:
            self.set_att_dc(b, smurf_init_config[band_str]['att_dc'],
                            write_log=write_log)

        # Things that have to be done for both AMC bays, regardless of whether or not an AMC
        # is plugged in there.
        for bay in [0,1]:
            self.set_trigger_hw_arm(bay, 0, write_log=write_log)

        self.set_trigger_width(0, 10, write_log=write_log)  # mystery bit that makes triggering work
        self.set_trigger_enable(0, 1, write_log=write_log)
        self.set_evr_channel_reg_enable(0, True, write_log=write_log)
        ## only sets enable, but is initialized to True already by
        ## default, and crashing for unknown reasons in rogue 4.
        #self.set_evr_trigger_reg_enable(0, True, write_log=write_log)
        self.set_evr_trigger_channel_reg_dest_sel(0, 0x20000, write_log=write_log)

        self.set_enable_ramp_trigger(1, write_log=write_log)

        flux_ramp_cfg = self.config.get('flux_ramp')
        self.set_select_ramp(flux_ramp_cfg['select_ramp'], write_log=write_log)

        self.set_cpld_reset(0, write_log=write_log)
        self.cpld_toggle(write_log=write_log)

        # Make sure flux ramp starts off
        self.flux_ramp_off(write_log=write_log)
        self.flux_ramp_setup(4, .5, write_log=write_log)


        # Turn on stream enable for all bands
        self.set_stream_enable(1, write_log=write_log)

        # self.set_smurf_to_gcp_clear(1, write_log=write_log, wait_after=1)
        # self.set_smurf_to_gcp_clear(0, write_log=write_log)

        self.set_amplifier_bias(write_log=write_log)
        self.get_amplifier_bias()
        self.log("Cryocard temperature = "+ str(self.C.read_temperature())) # also read the temperature of the CC

        # if no timing section present, assumes your defaults.yml
        # has set you up...good luck.
        if self.config.get('timing') is not None and self.config.get('timing').get('timing_reference') is not None:
            timing_reference=self.config.get('timing').get('timing_reference')

            # check if supported
            timing_options=['ext_ref','backplane']
            assert (timing_reference in timing_options), 'timing_reference in cfg file (={}) not in timing_options={}'.format(timing_reference,str(timing_options))

            self.log('Configuring the system to take timing from {}'.format(timing_reference))

            if timing_reference=='ext_ref':
                for bay in self.bays:
                    self.log(f'Select external reference for bay {bay}')
                    self.sel_ext_ref(bay)

                # make sure RTM knows there's no timing system
                self.set_ramp_start_mode(0,write_log=write_log)

            # https://confluence.slac.stanford.edu/display/SMuRF/Timing+Carrier#TimingCarrier-Howtoconfiguretodistributeoverbackplanefromslot2
            if timing_reference=='backplane':
                # Set SMuRF carrier crossbar to use the backplane
                # distributed timing.
                # OutputConfig[1] = 0x2 configures the SMuRF carrier's
                # FPGA to take the timing signals from the backplane
                # (TO_FPGA = FROM_BACKPLANE)
                self.log('Setting crossbar OutputConfig[1]=0x2 (TO_FPGA=FROM_BACKPLANE)')
                self.set_crossbar_output_config(1,2)

                self.log('Waiting 1 sec for timing up-link...')
                time.sleep(1)

                # Check if link is up - just printing status to
                # screen, not currently taking any action if it's not.
                timingRxLinkUp=self.get_timing_link_up()
                self.log(f'Timing RxLinkUp = {timingRxLinkUp}', self.LOG_USER if
                         timingRxLinkUp else self.LOG_ERROR)

                # Set LMK to use timing system as reference
                for bay in self.bays:
                    self.log(f'Configuring bay {bay} LMK to lock to the timing system')
                    self.set_lmk_reg(bay, 0x147, 0xA)

                # Configure RTM to trigger off of the timing system
                self.set_ramp_start_mode(1, write_log=write_log)

        self.log('Done with setup')

        # If active, re-enable hardware logging after setup.
        if self._hardware_logging_thread is not None:
            self.log('Resuming hardware logging.', (self.LOG_USER))
            self.resume_hardware_logging()

    def make_dir(self, directory):
        """
        check if a directory exists; if not, make it

        Args:
        -----
        directory (str): path of directory to create
        """
        if not os.path.exists(directory):
            os.makedirs(directory)


    def get_timestamp(self, as_int=False):
        """
        Gets the unixtime timestamp.

        Opt Args:
        ---------
        as_int (bool) : Whether to returnt the timestamp as an
            integer.

        Returns:
        --------
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
