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
"""Defines the SmurfControl class.
"""
import glob
import os
import time

import numpy as np

from pysmurf.client.base.smurf_config import SmurfConfig
from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
from pysmurf.client.command.smurf_atca_monitor import SmurfAtcaMonitorMixin
from pysmurf.client.command.smurf_command import SmurfCommandMixin
from pysmurf.client.debug.smurf_iv import SmurfIVMixin
from pysmurf.client.debug.smurf_noise import SmurfNoiseMixin
from pysmurf.client.tune.smurf_tune import SmurfTuneMixin
from pysmurf.client.util.smurf_util import SmurfUtilMixin

class SmurfControl(SmurfCommandMixin,
        SmurfAtcaMonitorMixin, SmurfUtilMixin, SmurfTuneMixin,
        SmurfNoiseMixin, SmurfIVMixin,
        SmurfConfigPropertiesMixin):

    """Base class for controlling SMuRF.

    Loads all the mixins.  NEED LONGER DESCRIPTION OF THE SMURF
    CONTROL CLASS HERE.  Inherits from the following mixins:

    * :class:`~pysmurf.client.command.smurf_command.SmurfCommandMixin` for WHAT IS THIS MIXIN FOR.
    * :class:`~pysmurf.client.command.smurf_atca_monitor.SmurfAtcaMonitorMixin` for WHAT IS THIS MIXIN FOR.
    * :class:`~pysmurf.client.util.smurf_util.SmurfUtilMixin` for WHAT IS THIS MIXIN FOR.
    * :class:`~pysmurf.client.tune.smurf_tune.SmurfTuneMixin` for WHAT IS THIS MIXIN FOR.
    * :class:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin` for WHAT IS THIS MIXIN FOR.
    * :class:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin` for WHAT IS THIS MIXIN FOR.
    * :class:`~pysmurf.client.command.smurf_config_properties.SmurfConfigPropertiesMixin` for WHAT IS THIS MIXIN FOR.

    Args
    ----
    epics_root : str, optional, default None
       The epics root to be used.
    cfg_file : str, optional, default None
       Config file path.  Must be provided if not on offline mode.
    data_dir : str, optional, default None
       Path to the data directory.
    name : str, optional, default None
       The name of the output directory. If None, it will use the
       current timestamp as the output directory name.
    make_logfile : bool, optional, default True
       Whether to make a log file. If False, outputs will go to the
       screen.
    setup : bool, optional, default False
       Whether to run the setup step.
    offline : bool, optional, default False
       Whether or not to instantiate in offline mode.
    smurf_cmd_mode : bool, optional, default False
       This mode tells the system that the input is coming in from the
       command line (rather than a python session).  Everything
       implemented here are in smurf_cmd.py.
    no_dir :  bool, optional, default False
       Whether to make a skip making a directory.
    shelf_manager : str, optional, default 'shm-smrf-sp01'
       Shelf manager ip or network name.  Usually each SMuRF server is
       connected one-to-one with a SMuRF crate, and the default shelf
       manager name is configured to be 'shm-smrf-sp01'
    validate_config : bool, optional, default True
       Whether to check if the input config file is correct.

    Attributes
    ----------
    config : :class:`~pysmurf.client.base.smurf_config.SmurfConfig` or None
       ???
    output_dir : str or None
       ???

    Raises
    ------
    ValueError
       If not `offline` and `cfg_file` is None.

    See Also
    --------
    initialize
    """

    def __init__(self, epics_root=None,
                 cfg_file=None, data_dir=None, name=None, make_logfile=True,
                 setup=False, offline=False, smurf_cmd_mode=False,
                 no_dir=False, shelf_manager='shm-smrf-sp01',
                 validate_config=True, data_path_id=None, **kwargs):
        """Constructor for the SmurfControl class.

        See the SmurfControl class docstring for more details.
        """

        # Class attributes
        self.config = None
        self.output_dir = None

        # Require specification of configuration file if not in
        # offline mode.  If configuration file is specified, load into
        # config attribute.
        SmurfConfigPropertiesMixin.__init__(self)
        if not offline and cfg_file is None:
            raise ValueError('Must provide config file.')
        elif cfg_file is not None:
            self.config = SmurfConfig(cfg_file)
            # Populate SmurfConfigPropertiesMixin properties with
            # values from loaded pysmurf configuration file.
            self.copy_config_to_properties(self.config)

        # Save shelf manager - Should this be in the config?
        self.shelf_manager = shelf_manager

        # Setting epics_root
        #
        # self.epics_root is already populated by the above call to
        # copy_config_to_properties() from the pysmurf configuration
        # file (if a configuration file is provided).
        #
        # Override epics_root from pysmurf configuration file if user
        # provides a different one.
        if epics_root is not None:
            # If user provides an epics root, override whatever's in
            # the pysmurf cfg file with it.
            self.epics_root = epics_root
        # In offline mode, epics root is not needed.
        if offline:
            self.epics_root = ''
        # Done setting epics_root

        super().__init__(offline=offline, **kwargs)

        if cfg_file is not None or data_dir is not None:
            self.initialize(data_dir=data_dir,
                name=name, make_logfile=make_logfile, setup=setup,
                smurf_cmd_mode=smurf_cmd_mode, no_dir=no_dir,
                data_path_id=data_path_id, **kwargs)

    def initialize(self, data_dir=None, name=None,
                   make_logfile=True, setup=False,
                   smurf_cmd_mode=False, no_dir=False, publish=False,
                   payload_size=2048, data_path_id=None,
                   **kwargs):
        """Initializes SMuRF system.

        Longer description of initialize routine here.

        Args
        ----
        data_dir : str, optional, default None
              Path to the data directory.
        name : str, optional, default None
              The name of the output directory. If None, it will use
              the current timestamp as the output directory name.
        make_logfile : bool, optional, default True
              Whether to make a log file. If False, outputs will
              go to the screen.
        setup : bool, optional, default False
              Whether to run the setup step.
        smurf_cmd_mode : bool, optional, default False
              This mode tells the system that the input is coming in
              from the command line (rather than a python session).
              Everything implemented here are in smurf_cmd.py.
        no_dir : bool, optional, default False
              Whether to make a skip making a directory.
        publish : bool, optional, default False
              Whether to send messages to the OCS publisher.
        payload_size : int, optional, default 2048
              The payload size to set on setup.
        data_path_id : str, optional, default None
              If set, this will add the path-id to the output and plot dir
              paths to avoid possible collisions between multiple smurf
              instances running simultaneously. For instance, if set to
              ``crate1slot2`` the outputs directory will be::
              ``<data_dir>/<date>/crate1slot2/<ctime>/outputs``

        See Also
        --------
        setup

        """
        if no_dir:
            print('Warning! Not making output directories!' +
                  'This will break may things!')
        elif smurf_cmd_mode:
            # Get data dir
            self.data_dir = self._smurf_cmd_dir
            self.start_time = self.get_timestamp()

            # Define output and plot dirs
            self.base_dir = os.path.abspath(self.data_dir)
            self.output_dir = os.path.join(self.base_dir, 'outputs')
            self.tune_dir = self._tune_dir
            self.plot_dir = os.path.join(self.base_dir, 'plots')
            self.status_dir = self._status_dir
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
                self.data_dir = self._default_data_dir

            self.date = time.strftime("%Y%m%d")

            # name
            self.start_time = self.get_timestamp()
            if name is None:
                name = self.start_time
            self.name = name

            self.base_dir = os.path.abspath(self.data_dir)

            # create output and plot directories
            if data_path_id is None:
                self.output_dir = os.path.join(self.base_dir, self.date, name,
                    'outputs')
            else:
                self.output_dir = os.path.join(self.base_dir, self.date,
                                               data_path_id, name, 'outputs')

            self.tune_dir = self._tune_dir

            if data_path_id is None:
                self.plot_dir = os.path.join(self.base_dir, self.date, name,
                                             'plots')
            else:
                self.plot_dir = os.path.join(self.base_dir, self.date,
                                             data_path_id, name, 'plots')

            self.status_dir = self._status_dir
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

            # Which bays were enabled on pysmurf server startup?
            self.bays = self.which_bays()

            # Crate/carrier configuration details that won't change.
            self.crate_id = self.get_crate_id()
            self.slot_number = self.get_slot_number()

            # Channel assignment files
            self.channel_assignment_files = {}
            if not no_dir:
                for band in self._bands:
                    all_channel_assignment_files = glob.glob(
                        os.path.join(
                            self.tune_dir,
                            f'*channel_assignment_b{band}.txt'))
                    if len(all_channel_assignment_files):
                        self.channel_assignment_files[f'band_{band}'] = \
                            np.sort(all_channel_assignment_files)[-1]

            # Which bands are usable, based on which bays are enabled.
            # Will use to check if pysmurf configuration file has unusable
            # bands defined, or no definition for usable bands.
            usable_bands=[]
            for bay in self.bays:
                # There are four bands per bay.  Bay 0 provides bands 0,
                # 1, 2, and 3, and bay 1 provides bands 4, 5, 6 and 7.
                usable_bands+=range(bay*4,4*(bay+1))

            # Compare usable bands to bands defined in pysmurf
            # configuration file.

            # Check if an unusable band is defined in the pysmurf cfg
            # file.
            for band in self._bands:
                if band not in usable_bands:
                    self.log(f'ERROR : band {band} is present in ' +
                             'pysmurf cfg file, but its bay is not ' +
                             'enabled!', self.LOG_ERROR)

            # Check if a usable band is not defined in the pysmurf cfg
            # file.
            for band in usable_bands:
                if band not in self._bands:
                    self.log(f'WARNING : band {band} bay is enabled, ' +
                             'but no configuration information ' +
                             'provided!', self.LOG_ERROR)

            ## Make band dictionaries
            self.freq_resp = {}
            for band in self._bands:
                self.freq_resp[band] = {}
                self.freq_resp[band]['lock_status'] = {}

        if setup:
            success = self.setup(payload_size=payload_size, **kwargs)
            # Log an error if system setup failed.
            if not success:
                self.log(
                    'ERROR : System setup failed!  Proceed at your own'
                    ' risk!',
                    self.LOG_ERROR)

        # initialize outputs cfg
        self.config.update('outputs', {})

    def setup(self, write_log=True, payload_size=2048, **kwargs):
        r"""Configures SMuRF system.

        TODO: NEED TO BE MORE DETAILED, CLEARER.

        Sets up the SMuRF system by first loading hardware register
        defaults followed by overriding the hardware default register
        values with defaults from the pysmurf configuration file.

        Setup steps (in order of execution):

        - Disables hardware logging if it’s active (to avoid register
          access collisions).
        - Sets FPGA OT limit (if one is specified in pysmurf cfg).
        - Resets the RF DACs on AMCs in use.
        - Sets hardware defaults using
          :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_defaults_pv`.
        - Checks if JESD is locked using
          :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_check_jesd`.
        - Overrides hardware defaults register values for subset of
          registers controlled by the pysmurf configuration file.
        - Resets the RTM CPLD.
        - Turns off flux ramp, but configures based on values for
          reset rate and fraction full scale provided in pysmurf
          configuration file.
        - Enables data streaming.
        - Sets mask and payload size to a single channel.
        - Sets RF amplifier biases (without enabling drain voltages).
        - Configures timing based on pysmurf configuration file settings.
        - Resumes hardware logging if it was active at beginning.

        If system configuration fails, returns `False`, otherwise
        returns `True`.  Failure modes (which will return `False`) are
        as follows:

        - :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_defaults_pv`
          fails (only supported for pysmurf core code versions
          >=4.1.0).  If failure is detected, doesn't attempt to
          execute the subsequent setup steps.
        - :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_check_jesd`
          fails (only supported for Rogue ZIP file versions >=0.3.0).
          If failure is detected, doesn't attempt to execute the
          subsequent setup steps.

        Args
        ----
        write_log : bool, optional, default True
            Whether to write to the log file.
        payload_size : int, optional, default 2048
            The starting size of the payload.
        \**kwargs
            Arbitrary keyword arguments.  Passed to many, but not all,
            of the `_caput` calls.

        Returns
        -------
        success : bool
           Returns `True` if system setup succeeded, otherwise
           `False`.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_defaults_pv` :
              Loads the hardware defaults.

        """
        success=True
        self.log('Setting up...', (self.LOG_USER))

        # If active, disable hardware logging while doing setup.
        if self._hardware_logging_thread is not None:
            self.log('Hardware logging is enabled.  Pausing for setup.',
                     (self.LOG_USER))
            self.pause_hardware_logging()

        # Thermal OT protection - should this be moved after
        # setDefaults?
        ultrascale_temperature_limit_degC = (
            self._ultrascale_temperature_limit_degC)
        if ultrascale_temperature_limit_degC is not None:
            self.log('Setting ultrascale OT protection limit '+
                     f'to {ultrascale_temperature_limit_degC}C', self.LOG_USER)
            # OT threshold in degrees C
            self.set_ultrascale_ot_threshold(
                self._ultrascale_temperature_limit_degC,
                write_log=write_log)

        # Which bands are we configuring?
        bands = self._bands

        # Right now, resetting both DACs in both MicrowaveMuxCore blocks,
        # but may want to determine at runtime which are actually needed and
        # only reset the DAC in those.
        self.log('Toggling DACs')
        dacs = [0, 1]
        for val in [1, 0]:
            for bay in self.bays:

                # In newer software versions, setDefaults disables
                # DBG:enable after loading the defaults.yml.  This
                # makes sure we can reset the RF DACs.
                self.set_dbg_enable(bay, True)

                # Reset all RF DACs in use.
                for dac in dacs:
                    self.set_dac_reset(
                        bay, dac, val, write_log=write_log)

        #
        # setDefaults
        self.set_read_all(write_log=write_log)
        set_defaults_success = self.set_defaults_pv(write_log=write_log)

        # Checking if setDefaults succeeded is only supported for
        # pysmurf core code versions >=4.1.0.  If it's not supported,
        # self.set_defaults_pv will return None.  Overriding None with
        # True to skip this check for older versions of pysmurf that
        # don't support checking if setDefaults succeeded.
        if set_defaults_success is None:
            set_defaults_success = True

        # Log an error if setDefaults failed.
        if not set_defaults_success:
            self.log(
                'ERROR : System configuration/setDefaults failed!  Do'
                ' not proceed!  Reboot or ask someone for help.  You'
                ' are strongly encouraged to report this as an issue'
                ' on the pysmurf github repo at'
                ' https://github.com/slaclab/pysmurf/issues (please'
                ' provide a state dump using the pysmurf'
                ' set_read_all/save_state functions).',
                self.LOG_ERROR)
            success = False

        # Only proceed with the rest of setup if the defaults were set
        # correctly.
        if success:
            # setDefaults runs the JesdHealth check, so just need to poll
            # status.
            jesd_health_status = self.get_jesd_status(write_log=write_log)

            # Checking if JesdHealth is Locked is only supported for Rogue
            # ZIP file versions >=0.3.0 and pysmurf core code versions
            # >=4.1.0.  If it's not supported, self.set_check_jesd will
            # return None if the JesdHealth registers in SmurfApplication
            # aren't present or 'Not found' if they are present but the
            # JesdHealth method isn't implemented in the loaded Rogue ZIP
            # file.  Overriding None with True to skip this check for
            # older versions of pysmurf that don't support the JesdHealth
            # check.  Log an error if the JesdHealth check reports that
            # JESD is unlocked.
            if ( jesd_health_status is not None and
                 jesd_health_status != 'Not found' and
                 jesd_health_status != 'Locked' ):
                self.log(
                    'ERROR : JESD is not locked!  Do not proceed!'
                    ' Reboot or ask someone for help.  You are strongly'
                    ' encouraged to report this as an issue on the'
                    ' pysmurf github repo at'
                    ' https://github.com/slaclab/pysmurf/issues (please'
                    ' provide a state dump using the pysmurf'
                    ' set_read_all/save_state functions).',
                    self.LOG_ERROR)
                success = False

        # Only proceed with the rest of setup if defaults were set
        # correctly and basic system checks succeeded, otherwise we
        # risk giving users false hope.
        if success:
            # The per band configs. May want to make available per-band
            # values.
            for band in bands:
                self.set_iq_swap_in(band, self._iq_swap_in[band],
                                    write_log=write_log, **kwargs)
                self.set_iq_swap_out(band, self._iq_swap_out[band],
                                     write_log=write_log, **kwargs)

                if self._ref_phase_delay[band]:
                    self.set_ref_phase_delay(
                        band,
                        self._ref_phase_delay[band],
                        write_log=write_log, **kwargs)
                    self.set_ref_phase_delay_fine(
                        band,
                        self._ref_phase_delay_fine[band],
                        write_log=write_log, **kwargs)

                    # in DSPv3, lmsDelay should be 4*refPhaseDelay (says
                    # Mitch).  If none provided in cfg, enforce that
                    # constraint.  If provided in cfg, override with provided
                    # value.
                    if self._lms_delay[band] is None:
                        self.set_lms_delay(
                            band, int(4*self._ref_phase_delay[band]),
                            write_log=write_log, **kwargs)
                    else:
                        self.set_lms_delay(
                            band, self._lms_delay[band],
                            write_log=write_log, **kwargs)
                # we'll use the next band_delay_us
                else:
                    if self._band_delay_us[band] is None:
                        raise RuntimeError("Must define either refPhaseDelay " +
                                           "and refPhaseDelayFine or bandDelayUs")
                    self.set_band_delay_us(
                        band,
                        self._band_delay_us[band],
                        write_log=write_log, **kwargs)

                self.set_lms_gain(
                    band, self._lms_gain[band],
                    write_log=write_log, **kwargs)

                self.set_trigger_reset_delay(
                    band, self._trigger_reset_delay[band],
                    write_log=write_log, **kwargs)

                self.set_feedback_enable(
                    band, self._feedback_enable[band],
                    write_log=write_log, **kwargs)
                self.set_feedback_gain(
                    band, self._feedback_gain[band],
                    write_log=write_log, **kwargs)
                self.set_feedback_limit_khz(
                    band, self._feedback_limit_khz[band],
                    write_log=write_log, **kwargs)
                self.set_feedback_polarity(
                    band, self._feedback_polarity[band],
                    write_log=write_log, **kwargs)

                for dmx in np.array(self._data_out_mux[band]):
                    self.set_data_out_mux(
                        int(self.band_to_bay(band)), int(dmx),
                        "UserData", write_log=write_log, **kwargs)

                self.set_dsp_enable(
                    band, self._dsp_enable,
                    write_log=write_log, **kwargs)

                # Tuning defaults
                self.set_gradient_descent_gain(
                    band, self._gradient_descent_gain[band],
                    write_log=write_log, **kwargs)
                self.set_gradient_descent_averages(
                    band, self._gradient_descent_averages[band],
                    write_log=write_log, **kwargs)
                self.set_gradient_descent_converge_hz(
                    band, self._gradient_descent_converge_hz[band],
                    write_log=write_log, **kwargs)
                self.set_gradient_descent_step_hz(
                    band, self._gradient_descent_step_hz[band],
                    write_log=write_log, **kwargs)
                self.set_gradient_descent_momentum(
                    band, self._gradient_descent_momentum[band],
                    write_log=write_log, **kwargs)
                self.set_gradient_descent_beta(
                    band, self._gradient_descent_beta[band],
                    write_log=write_log, **kwargs)
                self.set_eta_scan_averages(
                    band, self._eta_scan_averages[band],
                    write_log=write_log, **kwargs)
                self.set_eta_scan_del_f(
                    band, self._eta_scan_del_f[band],
                    write_log=write_log, **kwargs)

            # Set UC and DC attenuators
            for band in bands:
                self.set_att_uc(
                    band, self._att_uc[band],
                    write_log=write_log)
                self.set_att_dc(
                    band, self._att_dc[band],
                    write_log=write_log)

            # Things that have to be done for both AMC bays, regardless of whether or not an AMC
            # is plugged in there.
            for bay in [0, 1]:
                self.set_trigger_hw_arm(bay, 0, write_log=write_log)

            self.set_trigger_width(0, 10, write_log=write_log)  # mystery bit that makes triggering work
            self.set_trigger_enable(0, 1, write_log=write_log)
            ## only sets enable, but is initialized to True already by
            ## default, and crashing for unknown reasons in rogue 4.
            self.set_evr_channel_reg_enable(0, True, write_log=write_log)
            self.set_evr_trigger_channel_reg_dest_sel(0,
                                                      0x20000,
                                                      write_log=write_log)

            self.set_enable_ramp_trigger(1, write_log=write_log)

            # 0x1 selects fast flux ramp, 0x0 selects slow flux ramp.  The
            # slow flux ramp only existed on the first rev of RTM boards,
            # C0, and wasn't ever really used.
            self.set_select_ramp(0x1, write_log=write_log)

            self.set_cpld_reset(0, write_log=write_log)
            self.cpld_toggle(write_log=write_log)
            self.all_off()

            # Make sure flux ramp starts off
            self.flux_ramp_off(write_log=write_log)
            self.flux_ramp_setup(self._reset_rate_khz,
                                 self._fraction_full_scale,
                                 write_log=write_log)

            # Turn on stream enable for all bands
            self.set_stream_enable(1, write_log=write_log)

            # Set payload size and mask to a single channel
            self.set_payload_size(payload_size)
            self.set_channel_mask([0])

            self.set_amplifier_bias(write_log=write_log)
            self.get_amplifier_bias()

            # also read the temperature of the CC
            self.log(f"Cryocard temperature = {self.C.read_temperature()}")

            # Setup how this slot handles timing. To take science data, each
            # SMuRF slot should receive timing from the backplane or RTM fiber
            # cable. Otherwise, the front panel external reference may also
            # receive timing. If from fiber, assume that we're on slot 2, and
            # distribute across the backplane. If from backplane, assume we're
            # not on slot 2, and receive timing from backplane. If external,
            # receive external reference from the front of the panel.
            if self._timing_reference is not None:
                timing_reference = self._timing_reference

                timing_options = ['ext_ref', 'backplane', 'fiber']
                assert (timing_reference in timing_options), (
                    'timing_reference in cfg file ' +
                    f'(={timing_reference}) not in ' +
                    f'timing_options={timing_options}')

                self.log(
                    'Configuring the system to take timing ' +
                    f'from {timing_reference}')

                if timing_reference == 'ext_ref':
                    for bay in self.bays:
                        self.log(f'Select external reference for bay {bay}' +
                                'or free running if there is no reference.')
                        self.sel_ext_ref(bay)

                    # Ramp on the internal clock.
                    self.set_ramp_start_mode(0, write_log=write_log)

                # The expected setup is that this slot is slot 2, and it
                # should distribute its fiber timing to the carrier's
                # backplane. Order does not matter.
                if timing_reference == 'fiber':
                    # FPGA_TIMING_OUT to RTM Timing In 0
                    self.set_crossbar_output_config(1, 0x0)
                    # Backplane DIST0 to RTM Timing In 0
                    self.set_crossbar_output_config(2, 0x0)
                    # Backplane Dist1 to RTM Timing In 0
                    self.set_crossbar_output_config(3, 0x0)

                    # EvrV2CoreTriggers EvrV2ChannelReg[0] EnableReg True
                    self.set_evr_channel_reg_enable(0, True)

                    # EvrV2CoreTriggers EvrV2ChannelReg[0] DestType All
                    self.set_evr_trigger_dest_type(0, 0)

                    # EvrV2CoreTriggers EVrV2TriggerReg[0] Enable Trig True
                    self.set_trigger_enable(0, True)

                    # RtmCryoDet RampStartMode 0x1
                    self.set_ramp_start_mode(1, write_log=write_log)

                    # MicrowaveMuxCore[0] LMK LmkReg_0x0147 0xA
                    for bay in self.bays:
                        self.log(f'Setting Bay {bay} LMK 0x146 to 0x08')
                        self.set_lmk_reg(bay, 0x146, 0x08)
                        self.log(f'Setting Bay {bay} LMK 0x147 to 0x0A')
                        self.set_lmk_reg(bay, 0x147, 0x0A)

                # https://confluence.slac.stanford.edu/display/SMuRF/Timing+Carrier#TimingCarrier-Howtoconfiguretodistributeoverbackplanefromslot2

                # Take timing from the backplane. The expected setup is
                # that this carrier is not in slot 2, and slot 2 is
                # distributing timing to the backplane. The order of these
                # commands does not matter.
                if timing_reference == 'backplane':
                    self.log('The cfg file requests backplane timing.')
                    # OutputConfig[1] = 0x2 configures the SMuRF carrier's
                    # FPGA to take the timing signals from the backplane
                    # (TO_FPGA = FROM_BACKPLANE)
                    self.log('Setting crossbar OutputConfig[1]=0x2 (TO_FPGA=FROM_BACKPLANE)')
                    self.set_crossbar_output_config(1, 2)

                    # EvrV2CoreTriggers EvrV2ChannelReg[0] EnableReg True
                    self.set_evr_channel_reg_enable(0, True)

                    # EvrV2CoreTriggers EvrV2ChannelReg[0] DestType All
                    self.set_evr_trigger_dest_type(0, 0)

                    # EvrV2CoreTriggers EVrV2TriggerReg[0] Enable Trig True
                    self.set_trigger_enable(0, True)

                    # Set the bay AMC LMK to CLKin0 or CLKin1.
                    for bay in self.bays:
                        self.log(f'Setting Bay {bay} LMK 0x146 to 0x08')
                        self.set_lmk_reg(bay, 0x146, 0x08)
                        self.log(f'Setting Bay {bay} LMK 0x147 to 0x0A')
                        self.set_lmk_reg(bay, 0x147, 0x0A)

                    # Configure RTM to trigger off of the timing system
                    self.set_ramp_start_mode(1, write_log=write_log)

            self.log('Done with setup.', self.LOG_USER)
        else:
            self.log('Setup failed!', self.LOG_ERROR)

        # If active, re-enable hardware logging after setup.
        if self._hardware_logging_thread is not None:
            self.log('Resuming hardware logging.', self.LOG_USER)
            self.resume_hardware_logging()

        # Assume if we made it here that configuration was successful.
        return success

    def make_dir(self, directory):
        """Create directory on file system at the specified path.

        Checks if a directory exists on the file system.  If directory
        does not already exist on the file system, creates it.

        Args
        ----
        directory : str
              Full path of directory to create on the file system.
        """
        # if directory doesn't already exist on the file system
        if not os.path.exists(directory):
            # create directory on file system.
            os.makedirs(directory)


    def get_timestamp(self, as_int=False):
        """Gets the current unix time timestamp.

        Gets the number of seconds that have elapsed since the Unix
        epoch, that is the time 00:00:00 UTC on 1 January 1970, minus
        leap seconds.

        Args
        ----
        as_int : bool, optional, default False
            Whether to return the timestamp as an integer.  If False,
            timestamp is returned as an integer.

        Returns
        -------
        timestamp : str or int
            Timestamp as a string, unless optional argument
            as_int=True, in which case returns timestamp as an
            integer.
        """
        timestamp = f'{time.time():.0f}'

        if as_int:
            return int(timestamp)

        return timestamp

    def add_output(self, key, val):
        """Adds key/value pair to pysmurf configuration dictionary.

        NEED LONGER DESCRIPTION OF ADD OUTPUT MEMBER FUNCTION HERE.

        Args
        ----
        key : any
            The name of the key to update.
        val : any
            The value to assign to the key.
        """
        self.config.update_subkey('outputs', key, val)


    def write_output(self, filename=None):
        """Writes internal pysmurf configuration to disk.

        Dump the current configuration to a file. This wraps around the config
        file writing in the config object. Files are timestamped and dumped to
        the S.output_dir by default.

        Args
        ----
        filename : str, optional, default None
              Full path of output configuration file to write to disk.
        """

        timestamp = self.get_timestamp()
        if filename is not None:
            output_file = filename
        else:
            output_file = timestamp + '.cfg'

        full_path = os.path.join(self.output_dir, output_file)
        self.config.write(full_path)
