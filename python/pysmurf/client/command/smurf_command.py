#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf command module - SmurfCommandMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/command/smurf_command.py
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
from pysmurf.client.base import SmurfBase
from pysmurf.client.command.sync_group import SyncGroup as SyncGroup
from pysmurf.client.util import tools
try:
    import epics
except ModuleNotFoundError:
    print("smurf_command.py - epics not found.")

class SmurfCommandMixin(SmurfBase):

    _global_poll_enable = ':AMCc:enable'

    def _caput(self, cmd, val, write_log=False, execute=True,
            wait_before=None, wait_after=None, wait_done=True,
            log_level=0, enable_poll=False, disable_poll=False,
            new_epics_root=None, **kwargs):
        r"""Puts variables into epics.

        Wrapper around pyrogue lcaput. Puts variables into epics.

        Args
        ----
        cmd : str
            The pyrogue command to be executed.
        val: any
            The value to put into epics
        write_log : bool, optional, default False
            Whether to log the data or not.
        execute : bool, optional, default True
            Whether to actually execute the command.
        wait_before : int, optional, default None
            If not None, the number of seconds to wait before issuing
            the command.
        wait_after : int, optional, default None
            If not None, the number of seconds to wait after issuing
            the command.
        wait_done : bool, optional, default True
            Wait for the command to be finished before returning.
        log_level : int, optional, default 0
            Log level.
        enable_poll : bool, optional, default False
            Allows requests of all PVs.
        disable_poll : bool, optional, default False
            Disables requests of all PVs after issueing command.
        new_epics_root : str, optional, default None
            Temporarily replaces current epics root with a new one.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `epics.caput` call.
        """
        if new_epics_root is not None:
            self.log(f'Temporarily using new epics root: {new_epics_root}')
            old_epics_root = self.epics_root
            self.epics_root = new_epics_root
            cmd = cmd.replace(old_epics_root, self.epics_root)

        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        if wait_before is not None:
            if write_log:
                self.log(f'Waiting {wait_before:3.2f} seconds before...',
                         self.LOG_USER)
            time.sleep(wait_before)

        if write_log:
            log_str = 'caput ' + cmd + ' ' + str(val)
            if self.offline:
                log_str = 'OFFLINE - ' + log_str
            self.log(log_str, log_level)

        if execute and not self.offline:
            epics.caput(cmd, val, wait=wait_done, **kwargs)

        if wait_after is not None:
            if write_log:
                self.log(f'Waiting {wait_after:3.2f} seconds after...',
                    self.LOG_USER)
            time.sleep(wait_after)
            if write_log:
                self.log('Done waiting.', self.LOG_USER)

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        if new_epics_root is not None:
            self.epics_root = old_epics_root
            self.log('Returning back to original epics root'+
                     f' : {self.epics_root}')

    def _caget(self, cmd, write_log=False, execute=True, count=None,
               log_level=0, enable_poll=False, disable_poll=False,
               new_epics_root=None, yml=None, **kwargs):
        r"""Gets variables from epics.

        Wrapper around pyrogue lcaget. Gets variables from epics.

        Args
        ----
        cmd : str
            The pyrogue command to be exectued.
        write_log : bool, optional, default False
            Whether to log the data or not.
        execute : bool, optional, default True
            Whether to actually execute the command.
        count : int or None, optional, default None
            Number of elements to return for array data.
        log_level : int, optional, default 0
            Log level.
        enable_poll : bool, optional, default False
            Allows requests of all PVs.
        disable_poll : bool, optional, default False
            Disables requests of all PVs after issueing command.
        new_epics_root : str or None, optional, default None
            Temporarily replaces current epics root with a new one.
        yml : str or None, optional, default None
            If not None, yaml file to parse for the result.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `epics.caget` call.

        Returns
        -------
        ret : str
            The requested value.
        """
        if new_epics_root is not None:
            self.log(f'Temporarily using new epics root: {new_epics_root}')
            old_epics_root = self.epics_root
            self.epics_root = new_epics_root
            cmd = cmd.replace(old_epics_root, self.epics_root)

        if enable_poll:
            epics.caput(self.epics_root+ self._global_poll_enable, True)

        if write_log:
            self.log('caget ' + cmd, log_level)

        # load the data from yml file if provided
        if yml is not None:
            if write_log:
                self.log(f'Reading from yml file\n {cmd}')
            return tools.yaml_parse(yml, cmd)

        elif execute and not self.offline:
            ret = epics.caget(cmd, count=count, **kwargs)
            if write_log:
                self.log(ret)
        else:
            ret = None

        if disable_poll:
            epics.caput(self.epics_root+ self._global_poll_enable, False)

        if new_epics_root is not None:
            self.epics_root = old_epics_root
            self.log('Returning back to original epics root'+
                     f' : {self.epics_root}')

        return ret


    #### Start SmurfApplication gets/sets
    _smurf_version = 'SmurfVersion'

    def get_pysmurf_version(self, **kwargs):
        r"""Returns the pysmurf version.

        Alias for `pysmurf.__version__`.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            pysmurf version.
        """
        return self._caget(self.smurf_application +
                           self._smurf_version, as_string=True,
                           **kwargs)

    _smurf_directory = 'SmurfDirectory'

    def get_pysmurf_directory(self, **kwargs):
        r"""Returns path to the pysmurf python files.

        Path to the files from which the pysmurf module was loaded.
        Alias for `pysmurf__file__`.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            Path to pysmurf files.
        """
        return self._caget(self.smurf_application +
                           self._smurf_directory, as_string=True,
                           **kwargs)

    _smurf_startup_script = 'StartupScript'

    def get_smurf_startup_script(self, **kwargs):
        r"""Returns path to the pysmurf server startup script.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            Path to pysmurf server startup script.
        """
        return self._caget(self.smurf_application +
                           self._smurf_startup_script, as_string=True,
                           **kwargs)

    _smurf_startup_arguments = 'StartupArguments'

    def get_smurf_startup_args(self, **kwargs):
        r"""Returns pysmurf server startup arguments.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            pysmurf server startup arguments.
        """
        return self._caget(self.smurf_application +
                           self._smurf_startup_arguments,
                           as_string=True, **kwargs)

    #### End SmurfApplication gets/sets

    _rogue_version = 'RogueVersion'

    def get_rogue_version(self, **kwargs):
        r"""Get rogue version

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            The rogue version
        """
        return self._caget(self.amcc + self._rogue_version,
                           as_string=True, **kwargs)

    def get_enable(self, **kwargs):
        """
        Returns
        -------
        str
            The status of the global poll bit epics_root:AMCc:enable.
            If False, pyrogue is not currently polling the server. PVs
            will not be updating.
        """
        return self._caget(self.epics_root + self._global_poll_enable,
                           enable_poll=False, disable_poll=False, **kwargs)


    _number_sub_bands = 'numberSubBands'

    def get_number_sub_bands(self, band=None, **kwargs):
        """
        Returns the number of subbands in a band.
        To do - possibly hide this function.

        Args
        ----
        band : int or None, optional, default None
            Which band.  If None, assumes all bands have the same
            number of sub bands, and pulls the number of sub bands
            from the first band in the list of bands specified in the
            experiment.cfg.

        Returns
        -------
        int
            The number of subbands in the band.
        """
        if self.offline:
            return 128

        if band is None:
            # assume all bands have the same number of channels, and
            # pull the number of channels from the first band in the
            # list of bands specified in experiment.cfg.
            bands = self.config.get('init').get('bands')
            band = bands[0]

        return self._caget(self._band_root(band) + self._number_sub_bands,
            **kwargs)


    _number_channels = 'numberChannels'

    def get_number_channels(self, band=None, **kwargs):
        """
        Returns the number of channels in a band.

        Args
        ----
        band : int or None, optional, default None
            Which band.  If None, assumes all bands have the same
            number of channels, and pulls the number of channels from
            the first band in the list of bands specified in the
            experiment.cfg.

        Returns
        -------
        int
            The number of channels in the band.
        """
        if self.offline:
            return 512  # Hard coded offline mode 512

        if band is None:
            # assume all bands have the same number of channels, and
            # pull the number of channels from the first band in the
            # list of bands specified in experiment.cfg.
            bands = self.config.get('init').get('bands')
            band = bands[0]

        return self._caget(self._band_root(band) + self._number_channels,
            **kwargs)

    def get_number_processed_channels(self, band=None, **kwargs):
        """
        Returns the number of processed channels in a band.

        Args
        ----
        band : int or None, optional, default None
            Which band.  If None, assumes all bands have the same
            number of channels, and pulls the number of channels from
            the first band in the list of bands specified in the
            experiment.cfg.

        Returns
        -------
        n_processed_channels : int
            The number of processed channels in the band.
        """
        n_channels=self.get_number_channels(band)

        n_processed_channels=int(0.8125*n_channels)
        return n_processed_channels

    def set_defaults_pv(self, **kwargs):
        """
        Sets the default epics variables
        """
        self._caput(self.epics_root + ':AMCc:setDefaults', 1, wait_after=20,
            **kwargs)
        self.log('Defaults are set.', self.LOG_INFO)


    def set_read_all(self, **kwargs):
        """
        ReadAll sends a command to read all register to the pyrogue server
        Registers must upated in order to PVs to update.
        This call is necesary to read register with pollIntervale=0.
        """
        self._caput(self.epics_root + ':AMCc:ReadAll', 1, wait_after=5,
            **kwargs)
        self.log('ReadAll sent', self.LOG_INFO)


    def run_pwr_up_sys_ref(self,bay, **kwargs):
        """
        """
        triggerPV=self.lmk.format(bay) + 'PwrUpSysRef'
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    _eta_scan_in_progress = 'etaScanInProgress'

    def get_eta_scan_in_progress(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_in_progress,
                    **kwargs)

    _gradient_descent_max_iters = 'gradientDescentMaxIters'

    def set_gradient_descent_max_iters(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_max_iters, val,
                    **kwargs)

    def get_gradient_descent_max_iters(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_max_iters,
                           **kwargs)


    _gradient_descent_averages = 'gradientDescentAverages'

    def set_gradient_descent_averages(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_averages, val,
                    **kwargs)

    def get_gradient_descent_averages(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_averages,
                           **kwargs)

    _gradient_descent_gain = 'gradientDescentGain'

    def set_gradient_descent_gain(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_gain, val,
                    **kwargs)

    def get_gradient_descent_gain(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_gain,
                           **kwargs)


    _gradient_descent_converge_hz = 'gradientDescentConvergeHz'

    def set_gradient_descent_converge_hz(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_converge_hz, val,
                    **kwargs)

    def get_gradient_descent_converge_hz(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_converge_hz,
                           **kwargs)

    _gradient_descent_step_hz = 'gradientDescentStepHz'

    def set_gradient_descent_step_hz(self, band, val, **kwargs):
        """
        Sets the step size of the gradient descent in units of Hz
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_step_hz, val,
                    **kwargs)

    def get_gradient_descent_step_hz(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_step_hz,
                           **kwargs)


    _gradient_descent_momentum = 'gradientDescentMomentum'

    def set_gradient_descent_momentum(self, band, val, **kwargs):
        """
        Sets the momentum term of the gradient descent
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_momentum, val,
                    **kwargs)

    def get_gradient_descent_momentum(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_momentum,
                           **kwargs)

    _gradient_descent_beta = 'gradientDescentBeta'

    def set_gradient_descent_beta(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._gradient_descent_beta, val,
                    **kwargs)

    def get_gradient_descent_beta(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._gradient_descent_beta,
                           **kwargs)

    def run_parallel_eta_scan(self, band, sync_group=True, **kwargs):
        """
        runParallelScan
        """
        triggerPV=self._cryo_root(band) + 'runParallelEtaScan'
        monitorPV=self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

        if sync_group:
            sg = SyncGroup([monitorPV])
            sg.wait()
            vals = sg.get_values()
            self.log('parallel etaScan complete ; etaScanInProgress = ' +
                f'{vals[monitorPV]}', self.LOG_USER)

    _run_serial_eta_scan = 'runSerialEtaScan'

    def run_serial_eta_scan(self, band, sync_group=True, timeout=240,
                            **kwargs):
        """
        Does an eta scan serially across the entire band. You must
        already be tuned close to the resontor dip. Use
        run_serial_gradient_descent to get it.

        Args
        ----
        band  : int
            The band to eta scan.
        sync_group : bool, optional, default True
            Whether to use the sync group to monitor the PV.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_eta_scan
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            sg.get_values()


    _run_serial_min_search = 'runSerialMinSearch'

    def run_serial_min_search(self, band, sync_group=True, timeout=240,
                              **kwargs):
        """
        Does a brute force search for the resonator minima. Starts at
        the currently set frequency.

        Args
        ----
        band : int
            The band the min search.
        sync_group : bool, optional, default True
            Whether to use the sync group to monitor the PV.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """
        triggerPV = self._cryo_root(band) + self._run_serial_min_search
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            sg.get_values()


    _run_serial_gradient_descent = 'runSerialGradientDescent'

    def run_serial_gradient_descent(self, band, sync_group=True,
                                    timeout=240, **kwargs):
        """
        Does a gradient descent search for the minimum.

        Args
        ----
        band : int
            The band to run serial gradient descent on.
        sync_group : bool, optional, default True
            Whether to use the sync group to monitor the PV.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_gradient_descent
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], timeout=timeout)
            sg.wait()
            sg.get_values()


    _selextref = "SelExtRef"

    def sel_ext_ref(self, bay, **kwargs):
        """
        Selects this bay to trigger off of external reference (through
        front panel)

        Args
        ----
        bay : int
            Which bay to set to ext ref.  Either 0 or 1.
        """
        assert (bay in [0,1]),'bay must be an integer and in [0,1]'
        triggerPV=self.microwave_mux_core.format(bay) + self._selextref
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    # name changed in Rogue 4 from WriteState to SaveState.  Keeping
    # the write_state function for backwards compatibilty.
    _savestate = ":AMCc:SaveState"

    def save_state(self, val, **kwargs):
        """
        Dumps all PyRogue state variables to a yml file.

        Args
        ----
        val : str
            The path (including file name) to write the yml file to.
        """
        self._caput(self.epics_root + self._savestate,
                    val, **kwargs)
    # alias older rogue 3 write_state function to save_state
    write_state = save_state

    # name changed in Rogue 4 from WriteConfig to SaveConfig.  Keeping
    # the write_config function for backwards compatibilty.
    _saveconfig = ":AMCc:SaveConfig"

    def save_config(self, val, **kwargs):
        """
        Writes the current (un-masked) PyRogue settings to a yml file.

        Args
        ----
        val : str
            The path (including file name) to write the yml file to.
        """
        self._caput(self.epics_root + self._saveconfig,
                    val, **kwargs)
    # alias older rogue 3 write_config function to save_config
    write_config = save_config

    _tone_file_path = 'CsvFilePath'

    def get_tone_file_path(self, bay, **kwargs):
        r"""Get tone file path.

        Returns the tone file path that's currently being used for
        this bay.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            Full path to tone file.
        """

        return self._caget(self.dac_sig_gen.format(bay) +
                           self._tone_file_path, as_string=True, **kwargs)

    def set_tone_file_path(self, bay, val, **kwargs):
        """
        Sets the tone file path for this bay.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        val : str
            Path (including csv file name) to tone file.
        """
        # make sure file exists before setting

        if not os.path.exists(val):
            self.log(f'Tone file {val} does not exist!  Doing nothing!',
                     self.LOG_ERROR)
            raise ValueError('Must provide a path to an existing tone file.')

        self._caput(self.dac_sig_gen.format(bay) + self._tone_file_path,
                    val, **kwargs)

    _load_tone_file = 'LoadCsvFile'

    def load_tone_file(self, bay, val=None, **kwargs):
        """
        Loads tone file specified in tone_file_path.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        val : str or None, optional, default None
            Path (including csv file name) to tone file.  If none
            provided, assumes something valid has already been loaded
            into DacSigGen[#]:CsvFilePath
        """

        # Set tone file path if provided.
        if val is not None:
            self.set_tone_file_path(bay, val)
        else:
            val=self.get_tone_file_path(bay)


        self.log(f'Loading tone file : {val}',
                 self.LOG_USER)
        self._caput(self.dac_sig_gen.format(bay) + self._load_tone_file, val,
            **kwargs)

    _tune_file_path = 'tuneFilePath'

    def set_tune_file_path(self, val, **kwargs):
        """
        """
        self._caput(self.sysgencryo + self._tune_file_path,
                    val, **kwargs)

    def get_tune_file_path(self, **kwargs):
        """
        """
        return self._caget(self.sysgencryo + self._tune_file_path,
                           **kwargs)

    _load_tune_file = 'loadTuneFile'

    def set_load_tune_file(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._load_tune_file,
                    val, **kwargs)

    def get_load_tune_file(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._load_tune_file,
            **kwargs)

    _eta_scan_del_f_reg = 'etaScanDelF'

    def set_eta_scan_del_f(self, band, val, **kwargs):
        """Sets offset frequency for serial eta scan estimation.

        The rogue serial eta scan routine (run using
        :func:`run_serial_eta_scan`) estimates the eta parameter for each
        tone with nonzero amplitude in the provided `band` by sampling
        the frequency error at the tone frequency +/- this offset
        frequency.  Units are Hz.

        Args
        ----
        band : int
           Which band.
        val : int
           Offset frequency in Hz about each resonator's central
           frequency at which to sample the frequency error in order
           to estimate the eta parameters of each resonator in the
           rogue serial eta scan routine.

        See Also
        --------
        :func:`run_serial_eta_scan` : Runs rogue serial eta scan, which uses
		this parameter.
        :func:`get_eta_scan_del_f` : Gets the current value of this
		parameter in rogue.
        """
        self._caput(
            self._cryo_root(band) + self._eta_scan_del_f_reg, val,
            **kwargs)

    def get_eta_scan_del_f(self, band, **kwargs):
        """Gets offset frequency for serial eta scan estimation.

        The rogue serial eta scan routine (run using
        :func:`run_serial_eta_scan`) estimates the eta parameter for each
        tone with nonzero amplitude in the provided `band` by sampling
        the frequency error at the tone frequency +/- this offset
        frequency.  Units are Hz.

        Args
        ----
        band : int
           Which band.

        Returns
        -------
        val : int
           Offset frequency in Hz about each resonator's central
           frequency at which to sample the frequency error in order
           to estimate the eta parameters of each resonator in the
           rogue serial eta scan routine.

        See Also
        --------
        :func:`run_serial_eta_scan` : Runs rogue serial eta scan, which uses
		this parameter.
        :func:`set_eta_scan_del_f` : Sets the value of this parameter in
		rogue.
        """        
        return self._caget(
            self._cryo_root(band) + self._eta_scan_del_f_reg,
            **kwargs)        

    _eta_scan_freqs = 'etaScanFreqs'

    def set_eta_scan_freq(self, band, val, **kwargs):
        """
        Sets the frequency to do the eta scan

        Args
        ----
        band : int
            The band to count.
        val : int
            The frequency to scan.
        """
        self._caput(self._cryo_root(band) + self._eta_scan_freqs, val,
            **kwargs)


    def get_eta_scan_freq(self, band, **kwargs):
        """
        Args
        ----
        band : int
            The band to count.

        Returns
        -------
        freq : int
            The frequency of the scan.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_freqs,
            **kwargs)

    _eta_scan_amplitude = 'etaScanAmplitude'

    def set_eta_scan_amplitude(self, band, val, **kwargs):
        """
        Sets the amplitude of the eta scan.

        Args
        ----
        band : int
            The band to set.
        val : int
            The eta scan amplitude. Typical value is 9 to 11.
        """
        self._caput(self._cryo_root(band) + self._eta_scan_amplitude, val,
            **kwargs)

    def get_eta_scan_amplitude(self, band, **kwargs):
        """
        Gets the amplitude of the eta scan.

        Args
        ----
        band : int
            The band to set.

        Returns
        -------
        amp : int
            The eta scan amplitude.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_amplitude,
            **kwargs)

    _eta_scan_channel = 'etaScanChannel'

    def set_eta_scan_channel(self, band, val, **kwargs):
        """
        Sets the channel to eta scan.

        Args
        ----
        band : int
            The band to set.
        val : int
            The channel to set.
        """
        self._caput(self._cryo_root(band) + self._eta_scan_channel, val,
            **kwargs)

    def get_eta_scan_channel(self, band, **kwargs):
        """
        Gets the channel to eta scan.

        Args
        ----
        band : int
            The band to set.

        Returns
        -------
        chan : int
            The channel that is being eta scanned.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_channel,
            **kwargs)

    _eta_scan_averages = 'etaScanAverages'

    def set_eta_scan_averages(self, band, val, **kwargs):
        """
        Sets the number of frequency error averages to take at each point of
        the etaScan.

        Args
        ----
        band : int
            The band to set.
        val : int
            The channel to set.
        """
        self._caput(self._cryo_root(band) + self._eta_scan_averages, val,
            **kwargs)

    def get_eta_scan_averages(self, band, **kwargs):
        """
        Gets the number of frequency error averages taken at each point of
        the etaScan.

        Args
        ----
        band : int
            The band to set.

        Returns
        -------
        int
            The number of frequency error averages taken at each point
            of the etaScan.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_averages,
            **kwargs)

    _eta_scan_dwell = 'etaScanDwell'

    def set_eta_scan_dwell(self, band, val, **kwargs):
        """
        Swets how long to dwell while eta scanning.

        Args
        ----
        band : int
            The band to eta scan.
        val : int
            The time to dwell.
        """
        self._caput(self._cryo_root(band) + self._eta_scan_dwell, val, **kwargs)

    def get_eta_scan_dwell(self, band, **kwargs):
        """
        Gets how long to dwell

        Args
        ----
        band : int
            The band being eta scanned.

        Returns
        -------
        dwell : int
            The time to dwell during an eta scan.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_dwell,
            **kwargs)

    _run_eta_scan = 'runEtaScan'

    def set_run_eta_scan(self, band, val, **kwargs):
        """
        Runs the eta scan. Set the channel using set_eta_scan_channel()

        Args
        ----
        band : int
            The band to eta scan.
        val : bool
            Start the eta scan.
        """
        self._caput(self._cryo_root(band) + self._run_eta_scan, val, **kwargs)

    def get_run_eta_scan(self, band, **kwargs):
        """
        Gets the status of eta scan.

        Args
        ----
        band : int
            The band that is being checked.

        Returns
        -------
        status : int
            Whether the band is eta scanning.
        """
        return self._caget(self._cryo_root(band) + self._run_eta_scan, **kwargs)

    _eta_scan_results_real = 'etaScanResultsReal'

    def get_eta_scan_results_real(self, band, count, **kwargs):
        """
        Gets the real component of the eta scan.

        Args
        ----
        band : int
            The to get eta scans.
        count : int
            The number of samples to read.

        Returns
        -------
        resp : float array
            The real component of the most recent eta scan.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_results_real,
            count=count, **kwargs)

    _eta_scan_results_imag = 'etaScanResultsImag'

    def get_eta_scan_results_imag(self, band, count, **kwargs):
        """
        Gets the imaginary component of the eta scan.

        Args
        ----
        band : int
            The to get eta scans.
        count : int
            The number of samples to read.

        Returns
        -------
        resp : float array
            The imaginary component of the most recent eta scan.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_results_imag,
            count=count, **kwargs)

    _amplitude_scales = 'setAmplitudeScales'

    def set_amplitude_scales(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._amplitude_scales, val,
            **kwargs)

    def get_amplitude_scales(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._amplitude_scales,
            **kwargs)

    _amplitude_scale_array = 'amplitudeScaleArray'

    def set_amplitude_scale_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._amplitude_scale_array, val,
            **kwargs)

    def get_amplitude_scale_array(self, band, **kwargs):
        """
        Gets the array of amplitudes

        Args
        ----
        band : int
            The band to search.

        Returns
        -------
        amplitudes : array
            The tone amplitudes.
        """
        return self._caget(self._cryo_root(band) + self._amplitude_scale_array,
            **kwargs)

    def set_amplitude_scale_array_currentchans(self, band, drive, **kwargs):
        """
        Set only the currently on channels to a new drive power. Essentially
        a more convenient wrapper for set_amplitude_scale_array to only change
        the channels that are on.

        Args
        ----
        band : int
            The band to change.
        drive : int
            Tone power to change to.
        """

        old_amp = self.get_amplitude_scale_array(band, **kwargs)
        n_channels=self.get_number_channels(band)
        new_amp = np.zeros((n_channels,),dtype=int)
        new_amp[np.where(old_amp!=0)] = drive
        self.set_amplitude_scale_array(self, new_amp, **kwargs)

    _feedback_enable_array = 'feedbackEnableArray'

    def set_feedback_enable_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._feedback_enable_array, val,
            **kwargs)

    def get_feedback_enable_array(self, band, **kwargs):
        """
        Gets the array of feedbacks enables

        Args
        ----
        band : int
            The band to search.

        Returns
        -------
        fb_on : bool array
            An array of whether the feedback is on or off.
        """
        return self._caget(self._cryo_root(band) + self._feedback_enable_array,
            **kwargs)

    _single_channel_readout = 'singleChannelReadout'

    def set_single_channel_readout(self, band, val, **kwargs):
        """
        Sets the singleChannelReadout bit.

        Args
        ----
        band : int
            The band to set to single channel readout.
        """
        self._caput(self._band_root(band) + self._single_channel_readout, val,
            **kwargs)

    def get_single_channel_readout(self, band, **kwargs):
        """

        """
        return self._caget(self._band_root(band) + self._single_channel_readout,
            **kwargs)

    _single_channel_readout2 = 'singleChannelReadoutOpt2'

    def set_single_channel_readout_opt2(self, band, val, **kwargs):
        """
        Sets the singleChannelReadout2 bit.

        Args
        ----
        band : int
            The band to set to single channel readout.
        """
        self._caput(self._band_root(band) + self._single_channel_readout2, val,
            **kwargs)

    def get_single_channel_readout_opt2(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._single_channel_readout2,
            **kwargs)

    _readout_channel_select = 'readoutChannelSelect'

    def set_readout_channel_select(self, band, channel, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._readout_channel_select,
                    channel, **kwargs)

    def get_readout_channel_select(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) +
                           self._readout_channel_select, **kwargs)


    _stream_enable = 'enableStreaming'

    def set_stream_enable(self, val, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        self._caput(self.app_core + self._stream_enable, val, **kwargs)

    def get_stream_enable(self, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        return self._caget(self.app_core + self._stream_enable,
            **kwargs)

    _build_dsp_g = 'BUILD_DSP_G'

    def get_build_dsp_g(self, **kwargs):
        """
        BUILD_DSP_G encodes which bands the fw being used was built for.
        E.g. 0xFF means Base[0...7], 0xF is Base[0...3], etc.

        """
        return self._caget(self.app_core + self._build_dsp_g,
            **kwargs)

    _decimation = 'decimation'

    def set_decimation(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._decimation, val, **kwargs)

    def get_decimation(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._decimation, **kwargs)

    _filter_alpha = 'filterAlpha'

    def set_filter_alpha(self, band, val, **kwargs):
        """
        Coefficient for single pole low pass fitler before readout (when c
        hannels are multiplexed, decimated)
        y[n] = alpha*x[n] + (1 - alpha)*y[n-1]
        matlab to visualize
        h = fvtool([alpha], [1 -(1-alpha)]); h.Fs = 2.4e6;
        """
        self._caput(self._band_root(band) + self._filter_alpha, val, **kwargs)

    def get_filter_alpha(self, band, **kwargs):
        """
        Coefficient for single pole low pass fitler before readout (when
        channels are multiplexed, decimated)
        y[n] = alpha*x[n] + (1 - alpha)*y[n-1]
        matlab to visualize
        h = fvtool([alpha], [1 -(1-alpha)]); h.Fs = 2.4e6;
        """
        return self._caget(self._band_root(band) + self._filter_alpha, **kwargs)

    _iq_swap_in = 'iqSwapIn'

    def set_iq_swap_in(self, band, val, **kwargs):
        """
        Swaps I&Q into DSP (from ADC).  Tones being output by the system will
        flip about the band center (e.g. 4.25GHz, 5.25GHz etc.)
        """
        self._caput(self._band_root(band) + self._iq_swap_in, val, **kwargs)

    def get_iq_swap_in(self, band, **kwargs):
        """
        Swaps I&Q into DSP (from ADC).  Tones being output by the system will
        flip about the band center (e.g. 4.25GHz, 5.25GHz etc.)
        """
        return self._caget(self._band_root(band) + self._iq_swap_in, **kwargs)

    _iq_swap_out = 'iqSwapOut'

    def set_iq_swap_out(self, band, val, **kwargs):
        """
        Swaps I&Q out of DSP (to DAC).  Swapping I&Q flips spectrum around band
        center.
        """
        self._caput(self._band_root(band) + self._iq_swap_out, val, **kwargs)

    def get_iq_swap_out(self, band, **kwargs):
        """
        Swaps I&Q out of DSP (to DAC).  Swapping I&Q flips spectrum around band
        center.
        """
        return self._caget(self._band_root(band) + self._iq_swap_out, **kwargs)

    _ref_phase_delay = 'refPhaseDelay'

    def set_ref_phase_delay(self, band, val, **kwargs):
        """
        Corrects for roundtrip cable delay
        freqError = IQ * etaMag, rotated by etaPhase+refPhaseDelay
        """
        self._caput(self._band_root(band) + self._ref_phase_delay, val,
            **kwargs)

    def get_ref_phase_delay(self, band, **kwargs):
        """
        Corrects for roundtrip cable delay
        freqError = IQ * etaMag, rotated by etaPhase+refPhaseDelay
        """
        return self._caget(self._band_root(band) + self._ref_phase_delay,
            **kwargs)

    _ref_phase_delay_fine = 'refPhaseDelayFine'

    def set_ref_phase_delay_fine(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._ref_phase_delay_fine, val,
        **kwargs)

    def get_ref_phase_delay_fine(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._ref_phase_delay_fine,
            **kwargs)

    _tone_scale = 'toneScale'

    def set_tone_scale(self, band, val, **kwargs):
        """
        Scales the sum of 16 tones before synthesizer.
        """
        self._caput(self._band_root(band) + self._tone_scale, val, **kwargs)

    def get_tone_scale(self, band, **kwargs):
        """
        Scales the sum of 16 tones before synthesizer.
        """
        return self._caget(self._band_root(band) + self._tone_scale, **kwargs)

    _waveform_select = 'waveformSelect'

    def set_waveform_select(self, band, val, **kwargs):
        """
        0x0 select DSP -> DAC
        0x1 selects waveform table -> DAC (toneFile)
        """
        self._caput(self._band_root(band) + self._waveform_select, val,
            **kwargs)

    def get_waveform_select(self, band, **kwargs):
        """
        0x0 select DSP -> DAC
        0x1 selects waveform table -> DAC (toneFile)
        """
        return self._caget(self._band_root(band) + self._waveform_select,
            **kwargs)

    _waveform_start = 'waveformStart'

    def set_waveform_start(self, band, val, **kwargs):
        """
        0x1 enables waveform table
        """
        self._caput(self._band_root(band) + self._waveform_start, val,
            **kwargs)

    def get_waveform_start(self, band, **kwargs):
        """
        0x1 enables waveform table
        """
        return self._caget(self._band_root(band) + self._waveform_start,
            **kwargs)

    _rf_enable = 'rfEnable'

    def set_rf_enable(self, band, val, **kwargs):
        """
        0x0 output all 0s to DAC
        0x1 enable output to DAC (from DSP or waveform table)
        """
        self._caput(self._band_root(band) + self._rf_enable, val,
            **kwargs)

    def get_rf_enable(self, band, **kwargs):
        """
        0x0 output all 0s to DAC
        0x1 enable output to DAC (from DSP or waveform table)
        """
        return self._caget(self._band_root(band) + self._rf_enable, **kwargs)

    _analysis_scale = 'analysisScale'

    def set_analysis_scale(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._analysis_scale, val, **kwargs)

    def get_analysis_scale(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._analysis_scale,
            **kwargs)

    _feedback_enable = 'feedbackEnable'

    def set_feedback_enable(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._feedback_enable, val,
            **kwargs)

    def get_feedback_enable(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._feedback_enable,
            **kwargs)

    _loop_filter_output_array = 'loopFilterOutputArray'

    def get_loop_filter_output_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._loop_filter_output_array,
            **kwargs)

    _tone_frequency_offset_mhz = 'toneFrequencyOffsetMHz'

    def get_tone_frequency_offset_mhz(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) +
                           self._tone_frequency_offset_mhz,**kwargs)

    def set_tone_frequency_offset_mhz(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) +
                    self._tone_frequency_offset_mhz, val,
                    **kwargs)

    _center_frequency_array = 'centerFrequencyArray'

    def set_center_frequency_array(self, band, val, **kwargs):
        """
        Sets all the center frequencies in a band
        """
        self._caput(self._cryo_root(band) + self._center_frequency_array, val,
            **kwargs)

    def get_center_frequency_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._center_frequency_array,
            **kwargs)

    _feedback_gain = 'feedbackGain'

    def set_feedback_gain(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._feedback_gain, val, **kwargs)

    def get_feedback_gain(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._feedback_gain,
            **kwargs)

    _eta_phase_array = 'etaPhaseArray'

    def set_eta_phase_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._eta_phase_array, val,
            **kwargs)

    def get_eta_phase_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_phase_array,
            **kwargs)

    _frequency_error_array = 'frequencyErrorArray'

    def set_frequency_error_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._frequency_error_array, val,
            **kwargs)

    def get_frequency_error_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._frequency_error_array,
            **kwargs)

    _eta_mag_array = 'etaMagArray'

    def set_eta_mag_array(self, band, val, **kwargs):
        """
        """
        self._caput(self._cryo_root(band) + self._eta_mag_array, val, **kwargs)

    def get_eta_mag_array(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_mag_array,
            **kwargs)

    _feedback_limit = 'feedbackLimit'

    def set_feedback_limit(self, band, val, **kwargs):
        """
        freq = centerFreq + feedbackFreq
        abs(freq) < centerFreq + feedbackLimit
        """
        self._caput(self._band_root(band) + self._feedback_limit, val, **kwargs)

    def get_feedback_limit(self, band, **kwargs):
        """
        freq = centerFreq + feedbackFreq
        abs(freq) < centerFreq + feedbackLimit
        """
        return self._caget(self._band_root(band) + self._feedback_limit,
            **kwargs)

    _noise_select = 'noiseSelect'

    def set_noise_select(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._noise_select, val,
            **kwargs)

    def get_noise_select(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._noise_select,
            **kwargs)

    _lms_delay = 'lmsDelay'

    def set_lms_delay(self, band, val, **kwargs):
        """
        Match system latency for LMS feedback (2.4MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_delay, val, **kwargs)

    def get_lms_delay(self, band, **kwargs):
        """
        Match system latency for LMS feedback (2.4MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_delay, **kwargs)

    _lms_gain = 'lmsGain'

    def set_lms_gain(self, band, val, **kwargs):
        """
        LMS gain, powers of 2
        """
        self._caput(self._band_root(band) + self._lms_gain, val, **kwargs)

    def get_lms_gain(self, band, **kwargs):
        """
        LMS gain, powers of 2
        """
        return self._caget(self._band_root(band) + self._lms_gain, **kwargs)

    _trigger_reset_delay = 'trigRstDly'

    def set_trigger_reset_delay(self, band, val, **kwargs):
        """
        Trigger reset delay, set such that the ramp resets at the flux ramp glitch.  2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        self._caput(self._band_root(band) + self._trigger_reset_delay, val, **kwargs)

    def get_trigger_reset_delay(self, band, **kwargs):
        """
        Trigger reset delay, set such that the ramp resets at the flux ramp glitch.  2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        return self._caget(self._band_root(band) + self._trigger_reset_delay, **kwargs)

    _feedback_start = 'feedbackStart'

    def set_feedback_start(self, band, val, **kwargs):
        """
        The flux ramp DAC value at which to start applying feedback in each flux ramp cycle.
        In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        self._caput(self._band_root(band) + self._feedback_start, val, **kwargs)

    def get_feedback_start(self, band, **kwargs):
        """
        The flux ramp DAC value at which to start applying feedback in each flux ramp cycle.
        In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        return self._caget(self._band_root(band) + self._feedback_start, **kwargs)

    _feedback_end = 'feedbackEnd'

    def set_feedback_end(self, band, val, **kwargs):
        """
        The flux ramp DAC value at which to stop applying feedback in each flux ramp cycle.
        In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        self._caput(self._band_root(band) + self._feedback_end, val, **kwargs)

    def get_feedback_end(self, band, **kwargs):
        """
        The flux ramp DAC value at which to stop applying feedback in each flux ramp cycle.
        In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        return self._caget(self._band_root(band) + self._feedback_end, **kwargs)

    _lms_enable1 = 'lmsEnable1'

    def set_lms_enable1(self, band, val, **kwargs):
        """
        Enable 1st harmonic tracking
        """
        self._caput(self._band_root(band) + self._lms_enable1, val, **kwargs)

    def get_lms_enable1(self, band, **kwargs):
        """
        Enable 1st harmonic tracking
        """
        return self._caget(self._band_root(band) + self._lms_enable1, **kwargs)

    _lms_enable2 = 'lmsEnable2'

    def set_lms_enable2(self, band, val, **kwargs):
        """
        Enable 2nd harmonic tracking
        """
        self._caput(self._band_root(band) + self._lms_enable2, val, **kwargs)

    def get_lms_enable2(self, band, **kwargs):
        """
        Enable 2nd harmonic tracking
        """
        return self._caget(self._band_root(band) + self._lms_enable2, **kwargs)

    _lms_enable3 = 'lmsEnable3'

    def set_lms_enable3(self, band, val, **kwargs):
        """
        Enable 3rd harmonic tracking
        """
        self._caput(self._band_root(band) + self._lms_enable3, val, **kwargs)

    def get_lms_enable3(self, band, **kwargs):
        """
        Enable 3rd harmonic tracking
        """
        return self._caget(self._band_root(band) + self._lms_enable3, **kwargs)

    _lms_rst_dly = 'lmsRstDly'

    def set_lms_rst_dly(self, band, val, **kwargs):
        """
        Disable feedback after reset (2.4MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_rst_dly, val, **kwargs)

    def get_lms_rst_dly(self, band, **kwargs):
        """
        Disable feedback after reset (2.4MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_rst_dly, **kwargs)

    _lms_freq = 'lmsFreq'

    def set_lms_freq(self, band, val, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        self._caput(self._band_root(band) + self._lms_freq, val, **kwargs)

    def get_lms_freq(self, band, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        return self._caget(self._band_root(band) + self._lms_freq, **kwargs)

    _lms_freq_hz = 'lmsFreqHz'

    def set_lms_freq_hz(self, band, val, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        self._caput(self._band_root(band) + self._lms_freq_hz, val, **kwargs)

    def get_lms_freq_hz(self, band, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        return self._caget(self._band_root(band) + self._lms_freq_hz, **kwargs)

    _lms_dly_fine = 'lmsDlyFine'

    def set_lms_dly_fine(self, band, val, **kwargs):
        """
        fine delay control (38.4MHz ticks)
        """
        self._caput(self._band_root(band) + self._lms_dly_fine, val, **kwargs)

    def get_lms_dly_fine(self, band, **kwargs):
        """
        fine delay control (38.4MHz ticks)
        """
        return self._caget(self._band_root(band) + self._lms_dly_fine, **kwargs)

    _iq_stream_enable = 'iqStreamEnable'

    def set_iq_stream_enable(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._iq_stream_enable, val,
            **kwargs)

    def get_iq_stream_enable(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._iq_stream_enable,
            **kwargs)

    _feedback_polarity = 'feedbackPolarity'

    def set_feedback_polarity(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._feedback_polarity, val,
            **kwargs)

    def get_feedback_polarity(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._feedback_polarity,
            **kwargs)

    _band_center_mhz = 'bandCenterMHz'

    def set_band_center_mhz(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._band_center_mhz, val,
            **kwargs)

    def get_band_center_mhz(self, band, **kwargs):
        """
        Returns the center frequency of the band in MHz
        """
        if self.offline:
            bc = (4250 + band*500)
            return bc
        else:
            return self._caget(self._band_root(band) + self._band_center_mhz,
                **kwargs)


    _channel_frequency_mhz = 'channelFrequencyMHz'

    def get_channel_frequency_mhz(self, band=None, **kwargs):
        """
        Returns the channel frequency in MHz.  The channel frequency
        is the rate at which channels are processed.

        Args
        ----
        band : int or None, optional, default None
           Which band.  If None, assumes all bands have the same
           channel frequency, and pulls the channel frequency from the
           first band in the list of bands specified in the
           experiment.cfg.

        Returns
        -------
        float
            The rate at which channels in this band are processed.
        """

        if band is None:
            # assume all bands have the same number of channels, and
            # pull the number of channels from the first band in the
            # list of bands specified in experiment.cfg.
            bands = self.config.get('init').get('bands')
            band = bands[0]

        if self.offline:
            return 2.4
        else:
            return self._caget(self._band_root(band) +
                self._channel_frequency_mhz, **kwargs)

    _digitizer_frequency_mhz = 'digitizerFrequencyMHz'

    def get_digitizer_frequency_mhz(self, band=None, **kwargs):
        """
        Returns the digitizer frequency in MHz.

        Args
        ----
        band : int or None, optional, default None
           Which band.  If None, assumes all bands have the same
           channel frequency, and pulls the channel frequency from the
           first band in the list of bands specified in the
           experiment.cfg.

        Returns
        -------
        float
            The digitizer frequency for this band in MHz.
        """
        if self.offline:
            return 614.4

        if band is None:
            # assume all bands have the same number of channels, and
            # pull the number of channels from the first band in the
            # list of bands specified in experiment.cfg.
            bands = self.config.get('init').get('bands')
            band = bands[0]

        return self._caget(self._band_root(band) +
            self._digitizer_frequency_mhz, **kwargs)

    _synthesis_scale = 'synthesisScale'

    def set_synthesis_scale(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._synthesis_scale, val,
            **kwargs)

    def get_synthesis_scale(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._synthesis_scale,
            **kwargs)

    _dsp_enable = 'dspEnable'

    def set_dsp_enable(self, band, val, **kwargs):
        """
        """
        self._caput(self._band_root(band) + self._dsp_enable, val, **kwargs)

    def get_dsp_enable(self, band, **kwargs):
        """
        """
        return self._caget(self._band_root(band) + self._dsp_enable, **kwargs)

    # Single channel commands
    _feedback_enable = 'feedbackEnable'

    def set_feedback_enable_channel(self, band, channel, val, **kwargs):
        """
        Set the feedback for a single channel
        """
        self._caput(self._channel_root(band, channel) +
            self._feedback_enable, val, **kwargs)

    def get_feedback_enable_channel(self, band, channel, **kwargs):
        """
        Get the feedback for a single channel
        """
        return self._caget(self._channel_root(band, channel) +
            self._feedback_enable, **kwargs)

    _eta_mag_scaled_channel = 'etaMagScaled'

    def set_eta_mag_scaled_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) +
            self._eta_mag_scaled_channel, val, **kwargs)

    def get_eta_mag_scaled_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._eta_mag_scaled_channel, **kwargs)


    _center_frequency_mhz_channel = 'centerFrequencyMHz'

    def set_center_frequency_mhz_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) +
            self._center_frequency_mhz_channel, val, **kwargs)

    def get_center_frequency_mhz_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._center_frequency_mhz_channel, **kwargs)


    _amplitude_scale_channel = 'amplitudeScale'

    def set_amplitude_scale_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) +
            self._amplitude_scale_channel, val, **kwargs)

    def get_amplitude_scale_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._amplitude_scale_channel, **kwargs)

    _eta_phase_degree_channel = 'etaPhaseDegree'

    def set_eta_phase_degree_channel(self, band, channel, val, **kwargs):
        """
        """
        self._caput(self._channel_root(band, channel) +
            self._eta_phase_degree_channel, val, **kwargs)

    def get_eta_phase_degree_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._eta_phase_degree_channel, **kwargs)

    _frequency_error_mhz = 'frequencyErrorMHz'

    def get_frequency_error_mhz(self, band, channel, **kwargs):
        """
        """
        return self._caget(self._channel_root(band, channel) +
            self._frequency_error_mhz, **kwargs)


    def band_to_bay(self,b):
        """
        Returns the bay index for the band.
        Assumes LB is plugged into bay 0,  corresponding to bands [0,1,2,3] and that
        HB is plugged into bay 1, corresponding to bands [4,5,6,7].

        Args
        ----
        b : int
            Band number.
        """
        if b in [0,1,2,3]:
            bay=0
        elif b in [4,5,6,7]:
            bay=1
        else:
            assert False, 'band supplied to band_to_bay() must be and ' + \
                'integer in [0,1,2,3,4,5,6,7] ...'
        return bay

    # Attenuator
    _uc = 'UC[{}]'

    def set_att_uc(self, b, val, **kwargs):
        """
        Set the upconverter attenuator

        Args
        ----
        b : int
            The band number.
        val : int
            The attenuator value.
        """
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        self._caput(self.att_root.format(bay) + self._uc.format(int(att)), val,
            **kwargs)

    def get_att_uc(self, b, **kwargs):
        """
        Get the upconverter attenuator value

        Args
        ----
        b : int
            The band number.
        """
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        return self._caget(self.att_root.format(bay) + self._uc.format(int(att)),
            **kwargs)


    _dc = 'DC[{}]'

    def set_att_dc(self, b, val, **kwargs):
        """
        Set the down-converter attenuator

        Args
        ----
        b : int
            The band number.
        val : int
            The attenuator value
        """
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        self._caput(self.att_root.format(bay) + self._dc.format(int(att)), val,
            **kwargs)

    def get_att_dc(self, b, **kwargs):
        """
        Get the down-converter attenuator value

        Args
        ----
        b : int
            The band number.
        """
        att = self.band_to_att(b)
        bay = self.band_to_bay(b)
        return self._caget(self.att_root.format(bay) + self._dc.format(int(att)),
            **kwargs)

    # ADC commands
    _adc_remap = "Remap[0]"  # Why is this hardcoded 0

    def set_remap(self, **kwargs):
        """
        This command should probably be renamed to something more descriptive.
        """
        self._caput(self.adc_root + self._adc_remap, 1, **kwargs)

    # DAC commands
    _dac_temp = "Temperature"

    def get_dac_temp(self, bay, dac, **kwargs):
        """
        Get temperature of the DAC in celsius

        Args
        ----
        bay : int
            Which bay [0 or 1].
        dac : int
            Which DAC no. [0 or 1].
        """
        return self._caget(self.dac_root.format(bay,dac) + self._dac_temp, **kwargs)

    _dac_enable = "enable"

    def set_dac_enable(self, bay, dac, val, **kwargs):
        """
        Enables DAC

        Args
        ----
        bay : int
            Which bay [0 or 1].
        dac : int
            Which DAC no. [0 or 1].
        val : int
            Value to set the DAC enable register to [0 or 1].
        """
        self._caput(self.dac_root.format(bay,dac) + self._dac_enable, val, **kwargs)

    def get_dac_enable(self, bay, dac, **kwargs):
        """
        Gets enable status of DAC

        Args
        ----
        bay : int
            Which bay [0 or 1].
        dac : int
            Which DAC no. [0 or 1].
        """
        return self._caget(self.dac_root.format(bay,dac) + self._dac_enable, **kwargs)

    # Jesd commands
    _data_out_mux = 'dataOutMux[{}]'

    def set_data_out_mux(self, bay, b, val, **kwargs):
        """
        """
        self._caput(self.jesd_tx_root.format(bay) +
            self._data_out_mux.format(b), val, **kwargs)

    def get_data_out_mux(self, bay, b, **kwargs):
        """
        """
        return self._caget(self.jesd_tx_root.format(bay) + self._data_out_mux.format(b),
            **kwargs)

    # Jesd DAC commands
    _jesd_reset_n = "JesdRstN"

    def set_jesd_reset_n(self, bay, dac, val, **kwargs):
        """
        Set DAC JesdRstN

        Args
        ----
        bay : int
            Which bay [0 or 1].
        dac : int
            Which DAC no. [0 or 1].
        val : int
            Value to set JesdRstN to [0 or 1].
        """
        self._caput(self.dac_root.format(bay,dac) + self._jesd_reset_n, val, **kwargs)

    _jesd_rx_enable = 'Enable'

    def get_jesd_rx_enable(self, bay, **kwargs):
        return self._caget(self.jesd_rx_root.format(bay) + self._jesd_rx_enable, **kwargs)

    def set_jesd_rx_enable(self, bay, val, **kwargs):
        self._caput(self.jesd_rx_root.format(bay) + self._jesd_rx_enable, val, **kwargs)

    _jesd_rx_status_valid_cnt = 'StatusValidCnt'

    def get_jesd_rx_status_valid_cnt(self, bay, num, **kwargs):
        return self._caget(self.jesd_rx_root.format(bay) + self._jesd_rx_status_valid_cnt + f'[{num}]', **kwargs)

    _jesd_rx_data_valid = 'DataValid'

    def get_jesd_rx_data_valid(self, bay, **kwargs):
        return self._caget(self.jesd_rx_root.format(bay) +
            self._jesd_rx_data_valid, **kwargs)

    _link_disable = 'LINK_DISABLE'

    def set_jesd_link_disable(self, bay, val, **kwargs):
        """
        Disables jesd link
        """
        self._caput(self.jesd_rx_root.format(bay) + self._link_disable, val, **kwargs)

    def get_jesd_link_disable(self, bay, **kwargs):
        """
        Disables jesd link
        """
        return self._caget(self.jesd_rx_root.format(bay) + self._link_disable,
            **kwargs)

    _jesd_tx_enable = 'Enable'

    def get_jesd_tx_enable(self, bay, **kwargs):
        return self._caget(self.jesd_tx_root.format(bay) + self._jesd_tx_enable, **kwargs)

    def set_jesd_tx_enable(self, bay, val, **kwargs):
        self._caput(self.jesd_tx_root.format(bay) + self._jesd_tx_enable, val, **kwargs)

    _jesd_tx_data_valid = 'DataValid'

    def get_jesd_tx_data_valid(self, bay, **kwargs):
        return self._caget(self.jesd_tx_root.format(bay) +
            self._jesd_tx_data_valid, **kwargs)

    _jesd_tx_status_valid_cnt = 'StatusValidCnt'

    def get_jesd_tx_status_valid_cnt(self, bay, num, **kwargs):
        return self._caget(self.jesd_tx_root.format(bay) + self._jesd_tx_status_valid_cnt + f'[{num}]', **kwargs)

    _fpga_uptime = 'UpTimeCnt'

    def get_fpga_uptime(self, **kwargs):
        """
        Returns
        -------
        uptime : float
            The FPGA uptime.
        """
        return self._caget(self.axi_version + self._fpga_uptime, **kwargs)

    _fpga_version = 'FpgaVersion'

    def get_fpga_version(self, **kwargs):
        """
        Returns
        -------
        version : str
            The FPGA version.
        """
        return self._caget(self.axi_version + self._fpga_version, **kwargs)

    _fpga_git_hash = 'GitHash'

    def get_fpga_git_hash(self, **kwargs):
        r"""Get the full FPGA firmware SHA-1 git hash.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            The full git SHA-1 hash of the FPGA firmware.
        """
        return self._caget(self.axi_version + self._fpga_git_hash,
                           as_string=True, **kwargs)

    _fpga_git_hash_short = 'GitHashShort'

    def get_fpga_git_hash_short(self, **kwargs):
        r"""Get the short FPGA firmware SHA-1 git hash.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            The short git SHA-1 hash of the FPGA firmware.
        """
        return self._caget(self.axi_version +
                           self._fpga_git_hash_short, as_string=True,
                           **kwargs)


    _fpga_build_stamp = 'BuildStamp'

    def get_fpga_build_stamp(self, **kwargs):
        r"""Get the FPGA build stamp.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            The FPGA build stamp.
        """
        return self._caget(self.axi_version + self._fpga_build_stamp,
                           as_string=True, **kwargs)

    _input_mux_sel = 'InputMuxSel[{}]'

    def set_input_mux_sel(self, bay, lane, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) +
            self._input_mux_sel.format(lane), val, **kwargs)

    def get_input_mux_sel(self, bay, lane, **kwargs):
        """
        """
        self._caget(self.daq_mux_root.format(bay) + self._input_mux_sel.format(lane),
            **kwargs)

    _data_buffer_size = 'DataBufferSize'

    def set_data_buffer_size(self, bay, val, **kwargs):
        """
        Sets the data buffer size for the DAQx
        """
        self._caput(self.daq_mux_root.format(bay) + self._data_buffer_size, val, **kwargs)

    def get_data_buffer_size(self, bay, **kwargs):
        """
        Gets the data buffer size for the DAQs
        """
        return self._caget(self.daq_mux_root.format(bay) + self._data_buffer_size,
            **kwargs)

    # Waveform engine commands
    _start_addr = 'StartAddr[{}]'

    def set_waveform_start_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int or str
            What value to set.
        convert : bool, optional, default True
            Convert the input from an integer to a string of hex
            values before setting.
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._start_addr.format(engine), val, **kwargs)

    def get_waveform_start_addr(self, bay, engine, convert=True, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        convert : bool, optional, default True
            Convert the output from a string of hex values to an int.
        """

        val = self._caget(self.waveform_engine_buffers_root.format(bay) +
                          self._start_addr.format(engine), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _end_addr = 'EndAddr[{}]'

    def set_waveform_end_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int
            What val to set.
        convert : bool, optional, default True
            Convert the output from a string of hex values to an int.
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._end_addr.format(engine), val, **kwargs)

    def get_waveform_end_addr(self, bay, engine, convert=True, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        convert : bool, optional, default True
            Convert the output from a string of hex values to an int.

        Returns
        -------
        val : str or int
            Waveform end address (a string of hex values if convert is
            False, otherwise an integer if convert is True).
        """
        val = self._caget(self.waveform_engine_buffers_root.format(bay) +
            self._end_addr.format(engine), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _wr_addr = 'WrAddr[{}]'

    def set_waveform_wr_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int
            What val to set.
        convert : bool, optional, default True
            Convert the output from a string of hex values to an int.
        """
        if convert:
            val = self.int_to_hex_string(val)
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._wr_addr.format(engine), val, **kwargs)

    def get_waveform_wr_addr(self, bay, engine, convert=True, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        convert : bool, optional, default True
            Convert the output from a string of hex values to an int.

        Returns
        -------
        val : str or int
            Waveform end address (a string of hex values if convert is
            False, otherwise an integer if convert is True).
        """
        val = self._caget(self.waveform_engine_buffers_root.format(bay) +
            self._wr_addr.format(engine), **kwargs)
        if convert:
            return self.hex_string_to_int(val)
        else:
            return val

    _empty = 'Empty[{}]'

    def set_waveform_empty(self, bay, engine, val, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int
            What val to set.
        """
        self._caput(self.waveform_engine_buffers_root.format(bay) +
            self._empty.format(engine), **kwargs)

    def get_waveform_empty(self, bay, engine, **kwargs):
        """
        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        """
        return self._caget(self.waveform_engine_buffers_root.format(bay) +
            self._empty.format(engine), **kwargs)

    _data_file = 'DataFile'

    def set_streamdatawriter_datafile(self, datafile_path, **kwargs):
        """
        Sets the output path for the StreamDataWriter. This is what is
        used for take_debug_data.

        Args
        ----
        datafile_path : str
            The full path for the output.
        """
        self._caput(self.stream_data_writer_root + self._data_file,
            datafile_path, **kwargs)

    def get_streamdatawriter_datafile(self, as_str=True, **kwargs):
        r"""Gets output path for the StreamDataWriter.

        This is what is used for take_debug_data.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed to directly to the
            `_caget` call.

        Returns
        -------
        str
            The full path for the output.
        """
        return self._caget(self.stream_data_writer_root +
                           self._data_file, as_string=True, **kwargs)

    _datawriter_open = 'Open'

    def set_streamdatawriter_open(self, val, **kwargs):
        """
        """
        self._caput(self.stream_data_writer_root +
            self._datawriter_open, val, **kwargs)


    def get_streamdatawriter_open(self, **kwargs):
        """
        """
        return self._caget(self.stream_data_writer_root +
            self._datawriter_open, **kwargs)


    _datawriter_close = 'Close'

    def set_streamdatawriter_close(self, val, **kwargs):
        """
        """
        self._caput(self.stream_data_writer_root +
            self._datawriter_close, val, **kwargs)


    def get_streamdatawriter_close(self, **kwargs):
        """
        """
        return self._caget(self.stream_data_writer_root +
            self._datawriter_close, **kwargs)


    _trigger_daq = 'TriggerDaq'

    def set_trigger_daq(self, bay, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._trigger_daq, val,
            **kwargs)

    def get_trigger_daq(self, bay, **kwargs):
        """
        """
        self._caget(self.daq_mux_root.format(bay) + self._trigger_daq,
            **kwargs)

    _arm_hw_trigger = "ArmHwTrigger"

    def set_arm_hw_trigger(self, bay, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._arm_hw_trigger, val, **kwargs)

    _trigger_hw_arm = 'TriggerHwArm'

    def set_trigger_hw_arm(self, bay, val, **kwargs):
        """
        """
        self._caput(self.daq_mux_root.format(bay) + self._trigger_hw_arm, val, **kwargs)

    def get_trigger_hw_arm(self, bay, **kwargs):
        """
        """
        return self._caget(self.daq_mux_root.format(bay) + self._trigger_hw_arm, **kwargs)

    # rtm commands

    #########################################################
    ## start rtm arbitrary waveform

    _rtm_arb_waveform_lut_table = 'Lut[{}]:MemArray'

    def get_rtm_arb_waveform_lut_table(self, reg, **kwargs):
        """
        Gets the table currently loaded into the LUT table indexed by
        reg.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        return self._caget(self.rtm_lut_ctrl_root +
                           self._rtm_arb_waveform_lut_table.format(reg),
                           **kwargs)

    def set_rtm_arb_waveform_lut_table(self, reg, arr, pad=0, **kwargs):
        """
        Loads provided array into the LUT table indexed by reg.  If
        array is empty, loads zeros into table.  If array exceeds
        maximum length of 2048 entries, array is truncated.  If array
        is less than 2048 entries, the table is padded on the end with
        the value in the pad argument.

        Args
        ----
        reg : int
            LUT table index in [0,1].
        arr : int array
            Array of values to load into LUT table.  Each entry must
            be an integer and in [0,2^20).
        pad : int, optional, default 0
            Value to pad end of array with if provided array's length
            is less than 2048.  Default is 0.
        """
        # cast as numpy array
        lut_arr=np.pad(arr[:self._lut_table_array_length],
            (0, self._lut_table_array_length-len(arr[:self._lut_table_array_length])),
            'constant', constant_values=pad)

        # round entries and type as integer
        lut_arr=np.around(lut_arr).astype(int)

        # clip exceed max number of DAC output bits
        dac_nbits_fullscale=self._rtm_slow_dac_nbits
        # warn user if some points get clipped
        num_clip_above=len(np.where(lut_arr>(2**(dac_nbits_fullscale-1)-1))[0])
        if num_clip_above>0:
            self.log(f'{num_clip_above} points in LUT table exceed' +
                f' (2**{dac_nbits_fullscale-1})-1. ' +
                f' Will be clipped to (2**{dac_nbits_fullscale-1})-1.')
        num_clip_below=len(np.where(lut_arr<(-2**(dac_nbits_fullscale-1)))[0])
        if num_clip_below>0:
            self.log(f'{num_clip_below} points in LUT table are less than' +
                f' -2**{dac_nbits_fullscale-1}.  ' +
                f'Will be clipped to -2**{dac_nbits_fullscale-1}.')
        # clip the array
        lut_arr=np.clip(lut_arr,a_min=-2**(dac_nbits_fullscale-1),
            a_max=2**(dac_nbits_fullscale-1)-1)
        self._caput(self.rtm_lut_ctrl_root +
            self._rtm_arb_waveform_lut_table.format(reg), lut_arr, **kwargs)


    _rtm_arb_waveform_busy = 'Busy'

    def get_rtm_arb_waveform_busy(self, **kwargs):
        """
        =1 if waveform if Continuous=1 and the RTM arbitrary waveform
        is being continously generated.  Can be toggled low again by
        setting Continuous=0.
        """
        return self._caget(self.rtm_lut_ctrl +
                           self._rtm_arb_waveform_busy,
                           **kwargs)

    _rtm_arb_waveform_trig_cnt = 'TrigCnt'

    def get_rtm_arb_waveform_trig_cnt(self, **kwargs):
        """
        Counts the number of RTM arbitrary waveform software triggers
        since boot up or the last CntRst.
        """
        return self._caget(self.rtm_lut_ctrl +
                           self._rtm_arb_waveform_trig_cnt,
                           **kwargs)

    _rtm_arb_waveform_continuous = 'Continuous'

    def get_rtm_arb_waveform_continuous(self, **kwargs):
        """
        If =1, RTM arbitrary waveform generation is continuous and
        repeats, otherwise if =0, waveform in LUT tables is only
        broadcast once on software trigger.
        """
        return self._caget(self.rtm_lut_ctrl +
                           self._rtm_arb_waveform_continuous,
                           **kwargs)

    def set_rtm_arb_waveform_continuous(self, val, **kwargs):
        """
        If =1, RTM arbitrary waveform generation is continuous and
        repeats, otherwise if =0, waveform in LUT tables is only
        broadcast once on software trigger.

        Args
        ----
        val : int
            Whether or not arbitrary waveform generation is continuous
            on software trigger.  Must be in [0,1].
        """
        assert (val in range(2)), 'val must be in [0,1]'
        self._caput(self.rtm_lut_ctrl +
                    self._rtm_arb_waveform_continuous,
                    val,
                    **kwargs)

    _rtm_arb_waveform_software_trigger = 'SwTrig'

    def trigger_rtm_arb_waveform(self, continuous=False, **kwargs):
        """
        Software trigger for arbitrary waveform generation on the slow
        RTM DACs.  This will cause the RTM to play the LUT tables only
        once.

        Args
        ----
        continuous : bool, optional, default False
            Whether or not to continously broadcast the arbitrary
            waveform on software trigger.
        """
        if continuous is True:
            self.set_rtm_arb_waveform_continuous(1)
        else:
            self.set_rtm_arb_waveform_continuous(0)

        triggerPV=self.rtm_lut_ctrl + \
            self._rtm_arb_waveform_software_trigger
        self._caput(triggerPV,
                    1,
                    **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    _dac_axil_addr = 'DacAxilAddr[{}]'

    def get_dac_axil_addr(self, reg, **kwargs):
        """
        Gets the DacAxilAddr[#] registers.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        return self._caget(self.rtm_lut_ctrl +
                           self._dac_axil_addr.format(reg),
                           **kwargs)

    def set_dac_axil_addr(self, reg, val, **kwargs):
        """
        Sets the DacAxilAddr[#] registers.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        self._caput(self.rtm_lut_ctrl +
                    self._dac_axil_addr.format(reg), val, **kwargs)

    _rtm_arb_waveform_timer_size = 'TimerSize'

    def get_rtm_arb_waveform_timer_size(self, **kwargs):
        """
        Arbitrary waveforms are written to the slow RTM DACs with time
        between samples TimerSize*6.4ns.
        """
        return self._caget(self.rtm_lut_ctrl +
                           self._rtm_arb_waveform_timer_size,
                           **kwargs)

    def set_rtm_arb_waveform_timer_size(self, val, **kwargs):
        """
        Arbitrary waveforms are written to the slow RTM DACs with time
        between samples TimerSize*6.4ns.

        Args
        ----
        val : int
            The value to set TimerSize to.  Must be an integer in
            [0,2**24).
        """
        assert (val in range(2**24)), 'reg must be in [0,16777216)'
        self._caput(self.rtm_lut_ctrl +
                    self._rtm_arb_waveform_timer_size,
                    val,
                    **kwargs)

    _rtm_arb_waveform_max_addr = 'MaxAddr'

    def get_rtm_arb_waveform_max_addr(self, **kwargs):
        """
        Slow RTM DACs will play the sequence [0...MaxAddr] of points
        out of the loaded LUT tables before stopping or repeating on
        software trigger (if in continuous mode).  MaxAddr is an
        11-bit number (must be in [0,2048), because that's the maximum
        length of the LUT tables that store the waveforms.
        """
        return self._caget(self.rtm_lut_ctrl +
                           self._rtm_arb_waveform_max_addr,
                           **kwargs)

    def set_rtm_arb_waveform_max_addr(self, val, **kwargs):
        """
        Slow RTM DACs will play the sequence [0...MaxAddr] of points
        out of the loaded LUT tables before stopping or repeating on
        software trigger (if in continuous mode).  MaxAddr is an
        11-bit number (must be in [0,2048), because that's the maximum
        length of the LUT tables that store the waveforms.

        Args
        ----
        val : int
            The value to set MaxAddr to.  Must be an integer in
            [0,2048).
        """
        assert (val in range(2**11)), 'reg must be in [0,2048)'
        self._caput(self.rtm_lut_ctrl +
                    self._rtm_arb_waveform_max_addr,
                    val,
                    **kwargs)

    _rtm_arb_waveform_enable = 'EnableCh'

    def get_rtm_arb_waveform_enable(self, **kwargs):
        """
        Enable for generation of arbitrary waveforms on the RTM slow
        DACs.

        EnableCh = 0x0 is disable
        0x1 is Addr[0]
        0x2 is Addr[1]
        0x3 is Addr[0] and Addr[1]
        """
        return self._caget(self.rtm_lut_ctrl +
                           self._rtm_arb_waveform_enable,
                           **kwargs)

    def set_rtm_arb_waveform_enable(self, val, **kwargs):
        """
        Sets the enable for generation of arbitrary waveforms on the
        RTM slow DACs.

        Args
        ----
        val : int
            The value to set enable to.  EnableCh = 0x0 is disable,
            0x1 is Addr[0], 0x2 is Addr[1], and 0x3 is Addr[0] and
            Addr[1]
        """
        assert (val in range(4)), 'reg must be in [0,1,2,3]'
        self._caput(self.rtm_lut_ctrl +
                    self._rtm_arb_waveform_enable,
                    val,
                    **kwargs)

    ## end rtm arbitrary waveform
    #########################################################

    _reset_rtm = 'resetRtm'

    def reset_rtm(self, **kwargs):
        """
        Resets the rear transition module (RTM)
        """
        self._caput(self.rtm_cryo_det_root + self._reset_rtm, 1, **kwargs)

    _cpld_reset = 'CpldReset'

    def set_cpld_reset(self, val, **kwargs):
        """
        Args
        ----
        val : int
            Set to 1 for a cpld reset.
        """
        self._caput(self.rtm_cryo_det_root + self._cpld_reset, val, **kwargs)

    def get_cpld_reset(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._cpld_reset, **kwargs)

    def cpld_toggle(self, **kwargs):
        """
        Toggles the cpld reset bit.
        """
        self.set_cpld_reset(1, wait_done=True, **kwargs)
        self.set_cpld_reset(0, wait_done=True, **kwargs)

    _k_relay = 'KRelay'

    def set_k_relay(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._k_relay, val, **kwargs)

    def get_k_relay(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._k_relay, **kwargs)

    timing_crate_root = ":AMCc:FpgaTopLevel:AmcCarrierCore:AmcCarrierTiming:EvrV2CoreTriggers"
    _trigger_rate_sel = ":EvrV2ChannelReg[0]:RateSel"

    def set_ramp_rate(self, val, **kwargs):
        """
        flux ramp sawtooth reset rate in kHz

        Allowed rates: 1, 2, 3, 4, 5, 6, 8, 10, 12, 15kHz (hardcoded by timing)
        """
        rate_sel = self.flux_ramp_rate_to_PV(val)

        if rate_sel is not None:
            self._caput(self.epics_root + self.timing_crate_root +
                self._trigger_rate_sel, rate_sel, **kwargs)
        else:
            print("Rate requested is not allowed by timing triggers. Allowed " +
                "rates are 1, 2, 3, 4, 5, 6, 8, 10, 12, 15kHz only")

    def get_ramp_rate(self, **kwargs):
        """
        flux ramp sawtooth reset rate in kHz
        """

        rate_sel = self._caget(self.epics_root + self.timing_crate_root +
            self._trigger_rate_sel, **kwargs)

        reset_rate = self.flux_ramp_PV_to_rate(rate_sel)

        return reset_rate

    _trigger_delay = ":EvrV2TriggerReg[0]:Delay"

    def set_trigger_delay(self, val, **kwargs):
        """
        Adds an offset to flux ramp trigger.  Only really useful if
        you're using two carriers at once and you're trying to
        synchronize them.  Mitch thinks it's in units of 122.88MHz
        ticks.
        """
        self._caput(self.epics_root + self.timing_crate_root +
                    self._trigger_delay, val, **kwargs)

    def get_trigger_delay(self, **kwargs):
        """
        The flux ramp trigger offset.  Only really useful if you're
        using two carriers at once and you're trying to synchronize
        them.  Mitch thinks it's in units of 122.88MHz ticks.
        """

        trigger_delay = self._caget(self.epics_root + self.timing_crate_root +
                                    self._trigger_delay, **kwargs)
        return trigger_delay

    _debounce_width = 'DebounceWidth'

    def set_debounce_width(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._debounce_width, val,
            **kwargs)

    def get_debounce_width(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._debounce_width,
            **kwargs)

    _ramp_slope = 'RampSlope'

    def set_ramp_slope(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._ramp_slope, val, **kwargs)

    def get_ramp_slope(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._ramp_slope, **kwargs)

    _flux_ramp_dac = 'LTC1668RawDacData'

    def set_flux_ramp_dac(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._flux_ramp_dac, val, **kwargs)

    def get_flux_ramp_dac(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._flux_ramp_dac, **kwargs)

    _mode_control = 'ModeControl'

    def set_mode_control(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._mode_control, val, **kwargs)

    def get_mode_control(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._mode_control, **kwargs)

    _fast_slow_step_size = 'FastSlowStepSize'

    def set_fast_slow_step_size(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._fast_slow_step_size, val,
            **kwargs)

    def get_fast_slow_step_size(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._fast_slow_step_size,
            **kwargs)

    _fast_slow_rst_value = 'FastSlowRstValue'

    def set_fast_slow_rst_value(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._fast_slow_rst_value, val,
            **kwargs)

    def get_fast_slow_rst_value(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._fast_slow_rst_value,
            **kwargs)

    _enable_ramp_trigger = 'EnableRampTrigger'

    def set_enable_ramp_trigger(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_cryo_det_root + self._enable_ramp_trigger, val,
            **kwargs)

    def get_enable_ramp_trigger(self, **kwargs):
        """
        """
        return self._caget(self.rtm_cryo_det_root + self._enable_ramp_trigger,
            **kwargs)

    _cfg_reg_ena_bit = 'CfgRegEnaBit'

    def set_cfg_reg_ena_bit(self, val, **kwargs):
        """
        """
        self._caput(self.rtm_spi_root + self._cfg_reg_ena_bit, val, **kwargs)

    def get_cfg_reg_ena_bit(self, **kwargs):
        """
        """
        return self._caget(self.rtm_spi_root + self._cfg_reg_ena_bit, **kwargs)

    # Right now in pyrogue, this is named as if it's always a TesBias,
    # but pysmurf doesn't only use them as TES biases - e.g. in
    # systems using a 50K follow-on amplifier, one of these DACs is
    # used to drive the amplifier gate.
    _rtm_slow_dac_enable = 'TesBiasDacCtrlRegCh[{}]'

    def set_rtm_slow_dac_enable(self, dac, val, **kwargs):
        """
        Set DacCtrlReg for this DAC, which configures the AD5790
        analog output for the requested DAC number.  Set to 0x2 to
        enable for normal operation, which only needs to be done once
        for each DAC in a boot session.

        Args
        ----
        dac : int
            Which DAC to command.  1-indexed.  If a DAC index outside
            of the valid range is provided (must be within [1,32]),
            will assert.
        val : int
            Value to set the DAC enable to.
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'

        self._caput(self.rtm_spi_max_root +
                    self._rtm_slow_dac_enable.format(dac), val, **kwargs)


    def get_rtm_slow_dac_enable(self, dac, **kwargs):
        """
        Returns the DacCtrlReg for this DAC, which specifies the
        AD5790 analog output configuration for the requested DAC
        number.  Should be set to 0x2 in normal operation.

        Args
        ----
        dac : int
            Which DAC to query.  1-indexed.  If a DAC index outside of
            the valid range is provided (must be within [1,32]), will
            assert.

        Returns
        -------
        val : int
            The DacCtrlReg setting for the requested DAC.
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'

        return self._caget(self.rtm_spi_max_root +
                           self._rtm_slow_dac_enable.format(dac), **kwargs)

    _rtm_slow_dac_enable_array = 'TesBiasDacCtrlRegChArray'

    def set_rtm_slow_dac_enable_array(self, val, **kwargs):
        """
        Sets DacCtrlReg for all of the DACs at once.  DacCtrlReg
        configures the AD5790 analog outputs.  Setting to 0x2 enables
        normal operation, and only needs to be done once for each DAC
        in a boot session.  Writing the values as an array should be
        much faster than writing them to each DAC individually using
        the set_rtm_slow_dac_enable function (single versus multiple
        transactions).

        Args
        ----
        val : int array
            Length 32, addresses the DACs in DAC ordering.  If
            provided array is not length 32, asserts.
        """
        assert (len(val)==32),'len(val) must be 32, the number of DACs in hardware.'
        self._caput(self.rtm_spi_max_root +
                    self._rtm_slow_dac_enable_array, val, **kwargs)

    def get_rtm_slow_dac_enable_array(self, **kwargs):
        """
        Returns the current DacCtrlReg setting for all of the DACs at
        once (a 32 element integer array).  DacCtrlReg configures the
        AD5790 analog outputs.  If set to 0x2, then the DAC is
        configured for normal operation, which only needs to be done
        once for each DAC in a boot session.  Reading the values as an
        array should be much faster than reading them for each DAC
        individually using the get_rtm_slow_dac_enable function
        (single versus multiple transactions).

        Returns
        -------
        val : int array
            An array containing the DacCtrlReg settings for all of the
            slow RTM DACs.
        """
        return self._caget(self.rtm_spi_max_root +
                           self._rtm_slow_dac_enable_array, **kwargs)

    _rtm_slow_dac_data = 'TesBiasDacDataRegCh[{}]'

    def set_rtm_slow_dac_data(self, dac, val, **kwargs):
        """
        Sets the data register for the requested DAC, which sets the
        output voltage of the DAC.

        Args
        ----
        dac : int
            Which DAC to command.  1-indexed.  If a DAC index outside
            of the valid range is provided (must be within [1,32]),
            will assert.
        val : int
            The DAC voltage to set in DAC units.  Must be in
            [-2^19,2^19).  If requested value is less (greater) than
            -2^19 (2^19-1), sets DAC to -2^19 (2^19-1).
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'

        nbits=self._rtm_slow_dac_nbits
        if val > 2**(nbits-1)-1:
            val = 2**(nbits-1)-1
            self.log(f'Bias too high. Must be <= than 2^{nbits-1}-1.  ' +
                     'Setting to max value', self.LOG_ERROR)
        elif val < -2**(nbits-1):
            val = -2**(nbits-1)
            self.log(f'Bias too low. Must be >= than -2^{nbits-1}.  ' +
                     'Setting to min value', self.LOG_ERROR)
        self._caput(self.rtm_spi_max_root +
                    self._rtm_slow_dac_data.format(dac), val, **kwargs)

    def get_rtm_slow_dac_data(self, dac, **kwargs):
        """
        Gets the value in the data register for the requested DAC,
        which sets the output voltage of the DAC.

        Args
        ----
        dac : int
            Which DAC to command.  1-indexed.  If a DAC index outside
            of the valid range is provided (must be within [1,32]),
            will assert.

        Returns
        -------
        int
            The data register setting for the requested DAC, in DAC
            units.  The data register sets the output voltage of the
            DAC.
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'
        return self._caget(self.rtm_spi_max_root +
                           self._rtm_slow_dac_data.format(dac),
                           **kwargs)

    _rtm_slow_dac_data_array = 'TesBiasDacDataRegChArray'

    def set_rtm_slow_dac_data_array(self, val, **kwargs):
        """
        Sets the data registers for all 32 DACs, which sets their
        output voltages.  Must provide all 32 values.

        Args
        ----
        val : int array
            The DAC voltages to set in DAC units.  Each element of the
            array must Must be in [-2^19,2^19).  If a requested value
            is less (greater) than -2^19 (2^19-1), sets that DAC to
            -2^19 (2^19-1).  (32,) in DAC units.  If provided array is
            not 32 elements long, asserts.
        """
        assert (len(val)==32),'len(val) must be 32, the number of DACs in hardware.'

        nbits=self._rtm_slow_dac_nbits
        val=np.array(val)
        if len(np.ravel(np.where(val > 2**(nbits-1)-1))) > 0:
            self.log('Bias too high for some values. Must be ' +
                     f'<= 2^{nbits-1}-1. Setting to max value',
                     self.LOG_ERROR)
        val[np.ravel(np.where(val > 2**(nbits-1)-1))] = 2**(nbits-1)-1

        if len(np.ravel(np.where(val < - 2**(nbits-1)))) > 0:
            self.log('Bias too low for some values. Must be ' +
                     f'>= -2^{nbits-1}. Setting to min value',
                     self.LOG_ERROR)
        val[np.ravel(np.where(val < - 2**(nbits-1)))] = -2**(nbits-1)

        self._caput(self.rtm_spi_max_root + self._rtm_slow_dac_data_array, val, **kwargs)

    def get_rtm_slow_dac_data_array(self, **kwargs):
        """
        Gets the value in the data register, in DAC units, for all 32
        DACs.  The value in these registers set the output voltages of
        the DACs.

        Returns
        -------
        array : int array
            Size (32,) array of DAC values, in DAC units.  The value
            of these registers set the output voltages of the DACs.
        """
        return self._caget(self.rtm_spi_max_root + self._rtm_slow_dac_data_array,
            **kwargs)

    def set_rtm_slow_dac_volt(self, dac, val, **kwargs):
        """
        Sets the output voltage for the requested DAC.

        Args
        ----
        dac : int
            Which DAC to command.  1-indexed.  If a DAC index outside
            of the valid range is provided (must be within [1,32]),
            will assert.
        val : int
            The DAC voltage to set in volts.
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'
        self.set_rtm_slow_dac_data(dac, val/self._rtm_slow_dac_bit_to_volt, **kwargs)


    def get_rtm_slow_dac_volt(self, dac, **kwargs):
        """
        Gets the current output voltage for the requested DAC.

        Args
        ----
        dac : int
            Which DAC to query.  1-indexed.  If a DAC index outside of
            the valid range is provided (must be within [1,32]), will
            assert.

        Returns
        -------
        float
            The DAC voltage in volts.
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'
        return self._rtm_slow_dac_bit_to_volt * self.get_rtm_slow_dac_data(dac, **kwargs)

    def set_rtm_slow_dac_volt_array(self, val, **kwargs):
        """
        Sets the output voltage for all 32 DACs at once.  Writing the
        values as an array should be much faster than writing them to
        each DAC individually using the set_rtm_slow_dac_volt
        function (single versus multiple transactions).

        Args
        ----
        val : float array
            TES biases to set for each DAC in Volts. Expects an array
            of size (32,).  If provided array is not 32 elements,
            asserts.
        """
        assert (len(val)==32),'len(val) must be 32, the number of DACs in hardware.'
        int_val = np.array(np.array(val) / self._rtm_slow_dac_bit_to_volt, dtype=int)
        self.set_rtm_slow_dac_data_array(int_val, **kwargs)

    def get_rtm_slow_dac_volt_array(self, **kwargs):
        """
        Returns the output voltage for all 32 DACs at once, in volts.
        Reading the values as an array should be much faster than
        reading them for each DAC individually using the
        get_rtm_slow_dac_volt function (single versus multiple
        transactions).

        Returns
        -------
        volt_array : float array
            Size (32,) array of DAC values in volts.
        """
        return self._rtm_slow_dac_bit_to_volt * self.get_rtm_slow_dac_data_array(**kwargs)

    def set_50k_amp_gate_voltage(self, voltage, override=False, **kwargs):
        """
        Sets the 50K amplifier gate votlage.

        Args
        ----
        voltage : float
            The amplifier gate voltage between 0 and -1.
        override : bool, optional, default False
            Whether to override the software limit on the gate
            voltage. This allows you to go outside the range of 0 and
            -1.
        """
        if (voltage > 0 or voltage < -1.) and not override:
            self.log('Voltage must be between -1 and 0. Doing nothing.')
        else:
            self.set_rtm_slow_dac_data(self._fiftyk_dac_num, voltage/self._fiftyk_bit_to_V,
                **kwargs)

    def get_50k_amp_gate_voltage(self, **kwargs):
        """
        """
        return self._fiftyk_bit_to_V * self.get_rtm_slow_dac_data(self._fiftyk_dac_num,
            **kwargs)

    def set_50k_amp_enable(self, disable=False, **kwargs):
        """
        Sets the 50K amp bit to 2 for enable and 0 for disable.

        Args
        ----
        disable : bool, optional, default False
            Disable the 50K amplifier.
        """
        if disable:
            self.set_rtm_slow_dac_enable(self._fiftyk_dac_num, 0, **kwargs)
        else:
            self.set_rtm_slow_dac_enable(self._fiftyk_dac_num, 2, **kwargs)

    def flux_ramp_on(self, **kwargs):
        """
        Turns on the flux ramp - a useful wrapper for set_cfg_reg_ena_bit
        """
        self.set_cfg_reg_ena_bit(1, **kwargs)

    def flux_ramp_off(self, **kwargs):
        """
        Turns off the flux ramp - a useful wrapper for set_cfg_reg_ena_bit
        """
        self.set_cfg_reg_ena_bit(0, **kwargs)

    _ramp_max_cnt = 'RampMaxCnt'

    def set_ramp_max_cnt(self, val, **kwargs):
        """
        Internal Ramp's maximum count. Sets the trigger repetition rate. This
        is effectively the flux ramp frequency.

        RampMaxCnt = 307199 means flux ramp is 1kHz (307.2e6/(RampMaxCnt+1))
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_max_cnt, val, **kwargs)

    def get_ramp_max_cnt(self, **kwargs):
        """
        Internal Ramp's maximum count. Sets the trigger repetition rate. This
        is effectively the flux ramp frequency.

        RampMaxCnt = 307199 means flux ramp is 1kHz (307.2e6/(RampMaxCnt+1))
        """
        return self._caget(self.rtm_cryo_det_root + self._ramp_max_cnt,
            **kwargs)

    def set_flux_ramp_freq(self, val, **kwargs):
        """
        Wrapper function for set_ramp_max_cnt. Takes input in Hz.
        """
        cnt = 3.072E5/float(val)-1
        self.set_ramp_max_cnt(cnt, **kwargs)

    def get_flux_ramp_freq(self, **kwargs):
        """
        Returns flux ramp freq in units of Hz
        """
        if self.offline: # FIX ME - this is a stupid hard code
            return 4.0
        else:
            return 3.0725E5/(self.get_ramp_max_cnt(**kwargs)+1)


    _low_cycle = 'LowCycle'

    def set_low_cycle(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive).
        Along with HighCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(self.rtm_cryo_det_root + self._low_cycle, val, **kwargs)

    def get_low_cycle(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive).
        Along with HighCycle, sets the frequency of the clock going to the RTM.
        """
        return self._caget(self.rtm_cryo_det_root + self._low_cycle, **kwargs)

    _high_cycle = 'HighCycle'

    def set_high_cycle(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).
        Along with LowCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(self.rtm_cryo_det_root + self._high_cycle, val, **kwargs)

    def get_high_cycle(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).
        Along with LowCycle, sets the frequency of the clock going to the RTM.
        """
        return self._caget(self.rtm_cryo_det_root + self._high_cycle, **kwargs)

    _select_ramp = 'SelectRamp'

    def set_select_ramp(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._select_ramp, val, **kwargs)

    def get_select_ramp(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._select_ramp, **kwargs)

    _enable_ramp = 'EnableRamp'

    def set_enable_ramp(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._enable_ramp, val, **kwargs)

    def get_enable_ramp(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._enable_ramp, **kwargs)

    _ramp_start_mode = 'RampStartMode'

    def set_ramp_start_mode(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x2 = trigger from external system
        0x1 = trigger from timing system
        0x0 = trigger from internal system
        """
        self._caput(self.rtm_cryo_det_root + self._ramp_start_mode, val,
            **kwargs)

    def get_ramp_start_mode(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x2 = trigger from external system
        0x1 = trigger from timing system
        0x0 = trigger from internal system
        """
        return self._caget(self.rtm_cryo_det_root + self._ramp_start_mode,
            **kwargs)

    _pulse_width = 'PulseWidth'

    def set_pulse_width(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(self.rtm_cryo_det_root + self._pulse_width, val, **kwargs)

    def get_pulse_width(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(self.rtm_cryo_det_root + self._pulse_width, **kwargs)

    # can't write a get for this right now because read back isn't implemented
    # I think...
    _hemt_v_enable = 'HemtBiasDacCtrlRegCh[33]'

    def set_hemt_enable(self, disable=False, **kwargs):
        """
        Sets bit to 2 for enable and 0 for disable.

        Args
        ----
        disable : bool, optional, default False
            If True, sets the HEMT enable bit to 0.
        """
        if disable:
            self._caput(self.rtm_spi_max_root + self._hemt_v_enable, 0,
                **kwargs)
        else:
            self._caput(self.rtm_spi_max_root + self._hemt_v_enable, 2,
                **kwargs)

    def set_hemt_gate_voltage(self, voltage, override=False, **kwargs):
        """
        Sets the HEMT gate voltage in units of volts.

        Args
        ----
        voltage : float
            The voltage applied to the HEMT gate. Must be between 0
            and .75.
        override bool, optional, default False
            Override thee limits on HEMT gate voltage.
        """
        self.set_hemt_enable()
        if (voltage > self._hemt_gate_max_voltage or voltage <
                self._hemt_gate_min_voltage ) and not override:
            self.log('Input voltage too high. Not doing anything.' +
                ' If you really want it higher, use the override optional arg.')
        else:
            self.set_hemt_bias(int(voltage/self._hemt_bit_to_V),
                override=override, **kwargs)

    _hemt_v = 'HemtBiasDacDataRegCh[33]'

    def set_hemt_bias(self, val, override=False, **kwargs):
        """
        Sets the HEMT voltage in units of bits. Need to figure out the
        conversion into real units.

        There is a hardcoded maximum value. If exceeded, no voltage is set. This
        check can be ignored using the override optional argument.

        Args
        ----
        val : int
            The voltage in bits.
        override : bool, optional, default False
            Allows exceeding the hardcoded limit. Default False.
        """
        if val > 350E3 and not override:
            self.log('Input voltage too high. Not doing anything.' +
                ' If you really want it higher, use the override optinal arg.')
        else:
            self._caput(self.rtm_spi_max_root + self._hemt_v, val, **kwargs)

    def get_hemt_bias(self, **kwargs):
        """
        Returns the HEMT voltage in bits.
        """
        return self._caget(self.rtm_spi_max_root + self._hemt_v, **kwargs)

    def get_hemt_gate_voltage(self, **kwargs):
        """
        Returns the HEMT voltage in bits.
        """
        return self._hemt_bit_to_V*(self.get_hemt_bias(**kwargs))

    _stream_datafile = 'dataFile'

    def set_streaming_datafile(self, datafile, as_string=True, **kwargs):
        """
        Sets the datafile to write streaming data

        Args
        ----
        datafile : str or length 300 int array
            The name of the datafile.
        as_string : bool, optional, default True
            The input data is a string. If False, the input data must
            be a length 300 character int.
        """
        if as_string:
            datafile = [ord(x) for x in datafile]
            # must be exactly 300 elements long. Pad with trailing zeros
            datafile = np.append(datafile, np.zeros(300-len(datafile),
                dtype=int))
        self._caput(self.streaming_root + self._stream_datafile, datafile,
            **kwargs)

    def get_streaming_datafile(self, as_string=True, **kwargs):
        """
        Gets the datafile that streaming data is written to.

        Args
        ----
        as_string : bool, optional, default True
            The output data returns as a string. If False, the input
            data must be a length 300 character int.

        Returns
        -------
        datafile : str or length 300 int array
            The name of the datafile.
        """
        datafile = self._caget(self.streaming_root + self._stream_datafile,
            **kwargs)
        if as_string:
            datafile = ''.join([chr(x) for x in datafile])
        return datafile

    _streaming_file_open = 'open'

    def set_streaming_file_open(self, val, **kwargs):
        """
        Sets the streaming file open. 1 for streaming on. 0 for streaming off.

        Args
        ----
        val : int
            The streaming status.
        """
        self._caput(self.streaming_root + self._streaming_file_open, val,
            **kwargs)

    def get_streaming_file_open(self, **kwargs):
        """
        Gets the streaming file status. 1 is streaming, 0 is not.

        Returns
        -------
        val : int
            The streaming status.
        """
        return self._caget(self.streaming_root + self._streaming_file_open,
            **kwargs)

    # Carrier slot number
    _slot_number = "SlotNumber"

    def get_slot_number(self, **kwargs):
        """
        Gets the slot number of the crate that the carrier is installed into.

        Returns
        -------
        val : int
            The slot number of the crate that the carrier is installed into.
        """
        return self._caget(self.amc_carrier_bsi + self._slot_number, **kwargs)

    # Crate id
    _crate_id = "CrateId"

    def get_crate_id(self, **kwargs):
        """
        Gets the crate id.

        Returns
        -------
        val : int
            The crate id.
        """
        return self._caget(self.amc_carrier_bsi + self._crate_id, **kwargs)


    # UltraScale+ FPGA
    fpga_root = ":AMCc:FpgaTopLevel:AmcCarrierCore:AxiSysMonUltraScale"
    _fpga_temperature = ":Temperature"

    def get_fpga_temp(self, **kwargs):
        """
        Gets the temperature of the UltraScale+ FPGA.  Returns float32,
        the temperature in degrees Celsius.

        Returns
        -------
        val : float
            The UltraScale+ FPGA temperature in degrees Celsius.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_temperature, **kwargs)

    _fpga_vccint = ":VccInt"

    def get_fpga_vccint(self, **kwargs):
        """
        Returns
        -------
        val : float
            The UltraScale+ FPGA VccInt in Volts.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_vccint, **kwargs)

    _fpga_vccaux = ":VccAux"

    def get_fpga_vccaux(self, **kwargs):
        """
        Returns
        -------
        val : float
            The UltraScale+ FPGA VccAux in Volts.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_vccaux, **kwargs)

    _fpga_vccbram = ":VccBram"

    def get_fpga_vccbram(self, **kwargs):
        """
        Returns
        -------
        val : float
            The UltraScale+ FPGA VccBram in Volts.
        """
        return self._caget(self.epics_root + self.fpga_root + self._fpga_vccbram, **kwargs)

    # Regulator
    _regulator_iout = "IOUT"

    def get_regulator_iout(self, **kwargs):
        """
        Returns
        -------
        value : float
            Regulator current in amperes.
        """
        return float(
            float(self._caget(
                self.regulator + self._regulator_iout,
                as_string=True,
                **kwargs)))

    _regulator_temp1 = "TEMPERATURE[1]"

    def get_regulator_temp1(self, **kwargs):
        """
        Returns
        -------
        value : float
            Regulator PT temperature in C.
        """
        return float(
            float(self._caget(
                self.regulator + self._regulator_temp1,
                as_string=True,
                **kwargs)))

    _regulator_temp2 = "TEMPERATURE[2]"

    def get_regulator_temp2(self, **kwargs):
        """
        Returns
        -------
        value : float
            A regulator CTRL temperature in C.
        """
        return float(
            float(self._caget(
                self.regulator + self._regulator_temp2,
                as_string=True,
                **kwargs)))

    # Cryo card comands
    def get_cryo_card_temp(self, enable_poll=False, disable_poll=False):
        """
        Returns
        -------
        temp : float
            Temperature of the cryostat card in Celsius.
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        T = self.C.read_temperature()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return T


    def get_cryo_card_hemt_bias(self, enable_poll=False, disable_poll=False):
        """
        Returns
        -------
        bias : float
            The HEMT bias in volts.
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        hemt_bias = self.C.read_hemt_bias()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return hemt_bias

    def get_cryo_card_50k_bias(self, enable_poll=False, disable_poll=False):
        """
        Returns
        -------
        bias : float
            The 50K bias in volts.
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        bias = self.C.read_50k_bias()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return bias

    def get_cryo_card_cycle_count(self, enable_poll=False, disable_poll=False):
        """
        Returns
        -------
        cycle_count : float
            The cycle count.
        """
        self.log('Not doing anything because not implement in cryo_card.py')
        # return self.C.read_cycle_count()

    def get_cryo_card_relays(self, enable_poll=False, disable_poll=False):
        """
        Returns
        -------
        relays : hex
            The cryo card relays value.
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        relay = self.C.read_relays()

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

        return relay

    def set_cryo_card_relay_bit(self, bitPosition, oneOrZero):
        """
        Sets a single cryo card relay to the value provided

        Args
        ----
        bitPosition : int
            Which bit to set.  Must be in [0-16].
        oneOrZero : int
            What value to set the bit to.  Must be either 0 or 1.
        """
        assert (bitPosition in range(17)), 'bitPosition must be in [0,...,16]'
        assert (oneOrZero in [0,1]), 'oneOrZero must be either 0 or 1'
        currentRelay = self.get_cryo_card_relays()
        nextRelay = currentRelay & ~(1<<bitPosition)
        nextRelay = nextRelay | (oneOrZero<<bitPosition)
        self.set_cryo_card_relays(nextRelay)



    def set_cryo_card_relays(self, relay, write_log=False, enable_poll=False,
                             disable_poll=False):

        """
        Sets the cryo card relays

        Args
        ----
        relays : hex
            The cryo card relays
        """
        if write_log:
            self.log(f'Writing relay using cryo_card object. {relay}')

        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        self.C.write_relays(relay)

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)


    def set_cryo_card_delatch_bit(self, bit, write_log=False, enable_poll=False,
                                  disable_poll=False):
        """
        Delatches the cryo card for a bit.

        Args
        ----
        bit : int
            The bit to temporarily delatch.
        """
        if enable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, True)

        if write_log:
            self.log('Setting delatch bit using cryo_card ' +
                     f'object. {bit}')
        self.C.delatch_bit(bit)

        if disable_poll:
            epics.caput(self.epics_root + self._global_poll_enable, False)

    def set_cryo_card_hemt_ps_en(self, enable, write_log=False):
        """
        Set the cryo card HEMT power supply enable.

        Args
        ----
        enable : bool
            Power supply enable (True = enable, False = disable).
        """
        if write_log:
            self.log('Writing HEMT PS enable using cryo_card object '+
                f'to {enable}')

        # Read the current enable word and merge this bit in position 0
        current_en_value = self.C.read_ps_en()
        if (enable):
            # Set bit 0
            new_en_value = current_en_value | 0x1
        else:
            # Clear bit 0
            new_en_value = current_en_value & 0x2

        # Write back the new value
        self.C.write_ps_en(new_en_value)

    def set_cryo_card_50k_ps_en(self, enable, write_log=False):
        """
        Set the cryo card 50k power supply enable.

        Args
        ----
        enable : bool
            Power supply enable (True = enable, False = disable).
        """
        if write_log:
            self.log('Writing 50k PS enable using cryo_card object '+
                f'to {enable}')

        # Read the current enable word and merge this bit in position 1
        current_en_value = self.C.read_ps_en()
        if (enable):
            # Set bit 2
            new_en_value = current_en_value | 0x2
        else:
            # Clear bit 1
            new_en_value = current_en_value & 0x1

        # Write back the new value
        self.C.write_ps_en(new_en_value)

    def set_cryo_card_ps_en(self, enable=3, write_log=False):
        """
        Write the cryo card power supply enables. Can use this to set both
        power supplies at once rather than setting them individually

        Args
        ----
        enables : int, optional, default 3
            2-bit number with status of the power supplies enables
            Bit 0 for HEMT supply
            Bit 1 for 50K supply
            Bit == 1 means enabled
            Bit == 0 means disabled

            therefore:
            0 = all off
            1 = 50K on, HEMT off
            2 = HEMT on, 50K off
            3 = both on

            Default (enable=3) turns on both power supplies.
        """
        if write_log:
            self.log('Writing Cryocard PS enable using cryo_card ' +
                f'object to {enable}')
        self.C.write_ps_en(enable)

    def get_cryo_card_ps_en(self):
        """
        Read the cryo card power supply enable signals

        Returns
        -------
        enables : int
            2-bit number with status of the power supplies enables

            Bit 0 for HEMT supply
            Bit 1 for 50K supply
            Bit == 1 means enabled
            Bit == 0 means disabled

            therefore:
            0 = all off
            1 = 50K on, HEMT off
            2 = HEMT on, 50K off
            3 = both on
        """
        en_value = self.C.read_ps_en()
        return en_value


    def get_cryo_card_hemt_ps_en(self):
        """
        Get the cryo card HEMT power supply enable.

        Returns
        -------
        enable : bool
            Power supply enable (True = enable, False = disable).
        """

        # Read the power supply enable word and extract the status of bit 0
        en_value = self.C.read_ps_en()

        return (en_value & 0x1 == 0x1)

    def get_cryo_card_50k_ps_en(self):
        """
        Set the cryo card HEMT power supply enable.

        Returns
        -------
        enable : bool
            Power supply enable (True = enable, False = disable).
        """

        # Read the power supply enable word and extract the status of bit 1
        en_value = self.C.read_ps_en()

        return (en_value & 0x2 == 0x2)

    def get_cryo_card_ac_dc_mode(self):
        """
        Get the operation mode, AC or DC, based on the readback of the relays.

        Returns
        -------
        mode : str
            String describing the operation mode. If the relays
            readback don't match, then the string 'ERROR' is returned.
        """

        # Read the relays status
        status = self.C.read_ac_dc_relay_status()

        # Both bit
        if status == 0x0:
            # When both readbacks are '0' we are in DC mode
            return("DC")
        elif status == 0x3:
            # When both readback are '1' we are in AC mode
            return("AC")
        else:
            # Anything else is an error
            return("ERROR")


    _smurf_to_gcp_stream = 'userConfig[0]'  # bit for streaming

    def get_user_config0(self, as_binary=False, **kwargs):
        """
        """
        val =  self._caget(self.timing_header +
                           self._smurf_to_gcp_stream, **kwargs)
        if as_binary:
            val = bin(val)

        return val


    def set_user_config0(self, val, as_binary=False, **kwargs):
        """
        """
        self._caput(self.timing_header +
                   self._smurf_to_gcp_stream, val, **kwargs)


    def clear_unwrapping_and_averages(self, epics_poll=True, **kwargs):
        """
        Resets unwrapping and averaging for all channels, in all bands.
        """

        # Set bit 0 of userConfig[0] high.  Use SyncGroup to detect
        # when register changes so we're sure.
        user_config0_pv=self.timing_header + self._smurf_to_gcp_stream
        # Toggle using SyncGroup so we can confirm state as we toggle.
        sg=SyncGroup([user_config0_pv])

        # what is it now?
        sg.wait(epics_poll=epics_poll) # wait for value
        uc0=sg.get_values()[user_config0_pv]

        # set bit high, keeping all other bits the same
        self.set_user_config0(uc0 | (1 << 0))
        sg.wait(epics_poll=epics_poll) # wait for change
        uc0=sg.get_values()[user_config0_pv]
        assert ( ( uc0 >> 0) & 1 ),(
            'Failed to set averaging/clear bit high ' +
            f'(userConfig0={uc0})')

        # toggle bit back to low, keeping all other bits the same
        self.set_user_config0(uc0 & ~(1 << 0))
        sg.wait(epics_poll=epics_poll) # wait for change
        uc0=sg.get_values()[user_config0_pv]
        assert ( ~( uc0 >> 0) & 1 ),(
            'Failed to set averaging/clear bit low after setting ' +
            f'it high (userConfig0={uc0}).')

        self.log('Successfully toggled averaging/clearing bit ' +
                 f'(userConfig[0]={uc0}).',self.LOG_USER)

    # Triggering commands
    _trigger_width = 'EvrV2TriggerReg[{}]:Width'

    def set_trigger_width(self, chan, val, **kwargs):
        """
        Mystery value that seems to make the timing system work
        """
        self._caput(self.trigger_root + self._trigger_width.format(chan),
                    val, **kwargs)


    _trigger_enable = 'EvrV2TriggerReg[{}]:EnableTrig'

    def set_trigger_enable(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root + self._trigger_enable.format(chan),
                   val, **kwargs)

    _trigger_channel_reg_enable = 'EvrV2ChannelReg[{}]:EnableReg'

    def set_evr_channel_reg_enable(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root + self._trigger_channel_reg_enable.format(chan),
                   val, **kwargs)

    # Crashing in rogue 4, and not clear it's ever needed.
    _trigger_reg_enable = 'EvrV2TriggerReg[{}]:enable'

    def set_evr_trigger_reg_enable(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root +
                    self._trigger_reg_enable.format(chan), val,
                    **kwargs)

    _trigger_channel_reg_count = 'EvrV2ChannelReg[{}]:Count'

    def get_evr_channel_reg_count(self, chan, **kwargs):
        """
        """
        return self._caget(self.trigger_root +
                    self._trigger_channel_reg_count.format(chan),
                    **kwargs)

    _trigger_channel_reg_dest_sel = 'EvrV2ChannelReg[{}]:DestSel'

    def set_evr_trigger_channel_reg_dest_sel(self, chan, val, **kwargs):
        """
        """
        self._caput(self.trigger_root +
                    self._trigger_channel_reg_dest_sel.format(chan),
                    val, **kwargs)


    _dac_reset = 'dacReset[{}]'

    def set_dac_reset(self, bay, dac, val, **kwargs):
        """
        Toggles the physical reset line to DAC. Set to 1 then 0

        Args
        ----
        bay : int
            Which bay [0 or 1].
        dac : int
            Which DAC no. [0 or 1].
        """
        self._caput(self.DBG.format(bay) + self._dac_reset.format(dac), val,
                    **kwargs)

    def get_dac_reset(self, bay, dac, **kwargs):
        """
        Reads the physical reset DAC register.  Will be either 0 or 1.

        Args
        ----
        bay : int
            Which bay [0 or 1].
        dac : int
            Which DAC no. [0 or 1].
        """
        return self._caget(self.DBG.format(bay) + self._dac_reset.format(dac),
                           **kwargs)


    _debug_select = "DebugSelect[{}]"

    def set_debug_select(self, bay, val, **kwargs):
        """
        """
        self._caput(self.app_core + self._debug_select.format(bay),
                    val, **kwargs)

    def get_debug_select(self, bay, **kwargs):
        """
        """
        return self._caget(self.app_core + self._debug_select.format(bay),
                           **kwargs)

    ### Start Ultrascale OT protection
    _ultrascale_ot_upper_threshold = "OTUpperThreshold"

    def set_ultrascale_ot_upper_threshold(self, val, **kwargs):
        """
        Over-temperature (OT) upper threshold in degC for Ultrascale+
        FPGA.
        """
        self._caput(self.ultrascale + self._ultrascale_ot_upper_threshold,
                    val, **kwargs)

    def get_ultrascale_ot_upper_threshold(self, **kwargs):
        """
        Over-temperature (OT) upper threshold in degC for Ultrascale+
        FPGA.
        """
        return self._caget(self.ultrascale + self._ultrascale_ot_upper_threshold,
                           **kwargs)
    ### End Ultrascale OT protection

    _output_config = "OutputConfig[{}]"

    def set_crossbar_output_config(self, index, val, **kwargs):
        """
        """
        self._caput(self.crossbar + self._output_config.format(index),
                    val, **kwargs)

    def get_crossbar_output_config(self, index, **kwargs):
        """
        """
        return self._caget(self.crossbar + self._output_config.format(index),
                           **kwargs)

    _timing_link_up = "RxLinkUp"

    def get_timing_link_up(self, **kwargs):
        """
        """
        return self._caget(self.timing_status +
                           self._timing_link_up, **kwargs)

    # assumes it's handed the decimal equivalent
    _lmk_reg = "LmkReg_0x{:04X}"

    def set_lmk_reg(self, bay, reg, val, **kwargs):
        """
        Can call like this get_lmk_reg(bay=0,reg=0x147,val=0xA)
        to see only hex as in gui.
        """
        self._caput(self.lmk.format(bay) + self._lmk_reg.format(reg),
                    val, **kwargs)

    def get_lmk_reg(self, bay, reg, **kwargs):
        """
        Can call like this hex(get_lmk_reg(bay=0,reg=0x147))
        to see only hex as in gui.
        """
        return self._caget(self.lmk.format(bay) +
                           self._lmk_reg.format(reg), **kwargs)

    _mcetransmit_debug = ':AMCc:mcetransmitDebug'

    def set_mcetransmit_debug(self, val, **kwargs):
        """
        Sets the mcetransmit debug bit. If 1, the debugger will
        print to the pyrogue screen.

        Args
        ----
        val : int
            0 or 1 for the debug bit.
        """
        self._caput(self.epics_root + self._mcetransmit_debug, val,
                    **kwargs)

    _frame_count = 'FrameCnt'

    def get_frame_count(self, **kwargs):
        """
        Gets the frame count going into the SmurfProcessor. This
        must be incrementing if you are attempting to stream
        data.

        Returns
        -------
        int
            The frame count number
        """
        return self._caget(self.frame_rx_stats + self._frame_count,
                    **kwargs)

    _frame_size = 'FrameSize'

    def get_frame_size(self, **kwargs):
        """
        Gets the size of the frame going into the smurf processor.

        Returns
        -------
        int
            The size of the data frame into the smurf processor.
        """
        return self._caget(self.frame_rx_stats + self._frame_size,
                           **kwargs)

    _frame_loss_count = 'FrameLossCnt'

    def get_frame_loss_cnt(self, **kwargs):
        """
        The number of frames that did not make it to the smurf
        processor
        """
        return self._caget(self.frame_rx_stats+ self._frame_loss_count,
                           **kwargs)

    _frame_out_order_count = 'FrameOutOrderCnt'

    def get_frame_out_order_count(self, **kwargs):
        """
        """
        return self._caget(self.frame_rx_stats + self._frame_out_order_count,
                           **kwargs)


    _channel_mask = 'ChannelMapper:Mask'

    def set_channel_mask(self, mask, **kwargs):
        """
        Set the smurf processor channel mask.

        Args
        ----
        mask : list
            The channel mask.
        """
        self._caput(self.smurf_processor + self._channel_mask,
                mask, **kwargs)


    def get_channel_mask(self, **kwargs):
        """
        Gets the smuf processor channel mask.

        Returns
        -------
        mask : list
            The channel mask.
        """
        return self._caget(self.smurf_processor + self._channel_mask,
            **kwargs)


    _unwrapper_reset = 'Unwrapper:reset'

    def set_unwrapper_reset(self, **kwargs):
        """
        Resets the unwrap filter. There is no get function because
        it is an executed command.
        """
        self._caput(self.smurf_processor + self._unwrapper_reset, 1,
                    **kwargs)


    _filter_reset = 'Filter:reset'

    def set_filter_reset(self, **kwargs):
        """
        Resets the downsample filter
        """
        self._caput(self.smurf_processor + self._filter_reset,
                    1, **kwargs)


    _filter_a = 'Filter:A'

    def set_filter_a(self, coef, **kwargs):
        """
        Set the smurf processor filter A coefficients.

        Args
        ----
        coef : list
            The filter A coefficients.
        """
        self._caput(self.smurf_processor + self._filter_a, coef, **kwargs)


    def get_filter_a(self, **kwargs):
        """
        Gets the smurf processor filter A coefficients.

        Returns
        -------
        coef : list
            The filter A coefficients.
        """
        if self.offline:  # FIX ME - STUPPID HARDCODE
            return np.array([ 1., -3.74145562,  5.25726624, -3.28776591, 0.77203984])
        return self._caget(self.smurf_processor + self._filter_a, **kwargs)


    _filter_b = 'Filter:B'

    def set_filter_b(self, coef, **kwargs):
        """
        Set the smurf processor filter B coefficients.

        Args
        ----
        coef : list
            The filter B coefficients.
        """
        self._caput(self.smurf_processor + self._filter_b, coef, **kwargs)


    def get_filter_b(self, **kwargs):
        """
        Get the smurf processor filter B coefficients.

        Returns
        -------
        coef : list
            The filter B coefficients.
        """
        if self.offline:
            return np.array([5.28396689e-06, 2.11358676e-05, 3.17038014e-05,
                2.11358676e-05, 5.28396689e-06])
        return self._caget(self.smurf_processor + self._filter_b, **kwargs)


    _filter_order = 'Filter:Order'

    def set_filter_order(self, order, **kwargs):
        """
        Set the smurf processor filter order.

        Args
        ----
        int
            The filter order.
        """
        self._caput(self.smurf_processor + self._filter_order,
                order, **kwargs)

    def get_filter_order(self, **kwargs):
        """
        Get the smurf processor filter order.

        Args
        ----
        int
            The filter order.
        """
        return self._caget(self.smurf_processor + self._filter_order, **kwargs)


    _filter_gain = 'Filter:Gain'

    def set_filter_gain(self, gain, **kwargs):
        """
        Set the smurf processor filter gain.

        Args
        ----
        float
            The filter gain.
        """
        self._caput(self.smurf_processor + self._filter_gain, gain, **kwargs)


    def get_filter_gain(self, **kwargs):
        """
        Get the smurf processor filter gain.

        Returns
        -------
        float
            The filter gain.
        """
        return self._caget(self.smurf_processor + self._filter_gain, **kwargs)


    _downsampler_factor = 'Downsampler:Factor'

    def set_downsample_factor(self, factor, **kwargs):
        """
        Set the smurf processor down-sampling factor.

        Args
        ----
        int
            The down-sampling factor.
        """
        self._caput(self.smurf_processor + self._downsampler_factor, factor,
            **kwargs)


    def get_downsample_factor(self, **kwargs):
        """
        Get the smurf processor down-sampling factor.

        Returns
        -------
        int
            The down-sampling factor.
        """
        if self.offline:  # FIX ME - STUPID HARD CODE
            return 20

        else:
            return self._caget(self.smurf_processor + self._downsampler_factor,
                **kwargs)

    _filter_disable = "Filter:Disable"

    def set_filter_disable(self, disable_status, **kwargs):
        """
        If Disable is set to True, then the downsampling filter is off.

        Args
        ----
        bool
            The status of the Disable bit.
        """
        self._caput(self.smurf_processor + self._filter_disable,
                    disable_status, **kwargs)

    def get_filter_disable(self, **kwargs):
        """
        If Disable is set to True, then the downsampling filter is off.

        Returns
        -------
        bool
            The status of the Disable bit.
        """
        return self._caget(self.smurf_processor + self._filter_disable,
                           **kwargs)


    _data_file_name = 'FileWriter:DataFile'

    def set_data_file_name(self, name, **kwargs):
        """
        Set the data file name.

        Args
        ----
        str
            The file name.
        """
        self._caput(self.smurf_processor + self._data_file_name, name, **kwargs)


    def get_data_file_name(self, **kwargs):
        """
        Set the data file name.

        Returns
        -------
        str
            The file name.
        """
        return self._caget(self.smurf_processor + self._data_file_name,
            **kwargs)


    _data_file_open = 'FileWriter:Open'

    def open_data_file(self, **kwargs):
        """
        Open the data file.
        """
        self._caput(self.smurf_processor + self._data_file_open, 1, **kwargs)


    _data_file_close = 'FileWriter:Close'

    def close_data_file(self, **kwargs):
        """
        Close the data file.
        """
        self._caput(self.smurf_processor + self._data_file_close, 1, **kwargs)


    _num_channels = "NumChannels"

    def get_smurf_processor_num_channels(self, **kwargs):
        """
        This is the number of channels that smurf_processor (the thing that
        does the downsampling, filtering, etc and then swrites to disk/streams
        data to the DAQ) thinks are on.

        This value is read only.
        """
        return self._caget(self.channel_mapper + self._num_channels, **kwargs)


    _payload_size = "PayloadSize"

    def get_payload_size(self, **kwargs):
        """
        The payload size defines the number of available channels
        to write to disk/stream. Payload size must be larger than
        the number of channels going into the channel mapper

        Returns
        -------
        int
            The number of channels written to disk.  This is
            independent of the number of active channels.
        """
        return self._caget(self.channel_mapper + self._payload_size,
                           **kwargs)


    def set_payload_size(self, payload_size, **kwargs):
        """
        The payload size defines the number of available channels
        to write to disk/stream. Payload size must be larger than
        the number of channels going into the channel mapper

        Args
        ----
        int
            The number of channels written to disk.  This is
            independent of the number of active channels.
        """
        self._caput(self.channel_mapper + self._payload_size,
                    payload_size, **kwargs)
