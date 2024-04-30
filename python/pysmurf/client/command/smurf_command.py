
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
import os
import time
import subprocess

import numpy as np
from packaging import version

from pysmurf.client.base import SmurfBase
from pysmurf.client.command.sync_group import SyncGroup as SyncGroup
from pysmurf.client.util import tools, dscounters


class SmurfCommandMixin(SmurfBase):

    _global_poll_enable_reg = 'AMCc.enable'

    def _caput(self, pvname, val, atca=False, cast_type=True, **kwargs):
        """Puts variables into epics.

        Wrapper around pyrogue lcaput. Puts variables into epics.

        Args
        ----
        pvname : str
            The EPICs path of the PV to get.
        val: any
            The value to put into epics
        """
        err = False
        for k,v in kwargs.items():

            if v is not None:
                print(f"Unexpected kwargs: {k} = {v}")
                #err=True

        if err:
            raise Exception("Bad args passed to caput")

        if atca:
            var = self._atca.root.getNode(pvname)
        else:
            var = self._client.root.getNode(pvname)

        if var == None:
            raise Exception(f"Invalid node: {pvname}")

        # For the ATCA EPICS interface, a command is called with .set
        if not atca and var.isCommand:
            var.call(val)
        elif var.enum != None:
            # setDisp handles enum keys
            var.setDisp(val)
        else:
            if not atca and cast_type:
                # python types are better handled by rogue
                if isinstance(val, np.ndarray):
                    val = [i.item() for i in val]
                elif isinstance(val, np.generic):
                    val = val.item()
                # also check that the type matches variable type
                var_type = type(var.value())
                val = var_type(val)
            var.set(val)

    def _caget(self, pvname, atca=False, **kwargs):
        """Gets variables from epics.

        Wrapper around pyepics lcaget. Gets variables from epics.

        Args
        ----
        pvname : str
            The EPICs path of the PV to get.

        Returns
        -------
        ret : str
            The requested value.
        """
        as_string = kwargs.pop("as_string", False)
        err = False
        for k,v in kwargs.items():

            if v is not None:
                print(f"Unexpected kwargs: {k} = {v}")
                #err=True

        if err:
            raise Exception("Bad args passed to caput")

        if atca:
            var = self._atca.root.getNode(pvname)
        else:
            var = self._client.root.getNode(pvname)

        if var == None:
            raise Exception(f"Invalid node: {pvname}")

        if as_string:
            return var.getDisp()
        else:
            return var.get()

    #### Start SmurfApplication gets/sets
    _smurf_version_reg = 'SmurfVersion'

    def get_pysmurf_version(self, **kwargs):
        r"""Returns the pysmurf version.

        Alias for `pysmurf.__version__`.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            pysmurf version.
        """
        return self._caget(self.smurf_application +
                           self._smurf_version_reg, as_string=True,
                           **kwargs)

    _smurf_directory_reg = 'SmurfDirectory'

    def get_pysmurf_directory(self, **kwargs):
        r"""Returns path to the pysmurf python files.

        Path to the files from which the pysmurf module was loaded.
        Alias for `pysmurf__file__`.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            Path to pysmurf files.
        """
        return self._caget(self.smurf_application +
                           self._smurf_directory_reg, as_string=True,
                           **kwargs)

    _smurf_startup_script_reg = 'StartupScript'

    def get_smurf_startup_script(self, **kwargs):
        r"""Returns path to the pysmurf server startup script.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            Path to pysmurf server startup script.
        """
        return self._caget(self.smurf_application +
                           self._smurf_startup_script_reg, as_string=True,
                           **kwargs)

    _smurf_startup_arguments_reg = 'StartupArguments'

    def get_smurf_startup_args(self, **kwargs):
        r"""Returns pysmurf server startup arguments.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            pysmurf server startup arguments.
        """
        return self._caget(self.smurf_application +
                           self._smurf_startup_arguments_reg,
                           as_string=True, **kwargs)

    _enabled_bays_reg = "EnabledBays"

    def get_enabled_bays(self, **kwargs):
        r"""Returns list of enabled AMC bays.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        bays : list of int
            Which bays were enabled on pysmurf server startup.

        """
        enabled_bays = self._caget(
            self.smurf_application +
            self._enabled_bays_reg,
            **kwargs)
        try:
            return list(enabled_bays)
        except Exception:
            return enabled_bays

    _configuring_in_progress_reg = 'ConfiguringInProgress'

    def get_configuring_in_progress(self, **kwargs):
        r"""Whether or not configuration process in progress.

        Set to `True` when the rogue `setDefaults` command is called
        (usually by a call to :func:`set_defaults_pv`), and then set
        to `False` when the rogue `setDefaults` method exits.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        bool or None
           Boolean flag indicating whether or not the configuration
           process is in progress.  Returns `True` if the
           configuration process is in progress, otherwise returns
           `False`. If the underlying PV can not be read, this
           function returns `None`.

        See Also
        --------
        :func:`set_defaults_pv` : Loads the default configuration.

        :func:`get_system_configured` : Returns final state of
                configuration process.

        """
        ret = self._caget(self.smurf_application +
                          self._configuring_in_progress_reg,
                          as_string=True, **kwargs)
        if ret == 'True':
            return True
        elif ret == 'False':
            return False
        else:
            return None

    _system_configured_reg = 'SystemConfigured'

    def get_system_configured(self, **kwargs):
        r"""Returns final state of the configuration process.

        If the configuration was loaded without errors by the rogue
        `setDefaults` command (usually by a call to
        :func:`set_defaults_pv`) and all tests pass, this flag is set
        to `True` when the rogue `setDefaults` method exits.

        .. warning::
           The register used to check if the system has been
           configured, AMCc:SmurfApplication:SystemConfigured, is a
           software register, and does not persist if the Rogue server
           is restarted.  So this function will only tell you if the
           system has been successfully configured at any point during
           the current Rogue server session.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        bool or None
           Boolean flag indicating the final state of the
           configuration process.  If the configuration was loaded
           without errors and all tests passed, then this flag is set
           to `True`.  Otherwise it is set to `False`. If the
           underlying PV can not be read, this function returns
           `None`.

        See Also
        --------
        :func:`set_defaults_pv` : Loads the default configuration.

        :func:`get_configuring_in_progress` : Whether or not
                configuration process in progress.

        """
        ret = self._caget(self.smurf_application +
                          self._system_configured_reg,
                          as_string=True, **kwargs)

        if ret == 'True':
            return True
        elif ret == 'False':
            return False
        else:
            return None

    #### End SmurfApplication gets/sets

    _rogue_version_reg = 'RogueVersion'

    def get_rogue_version(self, **kwargs):
        r"""Get rogue version

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            The rogue version
        """
        return self._caget(self.amcc + self._rogue_version_reg,
                           as_string=True, **kwargs)

    def get_enable(self, **kwargs):
        """
        No description

        Returns
        -------
        str
            The status of the global poll bit AMCc.enable.
            If False, pyrogue is not currently polling the server. PVs
            will not be updating.
        """
        return self._caget(self._global_poll_enable_reg, **kwargs)


    _number_sub_bands_reg = 'numberSubBands'

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
            band = self._bands[0]

        return self._caget(self._band_root(band) + self._number_sub_bands_reg,
            **kwargs)


    _number_channels_reg = 'numberChannels'

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
            band = self._bands[0]

        return self._caget(self._band_root(band) + self._number_channels_reg,
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

    def set_defaults_pv(self, wait_after_sec=30.0,
                        max_timeout_sec=400.0, caget_timeout_sec=10.0,
                        **kwargs):
        r"""Loads the default configuration.

        Calls the rogue `setDefaults` command, which loads the default
        software and hardware configuration.

        If using pysmurf core code versions >=4.1.0 (as reported by
        :func:`get_pysmurf_version`), returns `True` if the
        `setDefaults` command was successfully executed on the rogue
        side, or failed.  Returns `None` for older versions.

        Args
        ----
        wait_after_sec : float or None, optional, default 30.0
            If not None, the number of seconds to wait after
            triggering the rogue `setDefaults` command.
        max_timeout_sec : float, optional, default 400.0
            Seconds to wait for system to configure before giving up.
            Only used for pysmurf core code versions >= 4.1.0.
        caget_timeout_sec : float, optional, default 10.0
            Seconds to wait for each poll of the configuration process
            status registers (see :func:`get_configuring_in_progress`
            and :func:`get_system_configured`).  Only used for pysmurf
            core code versions >= 4.1.0.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to
            all `_caget` calls.

        Returns
        -------
        success : bool or None
            Returns `True` if the system was successfully configured,
            otherwise returns `False`.  Configuration checking is only
            implemented for pysmurf core code versions >= 4.1.0.  If
            configuration validation is not available, returns None.

        See Also
        --------
        :func:`get_configuring_in_progress` : Whether or not
                configuration process in progress.
        :func:`get_system_configured` : Returns final state of
                configuration process.

        """
        # strip any commit info off the end of the pysmurf version
        # string
        pysmurf_version = self.get_pysmurf_version(**kwargs).split('+')[0]

        # Extra registers allow confirmation of successful
        # configuration for pysmurf versions >=4.1.0.
        # see https://github.com/slaclab/pysmurf/issues/462
        # for more details.
        if version.parse(pysmurf_version) >= version.parse('4.1.0'):
            # Will report how long the configuration process takes
            # once complete.
            start_time = time.time()

            # Start by calling the 'setDefaults' command. Set the 'wait' flag
            # to wait for the command to finish, although the server usually
            # gets unresponsive during setup and the connection is lost.
            self._caput('AMCc.setDefaults', 1, **kwargs)

            # Now let's wait until the process is finished. We define a maximum
            # time we will wait, 400 seconds in this case, divided in smaller
            # tries of 10 second each
            num_retries = int(max_timeout_sec/caget_timeout_sec)
            success = False
            for _ in range(num_retries):
                # Try to read the status of the
                # "ConfiguringInProgress" flag.
                #
                # We successfully exit the loop when we are able to
                # read the "ConfiguringInProgress" flag and it is set
                # to "False".  Otherwise we keep trying.
                # We disable the retry_on_fail feature and instead we catch any
                # RuntimeError exception and keep trying.
                try:
                    if self.get_configuring_in_progress(
                            timeout=caget_timeout_sec,
                            retry_on_fail=False, **kwargs) is False:
                        success=True
                        break
                except RuntimeError:
                    pass

            # If after out maximum defined timeout, we weren't able to
            # read the "ConfiguringInProgress" flags as "False", we
            # error on error.
            if not success:
                self.log(
                    'The system configuration did not finish after'
                    f' {max_timeout_sec} seconds.', self.LOG_ERROR)
                return False

            # At this point, we determine that the configuration
            # sequence ended in the server via the
            # "ConfiguringInProgress" flag.
            # The final status of the configuration sequence is
            # available in the "SystemConfigured" flag.
            # So, let's read it and use it as out return value.
            success = self.get_system_configured(
                timeout=caget_timeout_sec, **kwargs)

            # Measure how long the process take
            end_time = time.time()

            self.log(
                'System configuration finished after'
                f' {int(end_time - start_time)} seconds.'
                f' The final state was {success}.',
                self.LOG_USER)

            return success

        else:
            self._caput('AMCc.setDefaults', 1, **kwargs)
            return None

    def set_read_all(self, **kwargs):
        """
        ReadAll sends a command to read all register to the pyrogue server
        Registers must updated in order to PVs to update.
        This call is necessary to read register with pollIntervale=0.
        """
        self._caput('AMCc.ReadAll', 1, **kwargs)
        self.log('ReadAll sent', self.LOG_INFO)

    def run_pwr_up_sys_ref(self,bay, **kwargs):
        """
        """
        triggerPV=self.lmk.format(bay) + 'PwrUpSysRef'
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    _eta_scan_in_progress_reg = 'etaScanInProgress'

    def get_eta_scan_in_progress(self, band, **kwargs):
        """
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_in_progress_reg,
                    **kwargs)

    _gradient_descent_max_iters_reg = 'gradientDescentMaxIters'

    def set_gradient_descent_max_iters(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_max_iters_reg,
            val, **kwargs)

    def get_gradient_descent_max_iters(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_max_iters_reg,
            **kwargs)

    _gradient_descent_averages_reg = 'gradientDescentAverages'

    def set_gradient_descent_averages(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_averages_reg,
            val, **kwargs)

    def get_gradient_descent_averages(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_averages_reg,
            **kwargs)

    _gradient_descent_gain_reg = 'gradientDescentGain'

    def set_gradient_descent_gain(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_gain_reg,
            val, **kwargs)

    def get_gradient_descent_gain(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_gain_reg,
            **kwargs)

    _gradient_descent_converge_hz_reg = 'gradientDescentConvergeHz'

    def set_gradient_descent_converge_hz(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_converge_hz_reg,
            val, **kwargs)

    def get_gradient_descent_converge_hz(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_converge_hz_reg,
            **kwargs)

    _gradient_descent_step_hz_reg = 'gradientDescentStepHz'

    def set_gradient_descent_step_hz(self, band, val, **kwargs):
        """
        Sets the step size of the gradient descent in units of Hz
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_step_hz_reg,
            val, **kwargs)

    def get_gradient_descent_step_hz(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_step_hz_reg,
            **kwargs)

    _gradient_descent_momentum_reg = 'gradientDescentMomentum'

    def set_gradient_descent_momentum(self, band, val, **kwargs):
        """
        Sets the momentum term of the gradient descent
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_momentum_reg,
            val, **kwargs)

    def get_gradient_descent_momentum(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_momentum_reg,
            **kwargs)

    _gradient_descent_beta_reg = 'gradientDescentBeta'

    def set_gradient_descent_beta(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_beta_reg,
            val, **kwargs)

    def get_gradient_descent_beta(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_beta_reg,
            **kwargs)

    def run_parallel_eta_scan(self, band, sync_group=True, **kwargs):
        """
        runParallelScan
        """
        triggerPV=self._cryo_root(band) + 'runParallelEtaScan'
        monitorPV=(
            self._cryo_root(band) + self._eta_scan_in_progress_reg)

        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

        if sync_group:

            varList = [self._client.root.getNode(monitorPV)]


            sg = SyncGroup([monitorPV], self._client)
            sg.wait()
            vals = sg.get_values()
            self.log(
                'parallel etaScan complete ; etaScanInProgress = ' +
                f'{vals[monitorPV]}', self.LOG_USER)

    _run_serial_eta_scan_reg = 'runSerialEtaScan'

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

        triggerPV = self._cryo_root(band) + self._run_serial_eta_scan_reg
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], self._client, timeout=timeout)
            sg.wait()
            sg.get_values()


    _run_serial_min_search_reg = 'runSerialMinSearch'

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
        triggerPV = (
            self._cryo_root(band) + self._run_serial_min_search_reg)
        monitorPV = (
            self._cryo_root(band) + self._eta_scan_in_progress_reg)

        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        if sync_group:
            sg = SyncGroup([monitorPV], self._client, timeout=timeout)
            sg.wait()
            sg.get_values()


    _run_serial_gradient_descent_reg = 'runSerialGradientDescent'

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

        triggerPV = self._cryo_root(band) + self._run_serial_gradient_descent_reg
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg

        self._caput(triggerPV, 1, wait_after=5, **kwargs)

        if sync_group:
            sg = SyncGroup([monitorPV], self._client, timeout=timeout)
            sg.wait()
            sg.get_values()


    _sel_ext_ref_reg = "SelExtRef"

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
        triggerPV=(self.microwave_mux_core.format(bay) +
                   self._sel_ext_ref_reg)
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    # name changed in Rogue 4 from WriteState to SaveState.  Keeping
    # the write_state function for backwards compatibilty.
    _save_state_reg = "AMCc.SaveState"

    def save_state(self, val, **kwargs):
        """
        Dumps all PyRogue state variables to a yml file.

        Args
        ----
        val : str
            The path (including file name) to write the yml file to.
        """
        self._caput(self._save_state_reg, val, **kwargs)

    # alias older rogue 3 write_state function to save_state
    write_state = save_state

    # name changed in Rogue 4 from WriteConfig to SaveConfig.  Keeping
    # the write_config function for backwards compatibilty.
    _save_config_reg = "AMCc.SaveConfig"

    def save_config(self, val, **kwargs):
        """
        Writes the current (un-masked) PyRogue settings to a yml file.

        Args
        ----
        val : str
            The path (including file name) to write the yml file to.
        """
        self._caput(self._save_config_reg, val, **kwargs)

    # alias older rogue 3 write_config function to save_config
    write_config = save_config

    _tone_file_path_reg = 'CsvFilePath'

    def get_tone_file_path(self, bay, **kwargs):
        r"""Get tone file path.

        Returns the tone file path that's currently being used for
        this bay.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            Full path to tone file.
        """

        return self._caget(
            self.dac_sig_gen.format(bay) + self._tone_file_path_reg,
            as_string=True, **kwargs)

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

        self._caput(
            self.dac_sig_gen.format(bay) + self._tone_file_path_reg,
            val, **kwargs)

    _load_tone_file_reg = 'LoadCsvFile'

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
        self._caput(
            self.dac_sig_gen.format(bay) + self._load_tone_file_reg,
            val, **kwargs)

    _tune_file_path_reg = 'tuneFilePath'

    def set_tune_file_path(self, val, **kwargs):
        """
        """
        self._caput(
            self.sysgencryo + self._tune_file_path_reg,
            val, **kwargs)

    def get_tune_file_path(self, **kwargs):
        """
        """
        return self._caget(
            self.sysgencryo + self._tune_file_path_reg,
            **kwargs)

    _load_tune_file_reg = 'loadTuneFile'

    def set_load_tune_file(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._load_tune_file_reg,
            val, **kwargs)

    def get_load_tune_file(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._load_tune_file_reg,
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
            self._cryo_root(band) + self._eta_scan_del_f_reg,
            val, **kwargs)

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

    _eta_scan_freqs_reg = 'etaScanFreqs'

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
        self._caput(
            self._cryo_root(band) + self._eta_scan_freqs_reg,
            val, **kwargs)


    def get_eta_scan_freq(self, band, **kwargs):
        """
        No description

        Args
        ----
        band : int
            The band to count.

        Returns
        -------
        freq : int
            The frequency of the scan.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_freqs_reg,
            **kwargs)

    _eta_scan_amplitude_reg = 'etaScanAmplitude'

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
        self._caput(
            self._cryo_root(band) + self._eta_scan_amplitude_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._eta_scan_amplitude_reg,
            **kwargs)

    _eta_scan_channel_reg = 'etaScanChannel'

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
        self._caput(
            self._cryo_root(band) + self._eta_scan_channel_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._eta_scan_channel_reg,
            **kwargs)

    _eta_scan_averages_reg = 'etaScanAverages'

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
        self._caput(
            self._cryo_root(band) + self._eta_scan_averages_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._eta_scan_averages_reg,
            **kwargs)

    _eta_scan_dwell_reg = 'etaScanDwell'

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
        self._caput(
            self._cryo_root(band) + self._eta_scan_dwell_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._eta_scan_dwell_reg,
            **kwargs)

    _run_serial_find_freq_reg = 'runSerialFindFreq'

    def set_run_serial_find_freq(self, band, val, sync_group=True, **kwargs):
        """
        Runs the eta scan. Set the channel using set_eta_scan_channel()

        Args
        ----
        band : int
            The band to eta scan.
        val : bool
            Start the eta scan.
        """
        self._caput(
            self._cryo_root(band) + self._run_serial_find_freq_reg,
            val, **kwargs)

        monitorPV=(
            self._cryo_root(band) + self._eta_scan_in_progress_reg)

        inProgress = True
        if sync_group:
            while inProgress:
                sg = SyncGroup([monitorPV], self._client, timeout=360)
                sg.wait()
                vals = sg.get_values()
                inProgress = (vals[monitorPV] == 1)
            self.log(
                'serial find freq complete ; etaScanInProgress = ' +
                f'{vals[monitorPV]}', self.LOG_USER)

    _run_eta_scan_reg = 'runEtaScan'

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
        self._caput(
            self._cryo_root(band) + self._run_eta_scan_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._run_eta_scan_reg,
            **kwargs)

    _eta_scan_results_real_reg = 'etaScanResultsReal'

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
        return self._caget(
            self._cryo_root(band) + self._eta_scan_results_real_reg,
            count=count, **kwargs)

    _eta_scan_results_imag_reg = 'etaScanResultsImag'

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
        return self._caget(
            self._cryo_root(band) + self._eta_scan_results_imag_reg,
            count=count, **kwargs)

    _amplitude_scales_reg = 'setAmplitudeScales'

    def set_amplitude_scales(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._amplitude_scales_reg,
            val, **kwargs)

    def get_amplitude_scales(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._amplitude_scales_reg,
            **kwargs)

    _amplitude_scale_array_reg = 'amplitudeScaleArray'

    def set_amplitude_scale_array(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._amplitude_scale_array_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._amplitude_scale_array_reg,
            **kwargs)

    def set_amplitude_scale_array_currentchans(self, band, tone_power,
                                               **kwargs):
        """
        Set only the currently on channels to a new drive power. Essentially
        a more convenient wrapper for set_amplitude_scale_array to only change
        the channels that are on.

        Args
        ----
        band : int
            The band to change.
        tone_power : int
            Tone power to change to.
        """

        old_amp = self.get_amplitude_scale_array(band, **kwargs)
        n_channels=self.get_number_channels(band)
        new_amp = np.zeros((n_channels,),dtype=int)
        new_amp[np.where(old_amp!=0)] = tone_power
        self.set_amplitude_scale_array(self, new_amp, **kwargs)

    _feedback_enable_array_reg = 'feedbackEnableArray'

    def set_feedback_enable_array(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._feedback_enable_array_reg,
            val, **kwargs)

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
        return self._caget(
            self._cryo_root(band) + self._feedback_enable_array_reg,
            **kwargs)

    _single_channel_readout_reg = 'singleChannelReadout'

    def set_single_channel_readout(self, band, val, **kwargs):
        """
        Sets the singleChannelReadout bit.

        Args
        ----
        band : int
            The band to set to single channel readout.
        """
        self._caput(
            self._band_root(band) + self._single_channel_readout_reg,
            val, **kwargs)

    def get_single_channel_readout(self, band, **kwargs):
        """

        """
        return self._caget(
            self._band_root(band) + self._single_channel_readout_reg,
            **kwargs)

    _single_channel_readout2_reg = 'singleChannelReadoutOpt2'

    def set_single_channel_readout_opt2(self, band, val, **kwargs):
        """
        Sets the singleChannelReadout2 bit.

        Args
        ----
        band : int
            The band to set to single channel readout.
        """
        self._caput(
            self._band_root(band) + self._single_channel_readout2_reg,
            val, **kwargs)

    def get_single_channel_readout_opt2(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._single_channel_readout2_reg,
            **kwargs)

    _readout_channel_select_reg = 'readoutChannelSelect'

    def set_readout_channel_select(self, band, channel, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._readout_channel_select_reg,
            channel, **kwargs)

    def get_readout_channel_select(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._readout_channel_select_reg,
            **kwargs)

    _stream_enable_reg = 'enableStreaming'

    def set_stream_enable(self, val, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        self._caput(self.app_core + self._stream_enable_reg, val, **kwargs)

    def get_stream_enable(self, **kwargs):
        """
        Enable/disable streaming data, for all bands.
        """
        return self._caget(
            self.app_core + self._stream_enable_reg,
            **kwargs)

    _rf_iq_stream_enable_reg = 'rfIQStreamEnable'

    def set_rf_iq_stream_enable(self, band, val, **kwargs):
        """
        Sets the bit that turns on RF IQ streaming for take_debug_data

        Args
        ----
        band : int
            The 500 Mhz band
        val : int or bool
            Whether to set the mode to RF IQ.
        """
        self._caput(self._band_root(band) +
                    self._rf_iq_stream_enable_reg,
                    val, **kwargs)

    def get_rf_iq_stream_enable(self, band, **kwargs):
        """
        gets the bit that turns on RF IQ streaming for take_debug_data

        Args
        ----
        band : int
            The 500 MHz band

        Ret
        ---
        rf_iq_stream_bit : int
            The bit that sets the RF streaming
        """
        return self._caget(self._band_root(band) +
                           self._rf_iq_stream_enable_reg,
                           **kwargs)


    _build_dsp_g_reg = 'BUILD_DSP_G'

    def get_build_dsp_g(self, **kwargs):
        """
        BUILD_DSP_G encodes which bands the fw being used was built for.
        E.g. 0xFF means Base[0...7], 0xF is Base[0...3], etc.

        """
        return self._caget(
            self.app_core + self._build_dsp_g_reg,
            **kwargs)

    _decimation_reg = 'decimation'

    def set_decimation(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._decimation_reg,
            val, **kwargs)

    def get_decimation(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._decimation_reg,
            **kwargs)

    _filter_alpha_reg = 'filterAlpha'

    def set_filter_alpha(self, band, val, **kwargs):
        """
        Coefficient for single pole low pass fitler before readout
        (when channels are multiplexed, decimated)
        y[n] = alpha*x[n] + (1 - alpha)*y[n-1]
        matlab to visualize
        h = fvtool([alpha], [1 -(1-alpha)]); h.Fs = 2.4e6;
        """
        self._caput(
            self._band_root(band) + self._filter_alpha_reg,
            val, **kwargs)

    def get_filter_alpha(self, band, **kwargs):
        """
        Coefficient for single pole low pass fitler before readout
        (when channels are multiplexed, decimated)
        y[n] = alpha*x[n] + (1 - alpha)*y[n-1]
        matlab to visualize
        h = fvtool([alpha], [1 -(1-alpha)]); h.Fs = 2.4e6;
        """
        return self._caget(
            self._band_root(band) + self._filter_alpha_reg,
            **kwargs)

    _iq_swap_in_reg = 'iqSwapIn'

    def set_iq_swap_in(self, band, val, **kwargs):
        """
        Swaps I&Q into DSP (from ADC).  Tones being output by the
        system will flip about the band center (e.g. 4.25GHz, 5.25GHz
        etc.)
        """
        self._caput(
            self._band_root(band) + self._iq_swap_in_reg,
            val, **kwargs)

    def get_iq_swap_in(self, band, **kwargs):
        """
        Swaps I&Q into DSP (from ADC).  Tones being output by the
        system will flip about the band center (e.g. 4.25GHz, 5.25GHz
        etc.)
        """
        return self._caget(
            self._band_root(band) + self._iq_swap_in_reg,
            **kwargs)

    _iq_swap_out_reg = 'iqSwapOut'

    def set_iq_swap_out(self, band, val, **kwargs):
        """
        Swaps I&Q out of DSP (to DAC).  Swapping I&Q flips spectrum
        around band center.
        """
        self._caput(
            self._band_root(band) + self._iq_swap_out_reg,
            val, **kwargs)

    def get_iq_swap_out(self, band, **kwargs):
        """
        Swaps I&Q out of DSP (to DAC).  Swapping I&Q flips spectrum
        around band center.
        """
        return self._caget(
            self._band_root(band) + self._iq_swap_out_reg,
            **kwargs)

    _ref_phase_delay_reg = 'refPhaseDelay'

    def set_ref_phase_delay(self, band, val, **kwargs):
        """
        Deprecated.  Use set_band_delay_us instead.
        """
        self._caput(
            self._band_root(band) + self._ref_phase_delay_reg,
            val, **kwargs)

    def get_ref_phase_delay(self, band, **kwargs):
        """
        Deprecated.  Use get_band_delay_us instead.
        """
        return self._caget(
            self._band_root(band) + self._ref_phase_delay_reg,
            **kwargs)

    _ref_phase_delay_fine_reg = 'refPhaseDelayFine'

    def set_ref_phase_delay_fine(self, band, val, **kwargs):
        """
        Deprecated.  Use set_band_delay_us instead.
        """
        self._caput(
            self._band_root(band) + self._ref_phase_delay_fine_reg,
            val, **kwargs)

    def get_ref_phase_delay_fine(self, band, **kwargs):
        """
        Deprecated.  Use get_band_delay_us instead.
        """
        return self._caget(
            self._band_root(band) + self._ref_phase_delay_fine_reg,
            **kwargs)

    _band_delay_us_reg = 'bandDelayUs'

    def set_band_delay_us(self, band, val, **kwargs):
        """
        Set band delay compensation, microseconds.  Corrects
        for total system delay (cable, DSP, etc.).  Internally
        configures both ref_phase_delay and ref_phase_delay_fine
        """
        self._caput(
            self._band_root(band) + self._band_delay_us_reg,
            val, **kwargs)

    def get_band_delay_us(self, band, **kwargs):
        """
        Get band delay compensation, microseconds.  Corrects
        for total system delay (cable, DSP, etc.).
        """
        return self._caget(
            self._band_root(band) + self._band_delay_us_reg,
            **kwargs)

    _tone_scale_reg = 'toneScale'

    def set_tone_scale(self, band, val, **kwargs):
        """
        Scales the sum of 16 tones before synthesizer.
        """
        self._caput(
            self._band_root(band) + self._tone_scale_reg,
            val, **kwargs)

    def get_tone_scale(self, band, **kwargs):
        """
        Scales the sum of 16 tones before synthesizer.
        """
        return self._caget(
            self._band_root(band) + self._tone_scale_reg,
            **kwargs)

    _waveform_select_reg = 'waveformSelect'

    def set_waveform_select(self, band, val, **kwargs):
        """
        0x0 select DSP -> DAC
        0x1 selects waveform table -> DAC (toneFile)
        """
        self._caput(
            self._band_root(band) + self._waveform_select_reg,
            val, **kwargs)

    def get_waveform_select(self, band, **kwargs):
        """
        0x0 select DSP -> DAC
        0x1 selects waveform table -> DAC (toneFile)
        """
        return self._caget(
            self._band_root(band) + self._waveform_select_reg,
            **kwargs)

    _waveform_start_reg = 'waveformStart'

    def set_waveform_start(self, band, val, **kwargs):
        """
        0x1 enables waveform table
        """
        self._caput(
            self._band_root(band) + self._waveform_start_reg,
            val, **kwargs)

    def get_waveform_start(self, band, **kwargs):
        """
        0x1 enables waveform table
        """
        return self._caget(
            self._band_root(band) + self._waveform_start_reg,
            **kwargs)

    _rf_enable_reg = 'rfEnable'

    def set_rf_enable(self, band, val, **kwargs):
        """
        0x0 output all 0s to DAC
        0x1 enable output to DAC (from DSP or waveform table)
        """
        self._caput(
            self._band_root(band) + self._rf_enable_reg,
            val, **kwargs)

    def get_rf_enable(self, band, **kwargs):
        """
        0x0 output all 0s to DAC
        0x1 enable output to DAC (from DSP or waveform table)
        """
        return self._caget(
            self._band_root(band) + self._rf_enable_reg,
            **kwargs)

    _analysis_scale_reg = 'analysisScale'

    def set_analysis_scale(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._analysis_scale_reg,
            val, **kwargs)

    def get_analysis_scale(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._analysis_scale_reg,
            **kwargs)

    _feedback_enable_reg = 'feedbackEnable'

    def set_feedback_enable(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._feedback_enable_reg,
            val, **kwargs)

    def get_feedback_enable(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._feedback_enable_reg,
            **kwargs)

    _loop_filter_output_array_reg = 'loopFilterOutputArray'

    def get_loop_filter_output_array(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) +
            self._loop_filter_output_array_reg,
            **kwargs)

    _tone_frequency_offset_mhz_reg = 'toneFrequencyOffsetMHz'

    def set_tone_frequency_offset_mhz(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) +
            self._tone_frequency_offset_mhz_reg,
            val, **kwargs)

    def get_tone_frequency_offset_mhz(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) +
            self._tone_frequency_offset_mhz_reg,
            **kwargs)

    _center_frequency_array_reg = 'centerFrequencyArray'

    def set_center_frequency_array(self, band, val, **kwargs):
        """
        Sets all the center frequencies in a band
        """
        self._caput(
            self._cryo_root(band) + self._center_frequency_array_reg,
            val, **kwargs)

    def get_center_frequency_array(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._center_frequency_array_reg,
            **kwargs)

    _feedback_gain_reg = 'feedbackGain'

    def set_feedback_gain(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._feedback_gain_reg,
            val, **kwargs)

    def get_feedback_gain(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._feedback_gain_reg,
            **kwargs)

    _eta_phase_array_reg = 'etaPhaseArray'

    def set_eta_phase_array(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._eta_phase_array_reg,
            val, **kwargs)

    def get_eta_phase_array(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._eta_phase_array_reg,
            **kwargs)

    _frequency_error_array_reg = 'frequencyErrorArray'

    def set_frequency_error_array(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._frequency_error_array_reg,
            val, **kwargs)

    def get_frequency_error_array(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._frequency_error_array_reg,
            **kwargs)

    _eta_mag_array_reg = 'etaMagArray'

    def set_eta_mag_array(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._cryo_root(band) + self._eta_mag_array_reg,
            val, **kwargs)

    def get_eta_mag_array(self, band, **kwargs):
        """
        """
        return self._caget(
            self._cryo_root(band) + self._eta_mag_array_reg,
            **kwargs)

    _feedback_limit_reg = 'feedbackLimit'

    def set_feedback_limit(self, band, val, **kwargs):
        """
        freq = centerFreq + feedbackFreq
        abs(freq) < centerFreq + feedbackLimit
        """
        self._caput(
            self._band_root(band) + self._feedback_limit_reg,
            val, **kwargs)

    def get_feedback_limit(self, band, **kwargs):
        """
        freq = centerFreq + feedbackFreq
        abs(freq) < centerFreq + feedbackLimit
        """
        return self._caget(
            self._band_root(band) + self._feedback_limit_reg,
            **kwargs)

    _noise_select_reg = 'noiseSelect'

    def set_noise_select(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._noise_select_reg,
            val, **kwargs)

    def get_noise_select(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._noise_select_reg,
            **kwargs)

    _lms_delay_reg = 'lmsDelay'

    def set_lms_delay(self, band, val, **kwargs):
        """
        Match system latency for LMS feedback (2.4MHz ticks)
        """
        self._caput(
            self._band_root(band) + self._lms_delay_reg,
            val, **kwargs)

    def get_lms_delay(self, band, **kwargs):
        """
        Match system latency for LMS feedback (2.4MHz ticks)
        """
        return self._caget(
            self._band_root(band) + self._lms_delay_reg,
            **kwargs)

    _lms_gain_reg = 'lmsGain'

    def set_lms_gain(self, band, val, **kwargs):
        """
        LMS gain, powers of 2
        """
        self._caput(
            self._band_root(band) + self._lms_gain_reg,
            val, **kwargs)

    def get_lms_gain(self, band, **kwargs):
        """
        LMS gain, powers of 2
        """
        return self._caget(
            self._band_root(band) + self._lms_gain_reg,
            **kwargs)

    _trigger_reset_delay_reg = 'trigRstDly'

    def set_trigger_reset_delay(self, band, val, **kwargs):
        """
        Trigger reset delay, set such that the ramp resets at the flux
        ramp glitch.  2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        self._caput(
            self._band_root(band) + self._trigger_reset_delay_reg,
            val, **kwargs)

    def get_trigger_reset_delay(self, band, **kwargs):
        """
        Trigger reset delay, set such that the ramp resets at the flux
        ramp glitch.  2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        return self._caget(
            self._band_root(band) + self._trigger_reset_delay_reg,
            **kwargs)

    _feedback_start_reg = 'feedbackStart'

    def set_feedback_start(self, band, val, **kwargs):
        """
        The flux ramp DAC value at which to start applying feedback in
        each flux ramp cycle.  In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        self._caput(
            self._band_root(band) + self._feedback_start_reg,
            val, **kwargs)

    def get_feedback_start(self, band, **kwargs):
        """
        The flux ramp DAC value at which to start applying feedback in
        each flux ramp cycle.  In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        return self._caget(
            self._band_root(band) + self._feedback_start_reg,
            **kwargs)

    _feedback_end_reg = 'feedbackEnd'

    def set_feedback_end(self, band, val, **kwargs):
        """
        The flux ramp DAC value at which to stop applying feedback in
        each flux ramp cycle.  In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        self._caput(
            self._band_root(band) + self._feedback_end_reg,
            val, **kwargs)

    def get_feedback_end(self, band, **kwargs):
        """
        The flux ramp DAC value at which to stop applying feedback in
        each flux ramp cycle.  In 2.4 MHz ticks.

        Args
        ----
        band : int
            Which band.
        """
        return self._caget(
            self._band_root(band) + self._feedback_end_reg,
            **kwargs)

    _lms_enable1_reg = 'lmsEnable1'

    def set_lms_enable1(self, band, val, **kwargs):
        """
        Enable 1st harmonic tracking
        """
        self._caput(
            self._band_root(band) + self._lms_enable1_reg,
            val, **kwargs)

    def get_lms_enable1(self, band, **kwargs):
        """
        Enable 1st harmonic tracking
        """
        return self._caget(
            self._band_root(band) + self._lms_enable1_reg,
            **kwargs)

    _lms_enable2_reg = 'lmsEnable2'

    def set_lms_enable2(self, band, val, **kwargs):
        """
        Enable 2nd harmonic tracking
        """
        self._caput(
            self._band_root(band) + self._lms_enable2_reg,
            val, **kwargs),

    def get_lms_enable2(self, band, **kwargs):
        """
        Enable 2nd harmonic tracking
        """
        return self._caget(
            self._band_root(band) + self._lms_enable2_reg,
            **kwargs)

    _lms_enable3_reg = 'lmsEnable3'

    def set_lms_enable3(self, band, val, **kwargs):
        """
        Enable 3rd harmonic tracking
        """
        self._caput(
            self._band_root(band) + self._lms_enable3_reg,
            val, **kwargs)

    def get_lms_enable3(self, band, **kwargs):
        """
        Enable 3rd harmonic tracking
        """
        return self._caget(
            self._band_root(band) + self._lms_enable3_reg,
            **kwargs)

    _lms_rst_dly_reg = 'lmsRstDly'

    def set_lms_rst_dly(self, band, val, **kwargs):
        """
        Disable feedback after reset (2.4MHz ticks)
        """
        self._caput(
            self._band_root(band) + self._lms_rst_dly_reg,
            val, **kwargs)

    def get_lms_rst_dly(self, band, **kwargs):
        """
        Disable feedback after reset (2.4MHz ticks)
        """
        return self._caget(
            self._band_root(band) + self._lms_rst_dly_reg,
            **kwargs)

    _lms_freq_reg = 'lmsFreq'

    def set_lms_freq(self, band, val, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        self._caput(
            self._band_root(band) + self._lms_freq_reg,
            val, **kwargs)

    def get_lms_freq(self, band, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        return self._caget(
            self._band_root(band) + self._lms_freq_reg,
            **kwargs)

    _lms_freq_hz_reg = 'lmsFreqHz'

    def set_lms_freq_hz(self, band, val, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        self._caput(
            self._band_root(band) + self._lms_freq_hz_reg,
            val, **kwargs)

    def get_lms_freq_hz(self, band, **kwargs):
        """
        LMS frequency = flux ramp freq * nPhi0
        """
        return self._caget(
            self._band_root(band) + self._lms_freq_hz_reg,
            **kwargs)

    _lms_dly_fine_reg = 'lmsDlyFine'

    def set_lms_dly_fine(self, band, val, **kwargs):
        """
        fine delay control (38.4MHz ticks)
        """
        self._caput(
            self._band_root(band) + self._lms_dly_fine_reg,
            val, **kwargs)

    def get_lms_dly_fine(self, band, **kwargs):
        """
        fine delay control (38.4MHz ticks)
        """
        return self._caget(
            self._band_root(band) + self._lms_dly_fine_reg,
            **kwargs)

    _iq_stream_enable_reg = 'iqStreamEnable'

    def set_iq_stream_enable(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._iq_stream_enable_reg,
            val, **kwargs)

    def get_iq_stream_enable(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._iq_stream_enable_reg,
            **kwargs)

    _feedback_polarity_reg = 'feedbackPolarity'

    def set_feedback_polarity(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._feedback_polarity_reg,
            val, **kwargs)

    def get_feedback_polarity(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._feedback_polarity_reg,
            **kwargs)

    _band_center_mhz_reg = 'bandCenterMHz'

    def set_band_center_mhz(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._band_center_mhz_reg,
            val, **kwargs)

    def get_band_center_mhz(self, band, **kwargs):
        """
        Returns the center frequency of the band in MHz
        """
        if self.offline:
            bc = (4250 + band*500)
            return bc
        else:
            return self._caget(
                self._band_root(band) + self._band_center_mhz_reg,
                **kwargs)


    _channel_frequency_mhz_reg = 'channelFrequencyMHz'

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
            band = self._bands[0]

        if self.offline:
            return 2.4
        else:
            return self._caget(
                self._band_root(band) +
                self._channel_frequency_mhz_reg,
                **kwargs)

    _digitizer_frequency_mhz_reg = 'digitizerFrequencyMHz'

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
            band = self._bands[0]

        return self._caget(
            self._band_root(band) + self._digitizer_frequency_mhz_reg,
            **kwargs)

    _synthesis_scale_reg = 'synthesisScale'

    def set_synthesis_scale(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._synthesis_scale_reg,
            val, **kwargs)

    def get_synthesis_scale(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._synthesis_scale_reg,
            **kwargs)

    _dsp_enable_reg = 'dspEnable'

    def set_dsp_enable(self, band, val, **kwargs):
        """
        """
        self._caput(
            self._band_root(band) + self._dsp_enable_reg,
            val, **kwargs)

    def get_dsp_enable(self, band, **kwargs):
        """
        """
        return self._caget(
            self._band_root(band) + self._dsp_enable_reg,
            **kwargs)

    # Single channel commands
    _feedback_enable_reg = 'feedbackEnable'

    def set_feedback_enable_channel(self, band, channel, val,
                                    **kwargs):
        """
        Set the feedback for a single channel
        """
        self._caput(
            self._channel_root(band, channel) +
            self._feedback_enable_reg,
            val, **kwargs)

    def get_feedback_enable_channel(self, band, channel, **kwargs):
        """
        Get the feedback for a single channel
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._feedback_enable_reg,
            **kwargs)

    _eta_mag_scaled_channel_reg = 'etaMagScaled'

    def set_eta_mag_scaled_channel(self, band, channel, val,
                                   **kwargs):
        """
        """
        self._caput(
            self._channel_root(band, channel) +
            self._eta_mag_scaled_channel_reg,
            val, **kwargs)

    def get_eta_mag_scaled_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._eta_mag_scaled_channel_reg,
            **kwargs)

    _center_frequency_mhz_channel_reg = 'centerFrequencyMHz'

    def set_center_frequency_mhz_channel(self, band, channel, val,
                                         **kwargs):
        """
        """
        self._caput(
            self._channel_root(band, channel) +
            self._center_frequency_mhz_channel_reg,
            val, **kwargs)

    def get_center_frequency_mhz_channel(self, band, channel,
                                         **kwargs):
        """
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._center_frequency_mhz_channel_reg,
            **kwargs)


    _amplitude_scale_channel_reg = 'amplitudeScale'

    def set_amplitude_scale_channel(self, band, channel, val,
                                    **kwargs):
        """
        """
        self._caput(
            self._channel_root(band, channel) +
            self._amplitude_scale_channel_reg,
            val, **kwargs)

    def get_amplitude_scale_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._amplitude_scale_channel_reg,
            **kwargs)

    _eta_phase_degree_channel_reg = 'etaPhaseDegree'

    def set_eta_phase_degree_channel(self, band, channel, val,
                                     **kwargs):
        """
        """
        self._caput(
            self._channel_root(band, channel) +
            self._eta_phase_degree_channel_reg,
            val, **kwargs)

    def get_eta_phase_degree_channel(self, band, channel, **kwargs):
        """
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._eta_phase_degree_channel_reg,
            **kwargs)

    _frequency_error_mhz_reg = 'frequencyErrorMHz'

    def get_frequency_error_mhz(self, band, channel, **kwargs):
        """
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._frequency_error_mhz_reg,
            **kwargs)


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
    _uc_reg = 'UC[{}]'

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
        att = int(self.band_to_att(b))
        bay = self.band_to_bay(b)
        self._caput(
            self.att_root.format(bay) + self._uc_reg.format(att),
            val, **kwargs)

    def get_att_uc(self, b, **kwargs):
        """
        Get the upconverter attenuator value

        Args
        ----
        b : int
            The band number.
        """
        att = int(self.band_to_att(b))
        bay = self.band_to_bay(b)
        return self._caget(
            self.att_root.format(bay) + self._uc_reg.format(att),
            **kwargs)


    _dc_reg = 'DC[{}]'

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
        att = int(self.band_to_att(b))
        bay = self.band_to_bay(b)
        self._caput(
            self.att_root.format(bay) + self._dc_reg.format(att),
            val, **kwargs)

    def get_att_dc(self, b, **kwargs):
        """
        Get the down-converter attenuator value

        Args
        ----
        b : int
            The band number.
        """
        att = int(self.band_to_att(b))
        bay = self.band_to_bay(b)
        return self._caget(
            self.att_root.format(bay) + self._dc_reg.format(att),
            **kwargs)

    # ADC commands
    _adc_remap_reg = "Remap[0]"  # Why is this hardcoded 0

    def set_remap(self, **kwargs):
        """
        This command should probably be renamed to something more descriptive.
        """
        self._caput(
            self.adc_root + self._adc_remap_reg,
            1, **kwargs)

    # DAC commands
    _dac_temp_reg = "Temperature"

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
        return self._caget(
            self.dac_root.format(bay,dac) + self._dac_temp_reg,
            **kwargs)

    _dac_enable_reg = "enable"

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
        self._caput(
            self.dac_root.format(bay,dac) + self._dac_enable_reg,
            val, **kwargs)

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
        return self._caget(
            self.dac_root.format(bay,dac) + self._dac_enable_reg,
            **kwargs)

    # Jesd commands
    _data_out_mux_reg = 'dataOutMux[{}]'

    def set_data_out_mux(self, bay, b, val, **kwargs):
        """
        """
        self._caput(
            self.jesd_tx_root.format(bay) +
            self._data_out_mux_reg.format(b),
            val, **kwargs)

    def get_data_out_mux(self, bay, b, **kwargs):
        """
        """
        return self._caget(
            self.jesd_tx_root.format(bay) +
            self._data_out_mux_reg.format(b),
            **kwargs)

    # Jesd DAC commands
    _jesd_reset_n_reg = "JesdRstN"

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
        self._caput(
            self.dac_root.format(bay,dac) + self._jesd_reset_n_reg,
            val, **kwargs)

    _jesd_rx_enable_reg = 'Enable'

    def set_jesd_rx_enable(self, bay, val, **kwargs):
        self._caput(
            self.jesd_rx_root.format(bay) + self._jesd_rx_enable_reg,
            val, **kwargs)

    def get_jesd_rx_enable(self, bay, **kwargs):
        return self._caget(
            self.jesd_rx_root.format(bay) + self._jesd_rx_enable_reg,
            **kwargs)

    _jesd_rx_status_valid_cnt_reg = 'StatusValidCnt'

    def get_jesd_rx_status_valid_cnt(self, bay, num, **kwargs):
        return self._caget(
            self.jesd_rx_root.format(bay) +
            self._jesd_rx_status_valid_cnt_reg + f'[{num}]',
            **kwargs)

    _jesd_rx_data_valid_reg = 'DataValid'

    def get_jesd_rx_data_valid(self, bay, **kwargs):
        return self._caget(
            self.jesd_rx_root.format(bay) +
            self._jesd_rx_data_valid_reg,
            **kwargs)

    _link_disable_reg = 'LINK_DISABLE'

    def set_jesd_link_disable(self, bay, val, **kwargs):
        """
        Disables jesd link
        """
        self._caput(
            self.jesd_rx_root.format(bay) + self._link_disable_reg,
            val, **kwargs)

    def get_jesd_link_disable(self, bay, **kwargs):
        """
        Disables jesd link
        """
        return self._caget(
            self.jesd_rx_root.format(bay) + self._link_disable_reg,
            **kwargs)

    _jesd_tx_enable_reg = 'Enable'

    def set_jesd_tx_enable(self, bay, val, **kwargs):
        self._caput(
            self.jesd_tx_root.format(bay) + self._jesd_tx_enable_reg,
            val, **kwargs)

    def get_jesd_tx_enable(self, bay, **kwargs):
        return self._caget(
            self.jesd_tx_root.format(bay) + self._jesd_tx_enable_reg,
            **kwargs)

    _jesd_tx_data_valid_reg = 'DataValid'

    def get_jesd_tx_data_valid(self, bay, **kwargs):
        return self._caget(
            self.jesd_tx_root.format(bay) +
            self._jesd_tx_data_valid_reg,
            **kwargs)

    _jesd_tx_status_valid_cnt_reg = 'StatusValidCnt'

    def get_jesd_tx_status_valid_cnt(self, bay, num, **kwargs):
        return self._caget(
            self.jesd_tx_root.format(bay) +
            self._jesd_tx_status_valid_cnt_reg + f'[{num}]',
            **kwargs)

    def set_check_jesd(self, max_timeout_sec=60.0,
                       caget_timeout_sec=5.0, **kwargs):
        r"""Runs JESD health check and returns status.

        Toggles the pysmurf core code `SmurfApplication:CheckJesd`
        register, which triggers a call to the `AppTop.JesdHealth`
        method. The command will check if the Rogue ZIP file's
        `AppTop` device contains the `JesdHealth` method, call it if
        it exists, and return the result.

        The `SmurfApplication:CheckJesd` and
        `SmurfApplication:JesdStatus` (see :func:`get_jesd_status`)
        registers are only present in pysmurf core code versions
        >=4.1.0 ; returns `None` if pysmurf core code version is
        <4.1.0.  The `AppTop.JesdHealth` method is only present in
        Rogue ZIP file versions >=0.3.0 ; also returns `None` if the
        `AppTop.JesdHealth` method is not present in the Rogue ZIP
        file.

        Args
        ----
        max_timeout_sec : float, optional, default 60.0
            Seconds to wait for JESD health check to complete before
            giving up.
        caget_timeout_sec : float, optional, default 5.0
            Seconds to wait for each poll of the JESD health check
            status register (see :func:`get_jesd_status`).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to
            all `_caget` calls.

        Returns
        -------
        status : str or None
            Returns JESD health status (see :func:`get_jesd_status`
            for a description of the possible statuses).  Returns
            `None` for pysmurf core code versions < 4.1.0 and Rogue
            ZIP file versions that do not have the `AppTop.JesdHealth`
            method.

        See Also
        --------
        :func:`get_jesd_status` : Gets the status of the Rogue
              `AppTop.JesdHealth` method.

        """
        # strip any commit info off the end of the pysmurf version
        # string
        pysmurf_version = self.get_pysmurf_version(**kwargs).split('+')[0]

        # Extra registers allow confirmation of JESD lock for pysmurf
        # versions >=4.1.0 and Rogue ZIP file versions >=0.3.0.  see
        # https://github.com/slaclab/pysmurf/issues/467 for more
        # details.
        if version.parse(pysmurf_version) >= version.parse('4.1.0'):

            # Will report how long the JESD health check takes to
            # complete.
            start_time = time.time()

            # First, check if the 'JesdStatus' register is set to 'Not found'. That means that the
            # current ZIP file does not contain the new JesdHealth command.
            status = self.get_jesd_status(**kwargs)

            if status == 'Not found':
                self.log(
                    'The `JesdHealth` method is not present in the Rogue'
                    ' ZIP file.'  , self.LOG_ERROR)
                return status

            # If the command exists, then start by calling the `CheckJesd`
            # wrapper command.
            self._caput('AMCc.SmurfApplication.CheckJesd', 1, **kwargs)

            # Now let's wait for it to finish.
            num_retries = int(max_timeout_sec/caget_timeout_sec)
            success = False
            status = None
            for _ in range(num_retries):
                # Try to read the status register.
                status = self.get_jesd_status(timeout=caget_timeout_sec, **kwargs)

                if status not in [None, 'Checking']:
                    success = True
                    break

            # If after out maximum defined timeout, we weren't able to
            # read the "JesdStatus" status register with a valid status,
            # then we exit on error.
            if not success:
                self.log(
                    'JESD health check did not finish after'
                    f' {max_timeout_sec} seconds.', self.LOG_ERROR)
                return status

            # Measure how long the process take
            end_time = time.time()

            self.log(
                'JESD health check finished after'
                f' {int(end_time - start_time)} seconds.'
                f' The final status was {status}.',
                self.LOG_USER)

            return status
        else:
            self.log(
                'The `SmurfApplication:CheckJesd` and '
                ' `SmurfApplication:JesdStatus` registers are not'
                ' implemented for pysmurf core code versions <4.1.0'
                f' (current version is {pysmurf_version}).',
                self.LOG_ERROR)
            return None

    _jesd_status_reg = "JesdStatus"

    def get_jesd_status(self, **kwargs):
        r"""Gets the status of the Rogue `AppTop.JesdHealth` method.

        Returns the status of a call to the `AppTop.JesdHealth`
        method.  States are:

        * "Unlocked" : The `AppTop.JesdHealth` command has been run
          and has reported that the JESD were unlocked (**that's
          bad!**).
        * "Locked" : The `AppTop.JesdHealth` command has been run and
          has reported that the JESD were locked (that's good!).
        * "Checking" : The `AppTop.JesdHealth` command is in progress.
        * "Not found" : The Rogue ZIP file currently in use does not
          contain the `AppTop.JesdHealth` method.
        * `None` : The `SmurfApplication:JesdStatus` register is not
          implemented in pysmurf core code versions <4.1.0.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to
            all `_caget` calls.

        Returns
        -------
        status : str or None
            The status of the Rogue `AppTop.JesdHealth` method.
            Returns None for pysmurf core code versions <4.1.0 where
            the `SmurfApplication:JesdStatus` register is not
            implemented.

        See Also
        --------
        :func:`set_check_jesd` : Gets the status of the Rogue
              `AppTop.JesdHealth` method.

        """
        # strip any commit info off the end of the pysmurf version
        # string
        pysmurf_version = self.get_pysmurf_version(**kwargs).split('+')[0]

        # Extra registers were added to allow confirmation of JESD
        # lock for pysmurf versions >=4.1.0.
        # https://github.com/slaclab/pysmurf/issues/467 for more
        # details.
        if version.parse(pysmurf_version) >= version.parse('4.1.0'):
            # Must be as_string=True to compare below
            status =  self._caget(
                self.smurf_application + self._jesd_status_reg,
                as_string=True, **kwargs)

            if status == 'Not found':
                self.log(
                    'The `JesdHealth` method is not present in the Rogue'
                    ' ZIP file.'  , self.LOG_ERROR)

            return status
        else:
            self.log(
                'The `SmurfApplication:JesdStatus` register is not'
                ' implemented for pysmurf core code versions <4.1.0'
                f' (current version is {pysmurf_version}).',
                self.LOG_ERROR)
            return None

    _fpga_uptime_reg = 'UpTimeCnt'

    def get_fpga_uptime(self, **kwargs):
        """
        No description

        Returns
        -------
        uptime : float
            The FPGA uptime.
        """
        return self._caget(
            self.axi_version + self._fpga_uptime_reg,
            **kwargs)

    _fpga_version_reg = 'FpgaVersion'

    def get_fpga_version(self, **kwargs):
        """
        No description

        Returns
        -------
        version : str
            The FPGA version.
        """
        return self._caget(
            self.axi_version + self._fpga_version_reg,
            **kwargs)

    _fpga_git_hash_reg = 'GitHash'

    def get_fpga_git_hash(self, **kwargs):
        r"""Get the full FPGA firmware SHA-1 git hash.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            The full git SHA-1 hash of the FPGA firmware.
        """
        return self._caget(
            self.axi_version + self._fpga_git_hash_reg,
            as_string=True, **kwargs)

    _fpga_git_hash_short_reg = 'GitHashShort'

    def get_fpga_git_hash_short(self, **kwargs):
        r"""Get the short FPGA firmware SHA-1 git hash.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            The short git SHA-1 hash of the FPGA firmware.
        """
        return self._caget(
            self.axi_version + self._fpga_git_hash_short_reg,
            as_string=True, **kwargs)


    _fpga_build_stamp_reg = 'BuildStamp'

    def get_fpga_build_stamp(self, **kwargs):
        r"""Get the FPGA build stamp.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            The FPGA build stamp.
        """
        return self._caget(
            self.axi_version + self._fpga_build_stamp_reg,
            as_string=True, **kwargs)

    _input_mux_sel_reg = 'InputMuxSel[{}]'

    def set_input_mux_sel(self, bay, lane, val, **kwargs):
        """
        """
        self._caput(
            self.daq_mux_root.format(bay) +
            self._input_mux_sel_reg.format(lane),
            val, **kwargs)

    def get_input_mux_sel(self, bay, lane, **kwargs):
        """
        """
        self._caget(
            self.daq_mux_root.format(bay) +
            self._input_mux_sel_reg.format(lane),
            **kwargs)

    _data_buffer_size_reg = 'DataBufferSize'

    def set_data_buffer_size(self, bay, val, **kwargs):
        """
        Sets the data buffer size for the DAQx
        """
        self._caput(
            self.daq_mux_root.format(bay) +
            self._data_buffer_size_reg,
            val, **kwargs)

    def get_data_buffer_size(self, bay, **kwargs):
        """
        Gets the data buffer size for the DAQs
        """
        return self._caget(
            self.daq_mux_root.format(bay) +
            self._data_buffer_size_reg,
            **kwargs)

    # Waveform engine commands
    _start_addr_reg = 'StartAddr[{}]'

    def set_waveform_start_addr(self, bay, engine, val, **kwargs):
        """
        No description

        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int or str
            What value to set.
        """
        if isinstance(val, str):
            val = int(val, 16)
        self._caput(
            self.waveform_engine_buffers_root.format(bay) +
            self._start_addr_reg.format(engine),
            val, **kwargs)

    def get_waveform_start_addr(self, bay, engine, convert=True, **kwargs):
        """
        No description

        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        convert : bool, optional, default True
            Convert the output from a string of hex values to an int.
        """

        val = self._caget(
            self.waveform_engine_buffers_root.format(bay) +
            self._start_addr_reg.format(engine),
            as_string=True,
            **kwargs)

        if convert:
            return int(val, 16)
        else:
            return val

    _end_addr_reg = 'EndAddr[{}]'

    def set_waveform_end_addr(self, bay, engine, val, **kwargs):
        """
        No description

        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int
            What val to set.
        """
        if isinstance(val, str):
            val = int(val, 16)
        self._caput(
            self.waveform_engine_buffers_root.format(bay) +
            self._end_addr_reg.format(engine),
            val, **kwargs)

    def get_waveform_end_addr(self, bay, engine, convert=True, **kwargs):
        """
        No description

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
        val = self._caget(
            self.waveform_engine_buffers_root.format(bay) +
            self._end_addr_reg.format(engine),
            as_string=True,
            **kwargs)

        if convert:
            return int(val, 16)
        else:
            return val

    _wr_addr_reg = 'WrAddr[{}]'

    def set_waveform_wr_addr(self, bay, engine, val, convert=True, **kwargs):
        """
        No description

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
        if isinstance(val, str):
            val = int(val, 16)
        self._caput(
            self.waveform_engine_buffers_root.format(bay) +
            self._wr_addr_reg.format(engine),
            val, **kwargs)

    def get_waveform_wr_addr(self, bay, engine, convert=True, **kwargs):
        """
        No description

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
        val = self._caget(
            self.waveform_engine_buffers_root.format(bay) +
            self._wr_addr_reg.format(engine),
            as_string=True,
            **kwargs)

        if convert:
            return int(val, 16)
        else:
            return val

    _empty_reg = 'Empty[{}]'

    def set_waveform_empty(self, bay, engine, val, **kwargs):
        """
        No description

        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        val : int
            What val to set.
        """
        self._caput(
            self.waveform_engine_buffers_root.format(bay) +
            self._empty_reg.format(engine),
            **kwargs)

    def get_waveform_empty(self, bay, engine, **kwargs):
        """
        No description

        Args
        ----
        bay : int
            Which bay.
        engine : int
            Which waveform engine.
        """
        return self._caget(
            self.waveform_engine_buffers_root.format(bay) +
            self._empty_reg.format(engine),
            **kwargs)

    _data_file_reg = 'DataFile'

    def set_streamdatawriter_datafile(self, datafile_path, **kwargs):
        """
        Sets the output path for the StreamDataWriter. This is what is
        used for take_debug_data.

        Args
        ----
        datafile_path : str
            The full path for the output.
        """
        self._caput(
            self.stream_data_writer_root + self._data_file_reg,
            datafile_path, **kwargs)

    def get_streamdatawriter_datafile(self, as_str=True, **kwargs):
        r"""Gets output path for the StreamDataWriter.

        This is what is used for take_debug_data.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str
            The full path for the output.
        """
        return self._caget(
            self.stream_data_writer_root + self._data_file_reg,
            as_string=True, **kwargs)

    _datawriter_open_reg = 'Open'

    def set_streamdatawriter_open(self, val, **kwargs):
        """
        """
        self._caput(
            self.stream_data_writer_root + self._datawriter_open_reg,
            val, **kwargs)


    def get_streamdatawriter_open(self, **kwargs):
        """
        """
        return self._caget(
            self.stream_data_writer_root + self._datawriter_open_reg,
            **kwargs)

    _datawriter_close_reg = 'Close'

    def set_streamdatawriter_close(self, val, **kwargs):
        """
        """
        self._caput(
            self.stream_data_writer_root + self._datawriter_close_reg,
            val, **kwargs)

    def get_streamdatawriter_close(self, **kwargs):
        """
        """
        return self._caget(
            self.stream_data_writer_root + self._datawriter_close_reg,
            **kwargs)

    _trigger_daq_reg = 'TriggerDaq'

    def set_trigger_daq(self, bay, val, **kwargs):
        """
        """
        self._caput(
            self.daq_mux_root.format(bay) + self._trigger_daq_reg,
            val, **kwargs)

    def get_trigger_daq(self, bay, **kwargs):
        """
        """
        self._caget(
            self.daq_mux_root.format(bay) + self._trigger_daq_reg,
            **kwargs)

    _arm_hw_trigger_reg = "ArmHwTrigger"

    def set_arm_hw_trigger(self, bay, val, **kwargs):
        """
        """
        self._caput(
            self.daq_mux_root.format(bay) + self._arm_hw_trigger_reg,
            val, **kwargs)

    _trigger_hw_arm_reg = 'TriggerHwArm'

    def set_trigger_hw_arm(self, bay, val, **kwargs):
        """
        """
        self._caput(
            self.daq_mux_root.format(bay) + self._trigger_hw_arm_reg,
            val, **kwargs)

    def get_trigger_hw_arm(self, bay, **kwargs):
        """
        """
        return self._caget(
            self.daq_mux_root.format(bay) + self._trigger_hw_arm_reg,
            **kwargs)

    # rtm commands

    #########################################################
    ## start rtm arbitrary waveform

    _rtm_arb_waveform_lut_table_reg = 'Lut[{}].MemArray'

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

        self._caput(
            self.rtm_lut_ctrl_root +
            self._rtm_arb_waveform_lut_table_reg.format(reg),
            lut_arr, **kwargs)

    def get_rtm_arb_waveform_lut_table(self, reg, **kwargs):
        """
        Gets the table currently loaded into the LUT table indexed by
        reg.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        return self._caget(
            self.rtm_lut_ctrl_root +
            self._rtm_arb_waveform_lut_table_reg.format(reg),
            **kwargs)

    _rtm_arb_waveform_busy_reg = 'Busy'

    def get_rtm_arb_waveform_busy(self, **kwargs):
        """
        =1 if waveform if Continuous=1 and the RTM arbitrary waveform
        is being continously generated.  Can be toggled low again by
        setting Continuous=0.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_busy_reg,
            **kwargs)

    _rtm_arb_waveform_trig_cnt_reg = 'TrigCnt'

    def get_rtm_arb_waveform_trig_cnt(self, **kwargs):
        """
        Counts the number of RTM arbitrary waveform software triggers
        since boot up or the last CntRst.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_trig_cnt_reg,
            **kwargs)

    _rtm_arb_waveform_continuous_reg = 'Continuous'

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
        self._caput(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_continuous_reg,
            val, **kwargs)

    def get_rtm_arb_waveform_continuous(self, **kwargs):
        """
        If =1, RTM arbitrary waveform generation is continuous and
        repeats, otherwise if =0, waveform in LUT tables is only
        broadcast once on software trigger.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_continuous_reg,
            **kwargs)

    _rtm_arb_waveform_software_trigger_reg = 'SwTrig'

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
            self._rtm_arb_waveform_software_trigger_reg

        self._caput(triggerPV, 1, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    _dac_axil_addr_reg = 'DacAxilAddr[{}]'

    def set_dac_axil_addr(self, reg, val, **kwargs):
        """
        Sets the DacAxilAddr[#] registers.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        self._caput(
            self.rtm_lut_ctrl + self._dac_axil_addr_reg.format(reg),
            val, **kwargs)

    def get_dac_axil_addr(self, reg, **kwargs):
        """
        Gets the DacAxilAddr[#] registers.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        return self._caget(
            self.rtm_lut_ctrl + self._dac_axil_addr_reg.format(reg),
            **kwargs)

    _rtm_arb_waveform_timer_size_reg = 'TimerSize'

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
        self._caput(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_timer_size_reg,
            val, **kwargs)

    def get_rtm_arb_waveform_timer_size(self, **kwargs):
        """
        Arbitrary waveforms are written to the slow RTM DACs with time
        between samples TimerSize*6.4ns.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_timer_size_reg,
            **kwargs)

    _rtm_arb_waveform_max_addr_reg = 'MaxAddr'

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
        self._caput(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_max_addr_reg,
            val, **kwargs)

    def get_rtm_arb_waveform_max_addr(self, **kwargs):
        """
        Slow RTM DACs will play the sequence [0...MaxAddr] of points
        out of the loaded LUT tables before stopping or repeating on
        software trigger (if in continuous mode).  MaxAddr is an
        11-bit number (must be in [0,2048), because that's the maximum
        length of the LUT tables that store the waveforms.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_max_addr_reg,
            **kwargs)

    _rtm_arb_waveform_enable_reg = 'EnableCh'

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
        self._caput(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_enable_reg,
            val, **kwargs)

    def get_rtm_arb_waveform_enable(self, **kwargs):
        """
        Enable for generation of arbitrary waveforms on the RTM slow
        DACs.

        EnableCh = 0x0 is disable
        0x1 is Addr[0]
        0x2 is Addr[1]
        0x3 is Addr[0] and Addr[1]
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_enable_reg,
            **kwargs)

    ## end rtm arbitrary waveform
    #########################################################

    _reset_rtm_reg = 'resetRtm'

    def reset_rtm(self, **kwargs):
        """
        Resets the rear transition module (RTM)
        """
        self._caput(
            self.rtm_cryo_det_root + self._reset_rtm_reg,
            1, **kwargs)

    _cpld_reset_reg = 'CpldReset'

    def set_cpld_reset(self, val, **kwargs):
        """
        No description

        Args
        ----
        val : int
            Set to 1 for a cpld reset.
        """
        self._caput(
            self.rtm_cryo_det_root + self._cpld_reset_reg,
            val, **kwargs)

    def get_cpld_reset(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_cryo_det_root + self._cpld_reset_reg,
            **kwargs)

    def cpld_toggle(self, **kwargs):
        """
        Toggles the cpld reset bit.
        """
        self.reset_rtm(**kwargs)

    _k_relay_reg = 'KRelay'

    def set_k_relay(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_cryo_det_root + self._k_relay_reg,
            val, **kwargs)

    def get_k_relay(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_cryo_det_root + self._k_relay_reg,
            **kwargs)

    _timing_crate_root_reg = "AMCc.FpgaTopLevel.AmcCarrierCore.AmcCarrierTiming.EvrV2CoreTriggers"
    _trigger_rate_sel_reg = ".EvrV2ChannelReg[0].RateSel"

    def set_ramp_rate(self, val, **kwargs):
        """
        flux ramp sawtooth reset rate in kHz

        If using timing system, the allowed rates are: 1, 2, 3, 4, 5,
        6, 8, 10, 12, 15kHz (hardcoded by timing)
        """
        rate_sel = self.flux_ramp_rate_to_PV(val)

        if rate_sel is not None:
            self._caput(
                self._timing_crate_root_reg +
                self._trigger_rate_sel_reg,
                rate_sel, **kwargs)
        else:
            print(
                "Rate requested is not allowed by timing" +
                "triggers. Allowed rates are 1, 2, 3, 4, 5, 6, 8, 10," +
                "12, 15kHz only")

    def get_ramp_rate(self, **kwargs):
        """
        flux ramp sawtooth reset rate in kHz
        """

        rate_sel = self._caget(
            self._timing_crate_root_reg +
            self._trigger_rate_sel_reg,
            **kwargs)

        reset_rate = self.flux_ramp_PV_to_rate(rate_sel)

        return reset_rate

    _trigger_delay_reg = "EvrV2TriggerReg[0].Delay"

    def set_trigger_delay(self, val, **kwargs):
        """
        Adds an offset to flux ramp trigger.  Only really useful if
        you're using two carriers at once and you're trying to
        synchronize them.  Mitch thinks it's in units of 122.88MHz
        ticks.
        """
        self._caput(
            self._timing_crate_root_reg +
            self._trigger_delay_reg,
            val, **kwargs)

    def get_trigger_delay(self, **kwargs):
        """
        The flux ramp trigger offset.  Only really useful if you're
        using two carriers at once and you're trying to synchronize
        them.  Mitch thinks it's in units of 122.88MHz ticks.
        """

        trigger_delay = self._caget(
            self._timing_crate_root_reg +
            self._trigger_delay_reg,
            **kwargs)

        return trigger_delay

    _debounce_width_reg = 'DebounceWidth'

    def set_debounce_width(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_cryo_det_root + self._debounce_width_reg,
            val, **kwargs)

    def get_debounce_width(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_cryo_det_root + self._debounce_width_reg,
            **kwargs)

    _ramp_slope_reg = 'RampSlope'

    def set_ramp_slope(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_spi_root + self._ramp_slope_reg,
            val, **kwargs)

    def get_ramp_slope(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_spi_root + self._ramp_slope_reg,
            **kwargs)

    _flux_ramp_dac_reg = 'LTC1668RawDacData'

    def set_flux_ramp_dac(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_spi_root + self._flux_ramp_dac_reg,
            val, **kwargs)

    def get_flux_ramp_dac(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_spi_root + self._flux_ramp_dac_reg,
            **kwargs)

    _mode_control_reg = 'ModeControl'

    def set_mode_control(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_spi_root + self._mode_control_reg,
            val, **kwargs)

    def get_mode_control(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_spi_root + self._mode_control_reg,
            **kwargs)

    _fast_slow_step_size_reg = 'FastSlowStepSize'

    def set_fast_slow_step_size(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_spi_root + self._fast_slow_step_size_reg,
            val, **kwargs)

    def get_fast_slow_step_size(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_spi_root + self._fast_slow_step_size_reg,
            **kwargs)

    _fast_slow_rst_value_reg = 'FastSlowRstValue'

    def set_fast_slow_rst_value(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_spi_root + self._fast_slow_rst_value_reg,
            val, **kwargs)

    def get_fast_slow_rst_value(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_spi_root + self._fast_slow_rst_value_reg,
            **kwargs)

    _enable_ramp_trigger_reg = 'EnableRampTrigger'

    def set_enable_ramp_trigger(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_cryo_det_root + self._enable_ramp_trigger_reg,
            val, **kwargs)

    def get_enable_ramp_trigger(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_cryo_det_root + self._enable_ramp_trigger_reg,
            **kwargs)

    _cfg_reg_ena_bit_reg = 'CfgRegEnaBit'

    def set_cfg_reg_ena_bit(self, val, **kwargs):
        """
        """
        self._caput(
            self.rtm_spi_root + self._cfg_reg_ena_bit_reg,
            val, **kwargs)

    def get_cfg_reg_ena_bit(self, **kwargs):
        """
        """
        return self._caget(
            self.rtm_spi_root + self._cfg_reg_ena_bit_reg,
            **kwargs)

    # Right now in pyrogue, this is named as if it's always a TesBias,
    # but pysmurf doesn't only use them as TES biases - e.g. in
    # systems using a 50K follow-on amplifier, one of these DACs is
    # used to drive the amplifier gate.
    _rtm_slow_dac_enable_reg = 'TesBiasDacCtrlRegCh[{}]'

    def set_rtm_slow_dac_enable(self, dac, val=2, **kwargs):
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
            Value to set the DAC enable to. enabled is 0x2, disabled is 0xE.
            Power on default is 0xE.
        """
        assert (dac in range(1,33)),'dac must be an integer and in [1,32]'

        # only ever set this to 0x2 or 0xE (enable or disable)
        if (val != 0x2) and (val != 0xE):
            self.log("RTM dac val must be 0x2 or 0xE. Setting to 0x2 (enabled).")
            val = 0x2

        self._caput(self.rtm_spi_max_root +
            self._rtm_slow_dac_enable_reg.format(dac), val, **kwargs)

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

        return self._caget(
            self.rtm_spi_max_root +
            self._rtm_slow_dac_enable_reg.format(dac),
            **kwargs)

    _rtm_slow_dac_enable_array_reg = 'TesBiasDacCtrlRegChArray'

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
            provided array is not length 32, asserts. Power on
            default is 0xE, enabled is 0x2, disable is 0xE.
        """
        assert (len(val)==32),(
            'len(val) must be 32, the number of DACs in hardware.')

        # only ever set this to 0x2 or 0xE
        if np.any(np.logical_and(val != 0x2 , val != 0xE)):
            self.log("All values in val must be 0x2 or 0xE. " +
                "Setting incorrect values to 0x2 (enable)." )

        val = [0x2 if v != 0x2 and v != 0xE else v for v in val]

        self._caput(
            self.rtm_spi_max_root +
            self._rtm_slow_dac_enable_array_reg,
            val, **kwargs)

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
        return self._caget(
            self.rtm_spi_max_root +
            self._rtm_slow_dac_enable_array_reg,
            **kwargs)

    _rtm_slow_dac_data_reg = 'TesBiasDacDataRegCh[{}]'

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
        self._caput(
            self.rtm_spi_max_root +
            self._rtm_slow_dac_data_reg.format(dac),
            val, **kwargs)

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
        return self._caget(
            self.rtm_spi_max_root +
            self._rtm_slow_dac_data_reg.format(dac),
            **kwargs)

    _rtm_slow_dac_data_array_reg = 'TesBiasDacDataRegChArray'

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

        self._caput(
            self.rtm_spi_max_root + self._rtm_slow_dac_data_array_reg,
            val, **kwargs)

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
        return self._caget(
            self.rtm_spi_max_root + self._rtm_slow_dac_data_array_reg,
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
        self.set_rtm_slow_dac_data(
            dac, int(val/self._rtm_slow_dac_bit_to_volt), **kwargs)


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

        volt = self._rtm_slow_dac_bit_to_volt * self.get_rtm_slow_dac_data(dac, **kwargs)

        if volt > 9.9:
            self.log(f'Looks like DAC {dac} is close to max +10V output, {volt}V.')

        if volt < -9.9:
            self.log(f'Looks like DAC {dac} is close to min -10V output, {volt}V.')

        return volt

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
        assert (len(val)==32),(
            'len(val) must be 32, the number of DACs in hardware.')
        int_val = (
            np.array(np.array(val) /
                     self._rtm_slow_dac_bit_to_volt, dtype=int))
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
        return (self._rtm_slow_dac_bit_to_volt *
                self.get_rtm_slow_dac_data_array(**kwargs))

    # There are actually 33 RTM DACs but the 33rd is hacked in.  It's
    # the HEMT Gate on the C02, and HEMT1 Gate on the C04. Eventually
    # please remove this register and add index 33 to
    # TesBiasDacCtrlRegCh.
    _rtm_33_ctrl_reg = 'HemtBiasDacCtrlRegCh[33]'
    _rtm_33_data_reg = 'HemtBiasDacDataRegCh[33]'

    def get_amp_gate_voltage(self, amp):
        self.C.assert_amps_match_this_cryocard(list(amp))

        if amp == 'hemt' or amp =='hemt1':
            if amp == 'hemt':
                bit_to_volt = self.config.config['amplifier']['bit_to_V_hemt']
            else:
                bit_to_volt = self.config.config['amplifier']['hemt1']['gate_bit_to_volt']

            bits = self._caget(self.rtm_spi_max_root + self._rtm_33_data_reg)

        elif amp == '50k':
            dac_num = self.config.config['amplifier']['dac_num_50k']
            bit_to_volt = self.config.config['amplifier']['bit_to_V_50k']
            bits = self.get_rtm_slow_dac_data(dac_num)

        else:
            dac_num = self.config.config['amplifier'][amp]['gate_dac_num']
            bit_to_volt = self.config.config['amplifier'][amp]['gate_bit_to_volt']
            bits = self.get_rtm_slow_dac_data(dac_num)

        volts = bit_to_volt * bits
        return volts

    def set_amp_gate_voltage(self, amp, voltage, override = False, **kwargs):
        """
        Set the voltage out one of the RTM DACs, into the cryocard,
        such that the voltage out the cryocard is the given voltage.
        To do this, use the conversion factors "gate_bit_to_volt". The
        gate conversion is computed analytically.

        The DACs do not respond unless their DAC enable register is
        0x2, enabled.  The DACs still output voltage even if their
        enable register is 0xe, disabled.

        Params
        ------
        amp: str
            Use '50k' and 'hemt' for the C02 amps, and '50k1',
            '50k2', 'hemt1' and 'hemt2' for the C04 amps.

        voltage: float
            The desired voltage going out the cryocard amp.
        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        if amp == 'hemt' or amp == 'hemt1':
            if amp == 'hemt':
                bit_to_volt = self.config.config['amplifier']['bit_to_V_hemt']
            else:
                bit_to_volt = self.config.config['amplifier']['hemt1']['gate_bit_to_volt']

            bits = voltage / bit_to_volt
            nbits = self._rtm_slow_dac_nbits

            if bits > 2**(nbits-1)-1:
                self.log(f'{amp} voltage overflowed high, setting to max.')
                bits = 2**(nbits-1)-1

            elif bits < -2**(nbits-1):
                self.log(f'{amp} voltage overflowed low, setting to min.')
                bits = -2**(nbits-1)

            self.log(f'Setting hemt or hemt1 gate to {bits} bits given {voltage} volts.')
            self._caput(self.rtm_spi_max_root + self._rtm_33_data_reg, bits, **kwargs)

        elif amp == '50k':
            dac_num = self.config.config['amplifier']['dac_num_50k']
            bit_to_volt = self.config.config['amplifier']['bit_to_V_50k']
            bits = voltage / bit_to_volt
            self.set_rtm_slow_dac_data(dac_num, bits, **kwargs)

        else:
            min = self.config.get('amplifier')[amp]['gate_volt_min']
            max = self.config.get('amplifier')[amp]['gate_volt_max']

            if not override:
                assert voltage >= min and voltage <= max, f'Voltage {voltage} for amp {amp} out of bounds, {min}, {max}'

            dac_num = self.config.get('amplifier')[amp]['gate_dac_num']
            bit_to_volt = self.config.get('amplifier')[amp]['gate_bit_to_volt']
            bits = voltage / bit_to_volt
            self.log(f'Setting {amp} gate to {bits} via DAC {dac_num}, given {voltage} Volts')
            self.set_rtm_slow_dac_data(dac_num, bits, **kwargs)

    def get_amp_drain_voltage(self, amp):
        """
        C04 only.

        Args
        ----
        amp: str
          Choose '50k' or 'hemt' for the C00, C01, C02, and '50k1', '50k2', 'hemt1', 'hemt2' for the C04.
        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        if not self.get_amp_drain_enable(amp):
            self.log(f'get_amp_drain_voltage: The power supply for amp {amp} is off, therefore returning 0.0.')
            return 0.0

        dac_num = self.config.get('amplifier')[amp]['drain_dac_num']
        m = self.config.get('amplifier')[amp]['drain_conversion_m']
        b = self.config.get('amplifier')[amp]['drain_conversion_b']
        dac_volt = self.get_rtm_slow_dac_volt(dac_num)
        out_volt = m * dac_volt + b

        return out_volt

    def get_amp_drain_enable(self, amp):
        """Don't use directly, use get_amp_drain_voltage. Get if power supply
        are enabled for this cryocard amplifier. The drain will output
        voltage even if the power register is zero (ESCRYODET-851).

        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        power_bitmask = self.config.get('amplifier')[amp]['power_bitmask']
        power = self.C.read_ps_en()
        power_masked = power & power_bitmask

        return power_masked > 0

    def set_amp_drain_enable(self, amp, enable):
        """
        Enable the drain power supply. If C04, set_amp_drain_voltage sets
        this automatically. It is very important that this is on only
        when you want voltage going out the drain. See bug
        ESCRYODET-851. There are two drain outputs on the C02, and
        four drain outputs on the C04.
        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        power_bitmask = self.config.get('amplifier')[amp]['power_bitmask']
        power = self.C.read_ps_en()

        power_masked = power & ~power_bitmask

        if enable:
            power_masked = power | power_bitmask

        self.C.write_ps_en(power_masked)

    def set_amp_drain_voltage(self, amp, volt, override = False):
        """
        C04 only. Set the drain voltage going out of the given amplifier
        circuit, either 50k1, 50k2, hemt1, hemt2. The HEMT drain
        voltage range is different from the 50K voltage range, set the
        range limits in the SMuRF .cfg file. If given 0 volts, turn
        turn off the power supply entirely to avoid bug
        ESCRYODET-851. We cannot set the drain voltage directly,
        instead we can only set the RTM DAC voltage directly which
        approximately sets the drain voltage..
        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        dac_num = self.config.config['amplifier'][amp]['drain_dac_num']

        if volt == 0 or volt == 0.0:
            if self.get_amp_drain_enable(amp):
                self.log(f'set_amp_drain_voltage: {amp}: zero requested ; setting control DAC{dac_num} to 10V, disabling LDO, and then setting control DAC{dac_num} to 0V.  In the brief time for which the control DAC is set to 10V and the LDO is disabled, the cryocard will put out a small, load dependent voltage.')
                self.set_rtm_slow_dac_volt(dac_num, 9.999)
                self.set_amp_drain_enable(amp, False)
                self.set_rtm_slow_dac_volt(dac_num, 0.0)
            else:
                # just make sure control DAC is set to zero
                self.log(f'set_amp_drain_voltage: {amp}: drain already disabled.')
                # check if control DAC is nonzero - it shouldn't be if DAC is already disabled.
                if self.get_rtm_slow_dac_volt(dac_num)!=0.0:
                    self.log(f'set_amp_drain_voltage: {amp}: drain is disabled but control DAC{dac_num} is nonzero.  Setting to zero.')
                    self.set_rtm_slow_dac_volt(dac_num, 0.0)
        else:
            min = self.config.get('amplifier')[amp]['drain_volt_min']
            max = self.config.get('amplifier')[amp]['drain_volt_max']

            if not override:
                assert volt >= min and volt <= max, f'Voltage {volt} for amp {amp} out of bounds, {min}, {max}'

            # Set the voltage out the RTM Drain DACs to 10 V, which
            # implies that the voltage going out the cryocard drains
            # is minimized. See
            # https://confluence.slac.stanford.edu/display/SMuRF/Cryostat+board

            if not self.get_rtm_slow_dac_enable(dac_num) == 0x2:
                self.log(f'set_amp_drain_voltage: {amp}: DAC {dac_num} is not enabled, enabling it (0x2).')
#                self.set_rtm_slow_dac_enable(dac_num, 0x2)


            #self.log('Continuing.')

            if not self.get_amp_drain_enable(amp):
                self.log(f'set_amp_drain_voltage: {amp}: this drain is currently disabled ; setting control DAC{dac_num} to 10V, enabling LDO, and then setting control DAC{dac_num} to voltage required to achieve desired drain voltage.  In the brief time for which the control DAC is set to 10V and the LDO is disabled, the cryocard will put out a small, load dependent voltage.')
                self.set_rtm_slow_dac_volt(dac_num, 9.999)
                self.set_amp_drain_enable(amp, True)

            m = self.config.get('amplifier')[amp]['drain_conversion_m']
            b = self.config.get('amplifier')[amp]['drain_conversion_b']
            dac_volt = (volt - b)/m
            self.set_rtm_slow_dac_volt(dac_num, dac_volt)

    def get_amp_drain_current(self, amp):
        """Guess the current going out of the given amp by measuring the
        another voltage on the cryocard, for example 50K2_I. This
        function has high jitter, 1 mA at least. There is no
        set_amp_drain_current.

        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        address = self.config.get('amplifier')[amp]['drain_pic_address']
        drain_opamp_gain = self.config.get('amplifier')[amp]['drain_opamp_gain']

        if amp == 'hemt':
            drain_resistor = self.config.config['amplifier']['hemt_Vd_series_resistor']
            drain_offset = self.config.config['amplifier']['hemt_Id_offset']

        elif amp == '50k':
            drain_resistor = self.config.config['amplifier']['50K_amp_Vd_series_resistor']
            drain_offset = self.config.config['amplifier']['50k_Id_offset']

        else:
            drain_resistor = self.config.get('amplifier')[amp]['drain_resistor']
            drain_offset = self.config.get('amplifier')[amp]['drain_offset']

        volt = self.C.get_volt(address)
        amp = 2 * (volt / drain_opamp_gain) / drain_resistor
        out_milliamp = 1000 * amp
        out_milliamp_offset = out_milliamp - drain_offset

        return out_milliamp_offset

    def get_amp_drain_current_dict(self):
        """
        Return dictionary of all drain currents. This will return two drain
        currents if working on the C02, and four drain currents if working on
        the C04.
        """
        amp_gate_currents = dict()
        major, minor, patch = self.C.get_fw_version()

        if major == 4:
            for amp in self.C.list_of_c04_amps:
                current = self.get_amp_drain_current(amp)
                amp_gate_currents[amp] = current

        elif major == 1 or major == 10:
            for amp in self.C.list_of_c02_amps:
                current = self.get_amp_drain_current(amp)
                amp_gate_currents[amp] = current

        return amp_gate_currents

    def set_amp_defaults(self):
        """The pysmurf cfg file specifies the default power state, default
        gate voltage for the C02 amplifiers HEMT and 50K. Additionally
        it specifies the default drain voltage for the C04 hemt1,
        50k1, hemt2, 50k2. Read these values, then set them according
        to if the card is connected or not. See smurf_config.py. This
        assumes the gate and drain DAC are enabled, and the drain
        power supplies are disabled. Always set the gate voltage, and
        always zero the drain DAC voltage, which implies the drain
        voltage going out the cryocard is zero, and the drain current
        is zero. Also, send enable, 0x2, to all gate voltages and
        drain voltages, otherwise they cannot be controlled, even if
        the current register reports that they are supposedly enabled,
        0x2.

        """
        major, minor, patch = self.C.get_fw_version()

        # Enable the HEMT gate DAC on the C02, or HEMT1 gate DAC on the C04.
        self._caput(self.rtm_spi_max_root + self._rtm_33_ctrl_reg, 0x2)

        if major == 1 or major == 10:
            volt = self.config.get('amplifier')['LNA_Vg']
            self.set_amp_gate_voltage('50k', volt)

            volt = self.config.get('amplifier')['hemt_Vg']
            self.set_amp_gate_voltage('hemt', volt)

        if major == 4:
            for amp in self.C.list_of_c04_amps:

                # Set the gates to their defaults.

                gate_volt_default = self.config.config['amplifier'][amp]['gate_volt_default']
                self.set_amp_gate_voltage(amp, gate_volt_default)

    def get_amplifier_biases(self):
        """For the C00, C01 and C02, return dictionary of all gate voltages,
        drain currents, and drain power supply states. For the C04,
        also return all drain voltages.
        """

        amp_dict = dict()
        major, minor, patch = self.C.get_fw_version()

        if major == 1 or major == 10:
            for amp in self.C.list_of_c02_amps:
                voltage = self.get_amp_gate_voltage(amp)
                amp_dict[amp + '_gate_volt'] = voltage

                current = self.get_amp_drain_current(amp)
                amp_dict[amp + '_drain_current'] = current

                enable = self.get_amp_drain_enable(amp)
                amp_dict[amp + '_enable'] = enable

        if major == 4:
            for amp in self.C.list_of_c04_amps:
                voltage = self.get_amp_gate_voltage(amp)
                amp_dict[amp + '_gate_volt'] = voltage

                voltage = self.get_amp_drain_voltage(amp)
                amp_dict[amp + '_drain_volt'] = voltage

                current = self.get_amp_drain_current(amp)
                amp_dict[amp + '_drain_current'] = current

                enable = self.get_amp_drain_enable(amp)
                amp_dict[amp + '_enable'] = enable

        return amp_dict

    def set_hemt_enable(self, disable=False):
        enable = not disable
        self.log(f'set_hemt_enable: Deprecated. Calling set_amp_drain_enable("hemt", {enable}')
        self.set_amp_drain_enable('hemt', enable)

    def set_50k_amp_enable(self, disable=False):
        enable = not disable
        self.log(f'set_50k_enable: Deprecated. Calling set_amp_drain_enable("50k", {enable}')
        self.set_amp_drain_enable('50k', enable)

    def get_50k_amp_gate_voltage(self):
        self.log('get_50k_gate_voltage: Deprecated. Calling get_amp_gate_voltage("50k")')
        self.get_amp_get_voltage('50k')

    def set_50k_amp_gate_voltage(self, voltage, override=False):
        self.log(f'set_50k_gate_voltage: Deprecated. Calling set_amp_gate_voltage("50k", {voltage}, override={override})')
        self.set_amp_gate_voltage('50k', voltage, override)

    def set_hemt_gate_voltage(self, voltage, override=False):
        self.log(f'set_hemt_gate_voltage: Deprecated. Calling set_amp_gate_voltage("hemt", {voltage}, override={override})')
        self.set_amp_gate_voltage('hemt', voltage, override)

    def set_hemt_bias(self, voltage, override=False):
        self.log(f'set_hemt_bias: Deprecated. Calling set_amp_gate_voltage("hemt", {voltage}, override={override})')
        self.get_amp_get_voltage('hemt', voltage, override)

    def get_hemt_bias(self):
        self.log('get_hemt_bias: Deprecated. Calling get_amp_gate_voltage("hemt")')
        return self.get_amp_gate_voltage('hemt')

    def set_amplifier_bias(self, bias_hemt = None, bias_50k = None, **kwargs):
        self.log('set_amplifier_bias: Deprecated. Calling set_amp_gate_voltage')
        if bias_hemt is not None:
            self.set_amp_gate_voltage('hemt', bias_hemt, **kwargs)

        if bias_50k is not None:
            self.set_amp_gate_voltage('50k', bias_50k, **kwargs)

    def get_amplifier_bias(self):
        self.log('get_amplifier_bias: Deprecated. Calling get_amplifier_biases')
        return self.get_amplifier_biases()

    def get_hemt_drain_current(self):
        self.log('get_hemt_drain_current: Deprecated. Calling get_amp_drain_current("hemt")')
        return self.get_amp_drain_current("hemt")

    def get_50k_amp_drain_current(self):
        self.log('get_50k_amp_drain_current: Deprecated. Calling get_amp_drain_current("50k")')
        return self.get_amp_drain_current("50k")

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

    _ramp_max_cnt_reg = 'RampMaxCnt'

    def set_ramp_max_cnt(self, val, **kwargs):
        """
        Internal Ramp's maximum count. Sets the trigger repetition rate. This
        is effectively the flux ramp frequency.

        RampMaxCnt = 307199 means flux ramp is 1kHz (307.2e6/(RampMaxCnt+1))
        """
        self._caput(
            self.rtm_cryo_det_root + self._ramp_max_cnt_reg,
            val, **kwargs)

    def get_ramp_max_cnt(self, **kwargs):
        """
        Internal Ramp's maximum count. Sets the trigger repetition rate. This
        is effectively the flux ramp frequency.

        RampMaxCnt = 307199 means flux ramp is 1kHz (307.2e6/(RampMaxCnt+1))
        """
        return self._caget(
            self.rtm_cryo_det_root + self._ramp_max_cnt_reg,
            **kwargs)

    def set_flux_ramp_freq(self, val, **kwargs):
        r"""Sets flux ramp reset rate in kHz.

        Sets the flux ramp reset rate.  In units of kHz.  Wrapper
        function for :func:`set_ramp_max_cnt`.

        The flux ramp reset rate is specified by setting the trigger
        repetition rate (the `RampMaxCnt` register in
        `RtmCryoDet`).  `RampMaxCnt` is a 32-bit counter where each
        count represents a 307.2 MHz tick.

        Args
        ----
        val : float
            The frequency to set the flux ramp reset rate to in kHz.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        Note
        ----
           Because `RampMaxCnt` is specified in 307.2 MHz ticks, the
           flux ramp rate must be an integer divisor of 307.2 MHz.
           For example, for this reason it is not possible to set the
           flux ramp rate to exactly 7 kHz.
           :func:`set_flux_ramp_freq` rounds the `RampMaxCnt` computed
           for the desired flux ramp reset rate to the nearest integer
           using the built-in `round` routine, and if the
           computed `RampMaxCnt` differs from the rounded value,
           reports the flux ramp reset rate that will actually be
           programmed.

        .. warning::
           If `RampMaxCnt` is set too low, then it will invert and
           produce a train of pulses 1x or 2x 307.2 MHz ticks wide,
           but it will be mostly high.

        See Also
        --------
        get_flux_ramp_freq : Gets the flux ramp reset rate.
        set_ramp_max_cnt : Sets the flux ramp trigger repetition rate.
        """
        # the digitizer frequency is 2x the default fw rate because
        # the digitizer clocks an I and a Q sample both at 307.2MHz,
        # or data at 614.4MHz.  For some reason I don't understand,
        # there's too much precision error to get the right value out
        # of rogue, unless I poll the digitizer frequency as a string
        # and then cast to float in python.
        ramp_max_cnt_clock_hz = 1.e6*(
            float(
                self.get_digitizer_frequency_mhz(as_string=True)
            ) / 2.
        )
        ramp_max_cnt_rate_khz = ramp_max_cnt_clock_hz/1.e3
        cnt_estimate = ramp_max_cnt_rate_khz/float(val)-1.
        cnt_rounded  = round(cnt_estimate)
        if cnt_estimate != cnt_rounded:
            val_rounded = ramp_max_cnt_rate_khz/(cnt_rounded+1.)
            self.log(
                f'WARNING : Requested flux ramp reset rate of {val} '
                'kHz does not integer divide '
                f'{ramp_max_cnt_rate_khz/1e3} MHz.  Setting flux ramp'
                f' reset rate to {val_rounded} kHz instead.',
                self.LOG_ERROR)
        self.set_ramp_max_cnt(cnt_rounded, **kwargs)

    def get_flux_ramp_freq(self, **kwargs):
        r"""Returns flux ramp reset rate in kHz.

        Returns the current flux ramp reset rate.  In units of kHz.

        The flux ramp reset rate is determined by polling the trigger
        repetition rate (the `RampMaxCnt` register in `RtmCryoDet`).
        `RampMaxCnt` is a 32-bit counter where each count represents a
        307.2 MHz tick.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float
            Currently programmed flux ramp reset rate, in kHz.

        See Also
        --------
        set_flux_ramp_freq : Sets the flux ramp reset rate.
        get_ramp_max_cnt : Gets the flux ramp trigger repetition rate.
        """
        if self.offline: # FIX ME - this is a stupid hard code
            return 4.0
        else:
            # the digitizer frequency is 2x the default fw rate
            # because the digitizer clocks an I and a Q sample both at
            # 307.2MHz, or data at 614.4MHz.  For some reason I don't
            # understand, there's too much precision error to get the
            # right value out of rogue, unless I poll the digitizer
            # frequency as a string and then cast to float in python.
            ramp_max_cnt_clock_hz = 1.e6*(
                float(
                    self.get_digitizer_frequency_mhz(as_string=True)
                ) / 2.
            )
            ramp_max_cnt_rate_khz = ramp_max_cnt_clock_hz/1.e3
            return ramp_max_cnt_rate_khz/(
                self.get_ramp_max_cnt(**kwargs)+1)

    _low_cycle_reg = 'LowCycle'

    def set_low_cycle(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive).  Along with
        HighCycle, sets the frequency of the clock going to the RTM.
        """
        self._caput(
            self.rtm_cryo_det_root + self._low_cycle_reg,
            val, **kwargs)

    def get_low_cycle(self, val, **kwargs):
        """
        CPLD's clock: low cycle duration (zero inclusive).  Along with
        HighCycle, sets the frequency of the clock going to the RTM.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._low_cycle_reg,
            **kwargs)

    _high_cycle_reg = 'HighCycle'

    def set_high_cycle(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).  Along
        with LowCycle, sets the frequency of the clock going to the
        RTM.
        """
        self._caput(
            self.rtm_cryo_det_root + self._high_cycle_reg,
            val, **kwargs)

    def get_high_cycle(self, val, **kwargs):
        """
        CPLD's clock: high cycle duration (zero inclusive).  Along
        with LowCycle, sets the frequency of the clock going to the
        RTM.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._high_cycle_reg,
            **kwargs)

    _select_ramp_reg = 'SelectRamp'

    def set_select_ramp(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(
            self.rtm_cryo_det_root + self._select_ramp_reg,
            val, **kwargs)

    def get_select_ramp(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(
            self.rtm_cryo_det_root + self._select_ramp_reg,
            **kwargs)

    _ramp_start_mode_reg = 'RampStartMode'

    def set_ramp_start_mode(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x2 = trigger from external system
        0x1 = trigger from timing system
        0x0 = trigger from internal system
        """
        self._caput(
            self.rtm_cryo_det_root + self._ramp_start_mode_reg,
            val, **kwargs)

    def get_ramp_start_mode(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x2 = trigger from external system
        0x1 = trigger from timing system
        0x0 = trigger from internal system
        """
        return self._caget(
            self.rtm_cryo_det_root + self._ramp_start_mode_reg,
            **kwargs)

    _pulse_width_reg = 'PulseWidth'

    def set_pulse_width(self, val, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        self._caput(
            self.rtm_cryo_det_root + self._pulse_width_reg,
            val, **kwargs)

    def get_pulse_width(self, **kwargs):
        """
        Select Ramp to the CPLD
        0x1 = Fast flux Ramp
        0x0 = Slow flux ramp
        """
        return self._caget(
            self.rtm_cryo_det_root + self._pulse_width_reg,
            **kwargs)


    _stream_datafile_reg = 'dataFile'

    def set_streaming_datafile(self, datafile, as_string=True,
                               **kwargs):
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
            datafile = np.append(
                datafile, np.zeros(300-len(datafile), dtype=int))
        self._caput(
            self.streaming_root + self._stream_datafile_reg,
            datafile, **kwargs)

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
        datafile = self._caget(
            self.streaming_root + self._stream_datafile_reg,
            **kwargs)
        if as_string:
            datafile = ''.join([chr(x) for x in datafile])
        return datafile

    _streaming_file_open_reg = 'open'

    def set_streaming_file_open(self, val, **kwargs):
        """
        Sets the streaming file open. 1 for streaming on. 0 for
        streaming off.

        Args
        ----
        val : int
            The streaming status.
        """
        self._caput(
            self.streaming_root + self._streaming_file_open_reg,
            val, **kwargs)

    def get_streaming_file_open(self, **kwargs):
        """
        Gets the streaming file status. 1 is streaming, 0 is not.

        Returns
        -------
        val : int
            The streaming status.
        """
        return self._caget(
            self.streaming_root + self._streaming_file_open_reg,
            **kwargs)

    # Carrier slot number
    _slot_number_reg = "SlotNumber"

    def get_slot_number(self, **kwargs):
        """
        Gets the slot number of the crate that the carrier is installed into.

        Returns
        -------
        val : int
            The slot number of the crate that the carrier is installed into.
        """
        return self._caget(
            self.amc_carrier_bsi + self._slot_number_reg,
            **kwargs)

    # Crate id
    _crate_id_reg = "CrateId"

    def get_crate_id(self, **kwargs):
        """
        Gets the crate id.

        Returns
        -------
        val : int
            The crate id.
        """
        return self._caget(
            self.amc_carrier_bsi + self._crate_id_reg,
            **kwargs)

    # UltraScale+ FPGA
    _fpga_temperature_reg = "Temperature"

    def get_fpga_temp(self, **kwargs):
        """
        Gets the temperature of the UltraScale+ FPGA.  Returns float32,
        the temperature in degrees Celsius.

        Returns
        -------
        val : float
            The UltraScale+ FPGA temperature in degrees Celsius.
        """
        return self._caget(
            self.ultrascale +
            self._fpga_temperature_reg,
            **kwargs)

    _fpga_vccint_reg = "VccInt"

    def get_fpga_vccint(self, **kwargs):
        """
        No description

        Returns
        -------
        val : float
            The UltraScale+ FPGA VccInt in Volts.
        """
        return self._caget(
            self.ultrascale +
            self._fpga_vccint_reg,
            **kwargs)

    _fpga_vccaux_reg = "VccAux"

    def get_fpga_vccaux(self, **kwargs):
        """
        No description

        Returns
        -------
        val : float
            The UltraScale+ FPGA VccAux in Volts.
        """
        return self._caget(
            self.ultrascale +
            self._fpga_vccaux_reg,
            **kwargs)

    _fpga_vccbram_reg = "VccBram"

    def get_fpga_vccbram(self, **kwargs):
        """
        No description

        Returns
        -------
        val : float
            The UltraScale+ FPGA VccBram in Volts.
        """
        return self._caget(
            self.ultrascale +
            self._fpga_vccbram_reg,
            **kwargs)

    # Regulator
    _regulator_iout_reg = "IOUT"

    def get_regulator_iout(self, **kwargs):
        """
        No description

        Returns
        -------
        value : float
            Regulator current in amperes.
        """
        return float(
            self._caget(
                self.regulator + self._regulator_iout_reg,
                as_string=True, **kwargs))

    _regulator_temp1_reg = "TEMPERATURE[1]"

    def get_regulator_temp1(self, **kwargs):
        """
        No description

        Returns
        -------
        value : float
            Regulator PT temperature in C.
        """
        return float(
            self._caget(
                self.regulator + self._regulator_temp1_reg,
                as_string=True, **kwargs))

    _regulator_temp2_reg = "TEMPERATURE[2]"

    def get_regulator_temp2(self, **kwargs):
        """
        No description

        Returns
        -------
        value : float
            A regulator CTRL temperature in C.
        """
        return float(
            self._caget(
                self.regulator + self._regulator_temp2_reg,
                as_string=True, **kwargs))

    # Cryo card comands
    def get_cryo_card_temp(self, enable_poll=False, disable_poll=False):
        """
        Get the self-reported temperature of the cryocard. This value is typically
        around 20 Celcius. Anything higher than 30 would indicate a problem. Anything
        below 0 C indicates the board is not connected.

        Returns
        -------
        temp : float
            Temperature of the cryostat card in Celsius.
        """
        if enable_poll:
            self._caput(self._global_poll_enable_reg, True)

        T = self.C.read_temperature()

        if T < 0:
            self.log('get_cryo_card_temp: Temperature is below 0 C, is it connected?')

        if disable_poll:
            self._caput(self._global_poll_enable_reg, False)

        return T

    def get_cryo_card_cycle_count(self, enable_poll=False,
                                  disable_poll=False):
        """
        No description

        Returns
        -------
        cycle_count : float
            The cycle count.
        """
        self.log(
            'Not doing anything because not implemented in '
            'cryo_card.py')
        # return self.C.read_cycle_count()

    def get_cryo_card_relays(self, enable_poll=False,
                             disable_poll=False):
        """
        No description

        Returns
        -------
        relays : hex
            The cryo card relays value.
        """
        if enable_poll:
            self._caput(self._global_poll_enable_reg, True)

        relay = self.C.read_relays()

        if disable_poll:
            self._caput(self._global_poll_enable_reg, False)

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
        assert (bitPosition in range(17)), (
            'bitPosition must be in [0,...,16]')
        assert (oneOrZero in [0,1]), (
            'oneOrZero must be either 0 or 1')
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
            self._caput(self._global_poll_enable_reg, True)

        self.C.write_relays(relay)

        if disable_poll:
            self._caput(self._global_poll_enable_reg, False)

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
            self._caput(self._global_poll_enable_reg, True)

        if write_log:
            self.log('Setting delatch bit using cryo_card ' +
                     f'object. {bit}')
        self.C.delatch_bit(bit)

        if disable_poll:
            self._caput(self._global_poll_enable_reg, False)

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
            return ("DC")
        elif status == 0x3:
            # When both readback are '1' we are in AC mode
            return ("AC")
        else:
            # Anything else is an error
            return ("ERROR")


    _smurf_to_gcp_stream_reg = 'userConfig[0]'  # bit for streaming

    def get_user_config0(self, as_binary=False, **kwargs):
        """
        """
        val =  self._caget(
            self.timing_header + self._smurf_to_gcp_stream_reg,
            **kwargs)

        if as_binary:
            val = bin(val)

        return val


    def set_user_config0(self, val, as_binary=False, **kwargs):
        """
        """
        self._caput(
            self.timing_header + self._smurf_to_gcp_stream_reg,
            val, **kwargs)


    def clear_unwrapping_and_averages(self, **kwargs):
        """
        Resets unwrapping and averaging for all channels, in all bands.
        """

        # Set bit 0 of userConfig[0] high.  Use SyncGroup to detect
        # when register changes so we're sure.
        user_config0_pv=(
            self.timing_header + self._smurf_to_gcp_stream_reg)

        # Toggle using SyncGroup so we can confirm state as we toggle.
        sg=SyncGroup([user_config0_pv], self._client)

        # what is it now?
        sg.wait() # wait for value
        uc0=sg.get_values()[user_config0_pv]

        # set bit high, keeping all other bits the same
        self.set_user_config0(uc0 | (1 << 0))
        sg.wait() # wait for change
        uc0=sg.get_values()[user_config0_pv]
        assert ( ( uc0 >> 0) & 1 ),(
            'Failed to set averaging/clear bit high ' +
            f'(userConfig0={uc0})')

        # toggle bit back to low, keeping all other bits the same
        self.set_user_config0(uc0 & ~(1 << 0))
        sg.wait() # wait for change
        uc0=sg.get_values()[user_config0_pv]
        assert ( ~( uc0 >> 0) & 1 ),(
            'Failed to set averaging/clear bit low after setting ' +
            f'it high (userConfig0={uc0}).')

        self.log('Successfully toggled averaging/clearing bit ' +
                 f'(userConfig[0]={uc0}).',self.LOG_USER)

    # Triggering commands
    _trigger_width_reg = 'EvrV2TriggerReg[{}].Width'

    def set_trigger_width(self, chan, val, **kwargs):
        """
        Mystery value that seems to make the timing system work
        """
        self._caput(
            self.trigger_root + self._trigger_width_reg.format(chan),
            val, **kwargs)

    _trigger_enable_reg = 'EvrV2TriggerReg[{}].EnableTrig'

    def set_trigger_enable(self, chan, val, **kwargs):
        r"""Set trigger pulse generation enable for requested channel.

        The triggering firmware is broken into two parts: (1) the
        event selection logic "Channel", and (2) the trigger pulse
        generation "Trigger".  This enables or disables the "Trigger"
        component for the requested channel.

        Args
        ----
        chan : int
            Which trigger pulse generator channel to enable or
            disable.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_trigger_enable` : Get trigger pulse generation
        enable for requested channel.
        """
        self._caput(
            self.trigger_root + self._trigger_enable_reg.format(chan),
            val, **kwargs)

    def get_trigger_enable(self, chan, **kwargs):
        r"""Get trigger pulse generation enable for requested channel.

        The triggering firmware is broken into two parts: (1) the
        event selection logic "Channel", and (2) the trigger pulse
        generation "Trigger".  This returns whether or not the
        "Trigger" component for the requested channel is enabled or
        disabled.

        Args
        ----
        chan : int
            Return the enable for this trigger pulse generator
            channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            1 if trigger for this channel is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_trigger_enable` : Set trigger pulse generation
        enable for requested channel.
        """
        return self._caget(
            self.trigger_root + self._trigger_enable_reg.format(chan),
            **kwargs)

    _trigger_channel_reg_enable_reg = 'EvrV2ChannelReg[{}].EnableReg'

    def set_evr_channel_reg_enable(self, chan, val, **kwargs):
        r"""Set trigger channel enable.

        The triggering firmware is broken into two parts: (1) the
        event selection logic "Channel", and (2) the trigger pulse
        generation "Trigger".  Trigger pulse generation has several
        required inputs including which "Channel" to listen to.
        Setting this "Enable" register turns on the event selection
        logic for the requested channel.

        Args
        ----
        chan : int
            Which trigger event selection logic channel to enable or
            disable.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_evr_channel_reg_enable` : Get trigger channel enable.
        """
        self._caput(
            self.trigger_root +
            self._trigger_channel_reg_enable_reg.format(chan),
            val, **kwargs)

    def get_evr_channel_reg_enable(self, chan, **kwargs):
        r"""Get trigger channel enable.

        The triggering firmware is broken into two parts: (1) the
        event selection logic "Channel", and (2) the trigger pulse
        generation "Trigger".  Trigger pulse generation has several
        required inputs including which "Channel" to listen to.  This
        "Enable" register controls whether or not the event selection
        logic for the requested channel is on.

        Args
        ----
        chan : int
            Which trigger event selection logic channel to enable or
            disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            1 if trigger channel is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_evr_channel_reg_enable` : Get trigger channel enable.
        """
        return self._caget(
            self.trigger_root +
            self._trigger_channel_reg_enable_reg.format(chan),
            **kwargs)

    _trigger_reg_enable_reg = 'EvrV2TriggerReg[{}].enable'

    def set_evr_trigger_reg_enable(self, chan, val, **kwargs):
        """
        """
        self._caput(
            self.trigger_root +
            self._trigger_reg_enable_reg.format(chan),
            val, **kwargs)

    _trigger_channel_reg_count_reg = 'EvrV2ChannelReg[{}].Count'

    def get_evr_channel_reg_count(self, chan, **kwargs):
        """
        """
        return self._caget(
            self.trigger_root +
            self._trigger_channel_reg_count_reg.format(chan),
            **kwargs)

    _evr_trigger_dest_type_reg = 'EvrV2ChannelReg[{}].DestType'

    def set_evr_trigger_dest_type(self, chan, value, **kwargs):
        r"""Set trigger channel destination type.

        The triggering firmware is broken into two parts: (1) the
        event selection logic "Channel", and (2) the trigger pulse
        generation "Trigger".  Trigger pulse generation has several
        required inputs including which "Channel" to listen to.  The
        channel destination type, or DestType, is an optional logic
        selection (logical AND of) on the presence of a beam where
        this logic is also used for LCLS-2 on the accelerator.  For
        SMuRF we always use destination 0 (="All" if you're looking at
        the SMuRF Rogue gui) which tells the trigger channel logic to
        ignore all beam logic (a better name for "All" would be
        "DontCare".

        Args
        ----
        chan : int
            Which trigger event selection logic channel's destination
            type to set.
        val : int
            Destination type to set.  Although valid options are 0, 1,
            2 or 3, for SMuRF we always use 0 corresponding to "All"
            in the Rogue gui which instructs the channel to ignore any
            selection on the presence of beam.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_evr_trigger_dest_type` : Get trigger channel destination type.
        """
        self._caput(
            self.trigger_root +
            self._evr_trigger_dest_type_reg.format(chan),
            value, **kwargs)

    def get_evr_trigger_dest_type(self, chan, **kwargs):
        r"""Get trigger channel destination type.

        The triggering firmware is broken into two parts: (1) the
        event selection logic "Channel", and (2) the trigger pulse
        generation "Trigger".  Trigger pulse generation has several
        required inputs including which "Channel" to listen to.  The
        channel destination type, or DestType, is an optional logic
        selection (logical AND of) on the presence of a beam where
        this logic is also used for LCLS-2 on the accelerator.  For
        SMuRF we always use destination 0 (="All" if you're looking at
        the SMuRF Rogue gui) which tells the trigger channel logic to
        ignore all beam logic (a better name for "All" would be
        "DontCare".

        Args
        ----
        chan : int
            Which trigger event selection logic channel's destination
            type to get.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Channel destination type of the requested channel.
            Although valid options are 0, 1, 2, and 3, SMuRF should
            always use 0 corresponding to "All" in the Rogue gui which
            instructs the channel to ignore any selection on the
            presence of beam for LCLS-2.

        See Also
        --------
        :func:`set_evr_trigger_dest_type` : Set trigger channel destination type.
        """
        return self._caget(
            self.trigger_root +
            self._evr_trigger_dest_type_reg.format(chan),
            **kwargs)

    _trigger_channel_reg_dest_sel_reg = 'EvrV2ChannelReg[{}].DestSel'

    def set_evr_trigger_channel_reg_dest_sel(self, chan, val, **kwargs):
        """
        """
        self._caput(
            self.trigger_root +
            self._trigger_channel_reg_dest_sel_reg.format(chan),
            val, **kwargs)

    _dbg_enable_reg = "enable"

    def set_dbg_enable(self, bay, val, **kwargs):
        r"""Enables/disables write access to DBG registers.

        Args
        ----
        bay : int
            Which bay [0 or 1].
        val : bool
            True for enable, False for disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput(
            self.DBG.format(bay) + self._dbg_enable_reg,
            val, **kwargs)

    def get_dbg_enable(self, bay, **kwargs):
        r"""Whether or not write access is enabled for DBG registers.

        If disabled (=False), user cannot write to any of the DBG
        registers.

        Args
        ----
        bay : int
            Which bay [0 or 1].
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        bool
            True for enabled, False for disabled.
        """
        return self._caget(
            self.DBG.format(bay) + self._dbg_enable_reg,
            **kwargs)

    _dac_reset_reg = 'dacReset[{}]'

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
        self._caput(
            self.DBG.format(bay) + self._dac_reset_reg.format(dac),
            val, **kwargs)

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
        return self._caget(
            self.DBG.format(bay) + self._dac_reset_reg.format(dac),
            **kwargs)

    _debug_select_reg = "DebugSelect[{}]"

    def set_debug_select(self, bay, val, **kwargs):
        """
        """
        self._caput(
            self.app_core + self._debug_select_reg.format(bay),
            val, **kwargs)

    def get_debug_select(self, bay, **kwargs):
        """
        """
        return self._caget(
            self.app_core + self._debug_select_reg.format(bay),
            **kwargs)

    ### Start Ultrascale OT protection

    _ultrascale_ot_upper_threshold_reg = "OTUpperThreshold"

    def set_ultrascale_ot_upper_threshold(self, val, **kwargs):
        """
        Over-temperature (OT) upper threshold in degC for Ultrascale+
        FPGA.
        """
        self._caput(
            self.ultrascale + self._ultrascale_ot_upper_threshold_reg,
            val, **kwargs)

    def get_ultrascale_ot_upper_threshold(self, **kwargs):
        """
        Over-temperature (OT) upper threshold in degC for Ultrascale+
        FPGA.
        """
        return self._caget(
            self.ultrascale + self._ultrascale_ot_upper_threshold_reg,
            **kwargs)

    ### End Ultrascale OT protection

    _output_config_reg = "OutputConfig[{}]"

    def set_crossbar_output_config(self, index, val, **kwargs):
        """
        """
        self._caput(
            self.crossbar + self._output_config_reg.format(index),
            val, **kwargs)

    def get_crossbar_output_config(self, index, **kwargs):
        """
        This is the timing crossbar, which determins how the SMuRF carrier
        receives its timing signal, and how the backplane receives its
        timing signal.
        """
        return self._caget(
            self.crossbar + self._output_config_reg.format(index),
            **kwargs)

    _timing_link_up_reg = "RxLinkUp"

    def get_timing_link_up(self, **kwargs):
        r"""Return external timing link status.

        Return the value of RxLinkUp. This tells you if the FPGA
        recovered clock is receiving timing from somewhere, either the
        backplane or fiber. This doesn't directly tell you anything
        about the AMCs, JESDs, or LMKs.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            1 if link is up, 0 if link is down.
        """
        return self._caget(
            self.timing_status + self._timing_link_up_reg,
            **kwargs)

    _timing_crc_err_cnt = "CrcErrCount"

    def get_timing_crc_err_cnt(self, **kwargs):
        r"""Gets CRC error counter for received timing frames.

        This counter increments every time a cyclical redundancy check
        (=CRC) fails for a received timing packet.  The content of
        timing frames has a CRC on it which is a running sum for each
        packet.  Data transmitted on the timing link includes a lot of
        idle characters - unlike the counters returned by
        :func:`get_timing_rx_dec_err_cnt` and
        :func:`get_timing_rx_dsp_err_cnt` which increment for any
        errors detected for any received timing data, the CRC is only
        performed on actual timing frames.

        Common causes of timing system error counter increments
        include bad network connections between the external timing
        system and the SMuRF system) and providing the wrong frequency
        or amplitude 122.88 MHz clock reference signal to the timing
        system.

        Timing data is transmitted and received at a total data rate
        of 2.45 Gbps (requiring 10G SFPs and compatible fiber links
        between the external timing system and the SMuRF system(s)),
        and timing frames are transmitted and received at 480 kHz.
        The protocol used for communcation between the external timing
        and SMuRF system(s) is a serial 8B/10B encoding using the
        K-character symbols for byte and frame alignment.  The
        encoding/decoding and byte alignment is supporte by common
        Xilinx IP.

        .. warning::
           An increment in any of the timing system error counters
           (obtainable through :func:`get_timing_crc_err_cnt`,
           :func:`get_timing_rx_dec_err_cnt`, and
           :func:`get_timing_rx_dsp_err_cnt`) will cause a SMuRF
           timing firmware reset, resulting in a ~msec dropout of
           received timing data, including external triggers.  The
           :func:`get_timing_rx_rst_cnt` returns the value of the
           counter that increments everytime there is a reset.  In
           streamed data triggering on external timing, this will look
           like jumps in time without corresponding dropped frames.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            32-bit counter which increments every time the cyclical
            redundancy check fails for a received timing packet.

        See Also
        --------
        :func:`get_timing_rx_dec_err_cnt` : Gets decode error
            counter for received timing characters.

        :func:`get_timing_rx_dsp_err_cnt` : Gets disparity error counter
            for received timing characters.

        :func:`get_timing_rx_rst_cnt` : Gets timing data link reset
            counter.
        """
        return self._caget(
            self.timing_status + self._timing_crc_err_cnt,
            **kwargs)

    _timing_rx_dec_err_cnt = "RxDecErrCount"

    def get_timing_rx_dec_err_cnt(self, **kwargs):
        r"""Gets decode error counter for received timing characters.

        This counter increments every time the SMuRF carrier firmware
        tries to decode a 10-bit timing word but fails, implying the
        data must have gotten corrupted after transmission by the
        timing system.

        Common causes of timing system error counter increments
        include bad network connections between the external timing
        system and the SMuRF system) and providing the wrong frequency
        or amplitude 122.88 MHz clock reference signal to the timing
        system.

        Timing data is transmitted and received at a total data rate
        of 2.45 Gbps (requiring 10G SFPs and compatible fiber links
        between the external timing system and the SMuRF system(s)),
        and timing frames are transmitted and received at 480 kHz.
        The protocol used for communcation between the external timing
        and SMuRF system(s) is a serial 8B/10B encoding using the
        K-character symbols for byte and frame alignment.  The
        encoding/decoding and byte alignment is supporte by common
        Xilinx IP.

        .. warning::
           An increment in any of the timing system error counters
           (obtainable through :func:`get_timing_crc_err_cnt`,
           :func:`get_timing_rx_dec_err_cnt`, and
           :func:`get_timing_rx_dsp_err_cnt`) will cause a SMuRF
           timing firmware reset, resulting in a ~msec dropout of
           received timing data, including external triggers.  The
           :func:`get_timing_rx_rst_cnt` returns the value of the
           counter that increments everytime there is a reset.  In
           streamed data triggering on external timing, this will look
           like jumps in time without corresponding dropped frames.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            32-bit counter which increments every time the SMuRF
            carrier firmware fails to decode a 10-bit timing
            system word.

        See Also
        --------
        :func:`get_timing_crc_err_cnt` : Gets CRC error counter for
            received timing frames.

        :func:`get_timing_rx_dsp_err_cnt` : Gets disparity error counter
            for received timing characters.

        :func:`get_timing_rx_rst_cnt` : Gets timing data link reset
            counter.
        """
        return self._caget(
            self.timing_status + self._timing_rx_dec_err_cnt,
            **kwargs)

    _timing_rx_dsp_err_cnt = "RxDspErrCount"

    def get_timing_rx_dsp_err_cnt(self, **kwargs):
        r"""Gets disparity error counter for received timing characters.

        When the timing system sends out a character it has the choice
        of two to send ; each valid character has a dedicated on/off
        bit.  The timing system toggles this on/off bit every other
        character.  The SMuRF carrier then keeps a running sum of how
        many on vs off bits it receives from the timing system.  This
        disparity error counter increments every time the SMuRF
        carrier firmware detects too many "on" or "off" bits,
        registering that a timing character transmitted by the timing
        system must have gotten dropped.

        Common causes of timing system error counter increments
        include bad network connections between the external timing
        system and the SMuRF system) and providing the wrong frequency
        or amplitude 122.88 MHz clock reference signal to the timing
        system.

        Timing data is transmitted and received at a total data rate
        of 2.45 Gbps (requiring 10G SFPs and compatible fiber links
        between the external timing system and the SMuRF system(s)),
        and timing frames are transmitted and received at 480 kHz.
        The protocol used for communcation between the external timing
        and SMuRF system(s) is a serial 8B/10B encoding using the
        K-character symbols for byte and frame alignment.  The
        encoding/decoding and byte alignment is supporte by common
        Xilinx IP.

        .. warning::
           An increment in any of the timing system error counters
           (obtainable through :func:`get_timing_crc_err_cnt`,
           :func:`get_timing_rx_dec_err_cnt`, and
           :func:`get_timing_rx_dsp_err_cnt`) will cause a SMuRF
           timing firmware reset, resulting in a ~msec dropout of
           received timing data, including external triggers.  The
           :func:`get_timing_rx_rst_cnt` returns the value of the
           counter that increments everytime there is a reset.  In
           streamed data triggering on external timing, this will look
           like jumps in time without corresponding dropped frames.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            32-bit counter which increments every time the SMuRF
            system detects a disparity error in the stream of decoded
            timing characters.

        See Also
        --------
        :func:`get_timing_crc_err_cnt` : Gets CRC error counter for
            received timing frames.

        :func:`get_timing_rx_dec_err_cnt` : Gets decode error
            counter for received timing characters.

        :func:`get_timing_rx_rst_cnt` : Gets timing data link reset
            counter.
        """
        return self._caget(
            self.timing_status + self._timing_rx_dsp_err_cnt,
            **kwargs)

    _timing_rx_rst_cnt = "RxRstCount"

    def get_timing_rx_rst_cnt(self, **kwargs):
        r"""Gets timing data link reset counter.

        An increment in any of the timing system error counters
        (obtainable through :func:`get_timing_crc_err_cnt`,
        :func:`get_timing_rx_dec_err_cnt`, and
        :func:`get_timing_rx_dsp_err_cnt`) will cause a SMuRF timing
        firmware reset, resulting in a ~msec dropout of received
        timing data, including external triggers.  This function
        returns the value of the counter that increments everytime
        there is a reset.  In streamed data triggering on external
        timing, this will look like jumps in time without
        corresponding dropped frames.

        Common causes of timing system error counter increments which
        trigger timing data link resets include bad network
        connections between the external timing system and the SMuRF
        system) and providing the wrong frequency or amplitude 122.88
        MHz clock reference signal to the timing system.

        Timing data is transmitted and received at a total data rate
        of 2.45 Gbps (requiring 10G SFPs and compatible fiber links
        between the external timing system and the SMuRF system(s)),
        and timing frames are transmitted and received at 480 kHz.
        The protocol used for communcation between the external timing
        and SMuRF system(s) is a serial 8B/10B encoding using the
        K-character symbols for byte and frame alignment.  The
        encoding/decoding and byte alignment is supporte by common
        Xilinx IP.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            32-bit counter which increments every time there is a
            timing data link reset.

        See Also
        --------
        :func:`get_timing_crc_err_cnt` : Gets CRC error counter for
            received timing frames.

        :func:`get_timing_rx_dec_err_cnt` : Gets decode error
            counter for received timing characters.

        :func:`get_timing_rx_dsp_err_cnt` : Gets disparity error counter
            for received timing characters.
        """
        return self._caget(
            self.timing_status + self._timing_rx_rst_cnt,
            **kwargs)

    def set_lmk_enable(self, bay, val, **kwargs):
        r"""
        Enable the AMC LMK in bay 0. On boot, the LMK is enabled, however once
        the DACS are reset on SmurfControl.setup the LMK is disabled. If you
        need to modify LMK values, this value must be 1.

        Args
        ----
        bay : int
            0 ot 1.
        val : int
            0 or 1.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `epics.caput` call.
        """
        self._caput(self.lmk.format(bay) + 'enable', val, **kwargs)

    def get_lmk_enable(self, bay, **kwargs):
        r"""
        Set the LMK:Enable bit.

        Args
        ----
        bay : int
            0 or 1.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `epics.caget` call.
        """
        self._caget(self.lmk.format(bay) + 'Enable', **kwargs)

    # assumes it's handed the decimal equivalent
    _lmk_reg = "LmkReg_0x{:04X}"

    def set_lmk_reg(self, bay, reg, val, **kwargs):
        """
        Can call like this get_lmk_reg(bay=0,reg=0x147,val=0xA)
        to see only hex as in gui.
        """
        self._caput(
            self.lmk.format(bay) + self._lmk_reg.format(reg),
            val, **kwargs)

    def get_lmk_reg(self, bay, reg, **kwargs):
        """
        Can call like this hex(get_lmk_reg(bay=0,reg=0x147))
        to see only hex as in gui.
        """
        return self._caget(
            self.lmk.format(bay) + self._lmk_reg.format(reg),
            **kwargs)

    _mcetransmit_debug_reg = 'AMCc.mcetransmitDebug'

    def set_mcetransmit_debug(self, val, **kwargs):
        """
        Sets the mcetransmit debug bit. If 1, the debugger will
        print to the pyrogue screen.

        Args
        ----
        val : int
            0 or 1 for the debug bit.
        """
        self._caput(self._mcetransmit_debug_reg, val, **kwargs)

    _frame_count_reg = 'FrameCnt'

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
        return int(self._caget(
            self.frame_rx_stats + self._frame_count_reg,
            as_string=True,
            **kwargs))

    _frame_size_reg = 'FrameSize'

    def get_frame_size(self, **kwargs):
        """
        Gets the size of the frame going into the smurf processor.

        Returns
        -------
        int
            The size of the data frame into the smurf processor.
        """
        return self._caget(
            self.frame_rx_stats + self._frame_size_reg,
            **kwargs)

    _frame_loss_count_reg = 'FrameLossCnt'

    def get_frame_loss_cnt(self, **kwargs):
        """
        The number of frames that did not make it to the smurf
        processor
        """
        return self._caget(
            self.frame_rx_stats + self._frame_loss_count_reg,
            **kwargs)

    _frame_out_order_count_reg = 'FrameOutOrderCnt'

    def get_frame_out_order_count(self, **kwargs):
        """
        """
        return self._caget(
            self.frame_rx_stats + self._frame_out_order_count_reg,
            **kwargs)

    _channel_mask_reg = 'ChannelMapper.Mask'

    def set_channel_mask(self, mask, **kwargs):
        """
        Set the smurf processor channel mask.

        Args
        ----
        mask : list
            The channel mask.
        """
        self._caput(
            self.smurf_processor + self._channel_mask_reg,
            mask, **kwargs)


    def get_channel_mask(self, **kwargs):
        """
        Gets the smuf processor channel mask.

        Returns
        -------
        mask : list
            The channel mask.
        """
        return self._caget(
            self.smurf_processor + self._channel_mask_reg,
            **kwargs)

    _unwrapper_reset_reg = 'Unwrapper.reset'

    def set_unwrapper_reset(self, **kwargs):
        """
        Resets the unwrap filter. There is no get function because
        it is an executed command.
        """
        self._caput(
            self.smurf_processor + self._unwrapper_reset_reg,
            1, **kwargs)

    _filter_reset_reg = 'Filter.reset'

    def set_filter_reset(self, **kwargs):
        """
        Resets the downsample filter
        """
        self._caput(
            self.smurf_processor + self._filter_reset_reg,
            1, **kwargs)

    _filter_a_reg = 'Filter.A'

    def set_filter_a(self, coef, **kwargs):
        """
        Set the smurf processor filter A coefficients.

        Args
        ----
        coef : list
            The filter A coefficients.
        """
        self._caput(
            self.smurf_processor + self._filter_a_reg,
            coef, **kwargs)


    def get_filter_a(self, **kwargs):
        """
        Gets the smurf processor filter A coefficients.

        Returns
        -------
        coef : list
            The filter A coefficients.
        """
        if self.offline:  # FIX ME - STUPPID HARDCODE
            return np.array(
                [ 1., -3.74145562,  5.25726624,
                  -3.28776591, 0.77203984])

        return self._caget(
            self.smurf_processor + self._filter_a_reg,
            **kwargs)

    _filter_b_reg = 'Filter.B'

    def set_filter_b(self, coef, **kwargs):
        """
        Set the smurf processor filter B coefficients.

        Args
        ----
        coef : list
            The filter B coefficients.
        """
        self._caput(
            self.smurf_processor + self._filter_b_reg,
            coef, **kwargs)


    def get_filter_b(self, **kwargs):
        """
        Get the smurf processor filter B coefficients.

        Returns
        -------
        coef : list
            The filter B coefficients.
        """
        if self.offline:
            return np.array(
                [5.28396689e-06, 2.11358676e-05, 3.17038014e-05,
                 2.11358676e-05, 5.28396689e-06])

        return self._caget(
            self.smurf_processor + self._filter_b_reg,
            **kwargs)

    _filter_order_reg = 'Filter.Order'

    def set_filter_order(self, order, **kwargs):
        """
        Set the smurf processor filter order.

        Args
        ----
        int
            The filter order.
        """
        self._caput(
            self.smurf_processor + self._filter_order_reg,
            order, **kwargs)

    def get_filter_order(self, **kwargs):
        """
        Get the smurf processor filter order.

        Args
        ----
        int
            The filter order.
        """
        return self._caget(
            self.smurf_processor + self._filter_order_reg,
            **kwargs)

    _filter_gain_reg = 'Filter.Gain'

    def set_filter_gain(self, gain, **kwargs):
        """
        Set the smurf processor filter gain.

        Args
        ----
        float
            The filter gain.
        """
        self._caput(
            self.smurf_processor + self._filter_gain_reg,
            gain, **kwargs)

    def get_filter_gain(self, **kwargs):
        """
        Get the smurf processor filter gain.

        Returns
        -------
        float
            The filter gain.
        """
        return self._caget(
            self.smurf_processor + self._filter_gain_reg,
            **kwargs)

    _downsampler_mode_reg = 'Downsampler.DownsamplerMode'

    def set_downsample_mode(self, mode):
        """
        Set the downsampler mode. 0 is internal, 1 is external.

        Ref. SmurfHeader.h, SmurfHeader.cpp, _SmurfProcessor.py
        Ref. https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+Processor
        """
        if mode == 'internal':
            self._caput(self.smurf_processor + self._downsampler_mode_reg, 0)
        elif mode == 'external':
            self._caput(self.smurf_processor + self._downsampler_mode_reg, 1)
        else:
            self.log(f'set_downsample_mode: Unknown mode {mode}')

    def get_downsample_mode(self):
        """
        Get the downsampler mode. 0 is internal, 1 is external.

        Ref. SmurfHeader.h, SmurfHeader.cpp, _SmurfProcessor.py
        Ref. https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+Processor
        """
        mode = self._caget(self.smurf_processor + self._downsampler_mode_reg)

        ret = 'Unknown'

        if mode == 0:
            ret = 'internal'
        elif mode == 1:
            ret = 'external'
        else:
            self.log(f'get_downsample_mode: Unknown mode {mode}')

        return ret

    _downsampler_factor_reg = 'Downsampler.InternalFactor'

    def set_downsample_factor(self, factor, get_nearby=False, desperation=2, **kwargs):
        """
        Set the smurf processor down-sampling factor.

        Args
        ----
        int
            The down-sampling factor.
        get_nearby: bool
            If True, will try to find a bitmask close to the specified
            downsample factor if the given factor cannot be expressed
        desperation: int
            Desparation factor used when finding nearby ds-factors. This limits
            how far the ds-factor can be from the one requested
        """
        mode = self.get_downsample_mode()
        if mode == 'external':
            dsc = dscounters.DownsampleCounters(dscounters.configs['v3'])
            if get_nearby:
                bmask = dsc.get_nearby(factor, desperation=desperation)
            else:
                bmask = dsc.get_mask(factor)

            if bmask is None:
                raise ValueError(f"Could not create bitmask for factor {factor}")

            # The bitmask is shifted 10 bits to the left because the
            # first 10 bits of the timing system counters are the higher rate
            # FIXEDDIV counters.
            bmask = bmask << 10
            self.set_downsample_external_bitmask(bmask)

            return
        else:
            self._caput(
                self.smurf_processor + self._downsampler_factor_reg,
                factor, **kwargs)

    def get_downsample_factor(self, **kwargs):
        """
        Get the smurf processor down-sampling factor. This is only used
        when the downsampling mode is internal.  When the downsampling
        mode is internal, the server will receive frames, place them
        through SmurfProcessor.cpp, but it will not release any data
        to be saved until it has counted up to this factor.

        Returns
        -------
        int
            The down-sampling factor.
        """
        if self.offline:
            self.log("get_downsample_factor: offline is True, SmurfProcessor.cpp is not running..")
            return 20

        if self.get_downsample_mode() == 'external':
            dsc = dscounters.DownsampleCounters(dscounters.configs['v3'])
            bmask = self.get_downsample_external_bitmask() >> 10
            return dsc.period_from_mask(bmask)

        return self._caget(self.smurf_processor + self._downsampler_factor_reg, **kwargs)

    _downsampler_external_bitmask_reg = 'Downsampler.ExternalBitmask'

    def set_downsample_external_bitmask(self, bitmask):
        """
        Set the downsampler external bitmask.

        Ref. https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+Processor
        """
        self._caput(self.smurf_processor + self._downsampler_external_bitmask_reg, bitmask)

    def get_downsample_external_bitmask(self):
        """
        Get the downsampler external bitmask. This bitmask is only used
        when get_downsample_mode is external. For example, 0 means
        never trigger, 1 means trigger on the first bit, 2 will
        trigger on the second bit, 4 will trigger on the third bit. By
        bit, we mean when the bit flips in one direction, because the
        bit will flip back on the next incoming data stream.

        Ref. https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+Processor
        """
        return self._caget(self.smurf_processor + self._downsampler_external_bitmask_reg)

    _filter_disable_reg = "Filter.Disable"

    def set_filter_disable(self, disable_status, **kwargs):
        """
        If Disable is set to True, then the filter is off. Incoming data
        to SmurfProcessor.cpp may still be downsampled, however,
        filtering would not be applied.

        Args
        ----
        bool
            The status of the Disable bit.
        """
        self._caput(
            self.smurf_processor + self._filter_disable_reg,
            disable_status, **kwargs)

    def get_filter_disable(self, **kwargs):
        """
        If Disable is set to True, then the downsampling filter is
        off.

        Returns
        -------
        bool
            The status of the Disable bit.
        """
        return self._caget(
            self.smurf_processor + self._filter_disable_reg,
            **kwargs)

    _max_file_size_reg = 'FileWriter.MaxFileSize'

    def set_max_file_size(self, size, **kwargs):
        """Set maximum file size for streamed data.

        If nonzero, when streaming data to disk, will split data over
        files of this size, in bytes.  Files have the usual name but
        with an incrementing integer appended at the end, e.g. .dat.1,
        .dat.2, etc..

        Args
        ----
        size : int
            Number of bytes to limit the size of each file streamed to
            disk to before rolling over into a new file.  If zero, no
            limit.

        See Also
        --------
        :func:`get_max_file_size` : Get maximum file size for streamed data.

        """
        assert (isinstance(size,int)),f'size={size} should be type int, doing nothing'
        assert (size>=0),f'size={size} must be greater than zero, doing nothing'
        self._caput(
            self.smurf_processor + self._max_file_size_reg,
            str(size), **kwargs)

    def get_max_file_size(self, **kwargs):
        """Get maximum file size for streamed data.

        If nonzero, when streaming data to disk, will split data over
        files of this size, in bytes.  Files have the usual name but
        with an incrementing integer appended at the end, e.g. .dat.1,
        .dat.2, etc..


        Returns
        -------
        int
            Maximum file size for streamed data in bytes.  Returns
            zero if there's no limit in place.

        See Also
        --------
        :func:`set_max_file_size` : Get maximum file size for streamed data.

        """
        return int(self._caget(
            self.smurf_processor + self._max_file_size_reg,
            as_string=True, **kwargs))

    _data_file_name_reg = 'FileWriter.DataFile'

    def set_data_file_name(self, name, **kwargs):
        """
        Set the data file name.

        Args
        ----
        str
            The file name.
        """
        self._caput(
            self.smurf_processor + self._data_file_name_reg,
            name, **kwargs)

    def get_data_file_name(self, **kwargs):
        """
        Set the data file name.

        Returns
        -------
        str
            The file name.
        """
        return self._caget(self.smurf_processor + self._data_file_name_reg,
            **kwargs)

    _data_file_open_reg = 'FileWriter.Open'

    def open_data_file(self, **kwargs):
        """
        Open the data file.
        """
        self._caput(self.smurf_processor + self._data_file_open_reg, 1,
            **kwargs)

    _data_file_close_reg = 'FileWriter.Close'

    def close_data_file(self, **kwargs):
        """
        Close the data file.
        """
        self._caput(self.smurf_processor + self._data_file_close_reg, 1,
            **kwargs)

    _num_channels_reg = "NumChannels"

    def get_smurf_processor_num_channels(self, **kwargs):
        """
        This is the number of channels that smurf_processor (the thing that
        does the downsampling, filtering, etc and then swrites to disk/streams
        data to the DAQ) thinks are on.

        This value is read only.
        """
        return self._caget(
            self.channel_mapper + self._num_channels_reg,
            **kwargs)

    _payload_size_reg = "PayloadSize"

    def set_payload_size(self, payload_size, **kwargs):
        """
        The payload size defines the number of available channels to
        write to disk/stream. Payload size must be larger than the
        number of channels going into the channel mapper

        Args
        ----
        int
            The number of channels written to disk.  This is
            independent of the number of active channels.
        """
        self._caput(
            self.channel_mapper + self._payload_size_reg,
            payload_size, **kwargs)

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
        return self._caget(
            self.channel_mapper + self._payload_size_reg,
            **kwargs)

    ### Data emulator

    def set_predata_emulator_enable(self, val, **kwargs):
        """
        Turns on and off the predata emulator.
        """
        self._caput(self._predata_emulator + 'enable', val, **kwargs)

    def get_predata_emulator_enable(self, **kwargs):
        """
        Gets the enable bit for predata emulator.
        """
        return self._caget(self._predata_emulator + 'enable', **kwargs)

    _predata_emulator_disable = "Disable"

    def set_predata_emulator_disable(self, val, **kwargs):
        """
        Sets the predata emulator disable status.

        Args
        ----
        val : bool
        """
        self._caput(self._predata_emulator + self._predata_emulator_disable,
            val, **kwargs)

    def get_predata_emulator_disable(self, **kwargs):
        """
        Gets the predata emulator disable status.

        Returns
        -------
        type : bool
        """
        return self._caget(self._predata_emulator + self._predata_emulator_disable,
            **kwargs)

    _predata_emulator_type = "Type"

    def set_predata_emulator_type(self, val, **kwargs):
        """
        No description

        Args
        ----
        val : str
            The data type. Choices are - Zeros, ChannelNumber, Random, Square,
            Sawtooth, Triangle, Sine, and DropFrame
        """
        self._caput(self._predata_emulator + self._predata_emulator_type, val,
            **kwargs)

    def get_predata_emulator_type(self, **kwargs):
        """
        Gets the predata emulator type.

        Returns
        -------
        type : int
            0 - Zeros, 1 - ChannelNumber, 2 - Random, 3 - Square,
            4 - Sawtooth, 5 - Triangle, 6 - Sine, 7 - DropFrame
        """
        return self._caget(self._predata_emulator + self._predata_emulator_type,
            **kwargs)

    _predata_emulator_amplitude = "Amplitude"

    def set_predata_emulator_amplitude(self, val, **kwargs):
        """
        """
        self._caput(self._predata_emulator + self._predata_emulator_amplitude,
            val, **kwargs)

    def get_predata_emulator_amplitude(self, **kwargs):
        """
        """
        return self._caget(self._predata_emulator +
            self._predata_emulator_amplitude, **kwargs)

    _predata_emulator_offset = "Offset"

    def set_predata_emulator_offset(self, val, **kwargs):
        """
        """
        self._caput(self._predata_emulator + self._predata_emulator_offset, val,
            **kwargs)

    def get_predata_emulator_offset(self, **kwargs):
        """
        """
        return self._caget(self._predata_emulator +
            self._predata_emulator_offset, **kwargs)

    _predata_emulator_period = "Period"

    def set_predata_emulator_period(self, val, **kwargs):
        """
        Expressed as the number of incoming frames. It must be greater that 2.
        This period will be expressed in term of the period of the received
        frames, which in turn is related to the flux ramp period.

        Args
        ----
        val : int
            Number of frames that make up a period.
        """
        # Cast as str
        if not isinstance(val, str):
            val = str(val)
        self._caput(self._predata_emulator + self._predata_emulator_period, val,
            **kwargs)

    def get_predata_emulator_period(self, **kwargs):
        """
        Expressed as the number of incoming frames. It must be greater that 2.
        This period will be expressed in term of the period of the received
        frames, which in turn is related to the flux ramp period.

        Returns
        -------
        period : int
            Number of frames that make up a period
        """
        # Get as string and then cast as int
        return int(self._caget(self._predata_emulator +
            self._predata_emulator_period, as_string=True, **kwargs))

    def set_postdata_emulator_enable(self, val, **kwargs):
        """
        """
        self._caput(self._postdata_emulator + 'enable', val, **kwargs)

    def get_postdata_emulator_enable(self, **kwargs):
        """
        """
        return self._caget(self._postdata_emulator + 'enable', **kwargs)

    _postdata_emulator_type = "Type"

    def set_postdata_emulator_type(self, val, **kwargs):
        """
        No description

        Args
        ----
        val : str
            The data type. Choices are - Zeros, ChannelNumber, Random, Square,
            Sawtooth, Triangle, Sine, and DropFrame
        """
        self._caput(self._postdata_emulator + self._postdata_emulator_type, val,
            **kwargs)

    def get_postdata_emulator_type(self, **kwargs):
        """
        Gets the postdata emulator type.

        Returns
        -------
        type : int
            0 - Zeros, 1 - ChannelNumber, 2 - Random, 3 - Square,
            4 - Sawtooth, 5 - Triangle, 6 - Sine, 7 - DropFrame
        """
        return self._caget(self._postdata_emulator +
            self._postdata_emulator_type, **kwargs)

    _postdata_emulator_amplitude = "Amplitude"

    def set_postdata_emulator_amplitude(self, val, **kwargs):
        """
        """
        self._caput(self._postdata_emulator + self._postdata_emulator_amplitude,
            val, **kwargs)

    def get_postdata_emulator_amplitude(self, **kwargs):
        """
        """
        return self._caget(self._postdata_emulator +
            self._postdata_emulator_amplitude, **kwargs)

    _postdata_emulator_offset = "Offset"

    def set_postdata_emulator_offset(self, val, **kwargs):
        """
        """
        self._caput(self._postdata_emulator + self._postdata_emulator_offset,
            val, **kwargs)

    def get_postdata_emulator_offset(self, **kwargs):
        """
        """
        return self._caget(self._postdata_emulator +
            self._postdata_emulator_offset, **kwargs)

    _postdata_emulator_period = "Period"

    def set_postdata_emulator_period(self, val, **kwargs):
        """
        Expressed as the number of incoming frames. It must be greater that 2.
        This period will be expressed in terms of the downsampler periods. Note
        that this is different from the predata emulator.

        Args
        -----
        val : int
            Number of frames that make up a period.
        """
        if not isinstance(val, str):
            val = str(val)
        self._caput(self._postdata_emulator + self._postdata_emulator_period,
            val, **kwargs)

    def get_postdata_emulator_period(self, **kwargs):
        """
        No description

        Returns
        -------
        period : int
            Number of frames that make up a period.
        """
        return int(self._caget(self._postdata_emulator +
            self._postdata_emulator_period, as_string=True, **kwargs))

    _stream_data_source_enable = "SourceEnable"

    def set_stream_data_source_enable(self, val, **kwargs):
        """
        Sets the data stream enable bit. Must be set to True to stream data.
        """
        self._caput(self.stream_data_source + self._stream_data_source_enable,
            val, **kwargs)

    def get_stream_data_source_enable(self, **kwargs):
        """
        """
        return self._caget(self.stream_data_source +
            self._stream_data_source_enable, **kwargs)

    _stream_data_period = "Period"

    def set_stream_data_source_period(self, val, **kwargs):
        """
        Sets the data source period.
        """
        self._caput(self.stream_data_source + self._stream_data_period, val,
            **kwargs)

    def get_stream_data_source_period(self, **kwargs):
        """
        Gets the data source period.
        """
        return self._caget(self.stream_data_source + self._stream_data_period,
            **kwargs)

    def shell_command(self,cmd,**kwargs):
        r"""Runs command on shell and returns code, stdout, & stderr.

        Args
        ----
        cmd : str
            Command to run on shell.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `subprocess.run` call.

        Returns
        -------
        (stdout, stderr)
            stdout and stderr returned as str
        """
        result = subprocess.run(
            cmd.split(), stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=False, **kwargs
        )

        return result.stdout.decode(),result.stderr.decode()

    def get_fru_info(self,board,bay=None,slot_number=None,shelf_manager=None):
        r"""Returns FRU information for SMuRF board.

        Wrapper for dumping the FRU information for SMuRF boards using
        shell commands.

        Args
        ----
        board : str
            Which board to return FRU informationf for.  Valid options
            include 'amc', 'carrier', or 'rtm'.  If 'amc', must also
            provide the bay argument.
        bay : int, optional, default None
            Which bay to return the AMC FRU information for.  Used
            only if board='amc'.
        slot_number : int or None, optional, default None
            The crate slot number that the AMC is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        shelf_manager : str or None, optional, default None
            Shelf manager ip address.  If None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems the default name of the shelf manager
            is 'shm-smrf-sp01'.

        Returns
        -------
        fru_info_dict : dict
            Dictionary of requested FRU information.  Returns None if
            board not a valid option, board not present in slot, slot
            not present in shelf, or if no AMC is up in the requested
            bay.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if shelf_manager is None:
            shelf_manager=self.shelf_manager

        valid_board_options=['amc','rtm','carrier']
        if board not in valid_board_options:
            self.log(f'ERROR : {board} not in list of valid board options {valid_board_options}.  Returning None.',self.LOG_ERROR)
            return None

        shell_cmd=''
        shell_cmd_prefix=None
        if board=='amc':
            shell_cmd_prefix='amc'
            # require bay argument
            if bay is None:
                self.log('ERROR : Must provide AMC bay.  Returning None.',self.LOG_ERROR)
                return None
            if bay not in [0,1]:
                self.log('ERROR : bay argument can only be 0 or 1.  Returning None.',self.LOG_ERROR)
                return None
            shell_cmd+=f'/{bay*2}'
        elif board=='rtm':
            # require bay argument
            shell_cmd_prefix='rtm'
        else: # only carrier left
            shell_cmd_prefix='fru'

        shell_cmd=f'cba_{shell_cmd_prefix}_init -d {shelf_manager}/{slot_number}'+shell_cmd
        stdout,stderr=self.shell_command(shell_cmd)

        # Error handling
        if 'AMC not present in bay' in stdout:
            self.log('ERROR : AMC not present in bay!  Returning None.',
                     self.LOG_ERROR)
            return None
        if 'Slot not present in shelf' in stdout:
            self.log('ERROR : Slot not present in shelf!  Returning None.',
                     self.LOG_ERROR)
            return None
        if 'Board not present in slot' in stdout:
            self.log('ERROR : Board not present in slot!  Returning None.',
                     self.LOG_ERROR)
            return None
        else: # parse and return fru information for this board
            stdout=stdout.split('\n')
            fru_info_dict={}
            for line in stdout:
                if ':' in line:
                    splitline=line.split(':')
                    if len(splitline)==2:
                        fru_key=splitline[0].lstrip().rstrip()
                        fru_value=splitline[1].lstrip().rstrip()
                        if len(fru_value)>0: # skip header
                            fru_info_dict[fru_key]=fru_value

        return fru_info_dict
