
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
from typing import Literal

import numpy as np
from packaging import version
try:
    from pyrogue import VariableWait
except ModuleNotFoundError:
    # there will be warnings elsewhere
    pass

from pysmurf.client.base import SmurfBase
from pysmurf.client.command.sync_group import SyncGroup
from pysmurf.client.util import tools, dscounters


class SmurfCommandMixin(SmurfBase):

    def _skipifrfsoc(func):
        def skipper(self, *args,**kwargs):
            result = None
            if not self.is_rfsoc:
                result  = func(self, *args,**kwargs)
            else:
                print(f'Function {func.__name__} called, but not implemented on RFSoC.  Skipping call and returning None!')
            return result
        return skipper

    _global_poll_enable_reg = 'AMCc.enable'

    def _caput(self, pvname, val, index=-1, cast_type=True, write_log=False, log_level=None,
               execute=True, wait_before=None, wait_after=None, wait_done=True, **kwargs):
        """Sets to rogue variables in the root.

        Args
        ----
        pvname : str
            The path of the PV to set to.
        val: any
            The value to set.
        index: int
            Index into an array variable. Ignored if variable is scalar.
        cast_type: bool, default True
            Check the type of val and cast to that expected for the rogue
            variable.
        write_log : bool, optional, default False
            Whether to log the data or not.
        execute : bool, optional, default True
            Whether to actually execute the command.
        wait_before : float, optional, default None
            If not None, the number of seconds to wait before issuing
            the command.
        wait_after : float, optional, default None
            If not None, the number of seconds to wait after issuing
            the command.
        wait_done : bool, optional, default True
            Wait for the command to be finished before returning.
        log_level : int, optional, default to INFO
            Log level.
        """

        if log_level is None:
            log_level = self.LOG_INFO

        if kwargs:
            for k in kwargs:
                self.log(f"caput unexpected kwarg: {k}: {kwargs[k]}", self.LOG_ERROR)

        if wait_before is not None:
            if write_log:
                self.log(f'Waiting {wait_before:3.2f} seconds before...',
                         self.LOG_USER)
            time.sleep(wait_before)

        if write_log:
            log_str = 'caput ' + pvname + ' ' + str(val)
            if self.offline:
                log_str = 'OFFLINE - ' + log_str
            self.log(log_str, log_level)

        # only python integer type will be accepted by rogue
        index = int(index)

        # execute the set
        if execute and not self.offline:
            # NB this used to support getting the _atca root, but I can't
            # find any instances of this actually being used
            var = self._client.root.getNode(pvname)
            if var is None:
                raise ValueError(f"Invalid node: {pvname}")

            # handle different uses of `put`
            if var.isCommand:
                # a command is blocking so wait_done is moot
                var.call(val)
            elif var.enum is not None:
                # handle numpy type
                if isinstance(val, np.generic):
                    val = val.item()
                # setDisp handles enum values
                var.setDisp(val, index=index)
            elif cast_type:
                # rogue is strict about variable types for arrays
                var_val = var.value()  # like get(read=False)
                if isinstance(var_val, np.ndarray):
                    var_type = var_val.dtype.type
                else:
                    var_type = type(var_val)
                val = var_type(val)
                var.set(val, check=wait_done, index=index)
            else:
                var.set(val, check=wait_done, index=index)

        if wait_after is not None:
            if write_log:
                self.log(f'Waiting {wait_after:3.2f} seconds after...',
                    self.LOG_USER)
            time.sleep(wait_after)
            if write_log:
                self.log('Done waiting.', self.LOG_USER)


    def _caget(self, pvname, index=-1, write_log=False, log_level=None, execute=True,
               as_string=False, count=None, yml=None, **kwargs):
        """Gets variables from rogue root.

        Args
        ----
        pvname : str
            The path of the PV to get.
        index: int
            Index into an array variable. Ignored if variable is scalar.
        as_string : bool, default False
            Return the string provided by getDisp.
        write_log : bool, optional, default False
            Whether to log the data or not.
        log_level : int, optional, default INFO
            Log level.
        execute : bool, optional, default True
            Whether to actually execute the command.
        count : int or None, optional, default None
            Number of elements to return for array data.
        yml : str or None, optional, default None
            If not None, yaml file to parse for the result.

        Returns
        -------
        ret : any
            The requested value.
        """

        if log_level is None:
            log_level = self.LOG_INFO

        if kwargs:
            for k in kwargs:
                self.log(f"caget unexpected kwarg: {k}: {kwargs[k]}")

        # load the data from yml file if provided
        if yml is not None:
            if write_log:
                self.log(f'Reading from yml file\n {pvname}', log_level)
            ret = tools.yaml_parse(yml, pvname)
            if write_log:
                self.log(ret, log_level)
            return ret

        if not execute or self.offline:
            self.log(f"Not executing caget for {pvname} (execute={execute}, offline={self.offline})", log_level)
            # don't perform the read
            return None

        var = self._client.root.getNode(pvname)
        if var is None:
            raise ValueError(f"Invalid node: {pvname}")

        if write_log:
            self.log('caget ' + pvname, log_level)
        # Get the data
        if as_string:
            ret = var.getDisp(index=index)
        else:
            ret = var.get(index=index)

        if count is not None:
            try:
                ret = ret[:count]
            except (TypeError, IndexError):
                raise ValueError(
                    "Argument 'count' is present but cannot slice "
                    f"non-iterable variable of type {type(ret)}."
                )

        if write_log:
            self.log(ret, log_level)

        return ret


    def _wait_for(self, pvname, condition, timeout=None):
        """Wait for a variable to satisfy a certain condition.

        Args
        ----
        pvname : str
            The path of the PV to get.
        condition : function
            Returns True if the given variable value is such that we
            should stop waiting, False otherwise.
        timeout : float
            Timeout in seconds. Default is None.
        """
        var = self._client.root.getNode(pvname)
        if var is None:
            raise ValueError(f"Invalid node: {pvname}")

        if timeout is None:
            timeout = 0

        ret = VariableWait([var], lambda vals: condition(vals[0].value), timeout)
        if not ret:
            raise TimeoutError(f"Timed out after {timeout}s on PV {pvname}.")


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

    def wait_configuring_in_progress(self, timeout=None):
        """Wait until the system is no longer configuring.

        Wait until either the system configuring in progress flag is
        set to False or the timeout is reached.

        Args
        ----
        timeout : float
            Time in seconds to wait before raising a TimeoutError.
        """
        self._wait_for(
            self.smurf_application + self._configuring_in_progress_reg,
            lambda x: not x,  # condition for success is value of False
            timeout=timeout
        )

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

    def set_defaults_pv(self, max_timeout_sec=300.0, **kwargs):
        r"""Loads the default configuration.

        Calls the rogue `setDefaults` command, which loads the default
        software and hardware configuration.

        If using pysmurf core code versions >=4.1.0 (as reported by
        :func:`get_pysmurf_version`), returns `True` if the
        `setDefaults` command was successfully executed on the rogue
        side, or failed.  Returns `None` for older versions.

        Args
        ----
        max_timeout_sec : float, optional, default 300.0
            Seconds to wait for system to configure before giving up.
            Only used for pysmurf core code versions >= 4.1.0.
            The underlying process will give up after 240s by default,
            so if a shorter timeout is set here, it may not complete.
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

            # Start by calling the 'setDefaults' command.
            # This is now implemented as a rogue Process, so this call will
            # return immediately, and the wait loop that follows will begin
            self._caput('AMCc.setDefaults.Start', 1, **kwargs)

            # Now let's wait until the process is finished. We define a maximum
            # time we will wait, 400 seconds in this case
            try:
                self.wait_configuring_in_progress(max_timeout_sec)
            except TimeoutError:
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
            success = self.get_system_configured(**kwargs)

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
        r"""Reads all registers from hardware into the rogue server cache.

        Necessary to update registers that have pollInterval=0
        (no automatic polling).  Blocks for 20 seconds after issuing
        the command to allow the read to complete.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput('AMCc.ReadAll', 1, **kwargs)
        self.log('ReadAll sent', self.LOG_INFO)

    def run_pwr_up_sys_ref(self, bay, **kwargs):
        r"""Powers up the SYSREF signal on the LMK clock chip.

        Restores the SYSREF output on the LMK048xx clock generator
        after power-down. SYSREF is required for JESD204b
        synchronization. Waits 5 seconds after issuing the command.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        triggerPV=self.lmk.format(bay) + 'PwrUpSysRef'
        self._caput(triggerPV, 1, wait_after=5, **kwargs)
        self.log(f'{triggerPV} sent', self.LOG_USER)

    _eta_scan_in_progress_reg = 'etaScanInProgress'

    def get_eta_scan_in_progress(self, band, **kwargs):
        r"""Gets whether an eta scan or gradient descent is running.

        Returns the status flag indicating if a serial eta scan
        or serial gradient descent is in progress for the
        specified band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if a scan is in progress, 0 if idle.

        See Also
        --------
        :func:`run_serial_eta_scan` : Runs the serial eta scan.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        """
        return self._caget(self._cryo_root(band) + self._eta_scan_in_progress_reg,
                    **kwargs)

    _gradient_descent_max_iters_reg = 'gradientDescentMaxIters'

    def set_gradient_descent_max_iters(self, band, val, **kwargs):
        r"""Sets the maximum iterations for serial gradient descent.

        The gradient descent will stop after this many iterations
        even if it has not converged.

        Args
        ----
        band : int
            Which band.
        val : int
            Maximum number of iterations per channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_max_iters` : Gets the current value.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        :func:`set_gradient_descent_converge_hz` : Sets the convergence threshold.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_max_iters_reg,
            val, **kwargs)

    def get_gradient_descent_max_iters(self, band, **kwargs):
        r"""Gets the maximum iterations for serial gradient descent.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Maximum number of iterations per channel.

        See Also
        --------
        :func:`set_gradient_descent_max_iters` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_max_iters_reg,
            **kwargs)

    _gradient_descent_averages_reg = 'gradientDescentAverages'

    def set_gradient_descent_averages(self, band, val, **kwargs):
        r"""Sets the number of averages for gradient estimation.

        The gradient descent estimates the gradient by measuring
        frequency error at +/- an offset frequency. This sets how
        many measurements are averaged at each point to reduce
        noise.

        Args
        ----
        band : int
            Which band.
        val : int
            Number of frequency error measurements to average
            per gradient sample.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_averages` : Gets the current value.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_averages_reg,
            val, **kwargs)

    def get_gradient_descent_averages(self, band, **kwargs):
        r"""Gets the number of averages for gradient estimation.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Number of measurements averaged per gradient sample.

        See Also
        --------
        :func:`set_gradient_descent_averages` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_averages_reg,
            **kwargs)

    _gradient_descent_gain_reg = 'gradientDescentGain'

    def set_gradient_descent_gain(self, band, val, **kwargs):
        r"""Sets the gain (learning rate) for serial gradient descent.

        Scales the gradient before applying it as a frequency
        correction step. Larger values converge faster but risk
        overshooting. In momentum mode:
        v = beta*v + (1-beta)*gain*dx.

        Args
        ----
        band : int
            Which band.
        val : float
            Gain multiplier (no firmware-enforced limits).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_gain` : Gets the current value.
        :func:`set_gradient_descent_beta` : Sets the decay rate.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_gain_reg,
            val, **kwargs)

    def get_gradient_descent_gain(self, band, **kwargs):
        r"""Gets the gain (learning rate) for serial gradient descent.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Gain multiplier.

        See Also
        --------
        :func:`set_gradient_descent_gain` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_gain_reg,
            **kwargs)

    _gradient_descent_converge_hz_reg = 'gradientDescentConvergeHz'

    def set_gradient_descent_converge_hz(self, band, val, **kwargs):
        r"""Sets the convergence threshold for serial gradient descent.

        The gradient descent stops when the frequency step size
        falls below this threshold, indicating convergence.

        Args
        ----
        band : int
            Which band.
        val : float
            Convergence threshold in Hz.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_converge_hz` : Gets the current value.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        :func:`set_gradient_descent_max_iters` : Sets the max iterations.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_converge_hz_reg,
            val, **kwargs)

    def get_gradient_descent_converge_hz(self, band, **kwargs):
        r"""Gets the convergence threshold for serial gradient descent.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Convergence threshold in Hz.

        See Also
        --------
        :func:`set_gradient_descent_converge_hz` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_converge_hz_reg,
            **kwargs)

    _gradient_descent_step_hz_reg = 'gradientDescentStepHz'

    def set_gradient_descent_step_hz(self, band, val, **kwargs):
        r"""Sets the offset frequency for gradient estimation.

        The gradient is estimated by measuring frequency error at
        +/- this offset from the current center frequency.

        Args
        ----
        band : int
            Which band.
        val : float
            Offset frequency in Hz for gradient estimation.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_step_hz` : Gets the current value.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_step_hz_reg,
            val, **kwargs)

    def get_gradient_descent_step_hz(self, band, **kwargs):
        r"""Gets the offset frequency for gradient estimation.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Offset frequency in Hz.

        See Also
        --------
        :func:`set_gradient_descent_step_hz` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_step_hz_reg,
            **kwargs)

    _gradient_descent_momentum_reg = 'gradientDescentMomentum'

    def set_gradient_descent_momentum(self, band, val, **kwargs):
        r"""Sets the optimizer mode for serial gradient descent.

        When set to 1, uses momentum (exponential moving average
        of gradients for smoother convergence). When set to 0,
        uses an adaptive step size that scales inversely with
        the history of past gradients.

        Args
        ----
        band : int
            Which band.
        val : int
            1 for momentum mode, 0 for adaptive mode.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_momentum` : Gets the current value.
        :func:`set_gradient_descent_beta` : Sets the decay rate for both modes.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_momentum_reg,
            val, **kwargs)

    def get_gradient_descent_momentum(self, band, **kwargs):
        r"""Gets the optimizer mode for serial gradient descent.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 for momentum mode, 0 for adaptive mode.

        See Also
        --------
        :func:`set_gradient_descent_momentum` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_momentum_reg,
            **kwargs)

    _gradient_descent_beta_reg = 'gradientDescentBeta'

    def set_gradient_descent_beta(self, band, val, **kwargs):
        r"""Sets the decay rate for serial gradient descent.

        Controls the exponential decay rate for the running
        averages in both optimizer modes (where ``gain`` is set
        by :func:`set_gradient_descent_gain`):

        - Momentum mode: v = beta*v + (1-beta)*gain*dx
        - Adaptive mode: cache = beta*cache + (1-beta)*dx^2

        Values closer to 1.0 give more smoothing (longer memory),
        values closer to 0.0 give less smoothing.

        Args
        ----
        band : int
            Which band.
        val : float
            Decay rate (0.0 to 1.0).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_gradient_descent_beta` : Gets the current value.
        :func:`set_gradient_descent_gain` : Sets the gain multiplier.
        :func:`set_gradient_descent_momentum` : Sets the optimizer mode.
        :func:`run_serial_gradient_descent` : Runs the gradient descent.
        """
        self._caput(
            self._cryo_root(band) +
            self._gradient_descent_beta_reg,
            val, **kwargs)

    def get_gradient_descent_beta(self, band, **kwargs):
        r"""Gets the decay rate for serial gradient descent.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Decay rate (0.0 to 1.0).

        See Also
        --------
        :func:`set_gradient_descent_beta` : Sets the value.
        """
        return self._caget(
            self._cryo_root(band) +
            self._gradient_descent_beta_reg,
            **kwargs)

    _run_serial_eta_scan_reg = 'runSerialEtaScan'

    def run_serial_eta_scan(self, band, timeout=240, **kwargs):
        """
        Does an eta scan serially across the entire band. You must
        already be tuned close to the resontor dip. Use
        run_serial_gradient_descent to get it.

        Args
        ----
        band  : int
            The band to eta scan.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_eta_scan_reg
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg

        self._caput(triggerPV, 1, **kwargs)
        self._wait_for(monitorPV, lambda x: x == 0, timeout=timeout)


    _run_serial_gradient_descent_reg = 'runSerialGradientDescent'

    def run_serial_gradient_descent(self, band, timeout=240, **kwargs):
        """
        Does a gradient descent search for the minimum.

        Args
        ----
        band : int
            The band to run serial gradient descent on.
        timeout : float, optional, default 240
            The maximum amount of time to wait for the PV.
        """

        # need flux ramp off for this - enforce
        self.flux_ramp_off()

        triggerPV = self._cryo_root(band) + self._run_serial_gradient_descent_reg
        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg

        self._caput(triggerPV, 1, **kwargs)
        self._wait_for(monitorPV, lambda x: x == 0, timeout=timeout)


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
    # In rogue 6 this has moved to a process to avoid timing out
    # on a long-running command
    _save_state_reg = "AMCc.SaveConfigProcess"

    def _save_state_or_config(
        self, fname :  str, mode : Literal["Config", "Status"], timeout : float = 90.0,
        **kwargs
    ):
        # write out to a file
        self._caput(self._save_state_reg + ".SaveMode", "File", **kwargs)
        self._caput(self._save_state_reg + ".ConfigFile", fname, **kwargs)

        # select state or config
        self._caput(self._save_state_reg + ".DataType", mode, **kwargs)

        # start the process
        self._caput(self._save_state_reg + ".Start", 1, **kwargs)

        # wait for process to complete
        start = time.time()

        def keep_waiting():
            if (timeout == 0) or ((time.time() - start) <= timeout):
                return True
            raise TimeoutError(f"SaveConfigProcess timed out after {timeout}s.")

        while self._caget(self._save_state_reg + ".Running") and keep_waiting():
            time.sleep(0.1)
        # Check the return value from 'LoadConfig'.
        msg = self._caget(self._save_state_reg + ".Message")
        if msg != "Done":
            raise RuntimeError(f"SaveConfigProcess failed with '{msg}'")

    def save_state(self, val, **kwargs):
        """
        Dumps all PyRogue state variables to a yml file.

        Args
        ----
        val : str
            The path (including file name) to write the yml file to.
        """
        self._save_state_or_config(val, "Status", **kwargs)

    # alias older rogue 3 write_state function to save_state
    write_state = save_state

    # name changed in Rogue 4 from WriteConfig to SaveConfig.  Keeping
    # the write_config function for backwards compatibilty.
    def save_config(self, val, **kwargs):
        """
        Writes the current (un-masked) PyRogue settings to a yml file.

        Args
        ----
        val : str
            The path (including file name) to write the yml file to.
        """
        self._save_state_or_config(val, "Config", **kwargs)

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
        r"""Sets the path to the tune file for PyRogue loading.

        Sets the file path that the PyRogue ``loadTuneFile``
        command will read from. The tune file contains per-channel
        eta, center frequency, and amplitude parameters. When
        triggered, PyRogue reads the file on the server CPU and
        writes the parameters to firmware registers. Most users
        should use higher-level functions like :func:`load_tune`
        instead of setting this directly.

        Args
        ----
        val : str
            Path to the tune file (.npy format).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_tune_file_path` : Gets the current path.
        :func:`set_load_tune_file` : Triggers loading the tune file.
        :func:`load_tune` : Higher-level tune loading function.
        """
        self._caput(
            self.sysgencryo + self._tune_file_path_reg,
            val, **kwargs)

    def get_tune_file_path(self, **kwargs):
        r"""Gets the path to the tune file.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : str
            Path to the tune file.

        See Also
        --------
        :func:`set_tune_file_path` : Sets the path.
        """
        return self._caget(
            self.sysgencryo + self._tune_file_path_reg,
            **kwargs)

    _load_tune_file_reg = 'loadTuneFile'

    def set_load_tune_file(self, band, val, **kwargs):
        r"""Triggers loading the tune file into firmware registers.

        When set to 1, PyRogue reads the tune file (set by
        :func:`set_tune_file_path`) and writes the per-channel
        eta, center frequency, amplitude, and feedback enable
        parameters to firmware BRAM for the specified band.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to trigger loading.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_tune_file_path` : Sets the tune file path.
        :func:`load_tune` : Higher-level tune loading function.
        """
        self._caput(
            self._cryo_root(band) + self._load_tune_file_reg,
            val, **kwargs)


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
        r"""Sets the frequency array for find-freq or single-channel eta scan.

        Writes an array of scan frequencies in MHz (relative to subband
        center) into the rogue etaScanFreqs variable.  How the array is
        interpreted depends on which scan is triggered:

        - :func:`set_run_serial_find_freq`: reshapes to
          (n_channels × scan_points_per_channel), sweeping each channel
          through its row.  Channels with all-identical entries are
          skipped.
        - :func:`set_run_eta_scan`: uses the array as a flat 1D sweep
          applied to the single channel selected by
          :func:`set_eta_scan_channel`.

        Args
        ----
        band : int
            Which band.
        val : numpy.ndarray
            Array of scan frequencies in MHz.  For serial find-freq,
            flatten a (n_channels, n_scan_points) array before passing.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_run_serial_find_freq` : Triggers multi-channel sweep.
        :func:`set_run_eta_scan` : Triggers single-channel sweep.
        """
        self._caput(
            self._cryo_root(band) + self._eta_scan_freqs_reg,
            val, **kwargs)


    def get_eta_scan_freq(self, band, **kwargs):
        r"""Gets the frequency array for find-freq or single-channel eta scan.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        numpy.ndarray
            Array of scan frequencies in MHz previously set by
            :func:`set_eta_scan_freq`.

        See Also
        --------
        :func:`set_eta_scan_freq` : Sets this array.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_freqs_reg,
            **kwargs)

    _eta_scan_amplitude_reg = 'etaScanAmplitude'

    def set_eta_scan_amplitude(self, band, val, **kwargs):
        r"""Sets the tone amplitude used during eta scan or find-freq.

        This value is written to each channel's amplitudeScale register
        during :func:`set_run_eta_scan` or :func:`set_run_serial_find_freq`.
        Same scale as :func:`set_amplitude_scale_channel` — 4-bit unsigned
        (0–15), 3 dB per step.  Typical value is 12, which the
        firmware tone generation was optimized for for large numbers of
        tones.

        Args
        ----
        band : int
            Which band.
        val : int
            Tone amplitude for scanned channels (0–15).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_run_eta_scan` : Single-channel eta scan using this
                amplitude.
        :func:`set_run_serial_find_freq` : Multi-channel find-freq using
                this amplitude.
        """
        self._caput(
            self._cryo_root(band) + self._eta_scan_amplitude_reg,
            np.uint32(val), **kwargs)

    def get_eta_scan_amplitude(self, band, **kwargs):
        r"""Gets the tone amplitude used during eta scan or find-freq.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Tone amplitude for scanned channels (0–15, exactly 3 dB per
            step).

        See Also
        --------
        :func:`set_eta_scan_amplitude` : Sets this value.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_amplitude_reg,
            **kwargs)

    _eta_scan_channel_reg = 'etaScanChannel'

    def set_eta_scan_channel(self, band, val, **kwargs):
        r"""Sets the channel for the single-channel eta scan.

        Selects which channel :func:`set_run_eta_scan` will operate on.
        The scan sweeps this channel through the frequencies in
        :func:`set_eta_scan_freq` and records the frequency error
        response.

        Not used by :func:`set_run_serial_find_freq` or
        :func:`run_serial_eta_scan`, which operate on all channels.

        Args
        ----
        band : int
            Which band.
        val : int
            Channel number within the band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_run_eta_scan` : Triggers the scan on this channel.
        """
        self._caput(
            self._cryo_root(band) + self._eta_scan_channel_reg,
            val, **kwargs)

    def get_eta_scan_channel(self, band, **kwargs):
        r"""Gets the channel for the single-channel eta scan.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Channel number selected for the single-channel eta scan.

        See Also
        --------
        :func:`set_eta_scan_channel` : Sets this value.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_channel_reg,
            **kwargs)

    _eta_scan_averages_reg = 'etaScanAverages'

    def set_eta_scan_averages(self, band, val, **kwargs):
        r"""Sets the number of frequency error averages for serial eta scan.

        Used by :func:`run_serial_eta_scan` to average multiple frequency
        error readings at each measurement point.  Not used by
        :func:`set_run_eta_scan` (single-channel, no averaging).

        Args
        ----
        band : int
            Which band.
        val : int
            Number of frequency error samples to average at each point.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`run_serial_eta_scan` : Serial scan that uses this
                parameter.
        """
        self._caput(
            self._cryo_root(band) + self._eta_scan_averages_reg,
            val, **kwargs)

    def get_eta_scan_averages(self, band, **kwargs):
        r"""Gets the number of frequency error averages for serial eta scan.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Number of frequency error samples averaged at each point.

        See Also
        --------
        :func:`set_eta_scan_averages` : Sets this value.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_averages_reg,
            **kwargs)

    _run_serial_find_freq_reg = 'runSerialFindFreq'

    def set_run_serial_find_freq(self, band, val, **kwargs):
        r"""Triggers the serial find-freq scan across all channels.

        Starts the rogue SerialFindFreq process, which sweeps each
        channel through the frequencies previously loaded via
        :func:`set_eta_scan_freq` (reshaped to n_channels × scan_points).
        At each frequency, measures frequency error at eta phase 0° then
        -90°, storing results in etaScanResultsReal and etaScanResultsImag.
        Channels whose frequency rows are all identical are skipped.

        Blocks until the scan completes.

        Args
        ----
        band : int
            Which band.
        val : int
            Set to 1 to start the scan.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_eta_scan_freq` : Load frequencies before calling this.
        :func:`set_eta_scan_amplitude` : Set tone power before calling this.
        :func:`get_eta_scan_results_real` : Read back real results.
        :func:`get_eta_scan_results_imag` : Read back imaginary results.
        """
        self._caput(
            self._cryo_root(band) + self._run_serial_find_freq_reg,
            val, **kwargs)

        monitorPV = self._cryo_root(band) + self._eta_scan_in_progress_reg
        self._wait_for(monitorPV, lambda x: x == 0)
        self.log('serial find freq complete', self.LOG_USER)

    _run_eta_scan_reg = 'runEtaScan'

    def set_run_eta_scan(self, band, val, **kwargs):
        r"""Triggers the single-channel eta scan.

        Starts the rogue runEtaScan command, which sweeps the channel
        selected by :func:`set_eta_scan_channel` through the frequencies
        in :func:`set_eta_scan_freq`.  Measures frequency error at eta
        phase 90° then 0°, storing results in etaScanResultsReal and
        etaScanResultsImag.

        Blocks until the scan completes (rogue commands are synchronous).

        Args
        ----
        band : int
            Which band.
        val : int
            Set to 1 to start the scan.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_eta_scan_channel` : Select channel before calling this.
        :func:`set_eta_scan_freq` : Load frequencies before calling this.
        :func:`get_eta_scan_results_real` : Read back real results.
        :func:`get_eta_scan_results_imag` : Read back imaginary results.
        """
        self._caput(
            self._cryo_root(band) + self._run_eta_scan_reg,
            val, **kwargs)

    _eta_scan_results_real_reg = 'etaScanResultsReal'

    def get_eta_scan_results_real(self, band, count, **kwargs):
        r"""Gets the real component of the eta scan.

        Returns frequency error measured during the first sweep of
        the most recent :func:`set_run_eta_scan` or
        :func:`set_run_serial_find_freq`.  Treated as I when
        constructing the complex response as ``I + 1j*Q``.

        For serial find-freq, the array is flattened from shape
        (n_channels, n_scan_points) — reshape using the same dimensions
        passed to :func:`set_eta_scan_freq`.

        Args
        ----
        band : int
            Which band.
        count : int
            Number of samples to read from the results array.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        numpy.ndarray
            Frequency error array of length `count`.

        See Also
        --------
        :func:`get_eta_scan_results_imag` : Imaginary (Q) component.
        :func:`set_run_serial_find_freq` : Multi-channel scan that populates this.
        :func:`set_run_eta_scan` : Single-channel scan that populates this.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_results_real_reg,
            count=count, **kwargs)

    _eta_scan_results_imag_reg = 'etaScanResultsImag'

    def get_eta_scan_results_imag(self, band, count, **kwargs):
        r"""Gets the imaginary component of the eta scan.

        Returns frequency error measured during the second sweep of
        the most recent :func:`set_run_eta_scan` or
        :func:`set_run_serial_find_freq`.  Treated as Q when
        constructing the complex response as ``I + 1j*Q``.

        For serial find-freq, the array is flattened from shape
        (n_channels, n_scan_points) — reshape using the same dimensions
        passed to :func:`set_eta_scan_freq`.

        Args
        ----
        band : int
            Which band.
        count : int
            Number of samples to read from the results array.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        numpy.ndarray
            Frequency error array of length `count`.

        See Also
        --------
        :func:`get_eta_scan_results_real` : Real (I) component.
        :func:`set_run_serial_find_freq` : Multi-channel scan that populates this.
        :func:`set_run_eta_scan` : Single-channel scan that populates this.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_scan_results_imag_reg,
            count=count, **kwargs)


    _amplitude_scale_array_reg = 'amplitudeScale'

    def set_amplitude_scale_array(self, band, val, **kwargs):
        r"""Sets the tone amplitude for all channels in a band.

        Writes the full array of per-channel amplitudes to firmware
        BRAM. Each value controls the drive power for that channel.
        Each step is 3 dB. Channels with amplitude 0 output no tone
        and are not processed by firmware.

        Args
        ----
        band : int
            Which band.
        val : array-like
            Array of tone amplitudes, one per channel. 4-bit
            unsigned (0-15) per element.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_amplitude_scale_array` : Gets all channel amplitudes.
        :func:`set_amplitude_scale_channel` : Sets one channel.
        """
        self._caput(
            self._cryo_root(band) + self._amplitude_scale_array_reg,
            np.array(val).astype(np.uint32), **kwargs)

    def get_amplitude_scale_array(self, band, **kwargs):
        r"""Gets the tone amplitude for all channels in a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of tone amplitudes, one per channel. 4-bit
            unsigned (0-15) per element.

        See Also
        --------
        :func:`set_amplitude_scale_array` : Sets all channel amplitudes.
        :func:`get_amplitude_scale_channel` : Gets one channel.
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
        new_amp = np.zeros((n_channels,),dtype=np.uint32)
        new_amp[np.where(old_amp!=0)] = tone_power
        self.set_amplitude_scale_array(self, new_amp, **kwargs)

    _feedback_enable_array_reg = 'feedbackEnable'

    def set_feedback_enable_array(self, band, val, **kwargs):
        r"""Sets the per-channel feedback enable for all channels in a band.

        Writes an array of per-channel feedback enables to firmware
        BRAM. Each channel must have its individual feedback enabled
        here AND the global feedback enable must be set via
        :func:`set_feedback_enable` for tracking to be active on
        that channel.

        Args
        ----
        band : int
            Which band.
        val : array-like
            Array of feedback enable values, one per channel.
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_enable_array` : Gets all per-channel enables.
        :func:`set_feedback_enable` : Sets the global feedback enable.
        :func:`set_feedback_enable_channel` : Sets a single channel's enable.
        """
        self._caput(
            self._cryo_root(band) + self._feedback_enable_array_reg,
            val, **kwargs)

    def get_feedback_enable_array(self, band, **kwargs):
        r"""Gets the per-channel feedback enable for all channels in a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int array
            Array of feedback enable values, one per channel.
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_feedback_enable_array` : Sets all per-channel enables.
        :func:`get_feedback_enable` : Gets the global feedback enable.
        """
        return self._caget(
            self._cryo_root(band) + self._feedback_enable_array_reg,
            **kwargs)

    _single_channel_readout_reg = 'singleChannelReadout'

    def set_single_channel_readout(self, band, val, **kwargs):
        r"""Enables filtered/decimated single-channel debug readout.

        When enabled, debug data outputs only the channel specified
        by :func:`set_readout_channel_select`, passed through the
        IIR filter (``filterAlpha``) and decimated. When disabled,
        multichannel debug data is output. For single-channel
        readout at the full channel processing rate without
        filtering or decimation, use
        :func:`set_single_channel_readout_opt2` instead.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable single-channel mode, 0 for multichannel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_single_channel_readout` : Gets the current state.
        :func:`set_readout_channel_select` : Selects which channel.
        :func:`set_single_channel_readout_opt2` : Full-rate single-channel mode.
        :func:`take_debug_data` : Takes debug data.
        """
        self._caput(
            self._band_root(band) + self._single_channel_readout_reg,
            val, **kwargs)

    def get_single_channel_readout(self, band, **kwargs):
        r"""Gets the filtered/decimated single-channel debug readout state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if single-channel mode is enabled, 0 for multichannel.

        See Also
        --------
        :func:`set_single_channel_readout` : Sets this mode.
        """
        return self._caget(
            self._band_root(band) + self._single_channel_readout_reg,
            **kwargs)

    _single_channel_readout2_reg = 'singleChannelReadoutOpt2'

    def set_single_channel_readout_opt2(self, band, val, **kwargs):
        r"""Enables non-decimated single-channel debug readout.

        When enabled, debug data outputs the selected channel at
        the full channel processing rate (see
        :func:`get_channel_frequency_mhz`, default 2.4 MHz),
        bypassing the IIR filter and decimation. Channel is
        selected by :func:`set_readout_channel_select`.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable full-rate single-channel mode, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_single_channel_readout_opt2` : Gets the current state.
        :func:`set_single_channel_readout` : Filtered/decimated mode.
        :func:`set_readout_channel_select` : Selects which channel.
        :func:`take_debug_data` : Takes debug data.
        """
        self._caput(
            self._band_root(band) + self._single_channel_readout2_reg,
            val, **kwargs)

    def get_single_channel_readout_opt2(self, band, **kwargs):
        r"""Gets the non-decimated single-channel debug readout state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if full-rate single-channel mode is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_single_channel_readout_opt2` : Sets this mode.
        """
        return self._caget(
            self._band_root(band) + self._single_channel_readout2_reg,
            **kwargs)

    _readout_channel_select_reg = 'readoutChannelSelect'

    def set_readout_channel_select(self, band, channel, **kwargs):
        r"""Selects which channel to output in single-channel debug mode.

        Only used when single-channel readout is enabled via
        :func:`set_single_channel_readout`. Not all channel indices
        produce valid data; use :func:`which_on` to find channels
        with nonzero tone power, and :func:`get_processed_channels`
        to find which channel indices are processed by the
        channelizer.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel. Range is 0 to
            :func:`get_number_channels` - 1 (default firmware
            has 512 channels per band).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_readout_channel_select` : Gets the selected channel.
        :func:`set_single_channel_readout` : Enables single-channel debug mode.
        :func:`which_on` : Returns channels with nonzero amplitude.
        :func:`get_processed_channels` : Returns valid channel indices.
        """
        self._caput(
            self._band_root(band) + self._readout_channel_select_reg,
            channel, **kwargs)

    def get_readout_channel_select(self, band, **kwargs):
        r"""Gets the channel selected for single-channel debug mode.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        channel : int
            The currently selected channel.

        See Also
        --------
        :func:`set_readout_channel_select` : Sets the selected channel.
        """
        return self._caget(
            self._band_root(band) + self._readout_channel_select_reg,
            **kwargs)

    _stream_enable_reg = 'enableStreaming'

    def set_stream_enable(self, val, **kwargs):
        r"""Enables or disables streaming data output for all bands.

        Master enable for the data streaming path. When enabled,
        firmware generates a data frame on each flux ramp trigger
        containing the demodulated channel data. When disabled,
        flux ramp triggers are ignored and no streaming frames
        are produced.

        Args
        ----
        val : int
            1 to enable streaming, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_stream_enable` : Gets the current state.
        :func:`take_stream_data` : Takes streaming data for a duration.
        """
        self._caput(self.app_core + self._stream_enable_reg, val, **kwargs)

    def get_stream_enable(self, **kwargs):
        r"""Gets the streaming data output enable state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if streaming is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_stream_enable` : Sets the enable state.
        """
        return self._caget(
            self.app_core + self._stream_enable_reg,
            **kwargs)

    _mode_stream_reg = 'modeStream'

    def set_mode_stream(self, val, **kwargs):
        r"""Set the mode for data streaming.

        This function sets the mode for data streaming. If the mode is
        set to 0 (the default mode), the demodulated flux ramp phase
        will be streamed for each channel. If the mode is set to 1,
        the raw RF I and Q values will be streamed for each channel.

        When the mode is set to 1 (I/Q streaming mode), the
        `baySelStream` register, which can be set using the
        :func:`set_bay_sel_stream` routine, determines which bay's
        I/Q data is streamed. If `baySelStream` is 0, the I/Q data
        from bay 0 is streamed, and if it is 1, the I/Q data from bay
        1 is streamed.

        In I/Q streaming mode (modeSelStream=1), the I/Q data is
        truncated by a separate register, `lmsGain`, which can be set
        and retrieved using the `set_lms_gain` and `get_lms_gain`
        functions.

        When using the :func:`take_stream_data` and the
        :func:`read_stream_data` routines in I/Q streaming mode
        (modeSelStream=1), the `IQ_mode` input keyword must be set to
        `True` to account for the different channel mapping.

        Args
        ----
        val : int
            The mode to set, either 0 (default mode) or 1 (I/Q
            streaming mode).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_mode_stream` : Get the current data streaming mode.
        :func:`get_lms_gain` : Gets the current value of the `lmsGain` register.
        :func:`set_lms_gain` : Sets the value of the `lmsGain` register.
        :func:`read_stream_data` : Loads data taken with the `take_stream_data` function.
        :func:`take_stream_data` : Takes streaming data for a given amount of time.
        :func:`set_bay_sel_stream` : Set the bay selection for I/Q data streaming.
        :func:`get_bay_sel_stream` : Get the current bay selection for I/Q data streaming.
        """
        self._caput(self.app_core + self._mode_stream_reg, val, **kwargs)


    def get_mode_stream(self, **kwargs):
        r"""Get the current data streaming mode.

        This function returns the current mode for data streaming. If
        the mode is 0, the demodulated flux ramp phase is being
        streamed for each channel. If the mode is 1, the raw RF I and
        Q values are being streamed for each channel.

        See the docstring for :func:`set_mode_stream` for more
        details.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            The current mode for data streaming, either 0 (default
            mode) or 1 (I/Q streaming mode).

        See Also
        --------
        :func:`set_mode_stream` : Set the data streaming mode.
        :func:`get_lms_gain` : Gets the current value of the `lmsGain` register.
        :func:`set_lms_gain` : Sets the value of the `lmsGain` register.
        :func:`read_stream_data` : Loads data taken with the `take_stream_data` function.
        :func:`take_stream_data` : Takes streaming data for a given amount of time.
        :func:`set_bay_sel_stream` : Set the bay selection for I/Q data streaming.
        :func:`get_bay_sel_stream` : Get the current bay selection for I/Q data streaming.
        """
        return self._caget(
            self.app_core + self._mode_stream_reg,
            **kwargs)

    _bay_sel_stream_reg = 'baySelStream'

    def set_bay_sel_stream(self, val, **kwargs):
        r"""Set the bay selection for I/Q data streaming.

        This function sets the bay selection for I/Q data
        streaming. When the `modeStream` register (set with
        :func:`set_mode_stream` routine) is set to 1 (I/Q streaming
        mode), this parameter determines which bay's I/Q data is
        streamed.

        If set to 0, the I/Q data from bay 0 is streamed, and if it is
        1, the I/Q data from bay 1 is streamed instead.

        Args
        ----
        val : int
            The AMC bay to select for I/Q data streaming, either 0 or 1.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_bay_sel_stream` : Get the current bay selection for I/Q data streaming.
        :func:`get_mode_stream` : Get the current data streaming mode.
        :func:`set_mode_stream` : Set the data streaming mode.
        """
        self._caput(self.app_core + self._bay_sel_stream_reg, val, **kwargs)

    def get_bay_sel_stream(self, **kwargs):
        r"""Get the current bay selection for I/Q data streaming.

        This function returns the current bay selection for I/Q data
        streaming.  When the `modeStream` register (set with
        :func:`set_mode_stream` routine) is set to 1 (I/Q streaming
        mode), this parameter determines which bay's I/Q data is
        streamed.

        If the returned value is 0, the I/Q data from bay 0 is being
        streamed, and if it is 1, the I/Q data from bay 1 is being
        streamed.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val: int
            The current bay selection for I/Q data streaming, either 0 or 1.

        See Also
        --------
        :func:`set_bay_sel_stream` : Set the bay selection for I/Q data streaming.
        :func:`get_mode_stream` : Get the current data streaming mode.
        :func:`set_mode_stream` : Set the data streaming mode.
        """
        return self._caget(
            self.app_core + self._bay_sel_stream_reg,
            **kwargs)

    _rf_iq_stream_enable_reg = 'rfIQStreamEnable'

    def set_rf_iq_stream_enable(self, band, val, **kwargs):
        r"""Selects raw RF I/Q debug output from the analysis filter bank.

        When enabled (val=1), the debug readout path returns the raw I/Q
        output from the analysis filter bank (digital downconverter),
        before any tracking or flux ramp demodulation.  This and
        :func:`set_iq_stream_enable` are two bits of the same mux select
        in firmware — do not enable both simultaneously.  Used with
        :func:`take_debug_data` for diagnostics like measuring ADC power
        or verifying tone placement.

        Args
        ----
        band : int
            Which band.
        val : int
            0 to disable, 1 to select raw RF I/Q output.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_iq_stream_enable` : Selects demodulated I/Q instead.
        :func:`take_debug_data` : Takes data in the selected mode.
        """
        self._caput(self._band_root(band) +
                    self._rf_iq_stream_enable_reg,
                    val, **kwargs)

    def get_rf_iq_stream_enable(self, band, **kwargs):
        r"""Gets the raw RF I/Q debug output enable state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            0 for disabled, 1 for raw RF I/Q output enabled.

        See Also
        --------
        :func:`set_rf_iq_stream_enable` : Sets this value.
        """
        return self._caget(self._band_root(band) +
                           self._rf_iq_stream_enable_reg,
                           **kwargs)


    _build_dsp_g_reg = 'BUILD_DSP_G'

    def get_build_dsp_g(self, **kwargs):
        r"""Gets the firmware band bitmask.

        BUILD_DSP_G encodes which bands the firmware was built for.
        Each bit corresponds to one band (Base[n]).  For example, 0xFF
        means bands 0–7 are present, 0xF means bands 0–3.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Bitmask of available bands in this firmware build.
        """
        return self._caget(
            self.app_core + self._build_dsp_g_reg,
            **kwargs)

    _decimation_reg = 'decimation'

    def set_decimation(self, band, val, **kwargs):
        r"""Sets the debug data decimation factor.

        Direct divisor for the debug data output rate.  The debug
        output rate is approximately
        :func:`get_channel_frequency_mhz` / 2 / decimation MHz.
        Minimum value is 1 (no decimation).
        Applied after the IIR filter set by :func:`set_filter_alpha`.

        Args
        ----
        band : int
            Which band.
        val : int
            Decimation factor (15-bit unsigned, minimum 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_decimation` : Gets this value.
        :func:`set_filter_alpha` : IIR filter applied before decimation.
        :func:`take_debug_data` : Takes data using this path.
        """
        self._caput(
            self._band_root(band) + self._decimation_reg,
            val, **kwargs)

    def get_decimation(self, band, **kwargs):
        r"""Gets the debug data decimation factor.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Decimation factor (15-bit unsigned, minimum 1).

        See Also
        --------
        :func:`set_decimation` : Sets this value.
        """
        return self._caget(
            self._band_root(band) + self._decimation_reg,
            **kwargs)

    _filter_alpha_reg = 'filterAlpha'

    def set_filter_alpha(self, band, val, **kwargs):
        r"""Sets the IIR low-pass filter coefficient for debug data.

        Applies a single-pole IIR low-pass filter to each channel's
        debug data before decimation:
        y[n] = alpha * x[n] + (1 - alpha) * y[n-1].
        Larger alpha values give less filtering (higher bandwidth).
        This filter is applied to multichannel debug readout and
        single-channel readout (Opt1). It is NOT applied to
        singleChannelReadoutOpt2 (which picks off before the filter
        at the full channel processing rate) or to the main streamed
        data path. Most users should use
        :func:`set_debug_data_filter_cutoff` to set this by desired
        cutoff frequency in Hz.

        To visualize the filter response in Python::

            from scipy.signal import freqz
            import matplotlib.pyplot as plt
            alpha = val / 65536
            w, h = freqz([alpha], [1, -(1 - alpha)], fs=2.4e6)
            plt.semilogy(w, abs(h))

        Args
        ----
        band : int
            Which band.
        val : int
            Filter coefficient. 16-bit unsigned integer (0-65535).
            val/65536 gives the effective alpha (e.g. 0x4000 gives
            alpha=0.25, f3dB ~110 kHz; 0x10000 would give alpha=1.0
            but saturates at 0xFFFF).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_filter_alpha` : Gets the current filter coefficient.
        :func:`set_debug_data_filter_cutoff` : Sets filter by cutoff frequency.
        :func:`set_decimation` : Sets the decimation after filtering.
        :func:`set_single_channel_readout_opt2` : Bypasses this filter.
        :func:`take_debug_data` : Takes data using this filter/decimation path.
        """
        self._caput(
            self._band_root(band) + self._filter_alpha_reg,
            val, **kwargs)

    def get_filter_alpha(self, band, **kwargs):
        r"""Gets the IIR low-pass filter coefficient for debug data.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Filter coefficient. 16-bit unsigned integer (0-65535).
            val/65536 gives the effective alpha.

        See Also
        --------
        :func:`set_filter_alpha` : Sets the filter coefficient.
        :func:`get_debug_data_filter_cutoff` : Gets the cutoff frequency in Hz.
        """
        return self._caget(
            self._band_root(band) + self._filter_alpha_reg,
            **kwargs)

    _iq_swap_in_reg = 'iqSwapIn'

    def set_iq_swap_in(self, band, val, **kwargs):
        r"""Swaps I and Q on the analysis filter bank input.

        Swapping I and Q flips the input spectrum around the band center
        frequency.  Used to correct for hardware-dependent sideband
        conventions.  Set per-band in the pysmurf configuration file.

        Args
        ----
        band : int
            Which band.
        val : int
            0 for normal, 1 for swapped.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_iq_swap_out` : Swaps I/Q on the synthesis output.
        """
        self._caput(
            self._band_root(band) + self._iq_swap_in_reg,
            val, **kwargs)

    def get_iq_swap_in(self, band, **kwargs):
        r"""Gets the I/Q swap state on the analysis filter bank input.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            0 for normal, 1 for swapped.

        See Also
        --------
        :func:`set_iq_swap_in` : Sets this value.
        """
        return self._caget(
            self._band_root(band) + self._iq_swap_in_reg,
            **kwargs)

    _iq_swap_out_reg = 'iqSwapOut'

    def set_iq_swap_out(self, band, val, **kwargs):
        r"""Swaps I and Q on the synthesis filter bank output.

        Swapping I and Q flips the output spectrum around the band
        center frequency. Used to correct for hardware-dependent
        sideband conventions.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to swap, 0 for normal.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_iq_swap_out` : Gets the current state.
        :func:`set_iq_swap_in` : Swaps I/Q on the analysis input.
        """
        self._caput(
            self._band_root(band) + self._iq_swap_out_reg,
            val, **kwargs)

    def get_iq_swap_out(self, band, **kwargs):
        r"""Gets the I/Q swap state on the synthesis filter bank output.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if swapped, 0 if normal.

        See Also
        --------
        :func:`set_iq_swap_out` : Sets the swap state.
        """
        return self._caget(
            self._band_root(band) + self._iq_swap_out_reg,
            **kwargs)

    _ref_phase_delay_reg = 'refPhaseDelay'

    def set_ref_phase_delay(self, band, val, **kwargs):
        r"""Sets the coarse reference phase delay.

        Compensates for system round-trip latency at coarse
        resolution (channel processing rate ticks, see
        :func:`get_channel_frequency_mhz`, default 2.4 MHz).
        ``refPhaseDelayFine`` adds sub-tick correction at the JESD
        clock rate (see :func:`get_digitizer_frequency_mhz` / 2,
        default 307.2 MHz), and ``lmsDelay`` (set to the same
        value) aligns
        the LMS tracking loop separately. Most users should call
        :func:`estimate_phase_delay`, which measures and sets all
        three automatically.

        Args
        ----
        band : int
            Which band.
        val : int
            Coarse delay value.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_ref_phase_delay` : Gets the current value.
        :func:`set_ref_phase_delay_fine` : Sets the fine correction.
        :func:`set_band_delay_us` : Sets all delay registers together.
        :func:`estimate_phase_delay` : Measures and sets system latency.
        """
        self._caput(
            self._band_root(band) + self._ref_phase_delay_reg,
            val, **kwargs)

    def get_ref_phase_delay(self, band, **kwargs):
        r"""Gets the coarse reference phase delay.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Coarse delay in channel processing rate ticks.

        See Also
        --------
        :func:`set_ref_phase_delay` : Sets the value.
        :func:`get_band_delay_us` : Gets the total delay in microseconds.
        """
        return self._caget(
            self._band_root(band) + self._ref_phase_delay_reg,
            **kwargs)

    _ref_phase_delay_fine_reg = 'refPhaseDelayFine'

    def set_ref_phase_delay_fine(self, band, val, **kwargs):
        r"""Sets the fine reference phase delay.

        Fine correction to the DAC output timing at the JESD clock
        rate (see :func:`get_digitizer_frequency_mhz` / 2, default
        307.2 MHz, ~3.25 ns steps). Compensates for the rounding
        of ``refPhaseDelay`` to the coarser channel processing rate
        grid. Most users should call :func:`estimate_phase_delay`,
        which measures and sets this automatically.

        Args
        ----
        band : int
            Which band.
        val : int
            Fine delay value (8-bit unsigned).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_ref_phase_delay_fine` : Gets the current value.
        :func:`set_ref_phase_delay` : Sets the coarse delay.
        :func:`estimate_phase_delay` : Measures and sets system latency.
        """
        self._caput(
            self._band_root(band) + self._ref_phase_delay_fine_reg,
            val, **kwargs)

    def get_ref_phase_delay_fine(self, band, **kwargs):
        r"""Gets the fine reference phase delay.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Fine delay in JESD clock ticks
            (:func:`get_digitizer_frequency_mhz` / 2, default
            307.2 MHz).

        See Also
        --------
        :func:`set_ref_phase_delay_fine` : Sets the value.
        :func:`get_band_delay_us` : Gets the total delay in microseconds.
        """
        return self._caget(
            self._band_root(band) + self._ref_phase_delay_fine_reg,
            **kwargs)

    _band_delay_us_reg = 'bandDelayUs'

    def set_band_delay_us(self, band, val, **kwargs):
        r"""Sets the total band delay compensation in microseconds.

        Configures ``refPhaseDelay``, ``refPhaseDelayFine``, and
        ``lmsDelay`` together to compensate for the system
        round-trip latency. Most users should call
        :func:`estimate_phase_delay`, which measures the delay
        and calls this function automatically.

        Args
        ----
        band : int
            Which band.
        val : float
            Delay in microseconds.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_band_delay_us` : Gets the current value.
        :func:`estimate_phase_delay` : Measures and sets the delay.
        :func:`set_ref_phase_delay` : Sets coarse delay directly.
        :func:`set_ref_phase_delay_fine` : Sets fine delay directly.
        """
        self._caput(
            self._band_root(band) + self._band_delay_us_reg,
            val, **kwargs)

    def get_band_delay_us(self, band, **kwargs):
        r"""Gets the total band delay compensation in microseconds.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Delay in microseconds.

        See Also
        --------
        :func:`set_band_delay_us` : Sets the value.
        :func:`estimate_phase_delay` : Measures and sets the delay.
        """
        return self._caget(
            self._band_root(band) + self._band_delay_us_reg,
            **kwargs)

    _tone_scale_reg = 'toneScale'

    def set_tone_scale(self, band, val, **kwargs):
        r"""Sets the tone output scaling before the synthesis filter bank.

        Scales the combined tone output before it enters the
        synthesis filter bank. Each increment doubles the output
        amplitude.

        Args
        ----
        band : int
            Which band.
        val : int
            Scale factor. 2-bit unsigned (0-3).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_tone_scale` : Gets the current value.
        :func:`set_synthesis_scale` : Scales the synthesis filter bank output.
        """
        self._caput(
            self._band_root(band) + self._tone_scale_reg,
            val, **kwargs)

    def get_tone_scale(self, band, **kwargs):
        r"""Gets the tone output scaling before the synthesis filter bank.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Scale factor. 2-bit unsigned (0-3). Each increment
            doubles the output amplitude.

        See Also
        --------
        :func:`set_tone_scale` : Sets the value.
        """
        return self._caget(
            self._band_root(band) + self._tone_scale_reg,
            **kwargs)

    _waveform_select_reg = 'waveformSelect'

    def set_waveform_select(self, band, val, **kwargs):
        r"""Selects the DAC output source for a band.

        When set to 0, the DAC outputs the normal DSP synthesis path.
        When set to 1, the DAC outputs from the preloaded waveform table
        (tone file loaded via :func:`load_tone_file`).

        Args
        ----
        band : int
            Which band.
        val : int
            0 for DSP synthesis, 1 for waveform table (tone file).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_tone_file_path` : Sets the path to the tone file CSV.
        :func:`load_tone_file` : Loads the tone file into the waveform table.
        :func:`set_noise_select` : Selects random noise output instead.
        """
        self._caput(
            self._band_root(band) + self._waveform_select_reg,
            val, **kwargs)

    def get_waveform_select(self, band, **kwargs):
        r"""Gets the DAC output source selection for a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            0 for DSP synthesis, 1 for waveform table (tone file).

        See Also
        --------
        :func:`set_waveform_select` : Sets this value.
        """
        return self._caget(
            self._band_root(band) + self._waveform_select_reg,
            **kwargs)

    _rf_enable_reg = 'rfEnable'

    def set_rf_enable(self, band, val, **kwargs):
        r"""Enables or disables RF DAC output for a band.

        When set to 0, the DAC outputs all zeros regardless of the DSP
        or waveform table state.  When set to 1, the DAC outputs from
        whichever source is selected (:func:`set_waveform_select`,
        :func:`set_noise_select`, or DSP synthesis).

        Args
        ----
        band : int
            Which band.
        val : int
            0 to disable (output zeros), 1 to enable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_waveform_select` : Selects DSP vs. tone file output.
        :func:`set_noise_select` : Selects random noise output.
        """
        self._caput(
            self._band_root(band) + self._rf_enable_reg,
            val, **kwargs)

    def get_rf_enable(self, band, **kwargs):
        r"""Gets the RF DAC output enable state for a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            0 for disabled (output zeros), 1 for enabled.

        See Also
        --------
        :func:`set_rf_enable` : Sets this value.
        """
        return self._caget(
            self._band_root(band) + self._rf_enable_reg,
            **kwargs)

    _analysis_scale_reg = 'analysisScale'

    def set_analysis_scale(self, band, val, **kwargs):
        r"""Sets the analysis filter bank output scaling.

        Controls the output amplitude of the polyphase analysis
        (channelizer) filter bank. Each increment doubles the output amplitude. Too low risks overflow (check
        with the overflow status); too high loses dynamic range.
        Nominal value is 1.

        Args
        ----
        band : int
            Which band.
        val : int
            Scale factor. 2-bit unsigned (0-3). Each increment
            is a factor of 2. Nominal is 1.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_analysis_scale` : Gets the current analysis scale.
        :func:`set_synthesis_scale` : Sets the synthesis filter bank scaling.
        """
        self._caput(
            self._band_root(band) + self._analysis_scale_reg,
            val, **kwargs)

    def get_analysis_scale(self, band, **kwargs):
        r"""Gets the analysis filter bank output scaling.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Scale factor. 2-bit unsigned (0-3). Each increment
            is a factor of 2. Nominal is 1.

        See Also
        --------
        :func:`set_analysis_scale` : Sets the analysis scale.
        """
        return self._caget(
            self._band_root(band) + self._analysis_scale_reg,
            **kwargs)

    _feedback_enable_reg = 'feedbackEnable'

    def set_feedback_enable(self, band, val, **kwargs):
        r"""Sets the global feedback enable for a band.

        When enabled (val=1), the tone-tracking loop applies frequency
        corrections to all channels that also have per-channel feedback
        enabled. When disabled (val=0), no tracking corrections are
        applied regardless of per-channel settings.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable global feedback, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_enable` : Gets the global feedback enable state.
        :func:`set_feedback_enable_channel` : Sets per-channel feedback enable.
        """
        self._caput(
            self._band_root(band) + self._feedback_enable_reg,
            val, **kwargs)

    def get_feedback_enable(self, band, **kwargs):
        r"""Gets the global feedback enable for a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if global feedback is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_feedback_enable` : Sets the global feedback enable.
        """
        return self._caget(
            self._band_root(band) + self._feedback_enable_reg,
            **kwargs)

    _loop_filter_output_array_reg = 'loopFilterOutput'

    def get_loop_filter_output_array(self, band, **kwargs):
        r"""Gets the loop filter output for all channels in a band.

        Returns the accumulated frequency correction for each
        channel. This is the integrated feedback that is added to
        each tone's center frequency during tracking. Read-only
        from firmware BRAM.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of loop filter outputs, one per channel.

        See Also
        --------
        :func:`set_feedback_gain` : Sets the gain that drives this output.
        :func:`set_feedback_limit` : Limits this output's excursion.
        :func:`get_frequency_error_array` : Gets the frequency error input.
        """
        return self._caget(
            self._cryo_root(band) +
            self._loop_filter_output_array_reg,
            **kwargs)

    _tone_frequency_offset_mhz_reg = 'toneFrequencyOffsetMHz'

    def get_tone_frequency_offset_mhz(self, band, **kwargs):
        r"""Gets the subband center frequency offsets in MHz.

        Returns an array of precomputed frequency offsets from the
        band center for each subband. These are fixed by the
        channelizer architecture and are not user-configurable.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of subband center frequency offsets in MHz,
            relative to the band center.

        See Also
        --------
        :func:`get_band_center_mhz` : Gets the absolute band center frequency.
        """
        return self._caget(
            self._band_root(band) +
            self._tone_frequency_offset_mhz_reg,
            **kwargs)

    _center_frequency_array_reg = 'centerFrequencyMHz'

    def set_center_frequency_array(self, band, val, **kwargs):
        r"""Sets the tone center frequency for all channels in a band.

        Each value is the frequency offset from the band center at
        which that channel's tone is placed. This is the static
        tone position; the tracking loop adds corrections on top
        of this via the loop filter output. Typically set by
        :func:`setup_notches`.

        Args
        ----
        band : int
            Which band.
        val : array-like
            Array of center frequencies in MHz, one per channel.
            Range is +/-1.2 MHz (the half-bandwidth of one
            subband).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_center_frequency_array` : Gets all center frequencies.
        :func:`set_center_frequency_mhz_channel` : Sets one channel.
        :func:`setup_notches` : Measures and sets tone positions.
        """
        self._caput(
            self._cryo_root(band) + self._center_frequency_array_reg,
            val, **kwargs)

    def get_center_frequency_array(self, band, **kwargs):
        r"""Gets the tone center frequency for all channels in a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of center frequencies in MHz, one per channel.

        See Also
        --------
        :func:`set_center_frequency_array` : Sets all center frequencies.
        :func:`get_center_frequency_mhz_channel` : Gets one channel.
        """
        return self._caget(
            self._cryo_root(band) + self._center_frequency_array_reg,
            **kwargs)

    _feedback_gain_reg = 'feedbackGain'

    def set_feedback_gain(self, band, val, **kwargs):
        r"""Sets the integral gain of the tracking feedback loop.

        This gain scales the frequency error before it is accumulated
        into the loop filter output that adjusts each tone's center
        frequency. Higher values increase loop bandwidth but reduce
        stability margin. Distinct from ``lmsGain``, which controls
        how quickly the LMS flux ramp harmonic estimator adapts;
        ``feedbackGain`` controls how aggressively tone frequencies
        are corrected based on the measured error.

        Args
        ----
        band : int
            Which band.
        val : int
            Feedback gain. 16-bit unsigned integer (0-65535).
            val/4096 gives the effective gain multiplier.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_gain` : Gets the current feedback gain.
        :func:`set_feedback_limit` : Sets the maximum feedback excursion.
        :func:`set_lms_gain` : Sets the LMS harmonic estimator step size.
        """
        self._caput(
            self._band_root(band) + self._feedback_gain_reg,
            val, **kwargs)

    def get_feedback_gain(self, band, **kwargs):
        r"""Gets the integral gain of the tracking feedback loop.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Feedback gain. 16-bit unsigned integer (0-65535).
            val/4096 gives the effective gain multiplier.

        See Also
        --------
        :func:`set_feedback_gain` : Sets the feedback gain.
        """
        return self._caget(
            self._band_root(band) + self._feedback_gain_reg,
            **kwargs)

    _eta_phase_array_reg = 'etaPhase'

    def set_eta_phase_array(self, band, val, **kwargs):
        r"""Sets the eta phase for all channels in a band.

        Eta is the complex calibration parameter that rotates the
        measured resonator I/Q response so that frequency detuning
        appears as a single-axis signal. It is stored in firmware
        as Cartesian components (etaI, etaQ). Setting the phase
        preserves the current magnitude and recomputes etaI and
        etaQ as mag * cos(phase) and mag * sin(phase). Typically
        determined by :func:`setup_notches`, which calls
        :func:`run_serial_eta_scan` internally.

        Args
        ----
        band : int
            Which band.
        val : array-like
            Array of eta phases in radians, one per channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_eta_phase_array` : Gets the current eta phases.
        :func:`set_eta_mag_array` : Sets the eta magnitudes for all channels.
        :func:`setup_notches` : Measures and sets eta parameters.
        """
        self._caput(
            self._cryo_root(band) + self._eta_phase_array_reg,
            val, **kwargs)

    def get_eta_phase_array(self, band, **kwargs):
        r"""Gets the eta phase for all channels in a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of eta phases in radians, one per channel.

        See Also
        --------
        :func:`set_eta_phase_array` : Sets the eta phases.
        :func:`get_eta_mag_array` : Gets the eta magnitudes.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_phase_array_reg,
            **kwargs)

    _frequency_error_array_reg = 'frequencyError'

    def get_frequency_error_array(self, band, **kwargs):
        r"""Gets the frequency error for all channels in a band.

        Returns the measured detuning of each tone from its
        resonance after eta rotation. This is the input to the
        tracking loop filter. Read-only from firmware BRAM.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of frequency errors, one per channel.

        See Also
        --------
        :func:`get_loop_filter_output_array` : Gets the integrated feedback.
        :func:`get_frequency_error_mhz` : Gets a single channel's error in MHz.
        """
        return self._caget(
            self._cryo_root(band) + self._frequency_error_array_reg,
            **kwargs)

    _eta_mag_array_reg = 'etaMag'

    def set_eta_mag_array(self, band, val, **kwargs):
        r"""Sets the eta magnitude for all channels in a band.

        Eta is the complex calibration parameter that rotates the
        measured resonator I/Q response so that frequency detuning
        appears as a single-axis signal. It is stored in firmware
        as Cartesian components (etaI, etaQ). Setting the magnitude
        preserves the current phase and recomputes etaI and etaQ as
        mag * cos(phase) and mag * sin(phase). Typically determined
        by :func:`setup_notches`, which calls
        :func:`run_serial_eta_scan` internally.

        Args
        ----
        band : int
            Which band.
        val : array-like
            Array of eta magnitudes (real, positive), one per
            channel. Maximum safe value is ~2.0 (limited by the
            underlying etaI/etaQ registers). Setting above
            ~2.0 may overflow one component depending on phase.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_eta_mag_array` : Gets the current eta magnitudes.
        :func:`set_eta_phase_array` : Sets the eta phases for all channels.
        :func:`setup_notches` : Measures and sets eta parameters.
        """
        self._caput(
            self._cryo_root(band) + self._eta_mag_array_reg,
            val, **kwargs)

    def get_eta_mag_array(self, band, **kwargs):
        r"""Gets the eta magnitude for all channels in a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : array
            Array of eta magnitudes (real, positive), one per
            channel.

        See Also
        --------
        :func:`set_eta_mag_array` : Sets the eta magnitudes.
        :func:`get_eta_phase_array` : Gets the eta phases.
        """
        return self._caget(
            self._cryo_root(band) + self._eta_mag_array_reg,
            **kwargs)

    _feedback_limit_reg = 'feedbackLimit'

    def set_feedback_limit(self, band, val, **kwargs):
        r"""Sets the maximum feedback excursion for tone tracking.

        Limits how far the loop filter output can shift a tone's
        frequency from its programmed center frequency. If the
        accumulated feedback would exceed this limit, it is clamped.
        This prevents the tracker from pulling tones too far off
        resonance during transients or instability.

        Args
        ----
        band : int
            Which band.
        val : int
            Maximum allowed feedback excursion. 16-bit unsigned
            integer (0-65535) representing a fraction of the
            band width. val/65536 gives the fraction of the
            full 2.4 MHz subband.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_limit` : Gets the current feedback limit.
        :func:`set_feedback_gain` : Sets the feedback loop gain.
        """
        self._caput(
            self._band_root(band) + self._feedback_limit_reg,
            val, **kwargs)

    def get_feedback_limit(self, band, **kwargs):
        r"""Gets the maximum feedback excursion for tone tracking.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Maximum allowed feedback excursion. 16-bit unsigned
            integer (0-65535). val/65536 gives the fraction of
            the full 2.4 MHz subband.

        See Also
        --------
        :func:`set_feedback_limit` : Sets the feedback limit.
        """
        return self._caget(
            self._band_root(band) + self._feedback_limit_reg,
            **kwargs)

    _noise_select_reg = 'noiseSelect'

    def set_noise_select(self, band, val, **kwargs):
        r"""Enables or disables random noise output on the RF DACs.

        When enabled, firmware replaces the normal resonator tracking
        tones with pseudo-random noise (uniformly distributed) on the
        RF output for this band. This outputs broadband noise across
        the 500 MHz band instead of discrete tones. The noise is
        generated digitally by an FPGA shift-register PRNG, not by
        the DAC hardware. Used for system diagnostics such as
        measuring the RF transfer function.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to output random noise, 0 for normal tone output.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_noise_select` : Gets the current noise select state.
        :func:`set_waveform_select` : Selects preloaded waveform output instead.
        :func:`set_rf_enable` : Enables/disables RF output entirely.
        """
        self._caput(
            self._band_root(band) + self._noise_select_reg,
            val, **kwargs)

    def get_noise_select(self, band, **kwargs):
        r"""Gets the random noise output state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if noise output is enabled, 0 for normal tone output.

        See Also
        --------
        :func:`set_noise_select` : Sets the noise select state.
        """
        return self._caget(
            self._band_root(band) + self._noise_select_reg,
            **kwargs)

    _lms_delay_reg = 'lmsDelay'

    def set_lms_delay(self, band, val, **kwargs):
        r"""Sets the LMS loop delay compensation.

        Compensates for the round-trip latency of the system
        (ADC through channelizer, DSP processing, and back through
        the synthesis filter bank to DAC) so that the feedback
        correction is applied at the correct phase of the flux ramp.
        Typically set equal to ``refPhaseDelay``. Most users should
        call :func:`estimate_phase_delay`, which measures the system
        latency and sets this register (along with ``refPhaseDelay``
        and ``refPhaseDelayFine``) automatically.

        Args
        ----
        band : int
            Which band.
        val : int
            Delay in channel processing rate ticks (see
            :func:`get_channel_frequency_mhz`, default 2.4 MHz).
            6-bit unsigned (0-63).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_delay` : Gets the current LMS delay.
        :func:`estimate_phase_delay` : Measures and sets the system latency.
        :func:`set_band_delay_us` : Sets all delay registers directly.
        """
        self._caput(
            self._band_root(band) + self._lms_delay_reg,
            val, **kwargs)

    def get_lms_delay(self, band, **kwargs):
        r"""Gets the LMS loop delay compensation.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Delay in channel processing rate ticks (see
            :func:`get_channel_frequency_mhz`, default 2.4 MHz).
            6-bit unsigned (0-63).

        See Also
        --------
        :func:`set_lms_delay` : Sets the LMS delay.
        :func:`estimate_phase_delay` : Measures and sets the system latency.
        """
        return self._caget(
            self._band_root(band) + self._lms_delay_reg,
            **kwargs)

    _lms_gain_reg = 'lmsGain'

    def set_lms_gain(self, band, val, **kwargs):
        r"""Sets the LMS tracking loop gain.

        Controls the step size of the LMS adaptive filter that
        estimates flux ramp harmonic coefficients. The gain is
        applied as a power-of-2 bit shift, so the effective gain
        is 2^val. Larger values make the harmonic estimator adapt
        faster but can cause instability.

        Args
        ----
        band : int
            Which band.
        val : int
            LMS gain exponent. 3-bit unsigned (0-7), giving an
            effective gain of 2^val.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_gain` : Gets the current LMS gain.
        :func:`set_feedback_gain` : Sets the integrator gain (distinct from LMS).
        :func:`set_lms_enable1` : Enables 1st harmonic tracking.
        """
        self._caput(
            self._band_root(band) + self._lms_gain_reg,
            val, **kwargs)

    def get_lms_gain(self, band, **kwargs):
        r"""Gets the LMS tracking loop gain.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            LMS gain exponent. 3-bit unsigned (0-7). Effective
            gain is 2^val.

        See Also
        --------
        :func:`set_lms_gain` : Sets the LMS gain.
        """
        return self._caget(
            self._band_root(band) + self._lms_gain_reg,
            **kwargs)

    _trigger_reset_delay_reg = 'trigRstDly'

    def set_trigger_reset_delay(self, band, val, **kwargs):
        r"""Sets the trigger reset delay for the flux ramp.

        Delay in channel processing clock ticks between the flux ramp
        reset trigger and the actual integrator reset.  Adjusted so that
        the reset occurs at the flux ramp glitch.  Units are ticks of the
        channel processing clock (see :func:`get_channel_frequency_mhz`,
        default 2.4 MHz).

        Args
        ----
        band : int
            Which band.
        val : int
            Delay in processing clock ticks.  7-bit unsigned (0–127).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput(
            self._band_root(band) + self._trigger_reset_delay_reg,
            val, **kwargs)

    def get_trigger_reset_delay(self, band, **kwargs):
        r"""Gets the trigger reset delay for the flux ramp.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Delay in processing clock ticks (default 2.4 MHz clock).

        See Also
        --------
        :func:`set_trigger_reset_delay` : Sets this value.
        """
        return self._caget(
            self._band_root(band) + self._trigger_reset_delay_reg,
            **kwargs)

    _feedback_start_reg = 'feedbackStart'

    def set_feedback_start(self, band, val, **kwargs):
        r"""Sets the sample count at which to start applying feedback.

        Defines the start of the feedback-active window within each
        flux ramp cycle. Firmware only applies tracking corrections
        between ``feedbackStart`` and ``feedbackEnd`` sample counts,
        allowing the transient at the flux ramp reset to be blanked.

        Args
        ----
        band : int
            Which band.
        val : int
            Start sample count (32-bit unsigned). Units are ticks of
            the channel processing rate (see
            :func:`get_channel_frequency_mhz`, default 2.4 MHz)
            from the beginning of each flux ramp cycle.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_start` : Gets the current feedback start count.
        :func:`set_feedback_end` : Sets the end of the feedback window.
        """
        self._caput(
            self._band_root(band) + self._feedback_start_reg,
            val, **kwargs)

    def get_feedback_start(self, band, **kwargs):
        r"""Gets the sample count at which feedback starts.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Start sample count. Units are ticks of the channel
            processing rate (see :func:`get_channel_frequency_mhz`,
            default 2.4 MHz) from the beginning of each flux ramp
            cycle.

        See Also
        --------
        :func:`set_feedback_start` : Sets the feedback start count.
        :func:`get_feedback_end` : Gets the end of the feedback window.
        """
        return self._caget(
            self._band_root(band) + self._feedback_start_reg,
            **kwargs)

    _feedback_end_reg = 'feedbackEnd'

    def set_feedback_end(self, band, val, **kwargs):
        r"""Sets the sample count at which to stop applying feedback.

        Defines the end of the feedback-active window within each
        flux ramp cycle. Firmware only applies tracking corrections
        between ``feedbackStart`` and ``feedbackEnd`` sample counts.

        Args
        ----
        band : int
            Which band.
        val : int
            End sample count (32-bit unsigned). Units are ticks of
            the channel processing rate (see
            :func:`get_channel_frequency_mhz`, default 2.4 MHz)
            from the beginning of each flux ramp cycle.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_end` : Gets the current feedback end count.
        :func:`set_feedback_start` : Sets the start of the feedback window.
        """
        self._caput(
            self._band_root(band) + self._feedback_end_reg,
            val, **kwargs)

    def get_feedback_end(self, band, **kwargs):
        r"""Gets the sample count at which feedback stops.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            End sample count. Units are ticks of the channel
            processing rate (see :func:`get_channel_frequency_mhz`,
            default 2.4 MHz) from the beginning of each flux ramp
            cycle.

        See Also
        --------
        :func:`set_feedback_end` : Sets the feedback end count.
        :func:`get_feedback_start` : Gets the start of the feedback window.
        """
        return self._caget(
            self._band_root(band) + self._feedback_end_reg,
            **kwargs)

    _lms_enable1_reg = 'lmsEnable1'

    def set_lms_enable1(self, band, val, **kwargs):
        r"""Enables or disables 1st harmonic tracking.

        When enabled, the LMS adaptive filter tracks the fundamental
        frequency of the flux ramp modulation. The fundamental
        frequency is set by :func:`set_lms_freq_hz`.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_enable1` : Gets the current state.
        :func:`set_lms_enable2` : Enables 2nd harmonic tracking.
        :func:`set_lms_enable3` : Enables 3rd harmonic tracking.
        :func:`set_lms_freq_hz` : Sets the fundamental tracking frequency.
        """
        self._caput(
            self._band_root(band) + self._lms_enable1_reg,
            val, **kwargs)

    def get_lms_enable1(self, band, **kwargs):
        r"""Gets the 1st harmonic tracking enable state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_lms_enable1` : Sets the 1st harmonic tracking enable.
        """
        return self._caget(
            self._band_root(band) + self._lms_enable1_reg,
            **kwargs)

    _lms_enable2_reg = 'lmsEnable2'

    def set_lms_enable2(self, band, val, **kwargs):
        r"""Enables or disables 2nd harmonic tracking.

        When enabled, the LMS adaptive filter tracks the 2nd harmonic
        (2x the fundamental frequency) of the flux ramp modulation.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_enable2` : Gets the current state.
        :func:`set_lms_enable1` : Enables 1st harmonic tracking.
        :func:`set_lms_enable3` : Enables 3rd harmonic tracking.
        """
        self._caput(
            self._band_root(band) + self._lms_enable2_reg,
            val, **kwargs),

    def get_lms_enable2(self, band, **kwargs):
        r"""Gets the 2nd harmonic tracking enable state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_lms_enable2` : Sets the 2nd harmonic tracking enable.
        """
        return self._caget(
            self._band_root(band) + self._lms_enable2_reg,
            **kwargs)

    _lms_enable3_reg = 'lmsEnable3'

    def set_lms_enable3(self, band, val, **kwargs):
        r"""Enables or disables 3rd harmonic tracking.

        When enabled, the LMS adaptive filter tracks the 3rd harmonic
        (3x the fundamental frequency) of the flux ramp modulation.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_enable3` : Gets the current state.
        :func:`set_lms_enable1` : Enables 1st harmonic tracking.
        :func:`set_lms_enable2` : Enables 2nd harmonic tracking.
        """
        self._caput(
            self._band_root(band) + self._lms_enable3_reg,
            val, **kwargs)

    def get_lms_enable3(self, band, **kwargs):
        r"""Gets the 3rd harmonic tracking enable state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_lms_enable3` : Sets the 3rd harmonic tracking enable.
        """
        return self._caget(
            self._band_root(band) + self._lms_enable3_reg,
            **kwargs)


    _lms_freq_reg = 'lmsFreq'

    def set_lms_freq(self, band, val, **kwargs):
        r"""Sets the LMS tracking frequency in raw firmware units.

        The LMS frequency is the fundamental frequency at which
        the tracker demodulates the flux ramp signal. It should
        equal the flux ramp rate times the number of flux quanta
        per ramp cycle (nPhi0). Most users should use
        :func:`set_lms_freq_hz` which accepts Hz directly.

        Args
        ----
        band : int
            Which band.
        val : int
            LMS frequency as a fraction of the channel processing
            rate (see :func:`get_channel_frequency_mhz`, default
            2.4 MHz). 24-bit unsigned (0 to 2^24-1).
            To convert to Hz:
            val * get_channel_frequency_mhz() * 1e6 / 2^24.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_freq` : Gets the current raw LMS frequency.
        :func:`set_lms_freq_hz` : Sets the LMS frequency in Hz.
        :func:`get_channel_frequency_mhz` : Gets the channel processing rate.
        """
        self._caput(
            self._band_root(band) + self._lms_freq_reg,
            val, **kwargs)

    def get_lms_freq(self, band, **kwargs):
        r"""Gets the LMS tracking frequency in raw firmware units.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            LMS frequency as a fraction of the channel processing
            rate (see :func:`get_channel_frequency_mhz`, default
            2.4 MHz). 24-bit unsigned. To convert to Hz:
            val * get_channel_frequency_mhz() * 1e6 / 2^24.

        See Also
        --------
        :func:`set_lms_freq` : Sets the raw LMS frequency.
        :func:`get_lms_freq_hz` : Gets the LMS frequency in Hz.
        """
        return self._caget(
            self._band_root(band) + self._lms_freq_reg,
            **kwargs)

    _lms_freq_hz_reg = 'lmsFreqHz'

    def set_lms_freq_hz(self, band, val, **kwargs):
        r"""Sets the LMS tracking frequency in Hz.

        The LMS frequency is the fundamental frequency at which
        the tracker demodulates the flux ramp signal. It should
        equal the flux ramp rate times the number of flux quanta
        per ramp cycle (nPhi0). This is a convenience wrapper
        that handles the conversion from Hz to raw firmware units.

        Args
        ----
        band : int
            Which band.
        val : float
            LMS frequency in Hz. Range is 0 to just under the
            channel processing rate (see
            :func:`get_channel_frequency_mhz`, default 2.4 MHz),
            with resolution of ~0.14 Hz. Typical values are in
            the kHz range.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lms_freq_hz` : Gets the LMS frequency in Hz.
        :func:`set_lms_freq` : Sets the LMS frequency in raw units.
        """
        self._caput(
            self._band_root(band) + self._lms_freq_hz_reg,
            val, **kwargs)

    def get_lms_freq_hz(self, band, **kwargs):
        r"""Gets the LMS tracking frequency in Hz.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            LMS frequency in Hz.

        See Also
        --------
        :func:`set_lms_freq_hz` : Sets the LMS frequency in Hz.
        :func:`get_lms_freq` : Gets the LMS frequency in raw units.
        """
        return self._caget(
            self._band_root(band) + self._lms_freq_hz_reg,
            **kwargs)


    _iq_stream_enable_reg = 'iqStreamEnable'

    def set_iq_stream_enable(self, band, val, **kwargs):
        r"""Selects between frequency and demodulated I/Q debug output.

        Toggles what data the debug readout path returns. When
        disabled (val=0), the system returns frequency (F) and
        frequency error (dF) streams. When enabled (val=1), the
        system returns the demodulated flux ramp I/Q from the LMS
        harmonic estimator — the detector signal expressed as I
        and Q components. This is distinct from
        :func:`set_rf_iq_stream_enable`, which outputs the raw RF
        I/Q from the digital downconverter before any tracking or
        demodulation.

        Args
        ----
        band : int
            Which band.
        val : int
            0 for F/dF output, 1 for demodulated I/Q output.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_iq_stream_enable` : Gets the current state.
        :func:`set_rf_iq_stream_enable` : Selects raw RF I/Q output.
        :func:`take_debug_data` : Takes data in the selected mode.
        """
        self._caput(
            self._band_root(band) + self._iq_stream_enable_reg,
            val, **kwargs)

    def get_iq_stream_enable(self, band, **kwargs):
        r"""Gets the demodulated I/Q debug output enable state.

        When enabled, the debug path outputs the demodulated flux
        ramp I/Q from the LMS harmonic estimator instead of
        frequency (F) and frequency error (dF).

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            0 if outputting F/dF, 1 if outputting demodulated I/Q.

        See Also
        --------
        :func:`set_iq_stream_enable` : Sets this mode.
        """
        return self._caget(
            self._band_root(band) + self._iq_stream_enable_reg,
            **kwargs)

    _feedback_polarity_reg = 'feedbackPolarity'

    def set_feedback_polarity(self, band, val, **kwargs):
        r"""Sets the global feedback polarity for a band.

        Controls the sign of the feedback correction applied to tone
        frequencies. The correct polarity depends on the sign
        convention of the eta calibration and the physical wiring.

        Args
        ----
        band : int
            Which band.
        val : int
            0 or 1. Flips the sign of the feedback correction.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_polarity` : Gets the current feedback polarity.
        :func:`set_feedback_enable` : Global enable for the feedback loop.
        """
        self._caput(
            self._band_root(band) + self._feedback_polarity_reg,
            val, **kwargs)

    def get_feedback_polarity(self, band, **kwargs):
        r"""Gets the global feedback polarity for a band.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            0 or 1. The current feedback polarity.

        See Also
        --------
        :func:`set_feedback_polarity` : Sets the feedback polarity.
        """
        return self._caget(
            self._band_root(band) + self._feedback_polarity_reg,
            **kwargs)

    _band_center_mhz_reg = 'bandCenterMHz'

    def set_band_center_mhz(self, band, val, **kwargs):
        r"""Sets the band center frequency in MHz.

        This is a software-only variable that records the absolute
        RF frequency of the band center (set by the LO). It does
        not write to hardware. Used for converting between channel
        offsets and absolute frequencies.

        Args
        ----
        band : int
            Which band.
        val : float
            Band center frequency in MHz (e.g. 4250.0).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_band_center_mhz` : Gets the band center frequency.
        :func:`get_tone_frequency_offset_mhz` : Gets subband offsets.
        """
        self._caput(
            self._band_root(band) + self._band_center_mhz_reg,
            val, **kwargs)

    def get_band_center_mhz(self, band, **kwargs):
        r"""Gets the band center frequency in MHz.

        Returns the absolute RF frequency of the band center.
        This is a software-only variable (not read from hardware).

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Band center frequency in MHz.

        See Also
        --------
        :func:`set_band_center_mhz` : Sets the band center frequency.
        :func:`get_tone_frequency_offset_mhz` : Gets subband offsets.
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
        r"""Sets the synthesis filter bank output scaling.

        Controls the output amplitude of the polyphase synthesis
        (reconstruction) filter bank. Each increment scales the
        output amplitude by a factor of 2. Nominal value is 2.

        Args
        ----
        band : int
            Which band.
        val : int
            Scale factor. 2-bit unsigned (0-3). Each increment
            is a factor of 2. Nominal is 2.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_synthesis_scale` : Gets the current synthesis scale.
        :func:`set_analysis_scale` : Sets the analysis filter bank scaling.
        """
        self._caput(
            self._band_root(band) + self._synthesis_scale_reg,
            val, **kwargs)

    def get_synthesis_scale(self, band, **kwargs):
        r"""Gets the synthesis filter bank output scaling.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Scale factor. 2-bit unsigned (0-3). Each increment
            is a factor of 2. Nominal is 2.

        See Also
        --------
        :func:`set_synthesis_scale` : Sets the synthesis scale.
        """
        return self._caget(
            self._band_root(band) + self._synthesis_scale_reg,
            **kwargs)

    _dsp_enable_reg = 'dspEnable'

    def set_dsp_enable(self, band, val, **kwargs):
        r"""Enables or disables baseband DSP processing for a band.

        When disabled, the analysis and synthesis filter banks still
        run but the baseband processing (tone generation, tracking,
        feedback, streaming) is halted.

        Args
        ----
        band : int
            Which band.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_dsp_enable` : Gets the current DSP enable state.
        """
        self._caput(
            self._band_root(band) + self._dsp_enable_reg,
            val, **kwargs)

    def get_dsp_enable(self, band, **kwargs):
        r"""Gets the baseband DSP processing enable state.

        Args
        ----
        band : int
            Which band.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_dsp_enable` : Sets the DSP enable state.
        """
        return self._caget(
            self._band_root(band) + self._dsp_enable_reg,
            **kwargs)

    # Single channel commands
    _feedback_enable_reg = 'feedbackEnable'

    def set_feedback_enable_channel(self, band, channel, val,
                                    **kwargs):
        r"""Sets the feedback enable for a single channel.

        The channel must have feedback enabled here AND the global
        feedback enable must be set via :func:`set_feedback_enable`
        for tracking to be active on this channel.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_feedback_enable_channel` : Gets this channel's enable.
        :func:`set_feedback_enable_array` : Sets all channels at once.
        :func:`set_feedback_enable` : Sets the global feedback enable.
        """
        self._caput(
            self._channel_root(band, channel) +
            self._feedback_enable_reg,
            val, **kwargs)

    def get_feedback_enable_channel(self, band, channel, **kwargs):
        r"""Gets the feedback enable for a single channel.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_feedback_enable_channel` : Sets this channel's enable.
        :func:`get_feedback_enable_array` : Gets all channels at once.
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._feedback_enable_reg,
            **kwargs)

    _eta_mag_scaled_channel_reg = 'etaMagScaled'

    def set_eta_mag_scaled_channel(self, band, channel, val,
                                   **kwargs):
        r"""Sets the eta magnitude for a single channel.

        Per-channel accessor for the same data as
        :func:`set_eta_mag_array`. The "Scaled" name is historical;
        there is no additional scaling applied. Preserves the
        current eta phase and recomputes the underlying etaI and
        etaQ.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        val : float
            Eta magnitude (real, positive). Maximum safe value
            is ~2.0 (limited by the underlying etaI/etaQ
            registers). Setting above ~2.0 may overflow one
            component depending on phase.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_eta_mag_scaled_channel` : Gets this channel's eta magnitude.
        :func:`set_eta_phase_degree_channel` : Sets this channel's eta phase.
        :func:`set_eta_mag_array` : Sets all channels at once.
        """
        self._caput(
            self._channel_root(band, channel) +
            self._eta_mag_scaled_channel_reg,
            val, **kwargs)

    def get_eta_mag_scaled_channel(self, band, channel, **kwargs):
        r"""Gets the eta magnitude for a single channel.

        Per-channel accessor for the same data as
        :func:`get_eta_mag_array`.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Eta magnitude (real, positive).

        See Also
        --------
        :func:`set_eta_mag_scaled_channel` : Sets this channel's eta magnitude.
        :func:`get_eta_mag_array` : Gets all channels at once.
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._eta_mag_scaled_channel_reg,
            **kwargs)

    _center_frequency_mhz_channel_reg = 'centerFrequencyMHz'

    def set_center_frequency_mhz_channel(self, band, channel, val,
                                         **kwargs):
        r"""Sets the tone center frequency for a single channel.

        Per-channel accessor for the same data as
        :func:`set_center_frequency_array`.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        val : float
            Center frequency in MHz. Range is +/-1.2 MHz.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_center_frequency_mhz_channel` : Gets this channel's frequency.
        :func:`set_center_frequency_array` : Sets all channels at once.
        """
        self._caput(
            self._channel_root(band, channel) +
            self._center_frequency_mhz_channel_reg,
            val, **kwargs)

    def get_center_frequency_mhz_channel(self, band, channel,
                                         **kwargs):
        r"""Gets the tone center frequency for a single channel.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Center frequency in MHz.

        See Also
        --------
        :func:`set_center_frequency_mhz_channel` : Sets this channel's frequency.
        :func:`get_center_frequency_array` : Gets all channels at once.
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._center_frequency_mhz_channel_reg,
            **kwargs)


    _amplitude_scale_channel_reg = 'amplitudeScale'

    def set_amplitude_scale_channel(self, band, channel, val,
                                    **kwargs):
        r"""Sets the tone amplitude for a single channel.

        Controls the drive power of the tone output for this
        channel. Each step is 3 dB. When set to 0, no tone is
        output and the channel is not processed by firmware.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        val : int
            Tone amplitude. 4-bit unsigned (0-15). 0 means no
            tone output and no processing. Each increment is
            3 dB.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_amplitude_scale_channel` : Gets this channel's amplitude.
        :func:`set_amplitude_scale_array` : Sets all channels at once.
        """
        self._caput(
            self._channel_root(band, channel) +
            self._amplitude_scale_channel_reg,
            np.uint32(val), **kwargs)

    def get_amplitude_scale_channel(self, band, channel, **kwargs):
        r"""Gets the tone amplitude for a single channel.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Tone amplitude. 4-bit unsigned (0-15). 0 means no
            tone. Each increment is 3 dB.

        See Also
        --------
        :func:`set_amplitude_scale_channel` : Sets this channel's amplitude.
        :func:`get_amplitude_scale_array` : Gets all channels at once.
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._amplitude_scale_channel_reg,
            **kwargs)

    _eta_phase_degree_channel_reg = 'etaPhaseDegree'

    def set_eta_phase_degree_channel(self, band, channel, val,
                                     **kwargs):
        r"""Sets the eta phase for a single channel in degrees.

        Preserves the current eta magnitude and recomputes the
        underlying etaI and etaQ. Note this accepts degrees,
        unlike :func:`set_eta_phase_array` which uses radians.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        val : float
            Eta phase in degrees. Range -180 to 180 (values
            outside this wrap due to periodicity of cos/sin).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_eta_phase_degree_channel` : Gets this channel's eta phase.
        :func:`set_eta_mag_scaled_channel` : Sets this channel's eta magnitude.
        :func:`set_eta_phase_array` : Sets all channels (in radians).
        """
        self._caput(
            self._channel_root(band, channel) +
            self._eta_phase_degree_channel_reg,
            val, **kwargs)

    def get_eta_phase_degree_channel(self, band, channel, **kwargs):
        r"""Gets the eta phase for a single channel in degrees.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Eta phase in degrees (-180 to 180).

        See Also
        --------
        :func:`set_eta_phase_degree_channel` : Sets this channel's eta phase.
        :func:`get_eta_phase_array` : Gets all channels (in radians).
        """
        return self._caget(
            self._channel_root(band, channel) +
            self._eta_phase_degree_channel_reg,
            **kwargs)

    _frequency_error_mhz_reg = 'frequencyErrorMHz'

    def get_frequency_error_mhz(self, band, channel, **kwargs):
        r"""Gets the frequency error for a single channel in MHz.

        Returns the measured detuning of the specified tone from
        its resonance after eta rotation, converted to MHz by
        the PyRogue linked variable.

        Args
        ----
        band : int
            Which band.
        channel : int
            Which channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Frequency error in MHz.

        See Also
        --------
        :func:`get_frequency_error_array` : Gets all channels' errors (raw).
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

    @_skipifrfsoc
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

    @_skipifrfsoc
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

    @_skipifrfsoc
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

    @_skipifrfsoc
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
        Low-level function.  The RF DAC enable is configured
        automatically during ``setup()``; users should not normally
        need to call this directly.

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
        Low-level function.  The RF DAC enable is configured
        automatically during ``setup()``; users should not normally
        need to call this directly.

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
        r"""Sets the JESD transmit output data source for a lane.

        Selects what data is driven on the specified JESD
        transmit lane.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        b : int
            Which lane (0-9).
        val : int or str
            Output source: 0 or 'OutputZero' (zeros),
            1 or 'UserData' (normal FPGA data),
            2 or 'OutputOnes' (ones),
            3 or 'TestData' (test pattern).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_data_out_mux` : Gets the current selection.
        """
        self._caput(
            self.jesd_tx_root.format(bay) +
            self._data_out_mux_reg.format(b),
            val, **kwargs)

    def get_data_out_mux(self, bay, b, **kwargs):
        r"""Gets the JESD transmit output data source for a lane.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        b : int
            Which lane (0-9).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int or str
            Output source (0-3).

        See Also
        --------
        :func:`set_data_out_mux` : Sets the selection.
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
        r"""Sets the JESD receive lane enable mask for a bay.

        Each bit enables one JESD204b receive lane (ADC data into
        the FPGA). Only 8 of the 10 available lanes are used per
        bay; the active mask depends on the hardware configuration.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            Lane enable bitmask (up to 10 bits). Each bit enables
            one lane.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_jesd_rx_enable` : Gets the current enable mask.
        :func:`set_jesd_tx_enable` : Sets the JESD transmit lane mask.
        :func:`get_jesd_rx_data_valid` : Checks if receive data is valid.
        """
        self._caput(
            self.jesd_rx_root.format(bay) + self._jesd_rx_enable_reg,
            val, **kwargs)

    def get_jesd_rx_enable(self, bay, **kwargs):
        r"""Gets the JESD receive lane enable mask for a bay.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Lane enable bitmask. Default is 0x3F3.

        See Also
        --------
        :func:`set_jesd_rx_enable` : Sets the enable mask.
        """
        return self._caget(
            self.jesd_rx_root.format(bay) + self._jesd_rx_enable_reg,
            **kwargs)

    _jesd_rx_status_valid_cnt_reg = 'StatusValidCnt'

    def get_jesd_rx_status_valid_cnt(self, bay, num, **kwargs):
        r"""Gets the JESD receive synchronization count for a lane.

        Counts the number of times the specified lane has
        synchronized (rising edges of data valid). On a stable
        system (synced once at startup) this should not be
        incrementing. Any count incrementing after startup
        indicates link instability. There are 10 lanes per bay
        (indexed 0-9), but only 8 are in use (matching the Rx
        enable mask 0x3F3).

        Args
        ----
        bay : int
            Which bay (0 or 1).
        num : int
            Which lane (0-9).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Number of synchronizations for this lane.

        See Also
        --------
        :func:`get_jesd_rx_data_valid` : Checks if receive data is valid.
        """
        return self._caget(
            self.jesd_rx_root.format(bay) +
            self._jesd_rx_status_valid_cnt_reg + f'[{num}]',
            **kwargs)

    _jesd_rx_data_valid_reg = 'DataValid'

    def get_jesd_rx_data_valid(self, bay, **kwargs):
        r"""Gets the JESD receive data valid status for a bay.

        Returns a bitmask indicating which receive lanes have
        valid data. Used to verify the JESD link is up and
        synchronized. Should match the enable mask (0x3F3) when
        the link is healthy.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Data valid bitmask, one bit per lane.

        See Also
        --------
        :func:`get_jesd_rx_enable` : Gets the lane enable mask.
        :func:`get_jesd_rx_status_valid_cnt` : Gets per-lane valid count.
        """
        return self._caget(
            self.jesd_rx_root.format(bay) +
            self._jesd_rx_data_valid_reg,
            **kwargs)

    _jesd_tx_enable_reg = 'Enable'

    def set_jesd_tx_enable(self, bay, val, **kwargs):
        r"""Sets the JESD transmit lane enable mask for a bay.

        Each bit enables one JESD204b transmit lane (DAC data out
        of the FPGA). Only 8 of the 10 available lanes are used
        per bay; the active mask depends on the hardware
        configuration.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            Lane enable bitmask (up to 10 bits). Each bit enables
            one lane.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_jesd_tx_enable` : Gets the current enable mask.
        :func:`set_jesd_rx_enable` : Sets the JESD receive lane mask.
        :func:`get_jesd_tx_data_valid` : Checks if transmit data is valid.
        """
        self._caput(
            self.jesd_tx_root.format(bay) + self._jesd_tx_enable_reg,
            val, **kwargs)

    def get_jesd_tx_enable(self, bay, **kwargs):
        r"""Gets the JESD transmit lane enable mask for a bay.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Lane enable bitmask. Default is 0x3CF.

        See Also
        --------
        :func:`set_jesd_tx_enable` : Sets the enable mask.
        """
        return self._caget(
            self.jesd_tx_root.format(bay) + self._jesd_tx_enable_reg,
            **kwargs)

    _jesd_tx_data_valid_reg = 'DataValid'

    def get_jesd_tx_data_valid(self, bay, **kwargs):
        r"""Gets the JESD transmit data valid status for a bay.

        Returns a bitmask indicating which transmit lanes have
        valid data. Should match the enable mask (0x3CF) when
        the link is healthy.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Data valid bitmask, one bit per lane.

        See Also
        --------
        :func:`get_jesd_tx_enable` : Gets the lane enable mask.
        :func:`get_jesd_tx_status_valid_cnt` : Gets per-lane valid count.
        """
        return self._caget(
            self.jesd_tx_root.format(bay) +
            self._jesd_tx_data_valid_reg,
            **kwargs)

    _jesd_tx_status_valid_cnt_reg = 'StatusValidCnt'

    def get_jesd_tx_status_valid_cnt(self, bay, num, **kwargs):
        r"""Gets the JESD transmit synchronization count for a lane.

        Counts the number of times the specified lane has
        synchronized. On a stable system (synced once at startup)
        this should not be incrementing. There are 10 lanes per
        bay (indexed 0-9), but only 8 are in use (matching the Tx
        enable mask 0x3CF).

        Args
        ----
        bay : int
            Which bay (0 or 1).
        num : int
            Which lane (0-9).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Number of synchronizations for this lane.

        See Also
        --------
        :func:`get_jesd_tx_data_valid` : Checks if transmit data is valid.
        """
        return self._caget(
            self.jesd_tx_root.format(bay) +
            self._jesd_tx_status_valid_cnt_reg + f'[{num}]',
            **kwargs)

    def set_check_jesd(self, max_timeout_sec=60.0, **kwargs):
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
            try:
                self._wait_for(
                    self.smurf_application + self._jesd_status_reg,
                    lambda status: status not in [None, 'Checking'],
                    timeout=max_timeout_sec
                )
                status = self.get_jesd_status()
            except TimeoutError:
                # If after out maximum defined timeout, we weren't able to
                # read the "JesdStatus" status register with a valid status,
                # then we exit on error.
                self.log(
                    'JESD health check did not finish after'
                    f' {max_timeout_sec} seconds.', self.LOG_ERROR)
                return self.get_jesd_status()

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
        r"""Selects the data source for a DaqMux buffer lane.

        The DaqMux routes internal firmware signals to acquisition
        buffers for debug data capture. Mapping (with InputMuxSel
        enum offset of 2):

        - 0: Disabled
        - 1: Test (incrementing counter pattern)
        - 2-11: ADC lanes 0-9 (I/Q pairs for each converter)
        - 12-21: DAC lanes 0-9 (I/Q pairs for each converter)
        - 22-25: Debug outputs 0-3 (content depends on
          :func:`set_iq_stream_enable` and
          :func:`set_rf_iq_stream_enable`; band selected by
          :func:`set_debug_select`)

        Typically configured automatically by :func:`take_debug_data`.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        lane : int
            Which DaqMux buffer lane.
        val : int
            Input source selection (0-25).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_input_mux_sel` : Gets the current selection.
        :func:`set_debug_select` : Selects which band's debug routes to DaqMux.
        :func:`take_debug_data` : Sets this up automatically.
        """
        self._caput(
            self.daq_mux_root.format(bay) +
            self._input_mux_sel_reg.format(lane),
            val, **kwargs)

    def get_input_mux_sel(self, bay, lane, **kwargs):
        r"""Gets the data source selection for a DaqMux buffer lane.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        lane : int
            Which DaqMux buffer lane.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Input source selection (0-25).

        See Also
        --------
        :func:`set_input_mux_sel` : Sets the selection.
        """
        self._caget(
            self.daq_mux_root.format(bay) +
            self._input_mux_sel_reg.format(lane),
            **kwargs)

    _data_buffer_size_reg = 'DataBufferSize'

    def set_data_buffer_size(self, bay, val, **kwargs):
        r"""Sets the DaqMux data buffer size.

        Sets the number of 32-bit words to capture in the DaqMux
        waveform buffer.  Determines how many samples are collected
        per :func:`take_debug_data` acquisition.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            Buffer size in 32-bit words.  Minimum 4.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`take_debug_data` : Triggers a data capture using this buffer.
        """
        self._caput(
            self.daq_mux_root.format(bay) +
            self._data_buffer_size_reg,
            val, **kwargs)

    def get_data_buffer_size(self, bay, **kwargs):
        r"""Gets the DaqMux data buffer size.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Buffer size in 32-bit words.

        See Also
        --------
        :func:`set_data_buffer_size` : Sets this value.
        :func:`take_debug_data` : Triggers a data capture using this buffer.
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
        r"""Opens the stream data file writer.

        Args
        ----
        val : int
            1 to open.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_streamdatawriter_open` : Gets the current state.
        :func:`set_streamdatawriter_close` : Closes the writer.
        """
        self._caput(
            self.stream_data_writer_root + self._datawriter_open_reg,
            val, **kwargs)


    def get_streamdatawriter_open(self, **kwargs):
        r"""Gets the stream data file writer open state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Current state.

        See Also
        --------
        :func:`set_streamdatawriter_open` : Opens the writer.
        """
        return self._caget(
            self.stream_data_writer_root + self._datawriter_open_reg,
            **kwargs)

    _datawriter_close_reg = 'Close'

    def set_streamdatawriter_close(self, val, **kwargs):
        r"""Closes the stream data file writer.

        Args
        ----
        val : int
            1 to close.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_streamdatawriter_close` : Gets the current state.
        :func:`set_streamdatawriter_open` : Opens the writer.
        """
        self._caput(
            self.stream_data_writer_root + self._datawriter_close_reg,
            val, **kwargs)

    def get_streamdatawriter_close(self, **kwargs):
        r"""Gets the stream data file writer close state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Current state.

        See Also
        --------
        :func:`set_streamdatawriter_close` : Closes the writer.
        """
        return self._caget(
            self.stream_data_writer_root + self._datawriter_close_reg,
            **kwargs)

    _trigger_daq_reg = 'TriggerDaq'

    def set_trigger_daq(self, bay, val, **kwargs):
        r"""Triggers the DaqMux to start data acquisition.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            1 to trigger.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_trigger_daq` : Gets the current state.
        :func:`set_arm_hw_trigger` : Arms the hardware trigger.
        """
        self._caput(
            self.daq_mux_root.format(bay) + self._trigger_daq_reg,
            val, **kwargs)

    def get_trigger_daq(self, bay, **kwargs):
        r"""Gets the DaqMux trigger state.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Current trigger state.

        See Also
        --------
        :func:`set_trigger_daq` : Triggers acquisition.
        """
        self._caget(
            self.daq_mux_root.format(bay) + self._trigger_daq_reg,
            **kwargs)

    _arm_hw_trigger_reg = "ArmHwTrigger"

    def set_arm_hw_trigger(self, bay, val, **kwargs):
        r"""Arms the DaqMux hardware trigger (alternate register).

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            1 to arm.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_trigger_hw_arm` : Arms the hardware trigger.
        :func:`set_trigger_daq` : Software trigger for acquisition.
        """
        self._caput(
            self.daq_mux_root.format(bay) + self._arm_hw_trigger_reg,
            val, **kwargs)

    _trigger_hw_arm_reg = 'TriggerHwArm'

    def set_trigger_hw_arm(self, bay, val, **kwargs):
        r"""Arms the DaqMux hardware trigger.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            1 to arm.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_trigger_hw_arm` : Gets the current state.
        :func:`set_trigger_daq` : Software trigger for acquisition.
        """
        self._caput(
            self.daq_mux_root.format(bay) + self._trigger_hw_arm_reg,
            val, **kwargs)

    def get_trigger_hw_arm(self, bay, **kwargs):
        r"""Gets the DaqMux hardware trigger arm state.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Current arm state.

        See Also
        --------
        :func:`set_trigger_hw_arm` : Arms the trigger.
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
        r"""Gets the waveform LUT table contents.

        Args
        ----
        reg : int
            Which LUT table (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        array-like
            LUT contents, up to 2048 entries of 20-bit signed values.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        return self._caget(
            self.rtm_lut_ctrl_root +
            self._rtm_arb_waveform_lut_table_reg.format(reg),
            **kwargs)

    _rtm_arb_waveform_busy_reg = 'Busy'

    def get_rtm_arb_waveform_busy(self, **kwargs):
        r"""Gets whether the RTM arbitrary waveform generator is active.

        Returns 1 if the LUT controller state machine is not idle
        (waveform is being output).  Goes low when playback completes
        or when Continuous is set to 0.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            1 if busy (outputting waveform), 0 if idle.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_busy_reg,
            **kwargs)

    _rtm_arb_waveform_trig_cnt_reg = 'TrigCnt'

    def get_rtm_arb_waveform_trig_cnt(self, **kwargs):
        r"""Gets the RTM arbitrary waveform trigger count.

        Number of accepted software triggers since boot or the last
        counter reset.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Trigger count (16-bit).
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
        r"""Gets the RTM arbitrary waveform continuous mode flag.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            1 if continuous (repeating), 0 if single-shot.

        See Also
        --------
        :func:`set_rtm_arb_waveform_continuous` : Sets this value.
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
        Low-level function.  Used internally by
        ``set_tes_bias_bipolar`` and ``set_tes_bias_bipolar_array`` to
        route the RTM LUT to the correct DAC pair; users should not
        normally need to call this directly.

        Sets the DacAxilAddr[#] registers.
        """
        assert (reg in range(2)), 'reg must be in [0,1]'
        self._caput(
            self.rtm_lut_ctrl + self._dac_axil_addr_reg.format(reg),
            f"Dac[{val:d}]", **kwargs)

    def get_dac_axil_addr(self, reg, **kwargs):
        """
        Low-level function.  Used internally by the bipolar TES-bias
        path; users should not normally need to call this directly.

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
        r"""Gets the RTM arbitrary waveform sample interval.

        Time between DAC updates is TimerSize × 6.4 ns.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Timer size (24-bit unsigned).  Multiply by 6.4 ns for
            the sample interval.

        See Also
        --------
        :func:`set_rtm_arb_waveform_timer_size` : Sets this value.
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
        r"""Gets the last LUT address played by the RTM arbitrary waveform generator.

        The slow RTM DACs play entries [0, MaxAddr] from the LUT
        tables before stopping (single-shot) or repeating (continuous
        mode).

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            MaxAddr (11-bit, range [0, 2048)).

        See Also
        --------
        :func:`set_rtm_arb_waveform_max_addr` : Sets this value.
        :func:`trigger_rtm_arb_waveform` : Start waveform playback.
        :func:`set_rtm_arb_waveform_continuous` : Single-shot vs repeating.
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
        r"""Gets the enable for generation of arbitrary waveforms on the RTM slow DACs.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Bitmask: 0x0 = disabled, 0x1 = Addr[0], 0x2 = Addr[1],
            0x3 = both.

        See Also
        --------
        :func:`set_rtm_arb_waveform_enable` : Sets this value.
        :func:`trigger_rtm_arb_waveform` : Start waveform playback.
        """
        return self._caget(
            self.rtm_lut_ctrl + self._rtm_arb_waveform_enable_reg,
            **kwargs)

    ## end rtm arbitrary waveform
    #########################################################

    _reset_rtm_reg = 'resetRtm'

    def reset_rtm(self, **kwargs):
        r"""Resets the rear transition module (RTM) CPLD.

        Asserts the CPLD reset line for 100 ms, which resets the flux
        ramp counter, bias DAC SPI state machines, and all CPLD
        control logic.  After releasing reset, re-writes all RTM
        registers from cached values to restore prior state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_cpld_reset` : Direct control of the reset line.
        :func:`cpld_toggle` : Alias for this function.
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
        r"""Gets the RTM CPLD reset state.

        When asserted (1), holds the RTM CPLD in reset — all CPLD
        logic is inactive, the flux ramp counter is zeroed, and
        SPI communication with the CPLD is halted.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if CPLD is held in reset, 0 if released.

        See Also
        --------
        :func:`set_cpld_reset` : Sets the reset state.
        :func:`cpld_toggle` : Pulses the reset.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._cpld_reset_reg,
            **kwargs)

    def cpld_toggle(self, **kwargs):
        r"""Resets the RTM CPLD.

        Alias for :func:`reset_rtm`.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to
            :func:`reset_rtm`.
        """
        self.reset_rtm(**kwargs)


    _timing_crate_root_reg = "AMCc.FpgaTopLevel.AmcCarrierCore.AmcCarrierTiming.EvrV2CoreTriggers"
    _trigger_rate_sel_reg = ".EvrV2ChannelReg[0].RateSel"

    def set_ramp_rate(self, val, **kwargs):
        r"""Sets the flux ramp reset rate via the timing system.

        Only applies when using the timing system as the ramp trigger
        source (RampStartMode = 1, see :func:`set_ramp_start_mode`).
        When using internal triggering, the ramp rate is instead
        determined by :func:`set_low_cycle` and :func:`set_high_cycle`.

        The timing system only supports discrete rates: 1, 2, 3, 4,
        5, 6, 8, 10, 12, 15 kHz.

        Args
        ----
        val : int or float
            Desired reset rate in kHz.  Must be one of the allowed
            values listed above.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_ramp_rate` : Gets the timing system rate setting.
        :func:`set_ramp_start_mode` : Select trigger source.
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
        r"""Gets the flux ramp reset rate from the timing system in kHz.

        Reads the timing system trigger rate selector and converts
        to the corresponding reset rate.  Only meaningful when using
        the timing system as the ramp trigger source
        (RampStartMode = 1).

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            Flux ramp reset rate in kHz.

        See Also
        --------
        :func:`set_ramp_rate` : Sets the reset rate.
        :func:`set_ramp_start_mode` : Select trigger source.
        """

        rate_sel = self._caget(
            self._timing_crate_root_reg +
            self._trigger_rate_sel_reg,
            **kwargs)

        reset_rate = self.flux_ramp_PV_to_rate(rate_sel)

        return reset_rate

    _trigger_delay_reg = ".EvrV2TriggerReg[0].Delay"

    def set_trigger_delay(self, val, **kwargs):
        r"""Sets the flux ramp trigger delay offset.

        Adds a delay to the flux ramp trigger.  Used to synchronize
        multiple carriers.  Units are timing system clock ticks
        (default 122.88 MHz = :func:`get_digitizer_frequency_mhz` / 5).

        Args
        ----
        val : int
            Trigger delay in timing clock ticks (default 122.88 MHz).
            28-bit unsigned.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput(
            self._timing_crate_root_reg +
            self._trigger_delay_reg,
            val, **kwargs)

    def get_trigger_delay(self, **kwargs):
        r"""Gets the flux ramp trigger delay offset.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Trigger delay in timing clock ticks (default 122.88 MHz).

        See Also
        --------
        :func:`set_trigger_delay` : Sets this value.
        """

        trigger_delay = self._caget(
            self._timing_crate_root_reg +
            self._trigger_delay_reg,
            **kwargs)

        return trigger_delay

    _debounce_width_reg = 'DebounceWidth'

    def set_debounce_width(self, val, **kwargs):
        r"""Sets the external trigger debounce width.

        Controls how many JESD clock cycles (see
        :func:`get_digitizer_frequency_mhz` / 2, default
        307.2 MHz) the external flux ramp trigger input (LEMO1
        on the RTM front panel) must be stable before being
        accepted as a valid trigger event. Only relevant when
        the ramp start mode is set to external triggering.
        Prevents spurious triggers from noisy input signals.

        Args
        ----
        val : int
            Debounce count (16-bit unsigned). Debounce time =
            val / (get_digitizer_frequency_mhz() * 1e6 / 2).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_debounce_width` : Gets the current debounce width.
        :func:`set_enable_ramp_trigger` : Enables trigger pulses.
        """
        self._caput(
            self.rtm_cryo_det_root + self._debounce_width_reg,
            val, **kwargs)

    def get_debounce_width(self, **kwargs):
        r"""Gets the external trigger debounce width.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Debounce count (16-bit unsigned).

        See Also
        --------
        :func:`set_debounce_width` : Sets the debounce width.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._debounce_width_reg,
            **kwargs)

    _ramp_slope_reg = 'RampSlope'

    def set_ramp_slope(self, val, **kwargs):
        r"""Sets the flux ramp slope polarity.

        Controls whether the flux ramp sawtooth waveform ramps
        up (positive slope) or down (negative slope).

        Args
        ----
        val : int
            0 for positive slope, 1 for negative slope.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_ramp_slope` : Gets the current slope polarity.
        :func:`set_cfg_reg_ena_bit` : Enables/disables the flux ramp.
        """
        self._caput(
            self.rtm_spi_root + self._ramp_slope_reg,
            val, **kwargs)

    def get_ramp_slope(self, **kwargs):
        r"""Gets the flux ramp slope polarity.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            0 for positive slope, 1 for negative slope.

        See Also
        --------
        :func:`set_ramp_slope` : Sets the slope polarity.
        """
        return self._caget(
            self.rtm_spi_root + self._ramp_slope_reg,
            **kwargs)

    _flux_ramp_dac_reg = 'LTC1668RawDacData'

    def set_flux_ramp_dac(self, val, **kwargs):
        r"""Sets the raw flux ramp DAC value.

        Writes directly to the LTC1668 flux ramp DAC data register
        on the RTM. This value is only output to the DAC when
        ``ModeControl`` is set to 1 (test mode) via
        :func:`set_mode_control`. In normal operation
        (``ModeControl`` = 0), the internal ramp counter drives
        the DAC and this value is ignored. The flux ramp does not
        need to be enabled for this static value to appear at the
        DAC output; ``ModeControl`` = 1 is sufficient.

        Args
        ----
        val : int
            Raw DAC value. 16-bit unsigned (0-65535).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_flux_ramp_dac` : Gets the current DAC value.
        :func:`set_mode_control` : Switches between ramp and static DAC modes.
        """
        self._caput(
            self.rtm_spi_root + self._flux_ramp_dac_reg,
            val, **kwargs)

    def get_flux_ramp_dac(self, **kwargs):
        r"""Gets the raw flux ramp DAC value.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Raw DAC value. 16-bit unsigned (0-65535).

        See Also
        --------
        :func:`set_flux_ramp_dac` : Sets the DAC value.
        :func:`get_mode_control` : Gets the current DAC mode.
        """
        return self._caget(
            self.rtm_spi_root + self._flux_ramp_dac_reg,
            **kwargs)

    _mode_control_reg = 'ModeControl'

    def set_mode_control(self, val, **kwargs):
        r"""Sets the RTM DAC output mode.

        Switches the RTM between normal operation (flux ramp counter
        drives the ramp DAC) and test/load mode (DACs are programmed
        directly from control registers). In test mode, the flux ramp
        DAC outputs the static value from :func:`set_flux_ramp_dac`,
        and direct SPI programming of bias DACs is enabled.

        Args
        ----
        val : int
            0 for normal operation (flux ramp output),
            1 for test/load mode (direct DAC programming).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_mode_control` : Gets the current mode.
        :func:`set_flux_ramp_dac` : Sets the static ramp DAC value for test mode.
        """
        self._caput(
            self.rtm_spi_root + self._mode_control_reg,
            val, **kwargs)

    def get_mode_control(self, **kwargs):
        r"""Gets the RTM DAC output mode.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            0 for normal operation, 1 for test/load mode.

        See Also
        --------
        :func:`set_mode_control` : Sets the DAC output mode.
        """
        return self._caget(
            self.rtm_spi_root + self._mode_control_reg,
            **kwargs)

    _fast_slow_step_size_reg = 'FastSlowStepSize'

    def set_fast_slow_step_size(self, val, **kwargs):
        r"""Sets the flux ramp step size.

        Controls how much the RTM ramp DAC counter increments
        each clock tick. The counter clock is derived from the
        307.2 MHz JESD clock divided by
        (lowCycle + highCycle + 2), defaulting to 51.2 MHz. The
        counter is 32 bits wide and the top 16 bits drive the
        DAC, so the effective DAC increment per tick is
        val / 2^16. Together with the ramp rate
        (:func:`set_ramp_max_cnt`), this determines the
        peak-to-peak amplitude of the flux ramp sawtooth.

        Args
        ----
        val : int
            Step size (32-bit unsigned).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_fast_slow_step_size` : Gets the current step size.
        :func:`set_fast_slow_rst_value` : Sets the ramp reset value.
        :func:`set_ramp_max_cnt` : Sets the ramp repetition rate.
        :func:`set_low_cycle` : Sets the clock divider low phase.
        :func:`set_high_cycle` : Sets the clock divider high phase.
        """
        self._caput(
            self.rtm_spi_root + self._fast_slow_step_size_reg,
            val, **kwargs)

    def get_fast_slow_step_size(self, **kwargs):
        r"""Gets the flux ramp step size.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Step size (32-bit unsigned).

        See Also
        --------
        :func:`set_fast_slow_step_size` : Sets the step size.
        """
        return self._caget(
            self.rtm_spi_root + self._fast_slow_step_size_reg,
            **kwargs)

    _fast_slow_rst_value_reg = 'FastSlowRstValue'

    def set_fast_slow_rst_value(self, val, **kwargs):
        r"""Sets the flux ramp counter reset value.

        The value the RTM ramp DAC counter resets to at the start
        of each ramp cycle (on each trigger pulse). This sets the
        starting point of the sawtooth waveform.

        Args
        ----
        val : int
            Reset value (32-bit unsigned). The top 16 bits
            correspond to the DAC output at ramp start.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_fast_slow_rst_value` : Gets the current reset value.
        :func:`set_fast_slow_step_size` : Sets the ramp step size.
        """
        self._caput(
            self.rtm_spi_root + self._fast_slow_rst_value_reg,
            val, **kwargs)

    def get_fast_slow_rst_value(self, **kwargs):
        r"""Gets the flux ramp counter reset value.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Reset value (32-bit unsigned).

        See Also
        --------
        :func:`set_fast_slow_rst_value` : Sets the reset value.
        """
        return self._caget(
            self.rtm_spi_root + self._fast_slow_rst_value_reg,
            **kwargs)

    _enable_ramp_trigger_reg = 'EnableRampTrigger'

    def set_enable_ramp_trigger(self, val, **kwargs):
        r"""Enables or disables the flux ramp trigger pulses.

        Controls the FPGA-side gate that allows ramp trigger pulses
        to reach the RTM. When disabled, no trigger pulses are
        generated regardless of the internal ramp counter or
        external trigger state.

        Args
        ----
        val : int
            1 to enable trigger pulses, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_enable_ramp_trigger` : Gets the current state.
        :func:`set_cfg_reg_ena_bit` : Enables the flux ramp via SPI.
        """
        self._caput(
            self.rtm_cryo_det_root + self._enable_ramp_trigger_reg,
            val, **kwargs)

    def get_enable_ramp_trigger(self, **kwargs):
        r"""Gets the flux ramp trigger pulse enable state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if trigger pulses are enabled, 0 if disabled.

        See Also
        --------
        :func:`set_enable_ramp_trigger` : Sets the enable state.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._enable_ramp_trigger_reg,
            **kwargs)

    _cfg_reg_ena_bit_reg = 'CfgRegEnaBit'

    def set_cfg_reg_ena_bit(self, val, **kwargs):
        r"""Enables or disables the flux ramp.

        Controls the flux ramp enable via the RTM SPI register
        interface. When enabled, the FPGA generates periodic
        trigger pulses that drive the RTM flux ramp DAC sawtooth
        waveform. Most users should use :func:`flux_ramp_on` and
        :func:`flux_ramp_off` instead.

        Args
        ----
        val : int
            1 to enable the flux ramp, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_cfg_reg_ena_bit` : Gets the current state.
        :func:`flux_ramp_on` : Convenience wrapper to enable.
        :func:`flux_ramp_off` : Convenience wrapper to disable.
        :func:`set_ramp_max_cnt` : Sets the flux ramp frequency.
        """
        self._caput(
            self.rtm_spi_root + self._cfg_reg_ena_bit_reg,
            val, **kwargs)

    def get_cfg_reg_ena_bit(self, **kwargs):
        r"""Gets the flux ramp enable state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if flux ramp is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_cfg_reg_ena_bit` : Sets the flux ramp enable.
        """
        return self._caget(
            self.rtm_spi_root + self._cfg_reg_ena_bit_reg,
            **kwargs)

    # Right now in pyrogue, this is named as if it's always a TesBias,
    # but pysmurf doesn't only use them as TES biases - e.g. in
    # systems using a 50K follow-on amplifier, one of these DACs is
    # used to drive the amplifier gate.
    _rtm_slow_dac_enable_reg = 'TesBiasDacCtrlRegCh'

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
            self._rtm_slow_dac_enable_reg, val, index=dac - 1, **kwargs)

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
            self._rtm_slow_dac_enable_reg, index=dac - 1,
            **kwargs)

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
            self._rtm_slow_dac_enable_reg,
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
            self._rtm_slow_dac_enable_reg,
            **kwargs)

    _rtm_slow_dac_data_reg = 'TesBiasDacDataRegCh'

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
            self._rtm_slow_dac_data_reg, val, index=dac - 1,
            **kwargs)

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
            self._rtm_slow_dac_data_reg, index=dac - 1,
            **kwargs)

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
            self.rtm_spi_max_root + self._rtm_slow_dac_data_reg,
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
            self.rtm_spi_max_root + self._rtm_slow_dac_data_reg,
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
    _rtm_33_ctrl_reg = 'HemtBiasDacCtrlRegCh'
    _rtm_33_data_reg = 'HemtBiasDacDataRegCh'

    def get_amp_gate_voltage(self, amp):
        r"""Gets the gate voltage for a cryogenic RF amplifier.

        Reads the raw DAC value from the RTM and converts to volts
        using the ``bit_to_V`` calibration factor from the pysmurf
        config file (e.g. ``config['amplifier']['bit_to_V_hemt']``).
        The conversion is: volts = bit_to_volt * dac_bits. The
        RTM DAC output is routed through a voltage divider on the
        cryocard before reaching the amplifier gate.

        Args
        ----
        amp : str
            Which amplifier. Use '50k' and 'hemt' for C02
            cryocards, or '50k1', '50k2', 'hemt1', 'hemt2' for
            C04/C05 cryocards.

        Returns
        -------
        volts : float
            Gate voltage in volts.

        See Also
        --------
        :func:`set_amp_gate_voltage` : Sets the gate voltage.
        :func:`get_amplifier_biases` : Gets all amplifier states.
        """
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

        Parameters
        ----------
        amp : str
            Use '50k' and 'hemt' for the C02 amps, and '50k1',
            '50k2', 'hemt1' and 'hemt2' for the C04 amps.

        voltage : float
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
        """Gets whether the drain power supply is enabled for a cryogenic RF amplifier.

        Reads the cryocard power-supply enable register and checks
        whether the bit(s) for the specified amplifier are set.

        Args
        ----
        amp : str
            Amplifier name.  One of 'hemt', '50k' (C02 revision
            cryostat card) or 'hemt1', 'hemt2', '50k1', '50k2'
            (C04/C05 revision cryostat card).

        Returns
        -------
        bool
            True if the drain power supply is enabled.

        See Also
        --------
        :func:`set_amp_drain_enable` : Enable or disable.
        :func:`set_amp_drain_voltage` : Preferred interface (C04/C05).
        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        power_bitmask = self.config.get('amplifier')[amp]['power_bitmask']
        power = self.C.read_ps_en()
        power_masked = power & power_bitmask

        return power_masked > 0

    def set_amp_drain_enable(self, amp, enable):
        """Enables or disables the drain power supply for a cryogenic RF amplifier.

        On C04/C05 revision cryostat cards,
        :func:`set_amp_drain_voltage` manages this automatically —
        call this directly only if you need explicit control.
        Depending on the RF amplifier, applying drain voltage with
        the gate at zero can cause significant current draw and
        heating in the cryostat.

        Args
        ----
        amp : str
            Amplifier name.  One of 'hemt', '50k' (C02 revision
            cryostat card) or 'hemt1', 'hemt2', '50k1', '50k2'
            (C04/C05 revision cryostat card).
        enable : bool
            True to enable, False to disable.
        """
        self.C.assert_amps_match_this_cryocard(list(amp))

        power_bitmask = self.config.get('amplifier')[amp]['power_bitmask']
        power = self.C.read_ps_en()

        power_masked = power & ~power_bitmask

        if enable:
            power_masked = power | power_bitmask

        self.C.write_ps_en(power_masked)

    def set_amp_drain_voltage(self, amp, volt, override = False):
        """Sets the drain voltage for a cryogenic RF amplifier.

        C04/C05 revision cryostat card only.  Converts the requested
        drain voltage to the corresponding RTM DAC voltage using the
        linear calibration (m, b) from the pysmurf cfg file and writes
        it.  If 0 V is requested, the drain power supply is disabled
        to prevent residual output.  Allowable range differs between
        HEMT and 50K amplifiers — configure limits in the cfg file.

        Args
        ----
        amp : str
            Amplifier name: 'hemt1', 'hemt2', '50k1', or '50k2'.
        volt : float
            Desired drain voltage in volts.  Must be within the range
            [drain_volt_min, drain_volt_max] configured in the cfg
            file, unless override is True.
        override : bool, optional, default False
            If True, bypass the configured voltage range limits.

        See Also
        --------
        :func:`get_amp_drain_voltage` : Read back drain voltage.
        :func:`set_amp_drain_enable` : Direct enable/disable control.
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
        """Measures the drain current for a cryogenic RF amplifier.

        Reads a current-sense voltage from the cryostat card PIC ADC,
        then converts to milliamps using the configured op-amp gain
        and sense resistor.  Because the current is sensed before the
        LDO, there is a small offset (typically < 1 mA) which is
        subtracted using the Id_offset parameter in the cfg file
        (hemt_Id_offset, 50k_Id_offset, or per-amp drain_offset).

        Args
        ----
        amp : str
            Amplifier name.  One of 'hemt', '50k' (C02 revision
            cryostat card) or 'hemt1', 'hemt2', '50k1', '50k2'
            (C04/C05 revision cryostat card).

        Returns
        -------
        float
            Measured drain current in milliamps.

        See Also
        --------
        :func:`get_amp_drain_current_dict` : Measure all amplifiers.
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
        """Measures drain current for all cryogenic RF amplifiers.

        Returns two entries on C02 revision cryostat cards ('hemt',
        '50k') or four on C04/C05 revision cryostat cards ('hemt1',
        'hemt2', '50k1', '50k2').

        Returns
        -------
        dict
            Amplifier name to drain current in milliamps.

        See Also
        --------
        :func:`get_amp_drain_current` : Single-amplifier measurement.
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
        """Applies default gate bias to all cryogenic RF amplifiers.

        Reads default gate voltages from the pysmurf cfg file and sets
        them.  On C04/C05 revision cryostat cards, configures all four
        amplifier gates.  Does not touch drain power supplies or drain
        voltages — call :func:`set_amp_drain_voltage` separately after
        confirming gate bias is correct.

        Should be called when drain power supplies are disabled and
        RTM DAC outputs are enabled (typical state after boot).

        See Also
        --------
        :func:`set_amp_drain_voltage` : Set drain voltage (C04/C05).
        :func:`set_amp_gate_voltage` : Set individual gate voltage.
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
        """Returns bias state for all cryogenic RF amplifiers.

        On C02 revision cryostat cards, returns gate voltages, drain
        currents, and drain enable states for 'hemt' and '50k'.  On
        C04/C05 revision cryostat cards, also returns drain voltages
        for 'hemt1', 'hemt2', '50k1', '50k2'.

        Returns
        -------
        dict
            Keys are '{amp}_{param}' strings, e.g. 'hemt_gate_volt',
            'hemt_drain_current', 'hemt_enable', and on C04/C05 also
            'hemt1_drain_volt'.
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
        elif major == 4:
            for amp in self.C.list_of_c04_amps:
                voltage = self.get_amp_gate_voltage(amp)
                amp_dict[amp + '_gate_volt'] = voltage

                voltage = self.get_amp_drain_voltage(amp)
                amp_dict[amp + '_drain_volt'] = voltage

                current = self.get_amp_drain_current(amp)
                amp_dict[amp + '_drain_current'] = current

                enable = self.get_amp_drain_enable(amp)
                amp_dict[amp + '_enable'] = enable
        else:
            raise ValueError(
                f"Did not recognize cryo-card major version {major}. "
                f"Read version {major}.{minor}.{patch} from CC."
            )

        return amp_dict

    def set_hemt_enable(self, disable=False):
        """Deprecated. Use :func:`set_amp_drain_enable` instead."""
        enable = not disable
        self.log(f'set_hemt_enable: Deprecated. Calling set_amp_drain_enable("hemt", {enable}')
        self.set_amp_drain_enable('hemt', enable)

    def set_50k_amp_enable(self, disable=False):
        """Deprecated. Use :func:`set_amp_drain_enable` instead."""
        enable = not disable
        self.log(f'set_50k_enable: Deprecated. Calling set_amp_drain_enable("50k", {enable}')
        self.set_amp_drain_enable('50k', enable)

    def get_50k_amp_gate_voltage(self):
        """Deprecated. Use :func:`get_amp_gate_voltage` instead."""
        self.log('get_50k_gate_voltage: Deprecated. Calling get_amp_gate_voltage("50k")')
        self.get_amp_get_voltage('50k')

    def set_50k_amp_gate_voltage(self, voltage, override=False):
        """Deprecated. Use :func:`set_amp_gate_voltage` instead."""
        self.log(f'set_50k_gate_voltage: Deprecated. Calling set_amp_gate_voltage("50k", {voltage}, override={override})')
        self.set_amp_gate_voltage('50k', voltage, override)

    def set_hemt_gate_voltage(self, voltage, override=False):
        """Deprecated. Use :func:`set_amp_gate_voltage` instead."""
        self.log(f'set_hemt_gate_voltage: Deprecated. Calling set_amp_gate_voltage("hemt", {voltage}, override={override})')
        self.set_amp_gate_voltage('hemt', voltage, override)

    def set_hemt_bias(self, voltage, override=False):
        """Deprecated. Use :func:`set_amp_gate_voltage` instead."""
        self.log(f'set_hemt_bias: Deprecated. Calling set_amp_gate_voltage("hemt", {voltage}, override={override})')
        self.get_amp_get_voltage('hemt', voltage, override)

    def get_hemt_bias(self):
        """Deprecated. Use :func:`get_amp_gate_voltage` instead."""
        self.log('get_hemt_bias: Deprecated. Calling get_amp_gate_voltage("hemt")')
        return self.get_amp_gate_voltage('hemt')

    def set_amplifier_bias(self, bias_hemt = None, bias_50k = None, **kwargs):
        """Deprecated. Use :func:`set_amp_gate_voltage` instead."""
        self.log('set_amplifier_bias: Deprecated. Calling set_amp_gate_voltage')
        if bias_hemt is not None:
            self.set_amp_gate_voltage('hemt', bias_hemt, **kwargs)

        if bias_50k is not None:
            self.set_amp_gate_voltage('50k', bias_50k, **kwargs)

    def get_amplifier_bias(self):
        """Deprecated. Use :func:`get_amplifier_biases` instead."""
        self.log('get_amplifier_bias: Deprecated. Calling get_amplifier_biases')
        return self.get_amplifier_biases()

    def get_hemt_drain_current(self):
        """Deprecated. Use :func:`get_amp_drain_current` instead."""
        self.log('get_hemt_drain_current: Deprecated. Calling get_amp_drain_current("hemt")')
        return self.get_amp_drain_current("hemt")

    def get_50k_amp_drain_current(self):
        """Deprecated. Use :func:`get_amp_drain_current` instead."""
        self.log('get_50k_amp_drain_current: Deprecated. Calling get_amp_drain_current("50k")')
        return self.get_amp_drain_current("50k")

    def flux_ramp_on(self, **kwargs):
        r"""Enables flux ramp output.

        Wrapper for :func:`set_cfg_reg_ena_bit` with value 1.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to
            :func:`set_cfg_reg_ena_bit`.

        See Also
        --------
        :func:`flux_ramp_off` : Disable flux ramp output.
        :func:`set_cfg_reg_ena_bit` : Direct control.
        """
        self.set_cfg_reg_ena_bit(1, **kwargs)

    def flux_ramp_off(self, **kwargs):
        r"""Disables flux ramp output.

        Wrapper for :func:`set_cfg_reg_ena_bit` with value 0.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to
            :func:`set_cfg_reg_ena_bit`.

        See Also
        --------
        :func:`flux_ramp_on` : Enable flux ramp output.
        :func:`set_cfg_reg_ena_bit` : Direct control.
        """
        self.set_cfg_reg_ena_bit(0, **kwargs)

    _ramp_max_cnt_reg = 'RampMaxCnt'

    def set_ramp_max_cnt(self, val, **kwargs):
        r"""Sets the internal flux ramp maximum count.

        Controls the flux ramp reset rate when using internal
        triggering (RampStartMode = 0).  The ramp resets every
        (RampMaxCnt + 1) jesdClk cycles, giving a reset rate of
        :func:`get_digitizer_frequency_mhz` / 2 / (RampMaxCnt + 1)
        MHz.  For example, with the default 614.4 MHz digitizer
        clock, RampMaxCnt = 307199 gives 1 kHz.

        Args
        ----
        val : int
            Maximum count (32-bit unsigned).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_ramp_max_cnt` : Gets this value.
        :func:`set_ramp_rate` : Set rate via timing system instead.
        """
        self._caput(
            self.rtm_cryo_det_root + self._ramp_max_cnt_reg,
            val, **kwargs)

    def get_ramp_max_cnt(self, **kwargs):
        r"""Gets the internal flux ramp maximum count.

        The ramp reset rate is :func:`get_digitizer_frequency_mhz` /
        2 / (RampMaxCnt + 1) MHz.  Only meaningful when using internal
        triggering (RampStartMode = 0).

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Maximum count (32-bit unsigned).

        See Also
        --------
        :func:`set_ramp_max_cnt` : Sets this value.
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
        r"""Sets the RTM clock low cycle duration.

        Along with :func:`set_high_cycle`, sets the frequency of the
        RTM CPLD clock.  The RTM clock frequency is
        jesdClk / (LowCycle + HighCycle + 2), where jesdClk is
        :func:`get_digitizer_frequency_mhz` / 2 (default 307.2 MHz).
        Zero inclusive (a value of 0 means 1 tick low).

        Args
        ----
        val : int
            Low cycle duration in jesdClk ticks (zero inclusive).
            8-bit unsigned (0–255).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_high_cycle` : Sets the high cycle duration.
        """
        self._caput(
            self.rtm_cryo_det_root + self._low_cycle_reg,
            val, **kwargs)

    def get_low_cycle(self, val, **kwargs):
        r"""Gets the RTM clock low cycle duration.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Low cycle duration in jesdClk ticks (zero inclusive).
            jesdClk = :func:`get_digitizer_frequency_mhz` / 2
            (default 307.2 MHz).

        See Also
        --------
        :func:`set_low_cycle` : Sets this value.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._low_cycle_reg,
            **kwargs)

    _high_cycle_reg = 'HighCycle'

    def set_high_cycle(self, val, **kwargs):
        r"""Sets the RTM clock high cycle duration.

        Along with :func:`set_low_cycle`, sets the frequency of the
        RTM CPLD clock.  The RTM clock frequency is
        jesdClk / (LowCycle + HighCycle + 2), where jesdClk is
        :func:`get_digitizer_frequency_mhz` / 2 (default 307.2 MHz).
        Zero inclusive (a value of 0 means 1 tick high).

        Args
        ----
        val : int
            High cycle duration in jesdClk ticks (zero inclusive).
            8-bit unsigned (0–255).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_low_cycle` : Sets the low cycle duration.
        """
        self._caput(
            self.rtm_cryo_det_root + self._high_cycle_reg,
            val, **kwargs)

    def get_high_cycle(self, val, **kwargs):
        r"""Gets the RTM clock high cycle duration.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            High cycle duration in jesdClk ticks (zero inclusive).
            jesdClk = :func:`get_digitizer_frequency_mhz` / 2
            (default 307.2 MHz).

        See Also
        --------
        :func:`set_high_cycle` : Sets this value.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._high_cycle_reg,
            **kwargs)

    _ramp_start_mode_reg = 'RampStartMode'

    def set_ramp_start_mode(self, val, **kwargs):
        r"""Sets the flux ramp trigger source.

        Args
        ----
        val : int
            0 for internal trigger, 1 for timing system trigger,
            2 for external trigger.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput(
            self.rtm_cryo_det_root + self._ramp_start_mode_reg,
            val, **kwargs)

    def get_ramp_start_mode(self, **kwargs):
        r"""Gets the flux ramp trigger source.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            0 for internal, 1 for timing system, 2 for external.

        See Also
        --------
        :func:`set_ramp_start_mode` : Sets this value.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._ramp_start_mode_reg,
            **kwargs)

    _pulse_width_reg = 'PulseWidth'

    def set_pulse_width(self, val, **kwargs):
        r"""Sets the flux ramp pulse width on the RTM.

        Width of the start ramp pulse sent to the CPLD, in units of
        jesdClk = :func:`get_digitizer_frequency_mhz` / 2 (default
        307.2 MHz), same clock as :func:`set_low_cycle` and
        :func:`set_high_cycle`.  Typical value is 64.

        Args
        ----
        val : int
            Pulse width in jesdClk ticks (default 307.2 MHz).
            16-bit unsigned (0–65535).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput(
            self.rtm_cryo_det_root + self._pulse_width_reg,
            val, **kwargs)

    def get_pulse_width(self, **kwargs):
        r"""Gets the flux ramp pulse width on the RTM.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Pulse width in jesdClk ticks (default 307.2 MHz).

        See Also
        --------
        :func:`set_pulse_width` : Sets this value.
        """
        return self._caget(
            self.rtm_cryo_det_root + self._pulse_width_reg,
            **kwargs)


    _stream_datafile_reg = 'DataFile'

    def set_streaming_datafile(self, datafile, as_string=True,
                               **kwargs):
        """
        Sets the datafile to write streaming data

        Args
        ----
        datafile : str or length 300 int array
            The name of the datafile.
        as_string : bool, optional, default True
            DEPRECATED: Raises an error if set to False.
        """
        if not as_string:
            raise ValueError("Passing an int is deprecated.")
        self._caput(
            self.streaming_root + self._stream_datafile_reg,
            datafile, **kwargs)

    def get_streaming_datafile(self, **kwargs):
        """
        Gets the datafile that streaming data is written to.

        Returns
        -------
        datafile : str or length 300 int array
            The name of the datafile.
        """
        datafile = self._caget(
            self.streaming_root + self._stream_datafile_reg,
            **kwargs)
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
        r"""Gets the userConfig[0] timing frame header register.

        A general-purpose 32-bit field embedded in the data stream
        timing frame. Individual bits are used as control flags
        by the SmurfProcessor (e.g. bit 0 resets unwrapping and
        averaging when toggled).

        Args
        ----
        as_binary : bool, optional, default False
            If True, returns the value as a binary string.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int or str
            Register value, or binary string if as_binary=True.

        See Also
        --------
        :func:`set_user_config0` : Sets the value.
        :func:`clear_unwrapping_and_averages` : Toggles bit 0.
        """
        val =  self._caget(
            self.timing_header + self._smurf_to_gcp_stream_reg,
            **kwargs)

        if as_binary:
            val = bin(val)

        return val


    def set_user_config0(self, val, as_binary=False, **kwargs):
        r"""Sets the userConfig[0] timing frame header register.

        A general-purpose 32-bit field embedded in the data stream
        timing frame. Individual bits are used as control flags
        by the SmurfProcessor.

        Args
        ----
        val : int
            Register value (32-bit unsigned).
        as_binary : bool, optional, default False
            Unused (kept for API compatibility).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_user_config0` : Gets the current value.
        :func:`clear_unwrapping_and_averages` : Toggles bit 0.
        """
        self._caput(
            self.timing_header + self._smurf_to_gcp_stream_reg,
            val, **kwargs)


    def clear_unwrapping_and_averages(self, **kwargs):
        r"""Resets phase unwrapping and averaging for all channels.

        Toggles bit 0 of userConfig[0] high then low, which signals
        the downstream processor to clear its unwrapping accumulators
        and averaging state across all bands.  Uses a SyncGroup to
        confirm each transition completes before proceeding.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.

        See Also
        --------
        :func:`set_unwrapper_reset` : Per-band unwrapper reset.
        :func:`set_filter_reset` : Per-band filter reset.
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
        r"""Sets the trigger output pulse width.

        Duration of the trigger pulse output for the given channel,
        in timing system clock ticks (default 122.88 MHz =
        :func:`get_digitizer_frequency_mhz` / 5).

        Args
        ----
        chan : int
            Which trigger channel.
        val : int
            Pulse width in timing clock ticks.  28-bit unsigned.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
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


    _trigger_channel_reg_count_reg = 'EvrV2ChannelReg[{}].Count'

    def get_evr_channel_reg_count(self, chan, **kwargs):
        r"""Gets the EVR trigger channel event count.

        Returns the number of timing events received on the
        specified channel. Useful for verifying that timing
        events are being received.

        Args
        ----
        chan : int
            Which trigger channel.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Number of events received.

        See Also
        --------
        :func:`set_evr_trigger_dest_type` : Sets the trigger destination type.
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
        if not isinstance(value, str):
            self.log("set_evr_trigger_dest_type: "
                     "Use caution setting DestType to an int. enum mapping was "
                     "incorrect in previous code, and behaviour may have changed.",
                     self.LOG_ERROR)
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
        r"""Sets the EVR trigger channel destination select.

        Selects the destination mask for the specified EVR
        channel. Used during timing setup to configure which
        events are routed to the flux ramp trigger.

        Args
        ----
        chan : int
            Which trigger channel.
        val : int
            Destination select value.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_evr_trigger_dest_type` : Sets the destination type.
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
        Low-level function.  Called from ``SmurfControl.setup()`` to
        issue the physical reset to the RF DACs after
        ``setDefaults``; users should not normally need to call this
        directly.

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
        Low-level function.  The DAC reset line is exercised by
        ``SmurfControl.setup()``; users should not normally need to
        call this directly.

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
        r"""Selects which band's debug data is routed to the DaqMux.

        Each bay has 4 bands. This register selects which band's
        debug output is routed to that bay's DaqMux debug stream
        for acquisition via :func:`take_debug_data`.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        val : int
            Which band within the bay (0-3).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_debug_select` : Gets the current selection.
        :func:`take_debug_data` : Takes debug data.
        """
        self._caput(
            self.app_core + self._debug_select_reg.format(bay),
            val, **kwargs)

    def get_debug_select(self, bay, **kwargs):
        r"""Gets the band selected for debug data routing.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Which band within the bay (0-3).

        See Also
        --------
        :func:`set_debug_select` : Sets the selection.
        """
        return self._caget(
            self.app_core + self._debug_select_reg.format(bay),
            **kwargs)

    ### Start Ultrascale OT protection

    _ultrascale_ot_upper_threshold_reg = "OTUpperThreshold"

    def set_ultrascale_ot_upper_threshold(self, val, **kwargs):
        r"""Sets the FPGA over-temperature shutdown threshold.

        If the Ultrascale+ FPGA die temperature exceeds this value,
        the FPGA asserts an over-temperature alarm.

        Args
        ----
        val : float
            Temperature threshold in degrees C.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.
        """
        self._caput(
            self.ultrascale + self._ultrascale_ot_upper_threshold_reg,
            val, **kwargs)

    def get_ultrascale_ot_upper_threshold(self, **kwargs):
        r"""Gets the FPGA over-temperature shutdown threshold.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float
            Temperature threshold in degrees C.

        See Also
        --------
        :func:`set_ultrascale_ot_upper_threshold` : Sets this value.
        """
        return self._caget(
            self.ultrascale + self._ultrascale_ot_upper_threshold_reg,
            **kwargs)

    ### End Ultrascale OT protection

    _output_config_reg = "OutputConfig[{}]"

    def set_crossbar_output_config(self, index, val, **kwargs):
        r"""Sets the timing crossbar output configuration.

        Configures the Microchip SY56040 timing signal routing
        crossbar, which determines how timing signals are routed
        between the RTM, FPGA, and ATCA backplane. Each output
        index selects which input source drives it:

        - Index 0: RTM_TIMING_OUT0 source
        - Index 1: FPGA_TIMING_OUT source
        - Index 2: Backplane DIST0 source
        - Index 3: Backplane DIST1 source

        For each index, val selects the input:

        - 0x0: RTM_TIMING_IN0
        - 0x1: FPGA_TIMING_IN
        - 0x2: BP_TIMING_IN
        - 0x3: RTM_TIMING_IN1

        Args
        ----
        index : int
            Which output (0-3).
        val : int
            Which input source (0-3).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_crossbar_output_config` : Gets the current config.
        """
        self._caput(
            self.crossbar + self._output_config_reg.format(index),
            val, **kwargs)

    def get_crossbar_output_config(self, index, **kwargs):
        r"""Gets the timing crossbar output configuration.

        Args
        ----
        index : int
            Which output (0-3).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Which input source (0-3) is routed to this output.

        See Also
        --------
        :func:`set_crossbar_output_config` : Sets this value (includes
                full input/output mapping).
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
        self._caget(self.lmk.format(bay) + 'enable', **kwargs)

    # assumes it's handed the decimal equivalent
    _lmk_reg = "LmkReg_0x{:04X}"

    def set_lmk_reg(self, bay, reg, val, **kwargs):
        r"""Sets a register on the LMK clock distribution chip.

        Low-level access to individual registers of the LMK jitter
        cleaner on the specified AMC bay.  The LMK generates and
        distributes clocks and JESD204B SYSREF signals to the ADC
        and DAC chips on the ADC/DAC card.

        Args
        ----
        bay : int
            AMC bay number (0 or 1).
        reg : int
            Register address (e.g. 0x147).
        val : int
            Value to write.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_lmk_reg` : Read back a register.
        """
        self._caput(
            self.lmk.format(bay) + self._lmk_reg.format(reg),
            val, **kwargs)

    def get_lmk_reg(self, bay, reg, **kwargs):
        r"""Gets a register from the LMK clock distribution chip.

        Low-level access to individual registers of the LMK jitter
        cleaner on the specified AMC bay.  The LMK generates and
        distributes clocks and JESD204B SYSREF signals to the ADC
        and DAC chips on the ADC/DAC card.

        Args
        ----
        bay : int
            AMC bay number (0 or 1).
        reg : int
            Register address (e.g. 0x147).
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Register value.

        See Also
        --------
        :func:`set_lmk_reg` : Write a register.
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
        r"""Gets the count of frames lost before reaching the SmurfProcessor.

        A SmurfProcessor diagnostic counter.  Increments by the number
        of missing frames each time a gap in frame sequence numbers is
        detected (i.e. frames generated by the FPGA but never received
        by the SmurfProcessor).

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Number of lost frames since last reset.

        See Also
        --------
        :func:`get_frame_out_order_count` : Frames received out of order.
        """
        return self._caget(
            self.frame_rx_stats + self._frame_loss_count_reg,
            **kwargs)

    _frame_out_order_count_reg = 'FrameOutOrderCnt'

    def get_frame_out_order_count(self, **kwargs):
        r"""Gets the count of frames received out of order.

        A SmurfProcessor diagnostic counter. Increments each time
        a received frame has a lower sequence number than the
        previous frame. Such frames are discarded. A nonzero
        value may indicate network or data transport issues.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            Number of out-of-order frames received.
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
        # Smurf Processor stricly requires a python list
        mask = [int(i) for i in mask]
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
        r"""Resets the SmurfProcessor phase unwrapper.

        The unwrapper tracks wraps in the firmware's fixed-point
        frequency output, maintaining a per-channel counter to
        reconstruct the continuous signal.  This reset clears all
        wrap counters and previous-sample state, so subsequent
        output restarts from zero offset.  One-shot command (no
        corresponding get).

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_filter_reset` : Reset the downsample filter.
        :func:`clear_unwrapping_and_averages` : Reset unwrapper and averaging.
        """
        self._caput(
            self.smurf_processor + self._unwrapper_reset_reg,
            1, **kwargs)

    _filter_reset_reg = 'Filter.reset'

    def set_filter_reset(self, **kwargs):
        r"""Resets the SmurfProcessor downsample filter.

        Clears the filter state for all channels.  One-shot command
        (no corresponding get).

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`set_unwrapper_reset` : Reset the phase unwrapper.
        :func:`set_filter_a` : Set filter A coefficients.
        :func:`set_filter_b` : Set filter B coefficients.
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
        """Sets the SmurfProcessor downsampler trigger mode.

        In 'internal' mode, the downsampler outputs one frame every
        N input frames (N set by :func:`set_downsample_factor`).
        In 'external' mode, the downsampler uses timing bits from the
        frame header to trigger output, allowing synchronization to
        external timing markers.

        Args
        ----
        mode : str
            'internal' or 'external'.

        See Also
        --------
        :func:`get_downsample_mode` : Gets the current mode.
        :func:`set_downsample_factor` : Set internal decimation factor.
        :func:`set_downsample_external_bitmask` : Set timing bits for external mode.
        """
        if mode == 'internal':
            self._caput(self.smurf_processor + self._downsampler_mode_reg, 0)
        elif mode == 'external':
            self._caput(self.smurf_processor + self._downsampler_mode_reg, 1)
        else:
            self.log(f'set_downsample_mode: Unknown mode {mode}')

    def get_downsample_mode(self):
        """Gets the SmurfProcessor downsampler trigger mode.

        Returns
        -------
        str
            'internal' or 'external'.

        See Also
        --------
        :func:`set_downsample_mode` : Sets the mode.
        """
        mode = self._caget(self.smurf_processor + self._downsampler_mode_reg)

        if mode == 0:
            return 'internal'
        else:
            return 'external'

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
        """Sets the external downsampler bitmask.

        Only used when downsample mode is 'external'.  The
        SmurfProcessor reads timing bits from each frame header,
        ANDs them with this bitmask, and outputs a frame only when
        the result equals the bitmask (i.e. all masked bits are set
        simultaneously in that frame).

        The timing bits are: bits [9:0] are fixed-rate markers
        (1, 2, 3, 4, 5, 6, 8, 10, 12, 15 kHz), bits [27:10] are
        sequencer markers.  For example, bitmask = 0x1 triggers at
        1 kHz (the rate of the first fixed-rate marker).

        Args
        ----
        bitmask : int
            28-bit bitmask selecting which timing bits to match.

        See Also
        --------
        :func:`get_downsample_external_bitmask` : Gets this value.
        :func:`set_downsample_mode` : Must be 'external' for this to apply.
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
            bool(disable_status), **kwargs)

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
        r"""Opens the SmurfProcessor output data file for writing.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`close_data_file` : Close the file.
        """
        self._caput(self.smurf_processor + self._data_file_open_reg, 1,
            **kwargs)

    _data_file_close_reg = 'FileWriter.Close'

    def close_data_file(self, **kwargs):
        r"""Closes the SmurfProcessor output data file.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`open_data_file` : Open the file.
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
        r"""Enables or disables the SmurfProcessor pre-data emulator.

        When enabled, the emulator generates synthetic data frames
        upstream of the SmurfProcessor for testing without hardware.

        Args
        ----
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_predata_emulator_enable` : Gets the enable state.
        """
        self._caput(self._predata_emulator + 'enable', val, **kwargs)

    def get_predata_emulator_enable(self, **kwargs):
        r"""Gets the SmurfProcessor pre-data emulator enable state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_predata_emulator_enable` : Sets the enable state.
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
        r"""Sets the pre-data emulator signal amplitude.

        The pre-data emulator injects synthetic test data into
        the streaming pipeline before data processing.

        Args
        ----
        val : float
            Signal amplitude.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_predata_emulator_amplitude` : Gets the current value.
        :func:`set_predata_emulator_enable` : Enables the emulator.
        """
        self._caput(self._predata_emulator + self._predata_emulator_amplitude,
            val, **kwargs)

    def get_predata_emulator_amplitude(self, **kwargs):
        r"""Gets the pre-data emulator signal amplitude.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Signal amplitude.

        See Also
        --------
        :func:`set_predata_emulator_amplitude` : Sets the value.
        """
        return self._caget(self._predata_emulator +
            self._predata_emulator_amplitude, **kwargs)

    _predata_emulator_offset = "Offset"

    def set_predata_emulator_offset(self, val, **kwargs):
        r"""Sets the pre-data emulator signal offset.

        Args
        ----
        val : float
            Signal DC offset.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_predata_emulator_offset` : Gets the current value.
        """
        self._caput(self._predata_emulator + self._predata_emulator_offset, val,
            **kwargs)

    def get_predata_emulator_offset(self, **kwargs):
        r"""Gets the pre-data emulator signal offset.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Signal DC offset.

        See Also
        --------
        :func:`set_predata_emulator_offset` : Sets the value.
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
        r"""Enables or disables the post-data emulator.

        Args
        ----
        val : int
            1 to enable, 0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_postdata_emulator_enable` : Gets the current state.
        :func:`set_postdata_emulator_amplitude` : Sets the signal amplitude.
        """
        self._caput(self._postdata_emulator + 'enable', val, **kwargs)

    def get_postdata_emulator_enable(self, **kwargs):
        r"""Gets the post-data emulator enable state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if enabled, 0 if disabled.

        See Also
        --------
        :func:`set_postdata_emulator_enable` : Sets the enable state.
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
        r"""Sets the post-data emulator signal amplitude.

        The post-data emulator injects synthetic test data into
        the streaming pipeline after data processing.

        Args
        ----
        val : float
            Signal amplitude.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_postdata_emulator_amplitude` : Gets the current value.
        :func:`set_postdata_emulator_enable` : Enables the emulator.
        """
        self._caput(self._postdata_emulator + self._postdata_emulator_amplitude,
            val, **kwargs)

    def get_postdata_emulator_amplitude(self, **kwargs):
        r"""Gets the post-data emulator signal amplitude.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Signal amplitude.

        See Also
        --------
        :func:`set_postdata_emulator_amplitude` : Sets the value.
        """
        return self._caget(self._postdata_emulator +
            self._postdata_emulator_amplitude, **kwargs)

    _postdata_emulator_offset = "Offset"

    def set_postdata_emulator_offset(self, val, **kwargs):
        r"""Sets the post-data emulator signal offset.

        Args
        ----
        val : float
            Signal DC offset.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_postdata_emulator_offset` : Gets the current value.
        """
        self._caput(self._postdata_emulator + self._postdata_emulator_offset,
            val, **kwargs)

    def get_postdata_emulator_offset(self, **kwargs):
        r"""Gets the post-data emulator signal offset.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : float
            Signal DC offset.

        See Also
        --------
        :func:`set_postdata_emulator_offset` : Sets the value.
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
        r"""Enables or disables the StreamDataSource emulator.

        The StreamDataSource generates synthetic SMuRF frames at a
        configurable period for testing without hardware.  This is
        separate from the real data path.

        Args
        ----
        val : bool or int
            True/1 to enable, False/0 to disable.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_stream_data_source_enable` : Gets the enable state.
        :func:`set_stream_data_source_period` : Set frame period.
        """
        self._caput(self.stream_data_source + self._stream_data_source_enable,
            val, **kwargs)

    def get_stream_data_source_enable(self, **kwargs):
        r"""Gets the data stream source enable state.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        val : int
            1 if streaming is enabled, 0 if disabled.

        See Also
        --------
        :func:`set_stream_data_source_enable` : Sets the enable state.
        """
        return self._caget(self.stream_data_source +
            self._stream_data_source_enable, **kwargs)

    _stream_data_period = "Period"

    def set_stream_data_source_period(self, val, **kwargs):
        r"""Sets the StreamDataSource emulator frame period.

        Time between emulated frames in microseconds.

        Args
        ----
        val : int
            Period in microseconds.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caput` call.

        See Also
        --------
        :func:`get_stream_data_source_period` : Gets this value.
        :func:`set_stream_data_source_enable` : Enable the emulator.
        """
        self._caput(self.stream_data_source + self._stream_data_period, val,
            **kwargs)

    def get_stream_data_source_period(self, **kwargs):
        r"""Gets the StreamDataSource emulator frame period.

        Args
        ----
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        int
            Period in microseconds.

        See Also
        --------
        :func:`set_stream_data_source_period` : Sets this value.
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

    _readout_delay_reg = "readoutDelay"

    def get_readout_delay(self, **kwargs):
        """
        Get the value of a programmable delay between the startRamp pulse and when
        the readout FSM is triggered.

        The delay step is based on the AXI-Lite clock and its frequency is 156.25MHz (1/6.4ns)

        https://github.com/slaclab/cryo-det/blob/main/firmware/common/shared/rtl/CryoStream.vhd#L36
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/DspCoreLib/CryoDetCmbHcd/rtl/DspCoreWrapper.vhd#L234
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/DspCoreLib/CryoDetCmbHcd/rtl/DspCoreWrapper.vhd#L44
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/MicrowaveMuxApp/AppCore/hdl/AppCore.vhd#L317
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/MicrowaveMuxApp/AppCore/hdl/AppCore.vhd#L222
        """
        return self._caget(f"{self.app_core}{self._readout_delay_reg}", **kwargs)

    def set_readout_delay(self, delay, **kwargs):
        """
        Set the value of a programmable delay between the startRamp pulse and when
        the readout FSM is triggered.

        The delay step is based on the AXI-Lite clock and its frequency is 156.25MHz (1/6.4ns)

        https://github.com/slaclab/cryo-det/blob/main/firmware/common/shared/rtl/CryoStream.vhd#L36
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/DspCoreLib/CryoDetCmbHcd/rtl/DspCoreWrapper.vhd#L234
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/DspCoreLib/CryoDetCmbHcd/rtl/DspCoreWrapper.vhd#L44
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/MicrowaveMuxApp/AppCore/hdl/AppCore.vhd#L317
        https://github.com/slaclab/cryo-det/blob/main/firmware/common/MicrowaveMuxApp/AppCore/hdl/AppCore.vhd#L222
        """
        return self._caput(f"{self.app_core}{self._readout_delay_reg}", delay, **kwargs)

    _debug_timing_override_reg = "DebugTimingOverrideBay{}Ch{}"

    def get_debug_timing_override(self, bay, ch, **kwargs):
        """
        Get the value of the debug timing override register for a given bay and
        channel.

        If this register is enabled, the value of Counter0 will be substituted
        for one of the data channels (e.g. I/Q or f/df). This is useful for
        debugging timing issues in the system, as it allows you to see the value
        of Counter0 in the data stream.

        Counter0 increments at 480kHz and is reset by the PPS signal to the
        timing system.

        Args
        ----
        bay : int
            The bay number (0 or 1).
        ch : int
            The channel (0 or 1).
        """
        return self._caget(self.app_core + self._debug_timing_override_reg.format(bay, ch), **kwargs)

    def set_debug_timing_override(self, bay, ch, val, **kwargs):
        """
        Set the value of the debug timing override register for a given bay and
        channel.

        If this register is enabled, the value of Counter0 will be substituted
        for one of the data channels (e.g. I/Q or f/df). This is useful for
        debugging timing issues in the system, as it allows you to see the value
        of Counter0 in the data stream.

        Counter0 increments at 480kHz and is reset by the PPS signal to the
        timing system.

        Args
        ----
        bay : int
            The bay number (0 or 1).
        ch : int
            The channel (0 or 1).
        val : int
            Value to set (0 or 1).
        """
        return self._caput(self.app_core + self._debug_timing_override_reg.format(bay, ch), val, **kwargs)

    _counter_select_reg = "counterSelect"

    def get_counter_select(self, band, **kwargs):
        """
        Get the value of the counterSelect register for a given band.

        If this register is enabled, the value of I and Q will be substituted
        with the flux ramp frame counter. This makes it possible to see what
        frame the data belongs to, which is useful for debugging timing issues
        in the system.

        WARNING: If the counter-substituted IQ streams are allowed to go through
        phase estimation and downstream filtering, they will be hard to
        interpret. You can disable those steps with the following code::

            # disable downstream filtering
            S.set_filter_disable(True)
            S.set_downsample_factor(1)
            S._caput(S.smurf_processor + "Unwrapper:Disable",1)

            # select IQ streaming mode
            # bypasses CORDIC, send I and Q over both bays
            # in this case we select bands corresponding to bay 0
            S._caput(f'{S.app_core}baySelStream', 0, write_log=True)
            S._caput(f'{S.app_core}modeStream', 1, write_log=True)

        Args
        ----
        band : int
            The band number.

        Returns
        -------
        int
            The value of the counter select register.
        """
        return self._caget(self.band_root.format(band) + self._counter_select_reg, **kwargs)

    def set_counter_select(self, band, val, **kwargs):
        """
        Set the value of the counterSelect register for a given band.

        If this register is enabled, the value of I and Q will be substituted
        with the flux ramp frame counter. This makes it possible to see what
        frame the data belongs to, which is useful for debugging timing issues
        in the system.

        WARNING: If the counter-substituted IQ streams are allowed to go through
        phase estimation and downstream filtering, they will be hard to
        interpret. You can disable those steps with the following code::

            # disable downstream filtering
            S.set_filter_disable(True)
            S.set_downsample_factor(1)
            S._caput(S.smurf_processor + "Unwrapper:Disable",1)

            # select IQ streaming mode
            # bypasses CORDIC, send I and Q over both bays
            # in this case we select bands corresponding to bay 0
            S._caput(f'{S.app_core}baySelStream', 0, write_log=True)
            S._caput(f'{S.app_core}modeStream', 1, write_log=True)

        Args
        ----
        band : int
            The band number.
        val : int
            The value to set (0 or 1).

        Returns
        -------
        int
            The value of the counter select register.
        """
        return self._caput(self.band_root.format(band) + self._counter_select_reg, val, **kwargs)
