#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf command module - SmurfAtcaMonitorMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/command/smurf_atca_monitor.py
# Created    : 2019-07-22
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
"""Defines the :class:`SmurfAtcaMonitorMixin` class."""
from pysmurf.client.base import SmurfBase
import subprocess

class SmurfAtcaMonitorMixin(SmurfBase):
    """Mixin providing interface with the atca_monitor server.

    This Mixin provides the pysmurf interface to the atca_monitor
    registers.  The atca_monitor server is a Rogue application
    which uses IPMI to monitor information from the ATCA system
    [#atca_monitor]_.  The atca_monitor server must be
    running or all queries will timeout and return `None`.

    References
    ----------
    .. [#atca_monitor] https://github.com/slaclab/smurf-atca-monitor

    """

    _write_atca_monitor_state_reg = ":Crate:SaveState"

    def write_atca_monitor_state(self, val, **kwargs):
        """Writes atca_monitor state to yml file.

        Writes all current ATCA monitor values to a yml file.

        Args
        ----
        val : str
           The path (including file name) to write the yml file to.

        """
        self._caput(
            self.shelf_manager + self._write_atca_monitor_state_reg,
            val, **kwargs)

    _board_temp_fpga_reg = 'BoardTemp:FPGA'

    def get_board_temp_fpga(
            self, slot_number=None, atca_epics_root=None, **kwargs):
        r"""Returns the AMC carrier board temperature.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the AMC carrier is installed
            into.  If None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        atca_epics_root : str or None, optional, default None
            ATCA monitor server application EPICS root.  If None,
            defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems, atca_epics_root is the name of the
            shelf manager which for default systems is
            'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            AMC carrier board temperature in Celsius.  If None, either
            the EPICS query timed out or the atca_monitor server
            isn't running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Sensors:Slots:{slot_number}:' +
            self._board_temp_fpga_reg,**kwargs)

    _board_temp_rtm_reg = 'BoardTemp:RTM'

    def get_board_temp_rtm(
            self, slot_number=None, atca_epics_root=None, **kwargs):
        r"""Returns the RTM board temperature.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the RTM is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        atca_epics_root : str or None, optional, default None
            ATCA monitor server application EPICS root.  If None,
            defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems, atca_epics_root is the name of the
            shelf manager which for default systems is
            'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            RTM board temperature in Celsius.  If None, either the
            EPICS query timed out or the atca_monitor server isn't
            running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Sensors:Slots:{slot_number}:' +
            self._board_temp_rtm_reg,**kwargs)

    _junction_temp_fpga_reg = 'JunctionTemp:FPG'

    def get_junction_temp_fpga(
            self, slot_number=None, atca_epics_root=None, **kwargs):
        r"""Returns FPGA junction temperature.

        FPGA die temperature - probably from a sensor on the FPGA.  If
        you are looking at this, you probably should be looking at
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_fpga_temp`
        instead, which we think is more reliable.

        Args
        ----
        slot_number : int or None, optional, default None
            The crate slot number that the FPGA carrier is installed
            into.  If None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        atca_epics_root : str or None, optional, default None
            ATCA monitor server application EPICS root.  If None,
            defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems, atca_epics_root is the name of the
            shelf manager which for default systems is
            'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            FPGA junction temperature in Celsius.  If None, either the
            EPICS query timed out or the atca_monitor server isn't
            running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Sensors:Slots:{slot_number}:' +
            self._junction_temp_fpga_reg,**kwargs)

    _board_temp_amc_reg = 'BoardTemp:AMC{}'

    def get_board_temp_amc(self, bay, slot_number=None,
                           atca_epics_root=None, **kwargs):
        r"""Returns the AMC board temperature.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        slot_number : int or None, optional, default None
            The crate slot number that the AMC is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        atca_epics_root : str or None, optional, default None
            ATCA monitor server application EPICS root.  If None,
            defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems, atca_epics_root is the name of the
            shelf manager which for default systems is
            'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        float or None
            AMC board temperature in Celsius.  If None, either the
            EPICS query timed out or the atca_monitor server isn't
            running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        # For some reason, the bay 0 AMC is at AMC[0] and the bay 1
        # AMC is at AMC[2], hence the bay*2.
        return self._caget(
            f'{shelf_manager}:Crate:Sensors:Slots:{slot_number}:' +
            self._board_temp_amc_reg.format(bay*2),**kwargs)

    _amc_asset_tag_reg = 'Product_Asset_Tag'

    def get_amc_sn(
            self, bay, slot_number=None,
            atca_epics_root=None,
            shelf_manager=None,
            use_shell=False,
            **kwargs):
        r"""Returns the AMC serial number.

        The AMC serial number is the combination of its 'Product
        Version' and 'Product Asset Tag' from its FRU data.  A common
        example (the production AMCs built for Simons Observatory) is
        'Product Version'=C03 and 'Product Asset Tag'=A01-11', which
        combine to make the full AMC serial number C03-A01-11.

        By default, will try to get the serial number by querying the
        ATCA monitor EPICS server.  If you're not runing the ATCA
        monitor, can still get the AMC serial number more slowly via
        the shell by providing use_shell=True.

        C03 refers to the hardware revision of the AMC base board.
        The A## refers to the specific AMC baseboard loading.  The two
        most common SMuRF AMC base board loadings are A01 and A02
        corresponding to low band (4-6 GHz) and high band (6-8 GHz)
        AMCs.  The final number in the full serial number is the
        unique id assigned to each AMC base board which shares the
        same hardware revision and loading.  

        Typical SMuRF AMC assemblies are composed of two connected
        boards, an AMC base board and an AMC RF daughter board.  The
        serial number of an AMC RF daughter card in an AMC assembly
        cannot be obtained remotely; you must either know which card
        your AMC was assembled with (or you can ask SLAC which
        maintains a database with this information) or it should be
        labeled on the frontpanel of your AMC assembly.

        Args
        ----
        bay : int
            Which AMC bay (0 or 1).
        slot_number : int or None, optional, default None
            The crate slot number that the AMC is installed into.  If
            None, defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.slot_number`.
        atca_epics_root : str or None, optional, default None
            ATCA monitor server application EPICS root.  If None,
            defaults to the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems, atca_epics_root is the name of the
            shelf manager which for default systems is
            'shm-smrf-sp01'.
        use_shell : bool, optional, default False
            If False, polls the ATCA monitor EPICs server ; if True,
            runs slower shell command to poll this attribute.  This
            will be slower but provides an alternative if user is not
            running the ATCA monitor as part of their workflow, or if
            the ATCA monitor is down.
        shelf_manager : str or None, optional, default None
            Only used if use_shell=True.  If None, defaults to
            the
            :class:`~pysmurf.client.base.smurf_control.SmurfControl`
            class attribute
            :attr:`~pysmurf.client.base.smurf_control.SmurfControl.shelf_manager`.
            For typical systems the default name of the shelf manager
            is 'shm-smrf-sp01'.
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.

        Returns
        -------
        str or None
            AMC asset tag for the requested bay *e.g.* 'C03-A01-01'.
            If None, either the EPICS query timed out or the
            atca_monitor server isn't running, or if running with
            use_shell=True, the shell command failed.  Also
            returns None if there's no AMC in the requested bay if
            use_shell=True or if use_shell=True and the
            `cba_amc_init` command used to poll the AMC FRU fails.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            atca_epics_root=self.shelf_manager
        if shelf_manager is None:
            shelf_manager=self.shelf_manager            
        if use_shell:
            amc_desc_dict = self.cba_amc_init(bay)
            if amc_desc_dict is not None and amc_desc_dict.keys()>={'Product Version', 'Product Asset Tag'}:
                return f'{amc_desc_dict["Product Version"]}-{amc_desc_dict["Product Asset Tag"]}'
            else:
                self.log('cba_amc_init response missing "Product Version" and/or "Product Asset Tag" fields.  Returning None.',
                         self.LOG_ERROR)                
                return None
        else:
            # For some reason, the bay 0 AMC is at AMC[0] and the bay 1
            # AMC is at AMC[2], hence the bay*2.
            return self._caget(
                f'{atca_epics_root}:Crate:Sensors:Slots:{slot_number}:' +
                f'AMCInfo:{bay*2}:' +
                self._amc_asset_tag_reg, as_string=True,
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

    def cba_amc_init(self,bay,slot_number=None,shelf_manager=None):
        r"""Returns AMC FRU info for requested bay

        Wrapper for dumping the FRU information for the AMC in the
        requested bay using the cba_amc_init shell command.

        Args
        ----
        bay : int
            Which bay to return the AMC FRU information for.
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
        amc_desc_dict : dict
            Dictionary of AMC FRU information for requested bay.
            Returns None if slot not present in shelf, or if no AMC is
            up in the requested bay.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if shelf_manager is None:
            shelf_manager=self.shelf_manager            
        
        stdout,stderr=self.shell_command(f'cba_amc_init -d {shelf_manager}/{slot_number}/{bay*2}')
        # Parse for AMC asset tag for this bay
        if 'AMC not present in bay' in stdout:
            self.log('ERROR : AMC not present in bay!',
                     self.LOG_ERROR)
            return None
        if 'Slot not present in shelf' in stdout:
            self.log('ERROR : Slot not present in shelf!',
                     self.LOG_ERROR)
            return None                
        else: # return full AMC asset tag
            stdout=stdout.split('\n')
            amc_desc_dict={}
            for line in stdout:
                if ':' in line:
                    splitline=line.split(':')
                    if len(splitline)==2:
                        amcc_desc_key=splitline[0].lstrip().rstrip()
                        amcc_desc_value=splitline[1].lstrip().rstrip()
                        if len(amcc_desc_value)>0: # skip header
                            amc_desc_dict[amcc_desc_key]=amcc_desc_value

        return amc_desc_dict
