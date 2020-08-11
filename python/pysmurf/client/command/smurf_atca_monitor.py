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
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
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
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
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
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
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
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
            self._board_temp_amc_reg.format(bay*2),**kwargs)

    _amc_asset_tag_reg = 'AMC[{}]:Product_Asset_Tag'

    def get_amc_asset_tag(
            self, bay, slot_number=None, atca_epics_root=None,
            **kwargs):
        r"""Returns the AMC asset tag.

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
        str or None
            AMC asset tag for the requested bay *e.g.* 'C03-A01-01'.
            If None, either the EPICS query timed out or the
            atca_monitor server isn't running.
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        # For some reason, the bay 0 AMC is at AMC[0] and the bay 1
        # AMC is at AMC[2], hence the bay*2.
        return self._caget(
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
            self._amc_asset_tag_reg.format(bay*2), as_string=True,
            **kwargs)
