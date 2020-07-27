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
from pysmurf.client.base import SmurfBase

class SmurfAtcaMonitorMixin(SmurfBase):

    _write_atca_monitor_state_reg = ":Crate:WriteState"

    def write_atca_monitor_state(self, val, **kwargs):
        """
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
        """
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
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
            self._board_temp_rtm_reg,**kwargs)

    _board_temp_amc0_reg = 'BoardTemp:AMC0'

    def get_board_temp_amc0(
            self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
            self._board_temp_amc0_reg,**kwargs)

    _board_temp_amc2_reg = 'BoardTemp:AMC2'

    def get_board_temp_amc2(
            self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
            self._board_temp_amc2_reg,**kwargs)

    _junction_temp_fpga_reg = 'JunctionTemp:FPG'

    def get_junction_temp_fpga(
            self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(
            f'{shelf_manager}:Crate:Slot_{slot_number}:' +
            self._junction_temp_fpga_reg,**kwargs)

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
        \**kwargs
            Arbitrary keyword arguments.  Passed directly to the
            `_caget` call.
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
