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

    _write_atca_monitor_state = ":Crate:WriteState"

    def write_atca_monitor_state(self, val, **kwargs):

        """
        Writes all current ATCA monitor values to a yml file.

        Args
        ----
        val (str) : The path (including file name) to write the yml file to.
        """
        self._caput(self.shelf_manager + self._write_atca_monitor_state,
                    val, **kwargs)

    _board_temp_fpga = 'BoardTemp:FPGA'

    def get_board_temp_fpga(self, slot_number=None, atca_epics_root=None, **kwargs):

        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(f'{shelf_manager}:Crate:Slot_{slot_number}:' +
                           self._board_temp_fpga,**kwargs)

    _board_temp_rtm = 'BoardTemp:RTM'

    def get_board_temp_rtm(self, slot_number=None, atca_epics_root=None, **kwargs):

        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(f'{shelf_manager}:Crate:Slot_{slot_number}:' +
                           self._board_temp_rtm,**kwargs)

    _board_temp_amc0 = 'BoardTemp:AMC0'

    def get_board_temp_amc0(self, slot_number=None, atca_epics_root=None, **kwargs):

        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(f'{shelf_manager}:Crate:Slot_{slot_number}:' +
                           self._board_temp_amc0,**kwargs)

    _board_temp_amc2 = 'BoardTemp:AMC2'

    def get_board_temp_amc2(self, slot_number=None, atca_epics_root=None, **kwargs):

        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(f'{shelf_manager}:Crate:Slot_{slot_number}:' +
                           self._board_temp_amc2,**kwargs)

    _junction_temp_fpga = 'JunctionTemp:FPG'

    def get_junction_temp_fpga(self, slot_number=None, atca_epics_root=None, **kwargs):

        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager
        return self._caget(f'{shelf_manager}:Crate:Slot_{slot_number}:' +
                           self._junction_temp_fpga,**kwargs)
