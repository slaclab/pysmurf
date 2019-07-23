import numpy as np
import os
import epics
import time
from pysmurf.base import SmurfBase
from pysmurf.command.sync_group import SyncGroup as SyncGroup

class SmurfAtcaMonitorMixin(SmurfBase):

    _write_atca_monitor_state = ":Crate:WriteState"
    def write_atca_monitor_state(self, val, **kwargs):
        """
        Writes all current ATCA monitor values to a yml file.

        Args:
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
        return self._caget('{}:Crate:Slot_{}:'.format(shelf_manager,slot_number) + \
                           self._board_temp_fpga,**kwargs)

    _board_temp_rtm = 'BoardTemp:RTM'
    def get_board_temp_rtm(self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager            
        return self._caget('{}:Crate:Slot_{}:'.format(shelf_manager,slot_number) + \
                           self._board_temp_rtm,**kwargs)

    _board_temp_amc0 = 'BoardTemp:AMC0'
    def get_board_temp_amc0(self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager            
        return self._caget('{}:Crate:Slot_{}:'.format(shelf_manager,slot_number) + \
                           self._board_temp_amc0,**kwargs)

    _board_temp_amc2 = 'BoardTemp:AMC2'
    def get_board_temp_amc2(self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager            
        return self._caget('{}:Crate:Slot_{}:'.format(shelf_manager,slot_number) + \
                           self._board_temp_amc2,**kwargs)

    _junction_temp_fpga = 'JunctionTemp:FPG'
    def get_junction_temp_fpga(self, slot_number=None, atca_epics_root=None, **kwargs):
        """
        """
        if slot_number is None:
            slot_number=self.slot_number
        if atca_epics_root is None:
            shelf_manager=self.shelf_manager            
        return self._caget('{}:Crate:Slot_{}:'.format(shelf_manager,slot_number) + \
                           self._junction_temp_fpga,**kwargs)
    
