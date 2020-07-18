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
import time
import epics

class SmurfCommandMixin(object):

    def write_ps_en(self, enables):
        """
        Write the power supply enable signals.

        Args
        ----
        enables (int): 2-bit number to set the power supplies enables.
           Bit 0 set the enable for HEMT power supply.
           Bit 1 set the enable for 50k power supply.
           Bit set to 1 mean enable power supply.
           Bit set to 0 mean disable the power supply.

        """
        epics.caput(self.writepv, cmd_make(0, self.ps_en_address, enables))    
