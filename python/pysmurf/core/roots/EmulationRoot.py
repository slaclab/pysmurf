#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Emulation Root
#-----------------------------------------------------------------------------
# File       : EmulationRoot.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
# Emulation Root Class
#-----------------------------------------------------------------------------
# This file is part of the AmcCarrier Core. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the AmcCarrierCore, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue
import pysmurf
import rogue.protocols.srp
import Common

from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel

class EmulationRoot(Common.Common):
    def __init__(self, *,
                 config_file    = None,
                 epics_prefix   = "EpicsPrefix",
                 polling_en     = True,
                 pv_dump_file   = "",
                 disable_bay0   = False,
                 disable_bay1   = False,
                 txDevice       = None,
                 **kwargs):

        # Create the SRP Engine
        self._srp = pyrogue.interfaces.simulation.MemEmulate()

        # Instantiate Fpga top level
        self.add(FpgaTopLevel( memBase      = self._srp,
                               disableBay0  = disable_bay0,
                               disableBay1  = disable_bay1))

        # Create ddr stream interfaces for base class
        self._ddr_streams = [rogue.interfaces.stream.Master()] * 4

        # Set streaming variable for base class
        self._streaming_stream = rogue.itnterfaces.stream.Master()

        # Setup base class
        Common.Common.__init__(self, config_file    = config_file,
                                     epics_prefix   = epics_prefix,
                                     polling_en     = polling_en,
                                     pv_dump_file   = pv_dump_file,
                                     txDevice       = txDevice,
                                     **kwargs)

