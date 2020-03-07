#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Root Class For PCIE Express
#-----------------------------------------------------------------------------
# File       : CmbPcie.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
# Root class for PCI Express Ethernet
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
import rogue.hardware.axi
import rogue.protocols.srp

from pysmurf.core.roots.Common import Common

from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel

class CmbPcie(Common):
    def __init__(self, *,
                 pcie_rssi_lane = 0,
                 pcie_dev_rssi  = "/dev/datadev_0",
                 pcie_dev_data  = "/dev/datadev_1",
                 config_file    = None,
                 epics_prefix   = "EpicsPrefix",
                 polling_en     = True,
                 pv_dump_file   = "",
                 disable_bay0   = False,
                 disable_bay1   = False,
                 enable_pwri2c  = False,
                 txDevice       = None,
                 configure      = False,
                 VariableGroups = None,
                 server_port    = 0,
                 **kwargs):

        # TDEST 0 routed to streamr0 (SRPv3)
        self._srpStream = rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0),True)

        # Create the SRP Engine
        self._srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(self._srp, self._srpStream)

        # Instantiate Fpga top level
        self._fpga = FpgaTopLevel( memBase      = self._srp,
                                   disableBay0  = disable_bay0,
                                   disableBay1  = disable_bay1,
                                   enablePwrI2C = enable_pwri2c)

        # Create stream interfaces
        self._ddr_streams = []

        # DDR streams. We are only using the first 2 channel of each AMC daughter card, i.e.
        # channels 0, 1, 4, 5.
        for i in [0, 1, 4, 5]:
            tmp = rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0x80 + i), True)
            tmp.setZeroCopyEn(False)
            self._ddr_streams.append(tmp)

        # Streaming interface stream
        self._streaming_stream = \
            rogue.hardware.axi.AxiStreamDma(pcie_dev_data,(pcie_rssi_lane*0x100 + 0xC1), True)

        # Setup base class
        Common.__init__(self,
                        config_file    = config_file,
                        epics_prefix   = epics_prefix,
                        polling_en     = polling_en,
                        pv_dump_file   = pv_dump_file,
                        txDevice       = txDevice,
                        configure      = configure,
                        VariableGroups = VariableGroups,
                        server_port    = server_port,
                        **kwargs)
