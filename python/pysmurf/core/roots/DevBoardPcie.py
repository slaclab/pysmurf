#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Root Class For Dev Board using PCI Express Communication
#-----------------------------------------------------------------------------
# File       : CmbPcie.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
# Root Class For Dev Board using PCI Express Communication
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
import rogue.hardware.axi
import rogue.protocols.srp

from pysmurf.core.roots.Common import Common

from  CryoDevBoard.Kcu105Eth import FpgaTopLevel as FpgaTopLevel

class DevBoardPcie(Common):
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
                 txDevice       = None,
                 configure      = False,
                 server_port    = 0,
                 **kwargs):

        # TDEST 0 routed to streamr0 (SRPv3)
        self.dma = rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0),True)
        self.srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir( self.srp, self.dma )

        # Instantiate Fpga top level
        self._fpga = FpgaTopLevel( memBase      = self.srp,
                                   ipAddr       = "",
                                   commType     = "pcie-rssi-interleaved",
                                   pcieRssiLink = pcie_rssi_lane,
                                   disableBay0  = disable_bay0,
                                   disableBay1  = disable_bay1)

        # Create stream interfaces
        self._ddr_streams = []

        # DDR streams. We are only using the first 2 channel of each AMC daughter card, i.e.
        # channels 0, 1, 4, 5.
        for i in [0, 1, 4, 5]:
            self._ddr_streams.append(
                rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0x80 + i), True))

        # Streaming interface stream
        self._streaming_stream = \
            rogue.hardware.axi.AxiStreamDma(pcie_dev_data,(pcie_rssi_lane*0x100 + 0xC1), True)

        # Setup base class
        Common.__init__(self, config_file    = config_file,
                              epics_prefix   = epics_prefix,
                              polling_en     = polling_en,
                              pv_dump_file   = pv_dump_file,
                              txDevice       = txDevice,
                              configure      = configure,
                              server_port    = server_port,
                              **kwargs)

