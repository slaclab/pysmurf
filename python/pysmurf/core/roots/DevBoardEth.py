#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Root Class For Dev Board using Ethernet Communication
#-----------------------------------------------------------------------------
# File       : CmbPcie.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
# Root Class For Dev Board using Ethernet Communication
#-----------------------------------------------------------------------------
# This file is part of the AmcCarrier Core. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the AmcCarrierCore, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

from CryoDevBoard.Kcu105Eth import FpgaTopLevel as FpgaTopLevel
import pyrogue
import rogue.protocols.srp

import pysmurf
from pysmurf.core.roots.Common import Common

class DevBoardEth(Common):
    def __init__(self, *,
                 ip_addr        = "",
                 config_file    = None,
                 polling_en     = True,
                 pv_dump_file   = "",
                 disable_bay0   = False,
                 disable_bay1   = False,
                 txDevice       = None,
                 configure      = False,
                 server_port    = 0,
                 **kwargs):

        # Create Interleaved RSSI interface
        self._stream = pyrogue.protocols.UdpRssiPack(name='rudp', host=ip_addr, port=8198, packVer = 2, jumbo = True)

        # Connect the SRPv3 to tDest = 0x0
        self._srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir( self._srp, self._stream.application(dest=0x0) )

        # Instantiate Fpga top level
        self._fpga = FpgaTopLevel( memBase      = self._srp,
                                   ipAddr       = ip_addr,
                                   commType     = "eth-rssi-interleaved",
                                   pcieRssiLink = 0, # Not needed
                                   disableBay0  = disable_bay0,
                                   disableBay1  = disable_bay1)

        # Create stream interfaces
        self._ddr_streams = []

        # DDR streams. The FpgaTopLevel class will defined a 'stream' interface exposing them.
        # We are only using the first 2 channel of each AMC daughter card, i.e. channels 0, 1, 4, 5.
        for i in [0, 1, 4, 5]:
            self._ddr_streams.append(self._stream.application(0x80 + i))

        # Streaming interface stream. It comes over UDP, port 8195, without RSSI,
        # so we an UdpReceiver.
        self._streaming_stream = pysmurf.core.devices.UdpReceiver(ip_addr=ip_addr, port=8195)

        # Setup base class
        Common.__init__(self,
                        config_file    = config_file,
                        polling_en     = polling_en,
                        pv_dump_file   = pv_dump_file,
                        txDevice       = txDevice,
                        configure      = configure,
                        server_port    = server_port,
                        **kwargs)
