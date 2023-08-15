#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Root Class For Ethernet
#-----------------------------------------------------------------------------
# File       : CmbEth.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
# Root class for Ethernet
#-----------------------------------------------------------------------------
# This file is part of the AmcCarrier Core. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the AmcCarrierCore, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel
import pyrogue
import rogue.protocols.srp

import pysmurf
from pysmurf.core.roots.Common import Common

class CmbEth(Common):
    def __init__(self, *,
                 ip_addr        = "",
                 config_file    = None,
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


        # Create Interleaved RSSI interface
        self._stream = pyrogue.protocols.UdpRssiPack(name='rudp', host=ip_addr, port=8198, packVer = 2, jumbo = True)

        # Create the SRP Engine
        self._srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(self._srp, self._stream.application(dest=0x0))

        # Instantiate Fpga top level
        # In order to be backwards compatible for now, also support
        # FpgaTopLevel which doesn't have the enablePwrI2C argument.
        try:
            self._fpga = FpgaTopLevel( memBase      = self._srp,
                                       disableBay0  = disable_bay0,
                                       disableBay1  = disable_bay1,
                                       enablePwrI2C = enable_pwri2c)
        except TypeError as e:
            print(f"TypeError calling FpgaTopLevel: {e}")
            print("This FpgaTopLevel does not support the option 'enablePwrI2C'.")
            print("Please use a pyrogue zip file which is up to date.")
            print("Staring the server without using the 'enablePwrI2C' option.")
            self._fpga = FpgaTopLevel( memBase      = self._srp,
                                       disableBay0  = disable_bay0,
                                       disableBay1  = disable_bay1)

        # Create ddr stream interfaces for base class
        self._ddr_streams = []

        # DDR streams. The FpgaTopLevel class will defined a 'stream' interface exposing them.
        # We are only using the first 2 channel of each AMC daughter card, i.e. channels 0, 1, 4, 5.
        for i in [0, 1, 4, 5]:
            self._ddr_streams.append(self._stream.application(0x80 + i))

        # Streaming interface stream. It comes over UDP, port 8195, without RSSI,
        # so we an UdpReceiver.
        self._streaming_stream_int = pysmurf.core.devices.UdpReceiver(ip_addr=ip_addr, port=8195)

        # When Ethernet communication is used, We use a FIFO between the stream data and the receiver:
        # Stream -> FIFO -> smurf_processor receiver
        self._smurf_processor_fifo = rogue.interfaces.stream.Fifo(100000,0,True)
        pyrogue.streamConnect(self._streaming_stream_int, self._smurf_processor_fifo)

        # Set streaming variable for base class
        self._streaming_stream = self._smurf_processor_fifo

        # Setup base class
        Common.__init__(self,
                        config_file    = config_file,
                        polling_en     = polling_en,
                        pv_dump_file   = pv_dump_file,
                        txDevice       = txDevice,
                        configure      = configure,
                        VariableGroups = VariableGroups,
                        disable_bay0   = disable_bay0,
                        disable_bay1   = disable_bay1,
                        **kwargs)
