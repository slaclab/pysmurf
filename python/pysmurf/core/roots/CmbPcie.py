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


# For RFSoC systems the FpgaTopLevel subclass that defaults isRFSOC=True is used.
# For standard ATCA systems the BpEthGen2 class is used directly.
# The import is deferred to a function so the correct zip/PYTHONPATH is already
# in place before the import is attempted.
def _import_fpga_top_level(is_rfsoc=False):
    if is_rfsoc:
        from CryoDet._MicrowaveMuxZcu208 import FpgaTopLevel
    else:
        from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel
    return FpgaTopLevel

class CmbPcie(Common):
    def __init__(self, *,
                 pcie_rssi_lane = 0,
                 pcie_dev_rssi  = "/dev/datadev_0",
                 pcie_dev_data  = "/dev/datadev_1",
                 config_file    = None,
                 polling_en     = True,
                 pv_dump_file   = "",
                 disable_bay0   = False,
                 disable_bay1   = False,
                 is_rfsoc       = False,
                 is_prespectra  = False,
                 enable_pwri2c  = False,
                 txDevice       = None,
                 configure      = False,
                 VariableGroups = None,
                 server_port    = 0,
                 **kwargs):

        # Set this once, before creating any instances
        rogue.hardware.axi.AxiStreamDma.zeroCopyDisable(pcie_dev_rssi)

        # TDEST 0 routed to streamr0 (SRPv3)
        self._srpStream = rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0),True)

        # Create the SRP Engine
        self._srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir(self._srp, self._srpStream)

        # Instantiate Fpga top level. Import is deferred until here so that the
        # correct pyrogue zip is already on sys.path before the import is attempted.
        # For RFSoC systems this resolves to CryoDet._MicrowaveMuxZcu208.FpgaTopLevel
        # (which defaults isRFSOC=True); for ATCA systems to _MicrowaveMuxBpEthGen2.
        FpgaTopLevel = _import_fpga_top_level(is_rfsoc)

        # In order to be backwards compatible for now, also support
        # FpgaTopLevel which doesn't have the enablePwrI2C argument.
        try:
            self._fpga = FpgaTopLevel( memBase      = self._srp,
                                       disableBay0  = disable_bay0,
                                       disableBay1  = disable_bay1,
                                       isRFSOC      = is_rfsoc,
                                       isPreSpectra = is_prespectra,
                                       enablePwrI2C = enable_pwri2c)
        except TypeError as e:
            print(f"TypeError calling FpgaTopLevel: {e}")
            print("This FpgaTopLevel does not support the option 'enablePwrI2C'.")
            print("Please use a pyrogue zip file which is up to date.")
            print("Staring the server without using the 'enablePwrI2C' option.")
            self._fpga = FpgaTopLevel( memBase      = self._srp,
                                       disableBay0  = disable_bay0,
                                       disableBay1  = disable_bay1,
                                       isRFSOC      = is_rfsoc,
                                       isPreSpectra = is_prespectra)

        # Create stream interfaces
        self._ddr_streams = []

        # DDR streams. We are only using the first 2 channel of each AMC daughter card, i.e.
        # channels 0, 1, 4, 5.
        for i in [0, 1, 4, 5]:
            tmp = rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0x80 + i), True)
            self._ddr_streams.append(tmp)

        # Streaming interface stream
        self._streaming_stream = \
            rogue.hardware.axi.AxiStreamDma(pcie_dev_data,(pcie_rssi_lane*0x100 + 0xC1), True)

        # Setup base class
        Common.__init__(self,
                        config_file    = config_file,
                        polling_en     = polling_en,
                        pv_dump_file   = pv_dump_file,
                        txDevice       = txDevice,
                        configure      = configure,
                        VariableGroups = VariableGroups,
                        server_port    = server_port,
                        disable_bay0   = disable_bay0,
                        disable_bay1   = disable_bay1,
                        **kwargs)
