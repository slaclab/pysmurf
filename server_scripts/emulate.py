#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : PyRogue Server startup script
#-----------------------------------------------------------------------------
# File       : cmb_eth.py
# Created    : 2017-06-20
#-----------------------------------------------------------------------------
# Description:
# Python script to start a PyRogue server using ETH communication
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue

import pysmurf.core.devices
import pysmurf.core.transmitters
import pysmurf.core.server_scripts.Common as common

# Main body
if __name__ == "__main__":

    # Read Arguments
    args = common.get_args()

    # Import the root device after the python path is updated
    from pysmurf.core.roots.EmulationRoot import EmulationRoot

    with EmulationRoot ( config_file    = args['config_file'],
                         epics_prefix   = args['epics_prefix'],
                         polling_en     = args['polling_en'],
                         pv_dump_file   = args['pv_dump_file'],
                         disable_bay0   = args['disable_bay0'],
                         disable_bay1   = args['disable_bay1'],
                         server_port    = args['server_port'],
                         txDevice       = pysmurf.core.transmitters.BaseTransmitter(name='Transmitter')) as root:

        # Add dummy TES bias values ([-8:7]), for testing purposes.
        for i in range(16):
            root._smurf_processor.setTesBias(index=i, val=(i-8))

        # Start the GUI
        import pyrogue.gui
        print("Starting GUI...\n")
        pyrogue.gui.runGui(root=root)

