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

# Setup python path for testing
pyrogue.addLibraryPath("../python")
pyrogue.addLibraryPath("../lib")
pyrogue.addLibraryPath("../../cryo-det/firmware/python")
pyrogue.addLibraryPath("../../cryo-det/firmware/submodules/amc-carrier-core/python")
#pyrogue.addLibraryPath("../../cryo-det/firmware/submodules/axi-pcie-core/python")
pyrogue.addLibraryPath("../../cryo-det/firmware/submodules/lcls-timing-core/python")
pyrogue.addLibraryPath("../../cryo-det/firmware/submodules/surf/python")

import pyrogue.utilities.fileio

from pysmurf.core.roots.EmulationRoot import EmulationRoot

#rogue.Logging.setFilter('pyrogue.epicsV3.Value',rogue.Logging.Debug)
#rogue.Logging.setLevel(rogue.Logging.Debug)

#logger = logging.getLogger('pyrogue')
#logger.setLevel(logging.DEBUG)

# Main body
if __name__ == "__main__":

    with EmulationRoot ( config_file    = "",
                         epics_prefix   = "Test",
                         polling_en     = True,
                         pv_dump_file   = "epics_dump.txt",
                         disable_bay0   = False,
                         disable_bay1   = False) as root:

        root.saveVariableList("varlist.txt")

        # Start the GUI
        import pyrogue.pydm
        print("Starting GUI...\n")
        pyrogue.pydm.runPyDM(root=root)

