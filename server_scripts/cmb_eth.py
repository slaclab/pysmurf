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

import sys

import pyrogue

import pysmurf.core.devices

import pysmurf.core.server_scripts.Common as common

# Main body
if __name__ == "__main__":

    # Read Arguments
    args = common.get_args()

    # Import the root device after the python path is updated
    from pysmurf.core.roots.CmbEth import CmbEth

    if not args['ip_addr']:
        sys.exit("ERROR: Must specify an IP address for ethernet base communication devices.")

    common.verify_ip(args)
    common.ping_fpga(args)

    # Define variable groups (we use the provided example definition)
    # We can disable it by defining "VariableGroups = None" instead.
    from pysmurf.core.server_scripts._VariableGroupExample import VariableGroups

    # The PCIeCard object will take care of setting up the PCIe card (if present)
    with pysmurf.core.devices.PcieCard( lane      = args['pcie_rssi_lane'],
                                        comm_type = "eth-rssi-interleaved",
                                        ip_addr   = args['ip_addr'],
                                        dev_rssi  = args['pcie_dev_rssi'],
                                        dev_data  = args['pcie_dev_data']):

        with CmbEth( ip_addr        = args['ip_addr'],
                     config_file    = args['config_file'],
                     epics_prefix   = args['epics_prefix'],
                     polling_en     = args['polling_en'],
                     pv_dump_file   = args['pv_dump_file'],
                     disable_bay0   = args['disable_bay0'],
                     disable_bay1   = args['disable_bay1'],
                     enable_pwri2c  = args['enable_em22xx'],
                     configure      = args['configure'],
                     server_port    = args['server_port'],
                     VariableGroups = VariableGroups) as root:

            if args['use_gui']:
                # Start the GUI
                print("Starting GUI...\n")

                if args['use_qt']:
                    # Start the QT GUI, is selected by the user
                    import pyrogue.gui
                    pyrogue.gui.runGui(root=root,title=args['windows_title'])
                else:
                    # Otherwise, start the PyDM GUI
                    import pyrogue.pydm
                    pyrogue.pydm.runPyDM(root=root, title=args['windows_title'])

            else:
                # Stop the server when Crtl+C is pressed
                print("Running without GUI...")
                pyrogue.waitCntrlC()
