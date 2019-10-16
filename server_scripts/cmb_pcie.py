#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : PyRogue Server
#-----------------------------------------------------------------------------
# File       : python/pyrogue_server.py
# Created    : 2017-06-20
#-----------------------------------------------------------------------------
# Description:
# Python script to start a PyRogue Control Server
#-----------------------------------------------------------------------------
# This file is part of the pyrogue-control-server software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import argparse
import sys
import getopt
import socket
import os
import subprocess
import time
import struct
from packaging import version
import re

import pyrogue
import pyrogue.utilities.fileio
import rogue.interfaces.stream

import pysmurf.core.devices
import pysmurf.core.roots

# Print the usage message
def usage(name):
    print("Usage: {}".format(name))
    print("        [-a|--addr IP_address] [-s|--server] [-e|--epics prefix]")
    print("        [-n|--nopoll] [-l|--pcie-rssi-lane index]")
    print("        [-f|--stream-type data_type] [-b|--stream-size byte_size]")
    print("        [-d|--defaults config_file] [-u|--dump-pvs file_name] [--disable-gc]")
    print("        [--disable-bay0] [--disable-bay1] [-w|--windows-title title]")
    print("        [--pcie-dev-rssi pice_device] [--pcie-dev-data pice_device] [-h|--help]")
    print("")
    print("    -h|--help                   : Show this message")
    print("    -a|--addr IP_address        : FPGA IP address. Required when"\
        "the communication type is based on Ethernet.")
    print("    -d|--defaults config_file   : Default configuration file")
    print("    -e|--epics prefix           : Start an EPICS server with",\
        "PV name prefix \"prefix\"")
    print("    -s|--server                 : Server mode, without staring",\
        "a GUI (Must be used with -p and/or -e)")
    print("    -n|--nopoll                 : Disable all polling")
    print("    -l|--pcie-rssi-lane index   : PCIe RSSI lane (only needed with"\
        "PCIe). Supported values are 0 to 5")
    print("    -b|--stream-size data_size  : Expose the stream data as EPICS",\
        "PVs. Only the first \"data_size\" points will be exposed.",\
        "(Must be used with -e)")
    print("    -f|--stream-type data_type  : Stream data type (UInt16, Int16,",\
        "UInt32 or Int32). Default is UInt16. (Must be used with -e and -b)")
    print("    -u|--dump-pvs file_name     : Dump the PV list to \"file_name\".",\
        "(Must be used with -e)")
    print("    --disable-bay0              : Disable the instantiation of the"\
        "devices for Bay0")
    print("    --disable-bay1              : Disable the instantiation of the"\
        "devices for Bay1")
    print("    --disable-gc                : Disable python's garbage collection"\
        "(enabled by default)")
    print("    -w|--windows-title title    : Set the GUI windows title. If not"\
        "specified, the default windows title will be the name of this script."\
        "This value will be ignored when running in server mode.")
    print("    --pcie-dev-rssi pice_device : Set the PCIe card device name"\
        "used for RSSI (defaults to '/dev/datadev_0')")
    print("    --pcie-dev-data pice_device : Set the PCIe card device name"\
        "used for data (defaults to '/dev/datadev_1')")
    print("")
    print("Examples:")
    print("    {} -a IP_address              :".format(name),\
        " Start a local rogue server, with GUI, without an EPICS servers")
    print("    {} -a IP_address -e prefix    :".format(name),\
        " Start a local rogue server, with GUI, with and EPICS server")
    print("    {} -a IP_address -e prefix -s :".format(name),\
        " Start a local rogue server, without GUI, with an EPICS servers")
    print("")


# Main body
if __name__ == "__main__":
    ip_addr = ""
    epics_prefix = ""
    config_file = ""
    server_mode = False
    polling_en = True
    stream_pv_size = 0
    stream_pv_type = "UInt16"
    stream_pv_valid_types = ["UInt16", "Int16", "UInt32", "Int32"]
    pcie_rssi_lane=None
    pv_dump_file= ""
    pcie_dev_rssi="/dev/datadev_0"
    pcie_dev_data="/dev/datadev_1"
    disable_bay0=False
    disable_bay1=False
    disable_gc=False
    windows_title=""

    # Only Rogue version >= 2.6.0 are supported. Before this version the EPICS
    # interface was based on PCAS which is not longer supported.
    try:
        ver = pyrogue.__version__
        if (version.parse(ver) <= version.parse('2.6.0')):
            raise ImportError('Rogue version <= 2.6.0 is unsupported')
    except AttributeError:
        print("Error when trying to get the version of Rogue")
        pritn("Only version of Rogue > 2.6.0 are supported")
        raise

    # Read Arguments
    try:
        opts, _ = getopt.getopt(sys.argv[1:],
            "ha:se:d:nb:f:l:u:w:",
            ["help", "addr=", "server", "epics=", "defaults=", "nopoll",
            "stream-size=", "stream-type=", "pcie-rssi-link=", "dump-pvs=",
            "disable-bay0", "disable-bay1", "disable-gc", "windows-title=", "pcie-dev-rssi=",
            "pcie-dev-data="])
    except getopt.GetoptError:
        usage(sys.argv[0])
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(sys.argv[0])
            sys.exit()
        elif opt in ("-a", "--addr"):        # IP Address
            ip_addr = arg
        elif opt in ("-s", "--server"):      # Server mode
            server_mode = True
        elif opt in ("-e", "--epics"):       # EPICS prefix
            epics_prefix = arg
        elif opt in ("-n", "--nopoll"):      # Disable all polling
            polling_en = False
        elif opt in ("-b", "--stream-size"): # Stream data size (on PVs)
            try:
                stream_pv_size = int(arg)
            except ValueError:
                exit_message("ERROR: Invalid stream PV size")
        elif opt in ("-f", "--stream-type"): # Stream data type (on PVs)
            if arg in stream_pv_valid_types:
                stream_pv_type = arg
            else:
                print("Invalid data type. Using {} instead".format(stream_pv_type))
        elif opt in ("-d", "--defaults"):   # Default configuration file
            config_file = arg
        elif opt in ("-l", "--pcie-rssi-link"):       # PCIe RSSI Link
            pcie_rssi_lane = int(arg)
        elif opt in ("-u", "--dump-pvs"):   # Dump PV file
            pv_dump_file = arg
        elif opt in ("--disable-bay0"):
            disable_bay0 = True
        elif opt in ("--disable-bay1"):
            disable_bay1 = True
        elif opt in ("--disable-gc"):
            disable_gc = True
        elif opt in ("-w", "--windows-title"):
            windows_title = arg
        elif opt in ("--pcie-dev-rssi"):
            pcie_dev_rssi = arg
        elif opt in ("--pcie-dev-data"):
            pcie_dev_data = arg

    # Disable garbage collection if requested
    if disable_gc:
        import gc
        gc.disable()
        print("GARBAGE COLLECTION DISABLED")

    # Verify if IP address is valid
    if ip_addr:
        try:
            socket.inet_pton(socket.AF_INET, ip_addr)
        except socket.error:
            exit_message("ERROR: Invalid IP Address.")

    if server_mode and not (epics_prefix):
        exit_message("    ERROR: Can not start in server mode without the EPICS server enabled")

    # If EPICS server is enable, import the epics module
    if epics_prefix:
        import pyrogue.protocols.epics

    # Import the QT and gui modules if not in server mode
    if not server_mode:
        import pyrogue.gui

    # The PCIeCard object will take care of setting up the PCIe card (if present)
    with pysmurf.core.devices.PcieCard( lane      = pcie_rssi_lane,
                                        comm_type = "pcie-rssi-interleaved",
                                        ip_addr   = ip_addr,
                                        dev_rssi  = pcie_dev_rssi,
                                        dev_data  = pcie_dev_data):

        with pysmurf.core.roots.CmbPcie( ip_addr        = ip_addr,
                      config_file    = config_file,
                      epics_prefix   = epics_prefix,
                      polling_en     = polling_en,
                      pcie_rssi_lane = pcie_rssi_lane,
                      stream_pv_size = stream_pv_size,
                      stream_pv_type = stream_pv_type,
                      pv_dump_file   = pv_dump_file,
                      disable_bay0   = disable_bay0,
                      disable_bay1   = disable_bay1,
                      disable_gc     = disable_gc,
                      pcie_dev_rssi  = pcie_dev_rssi,
                      pcie_dev_data  = pcie_dev_data):

            if not server_mode:
                app_top = pyrogue.gui.application(sys.argv)
                app_top.setApplicationName(windows_title)
                gui_top = pyrogue.gui.GuiTop(group='GuiTop')
                gui_top.addTree(CmbPcie)
                print("Starting GUI...\n")
            else:
                # Stop the server when Crtl+C is pressed
                print("")
                print("Running in server mode now. Press Ctrl+C to stop...")
                try:
                    # Wait for Ctrl+C
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass

