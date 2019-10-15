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


parser = argparse.ArgumentParser('Pyrogue Client')

parser.add_argument('--server',
                    type=str, 
                    help="Server address: 'host:port' or list of addresses: 'host1:port1,host2:port2'",
                    default='localhost:9099')

parser.add_argument('--ui',
                    type=str, 
                    help='UI File for gui (cmd=gui)',
                    default=None)

parser.add_argument('--details',
                    help='Show log details with stacktrace (cmd=syslog)',
                    action='store_true')

parser.add_argument('cmd',    
                    type=str, 
                    choices=['gui','syslog','monitor','get','value','set','exec'], 
                    help='Client command to issue')

parser.add_argument('path',
                    type=str,
                    nargs='?',
                    help='Path to access')

parser.add_argument('value',
                    type=str, 
                    nargs='?',
                    help='Value to set')

args = parser.parse_args()
















# Print the usage message
def usage(name):
    print("Usage: {}".format(name))
    print("        [-a|--addr IP_address] [-s|--server] [-e|--epics prefix]")
    print("        [-n|--nopoll] [-c|--commType comm_type] [-l|--pcie-rssi-lane index]")
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
    print("    -c|--commType comm_type     : Communication type with the FPGA",\
        "(defaults to \"eth-rssi-non-interleaved\"")
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
    comm_type = "eth-rssi-non-interleaved";
    comm_type_valid_types = ["eth-rssi-non-interleaved", "eth-rssi-interleaved", "pcie-rssi-interleaved"]
    pcie_rssi_link=None
    pv_dump_file= ""
    pcie_dev="/dev/datadev_0"
    disable_bay0=False
    disable_bay1=False
    disable_gc=False
    windows_title=""


    # Read Arguments
    try:
        opts, _ = getopt.getopt(sys.argv[1:],
            "ha:se:d:nb:f:c:l:u:w:",
            ["help", "addr=", "server", "epics=", "defaults=", "nopoll",
            "stream-size=", "stream-type=", "commType=", "pcie-rssi-link=", "dump-pvs=",
            "disable-bay0", "disable-bay1", "disable-gc", "windows-title=", "pcie-dev="])
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
        elif opt in ("-c", "--commType"):   # Communication type
            if arg in comm_type_valid_types:
                comm_type = arg
            else:
                print("Invalid communication type. Valid choises are:")
                for c in comm_type_valid_types:
                    print("  - \"{}\"".format(c))
                exit_message("ERROR: Invalid communication type")
        elif opt in ("-l", "--pcie-rssi-link"):       # PCIe RSSI Link
            pcie_rssi_link = int(arg)
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
        elif opt in ("--pcie-dev"):
            pcie_dev = arg



    # Verify if IP address is valid
    if ip_addr:
        try:
            socket.inet_pton(socket.AF_INET, ip_addr)
        except socket.error:
            exit_message("ERROR: Invalid IP Address.")




    # Check connection with the board if using eth communication
    if "eth-" in comm_type:
        if not ip_addr:
            exit_message("ERROR: Must specify an IP address for ethernet base communication devices.")

        print("")
        print("Trying to ping the FPGA...")
        try:
           dev_null = open(os.devnull, 'w')
           subprocess.check_call(["ping", "-c2", ip_addr], stdout=dev_null, stderr=dev_null)
           print("    FPGA is online")
           print("")
        except subprocess.CalledProcessError:
           exit_message("    ERROR: FPGA can't be reached!")



    if server_mode and not (epics_prefix):
        exit_message("    ERROR: Can not start in server mode without the EPICS server enabled")



    # Try to import the FpgaTopLevel definition
    try:
        from FpgaTopLevel import FpgaTopLevel
    except ImportError as ie:
        print("Error importing FpgaTopLevel: {}".format(ie))
        exit()



    # If EPICS server is enable, import the epics module
    if epics_prefix:
        import pyrogue.protocols.epics



    # Import the QT and gui modules if not in server mode
    if not server_mode:
        import pyrogue.gui





    # The PCIeCard object will take care of setting up the PCIe card (if present)
    with pysmurf.core.devices.PcieCard(link=pcie_rssi_link, comm_type=comm_type, ip_addr=ip_addr, dev=pcie_dev):

        with CmbPcie( pcieDev       = '/dev/datadev_0',
                      pcieDataDev   = '/dev/datadev_1',
                      pcieRssiLink  = 0,
                      configFile    = "",
                      epicsPrefix   = "",
                      disableBay0   = False,
                      disableBay1   = False,
                      streamPvSize  = 0,
                      streamPvType  = 0,
                      pvDumpFile    = None,
                      pollEn        = 

            # while 1 or gui
            pass
