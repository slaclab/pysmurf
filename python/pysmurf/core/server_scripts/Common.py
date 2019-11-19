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
import getopt
import pyrogue
import socket
import subprocess
import os

# Print the usage message
def usage(name):
    print("Usage: {}".format(name))
    print("        [-z|--zip file] [-a|--addr IP_address] [-g|--gui]")
    print("        [-e|--epics prefix] [-n|--nopoll] [-l|--pcie-rssi-lane index]")
    print("        [-d|--defaults config_file] [-u|--dump-pvs file_name]")
    print("        [--disable-bay0] [--disable-bay1] [-w|--windows-title title]")
    print("        [--pcie-dev-rssi pice_device] [--pcie-dev-data pice_device] [-h|--help]")
    print("")
    print("    -h|--help                   : Show this message")
    print("    -z|--zip file               : Pyrogue zip file to be included in"\
        "the python path.")
    print("    -a|--addr IP_address        : FPGA IP address. Required when"\
        "the communication type is based on Ethernet.")
    print("    -d|--defaults config_file   : Default configuration file. If the path is"\
        "relative, it refers to the zip file (i.e: file.zip/config/config_file.yml).")
    print("    -e|--epics prefix           : Start an EPICS server with",\
        "PV name prefix \"prefix\"")
    print("    -g|--gui                    : Start the server with a GUI.")
    print("    -n|--nopoll                 : Disable all polling")
    print("    -l|--pcie-rssi-lane index   : PCIe RSSI lane (only needed with"\
        "PCIe). Supported values are 0 to 5")
    print("    -u|--dump-pvs file_name     : Dump the PV list to \"file_name\".",\
        "(Must be used with -e)")
    print("    --disable-bay0              : Disable the instantiation of the"\
        "devices for Bay0")
    print("    --disable-bay1              : Disable the instantiation of the"\
        "devices for Bay1")
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

# Parse the args
def get_args():

    # Defaults
    args = {'zip_file' : "",
            'ip_addr' : "",
            'epics_prefix' : "",
            'config_file' : "",
            'use_gui' : False,
            'polling_en' : True,
            'pcie_rssi_lane' : None,
            'pv_dump_file' : "",
            'pcie_dev_rssi' : "/dev/datadev_0",
            'pcie_dev_data' : "/dev/datadev_1",
            'disable_bay0' : False,
            'disable_bay1' : False,
            'windows_title' : "" }

    # Read Arguments
    try:
        opts, _ = getopt.getopt(sys.argv[1:],
            "hz:a:ge:d:nb:f:l:u:w:",
            ["help", "zip=", "addr=", "gui", "epics=", "defaults=", "nopoll",
            "pcie-rssi-link=", "dump-pvs=",
            "disable-bay0", "disable-bay1", "windows-title=", "pcie-dev-rssi=",
            "pcie-dev-data="])

    except getopt.GetoptError:
        usage(sys.argv[0])
        sys.exit()

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage(sys.argv[0])
            sys.exit()
        elif opt in ("-z", "--zip"):
            args['zip_file'] = arg
        elif opt in ("-a", "--addr"):        # IP Address
            args['ip_addr'] = arg
        elif opt in ("-g", "--gui"):         # Use a GUI
            args['use_gui'] = True
        elif opt in ("-e", "--epics"):       # EPICS prefix
            args['epics_prefix'] = arg
        elif opt in ("-n", "--nopoll"):      # Disable all polling
            args['polling_en'] = False
        elif opt in ("-d", "--defaults"):   # Default configuration file
            args['config_file'] = arg
        elif opt in ("-l", "--pcie-rssi-link"):       # PCIe RSSI Link
            args['pcie_rssi_lane'] = int(arg)
        elif opt in ("-u", "--dump-pvs"):   # Dump PV file
            args['pv_dump_file'] = arg
        elif opt in ("--disable-bay0"):
            args['disable_bay0'] = True
        elif opt in ("--disable-bay1"):
            args['disable_bay1'] = True
        elif opt in ("-w", "--windows-title"):
            args['windows_title'] = arg
        elif opt in ("--pcie-dev-rssi"):
            args['pcie_dev_rssi'] = arg
        elif opt in ("--pcie-dev-data"):
            args['pcie_dev_data'] = arg


    # If a zip file was specified and exist add it to the python path
    if args['zip_file'] and os.path.exists(args['zip_file']):
        pyrogue.addLibraryPath(args['zip_file']+"/python")

        # If the default configuration file was given using a relative path,
        # it is refereed to the zip file, so build the full path.
        if args['config_file'] and args['config_file'][0] != '/':
                args['config_file'] = args['zip_file'] + "/config/" + args['config_file']

    return args

def verify_ip(args):
    try:
        socket.inet_pton(socket.AF_INET, args['ip_addr'])
    except socket.error:
        sys.exit("ERROR: Invalid IP Address.")

def ping_fpga(args):
    print("")
    print("Trying to ping the FPGA...")
    try:
       dev_null = open(os.devnull, 'w')
       subprocess.check_call(["ping", "-c2", args['ip_addr']], stdout=dev_null, stderr=dev_null)
       print("    FPGA is online")
       print("")
    except subprocess.CalledProcessError:
       sys.exit("    ERROR: FPGA can't be reached!")

