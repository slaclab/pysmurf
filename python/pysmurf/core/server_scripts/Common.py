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

import argparse
import os
import socket
import subprocess
import sys
import zipfile
import logging

import pyrogue

# Name of the TopRoguePackage in the ZIP file
top_rogue_package_name="CryoDet"

def get_args():
    """
        Parses and processes args, returning the modified arguments as a dict.
        This is to maintain backwards compatibility with the old of parsing
        arguments.
    """
    parser = make_parser()
    args = parser.parse_args()
    process_args(args)
    return vars(args)


def process_args(args):
    """
        Processes args from argparse. Unzips zip_file and finds/sets the
        args.config_file. Also set the server port to  a default values
        if it was not defined.
    """

    # Setup logging
    logger = logging.getLogger()
    logger.setLevel(args.log_level)
    # set up a handler with timestamps
    handler = logging.StreamHandler()
    handler.setLevel(args.log_level)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s: %(msg)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Verify if the zip file was specified
    if args.zip_file:
        zip_file_name = args.zip_file

        # Verify if the zip file exist and if it is valid
        if os.path.exists(zip_file_name) and zipfile.is_zipfile(zip_file_name):

            # The ZIP file is valid
            print(f"Valid zip file '{zip_file_name}'.")

            # Add the zip file's python directory to the python path.
            pyrogue.addLibraryPath(f"{zip_file_name}/python")

            # If the default configuration file was given using a relative path,
            # it is refereed to the zip file, so build its full path.
            if args.config_file and args.config_file[0] != '/':

                # The configuration file will be in 'python/${TOP_ROGUE_PACKAGE_NAME}/config/'
                config_file = f"python/{top_rogue_package_name}/config/{args.config_file}"

                print("Default configuration was defined with a relative path.")
                print(f"Looking if the zip file contains: {config_file}")

                # Get a list of files in the zip file
                with zipfile.ZipFile(zip_file_name, 'r') as zf:
                    file_list = zf.namelist()

                # Verify that the file exist inside the zip file
                if config_file in file_list:
                    # If found, build the full path to it, including the path to the zip file
                    print("Found!")
                    args.config_file = f"{zip_file_name}/{config_file}"
                else:
                    # if not found, then clear the argument as it is invalid.
                    print("Not found. Omitting it.")
                    args.config_file = None

        else:
            print("Invalid zip file. Omitting it.")

    # If the server port was not defined, set it to a default value
    if args.server_port is None:
        # When using a real target (AMC carrier or dev board), set the server port to
        # (9000 + 2 * slot_number). Either the IP address or the RSSI lane number
        # must be defined, so calculate the slot number based on those two cases
        if args.pcie_rssi_lane:
            # If the RSSI lane number was defined, get the slot number from it
            args.server_port = 9000 + 2 * ( args.pcie_rssi_lane + 2 )
        elif args.ip_addr:
            # Otherwise, get th slot number from the last digit of the IP address
            args.server_port = 9000 + 2 * int(args.ip_addr[-1:])
        # Otherwise, for the emulator target set the server port to 9000.
        # This target doesn't require IP address nor RSSI lane.
        else:
            args.server_port = 9000

    return args


def verify_ip(args):

    if isinstance(args, argparse.Namespace):
        args = vars(args)

    try:
        socket.inet_pton(socket.AF_INET, args['ip_addr'])
    except socket.error:
        sys.exit("ERROR: Invalid IP Address.")


def ping_fpga(args):

    if isinstance(args, argparse.Namespace):
        args = vars(args)

    print("")
    print("Trying to ping the FPGA...")
    try:
        dev_null = open(os.devnull, 'w')
        subprocess.check_call(["ping", "-c2", args['ip_addr']], stdout=dev_null, stderr=dev_null)
        print("    FPGA is online")
        print("")
    except subprocess.CalledProcessError:
        sys.exit("    ERROR: FPGA can't be reached!")


def make_parser(parser=None):
    """
        Creates argparse parser containing smurf command-line options

        Args:
            parser (optional, argparse.ArgumentParser):
                Existing parser to add arguments to. If not specified a new one
                will be created and returned.
    """

    if parser is None:
        parser = argparse.ArgumentParser()

    group = parser.add_argument_group('SMuRF Args')

    group.add_argument('--zip', '-z', dest='zip_file', default="",
                       help="Pyrogue zip file to be included in python path."
                       )
    group.add_argument('--addr', '-a', dest='ip_addr', default="",
                       help="FPGA IP address. Required when the communication type is based on Ethernet."
                       )
    group.add_argument('--defaults', '-d', dest='config_file',
                       help="Default configuration file. If the path is relative, it refers to "
                       f"the zip file (i.e: <ZIP_FILE>/python/{top_rogue_package_name}/config/<CONFIG_FILE>)."
                       )
    group.add_argument('--configure', '-c', action='store_true',
                       help="Load the default configuration at startup."
                       )
    group.add_argument('--gui', '-g', action='store_true', dest='use_gui',
                       help="Starts the server with a gui."
                       )
    group.add_argument('--nopoll', '-n', action='store_false', dest='polling_en',
                       help="Disables all polling."
                       )
    group.add_argument('--pcie-rssi-lane', '-l', type=int, choices=[0,1,2,3,4,5],
                       help="PCIe RSSI lane. Only needed when using PCIe communication.", dest='pcie_rssi_lane'
                       )
    group.add_argument('--dump-pvs', '-u', dest='pv_dump_file', default="",
                       help="Dump the PV list to \"DUMP_FILE\". Must be used with -e."
                       )
    group.add_argument('--disable-bay0', action='store_true',
                       help="Disable the instantiation of devices for Bay 0."
                       )
    group.add_argument('--disable-bay1', action='store_true',
                       help="Disable the instantiation of devices for Bay 1"
                       )
    group.add_argument('--enable-em22xx', action='store_true',
                       help="Enable the EM22xx power monitor"
                       )
    group.add_argument('--windows-title', '-w', default=None,
                       help="Sets the GUI windows title. Defaults to name of this script. "
                            "This value will be ignored when running in server mode."
                       )
    group.add_argument('--pcie-dev-rssi', default="/dev/datadev_0",
                       help="Set the PCIe card device name used for RSSI "
                            "(defaults to '/dev/datadev_0')."
                       )
    group.add_argument('--pcie-dev-data', default="/dev/datadev_1",
                       help="Set the PCIe card device name used for data "
                            "(defaults to '/dev/datadev_1')."
                       )
    group.add_argument('--server-port', dest='server_port', type=int, default=None,
                       help="Set the server port. Defaults to 9000+2*slot_number"
                       )
    group.add_argument('--use-qt', action='store_true', dest='use_qt',
                       default=False, help="Use the QT ."
                       )
    group.add_argument('--log-level', type=str.upper, help="Set the logging level.",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                       default="WARNING"
                       )

    return parser
