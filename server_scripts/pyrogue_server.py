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

# Create gui interface
def create_gui(root, title=""):
    app_top = pyrogue.gui.application(sys.argv)
    app_top.setApplicationName(title)
    gui_top = pyrogue.gui.GuiTop(group='GuiTop')
    gui_top.addTree(root)
    print("Starting GUI...\n")

    try:
        app_top.exec_()
    except KeyboardInterrupt:
        # Catch keyboard interrupts while the GUI was open
        pass

    print("GUI was closed...")

# Exit with a error message
def exit_message(message):
    print(message)
    print("")
    exit()

class LocalServer(pyrogue.Root):
    """
    Local Server class. This class configure the whole rogue application.
    """
    def __init__(self, ip_addr, config_file, server_mode, epics_prefix,\
        polling_en, comm_type, pcie_rssi_lane, stream_pv_size, stream_pv_type,\
        pv_dump_file, disable_bay0, disable_bay1, disable_gc, windows_title,\
        pcie_dev_rssi, pcie_dev_data):

        try:
            pyrogue.Root.__init__(self, name='AMCc', description='AMC Carrier')

            # File writer for streaming interfaces
            # DDR interface (TDEST 0x80 - 0x87)
            stm_data_writer = pyrogue.utilities.fileio.StreamWriter(name='streamDataWriter')
            self.add(stm_data_writer)
            # Streaming interface (TDEST 0xC0 - 0xC7)
            stm_interface_writer = pyrogue.utilities.fileio.StreamWriter(name='streamingInterface')
            self.add(stm_interface_writer)

            # Workaround to FpgaTopLelevel not supporting rssi = None
            if pcie_rssi_lane == None:
                pcie_rssi_lane = 0

            # Instantiate Fpga top level
            fpga = FpgaTopLevel(ipAddr=ip_addr,
                commType=comm_type,
                pcieRssiLink=pcie_rssi_lane,
                disableBay0=disable_bay0,
                disableBay1=disable_bay1)

            # Add devices
            self.add(fpga)

            # Create stream interfaces
            self.ddr_streams = []       # DDR streams

            # Check if we are using PCIe or Ethernet communication.
            if 'pcie-' in comm_type:
                # If we are suing PCIe communication, used AxiStreamDmas to get the DDR and streaming streams.

                # DDR streams. We are only using the first 2 channel of each AMC daughter card, i.e.
                # channels 0, 1, 4, 5.
                for i in [0, 1, 4, 5]:
                    self.ddr_streams.append(
                        rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0x80 + i), True))

                # Streaming interface stream
                self.streaming_stream = \
                    rogue.hardware.axi.AxiStreamDma(pcie_dev_data,(pcie_rssi_lane*0x100 + 0xC1), True)

                # When PCIe communication is used, we connect the stream data directly to the receiver:
                # Stream -> smurf2mce receiver
                self.smurf_processor = pysmurf.core.devices.SmurfProcessor(
                    name="SmurfProcessor",
                    description="Process the SMuRF Streaming Data Stream",
                    master=self.streaming_stream)

            else:
                # If we are using Ethernet: DDR streams comes over the RSSI+packetizer channel, and
                # the streaming streams comes over a pure UDP channel.

                # DDR streams. The FpgaTopLevel class will defined a 'stream' interface exposing them.
                # We are only using the first 2 channel of each AMC daughter card, i.e. channels 0, 1, 4, 5.
                for i in [0, 1, 4, 5]:
                    self.ddr_streams.append(fpga.stream.application(0x80 + i))

                # Streaming interface stream. It comes over UDP, port 8195, without RSSI,
                # so we an UdpReceiver.
                self.streaming_stream = pysmurf.core.devices.UdpReceiver(ip_addr=ip_addr, port=8195)

                # When Ethernet communication is used, We use a FIFO between the stream data and the receiver:
                # Stream -> FIFO -> smurf_processor receiver
                self.smurf_processor_fifo = rogue.interfaces.stream.Fifo(100000,0,True)
                pyrogue.streamConnect(self.streaming_stream, self.smurf_processor_fifo)

                self.smurf_processor = pysmurf.core.devices.SmurfProcessor(
                    name="SmurfProcessor",
                    description="Process the SMuRF Streaming Data Stream",
                    master=self.smurf_processor_fifo)

            self.add(self.smurf_processor)

            # Add data streams (0-3) to file channels (0-3)
            for i in range(4):

                ## DDR streams
                pyrogue.streamConnect(self.ddr_streams[i],
                    stm_data_writer.getChannel(i))

                ## Streaming interface streams
                # We have already connected TDEST 0xC1 to the smurf_processor receiver,
                # so we need to tapping it to the data writer.
                pyrogue.streamTap(self.streaming_stream, stm_interface_writer.getChannel(0))

            # Look for the TesBias registers
            # TesBias register are located on
            # FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax
            # And their name is TesBiasDacDataRegCh[n], where x = [0:31]
            #self.TestBiasVars = []
            #self.TestBiasRegEx = re.compile('.*TesBiasDacDataRegCh\[(\d+)\]$')
            #for var in self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.variableList:
            #    m = self.TestBiasRegEx.match(var.name)
            #    if m:
            #        reg_index = int(m[1]) - 1
            #        if reg_index < 32:
            #            print(f'Found TesBias register: {var.name}, with index {reg_index}')
            #            self.TestBiasVars.append(var)

            ## Check that we have all 32 TesBias registers
            #if len(self.TestBiasVars) == 32:
            #    print(f'Found 32 TesBias registers. Assigning listener functions')
            #    # Add listener to the TesBias registers
            #    for var in self.TestBiasVars:
            #        var.addListener(self.send_test_bias)
            #    # Prepare a buffer to holds the TesBias register values
            #    self.TesBiasValue = [0] * 32
            #else:
            #    print(f'Error: {len(self.TestBiasVars)} TesBias register were found instead of 32. Aborting')

            # Run control for streaming interfaces
            self.add(pyrogue.RunControl(
                name='streamRunControl',
                description='Run controller',
                cmd=fpga.SwDaqMuxTrig,
                rates={
                    1:  '1 Hz',
                    10: '10 Hz',
                    30: '30 Hz'}))

            # lcaPut limits the maximun lenght of a string to 40 chars, as defined
            # in the EPICS R3.14 CA reference manual. This won't allowed to use the
            # command 'ReadConfig' with a long file path, which is usually the case.
            # This function is a workaround to that problem. Fomr matlab one can
            # just call this function without arguments an the function ReadConfig
            # will be called with a predefined file passed during startup
            # However, it can be usefull also win the GUI, so it is always added.
            self.config_file = config_file
            self.add(pyrogue.LocalCommand(
                name='setDefaults',
                description='Set default configuration',
                function=self.set_defaults_cmd))

            # If Garbage collection was disable, add this local variable to allow users
            # to manually run the garbage collection.
            if disable_gc:
                self.add(pyrogue.LocalCommand(
                    name='runGarbageCollection',
                    description='runGarbageCollection',
                    function=self.run_garbage_collection))

            #self.add(pyrogue.LocalVariable(
            #    name='smurfProcessorDebug',
            #    description='Enable smurf processor transmit debug',
            #    mode='RW',
            #    value=False,
            #    localSet=lambda value: self.smurf_processor.setDebug(value),
            #    hidden=False))

            ## Lost frame counter from smurf_processor
            #self.add(pyrogue.LocalVariable(
            #    name='frameLossCnt',
            #    description='Lost frame Counter',
            #    mode='RO',
            #    value=0,
            #    localGet=self.smurf_processor.getFrameLossCnt,
            #    pollInterval=1,
            #    hidden=False))

            ## Received frame counter from smurf_processor
            #self.add(pyrogue.LocalVariable(
            #    name='frameRxCnt',
            #    description='Received frame Counter',
            #    mode='RO',
            #    value=0,
            #    localGet=self.smurf_processor.getFrameRxCnt,
            #    pollInterval=1,
            #    hidden=False))

            ## Out-of-order frame counter from smurf_processor
            #self.add(pyrogue.LocalVariable(
            #    name='frameOutOrderCnt',
            #    description='Number of time out-of-order frames are detected',
            #    mode='RO',
            #    value=0,
            #    localGet=self.smurf_processor.getFrameOutOrderCnt,
            #    pollInterval=1,
            #    hidden=False))

            ## Command to clear all the frame counters on smurf_processor
            #self.add(pyrogue.LocalCommand(
            #    name='clearFrameCnt',
            #    description='Clear all frame counters',
            #    function=self.smurf_processor.clearFrameCnt))

            # Start the root
            print("Starting rogue server")
            self.start(pollEn=polling_en)

            self.ReadAll()

            # Call the get() method on the tesBias variable to force the call to
            # send_test_bias and update the array in Smurf2MCE
            # for var in self.TestBiasVars:
            #     var.get()

        except KeyboardInterrupt:
            print("Killing server creation...")
            super(LocalServer, self).stop()
            exit()

        # Show image build information
        try:
            print("")
            print("FPGA image build information:")
            print("===================================")
            print("BuildStamp              : {}"\
                .format(self.FpgaTopLevel.AmcCarrierCore.AxiVersion.BuildStamp.get()))
            print("FPGA Version            : 0x{:x}"\
                .format(self.FpgaTopLevel.AmcCarrierCore.AxiVersion.FpgaVersion.get()))
            print("Git hash                : 0x{:x}"\
                .format(self.FpgaTopLevel.AmcCarrierCore.AxiVersion.GitHash.get()))
        except AttributeError as attr_error:
            print("Attibute error: {}".format(attr_error))
        print("")

        # Start the EPICS server
        if epics_prefix:
            print("Starting EPICS server using prefix \"{}\"".format(epics_prefix))

            self.epics = pyrogue.protocols.epics.EpicsCaServer(base=epics_prefix, root=self)

            # PVs for stream data
            if stream_pv_size:

                print("Enabling stream data on PVs (buffer size = {} points, data type = {})"\
                    .format(stream_pv_size,stream_pv_type))

                self.stream_fifos  = []
                self.stream_slaves = []
                for i in range(4):
                    self.stream_slaves.append(self.epics.createSlave(name="AMCc:Stream{}".format(i), maxSize=stream_pv_size, type=stream_pv_type))

                    # Calculate number of bytes needed on the fifo
                    if '16' in stream_pv_type:
                        fifo_size = stream_pv_size * 2
                    else:
                        fifo_size = stream_pv_size * 4

                    self.stream_fifos.append(rogue.interfaces.stream.Fifo(1000, fifo_size, True)) # changes
                    self.stream_fifos[i]._setSlave(self.stream_slaves[i])
                    pyrogue.streamTap(self.ddr_streams[i], self.stream_fifos[i])

            self.epics.start()

            # Dump the PV list to the specified file
            if pv_dump_file:
                try:
                    # Try to open the output file
                    f = open(pv_dump_file, "w")
                except IOError:
                    print("Could not open the PV dump file \"{}\"".format(pv_dump_file))
                else:
                    with f:
                        print("Dumping PV list to \"{}\"...".format(pv_dump_file))
                        try:
                            try:
                                # Redirect the stdout to the output file momentarily
                                original_stdout, sys.stdout = sys.stdout, f
                                self.epics.dump()
                            finally:
                                sys.stdout = original_stdout

                            print("Done!")
                        except:
                            # Capture error from epics.dump() if any
                            print("Errors were found during epics.dump()")

        # If no in server Mode, start the GUI
        if not server_mode:
            create_gui(self, title=windows_title)
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

    # Function for setting a default configuration.
    def set_defaults_cmd(self):
        # Check if a default configuration file has been defined
        if not self.config_file:
            print('No default configuration file was specified...')
            return

        print('Setting defaults from file {}'.format(self.config_file))
        self.ReadConfig(self.config_file)

    def stop(self):
        print("Stopping servers...")
        if hasattr(self, 'epics'):
            print("Stopping EPICS server...")
            self.epics.stop()
        super(LocalServer, self).stop()

    def run_garbage_collection(self):
        print("Running garbage collection...")
        gc.collect()
        print( gc.get_stats() )

    # Send TesBias to Smurf2MCE
    def send_test_bias(self, path, value, disp):
        # Look for the register index
        m = self.TestBiasRegEx.match(path)
        if m:
            reg_index = int(m[1]) - 1
            if reg_index < 32:

                # Update reg value in the buffer
                self.TesBiasValue[reg_index] = value

                # The index  send to Smurf2MCE
                tes_bias_index = reg_index // 2

                # Calculate the difference between DAC bias values
                tes_bias_val = self.TesBiasValue[2*tes_bias_index+1] - self.TesBiasValue[2*tes_bias_index]

                # Send the difference value to smurf2mce
                #self.smurf_processor.setTesBias(tes_bias_index, tes_bias_val)
                self.smurf_processor.setTesBias(tes_bias_index, tes_bias_val)

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
            "ha:se:d:nb:f:c:l:u:w:",
            ["help", "addr=", "server", "epics=", "defaults=", "nopoll",
            "stream-size=", "stream-type=", "commType=", "pcie-rssi-link=", "dump-pvs=",
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
        elif opt in ("-c", "--commType"):   # Communication type
            if arg in comm_type_valid_types:
                comm_type = arg
            else:
                print("Invalid communication type. Valid choises are:")
                for c in comm_type_valid_types:
                    print("  - \"{}\"".format(c))
                exit_message("ERROR: Invalid communication type")
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
    with pysmurf.core.devices.PcieCard(lane=pcie_rssi_lane, comm_type=comm_type, ip_addr=ip_addr,
        dev_rssi=pcie_dev_rssi, dev_data=pcie_dev_data):

        # Start pyRogue server
        server = LocalServer(
            ip_addr=ip_addr,
            config_file=config_file,
            server_mode=server_mode,
            epics_prefix=epics_prefix,
            polling_en=polling_en,
            comm_type=comm_type,
            pcie_rssi_lane=pcie_rssi_lane,
            stream_pv_size=stream_pv_size,
            stream_pv_type=stream_pv_type,
            pv_dump_file=pv_dump_file,
            disable_bay0=disable_bay0,
            disable_bay1=disable_bay1,
            disable_gc=disable_gc,
            windows_title=windows_title,
            pcie_dev_rssi=pcie_dev_rssi,
            pcie_dev_data=pcie_dev_data)

    # Stop server
    server.stop()

    print("")
