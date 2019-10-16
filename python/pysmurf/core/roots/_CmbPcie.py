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
import pysmurf
import rogue.hardware.axi
import rogue.protocols.srp

import CryoDet.MicroMuxBpEthGen2.FpgaTopLevel as FpgaTopLevel
import AmcCarrierCore.AppTop as AppTop

class CmbPcie(AppTop.RootBase):
    def __init__(self, *,
                 ip_addr        = "",
                 config_file    = "",
                 epics_prefix   = "",
                 polling_en     = True,
                 pcie_rssi_lane = None,
                 stream_pv_size = 0,
                 stream_pv_type = "UInt16",
                 pv_dump_file   = "",
                 disable_bay0   = False,
                 disable_bay1   = False,
                 disable_gc     = False,
                 pcie_dev_rssi  = "/dev/datadev_0",
                 pcie_dev_data  = "/dev/datadev_1",
                 **kwargs):

        pyrogue.Root.__init__(self, initRead=True, pollEn=polling_en, **kwargs)

        self._pv_dump_file = pv_dump_file

        # Workaround to FpgaTopLelevel not supporting rssi = None
        if pcie_rssi_lane == None:
            pcie_rssi_lane = 0

        # TDEST 0 routed to streamr0 (SRPv3)
        self.dma = rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0),True)
        self.srp = rogue.protocols.srp.SrpV3()
        pyrogue.streamConnectBiDir( self.srp, self.dma )

        # Instantiate Fpga top level
        self._fpga = FpgaTopLevel( memBase      = self.srp,
                                   ipAddr       = ip_addr,
                                   commType     = "pcie-rssi-interleaved",
                                   pcieRssiLink = pcie_rssi_lane,
                                   disableBay0  = disable_bay0,
                                   disableBay1  = disable_bay1)

        # Add devices
        self.add(self._fpga)

        # File writer for streaming interfaces
        # DDR interface (TDEST 0x80 - 0x87)
        self._stm_data_writer = pyrogue.utilities.fileio.StreamWriter(name='streamDataWriter')
        self.add(self._stm_data_writer)

        # Streaming interface (TDEST 0xC0 - 0xC7)
        self._stm_interface_writer = pyrogue.utilities.fileio.StreamWriter(name='streamingInterface')
        self.add(self._stm_interface_writer)

        # Create stream interfaces
        self._ddr_streams = []

        # DDR streams. We are only using the first 2 channel of each AMC daughter card, i.e.
        # channels 0, 1, 4, 5.
        for i in [0, 1, 4, 5]:
            self._ddr_streams.append(
                rogue.hardware.axi.AxiStreamDma(pcie_dev_rssi,(pcie_rssi_lane*0x100 + 0x80 + i), True))

        # Streaming interface stream
        self._streaming_stream = \
            rogue.hardware.axi.AxiStreamDma(pcie_dev_data,(pcie_rssi_lane*0x100 + 0xC1), True)

        # When PCIe communication is used, we connect the stream data directly to the receiver:
        # Stream -> smurf2mce receiver
        self._smurf_processor = pysmurf.core.devices.SmurfProcessor(
            name="SmurfProcessor",
            description="Process the SMuRF Streaming Data Stream",
            master=self._streaming_stream)

        # Add data streams (0-3) to file channels (0-3)
        for i in range(4):
            ## DDR streams
            pyrogue.streamConnect(self._ddr_streams[i], self._stm_data_writer.getChannel(i))

        ## Streaming interface streams
        # We have already connected TDEST 0xC1 to the smurf_processor receiver,
        # so we need to tapping it to the data writer.
        pyrogue.streamTap(self._streaming_stream, self._stm_interface_writer.getChannel(0))

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
        #        var.addListener(self._send_test_bias)
        #    # Prepare a buffer to holds the TesBias register values
        #    self.TesBiasValue = [0] * 32
        #else:
        #    print(f'Error: {len(self.TestBiasVars)} TesBias register were found instead of 32. Aborting')

        # Run control for streaming interfaces
        self.add(pyrogue.RunControl(
            name='streamRunControl',
            description='Run controller',
            cmd=self._fpga.SwDaqMuxTrig,
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
        self._config_file = config_file
        self.add(pyrogue.LocalCommand(
            name='setDefaults',
            description='Set default configuration',
            function=self._set_defaults_cmd))

        # If Garbage collection was disable, add this local variable to allow users
        # to manually run the garbage collection.
        if disable_gc:
            self.add(pyrogue.LocalCommand(
                name='runGarbageCollection',
                description='runGarbageCollection',
                function=self._run_garbage_collection))

        # Add epics interface
        self._epics = None
        if epics_prefix:
            self._epics = pyrogue.protocols.epics.EpicsCaServer(base=epics_prefix, root=self)

            # PVs for stream data
            if stream_pv_size:

                print("Enabling stream data on PVs (buffer size = {} points, data type = {})"\
                    .format(stream_pv_size,stream_pv_type))

                self._stream_fifos  = []
                self._stream_slaves = []
                for i in range(4):
                    self._stream_slaves.append(self._epics.createSlave(name="AMCc:Stream{}".format(i),
                                                                       maxSize=stream_pv_size,
                                                                       type=stream_pv_type))

                    # Calculate number of bytes needed on the fifo
                    if '16' in stream_pv_type:
                        fifo_size = stream_pv_size * 2
                    else:
                        fifo_size = stream_pv_size * 4

                    self._stream_fifos.append(rogue.interfaces.stream.Fifo(1000, fifo_size, True)) # changes
                    self._stream_fifos[i]._setSlave(self._stream_slaves[i])
                    pyrogue.streamTap(self._ddr_streams[i], self._stream_fifos[i])


    def start(self):
        pyrogue.Root.start(self)

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

        if self._epics:
            self._epics.start()

            # Dump the PV list to the specified file
            # This should be made part of the Rogue dump!
            if self._pv_dump_file:
                try:
                    # Try to open the output file
                    f = open(self._pv_dump_file, "w")
                except IOError:
                    print("Could not open the PV dump file \"{}\"".format(self._pv_dump_file))
                else:
                    with f:
                        print("Dumping PV list to \"{}\"...".format(self._pv_dump_file))
                        try:
                            try:
                                # Redirect the stdout to the output file momentarily
                                original_stdout, sys.stdout = sys.stdout, f
                                self._epics.dump()
                            finally:
                                sys.stdout = original_stdout

                            print("Done!")
                        except:
                            # Capture error from epics.dump() if any
                            print("Errors were found during epics.dump()")

    def stop(self):
        print("Stopping servers...")
        if self._epics:
            self._epics.stop()
        pyrogue.Root.stop()

    # Function for setting a default configuration.
    def _set_defaults_cmd(self):
        # Check if a default configuration file has been defined
        if not self._config_file:
            print('No default configuration file was specified...')
            return

        print('Setting defaults from file {}'.format(self._config_file))
        self.LoadConfig(self._config_file)


    def _run_garbage_collection(self):
        print("Running garbage collection...")
        gc.collect()
        print( gc.get_stats() )


    # Send TesBias to Smurf2MCE
    def _send_test_bias(self, path, value, disp):
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
