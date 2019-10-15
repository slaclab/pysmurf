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

import CryoDet.MicroMuxBpEthGen2.FpgaTopLevel as FpgaTopLevel
import AmcCarrierCore.AppTop as AppTop

class CmbPcie(AppTop.RootBase):
    def __init__(self, *, 
                 pcieDev       = '/dev/datadev_0',
                 pcieDataDev   = '/dev/datadev_1',
                 pcieRssiLink  = 0,
                 configFile    = "",
                 epicsPrefix   = None,
                 disableBay0   = False,
                 disableBay1   = False,
                 streamPvSize  = 0,
                 streamPvType  = 0,
                 pvDumpFile    = None,
                 **kwargs):

        pyrogue.Root.__init__(self, initRead=True, **kwargs)

        # TDEST 0 routed to streamr0 (SRPv3)
        self.dma  = rogue.hardware.axi.AxiStreamDma(pcieDev,(pcieRssiLink*0x100 + 0),True)
        self.srp = rogue.protocols.srp.SrpV3()
        pr.streamConnectBiDir( self.srp, self.dma )

        # Streaming interface stream
        self._streaming_stream = rogue.hardware.axi.AxiStreamDma(pcieDataDev,(PcieRssiLane*0x100 + 0xC1), True)
        # Top level module should be added here.
        # Top level is a sub-class of AmcCarrierCore.AppTop.TopLevel
        # SRP interface should be passed as an arg
        self.add(FpgaTopLevel(memBase=self.srp))

        # Instantiate Fpga top level
        self._fpga = FpgaTopLevel( disableBay0=disableBay0, disableBay1=disableBay1)

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
        # Dests = 0x1
        self._ddr_streams = [rogue.hardware.axi.AxiStreamDma(pcieDev,(pcieRssiLink*0x100 + 0x80 + i), True) for i in [0, 1, 4, 5]]

        # Streaming interface stream
        self._streaming_stream = rogue.hardware.axi.AxiStreamDma(pcieDataDev,(PcieRssiLink*0x100 + 0xC1), True)
    
        # Our smurf_processor receiver
        # The data stream comes from TDEST 0xC1
        # We use a FIFO between the stream data and the receiver:
        # Stream -> FIFO -> smurf_processor receiver
        self._smurf_processor_fifo = rogue.interfaces.stream.Fifo(100000,0,True)
        pyrogue.streamConnect(self._streaming_stream, self._smurf_processor_fifo)

        self._smurf_processor = pysmurf.core.devices.SmurfProcessor(
                name="SmurfProcessor",
                description="Process the SMuRF Streaming Data Stream",
                master=self._smurf_processor_fifo)

        self.add(self._smurf_processor)
   
        # Add data streams (0-3) to file channels (0-3)
        for i in range(4):

            ## DDR streams
            pyrogue.streamConnect(self._ddr_streams[i], self._stm_data_writer.getChannel(i))

            ## Streaming interface streams
            # We have already connected TDEST 0xC1 to the smurf_processor receiver,
            # so we need to tapping it to the data writer.
            pyrogue.streamTap(self._streaming_stream, self._stm_interface_writer.getChannel(0))

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

        # Add epics interface
        self._epics = None
        if epicsPrefix:
            self._epics = pyrogue.protocols.epics.EpicsCaServer(base=epicsPrefix, root=self)

            # PVs for stream data
            if streamPvSize:

                print("Enabling stream data on PVs (buffer size = {} points, data type = {})"\
                    .format(streamPvSize,streamPvType))

                self._stream_fifos  = []
                self._stream_slaves = []
                for i in range(4):
                    self._stream_slaves.append(self._epics.createSlave(name="AMCc:Stream{}".format(i), 
                                                                       maxSize=streamPvSize, 
                                                                       type=streamPvType))

                    # Calculate number of bytes needed on the fifo
                    if '16' in streamPvType:
                        fifo_size = streamPvSize * 2
                    else:
                        fifo_size = streamPvSize * 4

                    self._stream_fifos.append(rogue.interfaces.stream.Fifo(1000, fifo_size, True)) # changes
                    self._stream_fifos[i]._setSlave(self._stream_slaves[i])
                    pyrogue.streamTap(self._ddr_streams[i], _self.stream_fifos[i])


    def start(self):
        pyrogue.Root.start()

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
            if pvDumpFile:
                try:
                    # Try to open the output file
                    f = open(pvDumpFile, "w")
                except IOError:
                    print("Could not open the PV dump file \"{}\"".format(pvDumpFile))
                else:
                    with f:
                        print("Dumping PV list to \"{}\"...".format(pvDumpFile))
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
            self._epics.start()
        pyrogue.Root.start()


    # Function for setting a default configuration.
    def _set_defaults_cmd(self):
        # Check if a default configuration file has been defined
        if not self._config_file:
            print('No default configuration file was specified...')
            return

        print('Setting defaults from file {}'.format(self._config_file))
        self.LoadConfig(self._config_file)


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

