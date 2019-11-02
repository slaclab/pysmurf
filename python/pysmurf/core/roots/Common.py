#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Common Root Class
#-----------------------------------------------------------------------------
# File       : Common.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
# Common Root class
#-----------------------------------------------------------------------------
# This file is part of the AmcCarrier Core. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the AmcCarrierCore, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import re

import pyrogue
import pysmurf
import rogue.hardware.axi
import rogue.protocols.srp

import pysmurf.core.utilities

from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel

from pyrogue.protocols import epics

class Common(pyrogue.Root):
    def __init__(self, *,
                 config_file    = None,
                 epics_prefix   = "EpicsPrefix",
                 polling_en     = True,
                 pv_dump_file   = None,
                 txDevice       = None,
                 fpgaTopLevel   = None,
                 stream_pv_size = 2**19,    # Not sub-classed
                 stream_pv_type = 'Int16',  # Not sub-classed 
                 **kwargs):

        pyrogue.Root.__init__(self, name="AMCc", initRead=True, pollEn=polling_en, **kwargs)

        #########################################################################################
        # The following interfaces are expected to be defined at this point by a sub-class
        # self._streaming_stream # Data stream interface
        # self._ddr_streams # 4 DDR Interface Streams
        # self._fpga = Top level FPGA

        # Add PySmurf Application Block
        self.add(pysmurf.core.devices.SmurfApplication())

        # Add FPGA
        self.add(self._fpga)

        # File writer for streaming interfaces
        # DDR interface (TDEST 0x80 - 0x87)
        self._stm_data_writer = pyrogue.utilities.fileio.StreamWriter(name='streamDataWriter')
        self.add(self._stm_data_writer)

        # Streaming interface (TDEST 0xC0 - 0xC7)
        self._stm_interface_writer = pyrogue.utilities.fileio.StreamWriter(name='streamingInterface')
        self.add(self._stm_interface_writer)

        # Add the SMuRF processor device
        self._smurf_processor = pysmurf.core.devices.SmurfProcessor(
            name="SmurfProcessor",
            description="Process the SMuRF Streaming Data Stream",
            root=self,
            txDevice=txDevice)
        self.add(self._smurf_processor)

        # Connect smurf processor
        pyrogue.streamConnect(self._streaming_stream, self._smurf_processor)

        # Add data streams (0-3) to file channels (0-3)
        for i in range(4):
            ## DDR streams
            pyrogue.streamConnect(self._ddr_streams[i], self._stm_data_writer.getChannel(i))

        ## Streaming interface streams
        # We have already connected TDEST 0xC1 to the smurf_processor receiver,
        # so we need to tapping it to the data writer.
        pyrogue.streamTap(self._streaming_stream, self._stm_interface_writer.getChannel(0))

        # TES Bias Update Function
        def _update_tes_bias(idx):
            v1 = self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.node(f'TesBiasDacDataRegCh[{2*idx+1}]').value()
            v2 = self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.node(f'TesBiasDacDataRegCh[{2*idx}]').value()
            val = v1 - v2

            # Pass to data processor
            self._smurf_processor.setTesBias(index=idx, val=val)

        # Register TesBias values configuration to update stream processor
        for i in range(32):
            idx = i // 2
            try:
                v = self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.node(f'.*TesBiasDacDataRegCh[{i}]')
                v.addVarListener(lambda idx=idx: _update_test_bias(idx))
            except:
                print(f"TesBiasDacDataRegCh[{i}] not found... Skipping!")

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
        print("Starting EPICS server using prefix \"{}\"".format(epics_prefix))
        self._epics = pyrogue.protocols.epics.EpicsCaServer(base=epics_prefix, root=self)
        self._pv_dump_file = pv_dump_file

        # PVs for stream data
        # This should be replaced with DataReceiver objects 
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

        # Start epics
        self._epics.start()
        self._epics.dump(self._pv_dump_file)

        # Add publisher, pub_root & script_id need to be updated
        self._pub = pysmurf.core.utilities.SmurfPublisher(root=self,pub_root=None,script_id=None)


    def stop(self):
        print("Stopping servers...")
        self._epics.stop()
        pyrogue.Root.stop(self)

    # Function for setting a default configuration.
    def _set_defaults_cmd(self):
        # Check if a default configuration file has been defined
        if self._config_file is None:
            print('No default configuration file was specified...')
            return

        print('Setting defaults from file {}'.format(self._config_file))
        self.LoadConfig(self._config_file)


