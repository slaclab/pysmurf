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

import pyrogue
import pysmurf
import rogue.hardware.axi
import rogue.protocols.srp

import pysmurf.core.utilities

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
                 configure      = False,
                 VariableGroups = None,
                 server_port    = 0,
                 pcie           = None,
                 **kwargs):

        pyrogue.Root.__init__(self, name="AMCc", initRead=True, pollEn=polling_en,
            streamIncGroups='stream', serverPort=server_port, **kwargs)

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
        # smurf_processor bias index 0 = TesBiasDacDataRegCh[2] - TesBiasDacDataRegCh[1]
        # smurf_processor bias index l = TesBiasDacDataRegCh[4] - TesBiasDacDataRegCh[3]
        def _update_tes_bias(idx):
            v1 = self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.node(f'TesBiasDacDataRegCh[{(2*idx)+2}]').value()
            v2 = self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.node(f'TesBiasDacDataRegCh[{(2*idx)+1}]').value()
            val = (v1 - v2) // 2

            # Pass to data processor
            self._smurf_processor.setTesBias(index=idx, val=val)

        # Register TesBias values configuration to update stream processor
        # Bias values are ranged 1 - 32, matching tes bias indexes 0 - 16
        for i in range(1,33):
            idx = (i-1) // 2
            try:
                v = self.FpgaTopLevel.AppTop.AppCore.RtmCryoDet.RtmSpiMax.node(f'TesBiasDacDataRegCh[{i}]')
                v.addListener(lambda path, value, lidx=idx: _update_tes_bias(lidx))
            except Exception:
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

        # Flag that indicates if the default configuration should be loaded
        # once the root is started.
        self._configure = configure

        # Variable groups
        self._VariableGroups = VariableGroups

        # Add epics interface
        self._epics = None
        if epics_prefix:
            print("Starting EPICS server using prefix \"{}\"".format(epics_prefix))
            self._epics = pyrogue.protocols.epics.EpicsCaServer(base=epics_prefix, root=self)
            self._pv_dump_file = pv_dump_file

            # PVs for stream data
            # This should be replaced with DataReceiver objects
            if stream_pv_size:
                print("Enabling stream data on PVs (buffer size = {} points, data type = {})"
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
                    pyrogue.streamConnect(self._stream_fifos[i],self._stream_slaves[i])
                    pyrogue.streamTap(self._ddr_streams[i], self._stream_fifos[i])


        # Update SaveState to not read before saving
        self.SaveState.replaceFunction(lambda arg: self.saveYaml(name=arg,
                                                                 readFirst=False,
                                                                 modes=['RW','RO','WO'],
                                                                 incGroups=None,
                                                                 excGroups='NoState',
                                                                 autoPrefix='state',
                                                                 autoCompress=True))

        # Update SaveConfig to not read before saving
        self.SaveConfig.replaceFunction(lambda arg: self.saveYaml(name=arg,
                                                                  readFirst=False,
                                                                  modes=['RW','WO'],
                                                                  incGroups=None,
                                                                  excGroups='NoConfig',
                                                                  autoPrefix='config',
                                                                  autoCompress=False))

        self._pcie = pcie

        if self._pcie:
            self.add(pyrogue.LocalCommand(
                name='RestartRssi',
                description='Restart RSSI Link',
                function=lambda : self._pcie.restart_rssi))

    def start(self):
        pyrogue.Root.start(self)

        # Setup groups
        pysmurf.core.utilities.setupGroups(self, self._VariableGroups)

        # Show image build information
        try:
            print("")
            print("FPGA image build information:")
            print("===================================")
            print("BuildStamp              : {}"
                .format(self.FpgaTopLevel.AmcCarrierCore.AxiVersion.BuildStamp.get()))
            print("FPGA Version            : 0x{:x}"
                .format(self.FpgaTopLevel.AmcCarrierCore.AxiVersion.FpgaVersion.get()))
            print("Git hash                : 0x{:x}"
                .format(self.FpgaTopLevel.AmcCarrierCore.AxiVersion.GitHash.get()))
        except AttributeError as attr_error:
            print("Attibute error: {}".format(attr_error))
        print("")

        # Start epics
        if self._epics:
            self._epics.start()

            # Dump the PV list to the expecified file
            if self._pv_dump_file:
                self._epics.dump(self._pv_dump_file)

        # Add publisher, pub_root & script_id need to be updated
        self._pub = pysmurf.core.utilities.SmurfPublisher(root=self)

        # Load default configuration, if requested
        if self._configure:
            self.setDefaults.call()



    def stop(self):
        print("Stopping servers...")
        if self._epics:
            self._epics.stop()

        print("Stopping root...")
        pyrogue.Root.stop(self)

    # Function for setting a default configuration.
    def _set_defaults_cmd(self):
        # Check if a default configuration file has been defined
        if self._config_file is None:
            print('No default configuration file was specified...')
            return

        print('Setting defaults from file {}'.format(self._config_file))
        self.LoadConfig(self._config_file)
