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
import rogue.hardware.axi
import rogue.protocols.srp

import pysmurf
import pysmurf.core.devices
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
                 disable_bay0   = False,
                 disable_bay1   = False,
                 **kwargs):

        pyrogue.Root.__init__(self, name="AMCc", initRead=True, pollEn=polling_en,
            timeout=5.0, streamIncGroups='stream', serverPort=server_port, **kwargs)

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
            print(f"Starting EPICS server using prefix \"{epics_prefix}\"")
            from pyrogue.protocols import epics
            self._epics = epics.EpicsCaServer(base=epics_prefix, root=self)
            self._pv_dump_file = pv_dump_file

            # PVs for stream data
            # This should be replaced with DataReceiver objects
            if stream_pv_size:
                print(
                    "Enabling stream data on PVs " +
                    f"(buffer size = {stream_pv_size} points, " +
                    f"data type = {stream_pv_type})")

                self._stream_fifos  = []
                self._stream_slaves = []
                for i in range(4):
                    self._stream_slaves.append(self._epics.createSlave(name=f"AMCc:Stream{i}",
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

        # List of enabled bays
        self._enabled_bays = [i for i,e in enumerate([disable_bay0, disable_bay1]) if not e]

    def start(self):
        pyrogue.Root.start(self)

        # Setup groups
        pysmurf.core.utilities.setupGroups(self, self._VariableGroups)

        # Show image build information
        try:
            print("")
            print("FPGA image build information:")
            print("===================================")
            print(
                "BuildStamp              : " +
                f"{self.FpgaTopLevel.AmcCarrierCore.AxiVersion.BuildStamp.get()}")
            print(
                "FPGA Version            : 0x" +
                f"{self.FpgaTopLevel.AmcCarrierCore.AxiVersion.FpgaVersion.get():x}")
            print(
                "Git hash                : 0x" +
                f"{self.FpgaTopLevel.AmcCarrierCore.AxiVersion.GitHash.get():x}")
        except AttributeError as attr_error:
            print(f"Attibute error: {attr_error}")
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

        # Update the 'EnabledBays' variable with the correct list of enabled bays.
        self.SmurfApplication.EnabledBays.set(self._enabled_bays)

        # Workaround: Set the Mask value to '[0]' by default when the server starts.
        # This is needed because the pysmurf-client by default stat data streaming
        # @4kHz without setting this mask, which by default has a value of '[0]*4096'.
        # Under these conditions, the software processing block is not able to keep
        # up, eventually making the PCIe card's buffer to get full, and that triggers
        # some known issues in the PCIe FW.
        self.SmurfProcessor.ChannelMapper.Mask.set([0])

    def stop(self):
        print("Stopping servers...")
        if self._epics:
            self._epics.stop()

        print("Stopping root...")
        pyrogue.Root.stop(self)

    # Function to load the configuration file, catching exceptions and checking the
    # return value of "LoadConfig", and trying if there were errors, up to
    # "max_retries" times.
    # This method return "True" if the configuration file was load correctly, and
    # "returns False" otherwise.
    def _load_config(self):
        success = False
        max_retries=10

        for i in range(max_retries):
            print(f'Setting defaults from file {self._config_file} (try number {i})')

            try:
                # Try to load the defaults file
                ret = self.LoadConfig(self._config_file)

                # Check the return value from 'LoadConfig'.
                if not ret:
                    print(f'  Setting defaults try number {i} failed. "LoadConfig" returned "False"')
                else:
                    success = True
                    break
            except pyrogue.DeviceError as err:
                print(f'  Setting defaults try number {i} failed with: {err}')

        # Check if we could load the defaults before 'max_retries' retires.
        if success:
            print('Defaults were set correctly!')
            return True
        else:
            print(f'ERROR: Failed to set defaults after {max_retries} retries!')
            return False

    # Function for setting a default configuration.
    def _set_defaults_cmd(self):
        # Check if a default configuration file has been defined
        if self._config_file is None:
            print('No default configuration file was specified...')
            return

        # Try to load the configuration file
        if not _load_config():
            return

        # After loading defaults successfully, check the status of the elastic buffers
        done = True
        max_retries=10
        for k in range(max_retries):
            print(f'Check elastic buffers ({k})...')
            retry_app_top_init = False

            for i in self._enabled_bays:
                # Workaround: Reading the individual registers does not work. So, we need to read
                # the whole device, and then use the 'value()' method to get the register values.
                self.FpgaTopLevel.AppTop.AppTopJesd[i].JesdRx.ReadDevice()

                for j in range(10):
                    # Check if the latency values are correct. The latency values should be:
                    # - 13 or 14, for lanes = 0:1,4:9
                    # - 255, for lanes 2:3
                    latency = self.FpgaTopLevel.AppTop.AppTopJesd[i].JesdRx.ElBuffLatency[j].value()
                    latency_ok = False

                    if j in [2,3]:
                        if latency == 255:
                            latency_ok = True
                    else:
                        if latency in [13,14]:
                            latency_ok = True

                    if latency_ok:
                        print(f'  OK - JesdRx[{i}].ElBuffLatency[{j}] = {latency}')
                    else:
                        print(f'  Test failed. JesdRx[{i}].ElBuffLatency[{j}] = {latency}')
                        retry_app_top_init = True
                        done = False

            # If the check failed, call 'AppTop.Init()' command again
            if retry_app_top_init:
                print('  Executing AppTop.Init()...')
                self.FpgaTopLevel.AppTop.Init()
            else:
                break

        # Check if the test passed before 'max_retries' retries
        if done:
            print('Elastic buffer check passed!')
        else:
            print(f'ERROR: Elastic buffer check failed {max_retries} times')
