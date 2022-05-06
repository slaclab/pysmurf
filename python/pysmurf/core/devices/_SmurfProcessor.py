#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Processor (Monolithic)
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Processor device, in its monolithic version.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import pyrogue
import pyrogue.utilities.fileio
import pyrogue.interfaces.stream

import pysmurf.core.counters
import pysmurf.core.conventers
import pysmurf.core.emulators
import smurf
import smurf.core.processors

class SmurfChannelMapper(pyrogue.Device):
    """
    SMuRF Channel Mapper Python Wrapper.
    """
    def __init__(self, name, device, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Channel Mapper', **kwargs)
        self.device = device

        # Add the number of enabled channels  variable
        self.add(pyrogue.LocalVariable(
            # NumChannels is defined here and accessible from pysmurf.
            name='NumChannels',
            description='Number enabled channels',
            mode='RO',
            value=0,
            pollInterval=1,
            # getNumCh is defined in SmurfProcessor.cpp.
            localGet=self.device.getNumCh))

        # Add "payloadSize" variable
        self.add(pyrogue.LocalVariable(
            name='PayloadSize',
            description='Define a fixed payload size. If 0, the payload will hold the number of mapped channels',
            mode='RW',
            value=528,
            localSet=lambda value: self.device.setPayloadSize(value),
            localGet=self.device.getPayloadSize))

        # Add variable to set the mapping mask
        # Rogue doesn't allow to have an empty list here. Also, the EPICS PV is created
        # with the initial size of this list, and can not be changed later, so we are doing
        # it big enough at this point using the maximum number of channels
        self.add(pyrogue.LocalVariable(
            name='Mask',
            description='Set the mapping mask',
            mode='RW',
            value=[0]*4096,
            localSet=lambda value: self.device.setMask(value),
            localGet=self.device.getMask))

class Unwrapper(pyrogue.Device):
    """
    SMuRF Data Unwrapper Python Wrapper.
    """
    def __init__(self, name, device, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Data Unwrapper', **kwargs)
        self.device = device

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self.device.setUnwrapperDisable(value),
            localGet=self.device.getUnwrapperDisable))

        # Command to reset the unwrapper
        self.add(pyrogue.LocalCommand(
            name='reset',
            description='Reset the unwrapper',
            function=self.device.resetUnwrapper))

class Downsampler(pyrogue.Device):
    """
    SMuRF Data Downsampler Python Wrapper.
    """
    def __init__(self, name, device, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Data Downsampler', **kwargs)
        self.device = device

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self.device.setDownsamplerDisable(value),
            localGet=self.device.getDownsamplerDisable))

        # Add the trigger mode variable
        self.add(pyrogue.LocalVariable(
            name='DownsamplerMode',
            description='Mode for the downsampler.',
            mode='RW',
            enum={0:'Internal', 1:'External'},
            value=0,
            localSet=lambda value : self.device.setDownsamplerMode(value),
            localGet=self.device.getDownsamplerMode))

        self.add(pyrogue.LocalVariable(
            name='InternalFactor',
            description='When on internal mode, of incoming frames to ignore.',
            mode='RW',
            value=20,
            localSet=lambda value : self.device.setDownsamplerInternalFactor(value),
            localGet=self.device.getDownsamplerInternalFactor))

        self.add(pyrogue.LocalVariable(
            name='ExternalBitmask',
            description='Mask for the external timing bits, size 18. First 10 are fixed rates from 1 to 15 kHz, last 18 are custom rates. 0 means never trigger.',
            mode='RW',
            value=0,
            localSet=lambda value : self.device.setDownsamplerExternalBitmask(value),
            localGet=self.device.getDownsamplerExternalBitmask))

        self.add(pyrogue.LocalVariable(
            name='outgoingCount',
            description='Outgoing number of frames from the downsampler.',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self.device.getDownsamplerOutgoingCount))



class GeneralAnalogFilter(pyrogue.Device):
    """
    SMuRF Data GeneralAnalogFilter Python Wrapper.
    """
    def __init__(self, name, device, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Data GeneralAnalogFilter', **kwargs)
        self.device = device

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self.device.setFilterDisable(value),
            localGet=self.device.getFilterDisable))

        # Add the filter order variable
        self.add(pyrogue.LocalVariable(
            name='Order',
            description='Filter order',
            mode='RW',
            value=4,
            localSet=lambda value : self.device.setOrder(value),
            localGet=self.device.getOrder))

        # Add the filter gain variable
        self.add(pyrogue.LocalVariable(
            name='Gain',
            description='Filter gain',
            mode='RW',
            value=1.0,
            localSet=lambda value : self.device.setGain(value),
            localGet=self.device.getGain))

        # Add the filter a coefficients variable
        # Rogue doesn't allow to have an empty list here. Also, the EPICS PV is created
        # with the initial size of this list, and can not be changed later, so we are doing
        # it big enough at this point (we are probably not going to use an order > 10)
        self.add(pyrogue.LocalVariable(
            name='A',
            description='Filter a coefficients',
            mode='RW',
            value= [  1.0,
                     -3.74145562,
                      5.25726624,
                     -3.28776591,
                      0.77203984 ] + [0] * 11,
            localSet=lambda value: self.device.setA(value),
            localGet=self.device.getA))

        # Add the filter b coefficients variable
        # Rogue doesn't allow to have an empty list here. Also, the EPICS PV is created
        # with the initial size of this list, and can not be changed later, so we are doing
        # it big enough at this point (we are probably not going to use an order > 10)
        self.add(pyrogue.LocalVariable(
            name='B',
            description='Filter b coefficients',
            mode='RW',
            value= [ 5.28396689e-06,
                     2.11358676e-05,
                     3.17038014e-05,
                     2.11358676e-05,
                     5.28396689e-06 ] + [0] * 11,
            localSet=lambda value: self.device.setB(value),
            localGet=self.device.getB))

        # Command to reset the filter
        self.add(pyrogue.LocalCommand(
            name='reset',
            description='Reset the unwrapper',
            function=self.device.resetFilter))


class SmurfProcessor(pyrogue.Device):
    """
    SMuRF Processor device.

    This is a slave device that accepts a raw SMuRF Streaming data
    stream from the FW application, and process it by doing channel
    mapping, data unwrapping, filtering and downsampling in a
    monolithic C++ module.

    Args
    ----
    name : str
        Name of the device.
    description : str
        Description of the device.
    root : pyrogue.Root or None, optional, default None
        The pyrogue root. The configuration status of this root will
        go to the data file as metadata.
    txDevice : pyrogue.Device or None, optional, default None
        A packet transmitter device.
    """
    def __init__(self, name, description, root=None, txDevice=None, **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)

        # Add a data emulator module, at the beginning of the chain
        self.pre_data_emulator = pysmurf.core.emulators.StreamDataEmulatorI16(name="PreDataEmulator")
        self.add(self.pre_data_emulator)

        # Add a frame statistics module
        self.smurf_frame_stats = pysmurf.core.counters.FrameStatistics(name="FrameRxStats")
        self.add(self.smurf_frame_stats)

        # Add the SmurfProcessor C++ object. This is the .so file
        # compiled by make, and Python finds it via PYTHONPATH. This
        # module implements: channel mapping, data unwrapping, filter,
        # and downsampling. Python wrapper for these functions are
        # added here to give the same tree structure as the modular
        # version. The functions are defined in setup_python in
        # SmurfProcessor.cpp.
        self.smurf_processor = smurf.core.processors.SmurfProcessor()

        self.smurf_mapper = SmurfChannelMapper(name="ChannelMapper", device=self.smurf_processor)
        self.add(self.smurf_mapper)

        self.smurf_unwrapper = Unwrapper(name="Unwrapper", device=self.smurf_processor)
        self.add(self.smurf_unwrapper)

        self.smurf_filter = GeneralAnalogFilter(name="Filter", device=self.smurf_processor)
        self.add(self.smurf_filter)

        self.smurf_downsampler = Downsampler(name="Downsampler", device=self.smurf_processor)
        self.add(self.smurf_downsampler)

        # This device doesn't have any user configurations, so we don't add it to the tree
        self.smurf_header2smurf = pysmurf.core.conventers.Header2Smurf(name="Header2Smurf")

        # Add a data emulator module, at the end of the chain
        self.post_data_emulator = pysmurf.core.emulators.StreamDataEmulatorI32(name="PostDataEmulator")
        self.add(self.post_data_emulator)

        # Use a standard Rogue file writer.
        # - Channel 0 will be use for the smurf data
        # - Channel 1 will be use for the configuration data (aka metadata)
        self.file_writer = pyrogue.utilities.fileio.StreamWriter(name='FileWriter')
        self.add(self.file_writer)

        # Connect devices
        pyrogue.streamConnect(self.pre_data_emulator,  self.smurf_frame_stats)
        pyrogue.streamConnect(self.smurf_frame_stats,  self.smurf_processor)
        pyrogue.streamConnect(self.smurf_processor,    self.smurf_header2smurf)
        pyrogue.streamConnect(self.smurf_header2smurf, self.post_data_emulator)
        pyrogue.streamConnect(self.post_data_emulator, self.file_writer.getChannel(0))

        # If a root was defined, connect it to the file writer, on channel 1
        if root:
            pyrogue.streamConnect(root, self.file_writer.getChannel(1))

        # If a transmitter device was defined, add it to the tree and connect it to the chain
        if txDevice:
            self.transmitter = txDevice

            # Add a fifo for the data frames. It will hold up to 100 copies of processed frames,
            # to be send to the transmitter device. If the fifo is full, new frames will be dropped.
            # The frames will be tapped after the data emulator, and before the file writer.
            self.fifo_data = pyrogue.interfaces.stream.Fifo(
                name='DataFifo', description="Data Fifo", maxDepth=100, trimSize=0, noCopy=False)
            self.add(self.fifo_data)

            # Tap the data frames at the output of the post data emulator, and send them to transmitter'
            # data channel, though the fifo.
            pyrogue.streamTap(    self.post_data_emulator, self.fifo_data)
            pyrogue.streamConnect(self.fifo_data,          self.transmitter.getDataChannel())

            # If a root was defined, connect  it to the transmitter's meta data channel.
            # Use streamTap as it was already connected to the file writer.
            if root:
                # Add a fifo for the meta data frames. It will hold up to 100 copies of the input frames,
                # to be send to the transmitter device. If the fifo is full, new frames will be dropped.
                # The frames will be tapped from the root, as the root was already connected to the file
                # writer.
                self.fifo_meta = pyrogue.interfaces.stream.Fifo(
                    name='MetaFifo', description="Meta data Fifo", maxDepth=100, trimSize=0, noCopy=False)
                self.add(self.fifo_meta)

                # Tap the metadata frames from the root, and send them to the transmitter's meta data
                # channel, through the fifo.
                pyrogue.streamTap(    root,           self.fifo_meta)
                pyrogue.streamConnect(self.fifo_meta, self.transmitter.getMetaChannel())

            # Add the transmitter device to the root here, after adding the fifos, so that it appears
            # in the correct order in the rogue tree.
            self.add(self.transmitter)

    def _getStreamSlave(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access slave.
        We will pass a reference to the smurf device of the first element in the chain,
        which is the 'FrameStatistics'.
        """
        return self.pre_data_emulator.getSmurfDevice()
