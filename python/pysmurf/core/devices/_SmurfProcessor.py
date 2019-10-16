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
import smurf
import pysmurf.core.counters
import smurf.core.processors
import pysmurf.core.conventers

class SmurfChannelMapper(pyrogue.Device):
    """
    SMuRF Channel Mapper Python Wrapper.
    """
    def __init__(self, name, device, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Channel Mapper', **kwargs)
        self.device = device

        # Add the number of enabled channels  variable
        self.add(pyrogue.LocalVariable(
            name='NumChannels',
            description='Number enabled channels',
            mode='RO',
            value=0,
            pollInterval=1,
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
            value=[0]*528,
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

        # Add the filter order variable
        self.add(pyrogue.LocalVariable(
            name='Factor',
            description='Downsampling factor',
            mode='RW',
            value=20,
            localSet=lambda value : self.device.setFactor(value),
            localGet=self.device.getFactor))

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
            value= [ 1.0,
                     -3.9999966334205683,
                      5.9999899002673720,
                     -3.9999899002730395,
                      0.9999966334262355 ] + [0] * 11,
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
            value= [ 1.7218423734035440e-25,
                     6.8873694936141760e-25,
                     1.0331054240421264e-24,
                     6.8873694936141760e-25,
                     1.7218423734035440e-25 ] + [0] * 11,
            localSet=lambda value: self.device.setB(value),
            localGet=self.device.getB))


class SmurfProcessor(pyrogue.Device):
    """
    SMuRF Processor device.

    This device accept a raw SMuRF Streaming data stream from
    the FW application, and process it by doing channel mapping,
    data unwrapping, filtering and downsampling in a monolithic
    C++ module.
    """
    def __init__(self, name, description, master, **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)

        self.master = master

        self.smurf_frame_stats = pysmurf.core.counters.FrameStatistics(name="FrameRxStats")
        self.add(self.smurf_frame_stats)

        # Add the SmurfProcessor C++ device. This module implements: channel mapping,
        # data unwrapping, filter, and downsampling. Python wrapper for these functions
        # are added here to give the same tree structure as the modular version.
        self.smurf_processor = smurf.core.processors.SmurfProcessor()

        self.smurf_mapper = SmurfChannelMapper(name="ChannelMapper", device=self.smurf_processor)
        self.add(self.smurf_mapper)

        self.smurf_unwrapper = Unwrapper(name="Unwrapper", device=self.smurf_processor)
        self.add(self.smurf_unwrapper)

        self.smurf_filter = GeneralAnalogFilter(name="Filter", device=self.smurf_processor)
        self.add(self.smurf_filter)

        self.smurf_downsampler = Downsampler(name="Downsampler", device=self.smurf_processor)
        self.add(self.smurf_downsampler)

        self.smurf_header2smurf = pysmurf.core.conventers.Header2Smurf(name="Header2Smurf")
        self.add(self.smurf_header2smurf)

        self.file_writer = pyrogue.utilities.fileio.StreamWriter(name='FileWriter')
        self.add(self.file_writer)

        pyrogue.streamConnect(self.master,             self.smurf_frame_stats)
        pyrogue.streamConnect(self.smurf_frame_stats,  self.smurf_processor)
        pyrogue.streamConnect(self.smurf_processor,    self.smurf_header2smurf)
        pyrogue.streamConnect(self.smurf_header2smurf, self.file_writer.getChannel(0))

    # Method called by streamConnect, streamTap and streamConnectBiDir to access master
    def _getStreamMaster(self):
        return self._FrameStatistics
