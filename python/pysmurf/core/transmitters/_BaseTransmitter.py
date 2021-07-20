#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Data Base Transmitter
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data Base Transmitter Python Package
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

class BaseTransmitter(pyrogue.Device):
    """
    SMuRF Data BaseTransmitter Python Wrapper.
    """
    def __init__(self, name, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Data BaseTransmitter', **kwargs)
        self._transmitter = smurf.core.transmitters.BaseTransmitter()

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self._transmitter.setDisable(value),
            localGet=self._transmitter.getDisable))


        # Add the data frame counter variable
        self.add(pyrogue.LocalVariable(
            name='dataFrameCnt',
            description='Number of data frame received',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self._transmitter.getDataFrameCnt))

        # Add the metaData frame counter variable
        self.add(pyrogue.LocalVariable(
            name='metaFrameCnt',
            description='Number of metadata frame received',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self._transmitter.getMetaFrameCnt))

        # Add the data dropped counter variable
        self.add(pyrogue.LocalVariable(
            name='dataDropCnt',
            description='Number of data frame dropped',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self._transmitter.getDataDropCnt))

        # Add the metaData dropped counter variable
        self.add(pyrogue.LocalVariable(
            name='metaDropCnt',
            description='Number of metadata frame dropped',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self._transmitter.getMetaDropCnt))

        # Command to clear all the counters
        self.add(pyrogue.LocalCommand(
            name='clearCnt',
            description='Clear all counters',
            function=self._transmitter.clearCnt))

    def getDataChannel(self):
        return self._transmitter.getDataChannel()

    def getMetaChannel(self):
        return self._transmitter.getMetaChannel()
