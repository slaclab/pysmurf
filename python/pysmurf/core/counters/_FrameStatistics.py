#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Frame Statistics
#-----------------------------------------------------------------------------
# File       : _FrameStatistics.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Frame Statistics Python Package
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

class FrameStatistics(pyrogue.Device):
    """
    SMuRF Frame Statistics Python Wrapper.
    """
    def __init__(self, name, **kwargs):
        self._FrameStatistics = smurf.core.counters.FrameStatistics()
        pyrogue.Device.__init__(self, name=name, description='SMuRF Frame Statistics', **kwargs)

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self._FrameStatistics.setDisable(value),
            localGet=self._FrameStatistics.getDisable))

        # Add the frame counter variable
        self.add(pyrogue.LocalVariable(
            name='FrameCnt',
            description='Frame counter',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._FrameStatistics.getFrameCnt))

        # Add the last frame size variable
        self.add(pyrogue.LocalVariable(
            name='FrameSize',
            description='Last frame size (bytes)',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._FrameStatistics.getFrameSize))

        # Add the frame lost counter  variable
        self.add(pyrogue.LocalVariable(
            name='FrameLossCnt',
            description='Number of lost frames',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._FrameStatistics.getFrameLossCnt))

        # Add the out-of-order frames variable
        self.add(pyrogue.LocalVariable(
            name='FrameOutOrderCnt',
            description='Number of time we have received out-of-order frames',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._FrameStatistics.getFrameOutOrderCnt))

        # Command to clear all the counters
        self.add(pyrogue.LocalCommand(
            name='clearCnt',
            description='Clear all counters',
            function=self._FrameStatistics.clearCnt))

    # Method called by streamConnect, streamTap and streamConnectBiDir to access slave
    def _getStreamSlave(self):
        return self._FrameStatistics

    # Method called by streamConnect, streamTap and streamConnectBiDir to access master
    def _getStreamMaster(self):
        return self._FrameStatistics
