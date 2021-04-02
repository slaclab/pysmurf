#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : SMuRF Band Phase Feedback Module
#-----------------------------------------------------------------------------
# File       : _BandPhaseFeedback.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Band Phase Feedback  Python Package
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

class BandPhaseFeedback(pyrogue.Device):
    """
    SMuRF Band Phase Feedback Python Wrapper.
    """
    def __init__(self, name, **kwargs):
        self._BandPhaseFeedback = smurf.core.feedbacks.BandPhaseFeedback()
        pyrogue.Device.__init__(self, name=name, description='SMuRF Band Phase Feedback', **kwargs)

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self._BandPhaseFeedback.setDisable(value),
            localGet=self._BandPhaseFeedback.getDisable))

        # Add the frame counter variable
        self.add(pyrogue.LocalVariable(
            name='FrameCnt',
            description='Frame counter',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getFrameCnt))

        # Add the bad frame counter variable
        self.add(pyrogue.LocalVariable(
            name='BadFrameCnt',
            description='Number of bad frames',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getBadFrameCnt))

        # Command to clear all the counters
        self.add(pyrogue.LocalCommand(
            name='clearCnt',
            description='Clear all counters',
            function=self._BandPhaseFeedback.clearCnt))

    # Method called by streamConnect, streamTap and streamConnectBiDir to access slave
    def _getStreamSlave(self):
        return self._BandPhaseFeedback

    # Method called by streamConnect, streamTap and streamConnectBiDir to access master
    def _getStreamMaster(self):
        return self._BandPhaseFeedback
