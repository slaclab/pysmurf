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

    Args
    ----
    namd : string
        Device name (must be unique).
    band : int
        Band number (0 to 7).
    """
    def __init__(self, name, band, **kwargs):
        self._BandPhaseFeedback = smurf.core.feedbacks.BandPhaseFeedback(band)
        pyrogue.Device.__init__(self, name=name, description='SMuRF Band Phase Feedback', **kwargs)

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self._BandPhaseFeedback.setDisable(value),
            localGet=self._BandPhaseFeedback.getDisable))

        # Add the band number variable
        self.add(pyrogue.LocalVariable(
            name='Band',
            description='Frame counter',
            mode='RO',
            value=0,
            typeStr='UInt64',
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getBand))

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

        # Add the number of channels in the incoming frame
        self.add(pyrogue.LocalVariable(
            name='NumChannels',
            description='Number of channels in the incoming frame',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getNumCh))

        # Command to clear all the counters
        self.add(pyrogue.LocalCommand(
            name='clearCnt',
            description='Clear all counters',
            function=self._BandPhaseFeedback.clearCnt))

        # Tone channels
        # Rogue doesn't allow to have an empty list here. Also, the EPICS PV is created
        # with the initial size of this list, and can not be changed later, so we are doing
        # it big enough at this point, using the maximum allowed number of tones.
        self.add(pyrogue.LocalVariable(
            name='toneChannels',
            description='Tone channel numbers',
            mode='RW',
            value= [0] * 10,
            localSet=lambda value: self._BandPhaseFeedback.setToneChannels(value),
            localGet=self._BandPhaseFeedback.getToneChannels))

        # Tone channels
        # Rogue doesn't allow to have an empty list here. Also, the EPICS PV is created
        # with the initial size of this list, and can not be changed later, so we are doing
        # it big enough at this point, using the maximum allowed number of tones.
        self.add(pyrogue.LocalVariable(
            name='toneFrequencies',
            description='Tone frequencies',
            mode='RW',
            value= [0.0] * 10,
            units='GHz',
            localSet=lambda value: self._BandPhaseFeedback.setToneFrequencies(value),
            localGet=self._BandPhaseFeedback.getToneFrequencies))

        # Add the data valid flag.
        self.add(pyrogue.LocalVariable(
            name='DataValid',
            description='The input parameters are valid',
            mode='RO',
            value=False,
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getDataValid))

        # Add the band phase slope estimation (tau).
        self.add(pyrogue.LocalVariable(
            name='Tau',
            description='Band estimated phase slope',
            mode='RO',
            value=0.0,
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getTau))

        # Add the band phase offset estimation (theta).
        self.add(pyrogue.LocalVariable(
            name='Theta',
            description='Band estimated phase offset',
            mode='RO',
            value=0.0,
            pollInterval=1,
            localGet=self._BandPhaseFeedback.getTheta))

    # Method called by streamConnect, streamTap and streamConnectBiDir to access slave
    def _getStreamSlave(self):
        return self._BandPhaseFeedback

    # Method called by streamConnect, streamTap and streamConnectBiDir to access master
    def _getStreamMaster(self):
        return self._BandPhaseFeedback
