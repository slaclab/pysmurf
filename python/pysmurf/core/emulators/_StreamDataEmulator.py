#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF StreamDataEmulator
#-----------------------------------------------------------------------------
# File       : _StreamDataEmulator.py
# Created    : 2019-10-29
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data Emulator
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
import smurf.core.emulators

class StreamDataEmulator(pyrogue.Device):
    """
    StreamDataEmulator Block
    """
    def __init__(self, name="StreamDataEmulator", description="SMURF Data Emulator", **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)

        self._emulator = smurf.core.emulators.StreamDataEmulator()

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=True,
            localSet=lambda value: self._emulator.setDisable(value),
            localGet=self._emulator.getDisable))

        # Add "Mode" variable
        self.add(pyrogue.LocalVariable(
            name='Type',
            description='Set type of signal',
            mode='RW',
            value=0,
            disp={
                0 : 'Zeros',
                1 : 'ChannelNumber',
                2 : 'Random',
                3 : 'Square',
                4 : 'Sawtooth',
                5 : 'Triangle',
                6 : 'Sine',
            },
            localSet=lambda value: self._emulator.setType(value),
            localGet=self._emulator.getType))

        # Add "Amplitude" variable
        self.add(pyrogue.LocalVariable(
            name='Amplitude',
            description='Signal amplitude (it is an uint16)',
            mode='RW',
            typeStr='UInt16',
            value=65535,
            localSet=lambda value: self._emulator.setAmplitude(value),
            localGet=self._emulator.getAmplitude))

        # Add "Offset" variable
        self.add(pyrogue.LocalVariable(
            name='Offset',
            description='Signal offset (it is an uint16)',
            mode='RW',
            typeStr='UInt16',
            value=0,
            localSet=lambda value: self._emulator.setOffset(value),
            localGet=self._emulator.getOffset))

        # Add "Period" variable
        self.add(pyrogue.LocalVariable(
            name='Period',
            description='Signal period, in multiples of flux ramp periods. Can not be set to zero (it is an uint32)',
            mode='RW',
            typeStr='UInt32',
            value=1,
            localSet=lambda value: self._emulator.setPeriod(value),
            localGet=self._emulator.getPeriod))

    def _getStreamSlave(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access slave.
        """
        return self._emulator

    def _getStreamMaster(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access master.
        """
        return self._emulator


