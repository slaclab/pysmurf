#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF StreamDataEmulator
#-----------------------------------------------------------------------------
# File       : _StreamDataEmulatorI16.py
# Created    : 2019-10-29
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data Emulator, with output data type Int16
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

class StreamDataEmulatorI16(pyrogue.Device):
    """
    StreamDataEmulatorI16 Block
    """
    def __init__(self, name="StreamDataEmulatorI16", description="SMURF Data Emulator, with output data type Int16", **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)

        self._emulator = smurf.core.emulators.StreamDataEmulatorI16()

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next subordinate.',
            mode='RW',
            value=True,
            localSet=lambda value: self._emulator.setDisable(value),
            localGet=self._emulator.getDisable))

        # Add "Mode" variable
        self.add(pyrogue.LocalVariable(
            name='Type',
            description='Set type of signal.',
            mode='RW',
            disp={
                0 : 'Zeros',
                1 : 'ChannelNumber',
                2 : 'Random',
                3 : 'Square',
                4 : 'Sawtooth',
                5 : 'Triangle',
                6 : 'Sine',
                7 : 'DropFrame',
            },
            localSet=lambda value: self._emulator.setType(value),
            localGet=self._emulator.getType))

        # Add "Amplitude" variable
        self.add(pyrogue.LocalVariable(
            name='Amplitude',
            description='Signal peak amplitude.',
            mode='RW',
            typeStr='UInt16',
            pollInterval=1,
            localSet=lambda value: self._emulator.setAmplitude(value),
            localGet=self._emulator.getAmplitude))

        # Add "Offset" variable
        self.add(pyrogue.LocalVariable(
            name='Offset',
            description='Signal offset.',
            mode='RW',
            typeStr='Int16',
            pollInterval=1,
            localSet=lambda value: self._emulator.setOffset(value),
            localGet=self._emulator.getOffset))

        # Add "Period" variable
        self.add(pyrogue.LocalVariable(
            name='Period',
            description='Signal period, in multiples of flux ramp periods. Can not be set to zero.',
            mode='RW',
            typeStr='UInt64',
            localSet=lambda value: self._emulator.setPeriod(value),
            localGet=self._emulator.getPeriod))

    def getSmurfDevice(self):
        """
        Returns a reference to the underlying smurf device.
        """
        return self._emulator

    def _getStreamSubordinate(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access subordinate.
        """
        return self._emulator

    def _getStreamMain(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access main.
        """
        return self._emulator
