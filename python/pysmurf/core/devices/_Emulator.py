#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Emulator
#-----------------------------------------------------------------------------
# File       : _Emulator.py
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
import smurf.core.processors

class Emulator(pyrogue.Device):
    """
    Emulator Block
    """
    def __init__(self, name="Emulator", description="SMURF Data Emulator", **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)

        self._emulator = smurf.core.processor.Emulator()

        self.add(pyrogue.LocalVariable(
            name='SinEnable',
            description='SIN Enable',
            mode='RW',
            value=False,
            localGet = lambda: self._emulator.getSinEnable(),
            localSet = lambda value: self._emulator.setSinEnable(value)))

        self.add(pyrogue.LocalVariable(
            name='SinChannel',
            description='SIN Channel',
            mode='RW',
            value=0,
            localGet = lambda: self._emulator.getSinChannel(),
            localSet = lambda value: self._emulator.setSinChannel(value)))

        self.add(pyrogue.LocalVariable(
            name='SinAmplitude',
            description='SIN Amplitude (16-bit ADC Value)',
            mode='RW',
            value=0,
            localGet = lambda: self._emulator.getSinAmplitude(),
            localSet = lambda value: self._emulator.setSinAmplitude(value)))

        self.add(pyrogue.LocalVariable(
            name='SinBaseline',
            description='SIN Baseline (16-bit ADC Value)',
            mode='RW',
            value=0,
            localGet = lambda: self._emulator.getSinBaseline(),
            localSet = lambda value: self._emulator.setSinBaseline(value)))

        self.add(pyrogue.LocalVariable(
            name='SinPeriod',
            description='SIN Period (16-bit Sample Count)',
            mode='RW',
            value=0,
            localGet = lambda: self._emulator.getSinPeriod(),
            localSet = lambda value: self._emulator.setSinPeriod(value)))

    def _getStreamSlave(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access slave.
        We will pass a reference to the smurf device of the first element in the chain,
        which is the 'FrameStatistics'.
        """
        return self._emulator

    def _getStreamMaster(self):
        return self._emulator


