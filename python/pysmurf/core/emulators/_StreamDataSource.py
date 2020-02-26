#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF StreamDataSource
#-----------------------------------------------------------------------------
# File       : _StreamDataSource.py
# Created    : 2019-10-29
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data Source
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

class StreamDataSource(pyrogue.Device):
    """
    StreamDataSource Block
    """
    def __init__(self, name="StreamDataSource", description="SMURF Data Source", **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)

        self._source = smurf.core.emulators.StreamDataSource()

        self.add(pyrogue.LocalVariable(
            name='SourceEnable',
            description='Frame generation enable',
            mode='RW',
            value=False,
            localGet = lambda: self._source.getSourceEnable(),
            localSet = lambda value: self._source.setSourceEnable(value)))

        self.add(pyrogue.LocalVariable(
            name='Period',
            description='Frame generation period in S',
            mode='RW',
            value=0.0,
            localGet = lambda: self._source.getSourcePeriod() / 1e6,
            localSet = lambda value: self._source.setSourcePeriod(int(value*1e6))))

        self.add(pyrogue.LocalVariable(
            name='CrateId',
            description='Frame generation crate ID',
            mode='RW',
            value=0,
            localGet = lambda: self._source.getCrateId(),
            localSet = lambda value: self._source.setCrateId(value)))

        self.add(pyrogue.LocalVariable(
            name='SlotNum',
            description='Frame generation slot #',
            mode='RW',
            value=0,
            localGet = lambda: self._source.getSlotNum(),
            localSet = lambda value: self._source.setSlotNum(value)))

    def _getStreamSlave(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access slave.
        """
        return self._source

    def _getStreamMaster(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access master.
        """
        return self._source
