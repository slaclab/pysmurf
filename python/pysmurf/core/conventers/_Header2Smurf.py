#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Data Header2Smurf
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data Header2Smurf Python Package
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

class Header2Smurf(pyrogue.Device):
    """
    SMuRF Data Header2Smurf Python Wrapper.
    """
    def __init__(self, name, **kwargs):
        self._header2smurf = smurf.core.conventers.Header2Smurf()
        pyrogue.Device.__init__(self, name=name, description='Convert the frame header to the SMuRF server', **kwargs)

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Data will just pass thorough to the next slave.',
            mode='RW',
            value=False,
            localSet=lambda value: self._header2smurf.setDisable(value),
            localGet=self._header2smurf.getDisable))

    # Method to set TES Bias values
    def setTesBias(self, index, value):
        self._header2smurf.setTesBias(index, value)

    # Method called by streamConnect, streamTap and streamConnectBiDir to access slave
    def _getStreamSlave(self):
        return self._header2smurf

    # Method called by streamConnect, streamTap and streamConnectBiDir to access master
    def _getStreamMaster(self):
        return self._header2smurf
