#-----------------------------------------------------------------------------
# Title      : PySMuRF Data GCP Transmitter
#-----------------------------------------------------------------------------
# File       : _GCPTransmitter.py
# Created    : 2020-10-23
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data GCPTransmitter Python Package
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

class GCPTransmitter(pyrogue.Device):
    def __init__(self, name, gcp_host, gcp_port, **kwargs):
        pyrogue.Device.__init__(self, name=name, description='SMuRF Data GCP Transmitter', **kwargs)
        self._transmitter = smurf.core.transmitters.GCPTransmitter(gcp_host, gcp_port)

        # Add "Disable" variable
        self.add(pyrogue.LocalVariable(
            name='Disable',
            description='Disable the processing block. Setting disable=False causes GCPTransmitter to re-connect to GCP.',
            mode='RW',
            value=False,
            localSet=lambda value: self._transmitter.setDisable(value),
            localGet=self._transmitter.getDisable))

        # Add a variable for the debugData flag
        self.add(pyrogue.LocalVariable(
            name='DebugData',
            description='Set the debug mode for the data',
            mode='RW',
            value=False,
            localSet=lambda value: self._transmitter.setDebugData(value),
            localGet=self._transmitter.getDebugData))

        # Add a variable for the debugMeta flag
        self.add(pyrogue.LocalVariable(
            name='DebugMeta',
            description='Set the debug mode for the metadata',
            mode='RW',
            value=False,
            localSet=lambda value: self._transmitter.setDebugMeta(value),
            localGet=self._transmitter.getDebugMeta))

        # Add the data dropped counter variable
        self.add(pyrogue.LocalVariable(
            name='DataDropCnt',
            description='Number of data frame dropped',
            mode='RO',
            value=0,
            pollInterval=1,
            localGet=self._transmitter.getDataDropCnt))

        # Add the metadata dropped counter variable
        self.add(pyrogue.LocalVariable(
            name='MetaDropCnt',
            description='Number of metadata frame dropped',
            mode='RO',
            value=0,
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
