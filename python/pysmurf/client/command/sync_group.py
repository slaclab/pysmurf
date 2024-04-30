#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf command module - SyncGroup class
#-----------------------------------------------------------------------------
# File       : pysmurf/command/sync_group.py
# Created    : 2018-09-18
# Author     : Mitch D'Ewart
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
from pyrogue import VariableWait

"""
This class is written by Mitch to read PVs simultanesouly
"""
class SyncGroup(object):
    def __init__(self, pvs, client, timeout=30.0):
        self.pvnames = pvs
        self.timeout = timeout
        self.pvs = []
        for pvname in pvs:
            node = client.root.getNode(pvname)
            if node is None:
                raise ValueError(f"Failed to get node {pvname}")
            self.pvs.append(node)

    def get_values(self):
        return {self.pvnames[i]: self.pvs[i].get() for i in range(len(self.pvs))}

    # blocking wait for all to complete
    def wait(self):
        VariableWait(
            self.pvs,
            lambda varValues: all([v.updated for v in varValues]),
            timeout=self.timeout
        )
