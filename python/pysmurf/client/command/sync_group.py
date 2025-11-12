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
from time import time
from pyrogue import VariableWaitClass

"""
Wait for a group of PVs to be updated before reading their values.
"""
class SyncGroup(object):
    def __init__(self, pvs, client, timeout=30.0):
        self.pvnames = pvs
        self.pvs = {}
        for pvname in pvs:
            node = client.root.getNode(pvname)
            if node is None:
                raise ValueError(f"Failed to get node {pvname}")
            self.pvs[pvname] = node
        self._vw = VariableWaitClass(
                list(self.pvs.values()), timeout=timeout)
        self._vw.arm()  # adds listeners to variables

    def get_values(self):
        return {k: self.pvs[k].value() for k in self.pvs}

    # blocking wait for all to complete
    def wait(self):
        done = self._vw.wait()
        if not done:
            raise TimeoutError(
                f"Timed out after {self._vw._timeout}s for variables "
                f"{self.pvnames} to update."
            )
