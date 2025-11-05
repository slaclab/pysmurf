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
import threading

"""
Wait for a group of PVs to be updated before reading their values.
"""
class SyncGroup(object):
    def __init__(self, pvs, client, timeout=30.0):
        self.pvnames = pvs
        self.timeout = timeout
        self.pvs = {}
        self.updated = {}
        # thread condition for listener to update
        self.cond = threading.Condition()
        for pvname in pvs:
            node = client.root.getNode(pvname)
            if node is None:
                raise ValueError(f"Failed to get node {pvname}")
            with self.cond:
                self.pvs[pvname] = node
                self.updated[pvname] = False
                node.addListener(self._receive_update)

    def _receive_update(self, path, val):
        with self.cond:
            if path in self.pvs:
                self.updated[path] = True
                # notify waiting thread
                self.cond.notify()

    def get_values(self, read=False):
        return {k: self.pvs[k].get() for k in self.pvs}

    # blocking wait for all to complete
    def wait(self):
        with self.cond:
            def done():
                return all(self.updated.values())
            start = time()
            while not done() and ((time() - start) < self.timeout):
                self.cond.wait(0.5)
                continue
        if not done():
            raise TimeoutError(
                f"Timed out after {self.timeout}s for variables "
                f"{self.pvnames} to update."
            )
