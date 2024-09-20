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
import time
import asyncio

"""
This class is written by Mitch to read PVs simultanesouly
"""
class SyncGroup(object):
    def __init__(self, pvs, client, timeout=30.0):
        self.pvnames = pvs
        self.timeout = timeout
        self.pvs = {}
        self.updated = {}
        #self.vals = {}
        for pvname in pvs:
            node = client.root.getNode(pvname)
            if node is None:
                raise ValueError(f"Failed to get node {pvname}")
            self.pvs[pvname] = node
            self.updated[pvname] = False
            node.addListener(self._receive_update)

    def _receive_update(self, path, val):
        if path in self.pvs:
            self.updated[path] = True
            #self.vals[path] = val.get()

    def get_values(self, read=False):
        #if read or (not all(self.updated.values())):
        #    return {self.pvnames[i]: self.pvs[i].get() for i in range(len(self.pvs))}
        #else:
        #    return self.vals
        return {k: self.pvs[k].get() for k in self.pvs}

    # blocking wait for all to complete
    def wait(self):
        done = lambda: all(self.updated.values())
        start = time.time()
        while not done() and ((time.time() - start) < self.timeout):
            continue

class AsyncPVWait(object):
    """Asynchronously wait for a PV to update to a given value."""

    def __init__(self, pv, client, check_val=None, timeout=30.0):
        self.client = client
        self.pvname = pv
        self.check_val = check_val
        self.timeout = timeout
        self.updated = False

        self.task = asyncio.Event()  # will signal to async wait
        self.loop = asyncio.get_running_loop()

        # set up listener
        node = client.root.getNode(pv)
        if node is None:
            raise ValueError(f"Failed to get node {pv}")
        node.addListener(self._receive_update)

    def _receive_update(self, path, val, *args, **kwargs):
        if path != self.pvname:
            self.log(f"Received an update for an unexpected PV: {path}.")
            return

        if (self.check_val is None) or (val.value == self.check_val):
            self.updated = True
            # this is probably being called by the PV on another thread
            self.loop.call_soon_threadsafe(self.task.set)

            # remove references to callback
            node = self.client.root.getNode(path)
            node.delListener(self._receive_update)

    async def async_wait(self):
        try:
            await asyncio.wait_for(self.task.wait(), self.timeout)
        except TimeoutError:
            raise TimeoutError(f"Timed out after waiting {self.timeout}s.")
