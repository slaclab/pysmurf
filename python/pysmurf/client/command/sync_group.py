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


class FuturePV(object):
    def __init__(self, loop, skip_first=True):
        self.future = loop.create_future()
        self.first = skip_first

    def callback(self, pvname, value, *args, **kwargs):
        # don't fill on initial connection
        if self.first:
            self.first = False
            return
        self.future.set_result(value)


class AsyncGroup(object):
    def __init__(self, pvs, skip_first=True):
        self.pvnames = pvs
        self.values = dict()
        self.futures = []
        self.skip_first = skip_first

    async def wait(self, timeout=30.0):
        # set up async callbacks
        loop = asyncio.get_running_loop()
        pvs = []
        self.futures = []
        for pv in self.pvnames:
            future = FuturePV(loop, self.skip_first)
            pvs.append(epics.PV(pv, callback=future.callback, auto_monitor=True))
            self.futures.append(future)

        async with asyncio.timeout(timeout):
            return [await f.future for f in self.futures]
