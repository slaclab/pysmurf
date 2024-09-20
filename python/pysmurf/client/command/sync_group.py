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
    def __init__(self, pv, loop, check_value=None, skip_first=True):
        self.loop = loop
        self.future = loop.create_future()
        self.first = skip_first
        self.val = check_value
        self.pv = epics.PV(pv, auto_monitor=True)
        self._cb_id = self.pv.add_callback(self.callback)

    def callback(self, pvname, value, *args, **kwargs):
        # don't fill on initial connection
        if self.first:
            print("skipping first callback")
            self.first = False
        elif (self.val is not None) and (value != self.val):
            print(f"not done yet: val = {value}")
            print(f"              type(val) = {type(value)}")
        else:
            print(f"callback: setting value: {value}")
            # must be careful because callback is being called from another thread
            self.loop.call_soon_threadsafe(self.future.set_result, value)
            self.pv.remove_callback(self._cb_id)


class AsyncGroup(object):
    def __init__(self, pvs, skip_first=True):
        self.pvnames = pvs
        self.skip_first = skip_first

    async def wait(self, check_value=None, timeout=30.0):
        # set up async callbacks
        loop = asyncio.get_running_loop()
        futures = []
        for pv in self.pvnames:
            futures.append(FuturePV(pv, loop, skip_first=self.skip_first, check_value=check_value))

        try:
            # for some reason the future is not finishing and this times out
            done, pending = await asyncio.wait([f.future for f in futures], timeout=timeout)
            if len(pending) > 0:
                raise Exception(f"Timed out after {timeout}s.")
            res = [f.result() for f in done]
            return res
        finally:
            # TODO check that this effectively removes callback
            del futures
