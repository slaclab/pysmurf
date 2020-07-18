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
import time

try:
    import epics
except ModuleNotFoundError:
    print("sync_group.py - epics not found.")

"""
This class is written by Mitch to read PVs simultanesouly
"""
class SyncGroup(object):
    def __init__(self, pvs, timeout=30.0, skip_first=True):
        self.pvnames = pvs
        self.values = dict()
        self.timeout = timeout
        self.first = [skip_first] * len(pvs)
        self.pvs = [epics.PV(pv, callback=self.channel_changed,
            auto_monitor=True) for pv in pvs]

    def channel_changed(self, pvname, value, *args, **kwargs):
        # don't fill on initial connection
        if self.first[self.pvnames.index(pvname)]:
            self.first[self.pvnames.index(pvname)] = False
            return
        self.values[pvname] = value

    def get_values(self):
        val  = self.values
        self.values = dict()
        return val

    # non-blocking check if done
    def check(self):
        return all([n in self.values for n in self.pvnames])

    # clear if want to force re-arm
    def clear(self):
        self.values = dict()

    # blocking wait for all to complete
    def wait(self,epics_poll=False):
        t0 = time.time()
        while not all([n in self.values for n in self.pvnames]):
            if epics_poll:
                # better performance using this over time.sleep() but
                # it can eat a lot of CPU, so better to use epics.poll
                # for short waits/acquisitions where latency is a
                # concern.  For more details see:
                # https://cars9.uchicago.edu/software/python/pyepics3/advanced.html#time-sleep-or-epics-poll
                epics.ca.poll()
            else:
                time.sleep(.001)
            if time.time() - t0 > self.timeout:
                raise Exception('Timeout waiting for PVs to update.')
