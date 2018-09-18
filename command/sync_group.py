import time
import epics

"""
This class is written by Mitch to read PVs simultanesouly
"""
class SyncGroup(object):
    def __init__(self, pvs, timeout=30.0):
        self.pvnames = pvs
        self.values = dict()
        self.timeout = timeout
        self.first = [True] * len(pvs)
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
    def wait(self):
        t0 = time.time()
        while not all([n in self.values for n in self.pvnames]):
            epics.ca.poll()  # better performance using this over time.sleep()
            if time.time() - t0 > self.timeout:
                raise Exception('Timeout waiting for PVs to update.')