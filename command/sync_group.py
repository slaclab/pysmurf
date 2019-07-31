import time
import epics

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
