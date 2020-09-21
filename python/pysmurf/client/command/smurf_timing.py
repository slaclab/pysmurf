#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf command module - SmurfTimingMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/command/smurf_timing.py
# Created    : 2020-09-17
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
"""Defines the :class:`SmurfTimingMixin` class."""
from pysmurf.client.base import SmurfBase

class SmurfTimingMixin(SmurfBase):
    """Mixin providing interface with the atca_monitor server.

    This Mixin provides the pysmurf interface to the atca_monitor
    registers.  The atca_monitor server is a Rogue application
    which uses IPMI to monitor information from the ATCA system
    [#atca_monitor]_.  The atca_monitor server must be
    running or all queries will timeout and return `None`.

    References
    ----------
    .. [#atca_monitor] https://github.com/slaclab/smurf-atca-monitor

    """

    # To do - Move to base
    timing_root = "TPG:SMRF:1:"

    _ckst = "CKST"

    def get_timing_state(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._ckst, **kwargs)


    _time_skew = "TIMESKEW"

    def get_timing_time_skew(self, **kwargs):
        """ Time skew between NPT and TPG (in micro-second)

        Returns
        -------
        skew : float
            Time between NPT and TPG in micro-seconds
        """
        return self._caget(self.timing_root + self._time_skew, **kwargs)

    _ratetxclk = "RATETXCLK"

    def get_timing_clock_rate(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._ratetxclk, **kwargs)

    _countpll = "COUNTPLL"

    def get_timing_pll_count(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._countpll, **kwargs)

    _countbrt = "COUNTBRT"

    def get_timing_base_rate(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._countbrt, **kwargs)

    _countsyncerr = "COUNTSYNCERR"

    def get_timing_sync_err(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._countsyncerr, **kwargs)

    _countintv = "COUNTINTV"

    def get_timing_interval_counter(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._countintv, **kwargs)

    _phyreadyrx = "PHYREADYRX"

    def get_timing_phyreadyrx(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._phyreadyrx, **kwargs)

    _phyreadytx = "PHYREADYTX"

    def get_timing_phyreadytx(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._phyreadytx, **kwargs)

    _loclnkready = "LOCLNKREADY"

    def get_timing_loclnkready(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._loclnkready, **kwargs)

    _remlnkready = "REMLNKREADY"

    def get_timing_remlnkready(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._remlnkready, **kwargs)

    def get_timing_all(self):
        """
        Polls all the timing PVs
        """
        timing_status = {}
        timing_status['timing_state'] = self.get_timing_state()
        timing_status['time_skew'] = self.get_timing_time_skew()
        timing_status['clock_rate'] = self.get_timing_clock_rate()
        timing_status['pll_count'] = self.get_timing_pll_count()
        timing_status['base_rate'] = self.get_timing_base_rate()
        timing_status['sync_err'] = self.get_timing_sync_err()
        timing_status['interval_counter'] = self.get_timing_interval_counter()
        timing_status['phyreadyrx'] = self.get_timing_phyreadyrx()
        timing_status['phyreadytx'] = self.get_timing_phyreadytx()
        timing_status['loclnkready'] = self.get_timing_loclnkready()
        timing_status['remlnkready'] = self.get_timing_remlnkready()

        return timing_status
