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
import time

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

    _ckst = "CLKST"

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

    _phyreadyrx = "MPSLNK:PHYREADYRX"

    def get_timing_phyreadyrx(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._phyreadyrx, **kwargs)

    _phyreadytx = "MPSLNK:PHYREADYTX"

    def get_timing_phyreadytx(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._phyreadytx, **kwargs)

    _loclnkready = "MPSLNK:LOCLNKREADY"

    def get_timing_loclnkready(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._loclnkready, **kwargs)

    _remlnkready = "MPSLNK:REMLNKREADY"

    def get_timing_remlnkready(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._remlnkready, **kwargs)

    def get_timing_all(self, **kwargs):
        """
        Polls all the timing PVs
        """
        timing_status = {}
        timing_status['timing_state'] = self.get_timing_state(**kwargs)
        timing_status['time_skew'] = self.get_timing_time_skew(**kwargs)
        timing_status['clock_rate'] = self.get_timing_clock_rate(**kwargs)
        timing_status['pll_count'] = self.get_timing_pll_count(**kwargs)
        timing_status['base_rate'] = self.get_timing_base_rate(**kwargs)
        timing_status['sync_err'] = self.get_timing_sync_err(**kwargs)
        timing_status['interval_counter'] = self.get_timing_interval_counter(**kwargs)
        timing_status['phyreadyrx'] = self.get_timing_phyreadyrx(**kwargs)
        timing_status['phyreadytx'] = self.get_timing_phyreadytx(**kwargs)
        timing_status['loclnkready'] = self.get_timing_loclnkready(**kwargs)
        timing_status['remlnkready'] = self.get_timing_remlnkready(**kwargs)

        return timing_status


    def check_timing(self, n=3, wait_time=1, **kwargs):
        """
        This takes a few seconds because it attempts to see if some values are
        constant over time.
        """

        # Query timing system several times
        timing_status_dict = {}
        for i in np.arange(n):
            self.log(f'{i+1} of {n}')
            timing_status_dict[i] = self.get_timing_all(**kwargs)
            time.sleep(wait_time)

        # Set default success bit
        success = True
        failure_list = np.array([], dtype='str')

        # check all constant values
        ready_bit = 0
        keys = ['phyreadyrx', 'phyreadytx', 'loclnkready', 'remlnkready']
        for i in np.arange(n):
            for k in keys:
                if timing_status_dict[i][k] != ready_bit:
                    success = False
                    failure_list = failure_list.append(k)

        normal_bit = 1
        keys = ['timing_state']
        for i in np.arange(n):
            for k in keys:
                if timing_status_dict[i][k] != normal_bit:
                    success = False
                    failure_list = failure_list.append(k)

        # checking time skew
        k = 'time_skew'
        skew_max = 50  # microseconds
        for i in np.arange(n):
            if np.abs(timing_status_dict[i][k]) > skew_max:
                success = False
                failure_list = failure_list.append(k)

        # check clock_rate
        k = 'clock_rate'
        clkrate_var = .1
        clkrate_med = 186.
        for i in np.arange(n):
            if np.abs(timing_status_dict[i][k]-clkrate_med) > clkrate_var:
                success = False
                failure_list = failure_list.append(k)

        # Check constant values
        keys = ['pll_count', 'base_rate', 'interval_counter']
        for i in np.arange(n-1):
            for k in keys:
                if timing_status_dict[i][k] != timing_status_dict[+1][k]:
                    success = False
                    failure_list = failure_list.append(k)

        # check sync error counter
        k = 'sync_err'
        for i in np.arange(n):
            if np.abs(timing_status_dict[i][k]) !=0:
                success = False
                failure_list = failure_list.append(k)

        if not success:
            self.log('Timing system not properly configured.')
            self.log(f'The error is likely in : ')
            for f in failure_list:
                self.log(f'   {f}')

        return success