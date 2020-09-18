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

    _time_skew = "TIMESKEW"
    def get_time_skew(self, **kwargs):
        """
        """
        return self._caget(self.timing_root + self._time_skew, **kwargs)
