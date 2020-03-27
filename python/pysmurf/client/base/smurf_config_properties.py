#!/usr/bin/env python
# -----------------------------------------------------------------------------
# Title : pysmurf smurf_config_properties module -
#         SmurfConfigPropertiesMixin class
# -----------------------------------------------------------------------------
# File : pysmurf/client/base/smurf_config_properties.py Created : 2020-03-27
# -----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level
# directory of this distribution and at:
# https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the
# terms contained in the LICENSE.txt file.
# -----------------------------------------------------------------------------
"""Defines the SmurfConfigPropertiesMixin class.
"""
from pysmurf.client.base import SmurfBase

class SmurfConfigPropertiesMixin(SmurfBase):
    """Mixin for ???x
    """

    ###########################################################################
    ## Start pA_per_phi0 property definition

    # Internal attribute 
    _pA_per_phi0=None
    
    # Getter
    @property
    def pA_per_phi0(self):
        """Conversion factor between demodulated phase and TES
        current.
        
        Gets or sets the conversion factor between the demodulated
        phase for every SMuRF channel and the equivalent TES current.
        Units are pA per Phi0, with Phi0 the magnetic flux quantum.
        """
        return self._pA_per_phi0

    # Setter
    @pA_per_phi0.setter
    def pA_per_phi0(self,value):
        self._pA_per_phi0=value

    # Deleter
    @pA_per_phi0.deleter
    def pA_per_phi0(self):
        del self._pA_per_phi0
    ## End pA_per_phi0 property definition
    ###########################################################################
        
