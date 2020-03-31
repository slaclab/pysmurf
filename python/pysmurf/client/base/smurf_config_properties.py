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

from pysmurf.client.base.smurf_config import SmurfConfig as SmurfConfig

class SmurfConfigPropertiesMixin(object):
    """Mixin for ???x
    """

    def __init__(self, *args, **kwargs):
        # Constants
        self._pA_per_phi0=None

        # Amplifiers
        self._hemt_Vg=None
        self._LNA_Vg=None

    def copy_config_to_properties(self,config):
        """Copy values from SmurfConfig instance to properties
        
        MORE EXPLANATION HERE. USES PROPERTY SETTERS IN CASE WE EVER
        WANT TO IMPOSE ANY CONDITIONS IN THEM.

        Args
        ----
        config : :class:`SmurfConfig`
              Select values in the provided :class:`SmurfConfig` will
              be copied into properties in this
              :class:`~pysmurf.client.command.smurf_config_properties.SmurfConfigPropertiesMixin`
              instance.
        """
        
        # Useful constants
        constant_cfg = config.get('constant')
        self.pA_per_phi0 = constant_cfg.get('pA_per_phi0')

        # Cold amplifier biases
        amp_cfg = self.config.get('amplifier')
        keys = amp_cfg.keys()
        if 'hemt_Vg' in keys:
            self.hemt_Vg=amp_cfg['hemt_Vg']
        if 'LNA_Vg' in keys:
            self.LNA_Vg=amp_cfg['LNA_Vg']

    ###########################################################################
    ## Start pA_per_phi0 property definition

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

    ## End pA_per_phi0 property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_Vg property definition

    # Getter
    @property
    def hemt_Vg(self):
        """4K HEMT Gate Voltage
        
        Gets or sets the desired value for the 4K HEMT Gate voltage.
        Units are Volts.
        """
        return self._hemt_Vg

    # Setter
    @hemt_Vg.setter
    def hemt_Vg(self,value):
        self._hemt_Vg=value

    ## End hemt_Vg property definition
    ###########################################################################

    ###########################################################################
    ## Start LNA_Vg property definition

    # Getter
    @property
    def LNA_Vg(self):
        """50K LNA Gate Voltage
        
        Gets or sets the desired value for the 50K LNA Gate voltage.
        Units are Volts.
        """
        return self._LNA_Vg

    # Setter
    @LNA_Vg.setter
    def LNA_Vg(self,value):
        self._LNA_Vg=value

    ## End LNA_Vg property definition
    ###########################################################################        
