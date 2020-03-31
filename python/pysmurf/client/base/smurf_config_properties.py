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
        ## 4K HEMT
        self._hemt_Vg=None
        self._hemt_bit_to_V=None
        self._hemt_Vd_series_resistor=None
        self._hemt_Id_offset=None
        self._hemt_gate_min_voltage=None
        self._hemt_gate_max_voltage=None        
        
        ## 50K LNA
        self._fiftyk_Vg=None
        self._fiftyk_dac_num=None
        self._fiftyk_bit_to_V=None
        self._fiftyk_amp_Vd_series_resistor=None
        self._fiftyk_Id_offset=None    
        
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

        ## 4K HEMT        
        if 'hemt_Vg' in keys:
            self.hemt_Vg=amp_cfg['hemt_Vg']
        if 'bit_to_V_hemt' in keys:
            self.hemt_bit_to_V=amp_cfg['bit_to_V_hemt']
        if 'hemt_Vd_series_resistor' in keys:
            self.hemt_Vd_series_resistor=amp_cfg['hemt_Vd_series_resistor']
        if 'hemt_Id_offset' in keys:
            self.hemt_Id_offset=amp_cfg['hemt_Id_offset']
        if 'hemt_gate_min_voltage' in keys:
            self.hemt_gate_min_voltage=amp_cfg['hemt_gate_min_voltage']
        if 'hemt_gate_max_voltage' in keys:
            self.hemt_gate_max_voltage=amp_cfg['hemt_gate_max_voltage']                        
            
        ## 50K HEMT                    
        if 'LNA_Vg' in keys:
            self.fiftyk_Vg=amp_cfg['LNA_Vg']
        if 'dac_num_50k' in keys:
            self.fiftyk_dac_num=amp_cfg['dac_num_50k']
        if 'bit_to_V_50k' in keys:
            self.fiftyk_bit_to_V=amp_cfg['bit_to_V_50k']
        if '50K_amp_Vd_series_resistor' in keys:
            self.fiftyk_amp_Vd_series_resistor=amp_cfg['50K_amp_Vd_series_resistor']
        if '50k_Id_offset' in keys:
            self.fiftyk_Id_offset=amp_cfg['50k_Id_offset']
            
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

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_noise.take_noise_psd`,
        :func:`~pysmurf.client.debug.smurf_noise.analyze_noise_vs_bias`,
        :func:`~pysmurf.client.debug.smurf_noise.analyze_noise_all_vs_noise_solo`,
        :func:`~pysmurf.client.debug.smurf_noise.analyze_noise_vs_tone`,
        :func:`~pysmurf.client.debug.smurf_iv.analyze_slow_iv`,
        :func:`~pysmurf.client.util.smurf_util.bias_bump`,
        :func:`~pysmurf.client.util.smurf_util.identify_bias_groups`
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
    ## Start hemt_bit_to_V property definition

    # Getter
    @property
    def hemt_bit_to_V(self):
        """???
        
        ???
        """
        return self._hemt_bit_to_V

    # Setter
    @hemt_bit_to_V.setter
    def hemt_bit_to_V(self,value):
        self._hemt_bit_to_V=value

    ## End hemt_bit_to_V property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_Vd_series_resistor property definition

    # Getter
    @property
    def hemt_Vd_series_resistor(self):
        """???
        
        ???
        """
        return self._hemt_Vd_series_resistor

    # Setter
    @hemt_Vd_series_resistor.setter
    def hemt_Vd_series_resistor(self,value):
        self._hemt_Vd_series_resistor=value

    ## End hemt_Vd_series_resistor property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_Id_offset property definition

    # Getter
    @property
    def hemt_Id_offset(self):
        """???
        
        ???
        """
        return self._hemt_Id_offset

    # Setter
    @hemt_Id_offset.setter
    def hemt_Id_offset(self,value):
        self._hemt_Id_offset=value

    ## End hemt_Id_offset property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_gate_min_voltage property definition

    # Getter
    @property
    def hemt_gate_min_voltage(self):
        """???
        
        ???
        """
        return self._hemt_gate_min_voltage

    # Setter
    @hemt_gate_min_voltage.setter
    def hemt_gate_min_voltage(self,value):
        self._hemt_gate_min_voltage=value

    ## End hemt_gate_min_voltage property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_gate_max_voltage property definition

    # Getter
    @property
    def hemt_gate_max_voltage(self):
        """???
        
        ???
        """
        return self._hemt_gate_max_voltage

    # Setter
    @hemt_gate_max_voltage.setter
    def hemt_gate_max_voltage(self,value):
        self._hemt_gate_max_voltage=value

    ## End hemt_gate_max_voltage property definition
    ###########################################################################            
    
    ###########################################################################
    ## Start fiftyk_Vg property definition

    # Getter
    @property
    def fiftyk_Vg(self):
        """50K LNA Gate Voltage
        
        Gets or sets the desired value for the 50K LNA Gate voltage.
        Units are Volts.
        """
        return self._fiftyk_Vg

    # Setter
    @fiftyk_Vg.setter
    def fiftyk_Vg(self,value):
        self._fiftyk_Vg=value

    ## End fiftyk_Vg property definition
    ###########################################################################        

    ###########################################################################
    ## Start fiftyk_dac_num property definition

    # Getter
    @property
    def fiftyk_dac_num(self):
        """???
        
        ???
        """
        return self._fiftyk_dac_num

    # Setter
    @fiftyk_dac_num.setter
    def fiftyk_dac_num(self,value):
        self._fiftyk_dac_num=value

    ## End fiftyk_dac_num property definition
    ###########################################################################

    ###########################################################################
    ## Start fiftyk_bit_to_V property definition

    # Getter
    @property
    def fiftyk_bit_to_V(self):
        """???
        
        ???
        """
        return self._fiftyk_bit_to_V

    # Setter
    @fiftyk_bit_to_V.setter
    def fiftyk_bit_to_V(self,value):
        self._fiftyk_bit_to_V=value

    ## End fiftyk_bit_to_V property definition
    ###########################################################################

    ###########################################################################
    ## Start fiftyk_amp_Vd_series_resistor property definition

    # Getter
    @property
    def fiftyk_amp_Vd_series_resistor(self):
        """???
        
        ???
        """
        return self._fiftyk_amp_Vd_series_resistor

    # Setter
    @fiftyk_amp_Vd_series_resistor.setter
    def fiftyk_amp_Vd_series_resistor(self,value):
        self._fiftyk_amp_Vd_series_resistor=value

    ## End fiftyk_amp_Vd_series_resistor property definition
    ###########################################################################

    ###########################################################################
    ## Start fiftyk_Id_offset property definition

    # Getter
    @property
    def fiftyk_Id_offset(self):
        """???
        
        ???
        """
        return self._fiftyk_Id_offset

    # Setter
    @fiftyk_Id_offset.setter
    def fiftyk_Id_offset(self,value):
        self._fiftyk_Id_offset=value

    ## End fiftyk_Id_offset property definition
    ###########################################################################

    
    
