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
"""Defines the mixin class :class:`SmurfConfigPropertiesMixin`.
"""

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
        """Demodulated SQUID phase to TES current conversion factor.

        Gets or sets the conversion factor between the demodulated
        SQUID phase for every SMuRF channel and the equivalent TES
        current.  Units are pA per Phi0, with Phi0 the magnetic flux
        quantum.

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.take_noise_psd`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_vs_bias`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_all_vs_noise_solo`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_vs_tone`,
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.analyze_slow_iv`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.bias_bump`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.identify_bias_groups`
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
        """4K HEMT gate voltage in volts.

        Gets or sets the desired value for the 4K HEMT gate voltage at
        the output of the cryostat card.  Units are Volts.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.set_amplifier_bias`
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
        """Bit to volts conversion for 4K HEMT gate DAC.

        Gets or sets the conversion from bits (the digital value the
        RTM DAC is set to) to Volts for the 4K amplifier gate
        (specified at the output of the cryostat card).  An important
        dependency is the voltage division on the cryostat card, which
        can be different from cryostat card to cryostat card.  Units
        are Volts/bit.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_hemt_gate_voltage`
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
        """4K HEMT drain current measurement resistor in Ohms.

        Gets or sets the resistance of the resistor that is inline
        with the 4K HEMT amplifier drain voltage source which is used
        to infer the 4K HEMT amplifier drain current.  This resistor
        is inline with but before the regulator that steps the main
        RF6.0V supply down to the lower 4K HEMT drain voltage, so the
        current flowing through this resistor includes both the drain
        current drawn by the 4K HEMT and any additional current drawn
        by the DC/DC regulator.  The default value of 200 Ohm is the
        standard value loaded onto cryostat card revision C02
        (PC-248-103-02-C02).  The resistor on that revision of the
        cryostat card is R44.  Units are Ohms.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_hemt_drain_current`
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
        """4K HEMT drain current offset in mA.

        Gets or sets the 4K HEMT drain current is measured before the
        regulator that steps the main RF6.0V supply down to the lower
        4K HEMT drain voltage using an inline resistor (see
        :func:`hemt_Vd_series_resistor`), so the total measured
        current through the series resistor includes both the drain
        current drawn by the 4K HEMT and any additional current drawn
        by the DC/DC regulator.  An accurate measurement of the 4K
        amplifier drain current requires subtracting the current drawn
        by that regulator from the measured total current.  This is
        the offset to subtract off the measured value.  Units are
        milliamperes.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_hemt_drain_current`
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
        """4K HEMT gate voltage minimum software limit

        Gets or sets the minimum voltage the 4K HEMT gate voltage can
        be set to using the
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`
        function unless this software limit is overriden with the
        boolean `override` argument of
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`.
        Units are Volts.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`
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
        """4K HEMT gate voltage maximum software limit

        Gets or sets the maximum voltage the 4K HEMT gate voltage can
        be set to using the
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`
        function unless this software limit is overriden with the
        boolean `override` argument of
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`.
        Units are Volts.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`
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

        Gets or sets the desired value for the 50K LNA Gate voltage at
        the output of the cryostat card.  Units are Volts.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.set_amplifier_bias`
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
        """The RTM DAC number which is wired to the 50K LNA gate.

        Gets or sets the DAC number of the DAC on the RTM that is
        wired to the 50K LNA gate.  Must be an integer between 1 and
        32, at least for RTM main board revision C01
        (PC-379-396-32-C01).  The DAC number corresponds to the number
        on the RTM schematic (e.g. see the nets named DAC1,...,DAC32.
        The connection between an RTM DAC and the 50K LNA gate is made
        on the cryostat card.  The default RTM DAC that's wired to the
        50K LNA gate for cryostat card revision C02
        (PC-248-103-02-C02) is DAC32 (if JMP4 on the cryostat card is
        populated correctly).

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_50k_amp_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_50k_amp_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_50k_amp_enable`
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
        """Bit to volts conversion for 50K LNA gate DAC.

        Gets or set the conversion from bits (the digital value the
        RTM DAC is set to) to Volts for the 50K LNA gate (specified at
        the output of the cryostat card).  An important dependency is
        the voltage division on the cryostat card, which can be
        different from cryostat card to cryostat card.  Units are
        Volts/bit.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_50k_amp_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_50k_amp_gate_voltage`
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
        """50K LNA drain current measurement resistor in Ohms.

        Gets or sets the resistance of the resistor that is inline
        with the 50K LNA drain voltage source which is used to infer
        the 50K RF amplifier drain current.  This resistor is inline
        with but before the regulator that steps the main RF6.0V
        supply down to the lower 50K LNA drain voltage, so the current
        flowing through this resistor includes both the drain current
        drawn by the 50K RF amplifier and any additional current drawn
        by the DC/DC regulator.  The default value of 10 Ohm is the
        standard value loaded onto cryostat card revision C02
        (PC-248-103-02-C02).  The resistor on that revision of the
        cryostat card is R54.  Units are Ohms.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_50k_amp_drain_current`
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
        """50K amplifier drain current offset in mA.

        Gets or sets the 50K RF amplifier drain current is measured
        before the regulator that steps the main RF6.0V supply down to
        the lower 50K RF amplifier drain voltage using an inline
        resistor (see :func:`fiftyk_amp_Vd_series_resistor`), so the
        total measured current through the series resistor includes
        both the drain current drawn by the 50K RF amplifier and any
        additional current drawn by the DC/DC regulator.  An accurate
        measurement of the 50K amplifier drain current requires
        subtracting the current drawn by that regulator from the
        measured total current.  This is the offset to subtract off
        the measured value.  Units are milliamperes.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_50k_amp_drain_current`
        """
        return self._fiftyk_Id_offset

    # Setter
    @fiftyk_Id_offset.setter
    def fiftyk_Id_offset(self,value):
        self._fiftyk_Id_offset=value

    ## End fiftyk_Id_offset property definition
    ###########################################################################
