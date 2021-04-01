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
"""Defines the mixin class :class:`SmurfConfigPropertiesMixin`."""
import numpy as np

class SmurfConfigPropertiesMixin:
    """Mixin for loading pysmurf configuration parameters.

    Defines properties used by pysmurf whose values are specified in
    the pysmurf configuration file.  More details on python properties
    can be found in the documentation for python "Built-in Functions"
    [#pyprop]_.

    The :class:`SmurfConfigPropertiesMixin` class has only one
    method :meth:`copy_config_to_properties` which can populate the
    properties from a provided
    :class:`~pysmurf.client.base.smurf_config.SmurfConfig` class
    instance.  :meth:`copy_config_to_properties` can be called by the
    user at any time, but it is called once to populate all of the
    :class:`SmurfConfigPropertiesMixin` attributes by a call to the
    :func:`~pysmurf.client.base.smurf_control.SmurfControl.initialize`
    method in the
    :class:`~pysmurf.client.base.smurf_control.SmurfControl`
    constructor (unless the
    :class:`~pysmurf.client.base.smurf_control.SmurfControl` instance
    is not instantiated with a configuration file).

    .. warning::
       If a user changes the value of a property but then writes the
       internal pysmurf configuration to disk with the
       :func:`~pysmurf.client.base.smurf_control.SmurfControl.write_output()`
       function, the value written to the file will not reflect the
       user's change.

    Examples
    --------
    Taking the `pA_per_phi0` property as an example, if `S` is a
    :class:`~pysmurf.client.base.smurf_control.SmurfControl` class
    instance, once the property has been set with a call to the
    :func:`copy_config_to_properties` routine, its value can be
    retrieved like this:

    >>> S.pA_per_phi0
    9000000.0

    and it can be set like this:

    >>> S.pA_per_phi0 = 9000000.0

    Each property has a corresponding class instance internal
    attribute that the property get/set operates on and which is used
    internally in all pysmurf functions that rely on the property.
    The internal attributes corresponding to each property have the
    same name as the property but with an underscore preceding the
    property name.  E.g. the internal attribute corresponding to the
    `pA_per_phi0` property is called `_pA_per_phi0`.  Users should
    never set or get the internal attribute directly.

    See Also
    --------
    :func:`~pysmurf.client.base.smurf_control.SmurfControl.initialize`

    References
    ----------
    .. [#pyprop] https://docs.python.org/3/library/functions.html#property

    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, *args, **kwargs):
        """SmurfConfigPropertiesMixin constructor."""
        # EPICS
        self._epics_root = None

        # Directories
        self._smurf_cmd_dir = None
        self._tune_dir = None
        self._status_dir = None
        self._default_data_dir = None

        # Constants
        self._pA_per_phi0 = None

        # Timing
        self._timing_reference = None

        # Amplifiers
        ## 4K HEMT
        self._hemt_Vg = None
        self._hemt_bit_to_V = None
        self._hemt_Vd_series_resistor = None
        self._hemt_Id_offset = None
        self._hemt_gate_min_voltage = None
        self._hemt_gate_max_voltage = None

        ## 50K LNA
        self._fiftyk_Vg = None
        self._fiftyk_dac_num = None
        self._fiftyk_bit_to_V = None
        self._fiftyk_amp_Vd_series_resistor = None
        self._fiftyk_Id_offset = None

        ## Tuning parameters
        self._default_tune = None
        self._gradient_descent_gain = None
        self._gradient_descent_averages = None
        self._gradient_descent_converge_hz = None
        self._gradient_descent_step_hz = None
        self._gradient_descent_momentum = None
        self._gradient_descent_beta = None
        self._feedback_start_frac = None
        self._feedback_end_frac = None
        self._eta_scan_del_f = None
        self._eta_scan_averages = None
        self._delta_freq = None

        # Reading/writing data
        self._fs = None

        # In fridge
        self._R_sh = None

        # Carrier
        self._ultrascale_temperature_limit_degC = None
        self._data_out_mux = None
        self._dsp_enable = None

        # AMC
        self._amplitude_scale = None
        self._bands = None
        self._attenuator = None
        self._iq_swap_in = None
        self._iq_swap_out = None
        self._ref_phase_delay = None
        self._ref_phase_delay_fine = None
        self._band_delay_us = None
        self._att_uc = None
        self._att_dc = None
        self._trigger_reset_delay = None

        # RTM
        self._num_flux_ramp_counter_bits = None
        self._fraction_full_scale = None
        self._reset_rate_khz = None

        # Cryocard
        self._bias_line_resistance = None
        self._high_low_current_ratio = None
        self._high_current_mode_bool = None
        self._pic_to_bias_group = None

        # Tracking algo
        self._lms_delay = None
        self._lms_gain = None
        self._lms_freq_hz = None
        self._feedback_enable = None
        self._feedback_gain = None
        self._feedback_limit_khz = None
        self._feedback_polarity = None

        # Mappings
        self._all_groups = None
        self._n_bias_groups = None
        self._bias_group_to_pair = None
        self._bad_mask = None

    def copy_config_to_properties(self, config):
        """Copy values from SmurfConfig instance to properties.

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
        ## EPICS
        self.epics_root = config.get('epics_root')

        ## Directories
        self.smurf_cmd_dir = config.get('smurf_cmd_dir')
        self.tune_dir = config.get('tune_dir')
        self.status_dir = config.get('status_dir')
        self.default_data_dir = config.get('default_data_dir')

        ## Useful constants
        constant_cfg = config.get('constant')
        self.pA_per_phi0 = constant_cfg.get('pA_per_phi0')

        ## Timing
        timing_cfg = config.get('timing')
        self.timing_reference = timing_cfg['timing_reference']

        ## Cold amplifier biases
        amp_cfg = config.get('amplifier')

        # 4K HEMT
        self.hemt_Vg = amp_cfg['hemt_Vg']
        self.hemt_bit_to_V = amp_cfg['bit_to_V_hemt']
        self.hemt_Vd_series_resistor = amp_cfg['hemt_Vd_series_resistor']
        self.hemt_Id_offset = amp_cfg['hemt_Id_offset']
        self.hemt_gate_min_voltage = amp_cfg['hemt_gate_min_voltage']
        self.hemt_gate_max_voltage = amp_cfg['hemt_gate_max_voltage']

        # 50K HEMT
        self.fiftyk_Vg = amp_cfg['LNA_Vg']
        self.fiftyk_dac_num = amp_cfg['dac_num_50k']
        self.fiftyk_bit_to_V = amp_cfg['bit_to_V_50k']
        self.fiftyk_amp_Vd_series_resistor = amp_cfg['50K_amp_Vd_series_resistor']
        self.fiftyk_Id_offset = amp_cfg['50k_Id_offset']

        ## Tune parameters
        tune_band_cfg = config.get('tune_band')
        self.default_tune = tune_band_cfg['default_tune']
        self.gradient_descent_gain = {
            int(band):v for (band,v) in
            tune_band_cfg['gradient_descent_gain'].items()}
        self.gradient_descent_averages = {
            int(band):v for (band,v) in
            tune_band_cfg['gradient_descent_averages'].items()}
        self.gradient_descent_converge_hz = {
            int(band):v for (band,v) in
            tune_band_cfg['gradient_descent_converge_hz'].items()}
        self.gradient_descent_step_hz = {
            int(band):v for (band,v) in
            tune_band_cfg['gradient_descent_step_hz'].items()}
        self.gradient_descent_momentum = {
            int(band):v for (band,v) in
            tune_band_cfg['gradient_descent_momentum'].items()}
        self.gradient_descent_beta = {
            int(band):v for (band,v) in
            tune_band_cfg['gradient_descent_beta'].items()}
        self.feedback_start_frac = {
            int(band):v for (band,v) in
            tune_band_cfg['feedback_start_frac'].items()}
        self.feedback_end_frac = {
            int(band):v for (band,v) in
            tune_band_cfg['feedback_end_frac'].items()}
        self.eta_scan_del_f = {
            int(band):v for (band,v) in
            tune_band_cfg['eta_scan_del_f'].items()}
        self.eta_scan_averages = {
            int(band):v for (band,v) in
            tune_band_cfg['eta_scan_averages'].items()}
        self.delta_freq = {
            int(band):v for (band,v) in
            tune_band_cfg['delta_freq'].items()}
        # Tracking algo
        self.lms_freq_hz = {
            int(band):v for (band,v) in
            tune_band_cfg['lms_freq'].items()}

        ## Reading/writing data
        self.fs = config.get('fs')

        ## In fridge
        self.R_sh = config.get('R_sh')

        ## Which bands are have their configurations specified in the
        ## pysmurf configuration file?
        smurf_init_config = config.get('init')
        bands = smurf_init_config['bands']

        ## Carrier
        self.dsp_enable = smurf_init_config['dspEnable']
        self.ultrascale_temperature_limit_degC = config.get('ultrascale_temperature_limit_degC')
        self.data_out_mux = {
            band:smurf_init_config[f'band_{band}']['data_out_mux']
            for band in bands}

        ## AMC
        # Which bands are present in the pysmurf configuration file?
        self.bands = bands
        self.amplitude_scale = {
            band:smurf_init_config[f'band_{band}']['amplitude_scale']
            for band in bands}
        self.iq_swap_in = {
            band:smurf_init_config[f'band_{band}']['iq_swap_in']
            for band in bands}
        self.iq_swap_out = {
            band:smurf_init_config[f'band_{band}']['iq_swap_out']
            for band in bands}
        self.ref_phase_delay = {
            band:smurf_init_config[f'band_{band}']['refPhaseDelay']
            for band in bands}
        self.ref_phase_delay_fine = {
            band:smurf_init_config[f'band_{band}']['refPhaseDelayFine']
            for band in bands}
        self.band_delay_us = {
            band:smurf_init_config[f'band_{band}']['bandDelayUs']
            for band in bands}
        self.att_uc = {
            band:smurf_init_config[f'band_{band}']['att_uc']
            for band in bands}
        self.att_dc = {
            band:smurf_init_config[f'band_{band}']['att_dc']
            for band in bands}
        self.trigger_reset_delay= {
            band:smurf_init_config[f'band_{band}']['trigRstDly']
            for band in bands}

        # Mapping from attenuator numbers to bands
        att_cfg = config.get('attenuator')
        att_cfg_keys = att_cfg.keys()
        attenuator = {}
        attenuator['band'] = np.zeros(len(att_cfg_keys),dtype=int)
        attenuator['att'] = np.zeros(len(att_cfg_keys),dtype=int)
        for i, k in enumerate(att_cfg_keys):
            attenuator['band'][i] = att_cfg[k]
            attenuator['att'][i] = k[-1]
        self.attenuator = attenuator

        ## RTM
        flux_ramp_cfg = config.get('flux_ramp')
        self.num_flux_ramp_counter_bits = flux_ramp_cfg['num_flux_ramp_counter_bits']
        self.reset_rate_khz = tune_band_cfg.get('reset_rate_khz')
        self.fraction_full_scale = tune_band_cfg.get('fraction_full_scale')

        ## Cryocard
        self.bias_line_resistance = config.get('bias_line_resistance')
        self.high_low_current_ratio = config.get('high_low_current_ratio')
        self.high_current_mode_bool = config.get('high_current_mode_bool')
        # Mapping from peripheral interface controller (PIC) to bias group
        pic_cfg = config.get('pic_to_bias_group')
        pic_cfg_keys = pic_cfg.keys()
        pic_to_bias_group = np.zeros((len(pic_cfg_keys), 2), dtype=int)
        for i, k in enumerate(pic_cfg_keys):
            val = pic_cfg[k]
            pic_to_bias_group[i] = [k, val]
        self.pic_to_bias_group = pic_to_bias_group

        ## Tracking algo
        # lmsGain ; this one's a little odd ; it's defined in each of
        # the band_# configuration file blocks, while the other main
        # tracking algorithm parameter, lms_freq_hz, is defined in the
        # tune_band configuration file block...
        self.lms_gain = {
            band:smurf_init_config[f'band_{band}']['lmsGain']
            for band in bands}
        self.lms_delay = {
            band:smurf_init_config[f'band_{band}']['lmsDelay']
            for band in bands}
        self.feedback_enable = {
            band:smurf_init_config[f'band_{band}']['feedbackEnable']
            for band in bands}
        self.feedback_gain = {
            band:smurf_init_config[f'band_{band}']['feedbackGain']
            for band in bands}
        self.feedback_limit_khz = {
            band:smurf_init_config[f'band_{band}']['feedbackLimitkHz']
            for band in bands}
        self.feedback_polarity = {
            band:smurf_init_config[f'band_{band}']['feedbackPolarity']
            for band in bands}

        ## Mappings
        # Bias groups available
        self.all_groups = config.get('all_bias_groups')

        # Number of bias groups and bias group to RTM DAC pair
        # mapping
        bias_group_cfg = config.get('bias_group_to_pair')
        bias_group_keys = bias_group_cfg.keys()

        # Number of bias groups
        self.n_bias_groups = len(bias_group_cfg)

        # Bias group to RTM DAC pair mapping
        bias_group_to_pair = np.zeros((len(bias_group_keys), 3), dtype=int)
        for i, k in enumerate(bias_group_keys):
            val = bias_group_cfg[k]
            bias_group_to_pair[i] = np.append([k], val)
        self.bias_group_to_pair = bias_group_to_pair

        # Bad resonator mask
        bad_mask_config = config.get('bad_mask')
        bad_mask_keys = bad_mask_config.keys()
        bad_mask = np.zeros((len(bad_mask_keys), 2))
        for i, k in enumerate(bad_mask_keys):
            bad_mask[i] = bad_mask_config[k]
        self.bad_mask = bad_mask

    ###########################################################################
    ## Start pA_per_phi0 property definition

    # Getter
    @property
    def pA_per_phi0(self):
        """Demodulated SQUID phase to TES current conversion factor.

        Gets or sets the conversion factor between the demodulated
        SQUID phase for every SMuRF channel and the equivalent TES
        current.  Units are picoamperes per Phi0, with Phi0 the
        magnetic flux quantum.

        Specified in the pysmurf configuration file as
        `constant:pA_per_phi0`.

        Returns
        -------
        float
           Conversion factor between demodulated SQUID phase and
           equivalent TES current in picoamperes per Phi0.

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.analyze_iv`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_all_vs_noise_solo`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_vs_bias`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_vs_tone`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.take_noise_psd`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.bias_bump`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.identify_bias_groups`

        """
        return self._pA_per_phi0

    # Setter
    @pA_per_phi0.setter
    def pA_per_phi0(self, value):
        self._pA_per_phi0 = value

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

        Specified in the pysmurf configuration file as
        `amplifier:hemt_Vg`.

        Returns
        -------
        float
           4K HEMT gate voltage in Volts.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.set_amplifier_bias`

        """
        return self._hemt_Vg

    # Setter
    @hemt_Vg.setter
    def hemt_Vg(self, value):
        self._hemt_Vg = value

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

        Specified in the pysmurf configuration file as
        `amplifier:bit_to_V_hemt`.

        Returns
        -------
        float
           Conversion factor from bits to volts for the 4K amplifier
           gate in Volts/bit.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_hemt_gate_voltage`

        """
        return self._hemt_bit_to_V

    # Setter
    @hemt_bit_to_V.setter
    def hemt_bit_to_V(self, value):
        self._hemt_bit_to_V = value

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

        Specified in the pysmurf configuration file as
        `amplifier:hemt_Vd_series_resistor`.

        Returns
        -------
        float
           Resistance in Ohms of the inline resistor used to measure
           the 4K HEMT amplifier drain current.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_hemt_drain_current`

        """
        return self._hemt_Vd_series_resistor

    # Setter
    @hemt_Vd_series_resistor.setter
    def hemt_Vd_series_resistor(self, value):
        self._hemt_Vd_series_resistor = value

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

        Specified in the pysmurf configuration file as
        `amplifier:hemt_Id_offset`.

        Returns
        -------
        float
           4K HEMT drain current offset in milliamperes.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_hemt_drain_current`

        """
        return self._hemt_Id_offset

    # Setter
    @hemt_Id_offset.setter
    def hemt_Id_offset(self, value):
        self._hemt_Id_offset = value

    ## End hemt_Id_offset property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_gate_min_voltage property definition

    # Getter
    @property
    def hemt_gate_min_voltage(self):
        """4K HEMT gate voltage minimum software limit.

        Gets or sets the minimum voltage the 4K HEMT gate voltage can
        be set to using the
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`
        function unless this software limit is overriden with the
        boolean `override` argument of
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`.
        Units are Volts.

        Specified in the pysmurf configuration file as
        `amplifier:hemt_gate_min_voltage`.

        Returns
        -------
        float
           Software limit on minimum 4K HEMT gate voltage user can
           apply, in Volts.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`

        """
        return self._hemt_gate_min_voltage

    # Setter
    @hemt_gate_min_voltage.setter
    def hemt_gate_min_voltage(self, value):
        self._hemt_gate_min_voltage = value

    ## End hemt_gate_min_voltage property definition
    ###########################################################################

    ###########################################################################
    ## Start hemt_gate_max_voltage property definition

    # Getter
    @property
    def hemt_gate_max_voltage(self):
        """4K HEMT gate voltage maximum software limit.

        Gets or sets the maximum voltage the 4K HEMT gate voltage can
        be set to using the
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`
        function unless this software limit is overriden with the
        boolean `override` argument of
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`.
        Units are Volts.

        Specified in the pysmurf configuration file as
        `amplifier:hemt_gate_max_voltage`.

        Returns
        -------
        float
           Software limit on maximum 4K HEMT gate voltage user can
           apply, in Volts.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_hemt_gate_voltage`

        """
        return self._hemt_gate_max_voltage

    # Setter
    @hemt_gate_max_voltage.setter
    def hemt_gate_max_voltage(self, value):
        self._hemt_gate_max_voltage = value

    ## End hemt_gate_max_voltage property definition
    ###########################################################################

    ###########################################################################
    ## Start fiftyk_Vg property definition

    # Getter
    @property
    def fiftyk_Vg(self):
        """50K LNA Gate Voltage.

        Gets or sets the desired value for the 50K LNA Gate voltage at
        the output of the cryostat card.  Units are Volts.

        Specified in the pysmurf configuration file as
        `amplifier:LNA_Vg`.

        Returns
        -------
        float
           50K LNA gate voltage in Volts.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.set_amplifier_bias`

        """
        return self._fiftyk_Vg

    # Setter
    @fiftyk_Vg.setter
    def fiftyk_Vg(self, value):
        self._fiftyk_Vg = value

    ## End fiftyk_Vg property definition
    ###########################################################################

    ###########################################################################
    ## Start fiftyk_dac_num property definition

    # Getter
    @property
    def fiftyk_dac_num(self):
        """RTM DAC number wired to the 50K LNA gate.

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

        Specified in the pysmurf configuration file as
        `amplifier:dac_num_50k`.

        Returns
        -------
        int
           DAC number of RTM DAC wired to the 50K LNA gate.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_50k_amp_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_50k_amp_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_50k_amp_enable`

        """
        return self._fiftyk_dac_num

    # Setter
    @fiftyk_dac_num.setter
    def fiftyk_dac_num(self, value):
        self._fiftyk_dac_num = value

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

        Specified in the pysmurf configuration file as
        `amplifier:bit_to_V_50k`.

        Returns
        -------
        float
           Conversion factor from bits to volts for the 50K LNA gate
           in Volts/bit.

        See Also
        --------
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_50k_amp_gate_voltage`,
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_50k_amp_gate_voltage`

        """
        return self._fiftyk_bit_to_V

    # Setter
    @fiftyk_bit_to_V.setter
    def fiftyk_bit_to_V(self, value):
        self._fiftyk_bit_to_V = value

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

        Specified in the pysmurf configuration file as
        `amplifier:50K_amp_Vd_series_resistor`.

        Returns
        -------
        float
           Resistance in Ohms of the inline resistor used to measure
           the 50K LNA drain current.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_50k_amp_drain_current`

        """
        return self._fiftyk_amp_Vd_series_resistor

    # Setter
    @fiftyk_amp_Vd_series_resistor.setter
    def fiftyk_amp_Vd_series_resistor(self, value):
        self._fiftyk_amp_Vd_series_resistor = value

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

        Specified in the pysmurf configuration file as
        `amplifier:50k_Id_offset`.

        Returns
        -------
        float
           50K LNA drain current offset in milliamperes.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.get_50k_amp_drain_current`

        """
        return self._fiftyk_Id_offset

    # Setter
    @fiftyk_Id_offset.setter
    def fiftyk_Id_offset(self, value):
        self._fiftyk_Id_offset = value

    ## End fiftyk_Id_offset property definition
    ###########################################################################

    ###########################################################################
    ## Start attenuator property definition

    # Getter
    @property
    def attenuator(self):
        """Map between the 500MHz bands and their RF attenuators.

        Gets or sets the assumed hardware correspondence between the
        500 MHz attenuators and their UC and DC RF attenuator numbers.
        Only the mapping for bands 0-3 should be specified, since the
        mapping is assumed to be identical for the two carrier AMC
        bays (e.g. the mapping is identical from band to attenuator
        number for band % 4).  Unitless.

        Specified in the pysmurf configuration file as
        `attenuator`.

        Returns
        -------
        dict
           A dictionary with two keys, 'band' and 'att', which map to
           two :py:class:`numpy.ndarray` of integers corresponding to
           the UC and DC attenuator numbers and their corresponding
           500 MHz band numbers, in the same order.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.att_to_band`
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.band_to_att`

        Examples
        --------
        The `attenuator` mapping is specified as a dictionary in the
        pysmurf configuration file like this::

           "attenuator" : {
              "att1" : 0,
              "att2" : 1,
              "att3" : 2,
              "att4" : 3
           },

        where the keys (e.g. "att1") specify which attenuator maps to
        which 500 MHz band (the integer value).  The mapping is stored
        in a very different format inside pysmurf - loading the above
        attenuator mapping results in the following value for the
        `attenuator` property (here `S` is a
        :class:`~pysmurf.client.base.smurf_control.SmurfControl` class
        instance):

        >>> print(S.attenuator)
        {'band': array([0, 1, 2, 3]), 'att': array([1, 2, 3, 4])}

        """
        return self._attenuator

    # Setter
    @attenuator.setter
    def attenuator(self, value):
        self._attenuator = value

    ## End attenuator property definition
    ###########################################################################

    ###########################################################################
    ## Start num_flux_ramp_counter_bits property definition

    # Getter
    @property
    def num_flux_ramp_counter_bits(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `flux_ramp:num_flux_ramp_counter_bits`.

        See Also
        --------
        ?

        """
        return self._num_flux_ramp_counter_bits

    # Setter
    @num_flux_ramp_counter_bits.setter
    def num_flux_ramp_counter_bits(self, value):
        self._num_flux_ramp_counter_bits = value

    ## End num_flux_ramp_counter_bits property definition
    ###########################################################################

    ###########################################################################
    ## Start all_groups property definition

    # Getter
    @property
    def all_groups(self):
        """Which TES bias groups are in use.

        Gets or sets the list of TES bias groups in use.  Each element
        of this list must be an integer corresponding to a TES bias
        group and in [0,16).  :func:`bias_group_to_pair` encodes the
        mapping from TES bias group number to bipolar RTM DAC pair.

        Specified in the pysmurf configuration file as
        `all_bias_groups`.

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.run_iv`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.overbias_tes_all`
        """
        return self._all_groups

    # Setter
    @all_groups.setter
    def all_groups(self, value):
        self._all_groups = value

    ## End all_groups property definition
    ###########################################################################

    ###########################################################################
    ## Start n_bias_groups property definition

    # Getter
    @property
    def n_bias_groups(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `?`.

        See Also
        --------
        ?

        """
        return self._n_bias_groups

    # Setter
    @n_bias_groups.setter
    def n_bias_groups(self, value):
        self._n_bias_groups = value

    ## End n_bias_groups property definition
    ###########################################################################

    ###########################################################################
    ## Start bias_group_to_pair property definition

    # Getter
    @property
    def bias_group_to_pair(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `bias_group_to_pair`.

        See Also
        --------
        ?

        """
        return self._bias_group_to_pair

    # Setter
    @bias_group_to_pair.setter
    def bias_group_to_pair(self, value):
        self._bias_group_to_pair = value

    ## End bias_group_to_pair property definition
    ###########################################################################

    ###########################################################################
    ## Start pic_to_bias_group property definition

    # Getter
    @property
    def pic_to_bias_group(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `pic_to_bias_group`.

        See Also
        --------
        ?

        """
        return self._pic_to_bias_group

    # Setter
    @pic_to_bias_group.setter
    def pic_to_bias_group(self, value):
        self._pic_to_bias_group = value

    ## End pic_to_bias_group property definition
    ###########################################################################

    ###########################################################################
    ## Start bias_line_resistance property definition

    # Getter
    @property
    def bias_line_resistance(self):
        """Total low current mode TES bias line resistance.

        Gets or sets the total low current mode TES bias line
        resistance.  Assumes the same resistance for every TES bias
        line.  Includes the inline resistance on the cryostat card
        (with the cryostat card relays set to the low current
        operation mode position) and cryocable impedance (including
        any cold resistors).  Technically includes the resistance
        added by the connected TES+shunt resistor chain, but that is
        typically negligible compared to the cryostat card and
        cryocable impedances.  Units are Ohms.

        Specified in the pysmurf configuration file as
        `bias_line_resistance`.

        Returns
        -------
        float
           Total low current mode TES bias line resistance in Ohms.

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.analyze_iv`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.NET_CMB`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.bias_bump`
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.identify_bias_groups`

        """
        return self._bias_line_resistance

    # Setter
    @bias_line_resistance.setter
    def bias_line_resistance(self, value):
        self._bias_line_resistance = value

    ## End bias_line_resistance property definition
    ###########################################################################

    ###########################################################################
    ## Start R_sh property definition

    # Getter
    @property
    def R_sh(self):
        """Resistance of the TES shunt resistors.

        Gets or sets the resistance of the TES shunt resistors.
        Assumes the same shunt resistance for every channel.  Units
        are Ohms.

        Specified in the pysmurf configuration file as `R_sh`.

        Returns
        -------
        float
           Resistance of the TES shunt resistors in Ohms.

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.analyze_iv`,
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.partial_load_curve_all`,
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.run_iv`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.NET_CMB`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.analyze_noise_vs_bias`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.bias_bump`

        """
        return self._R_sh

    # Setter
    @R_sh.setter
    def R_sh(self, value):
        self._R_sh = value

    ## End R_sh property definition
    ###########################################################################

    ###########################################################################
    ## Start high_low_current_ratio property definition

    # Getter
    @property
    def high_low_current_ratio(self):
        """Current ratio between low- to high-current modes.

        Gets or sets the ratio of currents sourced by the cryostat
        card for the same applied TES bias voltage (from the RTM, at
        the input to the cryostat card) in high- versus low-current
        cryostat card TES bias relay modes.  In typical applications,
        this ratio is >1, since otherwise the currents sourced in
        high-current mode wouldn't be higher than the currents sourced
        in low current mode!  Assumes the same ratio for every TES
        bias line.  In typical applications, this ratio is well
        approximated by the ratio of the resistances on the cryostat
        card alone (computed as the total resistance for the relay in
        the low-current mode position divided by the total resistance
        for the relay in the high-current mode position).  Unitless.

        Specified in the pysmurf configuration file as
        `high_low_current_ratio`.

        Returns
        -------
        float
           The ratio of currents sourced by the cryostat card for the
           same applied TES bias voltage (from the RTM, at the input
           to the cryostat card) in high- versus low-current modes
           (unitless).

        See Also
        --------
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.analyze_iv`,
        :func:`~pysmurf.client.debug.smurf_iv.SmurfIVMixin.partial_load_curve_all`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.NET_CMB`,
        :func:`~pysmurf.client.debug.smurf_noise.SmurfNoiseMixin.get_iv_data`,
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.bias_bump`
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.identify_bias_groups`

        """
        return self._high_low_current_ratio

    # Setter
    @high_low_current_ratio.setter
    def high_low_current_ratio(self, value):
        self._high_low_current_ratio = value

    ## End high_low_current_ratio property definition
    ###########################################################################

    ###########################################################################
    ## Start high_current_mode_bool property definition

    # Getter
    @property
    def high_current_mode_bool(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `high_current_mode_bool`.

        See Also
        --------
        ?

        """
        return self._high_current_mode_bool

    # Setter
    @high_current_mode_bool.setter
    def high_current_mode_bool(self, value):
        self._high_current_mode_bool = value

    ## End high_current_mode_bool property definition
    ###########################################################################

    ###########################################################################
    ## Start fs property definition

    # Getter
    @property
    def fs(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `fs`.

        See Also
        --------
        ?

        """
        return self._fs

    # Setter
    @fs.setter
    def fs(self, value):
        self._fs = value

    ## End fs property definition
    ###########################################################################

    ###########################################################################
    ## Start bad_mask property definition

    # Getter
    @property
    def bad_mask(self):
        """Resonator frequency ranges to ignore when relocking.

        Gets or sets the list of RF frequency intervals in which to
        ignore resonator candidates in calls to
        :func:`~pysmurf.client.tune.smurf_tune.SmurfTuneMixin.relock`
        Frequencies are in units of MHz.
        :func:`~pysmurf.client.tune.smurf_tune.SmurfTuneMixin.relock`
        is called at the end of many tuning functions including
        :func:`~pysmurf.client.tune.smurf_tune.SmurfTuneMixin.setup_notches`
        and
        :func:`~pysmurf.client.tune.smurf_tune.SmurfTuneMixin.track_and_check`.

        Specified in the pysmurf configuration file as
        `bad_mask`.

        See Also
        --------
        :func:`~pysmurf.client.tune.smurf_tune.SmurfTuneMixin.relock`

        Examples
        --------
        The `bad_mask` property is specified as a dictionary in the
        pysmurf configuration file like this (where the ellipsis
        denotes additional possible `bad_mask` entries)::

           "bad_mask" : {
              "0" : [5000, 5100],
              "1" : [5500, 5750],
              ...
           },

        where the keys are ignored (but must be unique) and the values
        specify intervals of RF frequencies in MHz over which to
        discard resonator candidates when relocking.  The frequency
        ranges are stored in pysmurf in the `bad_mask` property as a
        :py:class:`numpy.ndarray` of :py:class:`numpy.ndarray`.

        For example, the above would result in the following value for
        the `bad_mask` property (here `S` is a
        :class:`~pysmurf.client.base.smurf_control.SmurfControl` class
        instance and the ellipsis denotes additional possible
        entries):

        >>> import pprint
        >>> pprint.pprint(S.bad_mask)
        array([[5000., 5100.], [5500., 5750.], ...])

        """
        return self._bad_mask

    # Setter
    @bad_mask.setter
    def bad_mask(self, value):
        self._bad_mask = value

    ## End bad_mask property definition
    ###########################################################################

    ###########################################################################
    ## Start fraction_full_scale property definition

    # Getter
    @property
    def fraction_full_scale(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:fraction_full_scale`.

        See Also
        --------
        ?

        """
        return self._fraction_full_scale

    # Setter
    @fraction_full_scale.setter
    def fraction_full_scale(self, value):
        self._fraction_full_scale = value

    ## End fraction_full_scale property definition
    ###########################################################################

    ###########################################################################
    ## Start reset_rate_khz property definition

    # Getter
    @property
    def reset_rate_khz(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:reset_rate_khz`.

        See Also
        --------
        ?

        """
        return self._reset_rate_khz

    # Setter
    @reset_rate_khz.setter
    def reset_rate_khz(self, value):
        self._reset_rate_khz = value

    ## End reset_rate_khz property definition
    ###########################################################################

    ###########################################################################
    ## Start lms_gain property definition

    # Getter
    @property
    def lms_gain(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `?`.

        See Also
        --------
        ?

        """
        return self._lms_gain

    # Setter
    @lms_gain.setter
    def lms_gain(self, value):
        self._lms_gain = value

    ## End lms_gain property definition
    ###########################################################################

    ###########################################################################
    ## Start lms_freq_hz property definition

    # Getter
    @property
    def lms_freq_hz(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `?`.

        See Also
        --------
        ?

        """
        return self._lms_freq_hz

    # Setter
    @lms_freq_hz.setter
    def lms_freq_hz(self, value):
        self._lms_freq_hz = value

    ## End lms_freq_hz property definition
    ###########################################################################

    ###########################################################################
    ## Start gradient_descent_gain property definition

    # Getter
    @property
    def gradient_descent_gain(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:gradient_descent_gain`.

        See Also
        --------
        ?

        """
        return self._gradient_descent_gain

    # Setter
    @gradient_descent_gain.setter
    def gradient_descent_gain(self, value):
        self._gradient_descent_gain = value

    ## End gradient_descent_gain property definition
    ###########################################################################

    ###########################################################################
    ## Start gradient_descent_averages property definition

    # Getter
    @property
    def gradient_descent_averages(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:gradient_descent_averages`.

        See Also
        --------
        ?

        """
        return self._gradient_descent_averages

    # Setter
    @gradient_descent_averages.setter
    def gradient_descent_averages(self, value):
        self._gradient_descent_averages = value

    ## End gradient_descent_averages property definition
    ###########################################################################

    ###########################################################################
    ## Start gradient_descent_converge_hz property definition

    # Getter
    @property
    def gradient_descent_converge_hz(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:gradient_descent_converge_hz`.

        See Also
        --------
        ?

        """
        return self._gradient_descent_converge_hz

    # Setter
    @gradient_descent_converge_hz.setter
    def gradient_descent_converge_hz(self, value):
        self._gradient_descent_converge_hz = value

    ## End gradient_descent_converge_hz property definition
    ###########################################################################

    ###########################################################################
    ## Start gradient_descent_step_hz property definition

    # Getter
    @property
    def gradient_descent_step_hz(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:gradient_descent_step_hz`.

        See Also
        --------
        ?

        """
        return self._gradient_descent_step_hz

    # Setter
    @gradient_descent_step_hz.setter
    def gradient_descent_step_hz(self, value):
        self._gradient_descent_step_hz = value

    ## End gradient_descent_step_hz property definition
    ###########################################################################

    ###########################################################################
    ## Start gradient_descent_momentum property definition

    # Getter
    @property
    def gradient_descent_momentum(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:gradient_descent_momentum`.

        See Also
        --------
        ?

        """
        return self._gradient_descent_momentum

    # Setter
    @gradient_descent_momentum.setter
    def gradient_descent_momentum(self, value):
        self._gradient_descent_momentum = value

    ## End gradient_descent_momentum property definition
    ###########################################################################

    ###########################################################################
    ## Start gradient_descent_beta property definition

    # Getter
    @property
    def gradient_descent_beta(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:gradient_descent_beta`.

        See Also
        --------
        ?

        """
        return self._gradient_descent_beta

    # Setter
    @gradient_descent_beta.setter
    def gradient_descent_beta(self, value):
        self._gradient_descent_beta = value

    ## End gradient_descent_beta property definition
    ###########################################################################

    ###########################################################################
    ## Start eta_scan_del_f property definition

    # Getter
    @property
    def eta_scan_del_f(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:eta_scan_del_f`.

        See Also
        --------
        ?

        """
        return self._eta_scan_del_f

    # Setter
    @eta_scan_del_f.setter
    def eta_scan_del_f(self, value):
        self._eta_scan_del_f = value

    ## End eta_scan_del_f property definition
    ###########################################################################

    ###########################################################################
    ## Start eta_scan_averages property definition

    # Getter
    @property
    def eta_scan_averages(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:eta_scan_averages`.

        See Also
        --------
        ?

        """
        return self._eta_scan_averages

    # Setter
    @eta_scan_averages.setter
    def eta_scan_averages(self, value):
        self._eta_scan_averages = value

    ## End eta_scan_averages property definition
    ###########################################################################

    ###########################################################################
    ## Start delta_freq property definition

    # Getter
    @property
    def delta_freq(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:delta_freq`.

        See Also
        --------
        ?

        """
        return self._delta_freq

    # Setter
    @delta_freq.setter
    def delta_freq(self, value):
        self._delta_freq = value

    ## End delta_freq property definition
    ###########################################################################

    ###########################################################################
    ## Start bands property definition

    # Getter
    @property
    def bands(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `tune_band:bands`.

        See Also
        --------
        ?

        """
        return self._bands

    # Setter
    @bands.setter
    def bands(self, value):
        self._bands = value

    ## End bands property definition
    ###########################################################################

    ###########################################################################
    ## Start default_tune property definition

    # Getter
    @property
    def default_tune(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:default_tune`.

        See Also
        --------
        ?

        """
        return self._default_tune

    # Setter
    @default_tune.setter
    def default_tune(self, value):
        self._default_tune = value

    ## End default_tune property definition
    ###########################################################################

    ###########################################################################
    ## Start epics_root property definition

    # Getter
    @property
    def epics_root(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:epics_root`.

        See Also
        --------
        ?

        """
        return self._epics_root

    # Setter
    @epics_root.setter
    def epics_root(self, value):
        self._epics_root = value

    ## End epics_root property definition
    ###########################################################################

    ###########################################################################
    ## Start smurf_cmd_dir property definition

    # Getter
    @property
    def smurf_cmd_dir(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:smurf_cmd_dir`.

        See Also
        --------
        ?

        """
        return self._smurf_cmd_dir

    # Setter
    @smurf_cmd_dir.setter
    def smurf_cmd_dir(self, value):
        self._smurf_cmd_dir = value

    ## End smurf_cmd_dir property definition
    ###########################################################################

    ###########################################################################
    ## Start tune_dir property definition

    # Getter
    @property
    def tune_dir(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:tune_dir`.

        See Also
        --------
        ?

        """
        return self._tune_dir

    # Setter
    @tune_dir.setter
    def tune_dir(self, value):
        self._tune_dir = value

    ## End tune_dir property definition
    ###########################################################################

    ###########################################################################
    ## Start status_dir property definition

    # Getter
    @property
    def status_dir(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:status_dir`.

        See Also
        --------
        ?

        """
        return self._status_dir

    # Setter
    @status_dir.setter
    def status_dir(self, value):
        self._status_dir = value

    ## End status_dir property definition
    ###########################################################################

    ###########################################################################
    ## Start default_data_dir property definition

    # Getter
    @property
    def default_data_dir(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:default_data_dir`.

        See Also
        --------
        ?

        """
        return self._default_data_dir

    # Setter
    @default_data_dir.setter
    def default_data_dir(self, value):
        self._default_data_dir = value

    ## End default_data_dir property definition
    ###########################################################################

    ###########################################################################
    ## Start ultrascale_temperature_limit_degC property definition

    # Getter
    @property
    def ultrascale_temperature_limit_degC(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:ultrascale_temperature_limit_degC`.

        See Also
        --------
        ?

        """
        return self._ultrascale_temperature_limit_degC

    # Setter
    @ultrascale_temperature_limit_degC.setter
    def ultrascale_temperature_limit_degC(self, value):
        self._ultrascale_temperature_limit_degC = value

    ## End ultrascale_temperature_limit_degC property definition
    ###########################################################################

    ###########################################################################
    ## Start timing_reference property definition

    # Getter
    @property
    def timing_reference(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:timing_reference`.

        See Also
        --------
        ?

        """
        return self._timing_reference

    # Setter
    @timing_reference.setter
    def timing_reference(self, value):
        self._timing_reference = value

    ## End timing_reference property definition
    ###########################################################################

    ###########################################################################
    ## Start feedback_start_frac property definition

    # Getter
    @property
    def feedback_start_frac(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:feedback_start_frac`.

        See Also
        --------
        ?

        """
        return self._feedback_start_frac

    # Setter
    @feedback_start_frac.setter
    def feedback_start_frac(self, value):
        self._feedback_start_frac = value

    ## End feedback_start_frac property definition
    ###########################################################################

    ###########################################################################
    ## Start feedback_end_frac property definition

    # Getter
    @property
    def feedback_end_frac(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `smurf_to_mce:feedback_end_frac`.

        See Also
        --------
        ?

        """
        return self._feedback_end_frac

    # Setter
    @feedback_end_frac.setter
    def feedback_end_frac(self, value):
        self._feedback_end_frac = value

    ## End feedback_end_frac property definition
    ###########################################################################

    ###########################################################################
    ## Start amplitude_scale property definition

    # Getter
    @property
    def amplitude_scale(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `amplitude_scale`.

        See Also
        --------
        ?

        """
        return self._amplitude_scale

    # Setter
    @amplitude_scale.setter
    def amplitude_scale(self, value):
        self._amplitude_scale = value

    ## End amplitude_scale property definition
    ###########################################################################

    ###########################################################################
    ## Start iq_swap_in property definition

    # Getter
    @property
    def iq_swap_in(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `iq_swap_in`.

        See Also
        --------
        ?

        """
        return self._iq_swap_in

    # Setter
    @iq_swap_in.setter
    def iq_swap_in(self, value):
        self._iq_swap_in = value

    ## End iq_swap_in property definition
    ###########################################################################

    ###########################################################################
    ## Start iq_swap_out property definition

    # Getter
    @property
    def iq_swap_out(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `iq_swap_out`.

        See Also
        --------
        ?

        """
        return self._iq_swap_out

    # Setter
    @iq_swap_out.setter
    def iq_swap_out(self, value):
        self._iq_swap_out = value

    ## End iq_swap_out property definition
    ###########################################################################

    ###########################################################################
    ## Start ref_phase_delay property definition

    # Getter
    @property
    def ref_phase_delay(self):
        """DEPRECATED
        Coarse (analog + digital) round-trip delay.

        Gets or sets the coarse (analog + digital) round-trip delay.
        This is the total time it takes a tone to traverse the
        synthesis filter bank, get generated by the DAC, propagate
        through the external analog RF circuit (e.g. through a cold RF
        chain including MUX chips), get digitized by the ADC, and then
        traverse the analysis filter bank.  This register is named
        `refPhaseDelay` in firmware.

        Unit-less unsigned integer.  Each step is a clock tick whose
        frequency is given by
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_channel_frequency_mhz`
        in MHz.  Different carrier firmware versions support different
        ranges.

        This delay is applied to the phase of each generated tone, and
        an overall phase rotation of (`etaPhase` - `refPhaseDelay`) is
        applied to each tone after demodulation and downmix.
        `refPhaseDelay` can be measured using the
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.estimate_phase_delay`
        routine.

        The total effective delay can be fine tuned by also setting
        the `refPhaseDelayFine` register (see
        :func:`ref_phase_delay_fine`) which adjusts the compensated
        delay more finely by 307.2 MHz ticks (although they way
        `refPhaseDelayFine` compensates for delay in firmware is not
        the same as `refPhaseDelay` - see the docstring for the
        :func:`ref_phase_delay_fine` property for more details).

        Specified in the pysmurf configuration file as
        `init:band_#:refPhaseDelay` with # the SMuRF 500 MHz band
        number *e.g.* `init:band_0:refPhaseDelay` for band 0.

        .. warning::
           Because `refPhaseDelay` and `refPhaseDelayFine` include the
           digital delay, it will vary for different firmware
           versions.

        Examples
        --------
        For firmware where `refPhaseDelay` is measured in 2.4 MHz
        ticks, if `refPhaseDelay` is 6 that corresponds to a time
        delay of 6/(2.4 MHz) = 2.5 microseconds.  For example, a 2
        microsecond delay for a 100 kHz phi0 rate corresponds to an ~1
        rad phase shift -- so it's particularly important to set
        `refPhaseDelay` correctly if running at high phi0 rates (10
        kHz or higher).

        Returns
        -------
        int
           Coarse (analog + digital) round-trip delay.  Unit-less
           unsigned integer.  Each step is a clock tick whose rate is
           given by
           :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_channel_frequency_mhz`
           in MHz.  Different carrier firmware versions support
           different ranges.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.estimate_phase_delay` :
              Measures `ref_phase_delay` and `ref_phase_delay_fine`.

        :func:`ref_phase_delay_fine` : Fine adjustment for system roundtrip delay.

        :func:`lms_delay` : System roundtrip delay including the tracking algorithm.

        """
        return self._ref_phase_delay

    # Setter
    @ref_phase_delay.setter
    def ref_phase_delay(self, value):
        self._ref_phase_delay = value

    ## End ref_phase_delay property definition
    ###########################################################################

    ###########################################################################
    ## Start ref_phase_delay_fine property definition

    # Getter
    @property
    def ref_phase_delay_fine(self):
        """DEPRECATED

        Fine adjust for (analog + digital) round-trip delay.

        Gets or sets fine adjustment for the total (analog + digital)
        round-trip delay.  This allows for fine adjustment of the
        total effective system round-trip delay on top of the coarser
        correction provided by programming the `refPhaseDelay` register
        (see :func:`ref_phase_delay` and
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.set_ref_phase_delay`
        for more details).

        Unit-less unsigned integer.  Each step is a 307.2 MHz tick
        (3.255 ns), for all firmware versions.  Different carrier
        firmware versions support different ranges.

        This register is named `refPhaseDelayFine` in firmware and is
        implemented differently than `refPhaseDelay` : setting
        `refPhaseDelayFine` only adds a time lag to the RF DAC output.
        `refPhaseDelayFine` can be measured using the
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.estimate_phase_delay`
        routine.

        Specified in the pysmurf configuration file as
        `init:band_#:refPhaseDelayFine` with # the SMuRF 500 MHz band
        number *e.g.* `init:band_0:refPhaseDelayFine` for band 0.

        .. warning::
           Because `refPhaseDelay` and `refPhaseDelayFine` include the
           digital delay, it will vary for different firmware
           versions.

        Examples
        --------
        Say the total delay measured using
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.estimate_phase_delay`
        is 2.3 microseconds.  To program this delay, you'd first set
        `refPhaseDelay` to 6, which is 2.5 microseconds (assuming that
        `refPhaseDelay` is measured in 2.4 MHz ticks, and so 6/(2.4
        MHz) = 2.5 microseconds).  Because `refPhaseDelayFine` adds
        time lag to the RF DAC output, it subtracts from the total
        delay (so it compensates in the opposite direction that
        `refPhaseDelay` does).  So setting `refPhaseDelayFine` to 61
        would result in a programmed delay of:

        | `refPhaseDelay`/(2.4 MHz) - `refPhaseDelayFine`/(307.2 MHz)
        | = 6/(2.4 MHz) - 61/(307.2 MHz)
        | = 2.30143 microseconds

        which is as close to 2.3 microseconds as we can program the
        delay, given the clock rates.

        Returns
        -------
        int
           Fine adjustment for (analog + digital) round-trip delay.
           Unit-less unsigned integer.  Each step is a 307.2 MHz tick.
           Different carrier firmware versions support different
           ranges.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.estimate_phase_delay` :
              Measures `ref_phase_delay` and `ref_phase_delay_fine`.

        :func:`ref_phase_delay` : Coarse system roundtrip delay.

        :func:`lms_delay` : System roundtrip delay including the tracking algorithm.

        """
        return self._ref_phase_delay_fine

    # Setter
    @ref_phase_delay_fine.setter
    def ref_phase_delay_fine(self, value):
        self._ref_phase_delay_fine = value

    ## End ref_phase_delay_fine property definition
    ###########################################################################

    ###########################################################################
    ## Start band_delay_us property definition

    # Getter
    @property
    def band_delay_us(self):
        """Total compensation for system latency - cable, ADC/DAC, and DSP

        Returns
        -------
        double
           Adjustment for (analog + digital) round-trip delay in
           micro seconds.

        See Also
        --------
        :func:`~pysmurf.client.util.smurf_util.SmurfUtilMixin.estimate_phase_delay` :
              Measures `band_delay_us`.

        """
        return self._band_delay_us

    # Setter
    @band_delay_us.setter
    def band_delay_us(self, value):
        self._band_delay_us = value

    ## End band_delay_us property definition
    ###########################################################################

    ###########################################################################
    ## Start lms_delay property definition

    # Getter
    @property
    def lms_delay(self):
        """DEPRECATED

        Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `lms_delay`.

        See Also
        --------
        ?

        """
        return self._lms_delay

    # Setter
    @lms_delay.setter
    def lms_delay(self, value):
        self._lms_delay = value

    ## End lms_delay property definition
    ###########################################################################

    ###########################################################################
    ## Start trigger_reset_delay property definition

    # Getter
    @property
    def trigger_reset_delay(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `trigger_reset_delay`.

        See Also
        --------
        ?

        """
        return self._trigger_reset_delay

    # Setter
    @trigger_reset_delay.setter
    def trigger_reset_delay(self, value):
        self._trigger_reset_delay = value

    ## End trigger_reset_delay property definition
    ###########################################################################

    ###########################################################################
    ## Start feedback_enable property definition

    # Getter
    @property
    def feedback_enable(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `feedback_enable`.

        See Also
        --------
        ?

        """
        return self._feedback_enable

    # Setter
    @feedback_enable.setter
    def feedback_enable(self, value):
        self._feedback_enable = value

    ## End feedback_enable property definition
    ###########################################################################

    ###########################################################################
    ## Start feedback_gain property definition

    # Getter
    @property
    def feedback_gain(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `feedback_gain`.

        See Also
        --------
        ?

        """
        return self._feedback_gain

    # Setter
    @feedback_gain.setter
    def feedback_gain(self, value):
        self._feedback_gain = value

    ## End feedback_gain property definition
    ###########################################################################

    ###########################################################################
    ## Start feedback_limit_khz property definition

    # Getter
    @property
    def feedback_limit_khz(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `feedback_limit_khz`.

        See Also
        --------
        ?

        """
        return self._feedback_limit_khz

    # Setter
    @feedback_limit_khz.setter
    def feedback_limit_khz(self, value):
        self._feedback_limit_khz = value

    ## End feedback_limit_khz property definition
    ###########################################################################

    ###########################################################################
    ## Start feedback_polarity property definition

    # Getter
    @property
    def feedback_polarity(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `feedback_polarity`.

        See Also
        --------
        ?

        """
        return self._feedback_polarity

    # Setter
    @feedback_polarity.setter
    def feedback_polarity(self, value):
        self._feedback_polarity = value

    ## End feedback_polarity property definition
    ###########################################################################

    ###########################################################################
    ## Start data_out_mux property definition

    # Getter
    @property
    def data_out_mux(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `data_out_mux`.

        See Also
        --------
        ?

        """
        return self._data_out_mux

    # Setter
    @data_out_mux.setter
    def data_out_mux(self, value):
        self._data_out_mux = value

    ## End data_out_mux property definition
    ###########################################################################

    ###########################################################################
    ## Start dsp_enable property definition

    # Getter
    @property
    def dsp_enable(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `dsp_enable`.

        See Also
        --------
        ?

        """
        return self._dsp_enable

    # Setter
    @dsp_enable.setter
    def dsp_enable(self, value):
        self._dsp_enable = value

    ## End dsp_enable property definition
    ###########################################################################

    ###########################################################################
    ## Start att_uc property definition

    # Getter
    @property
    def att_uc(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `att_uc`.

        See Also
        --------
        ?

        """
        return self._att_uc

    # Setter
    @att_uc.setter
    def att_uc(self, value):
        self._att_uc = value

    ## End att_uc property definition
    ###########################################################################

    ###########################################################################
    ## Start att_dc property definition

    # Getter
    @property
    def att_dc(self):
        """Short description.

        Gets or sets ?.
        Units are ?.

        Specified in the pysmurf configuration file as
        `att_dc`.

        See Also
        --------
        ?

        """
        return self._att_dc

    # Setter
    @att_dc.setter
    def att_dc(self, value):
        self._att_dc = value

    ## End att_dc property definition
    ###########################################################################
