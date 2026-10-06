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

__all__ = ['SmurfConfigPropertiesMixin', 'PER_BAND_KEYS']

# The per-band keys exposed as {band: value} dictionaries.
PER_BAND_KEYS = ('amplitude_scale', 'att_uc', 'att_dc', 'iq_swap_in', 'iq_swap_out',
                 'data_out_mux', 'band_delay_us', 'trigger_reset_delay', 'lms_gain',
                 'feedback_enable', 'feedback_gain', 'feedback_limit_khz', 'feedback_polarity',
                 'lms_freq_hz', 'delta_freq', 'feedback_start_frac', 'feedback_end_frac',
                 'gradient_descent_gain', 'gradient_descent_averages',
                 'gradient_descent_converge_hz', 'gradient_descent_step_hz',
                 'gradient_descent_momentum', 'gradient_descent_beta', 'eta_scan_averages',
                 'eta_scan_del_f')


class SmurfConfigPropertiesMixin:
    """The configuration values pysmurf reads, as properties over the resolved configuration.

    ``self.config`` is the :class:`cryodaq.Resolved` the client was built from
    (or reattached to), and every property here reads from it, so the values a
    tuning or analysis method sees are the values the file resolved to. The
    per-band properties are ``{band: value}`` dictionaries and the wiring tables
    are numpy arrays in the shapes analysis code indexes:
    ``pic_to_bias_group`` ``(n, 2)``, ``bias_group_to_pair`` ``(n, 3)`` with the
    group first, ``bad_mask`` ``(n, 2)``.

    :meth:`copy_config_to_properties` builds those tables once from a
    resolution; the client calls it when it loads or reattaches to a
    configuration. Before that every property is ``None``, which is what an
    offline client with no configuration has always had.

    Examples
    --------
    >>> S.pA_per_phi0
    9000000.0
    >>> S.amplitude_scale[4]
    11
    """

    def __init__(self, *args, **kwargs):
        self.config = None
        # Built by copy_config_to_properties; the tables analysis code indexes,
        # and the per-band dictionaries it mutates in place.
        self._bands = None
        self._n_bias_groups = None
        self._pic_to_bias_group = None
        self._bias_group_to_pair = None
        self._bad_mask = None
        self._attenuator = None
        self._amplitude_scale = None
        self._per_band_tables = None
        self._pA_per_phi0 = None
        self._bias_line_resistance = None
        self._high_low_current_ratio = None
        # The three values live code assigns at run time.
        self._fraction_full_scale = None
        self._tune_dir = None
        self._status_dir = None

    def copy_config_to_properties(self, config):
        """Adopt a resolved configuration and build the tables from it.

        Parameters
        ----------
        config : :class:`cryodaq.Resolved`
            As :func:`pysmurf.client.config.load` returns it.
        """
        self.config = config
        v = config.values
        wiring = v['wiring']
        self._bands = sorted(v['bands'])
        self._pA_per_phi0 = wiring['pA_per_phi0']
        self._bias_line_resistance = wiring['bias_line_resistance']
        self._high_low_current_ratio = wiring['high_low_current_ratio']
        self._fraction_full_scale = v['tune']['fraction_full_scale']
        self._tune_dir = v['paths']['tune']
        self._status_dir = v['paths']['status']
        # Every per-band table is built once and handed out as the same
        # dictionary each time: callers write into them -- tracking_setup stores
        # the LMS frequency it measured, sodetlib the tone power it chose -- and
        # a copy would take the write and lose it.
        self._per_band_tables = {key: {band: block.get(key) for band, block in v['bands'].items()}
                                 for key in PER_BAND_KEYS}
        # The three deprecated delay views read the `delay` block, 0 / None where a
        # band gives band_delay_us instead, as the legacy properties did.
        delays = {band: block.get('delay') or {} for band, block in v['bands'].items()}
        self._per_band_tables['ref_phase_delay'] = {b: d.get('ref_phase', 0) for b, d in delays.items()}
        self._per_band_tables['ref_phase_delay_fine'] = {b: d.get('ref_phase_fine', 0) for b, d in delays.items()}
        self._per_band_tables['lms_delay'] = {b: d.get('lms') for b, d in delays.items()}
        self._amplitude_scale = self._per_band_tables['amplitude_scale']

        att = v['attenuator']
        self._attenuator = {'band': np.array([att[k] for k in att], dtype=int),
                            'att': np.array([int(k[-1]) for k in att], dtype=int)}

        # The two wiring tables are rows in ascending key order. The legacy
        # loader kept the file's order, and a record that has been through JSON
        # sorts its keys as strings ('10' before '2'); analysis code indexes
        # these arrays by row, so the order is fixed here to the one thing
        # every route agrees on.
        pic = wiring['pic_to_bias_group']
        self._pic_to_bias_group = np.array([[int(k), pic[k]] for k in sorted(pic, key=int)],
                                           dtype=int).reshape(len(pic), 2)
        pairs = wiring['bias_group_to_pair']
        self._n_bias_groups = len(pairs)
        self._bias_group_to_pair = np.array([[int(k), *pairs[k]] for k in sorted(pairs, key=int)],
                                            dtype=int).reshape(len(pairs), 3)
        self._bad_mask = np.array(wiring['bad_mask'], dtype=float).reshape(len(wiring['bad_mask']), 2)

    # ------------------------------------------------------------------

    def _value(self, *keys):
        """A value out of the resolved configuration, or None when there is none yet."""
        node = None if self.config is None else self.config.values
        for key in keys:
            if node is None:
                return None
            node = node.get(key)
        return node

    def _per_band(self, key):
        """The live ``{band: value}`` dictionary for one per-band key; None with no configuration."""
        if self.config is None:
            return None
        return self._per_band_tables[key]

    def delay_writes(self, band):
        """The band-delay registers ``setup()`` writes for one band, from the live properties.

        A band names its delay one of two ways: ``delay`` gives the three
        firmware registers directly and wins when present -- ``lms`` left unset
        means the same value as ``ref_phase``, as the firmware's own linked
        variable does -- and otherwise ``band_delay_us`` gives the total and
        the firmware derives the three. Read from the per-band tables, so a
        value changed on the instance is what is written, as for every other
        per-band register; ``ref_phase_delay`` is 0 where the configuration
        gave ``band_delay_us``, which is how the two are told apart.

        Parameters
        ----------
        band : int

        Returns
        -------
        tuple of (str, value)
            Setter names without their ``set_`` prefix, in the order they are called.
        """
        ref_phase = self.ref_phase_delay[band]
        if ref_phase:
            lms = self.lms_delay[band]
            return (('ref_phase_delay', ref_phase),
                    ('ref_phase_delay_fine', self.ref_phase_delay_fine[band]),
                    ('lms_delay', int(ref_phase) if lms is None else lms))
        return (('band_delay_us', self.band_delay_us[band]),)

    # ------------------------------------------------------------------
    # paths and scalars
    # ------------------------------------------------------------------

    @property
    def default_data_dir(self):
        """Root of the data tree, ``paths.data``."""
        return self._value('paths', 'data')

    @property
    def smurf_cmd_dir(self):
        """``paths.smurf_cmd``."""
        return self._value('paths', 'smurf_cmd')

    @property
    def tune_dir(self):
        """Where tune files are written, ``paths.tune``; the client may append a path id."""
        return self._tune_dir

    @tune_dir.setter
    def tune_dir(self, value):
        self._tune_dir = value

    @property
    def status_dir(self):
        """Where status dumps and the configuration records go, ``paths.status``."""
        return self._status_dir

    @status_dir.setter
    def status_dir(self, value):
        self._status_dir = value

    @property
    def pA_per_phi0(self):
        """Flux-ramp demodulated phase to current, pA per Phi0, ``wiring.pA_per_phi0``."""
        return self._pA_per_phi0

    @property
    def R_sh(self):
        """Shunt resistance in ohms, ``wiring.R_sh``."""
        return self._value('wiring', 'R_sh')

    @property
    def bias_line_resistance(self):
        """Bias line resistance in ohms, ``wiring.bias_line_resistance``."""
        return self._bias_line_resistance

    @property
    def high_low_current_ratio(self):
        """Ratio of high- to low-current-mode bias, ``wiring.high_low_current_ratio``."""
        return self._high_low_current_ratio

    @property
    def high_current_mode_bool(self):
        """Whether TES bias defaults to high-current mode, ``wiring.high_current_mode``."""
        return self._value('wiring', 'high_current_mode')

    @property
    def all_groups(self):
        """The bias groups this system drives, ``wiring.all_bias_groups``."""
        return self._value('wiring', 'all_bias_groups')

    @property
    def n_bias_groups(self):
        """How many bias groups ``bias_group_to_pair`` defines."""
        return self._n_bias_groups

    @property
    def pic_to_bias_group(self):
        """``(n, 2)`` int array of ``[PIC channel, bias group]``."""
        return self._pic_to_bias_group

    @property
    def bias_group_to_pair(self):
        """``(n, 3)`` int array of ``[bias group, DAC+, DAC-]``."""
        return self._bias_group_to_pair

    @property
    def bad_mask(self):
        """``(n, 2)`` float array of ``[lo, hi]`` MHz ranges never to tune."""
        return self._bad_mask

    @property
    def attenuator(self):
        """``{'band': int array, 'att': int array}`` -- which band each RF attenuator serves."""
        return self._attenuator

    @property
    def amplifier(self):
        """The ``amplifier`` block as a mapping; the amplifier bias commands read it."""
        return self._value('amplifier')

    @property
    def num_flux_ramp_counter_bits(self):
        """``flux_ramp.num_flux_ramp_counter_bits``, 20 or 32."""
        return self._value('flux_ramp', 'num_flux_ramp_counter_bits')

    @property
    def timing_reference(self):
        """``timing.timing_reference``: ``ext_ref``, ``backplane`` or ``fiber``."""
        return self._value('timing', 'timing_reference')

    @property
    def fs(self):
        """Sample rate in Hz, ``fs``."""
        return self._value('fs')

    @property
    def dsp_enable(self):
        """``dsp_enable``, written to the firmware at setup."""
        return self._value('dsp_enable')

    @property
    def ultrascale_temperature_limit_degC(self):
        """FPGA temperature above which setup refuses to run, or None."""
        return self._value('ultrascale_temperature_limit_degC')

    @property
    def default_tune(self):
        """A tune file to load at start, ``tune.default_tune``, or None."""
        return self._value('tune', 'default_tune')

    @property
    def reset_rate_khz(self):
        """Flux ramp reset rate in kHz, ``tune.reset_rate_khz``."""
        return self._value('tune', 'reset_rate_khz')

    @property
    def fraction_full_scale(self):
        """Flux ramp amplitude as a fraction of full scale, ``tune.fraction_full_scale``."""
        return self._fraction_full_scale

    @fraction_full_scale.setter
    def fraction_full_scale(self, value):
        self._fraction_full_scale = value

    @property
    def bands(self):
        """The bands the configuration declares, sorted."""
        return self._bands

    # ------------------------------------------------------------------
    # per band: {band: value}
    # ------------------------------------------------------------------

    @property
    def amplitude_scale(self):
        """Tone amplitude per band, 3 dB steps. The same dictionary each call, so it can be edited."""
        return self._amplitude_scale

    @property
    def att_uc(self):
        """Up-converter attenuation per band, 0.5 dB steps."""
        return self._per_band('att_uc')

    @property
    def att_dc(self):
        """Down-converter attenuation per band, 0.5 dB steps."""
        return self._per_band('att_dc')

    @property
    def iq_swap_in(self):
        """Swap I and Q on input, per band."""
        return self._per_band('iq_swap_in')

    @property
    def iq_swap_out(self):
        """Swap I and Q on output, per band."""
        return self._per_band('iq_swap_out')

    @property
    def data_out_mux(self):
        """Which two DAC outputs a band drives, per band."""
        return self._per_band('data_out_mux')

    @property
    def band_delay_us(self):
        """Total band delay in microseconds, per band; None where ``delay`` is given instead."""
        return self._per_band('band_delay_us')

    @property
    def ref_phase_delay(self):
        """``delay.ref_phase`` per band, or 0 where the band gives ``band_delay_us``."""
        return self._per_band('ref_phase_delay')

    @property
    def ref_phase_delay_fine(self):
        """``delay.ref_phase_fine`` per band, or 0."""
        return self._per_band('ref_phase_delay_fine')

    @property
    def lms_delay(self):
        """``delay.lms`` per band, or None meaning the same as ``ref_phase``."""
        return self._per_band('lms_delay')

    @property
    def trigger_reset_delay(self):
        """Flux ramp trigger reset delay per band, 2.4 MHz ticks."""
        return self._per_band('trigger_reset_delay')

    @property
    def lms_gain(self):
        """LMS feedback gain per band, powers of two."""
        return self._per_band('lms_gain')

    @property
    def feedback_enable(self):
        """Feedback enabled per band."""
        return self._per_band('feedback_enable')

    @property
    def feedback_gain(self):
        """Feedback gain per band."""
        return self._per_band('feedback_gain')

    @property
    def feedback_limit_khz(self):
        """Feedback limit per band, kHz."""
        return self._per_band('feedback_limit_khz')

    @property
    def feedback_polarity(self):
        """Feedback polarity per band."""
        return self._per_band('feedback_polarity')

    @property
    def lms_freq_hz(self):
        """Flux ramp demodulation frequency per band, Hz."""
        return self._per_band('lms_freq_hz')

    @property
    def delta_freq(self):
        """Frequency step for the eta scan per band, MHz."""
        return self._per_band('delta_freq')

    @property
    def feedback_start_frac(self):
        """Where in the flux ramp feedback starts, per band, as a fraction."""
        return self._per_band('feedback_start_frac')

    @property
    def feedback_end_frac(self):
        """Where in the flux ramp feedback ends, per band, as a fraction."""
        return self._per_band('feedback_end_frac')

    @property
    def gradient_descent_gain(self):
        """Gradient descent gain per band."""
        return self._per_band('gradient_descent_gain')

    @property
    def gradient_descent_averages(self):
        """Gradient descent averages per band."""
        return self._per_band('gradient_descent_averages')

    @property
    def gradient_descent_converge_hz(self):
        """Gradient descent convergence criterion per band, Hz."""
        return self._per_band('gradient_descent_converge_hz')

    @property
    def gradient_descent_step_hz(self):
        """Gradient descent step per band, Hz."""
        return self._per_band('gradient_descent_step_hz')

    @property
    def gradient_descent_momentum(self):
        """Gradient descent momentum per band."""
        return self._per_band('gradient_descent_momentum')

    @property
    def gradient_descent_beta(self):
        """Gradient descent beta per band."""
        return self._per_band('gradient_descent_beta')

    @property
    def eta_scan_averages(self):
        """Eta scan averages per band."""
        return self._per_band('eta_scan_averages')

    @property
    def eta_scan_del_f(self):
        """Eta scan frequency step per band, Hz."""
        return self._per_band('eta_scan_del_f')
