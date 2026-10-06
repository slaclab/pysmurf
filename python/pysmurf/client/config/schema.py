#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf Client Configuration Schema
#-----------------------------------------------------------------------------
# File       : schema.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
#    What the keys of a pysmurf client configuration mean, and what values
#    they take. `validate` is handed the merged values by `cryodaq.config.load`
#    and returns them checked and coerced, with `band_default` applied to every
#    declared band and the firmware's data_out_mux filled in where a band
#    leaves it unset.
#
#    Nothing here touches the filesystem: whether a directory exists or a tune
#    file is present is decided where the directory is used, so a file that is
#    valid on the system that runs it is valid everywhere.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import math
import re
from typing import Any, Dict

from schema import And, Optional, Or, Schema, SchemaError, Use

__all__ = ['validate', 'ConfigInvalid', 'DATA_OUT_MUX_DEFAULT', 'BAND_KEYS',
           'TUNING_KEYS']

# The firmware's data_out_mux per band; a band that does not set its own gets this.
DATA_OUT_MUX_DEFAULT = {0: [2, 3], 1: [0, 1], 2: [6, 7], 3: [8, 9],
                        4: [2, 3], 5: [0, 1], 6: [6, 7], 7: [8, 9]}

# Every key a band block takes, and the tuning parameters among them.
TUNING_KEYS = ('lms_freq_hz', 'delta_freq', 'feedback_start_frac', 'feedback_end_frac',
               'gradient_descent_gain', 'gradient_descent_averages',
               'gradient_descent_converge_hz', 'gradient_descent_momentum',
               'gradient_descent_step_hz', 'gradient_descent_beta',
               'eta_scan_averages', 'eta_scan_del_f')
BAND_KEYS = ('iq_swap_in', 'iq_swap_out', 'feedback_enable', 'feedback_polarity',
             'feedback_gain', 'feedback_limit_khz', 'att_uc', 'att_dc', 'amplitude_scale',
             'data_out_mux', 'trigger_reset_delay', 'lms_gain', 'band_delay_us',
             'delay') + TUNING_KEYS


class ConfigInvalid(ValueError):
    """A configuration value pysmurf cannot use.

    Parameters
    ----------
    key : str
        The dotted key at fault, e.g. ``'bands.4.att_uc'``.
    reason : str
        What is wrong with it, in a few words.
    """

    def __init__(self, key: str, reason: str):
        self.key = key
        self.reason = reason
        super().__init__(f"{key}: {reason}")


def _is_int(n: Any) -> bool:
    # YAML's `true` is a bool, which Python counts as an int; a register value is not a flag.
    return isinstance(n, int) and not isinstance(n, bool)


def _bool_int(n: Any) -> bool:
    return _is_int(n) and n in (0, 1)


def _in_range(lo: int, hi: int):
    return lambda n: _is_int(n) and lo <= n < hi


def _number(n: Any) -> float:
    """A finite number as float; a bool, a string or nan/inf is not one."""
    if isinstance(n, bool) or not isinstance(n, (int, float)):
        raise SchemaError(None, 'a number')
    if not math.isfinite(n):
        raise SchemaError(None, 'a finite number')
    return float(n)


def _schema():
    """The declarative schema, built on first use."""
    number = Use(_number)
    positive = And(number, lambda f: f > 0)
    unit = And(number, lambda f: 0 <= f <= 1)
    amp_block = {
        'drain_conversion_b': number, 'drain_conversion_m': number,
        'drain_dac_num': _in_range(1, 33), 'drain_offset': number, 'drain_opamp_gain': number,
        'drain_pic_address': _is_int, 'drain_resistor': number,
        'drain_volt_default': number, 'drain_volt_min': number,
        'drain_volt_max': number, 'gate_bit_to_volt': number,
        Optional('gate_dac_num'): _in_range(1, 33), 'gate_volt_default': number,
        'gate_volt_min': number, 'gate_volt_max': number, 'power_bitmask': _is_int,
    }
    band = {
        'iq_swap_in': _bool_int,
        'iq_swap_out': _bool_int,
        'feedback_enable': _bool_int,
        'feedback_polarity': _bool_int,
        'feedback_gain': _in_range(0, 2**16),
        'feedback_limit_khz': positive,
        'att_uc': _in_range(0, 2**5),
        'att_dc': _in_range(0, 2**5),
        'amplitude_scale': _in_range(0, 2**4),
        'data_out_mux': And([_in_range(0, 10)], lambda l: len(l) == 2 and l[0] != l[1]),
        'trigger_reset_delay': _in_range(0, 2**7),
        'lms_gain': _in_range(0, 2**3),
        # Either the total delay in microseconds, or the three firmware
        # registers directly. `delay` wins when both are given.
        'band_delay_us': Or(None, And(number, lambda f: 0 <= f < 30)),
        # A triple with ref_phase 0 was never written -- setup() has always read
        # a zero as "use band_delay_us" -- so a block that gives one is refused
        # rather than silently ignored.
        Optional('delay'): {
            'ref_phase': _in_range(1, 2**5),
            Optional('ref_phase_fine', default=0): _in_range(0, 2**8),
            Optional('lms', default=None): Or(None, _in_range(0, 2**6)),
        },
        'lms_freq_hz': positive,
        'delta_freq': positive,
        'feedback_start_frac': unit,
        'feedback_end_frac': unit,
        'gradient_descent_gain': positive,
        'gradient_descent_averages': And(_is_int, lambda n: n > 0),
        'gradient_descent_converge_hz': positive,
        'gradient_descent_momentum': And(_is_int, lambda n: n >= 0),
        'gradient_descent_step_hz': positive,
        'gradient_descent_beta': unit,
        'eta_scan_averages': And(_is_int, lambda n: n > 0),
        'eta_scan_del_f': And(_is_int, lambda n: n > 0),
    }
    return Schema({
        # A directory is named, not checked: whether it exists is decided where
        # it is used. An empty name is no directory at all.
        'paths': {'data': And(str, len), 'smurf_cmd': And(str, len), 'tune': And(str, len),
                  'status': And(str, len)},
        'wiring': {
            'R_sh': positive,
            'bias_line_resistance': positive,
            'high_low_current_ratio': And(number, lambda f: f >= 1),
            'high_current_mode': _bool_int,
            'pA_per_phi0': number,
            # Sixteen bias groups, sixteen PIC channels, thirty-two RTM DACs.
            'pic_to_bias_group': {_in_range(0, 16): _in_range(0, 16)},
            # A list schema checks each element, not the count: say two.
            'bias_group_to_pair': {_in_range(0, 16):
                                   And([_in_range(1, 33)], lambda l: len(l) == 2,
                                       error='a bias group names two DACs, [plus, minus]')},
            'all_bias_groups': And([_in_range(0, 16)], lambda l: len(set(l)) == len(l),
                                   error='a bias group is listed twice'),
            'bad_mask': [And([number], lambda l: len(l) == 2 and l[0] < l[1] and
                             all(4000 <= x <= 8000 for x in l))],
        },
        'attenuator': {'att1': _in_range(0, 4), 'att2': _in_range(0, 4),
                       'att3': _in_range(0, 4), 'att4': _in_range(0, 4)},
        'amplifier': {
            'hemt_Vg': number, 'LNA_Vg': number,
            'bit_to_V_hemt': positive, 'bit_to_V_50k': positive,
            'dac_num_50k': _in_range(1, 33),
            'hemt_Id_offset': number, '50k_Id_offset': number,
            'hemt_gate_min_voltage': number, 'hemt_gate_max_voltage': number,
            'hemt_Vd_series_resistor': positive, '50K_amp_Vd_series_resistor': positive,
            'hemt': {Optional('gate_dac_num'): _in_range(1, 33), str: object},
            '50k': {Optional('gate_dac_num'): _in_range(1, 33), str: object},
            'hemt1': amp_block, 'hemt2': amp_block, '50k1': amp_block, '50k2': amp_block,
        },
        'flux_ramp': {'num_flux_ramp_counter_bits': lambda n: isinstance(n, int) and n in (20, 32)},
        'timing': {'timing_reference': lambda s: s in ('ext_ref', 'backplane', 'fiber')},
        'fs': positive,
        'dsp_enable': _bool_int,
        'ultrascale_temperature_limit_degC': Or(None, And(number, lambda f: 0 <= f <= 99)),
        'tune': {'default_tune': Or(None, And(str, len)), 'fraction_full_scale': And(number, lambda f: 0 < f <= 1),
                 'reset_rate_khz': And(number, lambda f: 0 <= f <= 100)},
        'band_default': dict,
        'bands': {_in_range(0, 8): band},
    })


def validate(values: Dict[str, Any]) -> Dict[str, Any]:
    """Check and coerce a merged pysmurf configuration.

    Parameters
    ----------
    values : dict
        The merged layers, as ``cryodaq.config.load`` hands them over.

    Returns
    -------
    dict
        The same shape, with ``band_default`` applied under every band, the
        firmware's ``data_out_mux`` filled in where a band has none, numbers
        coerced to the types pysmurf uses, and band keys as ``int``.

    Raises
    ------
    ConfigInvalid
        Naming the key at fault and why: a required value left ``null``, an
        unknown key, a value out of range, a band with neither ``band_delay_us``
        nor ``delay``, a DAC in two bias groups.
    """
    values = dict(values)
    # Anything the schema does not know is a typo or a key that no longer exists.
    known = _schema().schema
    for key in values:
        if key not in known:
            raise ConfigInvalid(key, 'not a pysmurf configuration key')

    # The tables keyed by a number -- bands, PIC channels, bias groups -- take
    # the key as YAML's `4:` or JSON's "4", and nothing else.
    if isinstance(values.get('wiring'), dict):
        for table, what, hi in (('pic_to_bias_group', 'a PIC channel', 16),
                                ('bias_group_to_pair', 'a bias group', 16)):
            if isinstance(values['wiring'].get(table), dict):
                values['wiring'][table] = _numbered(values['wiring'][table], f"wiring.{table}", what, hi)
    default = _mapping_or_missing(values, 'band_default')
    bands = {}
    for number, block in _numbered(_mapping_or_missing(values, 'bands'), 'bands', 'a band', 8).items():
        if block is not None and not isinstance(block, dict):
            raise ConfigInvalid(f"bands.{number}", 'a band is a mapping of per-band keys')
        merged = {**default, **(block or {})}
        merged.setdefault('data_out_mux', DATA_OUT_MUX_DEFAULT.get(number))
        merged.setdefault('band_delay_us', None)
        for name in merged:
            if name not in BAND_KEYS:
                raise ConfigInvalid(f"bands.{number}.{name}", 'not a per-band key')
        bands[number] = merged
    values['bands'] = bands

    _refuse_nulls(values)
    try:
        checked = _schema().validate(values)
    except SchemaError as e:
        # An `error=` given in the schema is in `errors`; the library's own
        # wording is in `autos`.
        said = [m for m in e.errors if m] or [m for m in e.autos if m] or [str(e)]
        raise ConfigInvalid(_key_of(e), said[-1]) from None

    for number, block in checked['bands'].items():
        if block.get('band_delay_us') is None and 'delay' not in block:
            raise ConfigInvalid(f"bands.{number}", 'set band_delay_us or a delay block')
    _refuse_shared_dacs(checked)
    return checked


# Keys that may stay null: they mean "none" rather than "unset".
_MAY_BE_NULL = ('tune.default_tune', 'ultrascale_temperature_limit_degC',
                'band_delay_us', 'delay.lms')


def _numbered(table: Dict[Any, Any], where: str, what: str, hi: int) -> Dict[int, Any]:
    """``table`` with its keys as ``int`` in ``[0, hi)``.

    A key is an integer or a digit string; ``4.5``, ``True`` and ``'four'`` are
    refused, and so is the same number under two spellings, rather than one
    entry silently replacing the other.
    """
    out: Dict[int, Any] = {}
    for key, value in table.items():
        number = int(key) if (isinstance(key, int) and not isinstance(key, bool)) or \
            (isinstance(key, str) and key.isdigit()) else -1
        if not 0 <= number < hi:
            raise ConfigInvalid(f"{where}.{key}", f"{what} is numbered 0-{hi - 1}")
        if number in out:
            raise ConfigInvalid(f"{where}.{key}", f"{what.split()[-1]} {number} is given twice")
        out[number] = value
    return out


def _mapping_or_missing(values: Dict[str, Any], key: str) -> Dict[str, Any]:
    """The mapping at ``key``, ``{}`` if absent or null; anything else is refused, not emptied."""
    value = values.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ConfigInvalid(key, f"a mapping, not {type(value).__name__}")
    return value


def _refuse_nulls(values: Dict[str, Any], prefix: str = '') -> None:
    """A required value the default leaves null and no layer set.

    ``band_default`` is not walked: it is a template, applied to the bands and
    checked there, and a null in it means "each band says".
    """
    for key, value in values.items():
        dotted = f"{prefix}{key}"
        if dotted == 'band_default':
            continue
        if isinstance(value, dict):
            _refuse_nulls(value, f"{dotted}.")
        elif value is None and not any(dotted == k or dotted.endswith(f".{k}") for k in _MAY_BE_NULL):
            raise ConfigInvalid(dotted, 'required, and no layer sets it')


def _refuse_shared_dacs(values: Dict[str, Any]) -> None:
    pairs = values['wiring']['bias_group_to_pair']
    dacs = [d for pair in pairs.values() for d in pair]
    if len(set(dacs)) != len(dacs):
        dup = sorted(d for d in set(dacs) if dacs.count(d) > 1)
        raise ConfigInvalid('wiring.bias_group_to_pair',
                            f"DAC(s) {dup} assigned to more than one bias group")
    dac_50k = values['amplifier']['dac_num_50k']
    for group, pair in pairs.items():
        if dac_50k in pair:
            raise ConfigInvalid('amplifier.dac_num_50k',
                                f"DAC {dac_50k} also drives bias group {group}")


def _key_of(error: Any) -> str:
    """The dotted key a schema error is about, as far as the library says."""
    keys = []
    for line in error.autos:
        if line is None:
            continue
        m = re.search(r"[Kk]ey '?([^' ]+)'? error", line)
        if m:
            keys.append(m.group(1))
    return '.'.join(keys) or '<config>'
