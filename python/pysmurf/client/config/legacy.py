#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf Legacy Configuration Converter
#-----------------------------------------------------------------------------
# File       : legacy.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
#    Reads the JSON-with-comments `.cfg` files pysmurf used before the YAML
#    schema and produces the equivalent YAML mapping. Every value a site set
#    is carried across under its new key; the per-band firmware delay registers
#    (refPhaseDelay, refPhaseDelayFine) become the band's `band_delay_us`, the
#    total they encode, from which the firmware derives them again; lmsDelay is
#    dropped, the firmware setting it from the same total. Keys nothing read
#    any more are dropped and named in a warning.
#
#    Used in two ways: `convert()` in memory when SmurfControl is handed a
#    `.cfg`, and `python -m pysmurf.client.config convert` to write the YAML
#    once and move a site over.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import json
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml

__all__ = ['convert', 'read_json_with_comments', 'to_yaml', 'DROPPED_KEYS']

# Keys the shipped files carry that nothing has read for years, and the register
# values every file set to what the server's own defaults already hold; the ones
# the round-trip check expects to see named. Any other unread key is named too.
DROPPED_KEYS = ('epics_root', 'chip_to_freq', 'smurf_to_mce', 'band_to_chip',
                'channel_assignment', 'flux_ramp.select_ramp', 'flux_ramp.ramp_start_mode',
                'constant.pa_per_phi0', 'init.bands', 'init.dspEnable',
                'ultrascale_temperature_limit_degC',
                'tune_band.n_samples', 'tune_band.grad_cut', 'tune_band.amp_cut',
                'tune_band.freq_max', 'tune_band.freq_min', 'tune_band.eta_scan_amplitude')

# Old per-band key -> new one. The delay registers are handled apart.
_BAND_RENAMES = {
    'iq_swap_in': 'iq_swap_in', 'iq_swap_out': 'iq_swap_out',
    'feedbackEnable': 'feedback_enable', 'feedbackPolarity': 'feedback_polarity',
    'feedbackGain': 'feedback_gain', 'feedbackLimitkHz': 'feedback_limit_khz',
    'att_uc': 'att_uc', 'att_dc': 'att_dc', 'amplitude_scale': 'amplitude_scale',
    'trigRstDly': 'trigger_reset_delay',
    'lmsGain': 'lms_gain', 'bandDelayUs': 'band_delay_us',
}
_DELAY_KEYS = ('refPhaseDelay', 'refPhaseDelayFine', 'lmsDelay')
# The firmware's own reading of its delay registers as microseconds: refPhaseDelay
# counts 2.4 MHz ticks of delay, refPhaseDelayFine 307.2 MHz ticks of lag taken
# off it. Its setter inverts this exactly for every value a shipped file holds.
_REF_PHASE_TICKS_PER_US = 2.4
_REF_PHASE_FINE_TICKS_PER_US = 307.2
# Old tune_band per-band table -> new per-band key.
_TUNING_RENAMES = {
    'lms_freq': 'lms_freq_hz', 'delta_freq': 'delta_freq',
    'feedback_start_frac': 'feedback_start_frac', 'feedback_end_frac': 'feedback_end_frac',
    'gradient_descent_gain': 'gradient_descent_gain',
    'gradient_descent_averages': 'gradient_descent_averages',
    'gradient_descent_converge_hz': 'gradient_descent_converge_hz',
    'gradient_descent_momentum': 'gradient_descent_momentum',
    'gradient_descent_step_hz': 'gradient_descent_step_hz',
    'gradient_descent_beta': 'gradient_descent_beta',
    'eta_scan_averages': 'eta_scan_averages', 'eta_scan_del_f': 'eta_scan_del_f',
}
_BAND_BLOCK = re.compile(r'^band_([0-7])$')
# The top-level keys the converter reads; every other one is reported dropped.
_TAKEN = frozenset((
    'default_data_dir', 'smurf_cmd_dir', 'tune_dir', 'status_dir', 'R_sh', 'bias_line_resistance',
    'high_low_current_ratio', 'pic_to_bias_group', 'bias_group_to_pair', 'all_bias_groups',
    'high_current_mode_bool', 'constant', 'bad_mask', 'attenuator', 'amplifier', 'timing', 'fs',
    'flux_ramp', 'init', 'tune_band'))
# A `#` starts a comment unless it is inside a double-quoted string.
_COMMENT = re.compile(r'("(?:[^"\\]|\\.)*")|#.*')


def read_json_with_comments(path: Union[str, Path]) -> Dict[str, Any]:
    """Parse a ``.cfg``: JSON with ``#`` comments, which may sit mid-object.

    A ``#`` inside a string is not a comment. The old reader cut the line at
    the first ``#`` wherever it was; no shipped file has one in a string, and
    this reads the same files the same way.

    Parameters
    ----------
    path : str or Path
        The legacy file.
    """
    text = Path(path).read_text(encoding='utf-8')
    stripped = _COMMENT.sub(lambda m: m.group(1) or '', text)
    return json.loads(stripped)


def convert(path: Union[str, Path], *, warn: bool = True) -> Dict[str, Any]:
    """The YAML-schema mapping equivalent to the legacy ``.cfg`` at ``path``.

    Parameters
    ----------
    path : str or Path
        A legacy JSON-with-comments configuration file.
    warn : bool
        Emit one ``DeprecationWarning`` naming the file and the keys dropped.

    Returns
    -------
    dict
        A mapping in the new schema's shape, ready for the validator; nothing
        is filled in from the defaults here, that is the loader's.
    """
    old = read_json_with_comments(path)
    new: Dict[str, Any] = {}
    dropped: List[str] = []

    def take(section: Dict[str, Any], key: str, *dotted: str) -> None:
        if key in section:
            target = new
            for part in dotted[:-1]:
                target = target.setdefault(part, {})
            target[dotted[-1]] = section[key]

    take(old, 'default_data_dir', 'paths', 'data')
    take(old, 'smurf_cmd_dir', 'paths', 'smurf_cmd')
    take(old, 'tune_dir', 'paths', 'tune')
    take(old, 'status_dir', 'paths', 'status')
    for key in ('R_sh', 'bias_line_resistance', 'high_low_current_ratio',
                'pic_to_bias_group', 'bias_group_to_pair', 'all_bias_groups'):
        take(old, key, 'wiring', key)
    take(old, 'high_current_mode_bool', 'wiring', 'high_current_mode')
    if 'constant' in old:
        take(old['constant'], 'pA_per_phi0', 'wiring', 'pA_per_phi0')
        dropped.extend(f"constant.{k}" for k in old['constant'] if k != 'pA_per_phi0')
    if 'bad_mask' in old:
        # The old mapping's keys were labels nothing read; the ranges are the value.
        new.setdefault('wiring', {})['bad_mask'] = list(old['bad_mask'].values())
    for key in ('attenuator', 'amplifier', 'timing', 'fs'):
        take(old, key, key)
    if 'flux_ramp' in old:
        take(old['flux_ramp'], 'num_flux_ramp_counter_bits', 'flux_ramp', 'num_flux_ramp_counter_bits')
        dropped.extend(f"flux_ramp.{k}" for k in old['flux_ramp'] if k != 'num_flux_ramp_counter_bits')

    init = old.get('init', {})
    bands: Dict[int, Dict[str, Any]] = {}
    for key, block in init.items():
        m = _BAND_BLOCK.match(key)
        if not m:
            # `bands` the old validator rebuilt from the band_# blocks present,
            # as the new schema does from the keys under `bands`.
            dropped.append(f"init.{key}")
            continue
        band = bands.setdefault(int(m.group(1)), {})
        for old_key, value in block.items():
            if old_key in _BAND_RENAMES:
                band[_BAND_RENAMES[old_key]] = value
            elif old_key not in _DELAY_KEYS:
                dropped.append(f"init.{key}.{old_key}")
        # The old setup() wrote the three registers when refPhaseDelay was set
        # and nonzero, and bandDelayUs otherwise. The registers it wrote are
        # carried as the total they encode; lmsDelay, which the firmware now
        # sets from that total, and a bandDelayUs it passed over are dropped.
        if block.get('refPhaseDelay'):
            coarse = block['refPhaseDelay'] / _REF_PHASE_TICKS_PER_US
            band['band_delay_us'] = coarse - block.get('refPhaseDelayFine', 0) / _REF_PHASE_FINE_TICKS_PER_US
            dropped.extend(f"init.{key}.{old}" for old in ('lmsDelay', 'bandDelayUs') if old in block)
        else:
            dropped.extend(f"init.{key}.{old}" for old in _DELAY_KEYS if old in block)

    tune_band = old.get('tune_band', {})
    for key in ('fraction_full_scale', 'reset_rate_khz', 'default_tune'):
        take(tune_band, key, 'tune', key)
    for old_key, table in tune_band.items():
        if old_key in ('fraction_full_scale', 'reset_rate_khz', 'default_tune'):
            continue
        if old_key in _TUNING_RENAMES and isinstance(table, dict):
            for band_key, value in table.items():
                # The band is left as the file spelled it; the schema judges it.
                bands.setdefault(_band_number(band_key), {})[_TUNING_RENAMES[old_key]] = value
        else:
            dropped.append(f"tune_band.{old_key}")
    if bands:
        new['bands'] = dict(sorted(bands.items()))

    # Whatever was not taken above is named, listed or not: a key this converter
    # does not know is still a key the site wrote.
    dropped.extend(k for k in old if k not in _TAKEN)
    if warn:
        gone = (f" Dropped, nothing reads them: {', '.join(sorted(set(dropped)))}"
                if dropped else '')
        warnings.warn(f"{path}: the JSON .cfg format is deprecated; write it as YAML with "
                      f"`python -m pysmurf.client.config convert {path}`.{gone}",
                      DeprecationWarning, stacklevel=2)
    return new


def _band_number(key: Any) -> Any:
    """A digit-string band key as ``int``, so ``band_4`` and ``tune_band:...:"4"`` meet; anything else as is."""
    return int(key) if isinstance(key, str) and key.isdigit() else key


def to_yaml(mapping: Dict[str, Any]) -> str:
    """A mapping as YAML text, keys in the order given, floats in a form YAML reads as floats.

    Parameters
    ----------
    mapping : dict
        What ``convert`` returned, or any mapping in the schema's shape.
    """
    return yaml.safe_dump(mapping, sort_keys=False, default_flow_style=False)


def convert_files(pairs: List[Tuple[Path, Path]]) -> None:
    """Convert legacy files to YAML on disk, without the deprecation warning.

    Parameters
    ----------
    pairs : list of (Path, Path)
        ``(source .cfg, destination .yaml)`` per file; destinations are overwritten.
    """
    for src, dst in pairs:
        dst.write_text(to_yaml(convert(src, warn=False)))
