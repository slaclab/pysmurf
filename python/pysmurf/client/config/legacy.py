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
#    is carried across under its new key; the per-band firmware delay triple
#    (refPhaseDelay, refPhaseDelayFine, lmsDelay) becomes the band's `delay`
#    block verbatim, so what setup() writes does not change. Keys nothing
#    read any more are dropped and named in a warning.
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

# Keys the old files carry that nothing has read for years. Named so a site
# learns its file had them, and so the round-trip check can assert they are gone.
DROPPED_KEYS = ('epics_root', 'chip_to_freq', 'smurf_to_mce', 'band_to_chip',
                'channel_assignment', 'flux_ramp.select_ramp',
                'tune_band.n_samples', 'tune_band.grad_cut', 'tune_band.amp_cut',
                'tune_band.freq_max', 'tune_band.freq_min', 'tune_band.eta_scan_amplitude')

# Old per-band key -> new one. The delay triple is handled apart.
_BAND_RENAMES = {
    'iq_swap_in': 'iq_swap_in', 'iq_swap_out': 'iq_swap_out',
    'feedbackEnable': 'feedback_enable', 'feedbackPolarity': 'feedback_polarity',
    'feedbackGain': 'feedback_gain', 'feedbackLimitkHz': 'feedback_limit_khz',
    'att_uc': 'att_uc', 'att_dc': 'att_dc', 'amplitude_scale': 'amplitude_scale',
    'data_out_mux': 'data_out_mux', 'trigRstDly': 'trigger_reset_delay',
    'lmsGain': 'lms_gain', 'bandDelayUs': 'band_delay_us',
}
_DELAY_KEYS = {'refPhaseDelay': 'ref_phase', 'refPhaseDelayFine': 'ref_phase_fine',
               'lmsDelay': 'lms'}
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
    if 'bad_mask' in old:
        # The old mapping's keys were labels nothing read; the ranges are the value.
        new.setdefault('wiring', {})['bad_mask'] = list(old['bad_mask'].values())
    for key in ('attenuator', 'amplifier', 'timing', 'fs', 'ultrascale_temperature_limit_degC'):
        take(old, key, key)
    if 'flux_ramp' in old:
        take(old['flux_ramp'], 'num_flux_ramp_counter_bits', 'flux_ramp', 'num_flux_ramp_counter_bits')
        if 'select_ramp' in old['flux_ramp']:
            dropped.append('flux_ramp.select_ramp')

    init = old.get('init', {})
    take(init, 'dspEnable', 'dsp_enable')
    bands: Dict[int, Dict[str, Any]] = {}
    for key, block in init.items():
        m = _BAND_BLOCK.match(key)
        if not m:
            # `bands` the old validator rebuilt from the band_# blocks present,
            # as the new schema does from the keys under `bands`.
            if key != 'dspEnable':
                dropped.append(f"init.{key}")
            continue
        band = bands.setdefault(int(m.group(1)), {})
        delay = {}
        for old_key, value in block.items():
            if old_key in _BAND_RENAMES:
                band[_BAND_RENAMES[old_key]] = value
            elif old_key in _DELAY_KEYS:
                delay[_DELAY_KEYS[old_key]] = value
            else:
                dropped.append(f"init.{key}.{old_key}")
        # The old setup() wrote the triple only when refPhaseDelay was set and
        # nonzero, and bandDelayUs otherwise; a triple it never wrote is dropped.
        if delay.get('ref_phase'):
            band['delay'] = delay
        else:
            dropped.extend(f"init.{key}.{old}" for old, new in _DELAY_KEYS.items() if new in delay)

    tune_band = old.get('tune_band', {})
    for key in ('fraction_full_scale', 'reset_rate_khz', 'default_tune'):
        take(tune_band, key, 'tune', key)
    for old_key, table in tune_band.items():
        if old_key in ('fraction_full_scale', 'reset_rate_khz', 'default_tune'):
            continue
        if old_key in _TUNING_RENAMES and isinstance(table, dict):
            for band_key, value in table.items():
                bands.setdefault(int(band_key), {})[_TUNING_RENAMES[old_key]] = value
        else:
            dropped.append(f"tune_band.{old_key}")
    if bands:
        new['bands'] = dict(sorted(bands.items()))

    for key in old:
        if key in DROPPED_KEYS:
            dropped.append(key)
    if warn:
        gone = (f" Dropped, nothing reads them: {', '.join(sorted(set(dropped)))}"
                if dropped else '')
        warnings.warn(f"{path}: the JSON .cfg format is deprecated; write it as YAML with "
                      f"`python -m pysmurf.client.config convert {path}`.{gone}",
                      DeprecationWarning, stacklevel=2)
    return new


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
