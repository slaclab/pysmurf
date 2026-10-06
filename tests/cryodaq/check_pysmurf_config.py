#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : pysmurf Configuration Schema Checks
#-----------------------------------------------------------------------------
# File       : check_pysmurf_config.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
# Checks of pysmurf's client configuration content -- the schema, the shipped
# default and the legacy converter -- as opposed to the layering machinery,
# which check_config_resolves.py covers with a synthetic schema.
#
# The schema's refusals are checked one by one, each required to name the key
# at fault. Then every legacy .cfg in cfg_files/ is converted, resolved over the
# shipped default and turned into the client's configuration properties, and
# those are compared value for value with what the legacy loader and property
# mixin produced from the same file. The legacy code is read out of git history
# at the revision before it was replaced, so this comparison keeps working after
# the files are gone; it is the proof that no site's setup() sees a different
# value on the day the format changes.
#
# Needs PyYAML, schema and numpy: the property shapes are numpy arrays, and
# that is part of what is compared.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
"""Check pysmurf's configuration schema, default and legacy converter."""
import argparse
import copy
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / 'python'))

from cryodaq import ConfigError  # noqa: E402
from cryodaq.config import VALIDATED_LAYER, flatten  # noqa: E402
from pysmurf.client.config import DEFAULT, legacy, load, load_mapping, schema  # noqa: E402

# The legacy loader and property mixin, at the last revision that had them.
LEGACY_REVISION = '2ef76397'
LEGACY_FILES = ('python/pysmurf/client/base/smurf_config.py',
                'python/pysmurf/client/base/smurf_config_properties.py')

# Every legacy configuration file; the startup scripts share the suffix and are
# not JSON. Fourteen of them the legacy validator itself refuses -- a required
# key missing, left behind when the schema grew under them -- so those are
# checked to be refused by the new schema too, and the round trip runs over
# the rest.
CFG_FILES = sorted(p for p in (REPO / 'cfg_files').rglob('*.cfg')
                   if 'smurf_startup' not in p.name)
EXPECTED_CFG_COUNT = 49
EXPECTED_UNLOADABLE = 14

# The properties the legacy mixin exposed that live code reads: what a site's
# setup(), tuning and analysis see. The eleven amplifier ones had no reader and
# are gone; the three deprecated delay ones are compared through `delay`.
COMPARED = (
    'smurf_cmd_dir', 'tune_dir', 'status_dir', 'default_data_dir', 'pA_per_phi0',
    'timing_reference', 'default_tune', 'fs', 'R_sh', 'dsp_enable',
    'ultrascale_temperature_limit_degC', 'bands', 'num_flux_ramp_counter_bits',
    'reset_rate_khz', 'fraction_full_scale', 'bias_line_resistance',
    'high_low_current_ratio', 'high_current_mode_bool', 'all_groups', 'n_bias_groups',
    'attenuator', 'pic_to_bias_group', 'bias_group_to_pair', 'bad_mask',
    'gradient_descent_gain', 'gradient_descent_averages', 'gradient_descent_converge_hz',
    'gradient_descent_step_hz', 'gradient_descent_momentum', 'gradient_descent_beta',
    'feedback_start_frac', 'feedback_end_frac', 'eta_scan_del_f', 'eta_scan_averages',
    'delta_freq', 'lms_freq_hz', 'data_out_mux', 'amplitude_scale', 'iq_swap_in',
    'iq_swap_out', 'ref_phase_delay', 'ref_phase_delay_fine', 'band_delay_us', 'att_uc',
    'att_dc', 'trigger_reset_delay', 'lms_gain', 'lms_delay', 'feedback_enable',
    'feedback_gain', 'feedback_limit_khz', 'feedback_polarity',
)


# --------------------------------------------------------------------------
# the legacy side
# --------------------------------------------------------------------------

def legacy_modules():
    """The old loader and mixin, executed out of git history into namespaces."""
    out = {}
    for rel in LEGACY_FILES:
        source = subprocess.run(['git', 'show', f"{LEGACY_REVISION}:{rel}"], cwd=REPO,
                                check=True, capture_output=True, text=True).stdout
        namespace = {'__name__': f"legacy_{pathlib.Path(rel).stem}"}
        exec(compile(source, rel, 'exec'), namespace)                    # noqa: S102
        out[pathlib.Path(rel).stem] = namespace
    return out


_LEGACY = None


def legacy_properties(cfg_path):
    """What the old code made of a .cfg: {property: value} for every compared name."""
    global _LEGACY
    if _LEGACY is None:
        _LEGACY = legacy_modules()
    SmurfConfig = _LEGACY['smurf_config']['SmurfConfig']
    Mixin = _LEGACY['smurf_config_properties']['SmurfConfigPropertiesMixin']
    # The old validator insists the data directories exist and are writable;
    # a fixture-shaped tree is enough for it, and the values compared are the
    # strings it was given.
    config = SmurfConfig(str(cfg_path), validate=False)
    config.config = _with_existing_dirs(config.config)
    config.config = SmurfConfig.validate_config(config.config)
    holder = Mixin()
    holder.copy_config_to_properties(config)
    return {name: getattr(holder, name) for name in COMPARED}, config.config


_DIRS = None


def _with_existing_dirs(raw):
    """The legacy validator's directory checks, satisfied without touching /data."""
    global _DIRS
    if _DIRS is None:
        _DIRS = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_legacy_dirs_'))
    raw = dict(raw)
    for key in ('default_data_dir', 'smurf_cmd_dir', 'tune_dir', 'status_dir'):
        d = _DIRS / key
        d.mkdir(exist_ok=True)
        raw[key] = str(d)
    if raw.get('tune_band', {}).get('default_tune'):
        f = _DIRS / 'tune.npy'
        f.touch()
        raw['tune_band'] = dict(raw['tune_band'], default_tune=str(f))
    return raw


# --------------------------------------------------------------------------
# the new side
# --------------------------------------------------------------------------

def new_properties(cfg_path, legacy_raw):
    """What the new code makes of the same file, as the same property names."""
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    converted = legacy.convert(cfg_path, warn=False)
    # The directory strings the legacy side was given, so paths compare equal.
    converted['paths'] = {'data': legacy_raw['default_data_dir'],
                          'smurf_cmd': legacy_raw['smurf_cmd_dir'],
                          'tune': legacy_raw['tune_dir'], 'status': legacy_raw['status_dir']}
    if legacy_raw['tune_band'].get('default_tune'):
        converted.setdefault('tune', {})['default_tune'] = legacy_raw['tune_band']['default_tune']
    resolved = load_mapping(converted, name=str(cfg_path))
    holder = SmurfConfigPropertiesMixin()
    holder.copy_config_to_properties(resolved)
    return {name: getattr(holder, name) for name in COMPARED}, resolved


def same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, float) or isinstance(b, float):
        return a == b or (a is not None and b is not None and abs(float(a) - float(b)) < 1e-12)
    return a == b


# --------------------------------------------------------------------------
# checks: the schema
# --------------------------------------------------------------------------

def minimal():
    """The smallest mapping the schema accepts: one band, every required key set."""
    return {
        'wiring': {'R_sh': 4e-4, 'bias_line_resistance': 1e4, 'high_low_current_ratio': 6.0,
                   'pic_to_bias_group': {0: 0}, 'bias_group_to_pair': {0: [1, 2]},
                   'all_bias_groups': [0]},
        'attenuator': {'att1': 0, 'att2': 1, 'att3': 2, 'att4': 3},
        'amplifier': {'hemt_Vg': -0.6, 'LNA_Vg': -0.7, 'bit_to_V_hemt': 1e-5,
                      'bit_to_V_50k': 1e-5, 'dac_num_50k': 32, 'hemt_Id_offset': 0.0,
                      '50k_Id_offset': 0.0, 'hemt_gate_min_voltage': -1.0,
                      'hemt_gate_max_voltage': 0.0},
        'flux_ramp': {'num_flux_ramp_counter_bits': 20},
        'timing': {'timing_reference': 'ext_ref'},
        'fs': 4000.0,
        'tune': {'fraction_full_scale': 0.5, 'reset_rate_khz': 4.0},
        'bands': {4: {'feedback_gain': 256, 'feedback_limit_khz': 225.0, 'att_uc': 12,
                      'att_dc': 0, 'amplitude_scale': 11, 'trigger_reset_delay': 60,
                      'lms_gain': 7, 'band_delay_us': 2.4, 'lms_freq_hz': 20000.0,
                      'delta_freq': 0.01, 'feedback_start_frac': 0.02,
                      'feedback_end_frac': 0.94, 'gradient_descent_gain': 0.05,
                      'gradient_descent_averages': 2, 'gradient_descent_converge_hz': 0.02,
                      'gradient_descent_momentum': 1, 'gradient_descent_step_hz': 0.005,
                      'gradient_descent_beta': 0.4, 'eta_scan_averages': 2,
                      'eta_scan_del_f': 40000}},
    }


def refused(values, key, saying=''):
    try:
        load_mapping(values)
    except schema.ConfigInvalid as e:
        assert e.key == key, f"refused at {e.key!r}, expected {key!r}: {e}"
        assert saying in str(e), f"{e} does not say {saying!r}"
        return e
    raise AssertionError(f"accepted a configuration wrong at {key}")


def check_the_default_alone_is_a_complete_template_that_needs_a_site():
    # Every key is present, and a resolution of the default alone is refused
    # for a required value rather than for shape: the template is complete.
    from cryodaq import config
    bare = config.load(DEFAULT)
    assert 'bands' in bare.values and bare.values['bands'] == {}
    for key in ('paths', 'wiring', 'attenuator', 'amplifier', 'flux_ramp', 'timing',
                'fs', 'dsp_enable', 'tune', 'band_default'):
        assert key in bare.values, f"default.yaml lacks {key}"
    try:
        schema.validate(dict(bare.values))
    except schema.ConfigInvalid as e:
        assert 'required' in str(e), str(e)
    else:
        raise AssertionError('the default alone was accepted; a site must set its wiring')


def check_a_minimal_configuration_resolves_with_defaults_filled():
    resolved = load_mapping(minimal())
    band = resolved.values['bands'][4]
    assert band['iq_swap_in'] == 0 and band['feedback_enable'] == 1, 'band_default not applied'
    assert band['data_out_mux'] == [2, 3], 'the firmware data_out_mux for band 4'
    assert resolved.values['wiring']['pA_per_phi0'] == 9e6
    assert resolved.values['amplifier']['hemt1']['drain_dac_num'] == 31
    assert resolved.values['paths']['tune'] == '/data/smurf_data/tune'
    assert list(resolved.values['bands']) == [4] and isinstance(list(resolved.values['bands'])[0], int)
    assert resolved.get('bands.4.att_uc') == 12, 'a dotted path must reach an int-keyed band'
    assert resolved.get('bands.5.att_uc', 'none') == 'none'
    assert resolved.provenance['wiring.R_sh'][0] == 'in-memory'
    assert resolved.provenance['wiring.pA_per_phi0'][0] == str(DEFAULT)


def check_the_schema_refuses_bad_content_naming_the_key():
    v = minimal()
    v['not_a_key'] = 1
    refused(v, 'not_a_key', 'not a pysmurf configuration key')
    v = minimal()
    v['bands'][4]['no_such'] = 1
    refused(v, 'bands.4.no_such', 'not a per-band key')
    v = minimal()
    del v['wiring']['R_sh']
    refused(v, 'wiring.R_sh', 'required')
    v = minimal()
    v['bands'][4]['att_uc'] = 40
    refused(v, 'bands.4.att_uc')
    v = minimal()
    v['timing']['timing_reference'] = 'moon'
    refused(v, 'timing.timing_reference')
    v = minimal()
    v['bands'][9] = v['bands'].pop(4)
    refused(v, 'bands.9', '0-7')
    v = minimal()
    v['bands'][4]['band_delay_us'] = None
    refused(v, 'bands.4', 'band_delay_us or a delay block')
    # A zero ref_phase is a triple setup() never wrote: the block is refused
    # rather than accepted and then passed over for band_delay_us.
    v = minimal()
    v['bands'][4]['delay'] = {'ref_phase': 0, 'ref_phase_fine': 0, 'lms': 0}
    refused(v, 'bands.4.delay.ref_phase')
    # The directories are names, not checked -- but an empty name is none.
    for key in ('data', 'smurf_cmd', 'tune', 'status'):
        v = minimal()
        v.setdefault('paths', {})[key] = ''
        refused(v, f'paths.{key}')
    v = minimal()
    v['wiring']['bias_group_to_pair'] = {0: [1, 2], 1: [2, 3]}
    refused(v, 'wiring.bias_group_to_pair', '[2]')
    v = minimal()
    v['wiring']['bias_group_to_pair'] = {0: [32, 2]}
    refused(v, 'amplifier.dac_num_50k', 'bias group 0')
    for pair in ([1], [1, 2, 3]):
        v = minimal()
        v['wiring']['bias_group_to_pair'] = {0: pair}
        refused(v, 'wiring.bias_group_to_pair.0', 'two DACs')
    # A band is an integer 0-7 however spelled; 4.5 is not band 4, True is not
    # band 1, and the same band under two spellings is not two bands.
    for key in (True, 'four', -1, None):
        v = minimal()
        v['bands'][key] = v['bands'].pop(4)
        refused(v, f"bands.{key}", '0-7')
    # A key with a dot in it never reaches the schema: the loader refuses it.
    for key in (4.5, '4.0'):
        v = minimal()
        v['bands'][key] = v['bands'].pop(4)
        try:
            load_mapping(v)
        except ConfigError as e:
            assert e.key == f"bands.{key}" and "'.'" in str(e), e
        else:
            raise AssertionError(f"bands.{key} was accepted")
    v = minimal()
    v['bands']['4'] = dict(v['bands'][4], att_uc=1)
    refused(v, 'bands.4', 'twice')
    v = minimal()
    v['bands']['4'] = v['bands'].pop(4)
    assert load_mapping(v).values['bands'][4]['att_uc'] == 12, 'a digit string names its band'
    v = minimal()
    v['wiring']['pic_to_bias_group'] = {0: 0, '0': 1}
    refused(v, 'wiring.pic_to_bias_group.0', 'twice')
    v = minimal()
    v['wiring']['bias_group_to_pair'] = {True: [1, 2]}
    refused(v, 'wiring.bias_group_to_pair.True', '0-15')
    v = minimal()
    v['wiring']['pic_to_bias_group'] = {16: 0}
    refused(v, 'wiring.pic_to_bias_group.16', '0-15')
    v = minimal()
    v['wiring']['bias_group_to_pair'] = {0: [1, 33]}
    refused(v, 'wiring.bias_group_to_pair.0', 'two DACs')
    v = minimal()
    v['wiring']['all_bias_groups'] = [0, 0]
    refused(v, 'wiring.all_bias_groups', 'twice')
    # A number is a number: not a bool, not a string, not nan or inf.
    for bad, saying in ((True, 'a number'), ('4000', 'a number'), (float('inf'), 'finite'),
                        (float('nan'), 'finite')):
        v = minimal()
        v['fs'] = bad
        refused(v, 'fs', saying)
    v = minimal()
    v['attenuator']['att1'] = True
    refused(v, 'attenuator.att1')
    v = minimal()
    v['bands'][4]['eta_scan_averages'] = 2.0
    refused(v, 'bands.4.eta_scan_averages')
    v = minimal()
    v['tune']['default_tune'] = ''
    refused(v, 'tune.default_tune')
    # A falsey value of the wrong type is refused, not read as "nothing here".
    for key, bad in (('bands', []), ('bands', 0), ('band_default', []), ('band_default', '')):
        v = minimal()
        v[key] = bad
        refused(v, key, 'a mapping')
    v = minimal()
    v['bands'][4] = []
    refused(v, 'bands.4', 'a mapping')
    v = minimal()
    v['flux_ramp']['num_flux_ramp_counter_bits'] = 20.0
    refused(v, 'flux_ramp.num_flux_ramp_counter_bits')


def check_a_delay_block_is_accepted_and_wins_over_band_delay_us():
    v = minimal()
    v['bands'][4]['delay'] = {'ref_phase': 6, 'lms': 24}
    band = load_mapping(v).values['bands'][4]
    assert band['delay'] == {'ref_phase': 6, 'ref_phase_fine': 0, 'lms': 24}
    assert band['band_delay_us'] == 2.4, 'both are kept; setup() prefers delay'
    v = minimal()
    v['bands'][4]['band_delay_us'] = None
    v['bands'][4]['delay'] = {'ref_phase': 6}
    band = load_mapping(v).values['bands'][4]
    assert band['delay']['lms'] is None, 'lms defaults to none, meaning "same as ref_phase"'


def check_a_layered_site_file_resolves_over_the_default():
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_pysmurf_cfg_'))
    site = copy.deepcopy(minimal())
    band = site['bands'].pop(4)
    site['band_default'] = {k: v for k, v in band.items() if k not in ('att_uc', 'att_dc')}
    (d / 'site.yaml').write_text(legacy.to_yaml(site))
    (d / 'slot.yaml').write_text('inherit: site.yaml\nbands:\n  4: {att_uc: 12, att_dc: 0}\n'
                                 '  5: {att_uc: 14, att_dc: 2}\n')
    resolved = load(d / 'slot.yaml')
    assert sorted(resolved.values['bands']) == [4, 5]
    assert resolved.values['bands'][5]['att_uc'] == 14
    assert resolved.values['bands'][5]['lms_gain'] == 7, 'from band_default'
    assert resolved.provenance['bands.5.att_uc'][0].endswith('slot.yaml')
    assert resolved.provenance['wiring.R_sh'][0].endswith('site.yaml')
    # Every leaf has provenance: a value copied from band_default is credited to
    # the line that set the default, and one the schema filled in says so.
    assert set(resolved.provenance) == set(flatten(resolved.values))
    assert resolved.provenance['bands.5.lms_gain'] == resolved.provenance['band_default.lms_gain']
    assert resolved.provenance['bands.5.lms_gain'][0].endswith('site.yaml')
    assert resolved.provenance['bands.5.data_out_mux'] == (VALIDATED_LAYER, 0), \
        'the firmware default for data_out_mux came from the validator'
    # A layering fault is cryodaq's to refuse, unchanged by the schema.
    (d / 'loop.yaml').write_text('inherit: loop.yaml\n')
    try:
        load(d / 'loop.yaml')
    except ConfigError as e:
        assert 'loops' in str(e)
    else:
        raise AssertionError('a loop was accepted')


# --------------------------------------------------------------------------
# checks: the legacy files
# --------------------------------------------------------------------------

def check_every_legacy_file_converts_to_what_the_old_code_read():
    assert len(CFG_FILES) == EXPECTED_CFG_COUNT, \
        f"{len(CFG_FILES)} legacy files, expected {EXPECTED_CFG_COUNT}"
    failures = []
    unloadable = []
    for cfg in CFG_FILES:
        try:
            old, raw = legacy_properties(cfg)
        except Exception as e:                                   # noqa: BLE001
            # The legacy loader refuses it; the new one must too, naming a key.
            unloadable.append(cfg.name)
            try:
                load_mapping(legacy.convert(cfg, warn=False))
            except schema.ConfigInvalid as new_e:
                assert new_e.key, f"{cfg.name}: refused without a key"
            else:
                raise AssertionError(f"{cfg.name}: the legacy loader refuses this file "
                                     f"({str(e)[:60]!r}) and the new one accepts it")
            continue
        new, resolved = new_properties(cfg, raw)
        for name in COMPARED:
            if not same(old[name], new[name]):
                failures.append(f"{cfg.relative_to(REPO)}: {name}: old {old[name]!r} new {new[name]!r}")
        # Every band block resolves to the same delay decision setup() takes.
        for band in old['bands']:
            assert _delay_writes_old(old, band) == _delay_writes_new(resolved, band), \
                f"{cfg.name} band {band}: delay writes differ"
    assert not failures, f"{len(failures)} value(s) differ:\n  " + '\n  '.join(failures[:20])
    assert len(unloadable) == EXPECTED_UNLOADABLE, \
        f"{len(unloadable)} files the legacy loader refuses, expected {EXPECTED_UNLOADABLE}: {unloadable}"


def _delay_writes_old(old, band):
    """The registers setup() wrote from the legacy properties, as (name, value) pairs."""
    if old['ref_phase_delay'][band]:
        lms = old['lms_delay'][band]
        return (('ref_phase_delay', old['ref_phase_delay'][band]),
                ('ref_phase_delay_fine', old['ref_phase_delay_fine'][band]),
                ('lms_delay', int(old['ref_phase_delay'][band]) if lms is None else lms))
    return (('band_delay_us', old['band_delay_us'][band]),)


def _delay_writes_new(resolved, band):
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    holder = SmurfConfigPropertiesMixin()
    holder.copy_config_to_properties(resolved)
    return holder.delay_writes(band)


def check_a_record_read_back_through_json_gives_the_same_properties():
    # What the crate found: a Resolved recorded on the server or on disk
    # travels as JSON, which has no integer keys, so every per-band and per-group
    # table came back keyed by string and five properties disagreed with the
    # file-driven instance's. adopt() re-validates and the properties must agree.
    import json
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    from pysmurf.client.config import adopt
    from cryodaq import Resolved
    m = minimal()
    # Keys past 9, so that a string sort ('10' < '2') would reorder the rows: the
    # second thing the crate found, after the key type.
    m['wiring']['pic_to_bias_group'] = {0: 0, 2: 1, 10: 2, 11: 3}
    m['wiring']['bias_group_to_pair'] = {0: [1, 2], 2: [3, 4], 10: [5, 6], 11: [7, 8]}
    m['wiring']['all_bias_groups'] = [0, 2, 10, 11]
    original = load_mapping(m)
    # The file and the server record are both written with sort_keys=True.
    travelled = Resolved.from_dict(json.loads(json.dumps(original.to_dict(), sort_keys=True)))
    assert list(travelled.values['bands']) == ['4'], 'JSON did not stringify the band key; the case is moot'
    assert list(travelled.values['wiring']['pic_to_bias_group']) == ['0', '10', '11', '2'], \
        'the record did not come back string-sorted; the case is moot'
    adopted = adopt(travelled)
    assert adopted.hash == original.hash
    a, b = SmurfConfigPropertiesMixin(), SmurfConfigPropertiesMixin()
    a.copy_config_to_properties(original)
    b.copy_config_to_properties(adopted)
    differ = [n for n in COMPARED if hasattr(a, n) and not same(getattr(a, n), getattr(b, n))]
    assert not differ, f"properties differ after a JSON round trip: {differ}"
    assert list(adopted.values['bands']) == [4] and isinstance(list(adopted.values['bands'])[0], int)


def check_a_reattached_client_records_under_the_adopted_status_directory():
    # A no-file client opens its session before it has a configuration, so the
    # session's paths are the packaged default's; the configuration it adopts
    # from the server may put `paths.status` elsewhere, and the record its own
    # later setup() writes has to go there. Driven on a stand-in session: the
    # resolution is what matters, not the transport.
    import cryodaq
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    from pysmurf.client.base.smurf_control import SmurfControl
    site = minimal()
    site['paths'] = {'status': '/elsewhere/status'}
    adopted = load_mapping(site)
    default_status = load_mapping(minimal()).values['paths']['status']
    assert default_status != '/elsewhere/status', 'the case needs the site to move the directory'

    class _Session:
        endpoint = 'stand-in:9012'

        def __init__(self):
            self.paths = cryodaq.Paths.under('/data')

        def resolved_config(self):
            return adopted

    S = SmurfControl.__new__(SmurfControl)
    S._session = _Session()
    S.log = lambda *a, **k: None
    SmurfConfigPropertiesMixin.__init__(S)               # the None tables, no connection
    assert str(S._session_paths().status) == default_status, 'before: the default'
    S._reattach()
    assert S.status_dir == '/elsewhere/status'
    assert str(S._session.paths.status) == '/elsewhere/status', \
        f"the session still records under {S._session.paths.status}"


def check_a_refused_reattach_closes_the_session_it_opened():
    # The constructor opens the session before it knows whether the server is
    # configured. When the reattach fails -- no record, a record the server
    # cannot hand back, or one that does not re-validate -- no instance is
    # returned, so nothing could close that session later: the refusal has to,
    # or the client stays in the process's open-client table until exit.
    import json
    import cryodaq
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    from pysmurf.client.base.smurf_control import SmurfControl

    good = load_mapping(minimal())
    tampered = cryodaq.Resolved(dict(good.values), good.provenance, 'not-the-hash', good.layers)

    class _Session:
        endpoint = 'stand-in:9012'

        def __init__(self, answer):
            self.paths = cryodaq.Paths.under('/data')
            self.answer, self.closed = answer, 0

        def resolved_config(self):
            if isinstance(self.answer, Exception):
                raise self.answer
            return self.answer

        def close(self):
            self.closed += 1

    cases = (
        ('no record', None, RuntimeError, 'not configured'),
        ('a record the server cannot hand back', json.JSONDecodeError('truncated', '{', 1),
         ValueError, 'truncated'),
        ('a record that does not re-validate', tampered, cryodaq.ConfigError, 'changed the hash'),
    )
    for label, answer, kind, said in cases:
        S = SmurfControl.__new__(SmurfControl)
        S._session = _Session(answer)
        S.log = lambda *a, **k: None
        SmurfConfigPropertiesMixin.__init__(S)
        try:
            S._reattach()
        except kind as e:
            assert said in str(e), f"{label}: {e}"
        else:
            raise AssertionError(f"{label}: adopted")
        assert S._session.closed == 1, f"{label}: close() called {S._session.closed} times"
        assert S.config is None, f"{label}: a refusal left a configuration behind"

    # And the happy path does not close what it is about to use.
    S = SmurfControl.__new__(SmurfControl)
    S._session = _Session(good)
    S.log = lambda *a, **k: None
    SmurfConfigPropertiesMixin.__init__(S)
    S._reattach()
    assert S._session.closed == 0 and S.config is not None, 'a successful reattach closed its session'


def check_setup_resumes_hardware_logging_whatever_happens():
    # setup() pauses the hardware-logging thread for the duration. Anything
    # raising in between -- including the record written at the end -- must
    # still resume it, or a failed setup leaves the hardware unlogged for good.
    from pysmurf.client.base.smurf_control import SmurfControl

    class Boom(Exception):
        pass

    S = SmurfControl.__new__(SmurfControl)
    calls = []
    S.log = lambda *a, **k: None
    S._hardware_logging_thread = object()
    S.pause_hardware_logging = lambda: calls.append('pause')
    S.resume_hardware_logging = lambda: calls.append('resume')
    S.get_system_configured = lambda: False
    S._session = type('_S', (), {'clear_config': lambda self: calls.append('clear')})()

    def body(write_log, payload_size, **kw):
        calls.append('body')
        raise Boom()
    S._setup_hardware = body
    try:
        S.setup()
    except Boom:
        pass
    else:
        raise AssertionError('the failure was swallowed')
    assert calls == ['clear', 'pause', 'body', 'resume'], calls
    # And a body that returns hands its verdict through, still resuming.
    del calls[:]
    S._setup_hardware = lambda write_log, payload_size, **kw: False
    assert S.setup() is False
    assert calls == ['clear', 'pause', 'resume'], calls


def check_a_write_into_a_per_band_property_persists():
    # Callers write into these dictionaries -- tracking_setup stores the LMS
    # frequency it measured, sodetlib the tone power it chose -- and read the
    # value back later. A property rebuilt on each access takes the write on a
    # temporary and loses it; review found exactly that at smurf_tune.py's
    # `self.lms_freq_hz[band] = lms_freq_hz`. Every per-band property must hand
    # out the same dictionary each time.
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    holder = SmurfConfigPropertiesMixin()
    holder.copy_config_to_properties(load_mapping(minimal()))
    # The per-band properties are found, not listed: whatever answers with a
    # dict keyed by band is one, so a new one is covered without an edit here.
    def is_per_band(n):
        return (not n.startswith('_') and isinstance(getattr(type(holder), n, None), property) and
                isinstance(getattr(holder, n), dict) and 4 in getattr(holder, n))
    per_band = [n for n in dir(holder) if is_per_band(n)]
    assert len(per_band) >= 25, f"only {len(per_band)} per-band properties found: {per_band}"
    lost = []
    for name in per_band:
        table = getattr(holder, name)
        assert isinstance(table, dict) and 4 in table, f"{name} is not a per-band dict: {table!r}"
        table[4] = 'written'
        if getattr(holder, name)[4] != 'written' or getattr(holder, name) is not table:
            lost.append(name)
    assert not lost, f"a write into these per-band properties is lost: {lost}"
    # And the one sodetlib reads back after tracking_setup, by name, the way it is written.
    holder.lms_freq_hz[4] = 12345.0
    assert holder.lms_freq_hz[4] == 12345.0
    assert holder._amplitude_scale is holder.amplitude_scale, 'the private field is the same dict'


def check_setup_writes_the_live_delay_a_caller_set():
    # The old setup() read the delay from the mutable properties, as it reads
    # every other per-band value; `S.band_delay_us[0] = 8.8` before setup()
    # wrote 8.8. The delay writes are decided from the same tables, so a value
    # changed on the instance is what reaches the firmware -- and switching
    # representation works the way it did: a nonzero ref_phase_delay selects the
    # triple, zero selects band_delay_us.
    from pysmurf.client.base.smurf_config_properties import SmurfConfigPropertiesMixin
    holder = SmurfConfigPropertiesMixin()
    holder.copy_config_to_properties(load_mapping(minimal()))
    before = holder.delay_writes(4)
    assert before[0][0] == 'band_delay_us', before
    holder.band_delay_us[4] = 8.8
    assert holder.delay_writes(4) == (('band_delay_us', 8.8),), 'the live edit was not written'
    holder.ref_phase_delay[4] = 6
    holder.ref_phase_delay_fine[4] = 1
    assert holder.delay_writes(4) == (('ref_phase_delay', 6), ('ref_phase_delay_fine', 1), ('lms_delay', 6)), \
        'a ref_phase_delay set on the instance did not select the triple'
    holder.lms_delay[4] = 24
    assert holder.delay_writes(4)[2] == ('lms_delay', 24)


def check_the_converter_drops_only_what_nothing_read():
    seen = set()
    for cfg in CFG_FILES:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            converted = legacy.convert(cfg)
        assert len(caught) == 1, f"{cfg.name}: {len(caught)} warnings, expected one"
        text = str(caught[0].message)
        assert 'deprecated' in text and cfg.name in text, text
        if 'Dropped, nothing reads them: ' in text:
            seen.update(text.split('Dropped, nothing reads them: ', 1)[1].rstrip('.').split(', '))
        for key in ('epics_root', 'chip_to_freq', 'smurf_to_mce'):
            assert key not in converted, f"{cfg.name} kept {key}"
    allowed = set(legacy.DROPPED_KEYS)
    unexpected = {k for k in seen if k not in allowed and not k.startswith('init.band_')}
    assert not unexpected, f"the converter dropped keys not on its list: {sorted(unexpected)}"
    # The files that carry init.bands are told so; the old validator rebuilt it anyway.
    assert 'init.bands' in seen, 'init.bands was dropped without being named'


def check_the_converter_keeps_the_old_delay_decision_for_a_zero_ref_phase_delay():
    """No shipped file sets ``refPhaseDelay: 0``; a site's might, and the old setup() took it as "use bandDelayUs"."""
    base = legacy.read_json_with_comments(CFG_FILES[0])
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_cfg_'))
    block = base['init']['band_0']
    block.update(refPhaseDelay=0, refPhaseDelayFine=0, lmsDelay=0, bandDelayUs=8.8)
    (d / 'zero.cfg').write_text(json.dumps(base))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        converted = legacy.convert(d / 'zero.cfg')
    assert 'delay' not in converted['bands'][0], 'a triple the old setup() never wrote was kept'
    assert converted['bands'][0]['band_delay_us'] == 8.8
    assert 'init.band_0.refPhaseDelay' in str(caught[0].message), 'the drop was not named'
    assert _delay_writes_new(load_mapping(converted), 0) == (('band_delay_us', 8.8),)
    # Nonzero stays the direct triple, as the shipped files are checked for above.
    block.update(refPhaseDelay=6)
    (d / 'six.cfg').write_text(json.dumps(base))
    converted = legacy.convert(d / 'six.cfg', warn=False)
    assert converted['bands'][0]['delay'] == {'ref_phase': 6, 'ref_phase_fine': 0, 'lms': 0}


def check_the_converter_reads_a_hash_inside_a_string():
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_cfg_'))
    (d / 'x.cfg').write_text('{\n  # a comment\n  "a": "with # inside",  # trailing\n  "b": 1\n}\n')
    assert legacy.read_json_with_comments(d / 'x.cfg') == {'a': 'with # inside', 'b': 1}


def check_a_converted_file_written_as_yaml_resolves_to_the_same_hash():
    cfg = CFG_FILES[0]
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_cfg_'))
    (d / 'site.yaml').write_text(legacy.to_yaml(legacy.convert(cfg, warn=False)))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        via_cfg = load(cfg)
    via_yaml = load(d / 'site.yaml')
    assert via_cfg.hash == via_yaml.hash, 'the written YAML resolves differently from the .cfg'


# The legacy loader exposed the raw .cfg as `S.config.get(section)` /
# `S.config[section]` with these section names; `S.config` is a Resolved now, with
# `values`, `provenance`, `hash`, `layers` and a dotted `get`. A reader of the old
# shape passes every other gate -- the request-equivalence proof drives methods,
# not scripts -- and fails at the user's prompt, as smurf_cmd.py did.
_LEGACY_SECTIONS = ('init', 'tune_band', 'amplifier', 'attenuator', 'pic_to_bias_group',
                    'bias_group_to_pair', 'constant', 'timing', 'flux_ramp', 'smurf_to_mce',
                    'bad_mask', 'epics_root', 'default_data_dir', 'tune_dir', 'status_dir',
                    'smurf_cmd_dir', 'all_bias_groups', 'high_low_current_ratio',
                    'bias_line_resistance', 'high_current_mode_bool', 'chip_to_freq')
_LEGACY_READ = re.compile(r"\.config(?:\.get\(|\[)\s*['\"](" + '|'.join(_LEGACY_SECTIONS) + r")['\"]")
_RESOLVED_ATTRS = ('values', 'provenance', 'hash', 'layers', 'get', 'to_dict')
_CONFIG_ATTR = re.compile(r"\bself\.config\.([A-Za-z_]+)")
# Legacy config module and the converter (which names the old keys on purpose).
_EXEMPT = ('client/config/legacy.py',)


def check_no_code_reads_the_legacy_config_shape():
    hits = []
    for path in sorted((REPO / 'python' / 'pysmurf').rglob('*.py')):
        rel = path.relative_to(REPO / 'python' / 'pysmurf').as_posix()
        if rel in _EXEMPT:
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if _LEGACY_READ.search(line):
                hits.append(f"{rel}:{number}: {line.strip()}")
            for m in _CONFIG_ATTR.finditer(line):
                if m.group(1) not in _RESOLVED_ATTRS:
                    hits.append(f"{rel}:{number}: self.config.{m.group(1)} is not a Resolved attribute")
    assert not hits, 'readers of the legacy config shape:\n  ' + '\n  '.join(hits)


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items() if name.startswith('check_'))
    failed = []
    print(f"Checking the pysmurf configuration schema ({len(checks)} checks, "
          f"{len(CFG_FILES)} legacy files)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            failed.append(label)
            print(f"  FAIL  {label}")
            print(f"          {type(e).__name__}: {str(e)[:600]}")
        else:
            print(f"  ok    {label}")
    print("")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
