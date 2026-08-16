#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : SmurfConfig Cryostat-Card Cfg Validation Script
#-----------------------------------------------------------------------------
# File       : validate_cryostat_card_config.py
# Created    : 2026-05-01
#-----------------------------------------------------------------------------
# Description:
# Validates the cryostat-card cfg merge pass added for issue #86: a
# separate `Rcable` cfg variable plus a `cryostat_card_config_file`
# pointer that pulls card-specific values out of a shared cfg file.
#
# Exercises four scenarios:
#   1. Legacy path: an experiment cfg that sets `bias_line_resistance`
#      directly continues to load unchanged.
#   2. New path: an experiment cfg that sets `Rcable` and references a
#      card cfg via `cryostat_card_config_file` resolves
#      `bias_line_resistance = Rcable + R_cryostat_card`.
#   3. Override: when both legacy and new keys are present, the
#      explicit `bias_line_resistance` wins.
#   4. Missing: a cfg that references a card cfg but supplies neither
#      `Rcable` nor `bias_line_resistance` fails schema validation.
#
# Runs without hardware.  Exit 0 on pass.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level
# directory of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may
# be copied, modified, propagated, or distributed except according to
# the terms contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import importlib.util
import json
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..'))


def _load_smurf_config_class():
    """Import the SmurfConfig class without triggering pysmurf.client's
    eager import chain (which depends on a working pyrogue install).
    """
    module_path = os.path.join(
        REPO_ROOT, 'python', 'pysmurf', 'client', 'base', 'smurf_config.py')
    spec = importlib.util.spec_from_file_location(
        'pysmurf_smurf_config_under_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmurfConfig


SmurfConfig = _load_smurf_config_class()
TEMPLATE_CFG = os.path.join(REPO_ROOT, 'cfg_files', 'template', 'template.cfg')
CARD_CFG = os.path.join(
    REPO_ROOT, 'cfg_files', 'cryostat_cards', 'cc02-template.cfg')


def _load_template_dict(tmpdir):
    """Read the shipped template cfg and rewrite filesystem-validated
    directory keys to writable subdirs of ``tmpdir`` so the cfg can
    validate on hosts without ``/data/smurf_data``.
    """
    cfg_dict = SmurfConfig.read_json(TEMPLATE_CFG)
    for key in ('default_data_dir', 'smurf_cmd_dir', 'tune_dir',
                'status_dir'):
        sub = os.path.join(tmpdir, key)
        os.makedirs(sub, exist_ok=True)
        cfg_dict[key] = sub
    return cfg_dict


def _write_cfg(path, cfg_dict):
    """Write a cfg dict to ``path`` as JSON."""
    with open(path, 'w') as f:
        json.dump(cfg_dict, f, indent=2)


def test_legacy_path(tmpdir):
    """Legacy `bias_line_resistance` cfgs continue to load unchanged."""
    cfg_dict = _load_template_dict(tmpdir)
    cfg_path = os.path.join(tmpdir, 'experiment_legacy.cfg')
    _write_cfg(cfg_path, cfg_dict)

    cfg = SmurfConfig(cfg_path)
    assert cfg.config['bias_line_resistance'] == 15800, (
        f"expected 15800, got {cfg.config['bias_line_resistance']}")
    assert cfg.config['high_low_current_ratio'] == 6.08, (
        f"expected 6.08, got {cfg.config['high_low_current_ratio']}")
    assert 'pic_to_bias_group' in cfg.config
    assert 'bias_group_to_pair' in cfg.config
    print('PASS  legacy_path: template-style cfg loads with '
          'bias_line_resistance=15800')


def test_new_path(tmpdir):
    """`Rcable` + card cfg → bias_line_resistance is computed."""
    cfg_dict = _load_template_dict(tmpdir)
    # Strip the keys that should now flow in from the card cfg.
    for key in ('bias_line_resistance', 'high_low_current_ratio',
                'pic_to_bias_group', 'bias_group_to_pair'):
        cfg_dict.pop(key, None)
    cfg_dict['Rcable'] = 100.0
    cfg_dict['cryostat_card_config_file'] = CARD_CFG  # absolute path

    new_cfg_path = os.path.join(tmpdir, 'experiment_new_path.cfg')
    _write_cfg(new_cfg_path, cfg_dict)

    cfg = SmurfConfig(new_cfg_path)
    expected = 100.0 + 15800.0
    assert cfg.config['bias_line_resistance'] == expected, (
        f"expected {expected}, got {cfg.config['bias_line_resistance']}")
    assert cfg.config['high_low_current_ratio'] == 6.08
    assert cfg.config['pic_to_bias_group']
    assert cfg.config['bias_group_to_pair']
    assert cfg.config['Rcable'] == 100.0
    print('PASS  new_path: Rcable + R_cryostat_card sums to '
          f'bias_line_resistance={expected}')


def test_override_path(tmpdir):
    """Explicit `bias_line_resistance` wins over Rcable + card cfg."""
    cfg_dict = _load_template_dict(tmpdir)
    cfg_dict['bias_line_resistance'] = 99999.0  # explicit override
    cfg_dict['Rcable'] = 100.0
    cfg_dict['cryostat_card_config_file'] = CARD_CFG

    cfg_path = os.path.join(tmpdir, 'experiment_override.cfg')
    _write_cfg(cfg_path, cfg_dict)

    cfg = SmurfConfig(cfg_path)
    assert cfg.config['bias_line_resistance'] == 99999.0, (
        f"explicit override expected 99999.0, got "
        f"{cfg.config['bias_line_resistance']}")
    print('PASS  override_path: legacy bias_line_resistance=99999 '
          'wins over Rcable+card cfg')


def test_missing_path(tmpdir):
    """Card cfg without Rcable or bias_line_resistance → validation fails."""
    cfg_dict = _load_template_dict(tmpdir)
    for key in ('bias_line_resistance', 'high_low_current_ratio',
                'pic_to_bias_group', 'bias_group_to_pair'):
        cfg_dict.pop(key, None)
    cfg_dict['cryostat_card_config_file'] = CARD_CFG
    # Note: no Rcable, no bias_line_resistance.

    cfg_path = os.path.join(tmpdir, 'experiment_missing.cfg')
    _write_cfg(cfg_path, cfg_dict)

    raised = False
    try:
        SmurfConfig(cfg_path)
    except Exception as exc:  # noqa: BLE001 - schema raises SchemaError
        raised = True
        msg = str(exc)
        assert 'bias_line_resistance' in msg, (
            f"expected schema error to mention bias_line_resistance, "
            f"got: {msg}")
    assert raised, 'expected SmurfConfig to raise without bias_line_resistance'
    print('PASS  missing_path: schema rejects cfg with neither Rcable '
          'nor bias_line_resistance')


def test_relative_card_path(tmpdir):
    """`cryostat_card_config_file` is resolved relative to main cfg dir."""
    # Put a copy of the card cfg next to the experiment cfg so a bare
    # filename resolves correctly.
    local_card = os.path.join(tmpdir, 'cc02-template.cfg')
    shutil.copy(CARD_CFG, local_card)

    cfg_dict = _load_template_dict(tmpdir)
    for key in ('bias_line_resistance', 'high_low_current_ratio',
                'pic_to_bias_group', 'bias_group_to_pair'):
        cfg_dict.pop(key, None)
    cfg_dict['Rcable'] = 50.0
    cfg_dict['cryostat_card_config_file'] = 'cc02-template.cfg'  # relative

    cfg_path = os.path.join(tmpdir, 'experiment_relative.cfg')
    _write_cfg(cfg_path, cfg_dict)

    cfg = SmurfConfig(cfg_path)
    assert cfg.config['bias_line_resistance'] == 50.0 + 15800.0
    print('PASS  relative_card_path: relative card cfg path resolved '
          'against main cfg dir')


def main():
    failures = []
    tests = [test_legacy_path, test_new_path, test_override_path,
             test_missing_path, test_relative_card_path]
    for fn in tests:
        # Each test gets its own tempdir so tmpdir-rooted paths don't
        # collide between tests (e.g. each writes a different cfg file
        # into the same dir, but more importantly cached state from one
        # test's cfg file doesn't leak into the next).
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                fn(tmpdir)
            except Exception as exc:  # noqa: BLE001
                failures.append((fn.__name__, repr(exc)))
                print(f'FAIL  {fn.__name__}: {exc!r}')

    if failures:
        print(f'\n{len(failures)} test(s) failed:')
        for name, err in failures:
            print(f'  - {name}: {err}')
        sys.exit(1)

    print('\nAll cryostat-card cfg tests passed.')


if __name__ == '__main__':
    main()
