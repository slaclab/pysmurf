"""Offline-mode state-dump tests for issue #364.

These tests do not require a live rogue server, hardware, or a pysmurf
configuration file.  They exercise the path:

    SmurfBase(offline=True, state_file=...) -> _caget() -> tools.state_lookup

and verify that the per-method hard-coded offline returns yield to the
loaded state dump where one is provided.
"""
import gzip
import os
import tempfile

import numpy as np
import pytest
import yaml

from pysmurf.client.command.smurf_command import SmurfCommandMixin
from pysmurf.client.util import tools


def _state_dump(values):
    """Build a nested dict mirroring the rogue tree under
    ``AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.Base[band]``.

    ``values`` is ``{band_index: {register_name: value}}``.
    """
    base = {}
    for band, regs in values.items():
        base[f'Base[{band}]'] = dict(regs)
    return {
        'AMCc': {
            'FpgaTopLevel': {
                'AppTop': {
                    'AppCore': {
                        'SysgenCryo': base,
                    },
                },
            },
        },
    }


def _write_yaml(state, suffix='.yml'):
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    if suffix.endswith('.gz'):
        with gzip.open(path, 'wt') as f:
            yaml.safe_dump(state, f)
    else:
        with open(path, 'w') as f:
            yaml.safe_dump(state, f)
    return path


def _silent_log(*_, **__):
    pass


def _make_offline(state_file=None):
    """Construct a bare offline SmurfCommandMixin without needing a cfg."""
    S = SmurfCommandMixin(offline=True, state_file=state_file,
                          log=_silent_log)
    # _bands is normally populated by SmurfConfigPropertiesMixin; for the
    # band-aware getters we only need a populated default.
    S._bands = [0, 1, 2, 3]
    return S


# -- tools helpers ---------------------------------------------------------

def test_state_lookup_walks_dotted_path():
    state = _state_dump({0: {'numberChannels': 528}})
    val = tools.state_lookup(
        state,
        'AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.Base[0].numberChannels')
    assert val == 528


def test_state_lookup_missing_returns_default():
    state = _state_dump({0: {'numberChannels': 528}})
    assert tools.state_lookup(state, 'AMCc.NotPresent') is None
    assert tools.state_lookup(state, 'AMCc.NotPresent', default='x') == 'x'
    assert tools.state_lookup(None, 'AMCc.Anything') is None


def test_load_state_yaml_handles_gzip(tmp_path):
    state = _state_dump({0: {'numberChannels': 528}})
    plain = _write_yaml(state, suffix='.yml')
    gz = _write_yaml(state, suffix='.yml.gz')
    try:
        assert tools.load_state_yaml(plain) == state
        assert tools.load_state_yaml(gz) == state
    finally:
        os.unlink(plain)
        os.unlink(gz)


# -- offline _caget -------------------------------------------------------

def test_caget_returns_none_when_no_state_dump_loaded():
    S = _make_offline()
    assert S._offline_state is None
    assert S._caget('AMCc.SomeRegister') is None


def test_caget_reads_from_state_dump_when_loaded():
    state = _state_dump({0: {'digitizerFrequencyMHz': 614.4}})
    path = _write_yaml(state, suffix='.yml.gz')
    try:
        S = _make_offline(state_file=path)
        assert S._offline_state is not None
        val = S._caget(
            'AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.Base[0]'
            '.digitizerFrequencyMHz')
        assert val == 614.4
    finally:
        os.unlink(path)


def test_load_state_swaps_dump_at_runtime():
    first = _write_yaml(_state_dump({0: {'numberChannels': 100}}))
    second = _write_yaml(_state_dump({0: {'numberChannels': 200}}))
    try:
        S = _make_offline(state_file=first)
        assert S.get_number_channels(0) == 100
        S.load_state(second)
        assert S.get_number_channels(0) == 200
    finally:
        os.unlink(first)
        os.unlink(second)


# -- per-method hardcoded offline returns yield to the dump --------------

def test_hardcodes_used_when_no_dump():
    S = _make_offline()
    assert S.get_number_sub_bands(0) == 128
    assert S.get_number_channels(0) == 512
    assert S.get_band_center_mhz(2) == 4250 + 2 * 500
    assert S.get_channel_frequency_mhz(0) == 2.4
    assert S.get_digitizer_frequency_mhz(0) == 614.4
    assert S.get_flux_ramp_freq() == 4.0
    assert S.get_downsample_factor() == 20
    np.testing.assert_array_equal(
        S.get_filter_a(),
        np.array([1., -3.74145562, 5.25726624, -3.28776591, 0.77203984]))


def test_dump_overrides_hardcodes():
    state = _state_dump({0: {
        'numberSubBands': 64,
        'numberChannels': 528,
        'bandCenterMHz': 4321.0,
        'channelFrequencyMHz': 1.2,
        'digitizerFrequencyMHz': 700.0,
    }})
    path = _write_yaml(state)
    try:
        S = _make_offline(state_file=path)
        assert S.get_number_sub_bands(0) == 64
        assert S.get_number_channels(0) == 528
        assert S.get_band_center_mhz(0) == 4321.0
        assert S.get_channel_frequency_mhz(0) == 1.2
        assert S.get_digitizer_frequency_mhz(0) == 700.0
    finally:
        os.unlink(path)


def test_dump_missing_key_returns_none_without_falling_back_to_hardcode():
    # Document the documented tradeoff: a loaded dump that lacks a key
    # yields None rather than the legacy hardcode.  Callers opted in.
    state = _state_dump({0: {'numberChannels': 528}})
    path = _write_yaml(state)
    try:
        S = _make_offline(state_file=path)
        assert S.get_number_channels(0) == 528
        assert S.get_digitizer_frequency_mhz(0) is None
    finally:
        os.unlink(path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
