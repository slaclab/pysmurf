#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Unit tests for pysmurf.client.util.tools.coerce_value_for_var
#-----------------------------------------------------------------------------
# File       : pysmurf/client/test/test_tools.py
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import numpy as np
import pytest

from pysmurf.client.util.tools import (
    TYPEMAP,
    coerce_value_for_var,
)


class _StubVar:
    def __init__(self, type_str, path='stub.var'):
        self.typeStr = type_str
        self.path = path


# (typeStr, sample input, expected python/numpy type for the returned scalar)
_TYPEMAP_CASES = [
    ('UInt8',   1,        np.uint8),
    ('UInt16',  1,        np.uint16),
    ('UInt32',  1,        np.uint32),
    ('UInt64',  1,        np.uint64),
    ('Int8',    -1,       np.int8),
    ('Int16',   -1,       np.int16),
    ('Int32',   -1,       np.int32),
    ('Int64',   -1,       np.int64),
    ('Float32', 1.5,      np.float32),
    ('Double',  1.5,      np.float64),
    ('Bool',    1,        bool),
    ('String',  'x',      str),
]


@pytest.mark.parametrize('type_str,val,expected_type', _TYPEMAP_CASES)
def test_coerce_canonical_typemap(type_str, val, expected_type):
    out = coerce_value_for_var(_StubVar(type_str), val)
    assert isinstance(out, expected_type)


def test_typemap_has_no_extra_keys():
    expected = {
        'UInt8', 'UInt16', 'UInt32', 'UInt64',
        'Int8', 'Int16', 'Int32', 'Int64',
        'Float32', 'Double', 'Bool', 'String',
    }
    assert set(TYPEMAP.keys()) == expected


# Arbitrary-width integer fallback: round up to next native numpy width.
@pytest.mark.parametrize('type_str,expected_dtype', [
    ('UInt1',  np.uint8),
    ('UInt6',  np.uint8),
    ('UInt8',  np.uint8),
    ('UInt9',  np.uint16),
    ('UInt12', np.uint16),
    ('UInt16', np.uint16),
    ('UInt17', np.uint32),
    ('UInt32', np.uint32),
    ('UInt33', np.uint64),
    ('UInt64', np.uint64),
    ('Int1',   np.int8),
    ('Int7',   np.int8),
    ('Int8',   np.int8),
    ('Int9',   np.int16),
    ('Int16',  np.int16),
    ('Int24',  np.int32),
    ('Int48',  np.int64),
    ('Int64',  np.int64),
])
def test_coerce_arbitrary_integer_widths(type_str, expected_dtype):
    out = coerce_value_for_var(_StubVar(type_str), 1)
    assert isinstance(out, expected_dtype)


def test_coerce_float64_alias():
    # Rogue emits 'Float64' in some firmware (e.g. cryo-det); it is the
    # same dtype as 'Double'.
    out = coerce_value_for_var(_StubVar('Float64'), 1.5)
    assert isinstance(out, np.float64)


def test_coerce_array_canonical():
    out = coerce_value_for_var(_StubVar('UInt32[np]'), [1, 2, 3])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint32


def test_coerce_array_arbitrary_width():
    # Array form of a non-power-of-2 width.
    out = coerce_value_for_var(_StubVar('UInt6[np]'), [0, 1, 63])
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8


@pytest.mark.parametrize('type_str', [
    'Foo',          # nonsense
    'UInt0',        # width below range
    'UInt65',       # width above range
    'Int0',
    'Int65',
    'Float16',      # width not supported
    'Float128',
    'UInt',         # missing width
    '',             # empty
])
def test_coerce_invalid_typestr_raises(type_str):
    with pytest.raises(TypeError) as exc:
        coerce_value_for_var(_StubVar(type_str, path='abc.def'), 1)
    assert 'abc.def' in str(exc.value)
    assert type_str in str(exc.value)


def test_coerce_invalid_array_typestr_raises():
    with pytest.raises(TypeError) as exc:
        coerce_value_for_var(_StubVar('Foo[np]', path='abc.def'), [1, 2])
    assert 'abc.def' in str(exc.value)
    assert 'Foo[np]' in str(exc.value)
