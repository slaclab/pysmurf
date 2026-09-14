#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Map Checks
#-----------------------------------------------------------------------------
# File       : check_platform_map.py
# Created    : 2026-09-14
#-----------------------------------------------------------------------------
# Description:
# Checks of the platform layer's own logic: identifying which platform a system
# is, and enumerating the indexed scopes a tree has. Both are answered from
# values the caller supplies, so both can be checked without a tree -- these run
# in a second and need neither rogue nor hardware.
#
# Identification: a platform is the firmware it runs. Each map lists the image
# names it covers, the name is taken from the build stamp, and firmware that
# matches nothing is refused by name rather than guessed at. A system whose
# firmware cannot say what it is -- an emulated register space reads as zeros --
# has its platform declared instead, which is the one way past the check and is
# meant to look deliberate.
#
# Scope enumeration: which band, bay or attenuator indices exist is a property of
# the tree, discovered by probing. A firmware mask may leave an index out and keep
# a higher one, so a gap does not end a scope; the check below covers gapped,
# sparse, empty and full ranges, because collapsing any of them silently drops
# real hardware from every name listing and witness that follows.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import argparse
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from cryodaq import platform                                    # noqa: E402
from cryodaq._errors import ConnectError                         # noqa: E402

# A build stamp as an ATCA carrier running v2.5.1 reports it, verbatim from
# artifacts of a live read, so the parsing below is checked against the real
# shape of the string rather than a tidied one.
LIVE_STAMP = ('MicrowaveMuxBpEthGen2: Vivado v2020.2, rdsrv403 '
              '(Ubuntu 22.04.5 LTS), Built Tue Jun 16 10:56:22 AM PDT 2026 by ruckman')

# What an unwritten 256-character register reads back as.
EMPTY_STAMP = '\x00' * 256


def reader(stamp):
    """A `read(path)` that answers with one build stamp and nothing else."""
    def read(path):
        assert path == platform.TAG_PATH, f"identification read {path!r}"
        return stamp
    return read


def presence(*indices):
    """A `has(path)` that says only these indices are in the tree."""
    wanted = {f"[{i}]" for i in indices}
    return lambda path: any(i in path for i in wanted)


# --------------------------------------------------------------------------
# identification
# --------------------------------------------------------------------------

def check_every_map_is_named_and_tagged():
    assert platform.MAPS, "no platform maps"
    for pmap in platform.MAPS:
        assert pmap.name, "a map has no name"
        assert pmap.tags, f"{pmap.name} claims no firmware"
        for tag in pmap.tags:
            assert tag and not tag.startswith(' '), f"{pmap.name} has a blank tag"
        assert platform.by_name(pmap.name) is pmap


def check_tags_are_unique_across_maps():
    seen = {}
    for pmap in platform.MAPS:
        for tag in pmap.tags:
            assert tag not in seen, \
                f"firmware {tag!r} is claimed by both {seen[tag]} and {pmap.name}"
            seen[tag] = pmap.name


def check_each_map_is_identified_by_its_own_firmware():
    for pmap in platform.MAPS:
        for tag in pmap.tags:
            stamp = f"{tag}: Vivado v2020.2, host (os), Built today by someone"
            got = platform.identify(reader(stamp))
            assert got is pmap, f"{tag!r} identified as {got.name}, not {pmap.name}"


def check_the_live_carrier_stamp_identifies_the_carrier():
    assert platform.tag_of(LIVE_STAMP) == 'MicrowaveMuxBpEthGen2'
    assert platform.identify(reader(LIVE_STAMP)).name == 'umux-atca'


def check_a_tree_with_no_firmware_is_refused():
    for stamp in (EMPTY_STAMP, '', None, '   '):
        try:
            pmap = platform.identify(reader(stamp))
        except ConnectError as e:
            assert 'declare' in str(e), f"the error does not say what to do: {e}"
        else:
            raise AssertionError(f"{stamp!r} was identified as {pmap.name}")


def check_unknown_firmware_is_refused_by_name():
    stamp = 'SomeFutureImage: Vivado v2020.2, host (os), Built today by someone'
    try:
        pmap = platform.identify(reader(stamp))
    except ConnectError as e:
        assert 'SomeFutureImage' in str(e), f"the error does not quote the firmware: {e}"
        for known in platform.MAPS[0].tags:
            assert known in str(e), f"the error does not list what is known: {e}"
    else:
        raise AssertionError(f"unknown firmware was identified as {pmap.name}")


def check_a_declared_platform_is_taken_without_reading():
    def refuse(path):
        raise AssertionError('a declared platform still read the tree')

    for pmap in platform.MAPS:
        assert platform.identify(refuse, declared=pmap.name) is pmap


def check_a_declared_platform_that_does_not_exist_is_refused():
    try:
        platform.by_name('no-such-platform')
    except ConnectError as e:
        assert 'umux-atca' in str(e), f"the error does not list what is known: {e}"
    else:
        raise AssertionError('an unknown platform name was accepted')


# --------------------------------------------------------------------------
# scope enumeration
# --------------------------------------------------------------------------

def enumerated(platform_name, scope, present, expected, **fixed):
    """Assert which indices a scope reports, saying what it reported instead."""
    pmap = platform.by_name(platform_name)
    got = platform.indices(pmap, presence(*present), scope, **fixed)
    assert got == expected, \
        (f"{platform_name} {scope} with {sorted(present)} in the tree "
         f"enumerated as {got}, expected {expected}")


def check_a_contiguous_scope_is_enumerated():
    enumerated('umux-atca', 'band', (0, 1), (0, 1))


def check_a_gap_does_not_end_a_scope():
    # A firmware band mask that omits a band below one it keeps: stopping at the
    # first gap reports (1,) and loses band 3 entirely.
    enumerated('umux-atca', 'band', (1, 3), (1, 3))


def check_a_missing_first_index_does_not_empty_a_scope():
    # Bay 0 disabled with bay 1 populated -- a state this hardware runs in.
    enumerated('umux-atca', 'bay', (1,), (1,))


def check_an_absent_scope_is_empty_rather_than_an_error():
    enumerated('umux-rfsoc', 'bay', (), ())


def check_a_full_scope_is_enumerated_to_the_ceiling():
    every = tuple(range(platform.MAX_SCOPE_INDEX))
    enumerated('umux-atca', 'band', every, every)


def check_a_nested_scope_needs_its_parent():
    pmap = platform.by_name('umux-atca')
    try:
        platform.indices(pmap, presence(0), 'uc')
    except KeyError as e:
        assert 'bay' in str(e), f"the error does not name the parent scope: {e}"
    else:
        raise AssertionError('a nested scope was enumerated without its parent')


def check_an_unknown_scope_is_an_error():
    pmap = platform.by_name('umux-atca')
    try:
        platform.indices(pmap, presence(0), 'no_such_scope')
    except KeyError:
        pass
    else:
        raise AssertionError('an unknown scope was enumerated')


# --------------------------------------------------------------------------
# what the maps share
# --------------------------------------------------------------------------

def check_the_generation_shares_one_register_table():
    # The platforms of this generation are separate because they are configured
    # differently, not because they read different registers. Sharing the table
    # is the claim; a copy that drifted would be the failure.
    atca = platform.by_name('umux-atca')
    rfsoc = platform.by_name('umux-rfsoc')
    assert atca.patterns == rfsoc.patterns, \
        "the two platforms of this generation no longer share their register table"
    assert atca.witness == rfsoc.witness
    for pattern in atca.patterns:
        assert atca.registers[pattern] == rfsoc.registers[pattern], \
            f"{pattern} differs between the two platforms"


def check_every_witness_name_is_in_the_map():
    for pmap in platform.MAPS:
        for name in pmap.witness:
            assert name in pmap, f"{pmap.name} witnesses {name}, which it cannot resolve"


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()

    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []

    print(f"Checking the cryodaq platform maps ({len(checks)} checks)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            failed.append(label)
            print(f"  FAIL  {label}")
            print(f"          {type(e).__name__}: {e}")
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
