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
# has its platform named instead, which is a lookup by name and not a second way
# through identification: the two are separate functions and the checks below
# hold them apart.
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
from cryodaq._errors import ConnectError, UnresolvedName         # noqa: E402

# A build stamp as an AMC carrier running v2.5.1 reports it, verbatim from
# artifacts of a live read, so the parsing below is checked against the real
# shape of the string rather than a tidied one.
LIVE_STAMP = ('MicrowaveMuxBpEthGen2: Vivado v2020.2, rdsrv403 '
              '(Ubuntu 22.04.5 LTS), Built Tue Jun 16 10:56:22 AM PDT 2026 by ruckman')

# What an unwritten 256-character register reads back as.
EMPTY_STAMP = '\x00' * 256


def tree(stamp):
    """A tree that carries one build stamp and no other node.

    Identification asks a tree for a node and reads it, which is all of rogue's
    interface it uses -- so this stands in for one in two methods.
    """
    class Node:
        def get(self):
            return stamp

    class Tree:
        def getNode(self, path):
            assert path in {m.tag_path for m in platform.MAPS}, f"identification read {path!r}"
            return Node()

    return Tree()


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


def check_each_map_reads_its_stamp_where_its_table_says():
    """The stamp a platform is identified by is the register its own map names.

    Identification reads ``tag_path`` before it knows the platform; once known, a
    client reads ``firmware.build_stamp`` through the map. The two must be one
    register -- otherwise a platform could be recognised by one stamp and report
    another -- and it must sit in the carrier core's version block, which is the
    one device the server's probe builds before it has a tree.
    """
    for pmap in platform.MAPS:
        assert pmap.tag_path == pmap.path('firmware.build_stamp'), \
            f"{pmap.name} is identified at {pmap.tag_path} but reports its stamp elsewhere"
        assert pmap.tag_path.endswith('.AxiVersion.BuildStamp'), \
            f"{pmap.name}: the stamp is not in a version block"


def check_a_stamp_alone_names_the_platform():
    """A build stamp's text identifies the platform with no tree: what the server's probe uses.

    One version block serves every platform, so the platform is in the stamp's
    text and never in where the block was read from. The same stamp through a
    tree gives the same answer, and the two refuse the same inputs.
    """
    for pmap in platform.MAPS:
        for tag in pmap.tags:
            stamp = f"{tag}: Vivado v2020.2, host (os), Built today by someone"
            assert platform.from_stamp(stamp) is pmap
            assert platform.identify(tree(stamp)) is pmap
    assert platform.from_stamp(LIVE_STAMP).name == 'umux-amc'
    for stamp in (EMPTY_STAMP, '', None, '   '):
        try:
            platform.from_stamp(stamp)
        except ConnectError as e:
            assert 'declare' in str(e), e
        else:
            raise AssertionError(f"{stamp!r} named a platform")
    try:
        platform.from_stamp('SomeFutureImage: Vivado v2020.2, host (os), Built today by someone')
    except ConnectError as e:
        assert 'SomeFutureImage' in str(e), e
    else:
        raise AssertionError('unknown firmware named a platform')


def check_identification_reads_each_stamp_register_once():
    """A tree is asked for each distinct stamp register once, whatever the answer.

    Two platforms keep their stamp at one path, a third elsewhere; a tree that
    runs none of them must still have been asked at each path exactly once --
    not once per map -- and the error names every path that was read.
    """
    asked = []

    class Node:
        def get(self):
            return EMPTY_STAMP

    class Tree:
        def getNode(self, path):
            asked.append(path)
            return Node()

    try:
        platform.identify(Tree())
    except ConnectError as e:
        for path in set(asked):
            assert path in str(e), f"the error does not name {path}: {e}"
    else:
        raise AssertionError('an empty tree was identified')
    distinct = {m.tag_path for m in platform.MAPS}
    assert len(asked) == len(distinct) and set(asked) == distinct, asked
    assert len(asked) < len(platform.MAPS), 'no two maps share a stamp register; the case is empty'


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
            got = platform.identify(tree(stamp))
            assert got is pmap, f"{tag!r} identified as {got.name}, not {pmap.name}"


def check_the_live_carrier_stamp_identifies_the_carrier():
    assert platform.tag_of(LIVE_STAMP) == 'MicrowaveMuxBpEthGen2'
    assert platform.identify(tree(LIVE_STAMP)).name == 'umux-amc'


def check_a_tree_with_no_firmware_is_refused():
    for stamp in (EMPTY_STAMP, '', None, '   '):
        try:
            pmap = platform.identify(tree(stamp))
        except ConnectError as e:
            assert 'declare' in str(e), f"the error does not say what to do: {e}"
        else:
            raise AssertionError(f"{stamp!r} was identified as {pmap.name}")


def check_unknown_firmware_is_refused_by_name():
    stamp = 'SomeFutureImage: Vivado v2020.2, host (os), Built today by someone'
    try:
        pmap = platform.identify(tree(stamp))
    except ConnectError as e:
        assert 'SomeFutureImage' in str(e), f"the error does not quote the firmware: {e}"
        for known in platform.MAPS[0].tags:
            assert known in str(e), f"the error does not list what is known: {e}"
    else:
        raise AssertionError(f"unknown firmware was identified as {pmap.name}")


def check_identification_only_ever_reads_the_tree():
    """Naming a platform and discovering one are two functions, and stay two.

    A declaration reaches ``by_name`` and never ``identify``, so there is no path
    by which a tree that says which firmware it runs can be overruled from the
    outside, and none by which a tree that says nothing is guessed at.
    """
    for pmap in platform.MAPS:
        try:
            platform.identify(tree(EMPTY_STAMP), declared=pmap.name)
        except TypeError:
            pass
        except ConnectError:
            raise AssertionError('identify still takes a declared platform')
        else:
            raise AssertionError('identify still takes a declared platform')
        assert platform.by_name(pmap.name) is pmap


def check_a_declared_platform_that_does_not_exist_is_refused():
    try:
        platform.by_name('no-such-platform')
    except ConnectError as e:
        assert 'umux-amc' in str(e), f"the error does not list what is known: {e}"
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
    enumerated('umux-amc', 'band', (0, 1), (0, 1))


def check_a_gap_does_not_end_a_scope():
    # A firmware band mask that omits a band below one it keeps: stopping at the
    # first gap reports (1,) and loses band 3 entirely.
    enumerated('umux-amc', 'band', (1, 3), (1, 3))


def check_a_missing_first_index_does_not_empty_a_scope():
    # Bay 0 disabled with bay 1 populated -- a state this hardware runs in.
    enumerated('umux-amc', 'bay', (1,), (1,))


def check_an_absent_scope_is_empty_rather_than_an_error():
    enumerated('umux-rfsoc', 'bay', (), ())


def check_a_full_window_is_enumerated():
    every = tuple(range(platform.MAX_SCOPE_INDEX))
    enumerated('umux-amc', 'band', every, every)


def check_a_scope_is_enumerated_past_the_window():
    # A band has 512 channels, sixteen windows' worth. The enumeration has to reach
    # the last one, and stop where the tree does: 512 present, and nothing after.
    wanted = tuple(range(512))
    present = {f"CryoChannel[{i}]." for i in wanted}

    def has(path):
        # presence() matches an index anywhere in the path, and the parent band's
        # own [0] would match every probe; only the channel segment counts here.
        return any(p in path + '.' for p in present)

    for name in ('umux-amc', 'umux-rfsoc'):
        got = platform.indices(platform.by_name(name), has, 'channel', band=0)
        assert got == wanted, \
            f"{name} channel enumerated {len(got)} indices, last {got[-1:]}, expected 512"


def check_a_sampled_scope_expands_to_its_ends_and_middle():
    """``expand(sample=)`` fills a scope with three of its indices.

    A validation that samples the channels asks three of the 512 for each
    pattern; what it can and cannot see is asserted here so the trade-off is
    on record. The scope's extent still comes from the tree -- a band with 400
    channels samples 0, 200 and 399 -- so a name the sample offers and the
    tree lacks is a register missing at one of those three indices; a register
    missing at channel 100 alone is not seen.
    """
    pmap = platform.by_name('umux-amc')
    pattern = 'band[*].channel[*].tone.amplitude'

    def has_channels(n):
        def has(path):
            if 'CryoChannel[' not in path:
                return 'Base[0]' in path
            index = int(path.split('CryoChannel[')[1].split(']')[0])
            return index < n
        return has

    names = platform.expand(pmap, has_channels(512), pattern, sample=('channel',))
    assert names == ['band[0].channel[0].tone.amplitude', 'band[0].channel[256].tone.amplitude',
                     'band[0].channel[511].tone.amplitude'], names
    assert len(platform.expand(pmap, has_channels(512), pattern)) == 512
    fewer = platform.expand(pmap, has_channels(400), pattern, sample=('channel',))
    assert fewer == ['band[0].channel[0].tone.amplitude', 'band[0].channel[200].tone.amplitude',
                     'band[0].channel[399].tone.amplitude'], fewer
    # Nothing sampled: the full enumeration, as before.
    assert platform.expand(pmap, has_channels(512), pattern, sample=()) == \
        platform.expand(pmap, has_channels(512), pattern)


def check_a_scope_ends_after_an_empty_window():
    # Indices 0 and 40 are present: 40 is in the second window, so it is found; the
    # third window (64-95) is empty, so nothing past it is probed. A gap wider than a
    # window ends the scope -- that is the definition of its end.
    enumerated('umux-amc', 'band', (0, 40, 100), (0, 40))


def check_a_nested_scope_needs_its_parent():
    pmap = platform.by_name('umux-amc')
    try:
        platform.indices(pmap, presence(0), 'attenuator')
    except KeyError as e:
        assert 'bay' in str(e), f"the error does not name the parent scope: {e}"
    else:
        raise AssertionError('a nested scope was enumerated without its parent')


def check_an_unknown_scope_is_an_error():
    pmap = platform.by_name('umux-amc')
    try:
        platform.indices(pmap, presence(0), 'no_such_scope')
    except KeyError:
        pass
    else:
        raise AssertionError('an unknown scope was enumerated')


def check_a_name_carrying_whitespace_is_refused():
    """Every character of a name is part of it, including the last.

    A name arrives from somewhere -- a file, a command line, a copied cell in a
    table -- and arrives with what that somewhere put on the end. Anchoring a
    segment with ``$`` is not enough on its own, because in Python it also
    matches immediately before a final newline: the name would parse and the
    stray character would then be formatted into a register path.
    """
    good = 'band[4].tone.amplitude'
    platform.parse(good)                                     # the control
    for trailer in ('\n', '\r\n', ' ', '\t'):
        for name in (good + trailer, trailer + good):
            try:
                platform.parse(name)
            except UnresolvedName:
                pass
            else:
                raise AssertionError(f"{name!r} parsed")


def check_caching_a_parse_does_not_share_its_indices():
    """Parsing is cached, and two callers still get dictionaries of their own.

    A name parses to the same answer every time, so the answer is kept -- a tuning
    loop reaching a per-channel register pays for the regular expressions once
    rather than per call. The hazard a cache introduces is aliasing: handing the
    same dictionary to two callers lets one of them change what the other reads,
    and the symptom would be an index that is right until something else resolves
    the same name.
    """
    name = 'band[4].channel[17].tone.amplitude'
    first_pattern, first = platform.parse(name)
    assert first == {'band': 4, 'channel': 17}, first
    first['band'] = 99
    first['injected'] = 1
    second_pattern, second = platform.parse(name)
    assert second == {'band': 4, 'channel': 17}, \
        f'a second parse of {name} returned {second}'
    assert first_pattern == second_pattern == 'band[*].channel[*].tone.amplitude'
    # A malformed name must keep failing rather than being answered from a cache
    # of the exception, and a name fixed afterwards must resolve. The unhashable
    # ones are the cache's own failure mode: they must still get UnresolvedName,
    # not the TypeError the cache would raise while keying on them.
    for bad in ('band[4].tone.amplitude\n', '', None, [], ['band[4]'], {}):
        for _ in range(2):
            try:
                platform.parse(bad)
            except UnresolvedName:
                pass
            else:
                raise AssertionError(f'{bad!r} parsed')
    platform.parse('band[4].tone.amplitude')


# --------------------------------------------------------------------------
# what the maps share
# --------------------------------------------------------------------------

def check_the_generation_shares_every_register_it_can():
    """The platforms differ only by the hardware one of them does not have.

    They share most of their table and are separate platforms because they are
    configured differently. What they do not share is the hardware only one carries:
    a platform whose converters are on the FPGA's die has no RF front end and no
    serial links to one, so it does not offer those names at all.

    Two claims, and the second is the one that keeps the split honest: a name in both
    maps must mean the same register in both -- a copy that drifted would be two
    tables pretending to be one -- and no name may be offered by the platform with
    *fewer* devices and not by the other, which would mean the sharing is backwards.
    """
    amc = platform.by_name('umux-amc')
    rfsoc = platform.by_name('umux-rfsoc')
    shared = set(amc.patterns) & set(rfsoc.patterns)
    assert len(shared) > 100, \
        f"only {len(shared)} names are shared; the maps have diverged"
    for pattern in sorted(shared):
        assert amc.registers[pattern] == rfsoc.registers[pattern], \
            f"{pattern} differs between the two platforms"
    assert set(rfsoc.witness) <= set(amc.witness), \
        "the platform with fewer devices witnesses something the carrier does not"
    extra = sorted(set(rfsoc.patterns) - set(amc.patterns))
    assert not extra, (f"{rfsoc.name} offers name(s) the carrier does not: "
                       f"{', '.join(extra[:6])}")


def check_a_platform_offers_no_name_it_cannot_resolve():
    """Every name a map offers builds a path, and every scope it needs is declared.

    The split moved eleven names and four scopes into one platform's map. A name left
    behind whose scope went with it would raise on the first call rather than at
    import, so it is checked here: the pattern is filled with an index for each scope
    it carries and the result must contain no placeholder.
    """
    for pmap in platform.MAPS:
        for pattern in pmap.patterns:
            scopes = [segment.split('[')[0] for segment in pattern.split('.')
                      if segment.endswith('[*]')]
            for scope in scopes:
                assert scope in pmap.scopes, \
                    f"{pmap.name} offers {pattern}, whose scope {scope!r} it does not declare"
            name = pattern
            for scope in scopes:
                name = name.replace(f'{scope}[*]', f'{scope}[0]', 1)
            path = pmap.path(name)
            assert '{' not in path, \
                f"{pmap.name}: {pattern} resolves to {path}, which has a placeholder left"


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
