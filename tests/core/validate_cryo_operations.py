#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryo Operations Attachment Validation Script
#-----------------------------------------------------------------------------
# File       : validate_cryo_operations.py
# Created    : 2026-08-28
#-----------------------------------------------------------------------------
# Description:
# Script to validate pysmurf.core.operations' attachment of the cryo channel
# operations to a CryoChannels device.
#
# Needs no hardware and no CryoDet package: it builds throwaway pyrogue devices
# that stand in for one band's CryoChannels, so it can run anywhere rogue is
# importable. What it covers is that the attach puts all 31 nodes where they
# belong, and that a CryoDet package which still defines them itself is refused
# with a message that says so.
#
# It does not check that the operations themselves work; that needs a carrier.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import sys

import pyrogue as pr

import pysmurf.core.operations as ops
from pysmurf.core.operations import _CryoOperations

# Matches cryo-det's CryoChannels: 512 channels over a +/-1.2 MHz span.
N_CHANNELS = 512
FREQ_SPAN_MHZ = 1.2

# How many CryoChannel children to give the stand-in device. The real one has
# N_CHANNELS; a handful is enough here and keeps the script quick.
N_STUB_CHANNELS = 4

# What real CryoDet releases actually provide -- the four distinct sets of
# absent nodes across all 53 MicrowaveMuxBpEthGen2 tags, measured from
# cryo-det's git history. (v1.2.14 is a fifth shape, matching v0.0.1's absences
# but adding two nodes pysmurf does not know.) Every one of these is a package
# that predates the strip, so every one of them must be refused: pysmurf and the
# CryoDet package are updated together. They are listed by version rather than
# collapsed into one case so that the refusal is checked against the real
# variety, including the generation slaclab/zcu208-cryo-det pins for the RFSoC
# build. Regenerate with /workspace/scripts/stage1_firmware_ops_survey.py in the
# development notes.
FIRMWARE_RELEASES = {
    # v0.0.1 .. v1.0.1: before SerialFindFreq existed.
    'v0.0.1': ('etaScanMaxMag', 'UseNewSerialGradientDescent',
               'NewSerialGradientDescent', 'SerialFindFreq',
               'runSerialFindFreq'),
    # v1.0.3 .. v2.2.0: before NewSerialGradientDescent existed.
    'v2.2.0': ('etaScanMaxMag', 'UseNewSerialGradientDescent',
               'NewSerialGradientDescent'),
    # v2.3.0 .. v2.5.0: etaScanMaxMag not added yet. This is also the
    # generation slaclab/zcu208-cryo-det pins for the RFSoC build.
    'v2.5.0': ('etaScanMaxMag',),
    # v2.5.1 == the commit the operations were moved from.
    'v2.5.1': (),
}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

def make_cryo_channels():
    """A stand-in for one band's stripped CryoChannels device."""
    ch = pr.Device(name='CryoChannels', description='Stand-in for one band')
    ch._n_channels = N_CHANNELS
    ch._freqSpanMHz = FREQ_SPAN_MHZ
    for i in range(N_STUB_CHANNELS):
        ch.add(pr.Device(name=f'CryoChannel[{i}]'))
    return ch


def make_fpga(n_bands=8):
    """A stand-in for the FpgaTopLevel path the attach walks."""
    fpga = pr.Device(name='FpgaTopLevel')
    app_top = pr.Device(name='AppTop')
    app_core = pr.Device(name='AppCore')
    sysgen = pr.Device(name='SysgenCryo')
    fpga.add(app_top)
    app_top.add(app_core)
    app_core.add(sysgen)
    for i in range(n_bands):
        band = pr.Device(name=f'Base[{i}]')
        sysgen.add(band)
        band.add(make_cryo_channels())
    return fpga


def add_firmware_operations(ch, skip=()):
    """Add the operation nodes the way a pre-strip CryoDet package would."""
    for name in ops.OPERATION_NODES:
        if name in skip:
            continue
        if name in ops.OPERATION_PROCESSES:
            ch.add(pr.Device(name=name))
        elif name in ops.OPERATION_COMMANDS:
            ch.add(pr.LocalCommand(name=name, function=lambda: None))
        else:
            ch.add(pr.LocalVariable(name=name, value=0))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_fresh_attach():
    """A stripped device gets all 31 nodes, in cryo-det's original order."""
    ch = make_cryo_channels()
    before = set(ch.nodes)
    ops.attach_cryo_operations(ch)
    added = [n for n in ch.nodes if n not in before]
    assert added == list(ops.OPERATION_NODES), f"added: {added}"
    assert len(added) == 31, f"expected 31 nodes, got {len(added)}"


def check_old_package_is_refused():
    """A package that still defines the operations must not be attached over.

    Two copies of the tuning code cannot coexist at one set of rogue paths, and
    a partial attach would be worse than a refusal: it would let one package's
    algorithms run against the other package's parameters. So the attach
    refuses, and it does so before adding anything.
    """
    ch = make_cryo_channels()
    add_firmware_operations(ch)
    before = list(ch.nodes)
    try:
        ops.attach_cryo_operations(ch)
    except RuntimeError as e:
        assert 'already defines' in str(e), f"must explain the problem: {e}"
    else:
        raise AssertionError("a package that still has the operations must raise")
    assert list(ch.nodes) == before, "the tree was modified before the refusal"


def check_every_real_release_is_refused():
    """Not just the newest package -- every released one predates the strip.

    Each of these is missing a different subset of the 31 nodes, so this also
    pins down that the refusal keys on *any* node being present rather than on
    all of them.
    """
    for release, absent in FIRMWARE_RELEASES.items():
        ch = make_cryo_channels()
        add_firmware_operations(ch, skip=absent)
        before = list(ch.nodes)
        try:
            ops.attach_cryo_operations(ch)
        except RuntimeError as e:
            # The message has to name what it found, so an operator can tell
            # which package is loaded without reading pysmurf's source.
            assert 'etaScanChannel' in str(e), f"{release}: {e}"
        else:
            raise AssertionError(f"{release} must be refused")
        assert list(ch.nodes) == before, f"{release}: the tree was modified"


def check_any_single_node_is_enough_to_refuse():
    """One leftover node out of the 31 is a refusal, not a partial attach.

    Every name is checked individually because the collision surface is the
    whole node table: a node left out of the OPERATION_* tuples would be
    invisible to the pre-attach check and would blow up inside pyrogue's add()
    instead, halfway through.
    """
    for name in ops.OPERATION_NODES:
        ch = make_cryo_channels()
        if name in ops.OPERATION_PROCESSES:
            ch.add(pr.Device(name=name))
        elif name in ops.OPERATION_COMMANDS:
            ch.add(pr.LocalCommand(name=name, function=lambda: None))
        else:
            ch.add(pr.LocalVariable(name=name, value=0))
        before = list(ch.nodes)
        try:
            ops.attach_cryo_operations(ch)
        except RuntimeError as e:
            assert name in str(e), f"{name}: not named in the error: {e}"
        else:
            raise AssertionError(f"a leftover {name} must be refused")
        assert list(ch.nodes) == before, f"{name}: attached anyway"


def check_unrelated_nodes_do_not_block_the_attach():
    """Only the 31 names matter; the rest of CryoChannels is not our business.

    cryo-det keeps a large hardware map on this device, plus one LocalVariable
    (setCenterFrequencyDelay) that stayed behind. None of it may be mistaken for
    a leftover operation.
    """
    ch = make_cryo_channels()
    ch.add(pr.LocalVariable(name='setCenterFrequencyDelay', value=0))
    ch.add(pr.Device(name='GradientDescent'))   # dropped before the move
    ops.attach_cryo_operations(ch)
    for name in ops.OPERATION_NODES:
        assert name in ch.nodes, f"{name} was not attached"


def check_command_arguments():
    """Only setAmplitudeScales takes an argument.

    pyrogue decides that by inspecting the registered function's signature, so
    this is what breaks if a command is ever wrapped in a lambda on its way
    into pr.LocalCommand.
    """
    ch = make_cryo_channels()
    ops.attach_cryo_operations(ch)
    for name in ops.OPERATION_COMMANDS:
        cmd = ch.nodes[name]
        want = (name == 'setAmplitudeScales')
        assert cmd.arg is want, f"{name}: takes an argument = {cmd.arg}"


def check_descriptions_survive():
    """Every moved node keeps its description, which is tree documentation."""
    ch = make_cryo_channels()
    ops.attach_cryo_operations(ch)
    for name in ops.OPERATION_VARIABLES + ops.OPERATION_COMMANDS:
        assert ch.nodes[name].description, f"{name} lost its description"


def check_find_freq_geometry():
    """SerialFindFreq must get its sweep geometry from the device."""
    ch = make_cryo_channels()
    ops.attach_cryo_operations(ch)
    proc = ch.nodes['SerialFindFreq']
    assert proc._n_channels == N_CHANNELS, proc._n_channels
    assert proc._freq_span_mhz == FREQ_SPAN_MHZ, proc._freq_span_mhz


def check_attach_all_bands():
    fpga = make_fpga(8)
    bands = ops.attach_all_cryo_operations(fpga)
    assert bands == list(range(8)), f"attached bands: {bands}"
    for i in bands:
        ch = fpga.AppTop.AppCore.SysgenCryo.Base[i].CryoChannels
        for name in ops.OPERATION_NODES:
            assert name in ch.nodes, f"band {i} is missing {name}"


def check_attach_all_skips_missing_bands():
    """Not every platform builds all eight bands."""
    assert ops.attach_all_cryo_operations(make_fpga(3)) == [0, 1, 2]
    assert ops.attach_all_cryo_operations(pr.Device(name='FpgaTopLevel')) == []


def check_attach_all_propagates_errors():
    """A band that cannot be attached aborts startup rather than being skipped.

    Only the node *lookup* is allowed to be forgiving in attach_all; an error
    from the attach itself must reach the caller, or a half-attached band would
    start up and misbehave later.
    """
    fpga = make_fpga(2)
    ch = fpga.AppTop.AppCore.SysgenCryo.Base[1].CryoChannels
    add_firmware_operations(ch)
    try:
        ops.attach_all_cryo_operations(fpga)
    except RuntimeError as e:
        assert 'Base[1]' in str(e), f"must name the band: {e}"
    else:
        raise AssertionError("attach_all must not swallow the attach error")


def check_exists_matches_add():
    """_exists() must agree with Device.add() on all 31 names.

    This is the whole basis of the pre-attach check: if the two predicates could
    ever disagree, "would this collide?" would stop predicting "did it?".
    """
    for name in ops.OPERATION_NODES:
        ch = make_cryo_channels()
        assert not _CryoOperations._exists(ch, name), f"{name} exists too early"
        ch.add(pr.LocalVariable(name=name, value=0))
        assert _CryoOperations._exists(ch, name), f"{name} not seen after add"
        try:
            ch.add(pr.LocalVariable(name=name, value=0))
        except pr.NodeError:
            pass
        else:
            raise AssertionError(f"add() accepted a duplicate {name}")


# --------------------------------------------------------------------------

def main():
    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []

    print(f"Validating pysmurf.core.operations ({len(checks)} checks)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
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
