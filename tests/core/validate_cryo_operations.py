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
# importable. What it covers is that the attach puts all 29 nodes where they
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
# that defines the operations itself, so every one of them must be kept as it
# is, nothing added. They are listed by version rather than collapsed into one
# case so that the rule is checked against the real variety, including the
# generation slaclab/zcu208-cryo-det pins for the RFSoC build.
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
    # v2.5.1 and v2.6.0: all 29.
    'v2.6.0': (),
}

# Nodes releases up to v2.5.1 carry that pysmurf does not own: cryo-det removed
# the never-used ParrallelEtaScan pair in v2.6.0. The stand-ins add them
# too; they must neither change the decision nor be mistaken for something the
# attach should have provided.
RETIRED_NODES = ('ParrallelEtaScan', 'runParallelEtaScan')


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
    # every release also has the ParrallelEtaScan pair pysmurf retired
    ch.add(pr.Device(name=RETIRED_NODES[0]))
    ch.add(pr.LocalCommand(name=RETIRED_NODES[1], function=lambda: None))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_fresh_attach():
    """A stripped device gets all 29 nodes, in cryo-det's original order."""
    ch = make_cryo_channels()
    before = set(ch.nodes)
    ops.attach_cryo_operations(ch)
    added = [n for n in ch.nodes if n not in before]
    assert added == list(ops.OPERATION_NODES), f"added: {added}"
    assert len(added) == 29, f"expected 29 nodes, got {len(added)}"
    for name in RETIRED_NODES:
        assert name not in ch.nodes, f"{name} was removed in cryo-det v2.6.0"


class Captured:
    """The log records a device emits during one call, by level name."""

    def __init__(self, dev):
        self.dev, self.records = dev, []

    def __enter__(self):
        import logging

        class H(logging.Handler):
            def emit(h, record):
                self.records.append((record.levelname, record.getMessage()))
        self.h = H()
        self.level = self.dev._log.level
        self.dev._log.setLevel(logging.INFO)
        self.dev._log.addHandler(self.h)
        return self

    def __exit__(self, *exc):
        self.dev._log.removeHandler(self.h)
        self.dev._log.setLevel(self.level)

    def at(self, level):
        return [m for lvl, m in self.records if lvl == level]

    @staticmethod
    def names_in(message):
        """The node names a warning lists after its colon, whole words only."""
        listed = message.split('will not find: ', 1)[1].split('.', 1)[0]
        return {n.strip() for n in listed.split(',')}


def check_a_package_that_defines_the_operations_is_left_alone():
    """A package with its own operations keeps them: nothing is added, and the attach says so.

    Two copies of the tuning code cannot coexist at one set of rogue paths, and
    a partial attach would be worse than either: it would let one package's
    algorithms run against the other package's parameters. So a device that
    already has the operations is left exactly as the firmware built it.
    """
    ch = make_cryo_channels()
    add_firmware_operations(ch)
    before = list(ch.nodes)
    with Captured(ch) as log:
        attached = ops.attach_cryo_operations(ch)
    assert attached is False, "a package with its own operations was reported attached"
    assert list(ch.nodes) == before, "the tree was modified"
    assert any('keeping them' in m for m in log.at('INFO')), log.records
    assert log.at('WARNING') == [], f"a complete package drew a warning: {log.at('WARNING')}"


def check_every_real_release_is_left_alone_and_its_gaps_named():
    """Every released package keeps its own operations; the nodes it lacks are named in a warning.

    Each release is missing a different subset of the 29 nodes, so this also
    pins down that the decision keys on *any* node being present rather than on
    all of them -- an older release gets no partial fill-in, only a warning
    naming what a client will not find.
    """
    for release, absent in FIRMWARE_RELEASES.items():
        ch = make_cryo_channels()
        add_firmware_operations(ch, skip=absent)
        before = list(ch.nodes)
        with Captured(ch) as log:
            attached = ops.attach_cryo_operations(ch)
        assert attached is False, f"{release}: reported attached"
        assert list(ch.nodes) == before, f"{release}: the tree was modified"
        warnings = log.at('WARNING')
        if absent:
            assert len(warnings) == 1, f"{release}: {warnings}"
            assert Captured.names_in(warnings[0]) == set(absent), \
                f"{release}: the warning names {Captured.names_in(warnings[0])}, not {set(absent)}"
        else:
            assert warnings == [], f"{release}: a complete package drew a warning: {warnings}"


def check_any_single_node_is_enough_to_keep_the_firmwares():
    """One node out of the 29 means the firmware owns the operations: no partial attach.

    Every name is checked individually because the decision surface is the
    whole node table: a node left out of the OPERATION_* tuples would be
    invisible to the check and would blow up inside pyrogue's add() instead,
    halfway through.
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
        with Captured(ch) as log:
            attached = ops.attach_cryo_operations(ch)
        assert attached is False and list(ch.nodes) == before, f"{name}: attached anyway"
        assert log.at('WARNING') and name not in Captured.names_in(log.at('WARNING')[0]), f"{name}: {log.records}"


def check_unrelated_nodes_do_not_block_the_attach():
    """Only the 29 names matter; the rest of CryoChannels is not our business.

    cryo-det keeps a large hardware map on this device, plus one LocalVariable
    (setCenterFrequencyDelay) that stayed behind. None of it may be mistaken for
    a leftover operation.
    """
    ch = make_cryo_channels()
    ch.add(pr.LocalVariable(name='setCenterFrequencyDelay', value=0))
    ch.add(pr.Device(name='GradientDescent'))   # a device cryo-det dropped long ago
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
    """Every node keeps its description, which is tree documentation."""
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


def check_attach_all_reports_only_the_bands_it_attached():
    """A band whose firmware defines the operations is kept and not counted as attached."""
    fpga = make_fpga(3)
    add_firmware_operations(fpga.AppTop.AppCore.SysgenCryo.Base[1].CryoChannels)
    assert ops.attach_all_cryo_operations(fpga) == [0, 2]


def check_attach_all_skips_missing_bands():
    """Not every platform builds all eight bands."""
    assert ops.attach_all_cryo_operations(make_fpga(3)) == [0, 1, 2]
    assert ops.attach_all_cryo_operations(pr.Device(name='FpgaTopLevel')) == []


def check_attach_all_propagates_errors():
    """A band that cannot be attached aborts startup rather than being skipped.

    Only the node *lookup* is allowed to be forgiving in attach_all; an error
    from the attach itself must reach the caller, or a half-attached band would
    start up and misbehave later. Provoked here by a device whose channel count
    the processes cannot be sized from.
    """
    fpga = make_fpga(2)
    ch = fpga.AppTop.AppCore.SysgenCryo.Base[1].CryoChannels
    del ch._n_channels
    try:
        ops.attach_all_cryo_operations(fpga)
    except AttributeError:
        pass
    else:
        raise AssertionError("attach_all must not swallow the attach error")


def check_exists_matches_add():
    """_exists() must agree with Device.add() on all 29 names.

    This is the whole basis of the decision: if the two predicates could ever
    disagree, "does the firmware define this?" would stop predicting "would
    adding it collide?".
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
