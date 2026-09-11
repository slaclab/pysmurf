#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Emulated Client Validation Script
#-----------------------------------------------------------------------------
# File       : validate_client_emulated.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
# Script to validate the cryodaq client against a firmware register tree with no
# hardware: that every semantic name reaches the node its platform map says it
# does, and that a session can read, write and run over the connection.
#
# It builds an emulated root over a CryoDet package -- a checkout with
# --cryo-det, or a released pyrogue ZIP with --zip -- and serves it on a local
# port, then connects to it exactly as a client connects to a deployed server.
# There is one route in and it is the real one: no in-process shortcut, so what
# is exercised here is the code that talks to a crate.
#
# Run it once per supported tree: the ATCA carrier by default, the RFSoC
# generation with --rfsoc. Both use one map, and their trees differ by omission,
# so the scope check names what each is expected to have and fails if either
# changes.
#
# What it cannot show: that an operation does anything useful. The emulated
# memory reads back zeros, so a tuning process has nothing to tune. What is
# checked is the route -- name to register, name to process, failure to an
# exception that names what could not be resolved.
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
import socket
import sys

import pyrogue as pr

import cryodaq

# How long a tuning process gets before the bounded wait gives up. Nothing here
# has resonators to find, so a run that has not ended in this long is a stall,
# not slow progress.
PROCESS_TIMEOUT_S = 30.0

# The band every session check works on. One is enough: the names are per band by
# construction and the map checks resolve all of them on every band.
BAND = 0

# The twenty names the client contract reaches on every band: the tuning
# operations, their parameters, their results and the flag that says one is
# running. They are listed as names, not paths, because the names are what the
# contract is; a map that stopped offering one would break a caller even if the
# register were still there.
CONTRACT_NAMES = (
    'band[{band}].ops.in_progress',
    'band[{band}].ops.eta_scan.channel',
    'band[{band}].ops.eta_scan.frequencies',
    'band[{band}].ops.eta_scan.results_real',
    'band[{band}].ops.eta_scan.results_imag',
    'band[{band}].ops.eta_scan.delta_f',
    'band[{band}].ops.eta_scan.amplitude',
    'band[{band}].ops.eta_scan.averages',
    'band[{band}].ops.gradient_descent.max_iters',
    'band[{band}].ops.gradient_descent.averages',
    'band[{band}].ops.gradient_descent.gain',
    'band[{band}].ops.gradient_descent.converge_hz',
    'band[{band}].ops.gradient_descent.step_hz',
    'band[{band}].ops.gradient_descent.momentum',
    'band[{band}].ops.gradient_descent.beta',
    'band[{band}].ops.start_gradient_descent',
    'band[{band}].ops.start_eta_scan',
    'band[{band}].ops.start_find_freq',
    'band[{band}].ops.run_eta_scan',
    'band[{band}].ops.load_tune_file',
)

# What each supported tree is expected to have. The two generations share one
# map and differ by omission: an RFSoC has neither the per-bay data links nor
# the RF front end that carries the attenuators, so its bay scope is empty and
# every bay name is withheld rather than offered broken.
EXPECTED_BAYS = {False: True, True: False}      # keyed by args.rfsoc

# Set by main() once the tree is up and a session is open.
SESSION = None
RFSOC = False


# --------------------------------------------------------------------------
# the map against this tree
# --------------------------------------------------------------------------

def check_the_platform_map_was_identified():
    """The tree was recognised by its shape, and the session says which map it uses."""
    assert SESSION.pmap.name == 'umux', SESSION.pmap.name
    assert len(SESSION.pmap) >= 60, f"only {len(SESSION.pmap)} patterns in the map"
    assert SESSION.description['platform'] == SESSION.pmap.name


def check_every_offered_name_resolves():
    """Every name this tree offers reaches a node in it."""
    report = SESSION.validate()
    detail = '; '.join(f"{name}: {why}" for name, why in report.unresolved[:8])
    assert report.ok, f"{len(report.unresolved)} unresolved: {detail}"
    assert len(report.resolved) >= 60, len(report.resolved)
    on_band = [name for name in report.resolved if name.startswith(f"band[{BAND}].ops.")]
    assert len(on_band) >= 20, f"only {len(on_band)} operation names on band {BAND}"


def check_node_kinds_match_the_map():
    """A name the map calls a process is a process, a command a command, a value a value."""
    wrong = []
    for name in SESSION.names():
        declared = SESSION.pmap.kind(name)
        node = SESSION.node(name)
        if node.isinstance(pr.Process):
            found = cryodaq.PROCESS
        elif node.isCommand:
            found = cryodaq.COMMAND
        elif node.isDevice:
            found = 'device'
        else:
            found = cryodaq.VALUE
        if found != declared:
            wrong.append(f"{name}: map says {declared}, tree says {found}")
    assert not wrong, '; '.join(wrong[:8])


def check_the_client_contract_is_covered():
    """Every contract name is offered, on every band this tree has."""
    for band in SESSION.indices('band'):
        for template in CONTRACT_NAMES:
            name = template.format(band=band)
            SESSION.pmap.entry(name)                # raises if the map lacks it
            assert SESSION.node(name) is not None, name


def check_the_scopes_are_the_ones_this_tree_has():
    """Band and bay indices are discovered, and the expected generation difference holds."""
    bands = SESSION.indices('band')
    assert bands == tuple(range(len(bands))) and bands, bands
    bays = SESSION.indices('bay')
    if EXPECTED_BAYS[RFSOC]:
        assert bays, 'a carrier tree has bays with data links or an RF front end'
        attenuated = [b for b in bays if SESSION.indices('uc', bay=b)]
        assert attenuated, f"no bay of {list(bays)} carries attenuators"
        assert SESSION.indices('dc', bay=attenuated[0]), 'up-converters but no down-converters'
    else:
        assert bays == (), f"an RFSoC tree has no bays, found {list(bays)}"
        offered = [name for name in SESSION.names() if name.startswith('bay[')]
        assert not offered, f"bay names offered without bays: {offered[:4]}"


def check_the_witness_set_reads_back():
    """The registers that record how a system was left all read."""
    witness = SESSION.witness()
    assert len(witness) >= 10, len(witness)
    unread = [name for name, value in witness.items() if value is None]
    assert not unread, unread
    per_band = [name for name in witness if name.endswith('.dsp.enable')]
    assert len(per_band) == len(SESSION.indices('band')), per_band


# --------------------------------------------------------------------------
# the session over the connection
# --------------------------------------------------------------------------

def check_the_description_says_what_the_server_is():
    """Connecting recorded where the tree came from and what it says about itself."""
    description = SESSION.description
    for field in ('endpoint', 'platform', 'read_at', 'root', 'configured',
                  'firmware_version', 'firmware_build_stamp'):
        assert field in description, field
    assert description['firmware_version'] is not None, 'the firmware did not identify itself'
    assert description['configured'], 'the server was configured before this session opened'


def check_reading_and_writing_by_name():
    """set() then get() on the same name round trips through the register."""
    name = f"band[{BAND}].ops.gradient_descent.max_iters"
    original = SESSION.get(name)
    try:
        SESSION.set(name, 7)
        assert SESSION.get(name) == 7, SESSION.get(name)
    finally:
        SESSION.set(name, original)


def check_an_array_is_reached_by_index():
    """A per-channel register is one array, and a channel is an index into it."""
    name = f"band[{BAND}].tone.amplitude"
    whole = SESSION.get(name)
    assert len(whole) == SESSION.get(f"band[{BAND}].n_channels"), len(whole)
    original = SESSION.get(name, index=3)
    try:
        SESSION.set(name, 11, index=3)
        assert SESSION.get(name, index=3) == 11, SESSION.get(name, index=3)
    finally:
        SESSION.set(name, original, index=3)


def check_a_command_runs():
    """A name the map calls a command is called, and returns."""
    SESSION.call(f"band[{BAND}].ops.set_amplitude_scales", 0)
    assert SESSION.get(f"band[{BAND}].tone.amplitude", index=0) == 0


def check_a_process_runs_under_a_bounded_wait():
    """A process is started and awaited on the server's own Running flag.

    The emulated memory reads back zeros, so the run has nothing to tune and may
    end in failure; what must hold is that the wait is bounded, that it ends,
    and that what happened is readable from the process itself.
    """
    name = f"band[{BAND}].ops.gradient_descent"
    node = SESSION.call(name, wait=PROCESS_TIMEOUT_S)
    assert not node.Running.get(), 'the wait returned with the process still running'
    print(f"          (the run reported: {node.Message.get()!r})")


def check_both_gradient_descent_implementations_are_named():
    """The selector and both processes are reachable, so which one ran is visible.

    The command that dispatches a gradient descent reads
    ``use_new_gradient_descent`` to choose between two processes. A caller that
    polled only one of them would report success for a process that never ran.
    """
    selector = SESSION.get(f"band[{BAND}].ops.use_new_gradient_descent")
    assert selector in (0, 1, False, True), selector
    for name in ('gradient_descent', 'new_gradient_descent'):
        node = SESSION.node(f"band[{BAND}].ops.{name}")
        assert node.isinstance(pr.Process), name


def check_the_whole_tree_is_reachable():
    """Nothing is hidden: what the map does not name is still there through rogue."""
    assert SESSION.root.name, 'the root has no name'
    process = SESSION.node(f"band[{BAND}].ops.gradient_descent")
    for child in ('Progress', 'Message', 'Running'):
        assert process.node(child) is not None, f"the process has no {child}"
    assert process.Progress.get() is not None, 'a process child the map does not name'
    parameter = SESSION.node(f"band[{BAND}].ops.gradient_descent.max_iters")
    assert parameter.description, 'the tree carries its own documentation'


def check_names_and_kinds_are_refused_when_wrong():
    """A name the map does not have, or used the wrong way, raises rather than passing."""
    for bad in (f"band[{BAND}].ops.nonesuch", 'nonesuch', 'band[99].tone.amplitude',
                'not a name'):
        try:
            SESSION.get(bad)
        except cryodaq.UnresolvedName:
            pass
        else:
            raise AssertionError(f"get({bad!r}) was accepted")
    try:
        SESSION.get(f"band[{BAND}].ops.gradient_descent")
    except cryodaq.UnresolvedName as e:
        assert 'process' in str(e), str(e)
    else:
        raise AssertionError('a process was read as a value')
    try:
        SESSION.call(f"band[{BAND}].tone.amplitude")
    except cryodaq.UnresolvedName as e:
        assert 'value' in str(e), str(e)
    else:
        raise AssertionError('a value was called')


# --------------------------------------------------------------------------

def add_library_paths(args):
    """Put the CryoDet package on pyrogue's library path."""
    if args.zip:
        pr.addLibraryPath(f"{args.zip}/python")
        return
    pr.addLibraryPath(os.path.join(args.cryo_det, 'firmware/python'))
    for sub in ('amc-carrier-core', 'lcls-timing-core', 'surf'):
        path = os.path.join(args.cryo_det, 'firmware/submodules', sub, 'python')
        if os.path.isdir(path):
            pr.addLibraryPath(path)


def free_port():
    """A port triple starting here is free right now; the server takes three."""
    held = []
    try:
        for _ in range(3):
            s = socket.socket()
            s.bind(('localhost', 0 if not held else held[-1][1] + 1))
            held.append((s, s.getsockname()[1]))
    except OSError:
        for s, _ in held:
            s.close()
        return free_port()
    base = held[0][1]
    for s, _ in held:
        s.close()
    return base


def emulation_root(args, port):
    """Build the emulated root over the CryoDet package, serving on ``port``."""
    add_library_paths(args)
    from pysmurf.core.roots.EmulationRoot import EmulationRoot
    root = EmulationRoot(config_file='', polling_en=False, pv_dump_file='',
                         disable_bay0=False, disable_bay1=False,
                         is_rfsoc=args.rfsoc, is_prespectra=False, server_port=port)
    root.start()
    return root


def main():
    global SESSION, RFSOC
    ap = argparse.ArgumentParser(
        description='Drive a cryodaq session against an emulated firmware tree.')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--cryo-det', help='a CryoDet checkout')
    source.add_argument('--zip', help='a released pyrogue ZIP')
    ap.add_argument('--rfsoc', action='store_true',
                    help='the RFSoC tree instead of the carrier')
    ap.add_argument('--port', type=int, default=None,
                    help='port to serve the emulated tree on (a free one by default)')
    args = ap.parse_args()
    RFSOC = args.rfsoc

    label = ('RFSoC' if args.rfsoc else 'carrier') + ' tree from ' + (args.zip or args.cryo_det)
    port = args.port or free_port()
    endpoint = f"localhost:{port}"
    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []
    root = emulation_root(args, port)
    try:
        # Nothing here runs the server's own configuration procedure, so the
        # flag it would leave is set first, through the same interface, and the
        # session the checks use then finds a configured server as a client
        # normally would.
        with cryodaq.connect(endpoint) as boot:
            boot.set('application.configured', True)
        with cryodaq.connect(endpoint) as session:
            SESSION = session
            print(f"Validating the cryodaq client on the {label} "
                  f"({len(session.pmap)} patterns, {len(checks)} checks)")
            print(f"  {endpoint}: bands {list(session.indices('band'))}, "
                  f"bays {list(session.indices('bay'))}")
            for name, fn in checks:
                try:
                    fn()
                except Exception as e:                          # noqa: BLE001
                    failed.append(name)
                    print(f"  FAIL  {name}")
                    print(f"          {type(e).__name__}: {e}")
                else:
                    print(f"  ok    {name}")
            session.set('application.configured', False)
    finally:
        root.stop()

    print("")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
