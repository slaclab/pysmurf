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
# --cryo-det, or a pyrogue ZIP with --zip -- and serves it on a local port, then
# connects to it exactly as a client connects to a deployed server.
#
# The package has to be one whose tuning operations the server attaches rather
# than the firmware: a package that defines them itself is refused before a tree
# exists, and every released package so far defines them. So --zip takes a ZIP
# built from a package where they have been removed, and --cryo-det a checkout of
# one; a release off the shelf will not work here and says so when tried.
# There is one route in and it is the real one: no in-process shortcut, so what
# is exercised here is the code that talks to a crate.
#
# Run it once per platform: the ATCA carrier by default, the RFSoC with --rfsoc.
# The RFSoC firmware's own package is a subclass of this one that does nothing but
# default isRFSOC on, so setting the flag here builds that platform's tree without
# needing its repository on the path -- and what the flag changes is the JESD and
# signal-generator configuration, which is why that platform has no RF front end and
# no serial links to one. Both are bay-indexed even so, because the acquisition mux
# is; what differs is what sits inside a bay. The scope check names what each platform
# is expected to have and fails if either changes.
#
# An emulated register space reads back zeros, so the build stamp a platform is
# identified by is blank. The stamp of the platform being built is therefore
# written into the emulated memory before the client connects, so that
# identification here runs the same way it runs against a crate rather than being
# handed the answer, and so that the RFSoC run identifies as an RFSoC.
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
import contextlib
import os
import socket
import subprocess
import sys

import pyrogue as pr

import cryodaq
from cryodaq import platform

# How long a tuning process gets before the bounded wait gives up. Nothing here
# has resonators to find, so a run that has not ended in this long is a stall,
# not slow progress.
PROCESS_TIMEOUT_S = 30.0

# The band every session check works on, chosen by main() from the bands the tree
# reports. One is enough: the names are per band by construction and the map
# checks resolve all of them on every band. It is not band 0 by assumption --
# which band a tree has is the tree's business, and a build without band 0 is a
# tree this client is expected to work on.
BAND = None

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

# Which platform each run is expected to be, and the firmware image whose name
# says so -- both keyed by args.rfsoc. The image is one that platform really has;
# a name no map claims would be refused, which is the point of the check.
EXPECTED_PLATFORM = {False: 'umux-atca', True: 'umux-rfsoc'}
EXPECTED_IMAGE = {False: 'MicrowaveMuxBpEthGen2',
                  True: 'MicrowaveMuxZcu208_BaseBand'}

# What each is expected to have, so that the difference between them is asserted by
# name rather than noticed. Both platforms have bays: the acquisition mux is indexed
# by bay on either, which is why the scope is shared. What only the carrier has is
# what sits inside a bay -- the RF front end carrying the attenuators, and the serial
# links back from it -- so the difference is asserted there rather than on the scope.
EXPECTED_FRONT_END = {False: True, True: False}      # keyed by args.rfsoc

# The bands both trees have: the two are built from one firmware package, which
# defines eight either way. Asserted rather than derived, so a package that
# stopped defining one would fail here instead of quietly narrowing every check
# that iterates the bands. A tree whose bands are sparse or start above zero is
# legal and is covered by check_platform_map.py; this is what these trees are.
EXPECTED_BANDS = tuple(range(8))

# How long a child interpreter that opened a session gets to be gone. Generous
# on purpose: connecting to this tree takes about a second and the close at exit
# waits on the monitor thread, so anything near this bound is a hang and not a
# slow machine.
EXIT_DEADLINE_S = 30

# The one name below whose register the server owns and only reports: it is the
# state a configuration run leaves, so the tree is put in that state from the
# server's side and a client asking to write it is refused.
CONFIGURED_NAME = 'application.configured'
CONFIGURED_PATH = 'AMCc.SmurfApplication.SystemConfigured'

# Set by main() once the tree is up and a session is open.
SESSION = None
ROOT = None
ENDPOINT = None
RFSOC = False


def stamp_for(image):
    """A build stamp in the shape the firmware reports one, for ``image``."""
    return (f"{image}: Vivado v2020.2, emulated (no host), "
            f"Built Thu Jan  1 00:00:00 AM UTC 1970 by nobody")


@contextlib.contextmanager
def blank_build_stamp():
    """Leave the tree reporting no firmware, as an untouched emulation does."""
    write_build_stamp(ROOT, '\x00' * 256)
    try:
        yield
    finally:
        write_build_stamp(ROOT, stamp_for(EXPECTED_IMAGE[RFSOC]))


# --------------------------------------------------------------------------
# the map against this tree
# --------------------------------------------------------------------------

def check_the_platform_map_was_identified():
    """The tree was recognised by its firmware, and the session says which map it uses."""
    assert SESSION.pmap.name == EXPECTED_PLATFORM[RFSOC], SESSION.pmap.name
    assert len(SESSION.pmap) >= 60, f"only {len(SESSION.pmap)} patterns in the map"
    assert SESSION.description['platform'] == SESSION.pmap.name


def check_every_offered_name_resolves():
    """Every name this tree offers reaches a node in it.

    Except the ones an attached device brings: the point-of-load regulator is reached
    over I2C and is added by a *server* talking to real hardware, so an emulated tree
    built from a firmware package has none of it. Excluded by name rather than by
    letting the check pass on an empty set, and required to stay excluded -- a name
    that starts resolving here has stopped being server-attached and should lose its
    exemption.
    """
    report = SESSION.validate()
    attached = [name for name, _why in report.unresolved
                if name.startswith('carrier.regulator.')]
    assert attached, ('the regulator names resolve on an emulated tree, so they are no '
                      'longer server-attached and the exemption below is stale')
    unresolved = [(name, why) for name, why in report.unresolved
                  if not name.startswith('carrier.regulator.')]
    detail = '; '.join(f"{name}: {why}" for name, why in unresolved[:8])
    assert not unresolved, f"{len(unresolved)} unresolved: {detail}"
    assert len(report.resolved) >= 60, len(report.resolved)
    on_band = [name for name in report.resolved if name.startswith(f"band[{BAND}].ops.")]
    assert len(on_band) >= 20, f"only {len(on_band)} operation names on band {BAND}"


def check_node_kinds_match_the_map():
    """A name the map calls a process is a process, a command a command, a value a value."""
    wrong = []
    for name in SESSION.validate().resolved:
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
    """Band and bay indices are discovered, and the expected generation difference holds.

    The bands are asserted against what this package builds, not against a rule
    that they run from zero without gaps: a tree may have neither, and the
    platform layer's own checks cover the sparse and gapped cases.
    """
    bands = SESSION.indices('band')
    assert bands == EXPECTED_BANDS, bands
    # A band's channels run far past one probing window, so the enumeration has to
    # reach the end of what the tree has -- and how many that is comes from the
    # tree, not from this file: the per-channel array's length is the firmware's
    # own count, and what the channel scope enumerates has to agree with it.
    # (`n_channels` is a register, and reads zero over emulated memory.)
    per_channel = len(SESSION.get(f'band[{BAND}].tone.amplitude'))
    channels = SESSION.indices('channel', band=BAND)
    assert per_channel > platform.MAX_SCOPE_INDEX, per_channel
    assert channels == tuple(range(per_channel)), \
        (f"band {BAND} enumerated {len(channels)} channels, last {channels[-1:]}; "
         f"the per-channel array has {per_channel}")
    # Both platforms are bay-indexed, because the acquisition mux is. A tree with no
    # bays at all would mean the scope stopped being probed rather than that this
    # platform lacks the hardware, so it fails on either.
    bays = SESSION.indices('bay')
    assert bays, 'neither platform has a tree without bays; the scope found none'
    if EXPECTED_FRONT_END[RFSOC]:
        # Every carrier bay has a front end, so every bay must enumerate attenuators;
        # a path is a path in both directions, and the scope is proved by either -- so
        # both nodes have to exist on every path the scope found. Asked of the tree
        # through node(), which resolves the name and then looks the node up: the
        # map's path() alone only formats the template and proves nothing about
        # what this tree has.
        for bay in bays:
            atts = SESSION.indices('attenuator', bay=bay)
            assert atts, f"bay {bay} carries no attenuators"
            for att in atts:
                for direction in ('uc', 'dc'):
                    SESSION.node(f'bay[{bay}].attenuator[{att}].{direction}')
    else:
        # No front end, so the scopes it provides are not declared at all -- asking for
        # one is a KeyError and that is the map's statement of what this platform has,
        # rather than a scope that exists and enumerates empty.
        try:
            SESSION.indices('attenuator', bay=bays[0])
        except KeyError:
            pass
        else:
            raise AssertionError(
                "a platform with no RF front end declares the 'attenuator' scope")
        report = SESSION.validate()
        def front_end_name(name):
            return '.attenuator.' in name or '.jesd.' in name

        front_end = [name for name in report.resolved if front_end_name(name)]
        tried = [name for name, _ in report.unresolved if front_end_name(name)]
        assert not front_end + tried, \
            f"front-end names without a front end: {(front_end + tried)[:4]}"


def check_the_witness_set_reads_back():
    """The registers that record how a system was left all read."""
    witness = SESSION.witness()
    assert len(witness) >= 10, len(witness)
    unread = [name for name, value in witness.items() if value is None]
    assert not unread, unread
    per_band = [name for name in witness if name.endswith('.dsp.enable')]
    assert len(per_band) == len(SESSION.indices('band')), per_band


# --------------------------------------------------------------------------
# connecting: what happens before there is a session
#
# These run one at a time, each opening and closing its own session, before the
# long-lived session below exists -- because pyrogue caches a client per address
# and port, so two sessions on one endpoint are one transport and closing either
# closes both. That is a property of the transport rather than of these checks,
# and it is why they cannot simply join the group after it.
# --------------------------------------------------------------------------

def precheck_a_tree_that_reports_no_firmware_is_refused():
    """With nothing in the stamp there is no platform, and connecting says so."""
    with blank_build_stamp():
        try:
            session = cryodaq.connect(ENDPOINT)
        except cryodaq.ConnectError as e:
            assert 'declare' in str(e), f"the error does not say what to do: {e}"
        else:
            session.close()
            raise AssertionError(f"identified as {session.pmap.name} with no firmware")


def precheck_a_platform_can_be_declared_when_the_firmware_cannot_say():
    """The way past the refusal, for a tree with no firmware behind it."""
    with blank_build_stamp():
        with cryodaq.connect(ENDPOINT, platform_name=EXPECTED_PLATFORM[RFSOC]) as session:
            assert session.pmap.name == EXPECTED_PLATFORM[RFSOC], session.pmap.name
            assert session.get('application.configured') is not None


def precheck_a_request_deadline_is_installed_on_the_client():
    """A session opened with a bound still connects and reads.

    What cannot be checked here is the bound firing: rogue takes it write-only
    (there is no accessor to read it back), and tripping it needs a server that
    stops answering. So this holds the deadline to not breaking a live one.
    """
    with cryodaq.connect(ENDPOINT, timeout=20.0) as session:
        assert session.get('application.configured') is not None
    with cryodaq.connect(ENDPOINT, timeout=None) as session:
        assert session.get('application.configured') is not None


def precheck_a_deadline_that_is_not_a_duration_is_refused():
    """A deadline outside what the transport can express is refused, by rogue.

    The client converts seconds to milliseconds and passes them on; the range is
    rogue's, a ``uint32_t`` with a zero rejected outright, so this asserts that no
    such deadline is quietly accepted rather than that any particular exception
    arrives. What is raised varies with how the value fails -- ``math.ceil`` on
    infinity and a NaN, Boost.Python's converter on a negative or one past the
    ceiling, rogue itself on a zero -- and pinning that would be pinning three
    libraries' error text. ``None`` remains the way to ask for no deadline, and a
    session must not survive a refusal: the client is stopped on the way out, so a
    leaked one would show up as the next check finding the transport gone.
    """
    for bad in (0, -1.0, float('inf'), float('-inf'), float('nan'), 5e6, 1e305):
        try:
            session = cryodaq.connect(ENDPOINT, timeout=bad)
        except Exception:                                        # noqa: BLE001
            pass
        else:
            session.close()
            raise AssertionError(f"timeout={bad!r} was accepted")


def precheck_declaring_a_platform_that_does_not_exist_is_refused():
    try:
        session = cryodaq.connect(ENDPOINT, platform_name='no-such-platform')
    except cryodaq.ConnectError as e:
        assert EXPECTED_PLATFORM[RFSOC] in str(e), f"the error does not list what is known: {e}"
    else:
        session.close()
        raise AssertionError('an unknown platform name was accepted')


# --------------------------------------------------------------------------
# the session over the connection
# --------------------------------------------------------------------------

def check_the_platform_is_identified_from_the_firmware():
    """The session found its platform by reading the stamp, not by being told."""
    assert SESSION.pmap.name == EXPECTED_PLATFORM[RFSOC], SESSION.pmap.name
    stamp = SESSION.description['firmware_build_stamp']
    assert platform.tag_of(stamp) in SESSION.pmap.tags, \
        f"identified as {SESSION.pmap.name} from {stamp!r}"


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


def check_a_read_only_name_is_refused():
    """A name the tree only reports cannot be written through the interface.

    Nothing below refuses this: rogue takes the write, drops it silently where a
    firmware register would be built into a transaction, and keeps it where the
    value is the server's own -- which is this name, so the write would land and
    a server would then report a configuration it never ran. The value is read
    back afterwards because that is the half which fails if the refusal goes.
    """
    before = SESSION.get(CONFIGURED_NAME)
    try:
        SESSION.set(CONFIGURED_NAME, not before)
    except cryodaq.UnresolvedName as e:
        assert CONFIGURED_NAME in str(e), f"the error does not name it: {e}"
        assert 'read-only' in str(e), f"the error does not say why: {e}"
    else:
        raise AssertionError(f"{CONFIGURED_NAME} was written through the client")
    assert SESSION.get(CONFIGURED_NAME) == before, 'the refused write landed anyway'


def check_a_read_never_answers_from_a_cache():
    """What the server holds now is what a read returns now.

    Everything that waits on this interface depends on it: ``call(..., wait=…)``
    polls a flag the server owns, and a poll that could answer from something
    held on this side would end a wait on a value from before the process
    started. rogue reads on every ``get`` and the client proxies each one, so a
    value changed underneath -- here on the tree itself, which is the only way
    to change this one -- is visible to the next read with nothing invalidated
    in between.
    """
    before = SESSION.get(CONFIGURED_NAME)
    try:
        set_configured(ROOT, not before)
        assert SESSION.get(CONFIGURED_NAME) == (not before), \
            'a value changed on the server read back as the value before it'
    finally:
        set_configured(ROOT, before)
    assert SESSION.get(CONFIGURED_NAME) == before, 'the restore was not seen either'


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
    and that what happened is readable from the process itself. The bounds
    themselves are then checked, because a wait or a poll interval that cannot
    mean anything has to be refused before a process is started rather than
    after -- otherwise the call has left something running behind it.
    """
    name = f"band[{BAND}].ops.gradient_descent"
    node = SESSION.call(name, wait=PROCESS_TIMEOUT_S)
    assert not node.Running.get(), 'the wait returned with the process still running'
    print(f"          (the run reported: {node.Message.get()!r})")
    # A bound that makes no sense is refused before Start, so the refusal is what
    # the caller gets rather than a ValueError out of time.sleep once the process
    # is already running. Each case names the argument at fault, so a guard that
    # refused the right call for the wrong reason still fails here.
    for kwargs, argument in (({'wait': -1}, 'wait'),
                             ({'wait': float('nan')}, 'wait'),
                             ({'poll': 0}, 'poll'),
                             ({'poll': -0.5}, 'poll'),
                             ({'poll': float('inf')}, 'poll'),
                             ({'wait': 1, 'poll': 0}, 'poll')):
        try:
            SESSION.call(name, **kwargs)
        except ValueError as e:
            assert argument in str(e), \
                f"the error does not name {argument!r}: call({kwargs}) -> {e}"
        else:
            raise AssertionError(f"call({kwargs}) was accepted")
    # Zero is a legal wait: read the flag once, and then either return or report
    # the process still running. Both are documented outcomes and which one comes
    # back is a race with a process that finishes in under a millisecond, so both
    # are accepted here -- asserting only the first would be asserting the timing
    # of the emulator. Infinity is legal too and is deliberately not exercised: a
    # wait with no bound cannot be given a deadline by the thing testing it, so it
    # is asserted in the docstring only.
    try:
        SESSION.call(name, wait=0)
    except TimeoutError:
        pass


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


def check_a_session_left_open_still_lets_the_interpreter_exit():
    """Forgetting to close costs the transport nothing, and the process nothing.

    The link monitor runs in a thread that is not a daemon, so the interpreter
    waits for it on the way out; a session nobody closed would wait for good, and
    the case is an interactive session, where nothing guarantees a ``close``.
    It cannot be checked in this process -- what is being checked is an exit --
    so a child opens a session, returns without closing it, and has to be gone
    before the deadline. A child that hangs is the defect, and it is reported as
    the timeout it is rather than as a failure to connect.
    """
    for kwargs in ({}, {'monitor': False}):
        source = ('import cryodaq\n'
                  f"session = cryodaq.connect({ENDPOINT!r}, **{kwargs!r})\n"
                  "print(session.pmap.name)\n")
        try:
            done = subprocess.run([sys.executable, '-c', source], text=True,
                                  timeout=EXIT_DEADLINE_S, capture_output=True)
        except subprocess.TimeoutExpired:
            raise AssertionError(
                f"a child that connected with {kwargs or 'the defaults'} and never "
                f"closed was still running after {EXIT_DEADLINE_S}s: an unclosed "
                "session holds the interpreter open") from None
        assert done.returncode == 0, \
            f"the child exited {done.returncode}: {done.stderr.strip()[-400:]}"
        assert EXPECTED_PLATFORM[RFSOC] in done.stdout, \
            f"the child never connected: {done.stdout.strip()!r} {done.stderr.strip()[-200:]}"


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


def describe_source(args):
    """Where the tree came from, in one line, for a dump's provenance.

    The checkout's revision or the ZIP's name -- what was measured, read from the
    input rather than asserted: a fixture built from another checkout must say so.
    """
    if args.zip:
        return f"zip {os.path.basename(args.zip)}"
    try:
        head = subprocess.run(['git', '-C', args.cryo_det, 'describe', '--tags',
                               '--always', '--dirty'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        head = 'unknown revision'
    return f"cryo-det {head}" + (' (RFSoC)' if args.rfsoc else '')


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


def write_build_stamp(root, stamp):
    """Put a build stamp in the emulated memory, so the tree can be identified.

    The emulator answers from a dictionary of bytes, and unwritten addresses read
    as zero -- which is why an emulated tree reports no firmware at all. Writing
    the stamp where the register reads from gives identification the same thing to
    work with here as on a crate. The register is read-only from the tree's side,
    as it is in the firmware, so this goes in underneath it.
    """
    node = root.getNode(platform.TAG_PATH)
    assert node is not None, f"the tree has no {platform.TAG_PATH}"
    for offset, byte in enumerate(stamp.encode()):
        root._srp._data[node.address + offset] = byte
    # The block was read once while the tree was built, and cached what it found
    # then: zeros. Without a forced re-read the stamp would never be seen.
    assert platform.tag_of(node.get(read=True)) == platform.tag_of(stamp), \
        'the build stamp did not read back'


def set_configured(root, configured):
    """Leave the tree saying the system is configured, as the server leaves it.

    The flag is the server's own to set -- read-only from a client, as it is from
    anywhere else the value is only reported -- so this goes in on the server
    side of the socket, next to the tree, exactly as the build stamp above does.
    Nothing here runs the configuration sequence; what the checks need is a
    session that finds the state one would have left.
    """
    node = root.getNode(CONFIGURED_PATH)
    assert node is not None, f"the tree has no {CONFIGURED_PATH}"
    node.set(configured)
    assert node.get() is configured, 'the configured flag did not take'


def emulation_root(args, port):
    """Build and start the emulated root over the CryoDet package, on ``port``.

    Returns a started root, which the caller owns and has to stop: a rogue root
    runs threads that are not daemons, so one left started holds the interpreter
    open at exit rather than failing visibly. The build stamp is written by the
    caller, inside that cleanup, for the same reason -- a malformed tree is a
    thing this script exists to report, and reporting it must not hang.
    """
    add_library_paths(args)
    from pysmurf.core.roots.EmulationRoot import EmulationRoot
    # is_rfsoc is the package's own construction flag, and all it does is leave
    # out the JESD lanes and signal generators -- so the tree is this package
    # without its bays, not the RFSoC firmware's tree. See the header.
    root = EmulationRoot(config_file='', polling_en=False, pv_dump_file='',
                         disable_bay0=False, disable_bay1=False,
                         is_rfsoc=args.rfsoc, is_prespectra=False, server_port=port)
    try:
        root.start()
    except Exception:
        # A start that raised still leaves threads behind -- an occupied port is
        # the easy way to see it -- and stop() is what ends them, so the failure
        # is reported by exiting rather than by hanging.
        root.stop()
        raise
    return root


def collect(prefix):
    """The checks whose names start with ``prefix``, in a fixed order."""
    return sorted((name[len(prefix):], fn) for name, fn in globals().items()
                  if name.startswith(prefix))


def run(checks):
    """Run each check, reporting one line apiece; return the names that failed."""
    failed = []
    for name, fn in checks:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            failed.append(name)
            print(f"  FAIL  {name}")
            print(f"          {type(e).__name__}: {e}")
        else:
            print(f"  ok    {name}")
    return failed


def main():
    global SESSION, ROOT, ENDPOINT, RFSOC, BAND
    ap = argparse.ArgumentParser(
        description='Drive a cryodaq session against an emulated firmware tree.')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--cryo-det', help='a CryoDet checkout with the tuning '
                                           'operations removed')
    source.add_argument('--zip', help='a pyrogue ZIP built from such a checkout; a '
                                      'released ZIP still defines the operations '
                                      'and is refused before a tree is built')
    ap.add_argument('--rfsoc', action='store_true',
                    help='the RFSoC platform instead of the ATCA carrier: its own '
                         'firmware package is this one with isRFSOC defaulted on, '
                         'so the flag builds that tree without needing it')
    ap.add_argument('--port', type=int, default=None,
                    help='port to serve the emulated tree on (a free one by default)')
    ap.add_argument('--dump-tree', metavar='PATH',
                    help='also write the full node listing of the tree the checks ran '
                         'on, as saveVariableList() writes it; prune_fixtures.py reads '
                         'these to rebuild the committed fixtures')
    args = ap.parse_args()
    RFSOC = args.rfsoc

    label = ('RFSoC' if args.rfsoc else 'carrier') + ' tree'
    label += ' from ' + (args.zip or args.cryo_det)
    port = args.port or free_port()
    endpoint = ENDPOINT = f"localhost:{port}"
    prechecks = collect('precheck_')
    checks = collect('check_')
    failed = []
    root = ROOT = emulation_root(args, port)
    try:
        # Inside the cleanup, not before it: a stamp that will not write is a
        # failure worth reporting, and it is reported by exiting, which a started
        # root that nobody stopped would prevent.
        write_build_stamp(root, stamp_for(EXPECTED_IMAGE[args.rfsoc]))
        if args.dump_tree:
            root.saveVariableList(args.dump_tree)
            with open(args.dump_tree + '.source', 'w', encoding='utf-8') as fh:
                fh.write(describe_source(args) + '\n')
            print(f"  tree written to {args.dump_tree} (source in .source)")
        print(f"Validating the cryodaq client on the {label} "
              f"({len(prechecks)} + {len(checks)} checks)")
        failed += run(prechecks)
        # Nothing here runs the server's own configuration procedure, so the
        # flag it would leave is set on the server side first, and the session
        # the checks use then finds a configured server as a client would.
        set_configured(root, True)
        with cryodaq.connect(endpoint) as session:
            SESSION = session
            BAND = session.indices('band')[0]
            print(f"  {endpoint}: platform {session.pmap.name} from "
                  f"{platform.tag_of(session.description['firmware_build_stamp'])!r}, "
                  f"{len(session.pmap)} patterns, "
                  f"bands {list(session.indices('band'))} (checks on {BAND}), "
                  f"bays {list(session.indices('bay'))}")
            failed += run(checks)
        set_configured(root, False)
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
