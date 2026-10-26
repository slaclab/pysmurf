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
# Run it once per platform, named with --platform (the ATCA carrier by default).
# The RFSoC firmware's own package is a subclass of this one that does nothing but
# default isRFSOC on, so that platform's tree is built from the same package with
# the flag set, without needing its repository on the path -- and what the flag
# changes is the JESD and signal-generator configuration, which is why that
# platform has no RF front end and no serial links to one. The RFMC carrier has a
# package of its own (--rfmc, a release ZIP), which carries this one as a submodule
# *with* the tuning operations still in it: the operation-less checkout given here
# goes ahead of it on the path, so the tree is the RFMC's over this CryoDet, and
# that release declares rogue >= 6.15.0 (the server image's), so the RFMC run
# needs an interpreter with it. All three are bay-indexed, because the
# acquisition mux is; what differs is what sits
# inside a bay. The scope check names what each platform is expected to have and
# fails if any changes.
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
import tempfile
import threading
import time
import zipfile

import pyrogue as pr
import pyrogue.interfaces

import cryodaq
import cryodaq.server
import cryodaq.server._firmware
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
# says so -- both keyed by the --platform argument. The image is one that
# platform really has; a name no map claims would be refused, which is the point
# of the check.
EXPECTED_PLATFORM = {'atca': 'umux-atca', 'rfsoc': 'umux-rfsoc', 'rfmc': 'umux-rfmc'}
EXPECTED_IMAGE = {'atca': 'MicrowaveMuxBpEthGen2',
                  'rfsoc': 'MicrowaveMuxZcu208_BaseBand',
                  'rfmc': 'MicrowaveMuxRfmcCarrier'}

# What each is expected to have, so that the difference between them is asserted by
# name rather than noticed. Both platforms have bays: the acquisition mux is indexed
# by bay on either, which is why the scope is shared. What only the carrier has is
# what sits inside a bay -- the RF front end carrying the attenuators, and the serial
# links back from it -- so the difference is asserted there rather than on the scope.
EXPECTED_FRONT_END = {'atca': True, 'rfsoc': False, 'rfmc': True}      # keyed by --platform

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

# How long a child interpreter that builds the SMuRF client against three servers
# gets: each connect takes a few seconds; this bounds a hang.
CLIENT_CHILD_DEADLINE_S = 120

# How long the entry point, run as a child, may take to serve its tree. A full
# emulated tree takes a minute or two on an idle machine and several on a busy
# one; a child that dies ends the wait early, so this bounds a hang, not a run.
ENTRY_POINT_READY_S = 600
# How long that child may take to exit once told to. A server stopped soon after
# it started waits for rogue to finish rendering the operations' large arrays,
# which it does once, after start: about 30 s on an idle machine, measured. The
# bound is three times that, so a server that will not stop is told apart from
# one that is slow to.
ENTRY_POINT_EXIT_S = 90
# The firmware-package arguments this run was given, handed on to the child
# entry point so it builds the same tree. Set in main().
FIRMWARE_ARGS = []
# The PCIe offload card's firmware package (`SmurfPcie`, a release archive or a
# checkout of smurf-pcie), when given: the card module binds its register map,
# so the card check needs it. Without it the check says so and does nothing.
SMURF_PCIE = None

# The one name below whose register the server owns and only reports: it is the
# state a configuration run leaves, so the tree is put in that state from the
# server's side and a client asking to write it is refused.
CONFIGURED_NAME = 'application.configured'
CONFIGURED_PATH = 'AMCc.SmurfApplication.SystemConfigured'

# Set by main() once the tree is up and a session is open.
SESSION = None
ROOT = None
ENDPOINT = None
PLATFORM = 'atca'
# Whether the RFMC firmware package was put on the path (--rfmc); its builder is asked only then.
RFMC_ON_PATH = False


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
        write_build_stamp(ROOT, stamp_for(EXPECTED_IMAGE[PLATFORM]))


# --------------------------------------------------------------------------
# the map against this tree
# --------------------------------------------------------------------------

def check_the_platform_map_was_identified():
    """The tree was recognised by its firmware, and the session says which map it uses."""
    assert SESSION.pmap.name == EXPECTED_PLATFORM[PLATFORM], SESSION.pmap.name
    assert len(SESSION.pmap) >= 60, f"only {len(SESSION.pmap)} patterns in the map"
    assert SESSION.description['platform'] == SESSION.pmap.name


def check_every_offered_name_resolves():
    """Every name this tree offers reaches a node in it.

    Except the ones an attached device brings: the point-of-load regulator is reached
    over I2C and is added by a *server* talking to real hardware, so an emulated tree
    built from a firmware package has none of it. Excluded by name rather than by
    letting the check pass on an empty set, and required to stay excluded -- a name
    that starts resolving here has stopped being server-attached and should lose its
    exemption. The channels are sampled -- first, middle and last of each
    band -- so a register missing from one channel in between is not seen here;
    the committed tree listings cover every path.
    """
    report = SESSION.validate(sample=('channel',))
    attached = [name for name, _why in report.unresolved
                if name.startswith('carrier.regulator.')]
    # The regulator is the AMC carrier's; a platform on another board does not
    # offer the names at all, which is the map's statement and not an exemption.
    has_regulator = 'carrier.regulator.current' in SESSION.pmap
    assert bool(attached) == has_regulator, (
        'the regulator names resolve on an emulated tree, so they are no '
        'longer server-attached and the exemption below is stale' if has_regulator
        else f"{SESSION.pmap.name} offers no regulator, yet names of it went unresolved: {attached}")
    # ... and none of them, not merely not all of them: one regulator name resolving
    # while another stays absent is the same staleness, half-way.
    partly = [name for name in report.resolved if name.startswith('carrier.regulator.')]
    assert not partly, f"server-attached names resolving on an emulated tree: {partly}"
    # Likewise the offload card's link restart, which the application adds only
    # when it runs behind such a card; the emulated server has none.
    assert 'server.restart_rssi' in dict(report.unresolved), \
        "server.restart_rssi resolves on an emulated tree, so it is no longer card-only"
    unresolved = [(name, why) for name, why in report.unresolved
                  if not name.startswith('carrier.regulator.') and name != 'server.restart_rssi']
    detail = '; '.join(f"{name}: {why}" for name, why in unresolved[:8])
    assert not unresolved, f"{len(unresolved)} unresolved: {detail}"
    assert len(report.resolved) >= 60, len(report.resolved)
    on_band = [name for name in report.resolved if name.startswith(f"band[{BAND}].ops.")]
    assert len(on_band) >= 20, f"only {len(on_band)} operation names on band {BAND}"


def check_node_kinds_match_the_map():
    """A name the map calls a process is a process, a command a command, a value a value."""
    wrong = []
    for name in SESSION.validate(sample=('channel',)).resolved:
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
    if EXPECTED_FRONT_END[PLATFORM]:
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
        report = SESSION.validate(sample=('channel',))
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
        with cryodaq.connect(ENDPOINT, platform_name=EXPECTED_PLATFORM[PLATFORM]) as session:
            assert session.pmap.name == EXPECTED_PLATFORM[PLATFORM], session.pmap.name
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


def precheck_a_relative_layer_name_means_the_archive_and_nothing_else():
    """With a release archive, a relative configuration name is a member of it, never a local file.

    Deployments name the defaults a package ships by their bare file name; a file
    of the same name in the working directory must not be taken instead, since
    it would configure the registers from the wrong place without a word. An
    absolute name is used as given, and so is a relative one with no archive to
    look in. Asserted here on a throwaway archive beside a same-named local file.
    """
    import zipfile
    from cryodaq.server._firmware import resolve_layers
    # The directory is the platform's: each package keeps its files in its own.
    config_dir = platform.by_name(EXPECTED_PLATFORM[PLATFORM]).package_config_dir
    assert config_dir and not config_dir.startswith('/'), config_dir
    with tempfile.TemporaryDirectory() as d:
        archive = os.path.join(d, 'fw.zip')
        with zipfile.ZipFile(archive, 'w') as zf:
            zf.writestr(f'{config_dir}/defaults.yml', 'in: archive\n')
        local = os.path.join(d, 'defaults.yml')
        with open(local, 'w') as f:
            f.write('in: directory\n')
        here = os.getcwd()
        os.chdir(d)
        try:
            resolved = resolve_layers(['defaults.yml'], archive, config_dir)
            assert resolved == [f'{archive}/{config_dir}/defaults.yml'], \
                f"a relative name beside a same-named local file resolved to {resolved}"
            assert resolve_layers([local], archive, config_dir) == [local], \
                "an absolute name was not used as given"
            assert resolve_layers(['defaults.yml'], None, config_dir) == ['defaults.yml'], \
                "with no archive a relative name was not used as given"
            try:
                resolve_layers(['missing.yml'], archive, config_dir)
            except FileNotFoundError:
                pass
            else:
                raise AssertionError("a name the archive lacks was accepted")
        finally:
            os.chdir(here)


def precheck_declaring_a_platform_that_does_not_exist_is_refused():
    try:
        session = cryodaq.connect(ENDPOINT, platform_name='no-such-platform')
    except cryodaq.ConnectError as e:
        assert EXPECTED_PLATFORM[PLATFORM] in str(e), f"the error does not list what is known: {e}"
    else:
        session.close()
        raise AssertionError('an unknown platform name was accepted')


# --------------------------------------------------------------------------
# the session over the connection
# --------------------------------------------------------------------------

def check_the_platform_is_identified_from_the_firmware():
    """The session found its platform by reading the stamp, not by being told."""
    assert SESSION.pmap.name == EXPECTED_PLATFORM[PLATFORM], SESSION.pmap.name
    stamp = SESSION.description['firmware_build_stamp']
    assert platform.tag_of(stamp) in SESSION.pmap.tags, \
        f"identified as {SESSION.pmap.name} from {stamp!r}"


def check_the_description_says_what_the_server_is():
    """Connecting recorded where the tree came from and what it says about itself."""
    description = SESSION.description
    for field in ('endpoint', 'platform', 'read_at', 'root',
                  'firmware_version', 'firmware_build_stamp'):
        assert field in description, field
    assert description['firmware_version'] is not None, 'the firmware did not identify itself'
    # Identity only: what the server is doing is not cached on the session.
    for live in ('configured', 'configuring', 'ready', 'jesd_status', 'resolved_config_hash'):
        assert live not in description, f"{live} is state, not identity"
    assert SESSION.configured, 'the server was configured before this session opened'


def check_a_recorded_configuration_is_read_back_from_the_server():
    """record_config() writes the ApplicationConfig nodes the real server attaches; resolved_config() reads them.

    The tree here is the server's own root, so the ApplicationConfig device is the
    one a live server has, not a stand-in; this is where the semantic names
    for it are shown to reach real nodes, and where the answer is shown to come
    from the server and not from the record on disk it also wrote.
    """
    import tempfile
    from cryodaq import config
    resolved = config.Resolved(values={'probe': {'value': 1}}, provenance={'probe.value': ('here', 1)},
                               hash=config.hash_of({'probe': {'value': 1}}), layers=('here',))
    status = tempfile.mkdtemp(prefix='cryodaq_t1_status_')
    saved_paths, saved_hash = SESSION.paths, SESSION.get('application_config.hash')
    assert not saved_hash, f"a fresh emulated server already carries {saved_hash}"
    SESSION.paths = cryodaq.Paths(status, status, status, status, status)
    try:
        path = SESSION.record_config(resolved, extra={'source': 't1'})
        assert path.is_file(), path
        assert SESSION.get('application_config.hash') == resolved.hash
        again = SESSION.resolved_config()
        assert again is not None and again.hash == resolved.hash and again.values == resolved.values
        # The server is what answers: with the file gone the answer is unchanged.
        path.unlink()
        assert SESSION.resolved_config().hash == resolved.hash, 'the record did not come from the server'
        # And the configuring operation empties it first, so a setup that fails
        # part-way leaves nothing to reattach to while `configured` stays true.
        SESSION.clear_config()
        assert SESSION.configured and SESSION.resolved_config() is None, 'cleared, yet still offered'
        assert SESSION.get('application_config.hash') == ''
    finally:
        SESSION.paths = saved_paths
        for name in ('application_config.resolved', 'application_config.hash', 'application_config.written_at'):
            SESSION.set(name, '')


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


# How many times a child is run, per setting, for the exit check below. The abort
# it guards against is intermittent -- measured at 11 of 60 runs with the client
# left to its destructors -- so one clean exit is no evidence; at this count a rate
# that high escapes with probability 0.82**25, under one in a hundred, and each
# child costs two seconds.
EXIT_TRIALS = 25


def probe_transport(srp, link_nodes):
    """A transport with only what the platform probe reads: a register path and its link nodes."""
    from cryodaq.server import _transport
    return _transport.Transport(name='stand-in', srp=srp, ddr_streams=[], streaming_stream=None,
                                register_nodes=list(link_nodes))


def check_the_platform_probe_opens_the_link_before_reading():
    """The server reads the build stamp over a link it has opened, and closes it after.

    A transport's register path may need starting before it answers -- the RSSI
    link does, and it opens when the tree holding it starts. The platform probe
    runs before that tree exists, so it has to start the link itself around its
    one read and stop it again for the real tree to open. An emulated memory needs
    no link, so nothing else here would notice a probe that read over a closed
    one; the transport here is an emulated memory with a stamp planted in it and
    a stand-in link that records when it is started and stopped, and the order is
    what is asserted.
    """
    import pyrogue.interfaces.simulation
    events = []

    class Link(pr.Device):
        def __init__(self):
            pr.Device.__init__(self, name='link')

        def _start(self):
            events.append('start')
            pr.Device._start(self)

        def _stop(self):
            pr.Device._stop(self)
            events.append('stop')

    srp = pyrogue.interfaces.simulation.MemEmulate()
    address = ROOT.getNode(platform.by_name(EXPECTED_PLATFORM[PLATFORM]).tag_path).address
    for offset, byte in enumerate(stamp_for(EXPECTED_IMAGE[PLATFORM]).encode()):
        srp._data[address + offset] = byte
    original = platform.from_stamp

    def from_stamp(stamp):
        events.append('read')
        return original(stamp)
    platform.from_stamp = from_stamp
    try:
        pmap = cryodaq.server._firmware.probe_platform(probe_transport(srp, [Link()]))
    finally:
        platform.from_stamp = original
    assert pmap.name == EXPECTED_PLATFORM[PLATFORM], pmap.name
    assert events == ['start', 'read', 'stop'], f"the probe did not bracket its read: {events}"


def check_an_interrupted_platform_probe_closes_what_it_opened():
    """A probe interrupted mid-start stops the link it opened and leaves no thread.

    The probe is a root of its own: it starts the transport's link node, then
    its tree, whose threads are not daemons, and its context manager's exit
    never runs for an entry that raised. An interrupt is not an ``Exception``,
    so a guard that catches only those would let Ctrl-C leave the link open and
    the probe's threads holding the process. Two places it lands: inside the
    link's own start -- the RSSI link waits there, engine running, for a far end
    that may never answer, so the link must be stopped though its start never
    returned -- and inside the tree's start, after rogue has launched its
    threads (a node of the probe's tree raises: the tree must be stopped, the
    link with it). No non-daemon thread may survive either. A probe the guard
    failed to stop is stopped here, so a regression is reported as a failure
    and not as a run that never exits.
    """
    import pyrogue.interfaces.simulation
    events = []
    probes = []

    class Link(pr.Device):
        def __init__(self, name, interrupt=False):
            pr.Device.__init__(self, name=name)
            self._interrupt = interrupt

        def _start(self):
            pr.Device._start(self)
            events.append(f'start {self.name}')
            if self._interrupt:
                # The engine is up; the wait for the far end is where Ctrl-C lands.
                raise KeyboardInterrupt

        def _stop(self):
            pr.Device._stop(self)
            events.append(f'stop {self.name}')

    probe_cls = cryodaq.server._firmware._Probe

    class Probe(probe_cls):
        interrupt_in_tree = False

        def __init__(self, srp, link_nodes=(), offset=0):
            probe_cls.__init__(self, srp, link_nodes, offset)
            probes.append(self)
            if self.interrupt_in_tree:
                self.add(Link('in_tree', interrupt=True))

    def interrupted(link_nodes, in_tree):
        srp = pyrogue.interfaces.simulation.MemEmulate()
        before = set(threading.enumerate())
        Probe.interrupt_in_tree = in_tree
        cryodaq.server._firmware._Probe = Probe
        try:
            cryodaq.server._firmware.probe_platform(probe_transport(srp, link_nodes))
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("the probe ran to completion over a start that was interrupted")
        finally:
            cryodaq.server._firmware._Probe = probe_cls
        return [t.name for t in threading.enumerate() if t not in before and not t.daemon]

    try:
        left = interrupted([Link('link', interrupt=True)], in_tree=False)
        assert events == ['start link', 'stop link'], f"the probe interrupted in its link left: {events}"
        assert left == [], f"a probe interrupted in its link left threads running: {left}"
        events.clear()
        left = interrupted([Link('link')], in_tree=True)
        assert events == ['start link', 'start in_tree', 'stop in_tree', 'stop link'], \
            f"the probe interrupted in its tree left: {events}"
        assert left == [], f"a probe interrupted in its tree left threads running: {left}"
    finally:
        for probe in probes:
            if probe.running:
                probe.stop()


def check_the_bridge_transport_has_the_shape_the_rfmc_needs():
    """The TCP bridge: registers to the processor system, data as a carrier's, a second path for the converter.

    No bridge answers here: the transport is built against addresses nothing
    listens on and its parts judged without a board -- rogue's TCP client
    tolerates an absent server (it reconnects in the background until a
    transaction needs it). The register path is the client to the processor
    system's register port with the FPGA's window offset; the converter's path
    is a second client to its driver's port; the data path is Ethernet's (RSSI
    DDR streams and the plain UDP stream, pointed at the FPGA's own address) or
    the card's DMA when asked, and nothing else differs. The command line
    builds the same, and refuses to without both addresses.
    """
    import rogue.interfaces.memory
    from cryodaq.server import _cli, _transport
    # Documentation addresses (TEST-NET-1, RFC 5737): nothing answers and nothing
    # on the crate's subnet is touched. rogue's TCP client connects in the
    # background as soon as it is built and keeps retrying, so a real address
    # here would knock on a real door.
    PS, PL = '192.0.2.203', '192.0.2.103'
    for data in ('udp', 'pcie'):
        if data == 'pcie' and not os.path.exists('/dev/datadev_0'):
            # The DMA data path opens the card's devices; without a card in the
            # machine the udp half is what can be built here, and is.
            continue
        t = _transport.tcp_bridge(PS, PL, data=data, offload_card=False)
        assert t.name == 'tcp-bridge'
        assert isinstance(t.srp, rogue.interfaces.memory.TcpClient), type(t.srp)
        assert isinstance(t.srp_rfdc, rogue.interfaces.memory.TcpClient), type(t.srp_rfdc)
        assert t.srp is not t.srp_rfdc, 'one client serves both the registers and the converter'
        assert t.register_offset == _transport.BRIDGE_REGISTER_OFFSET == 0x4_0000_0000
        assert t.probes_firmware, 'a bridge reaches hardware, so the platform is read from it'
        assert len(t.ddr_streams) == len(_transport.DDR_TDEST) == 4
        assert not t.register_nodes, 'the bridge needs no link started before it answers'
        if data == 'udp':
            assert len(t.nodes) == 2, f"the RSSI link and the UDP receiver: {t.nodes}"
            assert (t.nodes[0]._host, t.nodes[0]._port) == (PL, _transport.RSSI_PORT), \
                'the data path is not pointed at the FPGA'
    try:
        _transport.tcp_bridge(PS, PL, data='rdma')
    except ValueError as e:
        assert 'udp' in str(e) and 'pcie' in str(e), e
    else:
        raise AssertionError('an unknown data path was accepted')
    # The other transports gained nothing: no converter path, no window offset.
    e = _transport.emulation(rfdc=False)
    assert e.srp_rfdc is None and e.register_offset == 0
    assert _transport.emulation().srp_rfdc is not None, 'emulation has no memory for the converter'
    # The command line's arm, with its two addresses required.
    import argparse
    parser = _cli.add_arguments(argparse.ArgumentParser())
    args = parser.parse_args(['--transport', 'tcp-bridge', '--ps-addr', PS, '--addr', PL, '--no-pcie-card'])
    t = _cli.transport_from(args)
    assert t.name == 'tcp-bridge' and isinstance(t.srp_rfdc, rogue.interfaces.memory.TcpClient)
    for argv in (['--transport', 'tcp-bridge', '--addr', PL],
                 ['--transport', 'tcp-bridge', '--ps-addr', PS],
                 ['--transport', 'tcp-bridge', '--ps-addr', 'nowhere', '--addr', PL]):
        try:
            _cli.transport_from(parser.parse_args(argv))
        except SystemExit as e:
            assert 'addr' in str(e), e
        else:
            raise AssertionError(f'{argv} built a bridge transport')


def check_every_builder_answers_for_the_transports_register_window():
    """A top level is built where the transport puts the FPGA's registers, or the platform refuses the transport.

    The window is the transport's fact: 0 over a link to the FPGA itself, the
    processor system's window over the bridge. A builder that silently built
    at 0 over a transport with a window elsewhere would address a tree the
    probe had identified somewhere else. So each platform's builder is asked,
    bare (nothing composed), over a transport with the bridge's offset: the
    RFMC's class takes the offset and its tree must start there; the carrier
    classes have no such argument (a rogue device's offset is fixed when it is
    built), so theirs must refuse, naming the platform and the window. The
    RFSoC's builder is asked as `firmware_support` stands it in, and the
    stand-in top level is asked too, since every check over `standin_root`
    relies on it. The RFMC's package is on the path only in the run given it
    (``--rfmc``); the other runs ask the two carrier builders.
    """
    from cryodaq.server import _transport
    saved = PLATFORM
    try:
        for key, name in EXPECTED_PLATFORM.items():
            globals()['PLATFORM'] = key
            supports = [StandInSupport()]
            if key != 'rfmc' or RFMC_ON_PATH:
                supports.append(firmware_support())
            for support in supports:
                transport = cryodaq.server.emulation()
                transport.register_offset = _transport.BRIDGE_REGISTER_OFFSET
                try:
                    fpga = support.top_level(transport)
                except cryodaq.ConnectError as e:
                    assert isinstance(support, StandInSupport) is False and key != 'rfmc', \
                        f"{name}: {type(support).__name__} refused the window: {e}"
                    assert name in str(e) and f"{_transport.BRIDGE_REGISTER_OFFSET:#x}" in str(e), e
                    continue
                assert key == 'rfmc' or isinstance(support, StandInSupport), \
                    f"{name}'s builder built a tree over a transport with a window at " \
                    f"{transport.register_offset:#x} without refusing it"
                assert fpga.offset == _transport.BRIDGE_REGISTER_OFFSET, \
                    f"{name}: {type(support).__name__} starts the tree at {fpga.offset:#x}, " \
                    f"not the transport's {_transport.BRIDGE_REGISTER_OFFSET:#x}"
    finally:
        globals()['PLATFORM'] = saved


def check_the_rfmc_composition_carries_its_converter_and_clock_file():
    """On the RFMC the root has the RF data converter beside the FPGA, and its procedure finds the package's LMK file.

    Both come from the platform's own support module: the converter is built over
    the transport's second register path and sits where the map says; the
    procedure resolves the clock file in the firmware package, the one the
    release ships. A transport without the second path is refused for this
    platform with the reason. On the other platforms the root carries neither.
    """
    from cryodaq.platform import rfmc
    root = ROOT
    if PLATFORM != 'rfmc':
        assert root.getNode(rfmc.RFDC) is None, f"{PLATFORM} carries an RF data converter"
        assert not hasattr(root.setup, 'lmk_config')
        return
    rfdc = root.getNode(rfmc.RFDC)
    assert rfdc is not None, 'the RFMC root has no RF data converter'
    assert rfdc.parent is root, 'the converter is not a device beside the FPGA'
    assert type(rfdc).__name__ == 'Rfdc', type(rfdc)
    # What the procedure drives: the multi-tile sync block, and the device's own
    # initialisation, which its class provides as a method rather than a node.
    assert rfdc.node('Mts') is not None and callable(rfdc.Init)
    # The clock file is the release's, extracted where the chip's driver can
    # open it, and is what TI's tool exports: register address and value per line.
    lmk = root.setup.lmk_config()
    assert lmk.endswith('HexRegisterValues.txt'), lmk
    with open(lmk) as fh:
        first = fh.readline().split()
    # `R0 (INIT) 0x000090`: the register, an optional note, the value -- which is
    # how the chip's driver reads it too.
    assert first[0].startswith('R') and int(first[-1], 16) >= 0, first
    import pysmurf.core.emulators
    import pysmurf.core.server
    transport = cryodaq.server.emulation(pysmurf.core.emulators.StreamDataSource(), rfdc=False)
    try:
        pysmurf.core.server.compose(transport=transport, platform_name='umux-rfmc',
                                    server_port=free_port(), polling=False)
    except cryodaq.ConnectError as e:
        assert 'srp_rfdc' in str(e) and 'emulation' in str(e), e
    else:
        raise AssertionError('an RFMC server was composed with no path to its converter')


def check_the_offload_card_is_configured_around_the_server():
    """A transport through the PCIe card opens its lane before the firmware is probed, and closes it after.

    The card's RSSI engine for the lane is what carries the register path to the
    carrier; until it is pointed at the carrier and opened, nothing can be read,
    and after a server exits a lane left open is what the next one finds closed.
    What the lane registers must say, and the Ethernet case, are
    `check_offload_card.py`'s and run here first over the same package; what only
    a composed server can show is asserted after: the lane is open before the
    composition reaches the firmware, the tree offers the link restart, and
    stopping the root closes the lane. The card is an emulated memory with its
    firmware's map bound over it; the probe's memory is separate.
    """
    import pysmurf.core.emulators
    if SMURF_PCIE is None:
        raise Skipped("no --smurf-pcie package given: the card's register map cannot be bound here")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import check_offload_card as coc
    failed = coc.run_all(SMURF_PCIE, on_path=True)
    assert failed == 0, f"{failed} of the card's own checks failed"

    events = []
    transport = cryodaq.server.emulation(pysmurf.core.emulators.StreamDataSource())
    transport.card = coc.recording(events)(coc.LANE, coc.CARRIER, memory=coc.card_memories())
    original = platform.by_name

    def by_name(name):
        events.append('platform')
        return original(name)
    platform.by_name = by_name
    try:
        # No operations on this root: the card is the transport's, not theirs,
        # and a started tree carrying the operations' large arrays takes half a
        # minute to stop while rogue renders them.
        root = standin_root(transport=transport, providers=())
    finally:
        platform.by_name = original
    # Emulation names its platform where hardware would be probed; the card has
    # to be open before either.
    assert events == [f'open {coc.LANE}', 'platform'], \
        f"the lane was not opened once, before the platform step: {events}"
    try:
        restart = root.getNode(root.pmap.path('server.restart_rssi'))
        assert restart is not None, "no RestartRssi command on a server behind a card"
        with root:
            restart()
            assert events[-2:] == [f'close {coc.LANE}', f'open {coc.LANE}'], \
                f"restart did not close and reopen: {events}"
    finally:
        root.release()      # a failed assertion must not leave the card's lane open
    assert events[-1] == f'close {coc.LANE}', f"stopping the root did not close the lane: {events}"
    assert transport.card._regs is None, "the card's register access was left open"


def check_the_server_reports_the_bays_it_was_built_with():
    """``EnabledBays`` is what the tree holds, not what a launcher said.

    A bay the composition leaves out keeps its DAQ mux but loses its converter
    front end, so a tree built with one bay reports one; a platform whose
    converters share the FPGA's die has no front end and reports none. A client
    addresses each reported bay's registers, so a bay reported and absent is a
    failure it meets at setup. Built here with bay 1 left out, on the platform
    under test.
    """
    import pysmurf.core.emulators
    import pysmurf.core.server
    if PLATFORM == 'rfmc':
        # The RFMC's package builds both quads unconditionally: there is no bay
        # to leave out, so what is checked is that both are reported off the
        # tree in hand -- by their front ends, which is what populates a bay.
        reported = list(ROOT.getNode(f'{platform.STATUS_DEVICE_PATH}.EnabledBays').get())
        assert reported == [0, 1], f"both quads are on the tree; reported {reported}"
        return
    root = pysmurf.core.server.compose(
        transport=cryodaq.server.emulation(pysmurf.core.emulators.StreamDataSource()),
        platform_name=EXPECTED_PLATFORM[PLATFORM], server_port=free_port(), polling=False,
        support=firmware_support(), top_level_options=dict(disableBay1=True))
    expected = [0] if EXPECTED_FRONT_END[PLATFORM] else []
    with root:
        reported = list(root.getNode(f'{platform.STATUS_DEVICE_PATH}.EnabledBays').get())
        assert reported == expected, f"bay 1 left out: reported {reported}, expected {expected}"
        for bay in reported:
            path = root.pmap.path(f'bay[{bay}].debug.enable')
            assert root.getNode(path) is not None, f"reported bay {bay} has no front end at {path}"


def check_the_application_server_has_its_capture_receivers_by_default():
    """``pysmurf.core.server.compose()`` builds the capture receivers unless told not to.

    The client's ADC and DAC reads go through the debug capture receivers, so a
    SMuRF server without them is one those reads fail against. The readout core
    alone builds none unless asked; the command line asks for the deployed
    depth, and a server composed from code -- a streaming application's -- must
    get the same tree without having to know the number. An explicit 0 still
    leaves them out.
    """
    def captures(root):
        return platform.indices(root.pmap, lambda path: root.getNode(path) is not None, 'capture')

    by_default = standin_root()
    try:
        found = captures(by_default)
        assert found == (0, 1, 2, 3), f"composed without a word, the server has captures {found}"
        depth = len(by_default.getNode(by_default.pmap.path('stream.capture[0].data')).value())
        assert depth == cryodaq.server.CAPTURE_POINTS, f"the default receiver buffers {depth} points"
    finally:
        by_default.release()
    without = standin_root(capture_points=0)
    try:
        assert captures(without) == (), f"asked for none, the server has captures {captures(without)}"
    finally:
        without.release()


def check_the_application_client_refuses_a_server_without_its_application():
    """``SmurfControl`` connects to a server pysmurf composed and refuses one it did not.

    A readout server names the application that composed it
    (``application.name``: ``'pysmurf'`` from ``pysmurf.core.server``, ``''``
    from the readout core alone). The SMuRF client is that application's; on
    the reduced server every method would fail at some register, so it is
    refused at connect, naming the reason. A server from before the name
    existed says nothing, and is a pysmurf server that cannot say so: it is
    used, with a line the user sees -- a client at this version must keep working against
    the servers deployed today. Three servers, then: the session's full one; a
    reduced one composed beside it with the core's ``compose()``; and a full one
    with the name taken off its status device before it starts, standing in for
    the older server. The client is built in a child interpreter --
    ``SmurfControl`` writes logs and reads its cfg, which this harness does not
    want in its own process -- and the verdicts come back on its stdout.
    """
    probe = subprocess.run([sys.executable, '-c', 'import pysmurf.client'], capture_output=True, text=True)
    if probe.returncode != 0:
        # The SMuRF client needs its own dependencies (schema, matplotlib ...)
        # beside rogue; an interpreter without them cannot build one.
        raise Skipped("pysmurf.client does not import in this interpreter: "
                      f"{probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else 'no detail'}")
    # No operations on either: what the client reads to decide is the status
    # device, and the operations' arrays make a started tree slow to stop.
    reduced = standin_root(reduced=True, transport=cryodaq.server.emulation(), providers=())
    older = standin_root(providers=())
    # A server built before the name existed has no such node; this one loses
    # it before the tree is sealed by start(). Reaching into the device's node
    # table is what it takes, since a built tree offers no removal.
    del older.status._nodes[older.status.Application.name]
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cfg_files', 'b33',
                       'experiment_2xLB_testing_Feb2021.cfg')
    source = (
        "import matplotlib; matplotlib.use('Agg')\n"
        "import os, tempfile\n"
        "import cryodaq, pysmurf.client\n"
        # The cfg names directories under /data, which this interpreter may not
        # have; a copy points them under a directory of the run's own.
        "tmp = tempfile.mkdtemp()\n"
        f"text = open({cfg!r}).read().replace('/data', tmp + '/data')\n"
        "cfg = os.path.join(tmp, 'experiment.cfg'); open(cfg, 'w').write(text)\n"
        "def attempt(port):\n"
        "    try:\n"
        "        S = pysmurf.client.SmurfControl(cfg_file=cfg, server_port=port, setup=False,\n"
        "                                        atca_monitor=False, make_logfile=False, offline=False,\n"
        "                                        data_dir=tmp + '/data/smurf_data')\n"
        "        return 'connected ' + repr(S._session.description['application'])\n"
        "    except cryodaq.ConnectError as e:\n"
        "        return 'refused ' + str(e)\n"
        f"print('FULL', attempt({ENDPOINT.rsplit(':', 1)[1]}))\n"
        "print('REDUCED', attempt(int(__import__('sys').argv[1])))\n"
        "print('OLDER', attempt(int(__import__('sys').argv[2])))\n")
    with reduced, older:
        # The client identifies the platform from the stamp before anything else,
        # so these trees get one too, as the session's tree did.
        for root in (reduced, older):
            write_build_stamp(root, stamp_for(EXPECTED_IMAGE[PLATFORM]))
        assert older.getNode(older.pmap.path('application.name')) is None, \
            "the stand-in for an older server still carries the application name"
        ports = [root.zmqServer.address.rsplit(':', 1)[1] for root in (reduced, older)]
        done = subprocess.run([sys.executable, '-c', source] + ports, text=True,
                              timeout=CLIENT_CHILD_DEADLINE_S, capture_output=True)
    lines = {l.split(' ', 1)[0]: l.split(' ', 1)[1] for l in done.stdout.splitlines()
             if l.startswith(('FULL ', 'REDUCED ', 'OLDER '))}
    assert done.returncode == 0 and set(lines) == {'FULL', 'REDUCED', 'OLDER'}, \
        f"the client child exited {done.returncode}: {done.stdout[-400:]!r} {done.stderr[-600:]!r}"
    assert lines['FULL'] == "connected 'pysmurf'", f"on the full server: {lines['FULL']}"
    assert lines['REDUCED'].startswith('refused ') and 'readout core alone' in lines['REDUCED'], \
        f"on the reduced server: {lines['REDUCED']}"
    assert lines['OLDER'] == 'connected None', f"on a server predating the name: {lines['OLDER']}"


@contextlib.contextmanager
def stood_in_procedure(_setup, fail_load=False):
    """The procedure's hardware steps stood in for.

    Loading the register layers and reading the converter links' elastic
    buffers have nothing to work on in emulation (the memory reads back
    zeros, so the buffer check would reload and fail for minutes), so both
    are stood in and told to succeed -- or the load told to fail, to see a
    failure reported. The whole-tree read is stood in too: on an emulated
    tree it is ~30 s of rendering the operations' arrays, and it is
    exercised by its own check.
    """
    # Each platform's procedure lives in its own support module, which bound the
    # loader by name at import; the stand-in goes into every one of them.
    import cryodaq.platform.atca.support as atca_support
    import cryodaq.platform.rfmc.support as rfmc_support
    import cryodaq.platform.rfsoc.support as rfsoc_support
    trees = (atca_support, rfsoc_support, rfmc_support)
    saved = (_setup.load_layers, atca_support.AtcaSetup.check_elastic_buffers, ROOT.ReadAll._function,
             ROOT.layers, rfmc_support.RfmcSetup.steps, _setup.check_layer_names_this_root)

    def load(root, layers):
        if fail_load:
            raise _setup.StepError('stood-in load failed')

    def rfmc_steps(self):
        # The RFMC's steps before and after the load drive clock chips and the
        # RF data converter -- hardware a register emulation has nothing of
        # (the converter is a device on the processor system, reached over its
        # own link) -- so only the two steps every platform has are run here;
        # the RFMC's own are exercised over a stand-in tree by
        # check_setup_steps.py, and on a board by the bench.
        return (('read all', self.root.ReadAll),
                ('load layers', lambda: load(self.root, self.layers)))
    try:
        _setup.load_layers = load
        for module in trees:
            module.load_layers = load
        # The stand-in layer is a name, not a file; the key check that reads
        # one is stood in with the load, and has a check of its own.
        _setup.check_layer_names_this_root = lambda root, layer: None
        atca_support.AtcaSetup.check_elastic_buffers = lambda self: None
        rfmc_support.RfmcSetup.steps = rfmc_steps
        ROOT.ReadAll.replaceFunction(lambda: None)
        ROOT.layers = ['<stand-in layer>']
        yield
    finally:
        (_setup.load_layers, atca_support.AtcaSetup.check_elastic_buffers, _, ROOT.layers,
         rfmc_support.RfmcSetup.steps, _setup.check_layer_names_this_root) = saved
        for module in trees:
            module.load_layers = saved[0]
        ROOT.ReadAll.replaceFunction(saved[2])


def check_the_application_client_configures_through_the_server():
    """``SmurfControl.setup()`` is the server's procedure plus the client's operating state.

    The client runs the server's procedure and -- once the server reports
    the system configured -- routes the triggers, sets the flux ramp up from
    its configuration, turns the stream on, sets the processor's payload and
    mask, zeroes the detector biases, and records the configuration. The
    layer load and the link check are stood in on the server, as in the
    procedure's own check; the cryocard steps skip as they do with no card.
    What is asserted is the client's end: ``setup()`` returns True, the
    witnessed ramp period and this client's per-band values are on the
    tree, the record is on the server, and a procedure the server fails is
    a False with the failure named. Built in a child interpreter, as the
    refusal check is, and for the same reasons.
    """
    from cryodaq.server import _setup
    probe = subprocess.run([sys.executable, '-c', 'import pysmurf.client'], capture_output=True, text=True)
    if probe.returncode != 0:
        raise Skipped("pysmurf.client does not import in this interpreter: "
                      f"{probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else 'no detail'}")
    cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'cfg_files', 'b33',
                       'experiment_2xLB_testing_Feb2021.cfg')
    source = (
        "import matplotlib; matplotlib.use('Agg')\n"
        "import os, sys, tempfile, json\n"
        "import cryodaq, pysmurf.client\n"
        "tmp = tempfile.mkdtemp()\n"
        f"text = open({cfg!r}).read().replace('/data', tmp + '/data')\n"
        "cfg = os.path.join(tmp, 'experiment.cfg'); open(cfg, 'w').write(text)\n"
        f"S = pysmurf.client.SmurfControl(cfg_file=cfg, server_port={ENDPOINT.rsplit(':', 1)[1]}, setup=False,\n"
        "                                atca_monitor=False, make_logfile=False, offline=False,\n"
        "                                data_dir=tmp + '/data/smurf_data')\n"
        "ok = S.setup(force_configure=True)\n"
        "print('SETUP', ok)\n"
        "b = S._bands[0]\n"
        "print('BAND', b, S.get_lms_gain(b), S.lms_gain[b], S.get_feedback_gain(b), S.feedback_gain[b],\n"
        "      S.get_gradient_descent_gain(b), S.gradient_descent_gain[b])\n"
        "print('RAMP', S.get_ramp_max_cnt(), S.get_enable_ramp_trigger(), S.get_stream_enable())\n"
        "print('MASK', json.dumps(list(S.get_channel_mask())), S.get_payload_size())\n"
        "rec = S._session.resolved_config()\n"
        "print('RECORD', rec is not None and rec.hash == S.config.hash, repr(S._session.get('ops.setup.message')))\n"
        "print('AGAIN', S.setup())\n"
        "print('FORCED', S.setup(force_configure=True))\n")
    with stood_in_procedure(_setup):
        done = subprocess.run([sys.executable, '-c', source], text=True,
                              timeout=CLIENT_CHILD_DEADLINE_S, capture_output=True)
        lines = {l.split(' ', 1)[0]: l.split(' ', 1)[1] for l in done.stdout.splitlines()
                 if l.split(' ', 1)[0] in ('SETUP', 'BAND', 'RAMP', 'MASK', 'RECORD', 'AGAIN', 'FORCED')}
        assert done.returncode == 0 and len(lines) == 7, \
            f"the client child exited {done.returncode}: {done.stdout[-600:]!r} {done.stderr[-800:]!r}"
        assert lines['SETUP'] == 'True', f"setup() returned {lines['SETUP']}: {done.stdout[-600:]!r}"
        # The client's per-band values are on the tree: register, configured, in pairs.
        band = lines['BAND'].split()
        assert band[1] == band[2] and band[3] == band[4] and float(band[5]) == float(band[6]), lines['BAND']
        assert lines['RAMP'].split() == ['76799', '1', '1'], lines['RAMP']
        assert lines['MASK'].startswith('[0] '), lines['MASK']
        assert lines['RECORD'].startswith("True 'Done'"), lines['RECORD']
        assert lines['AGAIN'] == 'False', 'a second setup() without force was not refused'
        assert lines['FORCED'] == 'True', lines['FORCED']
    # A procedure the server fails is a False from the client, the failure named.
    with stood_in_procedure(_setup, fail_load=True):
        failing = source.split("print('AGAIN'")[0].replace("print('RAMP'", "print('MESSAGE', repr(S._session.get('ops.setup.message')))\nprint('RAMP'")
        done = subprocess.run([sys.executable, '-c', failing], text=True,
                              timeout=CLIENT_CHILD_DEADLINE_S, capture_output=True)
        assert 'SETUP False' in done.stdout and "MESSAGE 'Error: load layers: stood-in load failed'" in done.stdout, \
            f"{done.stdout[-600:]!r} {done.stderr[-400:]!r}"
        assert 'RECORD False' in done.stdout, 'a failed setup left a record on the server'
    set_configured(ROOT, True)


def check_a_start_that_fails_in_the_tree_leaves_no_thread_behind():
    """A server whose tree does not come up is stopped, not left half-running.

    Rogue starts a root's threads -- the update worker, the heartbeat -- before it
    starts the link nodes, and none of them is a daemon. A link node whose start
    raises therefore raises out of ``start()`` with the threads running, and
    ``with compose(...) as root`` never reaches its exit for an entry that
    raised: the process would hold open with no server on it. (A register read
    that times out is not this case: rogue catches and logs it, and ``start()``
    returns with the root running.) The failure here is a link node that refuses
    to start, which is where a link that cannot open fails. Built over the
    stand-in top level: what is asserted is the guard, not the tree. A root the
    guard failed to stop is stopped here so that a regression is reported as a
    failure and not as a run that never exits.
    """
    import pysmurf.core.emulators

    class Link(pr.Device):
        def __init__(self):
            pr.Device.__init__(self, name='link')

        def _start(self):
            raise RuntimeError("the link did not open")

    transport = cryodaq.server.emulation(pysmurf.core.emulators.StreamDataSource())
    transport.nodes.append(Link())
    root = standin_root(transport=transport)
    before = set(threading.enumerate())
    try:
        try:
            with root:
                raise AssertionError("the root started over a link that did not open")
        except RuntimeError as e:
            assert 'did not open' in str(e), e
        assert not root.running, "the root reports running after a start that raised"
        left = [t.name for t in threading.enumerate() if t not in before and not t.daemon]
        assert left == [], f"a start that raised left threads running: {left}"
    finally:
        if root.running:
            root.stop()


def check_a_session_left_open_still_lets_the_interpreter_exit():
    """Forgetting to close costs the transport nothing, and the process nothing.

    Two ways an unclosed session could cost the process, one per monitor
    setting. With the monitor on, its thread is not a daemon, so the interpreter
    waits for it on the way out; a session nobody closed would wait for good, and
    the case is an interactive session, where nothing guarantees a ``close``.
    With it off, a rogue client left to its destructors aborts the process during
    finalisation some of the time (``FATAL: exception not rethrown``, exit 134)
    -- after everything the program meant to print, so a run looks clean and its
    exit code says otherwise. Neither can be checked in this process, what is
    being checked being an exit: a child opens a session, returns without closing
    it, and has to be gone, and gone cleanly, before the deadline. The abort is
    intermittent, so the child runs ``EXIT_TRIALS`` times and every exit counts.
    """
    for kwargs in ({}, {'monitor': False}):
        source = ('import cryodaq\n'
                  f"session = cryodaq.connect({ENDPOINT!r}, **{kwargs!r})\n"
                  "print(session.pmap.name)\n")
        exits = {}
        for _ in range(EXIT_TRIALS):
            try:
                done = subprocess.run([sys.executable, '-c', source], text=True,
                                      timeout=EXIT_DEADLINE_S, capture_output=True)
            except subprocess.TimeoutExpired:
                raise AssertionError(
                    f"a child that connected with {kwargs or 'the defaults'} and never "
                    f"closed was still running after {EXIT_DEADLINE_S}s: an unclosed "
                    "session holds the interpreter open") from None
            exits[done.returncode] = exits.get(done.returncode, 0) + 1
            assert EXPECTED_PLATFORM[PLATFORM] in done.stdout, \
                f"the child never connected: {done.stdout.strip()!r} {done.stderr.strip()[-200:]}"
        assert exits == {0: EXIT_TRIALS}, (
            f"with {kwargs or 'the defaults'}, {EXIT_TRIALS} children that never closed "
            f"exited {exits}: a non-zero exit after a clean run is the client being torn "
            f"down by its destructors rather than stopped (last stderr: "
            f"{done.stderr.strip()[-200:]!r})")


def check_the_configuration_procedure_ends_on_this_platform():
    """``setDefaults`` runs to its end on the tree in hand, and reports as the platform allows.

    The layer load and the link check are stood in for (see
    ``stood_in_procedure``). What is exercised is everything around them:
    that the procedure, started through the session as a client would,
    reaches its end on a platform with links and on one without, that
    ``SystemConfigured`` rises and ``ConfiguringInProgress`` falls, and that
    the links' verdict is set where
    the platform has links and nowhere otherwise. A procedure that reached
    for the verdict on a platform without links raised here before it was
    gated. A step that fails is reported with its reason, and the run is
    judged failed.
    """
    from cryodaq.server import _setup
    root = ROOT
    status = root.status
    has_links = 'application.jesd_status' in root.pmap
    assert ('JesdStatus' in status.nodes) == has_links, \
        f"the status device offers a JESD verdict ({'JesdStatus' in status.nodes}) " \
        f"where the map {'names' if has_links else 'does not name'} one"
    expected_setup = {'atca': 'AtcaSetup', 'rfsoc': 'RfsocSetup', 'rfmc': 'RfmcSetup'}[PLATFORM]
    assert type(root.setup).__name__ == expected_setup, type(root.setup)
    with stood_in_procedure(_setup):
        if has_links:
            status.JesdStatus.set(0)
        SESSION.call('ops.setup', wait=60)
        assert SESSION.get('ops.setup.message') == 'Done', SESSION.get('ops.setup.message')
        assert status.SystemConfigured.get() is True, "the procedure did not report success"
        assert status.ConfiguringInProgress.get() is False, "the procedure did not report its end"
        if has_links:
            assert status.JesdStatus.getDisp() == 'Locked', \
                f"the links' verdict was not set from the outcome: {status.JesdStatus.getDisp()}"
    with stood_in_procedure(_setup, fail_load=True):
        SESSION.call('ops.setup', wait=60)
        message = SESSION.get('ops.setup.message')
        assert message == 'Error: load layers: stood-in load failed', message
        assert status.SystemConfigured.get() is False, "a failed step was credited"
        if has_links:
            assert status.JesdStatus.getDisp() != 'Locked', "a failed procedure left the links' verdict Locked"
    set_configured(root, True)      # leave the tree as the other checks expect


def check_the_entry_point_runs_to_ready_and_exits_on_sigterm():
    """``python -m pysmurf.core.server`` itself, not just what it composes.

    Every other check composes the tree in this process, which is how a defect
    confined to ``main()`` -- an import order the card needs, a name shadowed in
    the function's own scope -- stays invisible until a server is started on a
    crate. So the entry point is run the way a launcher runs it: as a child,
    over the emulated transport, with this run's firmware package; a client must
    see ``Ready`` rise, and SIGTERM must end the child with exit 0 and no
    traceback. What the child cannot exercise -- the offload card, a real link
    -- it says so in its own arguments.
    """
    _entry_point_child('pysmurf.core.server').verdict(identity=True)


def check_the_reduced_entry_point_runs_to_ready_and_exits_on_sigterm():
    """``python -m cryodaq.server``, the same way: it has a ``main()`` of its own.

    The expected difference from the full server: no application identity on
    the status device, since no application composed it.
    """
    _entry_point_child('cryodaq.server').verdict(identity=False)


# The two entry points, each building a full tree in its own interpreter, are
# the slow checks of this run, and they need nothing from it; main() starts
# both children as soon as the session's own tree is up, and each check judges
# its own when its turn comes.
ENTRY_POINT_MODULES = ('pysmurf.core.server', 'cryodaq.server')
ENTRY_POINT_CHILDREN = {}


def start_entry_point_children():
    # Both ports before either child, so one cannot be handed the other's.
    ports = {name: free_port() for name in ENTRY_POINT_MODULES}
    for name in ENTRY_POINT_MODULES:
        ENTRY_POINT_CHILDREN[name] = _EntryPointChild(name, ports[name])


def _entry_point_child(module):
    # A check run on its own, outside main(), starts just the child it judges.
    if module not in ENTRY_POINT_CHILDREN:
        ENTRY_POINT_CHILDREN[module] = _EntryPointChild(module, free_port())
    return ENTRY_POINT_CHILDREN.pop(module)


class _EntryPointChild:
    """An entry point run as a child, to be judged once: Ready rose, SIGTERM ended it cleanly."""

    def __init__(self, module, port):
        # The child names the platform this run's package really builds: the
        # carrier for the CryoDet package (the RFSoC run's tree is a stand-in
        # made in-process, see the header, and the child has no such shortcut),
        # the RFMC for its own. What is under test is main() with the command
        # line a launcher would give it, not the platform.
        self.module, self.port = module, port
        self.platform = EXPECTED_PLATFORM['rfmc' if PLATFORM == 'rfmc' else 'atca']
        args = ['--transport', 'emulation', '--platform', self.platform, '--server-port', str(port),
                '--nopoll', '--capture-points', '0', '--log-level', 'WARNING'] + FIRMWARE_ARGS
        cmd = [sys.executable, '-m', module] + args
        # The child inherits this run's import path: the firmware packages in
        # the order add_library_paths() put them.
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
        # Output to a file, not a pipe: a pipe nobody drains fills, and a child
        # blocked on its own print looks exactly like one that never got there.
        self.log = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        self.child = subprocess.Popen(cmd, env=env, stdout=self.log, stderr=subprocess.STDOUT)

    def verdict(self, identity):
        child, module = self.child, self.module
        try:
            _wait_for_ready(child, self.port, identity, self.log, self.platform)
            child.terminate()
            try:
                child.wait(timeout=ENTRY_POINT_EXIT_S)
            except subprocess.TimeoutExpired:
                raise AssertionError(f"{module} ignored SIGTERM for {ENTRY_POINT_EXIT_S}s")
        finally:
            # Whatever happened -- an assertion, Ctrl-C -- no child is left
            # serving on the port.
            if child.poll() is None:
                child.kill()
                child.wait()
            self.log.seek(0)
            out = self.log.read()
            self.log.close()
        assert child.returncode == 0 and 'Traceback' not in out, (
            f"{module} exited {child.returncode} on SIGTERM; output ends: "
            f"{out.strip()[-600:]!r}")


def stop_entry_point_children():
    """Kill any child a check never came to judge (a run that stopped early)."""
    for pending in ENTRY_POINT_CHILDREN.values():
        if pending.child.poll() is None:
            pending.child.kill()
            pending.child.wait()
        pending.log.close()
    ENTRY_POINT_CHILDREN.clear()


def _wait_for_ready(child, port, identity, log, platform_name):
    """Poll the child's server until ``Ready`` is up, within ENTRY_POINT_READY_S.

    ``identity`` says whether the application's version must be on the status
    device once it is (an application server) or must not be (a reduced one).

    Not cryodaq.connect(): the child's emulated memory holds no build stamp,
    which connect() rightly refuses, and nothing outside the child can write
    one. A bare rogue client polls the flag the server raises last -- the ZMQ
    server answers before start() has finished -- and the deadline covers the
    read too: a client that connects to a stuck server would otherwise retry
    for good.
    """
    pmap = platform.by_name(platform_name)
    deadline = time.monotonic() + ENTRY_POINT_READY_S
    while True:
        try:
            client = pyrogue.interfaces.VirtualClient(addr='localhost', port=port)
            # Bound each request as a session does, so a server that answers
            # the connection and then stalls comes back here, not never.
            client.setTimeout(5000, 10000)
        except Exception:                                      # noqa: BLE001
            client = None                                      # nothing listens yet
        if client is not None:
            try:
                ready = client.root.getNode(pmap.path('server.ready')).get()
                if ready:
                    version = client.root.getNode(pmap.path('application.version'))
                    name = client.root.getNode(pmap.path('application.name')).get()
                    if identity:
                        assert version is not None and version.get(), \
                            "the application's identity is not on the status device"
                        assert name == 'pysmurf', f"the full server names its application {name!r}"
                    else:
                        assert version is None, \
                            "a reduced server carries an application's identity"
                        assert name == '', f"the reduced server names an application: {name!r}"
                    return
            finally:
                client.stop()
        if child.poll() is not None:
            log.seek(0)
            raise AssertionError(f"the entry point exited {child.returncode} before Ready; "
                                 f"output ends: {log.read().strip()[-800:]!r}")
        if time.monotonic() > deadline:
            raise AssertionError(f"the entry point did not raise Ready within "
                                 f"{ENTRY_POINT_READY_S}s")
        time.sleep(2.0)


# --------------------------------------------------------------------------

def add_library_paths(args):
    """Put the CryoDet package on pyrogue's library path, and the RFMC's behind it.

    `addLibraryPath` puts each path at the *front*, so what must be found first
    is added last.
    """
    if args.rfmc:
        # Behind CryoDet: the RFMC release carries its own CryoDet, with the
        # operations the server attaches, which the composition would refuse.
        pr.addLibraryPath(f"{args.rfmc}/python")
    if args.zip:
        # Never with --rfmc (main() refuses it): the archive would bring its own
        # surf ahead of the RFMC release's.
        pr.addLibraryPath(f"{args.zip}/python")
        return
    if not args.rfmc:
        # The RFMC release ships these three at the revisions its own package
        # needs; with it on the path, only CryoDet comes from the checkout.
        for sub in ('amc-carrier-core', 'lcls-timing-core', 'surf'):
            path = os.path.join(args.cryo_det, 'firmware/submodules', sub, 'python')
            if os.path.isdir(path):
                pr.addLibraryPath(path)
    pr.addLibraryPath(os.path.join(args.cryo_det, 'firmware/python'))


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
    source = f"cryo-det {head}" + ('' if args.platform == 'atca' else f" ({args.platform})")
    if args.rfmc:
        source += f" over {os.path.basename(args.rfmc)}"
    return source


# Port triples this run has handed out, so a server still on its way to binding
# its ports cannot have them offered to another.
PORTS_GIVEN = set()


def free_port():
    """A port triple starting here is free right now, and not given out before; the server takes three."""
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
    if any(base + i in PORTS_GIVEN for i in range(3)):
        return free_port()
    PORTS_GIVEN.update(base + i for i in range(3))
    return base


def firmware_support():
    """The platform's own support code, building the firmware's tree -- the RFSoC's stood in.

    The RFSoC firmware's own package is not to hand here. Its top-level class
    is the AMC carrier's with ``isRFSOC`` defaulted on -- all that does is leave
    out the JESD lanes and signal generators -- so that class is built with the
    flag, and the RFSoC tree is the carrier package without its bays, not the
    RFSoC firmware's. The procedure and the devices are the platform's own.
    """
    support = cryodaq.server.PlatformSupport.for_map(this_platform_map())
    if PLATFORM == 'rfsoc':
        real = support.top_level

        def top_level(transport, **options):
            if transport.register_offset:
                # Refused as the real builder refuses it, for the same reason.
                real(transport, **options)
            from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel
            return FpgaTopLevel(memBase=transport.srp, isRFSOC=True, **options)
        support.top_level = top_level
    return support


def this_platform_map():
    """The map under test, off the table -- not through `by_name`, which a check watches for the composition's own call."""
    return next(m for m in platform.MAPS if m.name == EXPECTED_PLATFORM[PLATFORM])


class StandInSupport(cryodaq.server.PlatformSupport):
    """A platform's support code with a stand-in top level and nothing beside it.

    The procedure is the platform's own, so a check that runs it over the
    stand-in tree runs the real one.
    """

    def __init__(self):
        self.setup = cryodaq.server.PlatformSupport.for_map(this_platform_map()).setup

    def top_level(self, transport, **options):
        # Over the transport, as the platform's builder is: the offset of the
        # FPGA's register window is the transport's fact and goes in here.
        assert not options, f"the stand-in takes no option; given {sorted(options)}"
        return StandInTopLevel(memBase=transport.srp, offset=transport.register_offset)


class StandInTopLevel(pr.Device):
    """A firmware top level with only what the composition itself needs.

    The real top level takes half a minute to build and as long again to stop
    (its eta-scan arrays are rendered on the way down). Some checks assert on
    what ``compose()`` does around the firmware -- the order it opens the card
    in, the receivers it adds, how a failed start is unwound, the application
    it names -- and for those the firmware's own contents are beside the point.
    This stands in for it there: the devices the map anchors the operations on,
    with the array variables those operations write, a bay scope, the DAQ
    trigger and a version block for the stamp; nothing else.

    Which checks may use it is a judgement about their assertion: anything
    asserting on what the real tree holds -- the bays a build leaves out, every
    mapped name resolving, the entry points -- stays on the real class. It
    spells firmware node names, as a stand-in for a firmware tree must; they
    are the ones the platform map anchors on, and a map change that moved
    them would fail the checks built over it.
    """

    def __init__(self, memBase=None, offset=0):
        import surf.axi
        pr.Device.__init__(self, name='FpgaTopLevel', memBase=memBase, offset=offset)
        app_top = pr.Device(name='AppTop')
        app_core = pr.Device(name='AppCore')
        sysgen = pr.Device(name='SysgenCryo')
        self.add(app_top)
        app_top.add(app_core)
        app_core.add(sysgen)
        for band in range(len(EXPECTED_BANDS)):
            sysgen.add(_StandInBand(band))
        for bay in range(2):
            app_top.add(pr.Device(name=f'DaqMuxV2[{bay}]'))
        self.add(pr.LocalCommand(name='SwDaqMuxTrig', function=lambda: None))
        # What the SMuRF client reaches as it is built, before it is asked for
        # anything: the cryostat card's mailboxes, and the crate and slot it
        # reads to name its files.
        rtm = pr.Device(name='RtmCryoDet')
        spi = pr.Device(name='SpiCryo')
        app_core.add(rtm)
        rtm.add(spi)
        for name in ('read', 'write'):
            spi.add(pr.LocalVariable(name=name, value=0, mode='RW'))
        # The carrier core and its board-support device are named for the
        # board; the map says what this platform's are called.
        paths = this_platform_map()
        carrier = pr.Device(name=paths.path('firmware.build_stamp').split('.')[-3])
        self.add(carrier)
        carrier.add(surf.axi.AxiVersion(offset=0))
        bsi = pr.Device(name=paths.path('carrier.crate_id').split('.')[-2])
        carrier.add(bsi)
        for name in ('CrateId', 'SlotNumber'):
            bsi.add(pr.LocalVariable(name=name, value=0, mode='RO'))


class _StandInBand(pr.Device):
    """One band: the anchor device and the per-channel variables the operations write."""

    CHANNELS = 4

    def __init__(self, band):
        import numpy as np
        pr.Device.__init__(self, name=f'Base[{band}]')
        channels = pr.Device(name='CryoChannels')
        # What the operations read off their anchor to size themselves.
        channels._n_channels = 512
        channels._freqSpanMHz = 1.2
        self.add(channels)
        for name in ('amplitudeScale', 'centerFrequencyMHz', 'feedbackEnable', 'etaMag',
                     'etaPhase', 'frequencyError', 'loopFilterOutput'):
            channels.add(pr.LocalVariable(name=name, value=np.zeros(self.CHANNELS), mode='RW'))
        for i in range(self.CHANNELS):
            channel = pr.Device(name=f'CryoChannel[{i}]')
            channels.add(channel)
            for name in ('amplitudeScale', 'centerFrequencyMHz', 'etaMagScaled',
                         'etaPhaseDegree', 'feedbackEnable', 'frequencyError'):
                channel.add(pr.LocalVariable(name=name, value=0.0, mode='RW'))


def standin_root(reduced=False, **kwargs):
    """A server composed over `StandInTopLevel`, full unless ``reduced``; not started."""
    import pysmurf.core.emulators
    import pysmurf.core.server
    compose = cryodaq.server.compose if reduced else pysmurf.core.server.compose
    kwargs.setdefault('transport',
                      cryodaq.server.emulation(pysmurf.core.emulators.StreamDataSource()))
    return compose(platform_name=EXPECTED_PLATFORM[PLATFORM], server_port=free_port(),
                   polling=False, support=StandInSupport(), **kwargs)


def write_build_stamp(root, stamp):
    """Put a build stamp in the emulated memory, so the tree can be identified.

    The emulator answers from a dictionary of bytes, and unwritten addresses read
    as zero -- which is why an emulated tree reports no firmware at all. Writing
    the stamp where the register reads from gives identification the same thing to
    work with here as on a crate. The register is read-only from the tree's side,
    as it is in the firmware, so this goes in underneath it.
    """
    tag_path = platform.by_name(EXPECTED_PLATFORM[PLATFORM]).tag_path
    node = root.getNode(tag_path)
    assert node is not None, f"the tree has no {tag_path}"
    for offset, byte in enumerate(stamp.encode()):
        root._transport.srp._data[node.address + offset] = byte
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
    import pysmurf.core.emulators
    import pysmurf.core.server
    # The tree is what the SMuRF server composes over an emulated transport,
    # built by the platform's own support code (the RFSoC's stood in, see
    # `firmware_support`). An emulated memory holds no build stamp, so the
    # platform is named here; the stamp the checks expect is written in afterwards.
    root = pysmurf.core.server.compose(
        transport=cryodaq.server.emulation(pysmurf.core.emulators.StreamDataSource()),
        platform_name=EXPECTED_PLATFORM[args.platform], server_port=port, polling=False,
        support=firmware_support())
    try:
        root.start()
    except Exception:
        # The server's own start() stops the root when it raises; this is the
        # harness's belt for a regression there, so the failure is reported by
        # exiting rather than by hanging on threads left behind.
        if root.running:
            root.stop()
        raise
    return root


def collect(prefix):
    """The checks whose names start with ``prefix``, in a fixed order."""
    return sorted((name[len(prefix):], fn) for name, fn in globals().items()
                  if name.startswith(prefix))


class Skipped(Exception):
    """A check that could not run here, and says why: counted apart from the passes."""


SKIPPED = []


def run(checks):
    """Run each check, reporting one line apiece; return the names that failed.

    A check that raises `Skipped` is neither a pass nor a failure: it is listed
    with its reason and counted in SKIPPED, so a run's summary says how many of
    its checks actually ran.
    """
    failed = []
    for name, fn in checks:
        started = time.monotonic()
        try:
            fn()
        except Skipped as e:
            SKIPPED.append(name)
            verdict, detail = 'skip', str(e)
        except Exception as e:                                   # noqa: BLE001
            failed.append(name)
            verdict, detail = 'FAIL', f"{type(e).__name__}: {e}"
        else:
            verdict, detail = 'ok', None
        # The time each check took, so a run that slows down says where.
        print(f"  {verdict:4}  {time.monotonic() - started:6.1f}s  {name}")
        if detail is not None:
            print(f"                  {detail}")
    return failed


def main():
    global SESSION, ROOT, ENDPOINT, PLATFORM, BAND, FIRMWARE_ARGS, SMURF_PCIE, RFMC_ON_PATH
    ap = argparse.ArgumentParser(
        description='Drive a cryodaq session against an emulated firmware tree.')
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument('--cryo-det', help='a CryoDet checkout with the tuning '
                                           'operations removed')
    source.add_argument('--zip', help='a pyrogue ZIP built from such a checkout; a '
                                      'released ZIP still defines the operations '
                                      'and is refused before a tree is built')
    ap.add_argument('--platform', choices=sorted(EXPECTED_PLATFORM), default='atca',
                    help='which platform\'s tree to build and check (the ATCA carrier '
                         'by default); the RFSoC\'s own firmware package is this one '
                         'with isRFSOC defaulted on, so that tree is built from the '
                         'same package without needing it; rfmc needs --rfmc too')
    ap.add_argument('--rfmc', metavar='ZIP',
                    help='the RFMC carrier\'s pyrogue release ZIP (rfmc-cryo-det), put on '
                         'the path behind the CryoDet package above; required with '
                         '--platform rfmc')
    ap.add_argument('--smurf-pcie', metavar='PATH',
                    help='the PCIe offload card\'s package (a smurf-pcie release ZIP or a '
                         'checkout\'s firmware/python): the card check builds its tree over '
                         'an emulated memory; without it that check is skipped and says so')
    ap.add_argument('--port', type=int, default=None,
                    help='port to serve the emulated tree on (a free one by default)')
    ap.add_argument('--dump-tree', metavar='PATH',
                    help='also write the full node listing of the tree the checks ran '
                         'on, as saveVariableList() writes it; prune_fixtures.py reads '
                         'these to rebuild the committed fixtures')
    ap.add_argument('--only', metavar='NAME', action='append', default=[],
                    help='run only the named check(s) (without the check_ prefix), '
                         'over the same served tree; the entry-point children are '
                         'started only if one of them is named')
    args = ap.parse_args()
    PLATFORM = args.platform
    RFMC_ON_PATH = bool(args.rfmc)
    if (args.platform == 'rfmc') != bool(args.rfmc):
        ap.error('--platform rfmc and --rfmc go together')
    if args.rfmc and args.zip:
        # A CryoDet archive carries its own surf/amc-carrier-core, older than
        # the RFMC release's; put whole in front of it, the RFMC's classes
        # fail on an argument that older surf lacks (`zynq`). The checkout
        # form adds CryoDet alone, which is what the RFMC run needs.
        ap.error('--rfmc goes with --cryo-det, not --zip: a CryoDet archive would '
                 "put its older surf ahead of the RFMC release's")
    FIRMWARE_ARGS = ['--firmware', args.zip] if args.zip else []
    if args.smurf_pcie:
        SMURF_PCIE = f"{args.smurf_pcie}/python" if zipfile.is_zipfile(args.smurf_pcie) else args.smurf_pcie
        # Behind the firmware package, never in front of it: a smurf-pcie release
        # carries its own `surf`, from a different revision than the firmware's, and
        # the carrier's tree built over that one has blocks overlapping (a run that
        # prepended it ended in rogue's overlap error). The card's own tree builds
        # over the firmware's `surf`, so the order costs the card nothing.
        sys.path.append(SMURF_PCIE)

    label = f"{EXPECTED_PLATFORM[args.platform]} tree"
    label += ' from ' + (args.zip or args.cryo_det)
    port = args.port or free_port()
    endpoint = ENDPOINT = f"localhost:{port}"
    prechecks = collect('precheck_')
    checks = collect('check_')
    if args.only:
        wanted = set(args.only)
        prechecks = [(n, f) for n, f in prechecks if n in wanted]
        checks = [(n, f) for n, f in checks if n in wanted]
        unknown = wanted - {n for n, _ in prechecks + checks}
        if unknown:
            ap.error(f"no such check(s): {', '.join(sorted(unknown))}")
    failed = []
    started = time.monotonic()
    root = ROOT = emulation_root(args, port)
    print(f"  tree composed and started in {time.monotonic() - started:.1f}s")
    try:
        # Inside the cleanup, not before it: a stamp that will not write is a
        # failure worth reporting, and it is reported by exiting, which a started
        # root that nobody stopped would prevent.
        write_build_stamp(root, stamp_for(EXPECTED_IMAGE[args.platform]))
        if args.dump_tree:
            root.saveVariableList(args.dump_tree)
            with open(args.dump_tree + '.source', 'w', encoding='utf-8') as fh:
                fh.write(describe_source(args) + '\n')
            print(f"  tree written to {args.dump_tree} (source in .source)")
        print(f"Validating the cryodaq client on the {label} "
              f"({len(prechecks)} + {len(checks)} checks)")
        if not args.only or any('entry_point' in n for n in args.only):
            start_entry_point_children()
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
        stop_entry_point_children()
        root.stop()

    print("")
    print(f"{time.monotonic() - started:.0f}s in all")
    if SKIPPED:
        print(f"SKIPPED ({len(SKIPPED)}), not run here: {', '.join(SKIPPED)}")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("All checks passed." if not SKIPPED else
          f"All checks that ran passed ({len(SKIPPED)} skipped).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
