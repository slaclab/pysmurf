#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Name Resolution Checks
#-----------------------------------------------------------------------------
# File       : check_catalog_resolves.py
# Created    : 2026-09-18
#-----------------------------------------------------------------------------
# Description:
# Checks that every register a semantic name claims is a register the firmware
# actually has. The platform map is the contract between firmware and software, so
# an entry that has drifted is a build failure here rather than a night on a crate.
#
# A tree dump is a tab-separated list of every node a firmware package defines,
# one per line, produced from the package itself with no hardware involved. Names
# are resolved through the platform layer -- the same lookup a running client does,
# so what passes here is what would resolve on a system -- the resulting paths are
# filled with the indices the dump turns out to have, and each one is looked up.
# Nothing here connects to anything.
#
# Two generations are checked, and they differ by omission: one carries per-bay
# data links and an RF front end the other does not. A name reaching those
# registers resolves on one and not the other, so the checks assert *which* names
# are absent rather than how many -- a name absent from a generation that carries
# the hardware, and a name present on one that does not, both fail. The absence is
# stated by device and not by scope, because the bay scope is not the difference:
# the DAQ mux is indexed by bay on both.
#
# The compatibility layer is checked against the same dumps, since the names it
# resolves are the ones that have to be there for a client to work at all: a name
# in the map that nothing reaches is a loose end, and a name reached by a method
# that resolves nowhere is a method that raises.
#
# This file reads its dumps from the repository, so it needs nothing installed and
# runs where rogue is absent. A full dump is some nine megabytes, so what is
# committed is pruned to the devices a name reaches, keeping every index of each --
# that is what lets the two generations still be told apart. The pruning is what
# makes this the fallback rather than the whole story: the run against every
# released firmware package belongs in a job that can download them, and this one
# is what keeps the check meaningful where those are out of reach.
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
import ast
import gzip
import os
import pathlib
import re
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'python'))

from cryodaq import platform                                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURES = HERE / 'fixtures'
COMMAND = (REPO / 'python' / 'pysmurf' / 'client' / 'command' / 'smurf_command.py')

# Which platform map each committed dump belongs to.
PLATFORM_OF = {'atca': 'umux-atca', 'rfsoc': 'umux-rfsoc'}

# The subtrees the *server* adds on top of the firmware package, by their top-level node
# name under the root. `pysmurf.core.roots.Common` adds each one explicitly -- the
# application status device, the two file writers, the data processor, the capture
# receivers, the configuration procedure and the readiness flag -- so a dump of a
# firmware package alone has none of them, and their absence from one says nothing about
# the firmware.
#
# This exists because a dump answers a narrower question than "does this register exist",
# and which question depends on where the dump came from. A dump taken from a *running
# server* has these; a dump built from a released ZIP does not. Without the distinction,
# checking the map against a released package reports ~54 false absences and the real
# ones are lost in them.
SERVER_ADDED_SUBTREES = (
    'SmurfApplication',      # Common.py: pysmurf.core.devices.SmurfApplication()
    'SmurfProcessor',        # the data processing chain
    'streamDataWriter',      # pyrogue.utilities.fileio.StreamWriter
    'streamingInterface',    # the second StreamWriter
    'StreamDataSource',      # EmulationRoot's generator; absent from a deployed root too
    'setDefaults',           # pyrogue.Process wrapping the configuration sequence
    'Ready',                 # LocalVariable set once start-up finishes
)

# The capture receivers are added in a loop as Stream0..3, so they are matched by prefix
# rather than named: a map entry reaches them as `Stream{capture}`.
SERVER_ADDED_PREFIXES = ('Stream',)

# The device trees only a platform with a separate converter board has: the RF front
# end and the serial links back from it. Used to check that the *dumps* differ the way
# the maps say they do -- which names are absent is derived from the maps themselves
# (see below), not listed here, so the two cannot disagree.
CARRIER_ONLY_DEVICES = ('MicrowaveMuxCore', 'AppTopJesd')

# Names whose nodes the server attaches to the tree at start-up, rather than the
# firmware package declaring them: the four resonator-tuning processes and the
# server's own configuration procedure. A dump records what a package defines, so
# these are absent from one by construction and their absence says nothing about the
# firmware. Named here so that "absent" means one thing everywhere else.
SERVER_ATTACHED = (
    # The point-of-load regulator, an I2C device the server attaches on top of the
    # firmware package. Named here for the same reason as the tuning processes: a dump
    # records what a package defines, so these are absent from one by construction. They
    # are on a live carrier -- 129 rows of EM22xx, read from a crate.
    'carrier.regulator.current',
    'carrier.regulator.temperature[*]',
    'band[*].ops.eta_scan',
    'band[*].ops.find_freq',
    'band[*].ops.gradient_descent',
    'band[*].ops.new_gradient_descent',
    'ops.setup',
)

# Writes to a read-only register that predate this check and are left in place. Each is
# a method nothing calls, scheduled for removal, whose write the firmware has always
# refused -- so it has never worked and fixing it would be inventing behaviour. Listed
# rather than tolerated silently, so that removing the method removes the entry too and
# a *new* such write still fails.
KNOWN_READ_ONLY_WRITES = (
    # set_waveform_wr_addr: the write pointer of a capture buffer is the firmware's to
    # advance, and it declares the node RO.
    'carrier.bsa.engine[*].buffer[*].write_address',
    # set_waveform_empty: likewise the buffer's empty flag, which the firmware sets. That
    # method also takes a value it never wrote, and a getter beside it reads the same
    # register, so all it has ever done is fail.
    'carrier.bsa.engine[*].buffer[*].empty',
)

# How far an index is probed for. Matches the platform layer's own ceiling, so a
# scope this check enumerates is the scope a session would enumerate.
MAX_INDEX = platform.MAX_SCOPE_INDEX

SCOPE = re.compile(r'\{(\w+)\}')


def load_dump(stem):
    """The set of node paths in a tree dump.

    A bare stem names a committed fixture; a path with a separator is read as given, so a
    dump built from a released firmware ZIP can be checked without committing 9 MB of it.
    Plain text and gzip are both accepted, since the fixtures are compressed and a fresh
    dump is not.
    """
    if os.sep in str(stem) or str(stem).endswith(('.txt', '.gz')):
        path = pathlib.Path(stem)
    else:
        path = FIXTURES / f'{stem}.varlist.txt.gz'
    if not path.exists():
        raise FileNotFoundError(f'no dump at {path}')
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt', encoding='utf-8', errors='replace') as fh:
        paths = {line.split('\t', 1)[0].strip() for line in fh if line.strip()}
    paths.discard('Path')                       # the header row
    if not paths:
        raise AssertionError(f'{path} holds no paths')
    return paths


def load_nodes(stem):
    """The declared type and access mode of every node in a committed dump, by path.

    ``RO``, ``RW``, ``WO`` as the firmware declares them. This is what says whether a
    register may be written, so it is read from the firmware rather than from anything
    that describes the firmware -- a second statement of it would be a second thing to
    keep in step, and the check below exists because those drift.
    """
    path = FIXTURES / f'{stem}.varlist.txt.gz'
    nodes = {}
    with gzip.open(path, 'rt', encoding='utf-8') as fh:
        for line in fh:
            parts = line.split('\t')
            if len(parts) > 2 and parts[0] != 'Path':
                nodes[parts[0]] = (parts[1].strip(), parts[2].strip())
    if not nodes:
        raise AssertionError(f'{path} declares no node types')
    return nodes


def load_client_names():
    """Every semantic name the client's accessors reach, and how each is reached.

    Read out of the source with ``ast``: the accessors are hand-written, so the names
    they resolve are in the calls themselves and a table of them beside the file would
    be a second description to keep in step. What is recorded is the name pattern, the
    direction, and the method -- enough to hold each to the firmware.

    A name built at run time from something static reading cannot see is skipped, and
    the count is asserted, so a parser that quietly stopped matching fails here rather
    than checking nothing.
    """
    source = COMMAND.read_text(encoding='utf-8')
    tree = ast.parse(source, filename=str(COMMAND))
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == 'SmurfCommandMixin'), None)
    if cls is None:
        raise AssertionError(f'{COMMAND} has no class SmurfCommandMixin')

    found = []
    for node in ast.walk(cls):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in {'_get_by_name', '_set_by_name'}:
            continue
        if not node.args:
            continue
        literal = node.args[0]
        if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
            pattern = literal.value
        elif isinstance(literal, ast.JoinedStr):
            pattern = ''
            for part in literal.values:
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    pattern += part.value
                elif isinstance(part, ast.FormattedValue):
                    pattern += '*'
                else:
                    pattern = None
                    break
            if pattern is None:
                continue
        else:
            continue
        # A literal index in a name -- ``evr_channel[0]`` where the accessor always
        # reaches channel zero -- is still that name's pattern with an index filled in,
        # so it is reduced to the pattern the map is keyed by. Without this a hand-written
        # call site reads as a name the map does not have.
        pattern = re.sub(r'\[\d+\]', '[*]', pattern)
        found.append({
            'name': pattern,
            'direction': 'get' if node.func.attr == '_get_by_name' else 'set',
            'line': node.lineno,
        })

    if len(found) < 150:
        raise AssertionError(
            f'only {len(found)} accessor call(s) found in {COMMAND.name}; the check has '
            f'stopped recognising them rather than the client having lost them')
    return found


def template_of(pmap, name):
    """The register path template a name pattern resolves to."""
    return pmap.registers[name][0]


def indices_present(paths, template, scope, fixed):
    """Which indices of one scope a dump has, with the outer scopes pinned.

    Every index below the ceiling is probed and a gap does not end the scope: a
    firmware mask may leave one out and keep a higher one, so stopping at the
    first miss would silently drop real hardware.

    An inner scope still unfilled is matched by prefix rather than substituted.
    Substituting a nominal ``0`` was wrong, and quietly: the attenuators are
    indexed from one, so probing for ``ATT.UC[0]`` found nothing and reported the
    *bay* as absent -- a whole platform's front end missing because an inner index
    does not start where the probe assumed.
    """
    found = []
    for i in range(MAX_INDEX):
        probe = template
        for name, value in dict(fixed, **{scope: i}).items():
            probe = probe.replace(f'{{{name}}}', str(value))
        remaining = SCOPE.search(probe)
        if remaining:
            prefix = probe[:remaining.start()]
            if any(path.startswith(prefix) for path in paths):
                found.append(i)
            continue
        if probe in paths:
            found.append(i)
    return found


def expand(paths, template):
    """Every concrete path a template reaches in this dump.

    Empty when a scope in the template has no indices here, which is how a
    platform without the hardware behind a name is seen.
    """
    names = []
    for name in SCOPE.findall(template):
        if name not in names:
            names.append(name)
    if not names:
        return [template] if template in paths else []

    out = []

    def walk(fixed, remaining):
        if not remaining:
            concrete = template
            for name, value in fixed.items():
                concrete = concrete.replace(f'{{{name}}}', str(value))
            if concrete in paths:
                out.append(concrete)
            return
        scope, rest = remaining[0], remaining[1:]
        for i in indices_present(paths, template, scope, fixed):
            walk(dict(fixed, **{scope: i}), rest)

    walk({}, names)
    return out


def reaches_a_server_added_subtree(template):
    """Is this template's node one the server adds rather than the firmware declaring it?

    Decided on the template's first segment under the root, which is where a subtree
    added to the root appears. Used only when resolving against a *package* dump: a dump
    from a running server has these nodes, so excluding them there would weaken the check
    for no reason.
    """
    parts = template.split('.')
    if len(parts) < 2:
        return False
    head = parts[1]
    if head in SERVER_ADDED_SUBTREES:
        return True
    bare = head.split('[')[0].split('{')[0]
    return bare in SERVER_ADDED_SUBTREES or any(
        bare.startswith(p) for p in SERVER_ADDED_PREFIXES)


def resolution(stem, names=None, package_only=False, platform_name=None):
    """Which names resolve in one dump, and which do not.

    Resolution goes through the platform map, which is what a client does, so a name
    that passes here is a name that would reach a register on a system running this
    firmware. Every name the platform's map offers by default -- not the documented
    subset, because a map is what a client resolves against and a name it offers
    without a register behind it is the failure being looked for.

    Some names are deliberately excluded: the operation nodes the server attaches at
    start-up, and its own procedure. A dump records what a firmware package defines,
    so those are absent from it by construction rather than by omission.

    `package_only` says the dump came from a released firmware ZIP rather than from a
    running server, so the whole server-added half of the tree is absent too and is
    excluded on the same grounds. The committed fixtures are *not* package-only -- they
    are built with the server's own root -- so the default is the stricter reading.
    """
    paths = load_dump(stem)
    pmap = platform.by_name(platform_name or PLATFORM_OF[stem])
    wanted = sorted(pmap.registers if names is None else names)
    resolved, absent = {}, {}
    for name in wanted:
        if name in SERVER_ATTACHED:
            continue
        if package_only and name in pmap \
                and reaches_a_server_added_subtree(template_of(pmap, name)):
            continue
        if name not in pmap:
            absent[name] = 'not in this platform map'
            continue
        hits = expand(paths, template_of(pmap, name))
        if hits:
            resolved[name] = hits[0]
        else:
            absent[name] = template_of(pmap, name)
    return resolved, absent


def check_each_platform_offers_only_names_its_firmware_has():
    """Every name a platform's map offers resolves on that platform's firmware.

    This is the generality claim in the form the reachable firmware can support it,
    and it is now the same assertion for both platforms rather than a special case
    for one: a map is a statement of what a platform has, so a name it offers has to
    reach something. A platform whose converters share the FPGA's die does not offer
    the front-end names at all -- asking for one is an unresolved name, not a path
    that reaches nothing.

    What each platform has is read from the maps, not listed here. That matters: the
    earlier version of this check carried its own list of carrier-only devices and
    matched it against path strings, which was a second place the same fact was
    written down and could have drifted from the maps it was checking.
    """
    for stem in sorted(PLATFORM_OF):
        resolved, absent = resolution(stem)
        named = ', '.join(sorted(absent)[:8])
        assert not absent, (f'{len(absent)} name(s) the {PLATFORM_OF[stem]} map offers '
                            f'do not resolve on its firmware: {named}')
        assert len(resolved) >= 100, \
            f'only {len(resolved)} names resolved on {stem}; too few to mean much'


def check_the_platforms_differ_by_the_hardware_one_lacks():
    """The two maps differ, and differ by exactly the hardware one platform lacks.

    The difference has to be real in both directions. If the maps were identical this
    whole comparison would be vacuous -- every name would resolve everywhere and
    nothing would be shown about resolution not depending on a device existing. If a
    name were missing for any other reason, the maps would disagree about the
    generation rather than about the hardware.
    """
    atca = platform.by_name(PLATFORM_OF['atca'])
    rfsoc = platform.by_name(PLATFORM_OF['rfsoc'])
    carrier_only = set(atca.registers) - set(rfsoc.registers)
    assert carrier_only, ('the two maps offer the same names, so nothing here shows '
                          'that resolution does not depend on a device existing')
    stray = sorted(n for n in carrier_only
                   if not any(d in template_of(atca, n) for d in CARRIER_ONLY_DEVICES))
    assert not stray, ('name(s) the carrier has and the other lacks that are not its '
                       'front end or data links: ' + ', '.join(stray[:6]))
    other_way = sorted(set(rfsoc.registers) - set(atca.registers))
    assert not other_way, ('name(s) offered by the platform with fewer devices and not '
                           'by the carrier: ' + ', '.join(other_way[:6]))
    # And the firmware has to agree: every carrier-only name must resolve there and
    # not on the other, or the maps are describing trees these dumps are not.
    atca_paths, rfsoc_paths = load_dump('atca'), load_dump('rfsoc')
    for name in sorted(carrier_only):
        template = template_of(atca, name)
        assert expand(atca_paths, template), \
            f'{name} is declared carrier-only and does not resolve on the carrier'
        assert not expand(rfsoc_paths, template), \
            f'{name} is declared carrier-only and resolves on the other platform too'


def check_both_generations_are_really_different_trees():
    """The two dumps are not the same file under two names.

    A check that compared a tree against itself would pass every assertion above and
    prove nothing, so the difference is asserted directly, and asserted to be
    the bay axis rather than merely non-empty.
    """
    atca, rfsoc = load_dump('atca'), load_dump('rfsoc')
    assert atca != rfsoc, 'the two dumps hold identical paths'
    only_atca = atca - rfsoc
    assert only_atca, 'the carrier dump has nothing the other lacks'
    stray = [p for p in only_atca
             if not any(d in p for d in CARRIER_ONLY_DEVICES)]
    assert not stray, ('the carrier has paths the other lacks that are not the per-bay '
                       'front end or data links: ' + ', '.join(sorted(stray)[:4]))


def check_the_client_reaches_only_names_the_map_resolves():
    """Every name the client's accessors reach resolves on the carrier.

    A name that does not resolve is a method that raises the first time it is called, so
    it is caught here rather than in a measurement. The names are read from the accessor
    calls themselves and the map is asked to resolve each, which is the lookup a running
    client does.

    The map deliberately carries names no accessor reaches -- the operations use those
    directly -- so only this direction is a failure.
    """
    resolved, _absent = resolution('atca')
    reached = load_client_names()
    # A name the server attaches rather than the firmware declaring it cannot be
    # verified against a package dump, so it is excluded here for the same reason it is
    # excluded above: its absence from a dump says nothing about whether it exists.
    def unresolved(call):
        if call['name'] in SERVER_ATTACHED:
            return False
        return call['name'] not in resolved

    broken = sorted({f"{call['name']} (line {call['line']})" for call in reached
                     if unresolved(call)})
    assert not broken, ('name(s) the client reaches that the carrier does not have: ' +
                        ', '.join(broken[:6]))
    assert len({call['name'] for call in reached}) >= 100, \
        'too few distinct names reached to mean much'


def check_the_map_declares_the_kind_the_firmware_declares():
    """A name the map calls a value is a value in the firmware, and a command a command.

    The map states what kind of node each name reaches, and a client branches on it: a
    value is read and written, a command is called. Getting it wrong is not caught by
    resolution -- the path is right and the node is there -- so it surfaces only when
    something tries to write a node that has to be called, which is a run against a tree
    rather than a build.

    What a dump can prove is only half of it, and the half it can is worth having. A
    command is always write-only, so a name the map calls a *command* whose node is
    readable is wrong and fails here. The converse does not follow: a write-only node may
    be a command or an ordinary write-only variable -- ``SpiCryo.write`` and ``ReadAll``
    are identical in every column a dump records -- so a value declared write-only is left
    alone rather than guessed at.

    The other direction is checked where the answer exists, against a real tree:
    ``validate_client_emulated.py`` asks rogue itself through ``node.isCommand``, which is
    what caught the four entries this check was added beside. Two checks, and each asserts
    only what its evidence supports.
    """
    pmap = platform.by_name(PLATFORM_OF['atca'])
    nodes = load_nodes('atca')
    paths = load_dump('atca')
    wrong = []
    for name in sorted(pmap.registers):
        if name in SERVER_ATTACHED:
            continue
        declared = pmap.registers[name][1]
        for concrete in expand(paths, template_of(pmap, name)):
            kind, mode = nodes.get(concrete, ('', ''))
            if declared == 'command' and mode and mode != 'WO':
                wrong.append(f'{name}: the map says command, but the firmware declares '
                             f'{kind}/{mode} at {concrete} -- a command is write-only')
            break
    assert not wrong, ('name(s) whose kind the firmware disagrees with: ' +
                       '; '.join(wrong[:6]))


def check_the_client_writes_no_register_the_firmware_makes_read_only():
    """No accessor writes a register the firmware declares readable only.

    Whether a register may be written is the firmware's statement, in the access mode it
    declares, so it is read from the dump rather than from anything describing the dump.
    A write to a read-only node is refused at run time by the tree -- on a good day; the
    point of checking here is the day it is not.
    """
    pmap = platform.by_name(PLATFORM_OF['atca'])
    nodes = load_nodes('atca')
    paths = load_dump('atca')
    wrong = []
    for call in load_client_names():
        if call['direction'] != 'set' or call['name'] not in pmap:
            continue
        if call['name'] in KNOWN_READ_ONLY_WRITES:
            continue
        for concrete in expand(paths, template_of(pmap, call['name'])):
            if nodes.get(concrete, ('', ''))[1] == 'RO':
                wrong.append(f"{call['name']} (line {call['line']}) -> {concrete}")
            break
    assert not wrong, (
        'accessor(s) writing a register the firmware declares read-only: ' +
        ', '.join(sorted(wrong)[:6]))


def selftest():
    """Prove the checks fail on a tree they should reject.

    Every assertion above passes when the map and the dumps agree. The risk is an
    assertion that cannot fail -- comparing a tree with itself, or accepting an
    empty resolution as success -- so each is driven with a deliberately wrong
    input and required to complain.
    """
    import tempfile
    from dataclasses import replace

    global FIXTURES, COMMAND, PLATFORM_OF
    saved = (FIXTURES, COMMAND, PLATFORM_OF, platform.MAPS)
    failures = 0

    def expect_failure(label, fn):
        nonlocal failures
        try:
            fn()
        except AssertionError as e:
            print(f'  ok    {label}')
            print(f'          refused with: {str(e)[:96]}')
        except Exception as e:                                   # noqa: BLE001
            failures += 1
            print(f'  FAIL  {label}: raised {type(e).__name__} rather than refusing')
            print(f'          {e}')
        else:
            failures += 1
            print(f'  FAIL  {label}: accepted input it should have refused')

    # Two fake maps standing in for the two real ones, differing the way they do: one
    # carries the per-bay front end and one does not. They have to differ, or every
    # check that compares them would pass while proving nothing.
    base = 'AMCc.FpgaTopLevel.AppTop.AppCore.'
    shared = {
        'band[*].delay_us': (base + 'SysgenCryo.Base[{band}].bandDelayUs', 'value'),
    }
    carrier_only = {
        'bay[*].attenuator.uc[*]': (
            base + 'MicrowaveMuxCore[{bay}].ATT.UC[{uc}]', 'value'),
    }
    scopes = {
        'band': ((base + 'SysgenCryo.Base[{band}].bandDelayUs',), ()),
        'bay': ((base + 'MicrowaveMuxCore[{bay}].ATT.UC[{uc}]',), ()),
        'uc': ((base + 'MicrowaveMuxCore[{bay}].ATT.UC[{uc}]',), ('bay',)),
    }
    fake = platform.PlatformMap(name='fake', tags=('Fake',),
                               registers=dict(shared, **carrier_only),
                               witness=(), scopes=scopes)
    fake_bayless = platform.PlatformMap(name='fake_rfsoc', tags=('FakeBayless',),
                                        registers=dict(shared), witness=(),
                                        scopes={'band': scopes['band']})
    # A stand-in client. The names are read out of accessor calls, so what this
    # selftest has to vary is source text rather than a table.
    def client_source(*calls):
        """A module defining one class whose methods make the given accessor calls."""
        lines = ['class SmurfCommandMixin:']
        for i, (accessor, literal, extra) in enumerate(calls):
            lines.append(f'    def m{i}(self, band=0, bay=0, uc=0, val=0):')
            lines.append(f'        return self.{accessor}({literal}{extra})')
        # The reader refuses a file with too few accessors to be the real client, so it
        # is padded to that floor with calls to a name the fake map resolves.
        for i in range(len(calls), 160):
            lines.append(f'    def pad{i}(self, band=0):')
            lines.append("        return self._get_by_name(f'band[{band}].delay_us')")
        return chr(10).join(lines) + chr(10)

    try:
        with tempfile.TemporaryDirectory() as tmp:
            tree = pathlib.Path(tmp)
            FIXTURES = tree
            PLATFORM_OF = {'atca': 'fake', 'rfsoc': 'fake_rfsoc'}
            platform.MAPS = (fake, fake_bayless)

            def write(stem, paths):
                with gzip.open(tree / f'{stem}.varlist.txt.gz', 'wt',
                               encoding='utf-8') as fh:
                    fh.write('Path\tTypeStr\n')
                    for p in paths:
                        fh.write(f'{p}\tUInt32\n')

            def write_client(text):
                COMMAND.write_text(text, encoding='utf-8')

            COMMAND = tree / 'smurf_command.py'
            write_client(client_source(
                ('_get_by_name', "f'band[{band}].delay_us'", ''),
                ('_set_by_name', "f'bay[{bay}].attenuator.uc[{uc}]'", ', val'),
            ))

            good_atca = [base + f'SysgenCryo.Base[{b}].bandDelayUs' for b in range(8)]
            # Indexed from one, as the real attenuators are: an inner scope that does
            # not start at zero is what the probe used to get wrong.
            good_atca += [base + f'MicrowaveMuxCore[{y}].ATT.UC[{u}]'
                          for y in range(2) for u in (1, 2, 3, 4)]
            good_rfsoc = [base + f'SysgenCryo.Base[{b}].bandDelayUs' for b in range(8)]

            # A name the firmware does not have must fail, on either platform.
            write('atca', [p for p in good_atca if 'bandDelayUs' not in p])
            write('rfsoc', good_rfsoc)
            expect_failure('a name the carrier lacks is caught',
                           check_each_platform_offers_only_names_its_firmware_has)

            # A name the *other* platform's map offers and its firmware lacks must fail
            # too. That is the assertion the split makes possible: before it, this name
            # was expected to be absent and its absence proved nothing.
            write('atca', good_atca)
            write('rfsoc', [p for p in good_rfsoc if 'bandDelayUs' not in p])
            expect_failure('a name the bayless platform lacks is caught',
                           check_each_platform_offers_only_names_its_firmware_has)

            # A front-end name that resolves on a platform whose map does not offer it
            # means the dumps are not the trees the maps describe.
            write('rfsoc', good_atca)
            expect_failure('a front-end register present without a front end is caught',
                           check_the_platforms_differ_by_the_hardware_one_lacks)

            # Two maps offering the same names make every comparison vacuous.
            write('atca', good_atca)
            write('rfsoc', good_rfsoc)
            platform.MAPS = (fake, replace(fake, name='fake_rfsoc'))
            expect_failure('two maps that offer the same names are caught',
                           check_the_platforms_differ_by_the_hardware_one_lacks)
            platform.MAPS = (fake, fake_bayless)

            # A carrier-only name that is not front-end hardware means the maps disagree
            # about the generation rather than about the hardware.
            odd = platform.PlatformMap(
                name='fake', tags=('Fake',),
                registers=dict(fake.registers,
                               **{'unrelated': (base + 'Something.Else', 'value')}),
                witness=(), scopes=scopes)
            platform.MAPS = (odd, fake_bayless)
            write('atca', good_atca + [base + 'Something.Else'])
            expect_failure('a carrier-only name that is not the front end is caught',
                           check_the_platforms_differ_by_the_hardware_one_lacks)
            platform.MAPS = (fake, fake_bayless)
            write('atca', good_atca)

            # The package-dump exclusion must be narrow: it may drop a name because the
            # *server* owns its subtree, and must not drop one under the firmware's own
            # tree. Asserted directly on the classifier, because a too-broad rule here
            # would silently excuse a real absence when checking a released ZIP.
            before_classifier = failures
            for tmpl in (base + 'SysgenCryo.Base[0].bandDelayUs',
                         'AMCc.FpgaTopLevel.AmcCarrierCore.AxiVersion.FpgaVersion',
                         'AMCc.ReadAll', 'AMCc.enable', 'AMCc.RogueVersion'):
                if reaches_a_server_added_subtree(tmpl):
                    failures += 1
                    print(f'  FAIL  {tmpl} wrongly classed as server-added')
            for tmpl in ('AMCc.SmurfProcessor.Filter.Disable',
                         'AMCc.SmurfApplication.SmurfVersion',
                         'AMCc.Stream{capture}.Updated',
                         'AMCc.streamDataWriter.Open', 'AMCc.setDefaults.Start',
                         'AMCc.Ready'):
                if not reaches_a_server_added_subtree(tmpl):
                    failures += 1
                    print(f'  FAIL  {tmpl} not recognised as server-added')
            if failures == before_classifier:
                print('  ok    the package-dump exclusion covers the server subtrees '
                      'and nothing under FpgaTopLevel')

            # The same tree under two names must fail.
            write('rfsoc', good_atca)
            expect_failure('the same tree under two names is caught',
                           check_both_generations_are_really_different_trees)

            # A tree differing by something other than the bay axis must fail.
            write('rfsoc', good_atca + [base + 'Extra.Register'])
            write('atca', good_atca + [base + 'Something.Unrelated'])
            expect_failure('a difference that is not the bay axis is caught',
                           check_both_generations_are_really_different_trees)
            write('atca', good_atca)
            write('rfsoc', good_rfsoc)

            # A name the map calls a command whose node the firmware lets you read must
            # fail: a command is write-only, so a readable one is not a command. This is
            # the half of the question a dump can answer -- the other half, a value that
            # is really a command, is caught against a real tree by
            # validate_client_emulated.py, which asks rogue rather than a dump.
            commanding = platform.PlatformMap(
                name='fake', tags=('Fake',),
                registers=dict(fake.registers,
                               **{'band[*].delay_us': (
                                   base + 'SysgenCryo.Base[{band}].bandDelayUs',
                                   'command')}),
                witness=(), scopes=scopes)
            platform.MAPS = (commanding, fake_bayless)
            expect_failure('a command the firmware declares readable is caught',
                           check_the_map_declares_the_kind_the_firmware_declares)
            platform.MAPS = (fake, fake_bayless)

            # A client reaching a name the map cannot resolve must fail: that is a
            # method which raises the first time it is called.
            write_client(client_source(
                ('_get_by_name', "f'band[{band}].ghost'", ''),
            ))
            expect_failure('an accessor reaching an unresolvable name is caught',
                           check_the_client_reaches_only_names_the_map_resolves)

            # A client writing a register the firmware declares read-only must fail.
            # The mode comes from the dump, so the dump is what states it here.
            with gzip.open(tree / 'atca.varlist.txt.gz', 'wt', encoding='utf-8') as fh:
                fh.write('Path\tTypeStr\tMode\n')
                for path in good_atca:
                    mode = 'RO' if 'bandDelayUs' in path else 'RW'
                    fh.write(f'{path}\tUInt32\t{mode}\n')
            write_client(client_source(
                ('_set_by_name', "f'band[{band}].delay_us'", ', val'),
            ))
            expect_failure('a write to a read-only register is caught',
                           check_the_client_writes_no_register_the_firmware_makes_read_only)

            # A client the reader cannot recognise at all must fail rather than pass
            # by finding nothing: a parser that stopped matching would otherwise
            # report success over an empty set.
            write_client('class SmurfCommandMixin:\n    pass\n')
            expect_failure('a client whose accessors cannot be read is caught',
                           check_the_client_reaches_only_names_the_map_resolves)
    finally:
        FIXTURES, COMMAND, PLATFORM_OF, platform.MAPS = saved

    print('')
    if failures:
        print(f'SELFTEST FAILED ({failures})')
        return 1
    print('SELFTEST PASS -- every check refuses the input it is meant to refuse.')
    return 0


def main():
    ap = argparse.ArgumentParser(description='Resolve every name against tree dumps.')
    ap.add_argument('--selftest', action='store_true',
                    help='drive each check with a wrong input and require a complaint')
    ap.add_argument('--package-dump', metavar='VARLIST',
                    help='resolve against a dump built from a released firmware ZIP '
                         'rather than the committed fixtures. Such a dump has no '
                         'server-added subtrees, so those names are excluded; every '
                         'name under the firmware tree still has to resolve.')
    ap.add_argument('--platform', default='umux-atca',
                    help='which map to resolve with --package-dump')
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    if args.package_dump:
        resolved, absent = resolution(args.package_dump, package_only=True,
                                      platform_name=args.platform)
        print(f'Resolving the {args.platform} map against a released firmware package')
        print(f'  dump     : {args.package_dump}')
        print(f'  resolved : {len(resolved)}')
        print(f'  absent   : {len(absent)}')
        for name, why in sorted(absent.items()):
            print(f'    {name:46s} {why}')
        if absent:
            print(f'\nFAILED: {len(absent)} name(s) the map offers are not in this '
                  f'released package.')
            return 1
        print(f'\nAll {len(resolved)} firmware-tree name(s) resolve against this '
              f'released package.')
        return 0

    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []

    print(f'Resolving cryodaq names against committed tree dumps '
          f'({len(checks)} checks)')
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            failed.append(label)
            print(f'  FAIL  {label}')
            print(f'          {type(e).__name__}: {e}')
        else:
            print(f'  ok    {label}')

    print('')
    if failed:
        print(f'FAILED ({len(failed)}): {", ".join(failed)}')
        return 1
    resolved, _ = resolution('atca')
    print(f'All checks passed ({len(resolved)} names resolved on the carrier, '
          f'{len(set(c["name"] for c in load_client_names()))} names the client reaches).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
