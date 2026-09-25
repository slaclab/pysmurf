#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Fixture Pruning
#-----------------------------------------------------------------------------
# File       : prune_fixtures.py
# Created    : 2026-09-25
#-----------------------------------------------------------------------------
# Description:
# Rebuilds the tree-dump fixtures under fixtures/ from full dumps of the two
# emulated trees. A full dump of one firmware package is some nine megabytes of
# tab-separated text, almost all of it the per-channel registers of eight bands;
# the repository carries a pruned copy, and this is what prunes it.
#
# A row survives when the device it sits in is one a register name reaches, with
# every index of that device kept -- all eight bands and both bays -- so the two
# generations can still be told apart and an off-by-one in an index scope is still
# caught. Only the per-channel arrays are thinned, to two channels, because no name
# indexes a channel through the tree. The pruning is then proved not to have changed
# an answer: every name is resolved against the full dump and against the fixture,
# and the two must agree on which names are absent. Provenance is written beside the
# fixtures; a fixture whose origin is not recorded is not evidence.
#
# The full dumps come from validate_client_emulated.py, which builds each tree in a
# container that has rogue and writes it out with --dump-tree:
#
#     validate_client_emulated.py --cryo-det <checkout>         --dump-tree atca.txt
#     validate_client_emulated.py --cryo-det <checkout> --rfsoc --dump-tree rfsoc.txt
#     prune_fixtures.py --atca atca.txt --rfsoc rfsoc.txt
#
# Each dump comes with a `.source` file naming the checkout revision or ZIP it was
# built from; that is copied into the provenance rather than asserted here.
#
# Rerun after any change to the platform maps that reaches a new device, or the
# fixtures will lack the rows the new names need and check_catalog_resolves.py will
# report them absent. --check reports what would be written without writing it.
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
import collections
import gzip
import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'python'))

from cryodaq import platform                                         # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURES = HERE / 'fixtures'

# How each tree is built; the stems are what check_catalog_resolves.py reads. *What* it
# is built from is not written here: validate_client_emulated.py records that beside
# each dump it writes (a `.source` file naming the checkout's revision or the ZIP),
# and it is copied through, so a fixture rebuilt from another checkout says so.
BUILT_BY = {
    'atca': 'pysmurf.core.roots.EmulationRoot over MemEmulate',
    'rfsoc': 'pysmurf.core.roots.EmulationRoot over MemEmulate, isRFSOC=True',
}

# How many per-channel entries to keep. Two rather than one so that a check which
# enumerated a channel scope would still see more than a single index.
KEEP_CHANNELS = 2

INDEX = re.compile(r'\[\d+\]')
CHANNEL = re.compile(r'CryoChannel\[(\d+)\]')
SCOPE = re.compile(r'\{(\w+)\}')


def generalise(path):
    """A path with every index replaced, so one device matches all its indices."""
    return INDEX.sub('[]', path)


def path_templates():
    """The register path template behind every name, across both platform maps.

    Both are read, so a device only one generation has is still kept -- pruning it
    out would flatten the very difference the check depends on.
    """
    out = {}
    for pmap in platform.MAPS:
        for name, (template, _) in pmap.registers.items():
            out.setdefault(name, template)
    return out


def wanted_devices(templates):
    """The devices a name reaches, index-generalised."""
    devices = set()
    for template in templates.values():
        device = template.rsplit('.', 1)[0]
        devices.add(generalise(SCOPE.sub('0', device)))
    return devices


def prune(lines, devices):
    """The dump's header and the rows worth keeping."""
    kept = [lines[0]]
    for line in lines[1:]:
        path = line.split('\t', 1)[0]
        if generalise(path).rsplit('.', 1)[0] not in devices:
            continue
        m = CHANNEL.search(path)
        if m and int(m.group(1)) >= KEEP_CHANNELS:
            continue
        kept.append(line)
    return kept


def scope_indices(rows, device):
    """Which indices of one device appear in these rows."""
    found = set()
    for row in rows:
        for m in re.finditer(re.escape(device) + r'\[(\d+)\]', row):
            found.add(int(m.group(1)))
    return sorted(found)


def resolves(paths, template):
    """Whether a template reaches at least one path, with any indices."""
    pattern = re.compile(
        '^' + re.escape(SCOPE.sub('\0', template)).replace('\0', r'\d+') + '$')
    return any(pattern.match(p) for p in paths)


def short_head():
    """The pysmurf revision the fixtures were pruned at, ``-dirty`` if the worktree was.

    The fixtures depend on the map and on this script, so a build from uncommitted
    edits to either must say so, as the dump's own provenance does for cryo-det.
    """
    try:
        return subprocess.run(['git', '-C', str(REPO), 'describe', '--always', '--dirty'],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return 'unknown'


def main():
    ap = argparse.ArgumentParser(
        description='Prune full tree dumps into the committed fixtures.')
    ap.add_argument('--atca', required=True, type=pathlib.Path,
                    help='full dump of the carrier tree (validate_client_emulated.py '
                         '--dump-tree)')
    ap.add_argument('--rfsoc', required=True, type=pathlib.Path,
                    help='full dump of the RFSoC tree (... --rfsoc --dump-tree)')
    ap.add_argument('--check', action='store_true',
                    help='report what would be written without writing it')
    args = ap.parse_args()
    sources = {'atca': args.atca, 'rfsoc': args.rfsoc}

    templates = path_templates()
    devices = wanted_devices(templates)
    print(f'{len(templates)} name(s) reach {len(devices)} device(s)')

    summary = collections.OrderedDict()
    pruned_paths = {}
    for stem, source in sources.items():
        lines = source.read_text(encoding='utf-8').splitlines()
        if len(lines) < 2:
            sys.exit(f'{source}: not a tree dump')
        sidecar = source.with_name(source.name + '.source')
        if not sidecar.exists():
            sys.exit(f'{sidecar}: no provenance beside the dump; write the dump with '
                     f'validate_client_emulated.py --dump-tree, which records it')
        origin = sidecar.read_text(encoding='utf-8').strip()
        kept = prune(lines, devices)
        pruned_paths[stem] = {line.split('\t', 1)[0] for line in kept[1:]}
        bands = scope_indices(kept, 'Base')
        bays = scope_indices(kept, 'MicrowaveMuxCore')
        daq = scope_indices(kept, 'DaqMuxV2')
        if not bands:
            sys.exit(f'{stem}: pruned to no bands at all; the fixture would be useless')

        # The pruning is only legitimate if it cannot change an answer. Every name is
        # resolved against the full dump and against the fixture, and the same set has
        # to be absent from both -- otherwise the fixture is a smaller tree that happens
        # to pass, which is the failure a pruned fixture invites.
        full = {line.split('\t', 1)[0] for line in lines[1:]}
        absent_full = {n for n, t in templates.items() if not resolves(full, t)}
        absent_pruned = {n for n, t in templates.items()
                         if not resolves(pruned_paths[stem], t)}
        if absent_full != absent_pruned:
            differ = sorted(absent_full ^ absent_pruned)[:6]
            sys.exit(f'{stem}: the pruned fixture answers differently from the full '
                     f'dump for {", ".join(differ)}; pruning must not change a verdict')

        summary[stem] = {'kept': kept, 'origin': origin, 'rows': len(kept) - 1,
                         'rows_before': len(lines) - 1, 'bands': bands,
                         'front_end_bays': bays, 'daq_mux_bays': daq}
        print(f'  {stem:6s} {len(lines) - 1:6d} -> {len(kept) - 1:5d} rows  '
              f'bands={bands} front-end bays={bays} daq bays={daq}; '
              f'pruned and full dumps agree ({len(absent_pruned)} name(s) absent)')

    # The property the catalog check turns on: one generation has the per-bay front
    # end, the other does not. If pruning ever flattened that, every comparison built
    # on it would pass while proving nothing.
    if summary['atca']['front_end_bays'] == summary['rfsoc']['front_end_bays']:
        sys.exit('both fixtures ended up with the same front-end bays; the pruning has '
                 'flattened the difference the catalog check depends on')
    if not summary['atca']['front_end_bays']:
        sys.exit('the carrier fixture kept no front-end bays')
    if summary['rfsoc']['front_end_bays']:
        sys.exit('the bayless fixture kept front-end bays')

    if args.check:
        print('\n--check: nothing written')
        return 0

    FIXTURES.mkdir(parents=True, exist_ok=True)
    for stem, facts in summary.items():
        with gzip.open(FIXTURES / f'{stem}.varlist.txt.gz', 'wt',
                       encoding='utf-8') as fh:
            fh.write('\n'.join(facts['kept']) + '\n')

    note = [
        'Tree-dump fixtures for tests/cryodaq/check_catalog_resolves.py',
        '',
        'Each file lists every node a firmware package defines, one per line,',
        'tab-separated, as pyrogue\'s Root.saveVariableList() writes it. They are',
        'checked in so the name resolution check needs no hardware, no rogue and no',
        'network -- see the README in this directory for what the check does with them.',
        '',
        'These are *pruned* by tests/cryodaq/prune_fixtures.py: a row is kept when the',
        'device it sits in is one a register name reaches, with every index of that',
        f'device retained, and the per-channel arrays thinned to {KEEP_CHANNELS}.',
        'Keeping every index is what matters: it is how the two generations can still',
        'be told apart, and how an off-by-one in an index scope is still caught.',
        '',
        'Do not edit them by hand. Regenerate them with that script (its header says',
        'how); a hand edit is a fixture that answers a question the firmware would',
        'answer differently.',
        '',
        f'Built at pysmurf {short_head()}.',
        '',
    ]
    for stem, facts in summary.items():
        note += [f'{stem}.varlist.txt.gz',
                 f'  built from      {facts["origin"]}',
                 f'  built by        {BUILT_BY[stem]}',
                 f'  rows            {facts["rows"]} of {facts["rows_before"]}',
                 f'  bands           {facts["bands"]}',
                 f'  front-end bays  {facts["front_end_bays"] or "none"}',
                 f'  daq mux bays    {facts["daq_mux_bays"]}',
                 '']
    note += [
        'Note what a dump of a firmware *package* does not contain: the subtrees a',
        'running server adds -- SmurfProcessor, the stream writers, SmurfApplication,',
        'the capture receivers, setDefaults, Ready. These fixtures are EmulationRoot',
        'dumps, so they do carry them; a dump built straight from a released ZIP does',
        'not, which is why check_catalog_resolves.py has a --package-dump mode that',
        'excludes those names. Absence from a dump is not evidence about the firmware',
        'unless you know which kind of dump you are holding.',
    ]
    (FIXTURES / 'PROVENANCE.txt').write_text('\n'.join(note) + '\n', encoding='utf-8')
    print(f'\nwrote {len(summary)} fixture(s) and PROVENANCE.txt under {FIXTURES}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
