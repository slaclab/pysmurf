#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Released Firmware Tree Dump
#-----------------------------------------------------------------------------
# File       : dump_released_tree.py
# Created    : 2026-09-22
#-----------------------------------------------------------------------------
# Description:
# Writes a node listing for the tree a released firmware package defines, so that
# the register names can be checked against the firmware SLAC shipped rather than
# against the tree this repository builds.
#
# The package is a ZIP as the server consumes one: `pyrogue.addLibraryPath` is
# pointed inside the archive exactly as `cryodaq.server` does, the
# top-level device is constructed over emulated memory, and the resulting tree is
# written out. No hardware, no network, and no server -- which is also the limit
# of what the dump can answer for, since a server adds subtrees of its own that a
# package has never heard of. `check_catalog_resolves.py --package-dump` knows
# which those are.
#
# Needs rogue, so it runs in the server container rather than beside the other
# checks here.
#
# Usage:
#     dump_released_tree.py --zip <package.zip> --out <varlist.txt> [--platform umux-rfmc]
#
# The top-level class is the one the platform's map names, imported from the
# package as the server imports it. The RFMC carrier's release declares a rogue
# newer than the server image's for its RF data converter alone; the declaration
# is stood in for here, since nothing it guards is used (the device is not built).
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
import importlib
import os
import sys
import zipfile

import pyrogue as pr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', required=True, help='released firmware package')
    ap.add_argument('--out', required=True, help='where to write the node listing')
    ap.add_argument('--platform', default='umux-atca',
                    help='whose top-level class to build the tree with (its map names it)')
    args = ap.parse_args()

    source = os.path.abspath(args.zip)
    if not zipfile.is_zipfile(source):
        raise SystemExit(f'FATAL: {source} is not a zip')

    # The trailing /python is a path inside the archive, not on disk; pyrogue mounts
    # the ZIP. This is what cryodaq.server.load_package does with a released package.
    pr.addLibraryPath(f'{source}/python')
    # The platform map is this repository's; the script may be run from a copy
    # outside it (the CI job copies it into the server container), so the
    # repository is looked for beside the script first and in the working
    # directory second.
    for root in (os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                 os.getcwd()):
        if os.path.isdir(os.path.join(root, 'python', 'cryodaq')):
            sys.path.insert(0, os.path.join(root, 'python'))
            break
    import rogue
    rogue.Version.minVersion = lambda *a, **k: None
    from cryodaq import platform
    import pyrogue.interfaces.simulation

    pmap = platform.by_name(args.platform)
    module_name, _, attribute = pmap.top_level.partition(':')
    FpgaTopLevel = getattr(importlib.import_module(module_name), attribute)

    kwargs = dict(memBase=pyrogue.interfaces.simulation.MemEmulate())
    # The AMC carrier's class takes which bays to build and which generation;
    # isRFSOC/isPreSpectra are recent additions a release predating them rejects, so
    # ask in decreasing order of specificity rather than assuming. The RFMC's takes
    # nothing of the kind.
    for extra in ({'disableBay0': False, 'disableBay1': False, 'isRFSOC': False, 'isPreSpectra': False},
                  {'disableBay0': False, 'disableBay1': False, 'isRFSOC': False},
                  {'disableBay0': False, 'disableBay1': False}, {}):
        try:
            fpga = FpgaTopLevel(**kwargs, **extra)
            break
        except TypeError as exc:
            if 'unexpected keyword argument' not in str(exc):
                raise
    else:
        raise SystemExit('FATAL: could not construct FpgaTopLevel from this package')

    # add() before start(): a started tree refuses new nodes, so the context-manager
    # form cannot be used here -- it starts on entry.
    root = pr.Root(name='AMCc', description='released package dump',
                   initRead=False, pollEn=False, timeout=5.0)
    root.add(fpga)
    root.start()
    try:
        root.saveVariableList(args.out)
    finally:
        root.stop()

    with open(args.out, encoding='utf-8', errors='replace') as fh:
        lines = sum(1 for _ in fh)
    print(f'wrote {args.out} ({lines} lines) from {os.path.basename(source)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
