#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf Client Configuration Tools
#-----------------------------------------------------------------------------
# File       : __main__.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
#    python -m pysmurf.client.config convert site.cfg [-o site.yaml]
#        writes the YAML equivalent of a legacy .cfg (stdout without -o)
#    python -m pysmurf.client.config resolve site.yaml
#        resolves a file over the default and prints the values and, with
#        --provenance, the file and line that set each key
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import argparse
import sys
import warnings
from pathlib import Path

from pysmurf.client.config import legacy, load


def main(argv=None):
    parser = argparse.ArgumentParser(prog='python -m pysmurf.client.config')
    sub = parser.add_subparsers(dest='command', required=True)
    conv = sub.add_parser('convert', help='write a legacy .cfg as YAML')
    conv.add_argument('cfg', type=Path)
    conv.add_argument('-o', '--output', type=Path, help='destination; stdout by default')
    res = sub.add_parser('resolve', help='resolve a configuration over the default')
    res.add_argument('path', type=Path)
    res.add_argument('--provenance', action='store_true', help='show which file set each key')
    args = parser.parse_args(argv)

    if args.command == 'convert':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            text = legacy.to_yaml(legacy.convert(args.cfg))
        for w in caught:
            print(f"note: {w.message}", file=sys.stderr)
        if args.output:
            args.output.write_text(text)
            print(f"wrote {args.output}", file=sys.stderr)
        else:
            sys.stdout.write(text)
        return 0

    resolved = load(args.path)
    print(f"# hash {resolved.hash}")
    print(f"# layers: {' <- '.join(resolved.layers)}")
    if args.provenance:
        from cryodaq.config import flatten
        width = max(len(k) for k in resolved.provenance) if resolved.provenance else 0
        for key, value in sorted(flatten(resolved.values).items()):
            layer, line = resolved.provenance.get(key, ('?', 0))
            print(f"{key:{width}s} = {value!r:24}  {Path(layer).name}:{line}")
    else:
        sys.stdout.write(legacy.to_yaml(dict(resolved.values)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
