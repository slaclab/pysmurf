#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Reduced Server
#-----------------------------------------------------------------------------
# File       : __main__.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    `python -m cryodaq.server`: a readout server with the platform, the core's
#    operations and no application -- every register, the bring-up procedure
#    and the same semantic names as a full server, from a plain Python
#    environment with rogue and the firmware package. What a firmware engineer
#    needs to bring a carrier up.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import argparse
import sys

import pyrogue

import cryodaq.server
from cryodaq.server import _cli


def main(argv=None) -> int:
    parser = _cli.add_arguments(argparse.ArgumentParser(
        prog='python -m cryodaq.server', description='Start a reduced readout server.'))
    args = parser.parse_args(argv)
    _cli.configure_logging(args)
    comp = cryodaq.server.compose(transport=_cli.transport_from(args),
                                  **_cli.composition_kwargs(args))
    with comp as root:
        print(f"Server on {root.zmqServer.address}; platform {comp.pmap.name}. Ctrl-C to stop.")
        pyrogue.waitCntrlC()
    return 0


if __name__ == '__main__':
    sys.exit(main())
