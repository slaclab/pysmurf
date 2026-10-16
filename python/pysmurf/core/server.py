#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Server Composition
#-----------------------------------------------------------------------------
# File       : server.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    The SMuRF server: cryodaq's readout composition with pysmurf's data
#    processing attached. `compose()` takes what a deployment knows -- the
#    transport, the firmware package, the register configuration files, a
#    transmitter for the processed data -- and returns the composition the
#    readout core builds, with the SmurfProcessor on the streaming interface,
#    the publisher, and pysmurf's identity on the status device. `python -m pysmurf.core.server` runs it; a
#    streaming application calls `compose()` with its own transmitter.
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
import logging
import os
import sys
from typing import Any, Mapping, Optional, Sequence

import pyrogue

import cryodaq.server
import pysmurf
import pysmurf.core.devices
import pysmurf.core.emulators
import pysmurf.core.utilities

__all__ = ['compose', 'main', 'add_arguments', 'VARIABLE_GROUPS']

_log = logging.getLogger(__name__)

# The variables published and streamed as metadata: a path to the groups it
# joins and its poll interval. A device path applies to everything under it;
# a poll interval of None leaves the interval alone and 0 disables polling.
VARIABLE_GROUPS = {
    'root.RogueVersion': {'groups': ['publish', 'stream'], 'pollInterval': None},
    'root.RogueDirectory': {'groups': ['publish', 'stream'], 'pollInterval': None},
    'root.SmurfApplication': {'groups': ['publish', 'stream'], 'pollInterval': None},
    'root.SmurfProcessor': {'groups': ['publish', 'stream'], 'pollInterval': None},
}


def compose(*, transport: Any, tx_device: Any = None,
            providers: Sequence[Any] = cryodaq.server.CORE_PROVIDERS,
            variable_groups: Optional[Mapping] = None,
            **kwargs: Any) -> cryodaq.server.ReadoutRoot:
    """Build the SMuRF server.

    Parameters
    ----------
    transport : cryodaq.server.Transport
        From `cryodaq.server.transport_from`, or built directly.
    tx_device : pyrogue.Device, optional
        A transmitter the processed data is handed to (a streaming application's).
    providers, kwargs
        As `cryodaq.server.compose`.
    variable_groups : mapping, optional
        Defaults to `VARIABLE_GROUPS`.

    Returns
    -------
    cryodaq.server.ReadoutRoot
        Enter it to start the server. As with `cryodaq.server.compose`, a
        root that is composed must be stopped (or released), since it holds
        the transport's offload card.
    """
    def processor(root):
        return pysmurf.core.devices.SmurfProcessor(
            name='SmurfProcessor', description='Process the SMuRF Streaming Data Stream',
            root=root, txDevice=tx_device)

    def on_start(root):
        root._pub = pysmurf.core.utilities.SmurfPublisher(root=root)
        # Workaround: the client starts streaming at 4 kHz without setting a
        # channel mask, and the default mask of every channel is more than the
        # processor keeps up with; the backlog fills the PCIe card's buffers,
        # which triggers known problems in its firmware.
        root.SmurfProcessor.ChannelMapper.Mask.set([0])

    root = cryodaq.server.compose(
        transport=transport, providers=providers, sinks=[processor],
        variable_groups=VARIABLE_GROUPS if variable_groups is None else variable_groups,
        on_start=on_start, application=pysmurf.APPLICATION, **kwargs)
    try:
        for variable in _identity():
            root.status.add(variable)
    except BaseException:
        root.release()      # the composed root holds the transport's card
        raise
    return root


def _identity():
    """pysmurf's identity variables on the server's status device."""
    return (
        pyrogue.LocalVariable(name='SmurfVersion', description='PySMuRF Version',
                              mode='RO', value=pysmurf.__version__),
        pyrogue.LocalVariable(name='SmurfDirectory', description='Path to the PySMuRF Python Files',
                              mode='RO', value=os.path.dirname(pysmurf.__file__)),
        pyrogue.LocalVariable(name='StartupScript', description='PySMuRF Server Startup Script',
                              mode='RO', value=sys.argv[0]),
        pyrogue.LocalVariable(name='StartupArguments', description='PySMuRF Server Startup Arguments',
                              mode='RO', value=' '.join(sys.argv[1:])),
    )


# -- command line ------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The server's options: the readout server's plus pysmurf's own."""
    cryodaq.server.add_arguments(parser)
    group = parser.add_argument_group('SMuRF application')
    # Accepted and ignored: the platform and firmware line are read from the
    # hardware's build stamp, and the management address belongs to the
    # launcher's firmware check. Kept so a launcher that still passes them
    # keeps working; to be removed.
    group.add_argument('--is-rfsoc', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('--is-prespectra', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('--rfsoc-mgmt-ip', default=None, help=argparse.SUPPRESS)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start the SMuRF server from the command line and run until interrupted."""
    parser = add_arguments(argparse.ArgumentParser(
        prog='python -m pysmurf.core.server', description='Start the SMuRF server.'))
    args = parser.parse_args(argv)
    cryodaq.server.configure_logging(args)
    for flag in ('is_rfsoc', 'is_prespectra'):
        if getattr(args, flag):
            _log.warning("--%s is ignored: the platform is read from the hardware's build stamp",
                         flag.replace('_', '-'))
    source = pysmurf.core.emulators.StreamDataSource() if args.transport == 'emulation' else None
    root = compose(transport=cryodaq.server.transport_from(args, emulation_source=source),
                   **cryodaq.server.composition_kwargs(args))
    with root:
        cryodaq.server.run(root, args)
    return 0


if __name__ == '__main__':
    sys.exit(main())
