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
from contextlib import nullcontext
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
            variable_groups: Optional[Mapping] = None, pcie_card: Any = None,
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
    pcie_card : PcieCard, optional
        The offload card the server runs behind, if any; a command to restart
        its RSSI link is added to the tree.

    Returns
    -------
    cryodaq.server.ReadoutRoot
        Enter it to start the server.
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
        on_start=on_start, **kwargs)
    for variable in _identity():
        root.status.add(variable)
    if pcie_card is not None:
        root.add(pyrogue.LocalCommand(name='RestartRssi', description='Restart RSSI Link',
                                      function=lambda: pcie_card.restart_rssi()))
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
    group.add_argument('--no-pcie-card', action='store_true',
                       help="Do not open the PCIe card's RSSI lanes around the server.")
    group.add_argument('--gui', '-g', action='store_true', help="Start a GUI beside the server.")
    group.add_argument('--windows-title', '-w', default=None, help="GUI window title.")
    # Accepted and ignored: the platform is read from the hardware, and the
    # management address belongs to the launcher's firmware check. Kept so a
    # launcher that still passes them keeps working; to be removed.
    group.add_argument('--is-rfsoc', action='store_true', help=argparse.SUPPRESS)
    group.add_argument('--rfsoc-mgmt-ip', default=None, help=argparse.SUPPRESS)
    return parser


def pcie_card(args: argparse.Namespace):
    """The offload card, opened around the server for the eth and pcie transports."""
    if args.no_pcie_card or args.transport == 'emulation':
        return nullcontext()
    # The card's own register tree imports from the firmware package (surf,
    # axipcie), so a release archive must be on the path before the card is
    # opened -- which is before compose() would put it there.
    cryodaq.server.load_package(args.firmware)
    comm = 'eth-rssi-interleaved' if args.transport == 'eth' else 'pcie-rssi-interleaved'
    return pysmurf.core.devices.PcieCard(lane=args.pcie_rssi_lane, comm_type=comm,
                                         ip_addr=args.ip_addr, dev_rssi=args.pcie_dev_rssi,
                                         dev_data=args.pcie_dev_data)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start the SMuRF server from the command line and run until interrupted."""
    parser = add_arguments(argparse.ArgumentParser(
        prog='python -m pysmurf.core.server', description='Start the SMuRF server.'))
    args = parser.parse_args(argv)
    cryodaq.server.configure_logging(args)
    if args.is_rfsoc:
        _log.warning("--is-rfsoc is ignored: the platform is read from the hardware's build stamp")
    # The card is a context of its own around the server's: its lanes are opened
    # before the tree is built and closed after it stops.
    with pcie_card(args) as card:
        source = pysmurf.core.emulators.StreamDataSource() if args.transport == 'emulation' else None
        root = compose(transport=cryodaq.server.transport_from(args, emulation_source=source),
                       pcie_card=card if isinstance(card, pysmurf.core.devices.PcieCard) else None,
                       **cryodaq.server.composition_kwargs(args))
        with root:
            if args.gui:
                _run_gui(root, args.windows_title)
            else:
                pyrogue.waitCntrlC()
    return 0


def _run_gui(root, title):
    # Imported here, not at the top: pydm pulls in Qt, which a headless server
    # does not have. A function of its own so the import cannot shadow the
    # module-level `pyrogue` inside main().
    import pyrogue.pydm
    pyrogue.pydm.runPyDM(serverList=root.zmqServer.address, title=title)


if __name__ == '__main__':
    sys.exit(main())
