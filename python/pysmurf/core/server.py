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
#    the capture receivers on the DDR streams, the publisher, and pysmurf's
#    identity on the status device. `python -m pysmurf.core.server` runs it; a
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
import rogue.interfaces.stream

import cryodaq.server
import pysmurf
import pysmurf.core.devices
import pysmurf.core.emulators
import pysmurf.core.utilities

__all__ = ['compose', 'main', 'add_arguments', 'VARIABLE_GROUPS']

_log = logging.getLogger(__name__)

# The variables published and streamed as metadata.
VARIABLE_GROUPS = {
    'root.RogueVersion': {'groups': ['publish', 'stream'], 'pollInterval': None},
    'root.RogueDirectory': {'groups': ['publish', 'stream'], 'pollInterval': None},
    'root.SmurfApplication': {'groups': ['publish', 'stream'], 'pollInterval': None},
    'root.SmurfProcessor': {'groups': ['publish', 'stream'], 'pollInterval': None},
}


def compose(*, transport: Any, firmware: Optional[os.PathLike] = None,
            layers: Sequence[os.PathLike] = (), platform_name: Optional[str] = None,
            tx_device: Any = None, stream_pv_size: int = 2**19, stream_pv_type: str = 'Int16',
            providers: Sequence[Any] = cryodaq.server.CORE_PROVIDERS,
            variable_groups: Optional[Mapping] = None,
            **kwargs: Any) -> cryodaq.server.Composition:
    """Build the SMuRF server.

    Parameters
    ----------
    transport : cryodaq.server.Transport
        From `transport_for`, or built directly.
    firmware, layers, platform_name, providers, kwargs
        As `cryodaq.server.compose`.
    tx_device : pyrogue.Device, optional
        A transmitter the processed data is handed to (a streaming application's).
    stream_pv_size, stream_pv_type
        Size and type of the capture receivers on the DDR streams; 0 for none.
    variable_groups : mapping, optional
        Defaults to `VARIABLE_GROUPS`.

    Returns
    -------
    cryodaq.server.Composition
        Enter it to start the server; ``root`` is the tree.
    """
    def processor(root):
        return pysmurf.core.devices.SmurfProcessor(
            name='SmurfProcessor', description='Process the SMuRF Streaming Data Stream',
            root=root, txDevice=tx_device)

    def on_start(root):
        root._pub = pysmurf.core.utilities.SmurfPublisher(root=root)
        # The client starts streaming at 4 kHz without setting a channel mask,
        # and the default mask of every channel is more than the processor keeps
        # up with.
        root.SmurfProcessor.ChannelMapper.Mask.set([0])

    comp = cryodaq.server.compose(
        transport=transport, firmware=firmware, layers=layers, providers=providers,
        sinks=[processor], platform_name=platform_name,
        variable_groups=VARIABLE_GROUPS if variable_groups is None else variable_groups,
        on_start=on_start, **kwargs)
    root = comp.root
    for variable in _identity():
        root.status.add(variable)
    if stream_pv_size:
        _add_capture_receivers(root, transport, stream_pv_size, stream_pv_type)
    return comp


def transport_for(args: argparse.Namespace):
    """The transport the command line names, with pysmurf's streaming receiver."""
    if args.transport == 'eth':
        receiver = pysmurf.core.devices.UdpReceiver(
            ip_addr=args.ip_addr, port=cryodaq.server._transport.STREAMING_PORT)
        return cryodaq.server.transport_from(args, streaming_receiver=receiver)
    if args.transport == 'emulation':
        return cryodaq.server.transport_from(
            args, streaming_source=pysmurf.core.emulators.StreamDataSource())
    return cryodaq.server.transport_from(args)


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


def _add_capture_receivers(root, transport, size, dtype):
    """One receiver per DDR stream, behind a FIFO sized to the receiver."""
    _log.info("enabling stream data on PVs (buffer size = %d points, data type = %s)", size, dtype)
    bytes_per_point = 2 if '16' in dtype else 4
    root._stream_fifos, root._stream_slaves = [], []
    for i, ddr in enumerate(transport.ddr_streams):
        receiver = pysmurf.core.utilities.SmurfDataReceiver(name=f"Stream{i}", rxSize=size, rxType=dtype)
        fifo = rogue.interfaces.stream.Fifo(1000, size * bytes_per_point, True)
        root.add(receiver)
        pyrogue.streamConnect(fifo, receiver)
        pyrogue.streamConnect(ddr, fifo)
        root._stream_slaves.append(receiver)
        root._stream_fifos.append(fifo)


# -- command line ------------------------------------------------------------

def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The server's options: the readout server's plus pysmurf's own."""
    cryodaq.server.add_arguments(parser)
    group = parser.add_argument_group('SMuRF application')
    group.add_argument('--stream-pv-size', type=int, default=2**19,
                       help="Points buffered by each capture receiver; 0 disables them.")
    group.add_argument('--stream-pv-type', default='Int16', choices=['Int16', 'Int32'])
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
    with pcie_card(args):
        comp = compose(transport=transport_for(args), stream_pv_size=args.stream_pv_size,
                       stream_pv_type=args.stream_pv_type,
                       **cryodaq.server.composition_kwargs(args))
        with comp as root:
            if args.gui:
                import pyrogue.pydm
                pyrogue.pydm.runPyDM(serverList=root.zmqServer.address, title=args.windows_title)
            else:
                pyrogue.waitCntrlC()
    return 0


if __name__ == '__main__':
    sys.exit(main())
