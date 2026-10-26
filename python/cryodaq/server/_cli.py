#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Command Line
#-----------------------------------------------------------------------------
# File       : _cli.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    The command-line options a readout server takes, how they become the
#    arguments of `compose()`, and `run()`, which holds the started server
#    open. Shared by the reduced server here and by any application entry
#    point, which adds its own options to the same parser.
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
import socket
from typing import Any, Dict, Optional

from cryodaq import platform
from cryodaq.server._capture import CAPTURE_POINTS

__all__ = ['add_arguments', 'configure_logging', 'transport_from', 'composition_kwargs', 'top_level_options',
           'server_port_for', 'run']

# The ZMQ port convention: a base, three ports per server, chosen by the slot.
SERVER_PORT_BASE = 9000
SERVER_PORT_STRIDE = 3


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the readout server's options to ``parser``."""
    group = parser.add_argument_group('readout server')
    group.add_argument('--transport', '-t', choices=['eth', 'pcie', 'tcp-bridge', 'emulation'],
                       default='eth',
                       help="How the FPGA's registers are reached: over its Ethernet port (eth), "
                            "through the PCIe offload card (pcie), or through the TCP bridge on an "
                            "RFSoC's processor system (tcp-bridge; the data path is then chosen "
                            "separately with --bridge-data).")
    group.add_argument('--addr', '-a', dest='ip_addr', default='',
                       help="FPGA IP address; required for eth and tcp-bridge (where it is the "
                            "FPGA's own port, the data's source). For pcie, the address the "
                            "card's lane is pointed at; omitted, the lane keeps its last one.")
    group.add_argument('--ps-addr', dest='ip_ps', default='',
                       help="For tcp-bridge: the RFSoC processor system's address, where the "
                            "register bridge runs.")
    group.add_argument('--bridge-data', choices=['udp', 'pcie'], default='udp',
                       help="For tcp-bridge: receive the data on the host's Ethernet port (udp), "
                            "or through the PCIe offload card's lane (pcie, with --pcie-rssi-lane "
                            "and the --pcie-dev-* nodes as for the pcie transport).")
    group.add_argument('--pcie-rssi-lane', '-l', type=int, choices=range(6), default=0,
                       help="PCIe RSSI lane, for pcie.")
    group.add_argument('--pcie-dev-rssi', default='/dev/datadev_0',
                       help="PCIe device node for registers.")
    group.add_argument('--pcie-dev-data', default='/dev/datadev_1',
                       help="PCIe device node for data.")
    group.add_argument('--no-pcie-card', action='store_true',
                       help="Do not configure the PCIe offload card: for pcie, take its lane "
                            "as already open; for eth, leave its lanes alone.")
    group.add_argument('--firmware', '--zip', '-z', dest='firmware', default=None,
                       help="Firmware Python package: a release archive or a checkout.")
    group.add_argument('--layer', '--defaults', '-d', dest='layers', action='append', default=[],
                       metavar='FILE', help="A register configuration file, applied in order "
                       "given; the population's defaults first, then overrides.")
    group.add_argument('--platform', dest='platform_name', default=None,
                       choices=[m.name for m in platform.MAPS],
                       help="The platform; required for emulation, otherwise read from the "
                       "hardware and checked against this if given.")
    group.add_argument('--configure', '-c', action='store_true',
                       help="Apply the register configuration at startup.")
    group.add_argument('--nopoll', '-n', action='store_false', dest='polling',
                       help="Disable all polling.")
    # What the deployment knows and the FPGA cannot report before its tree is
    # built: which bays are populated and which carrier revision this is. The
    # launcher learns both over the crate's management bus.
    group.add_argument('--disable-bay0', action='store_true',
                       help="Leave bay 0's devices out of the tree.")
    group.add_argument('--disable-bay1', action='store_true',
                       help="Leave bay 1's devices out of the tree.")
    group.add_argument('--enable-em22xx', action='store_true', dest='enable_pwri2c',
                       help="Enable the EM22xx power monitor.")
    group.add_argument('--capture-points', '--stream-pv-size', dest='capture_points', type=int,
                       default=CAPTURE_POINTS, metavar='N',
                       help="Points buffered by each debug capture receiver; 0 disables them.")
    group.add_argument('--capture-type', '--stream-pv-type', dest='capture_type',
                       default='Int16', choices=['Int16', 'Int8'],
                       help="Sample type of the debug capture receivers.")
    group.add_argument('--server-port', type=int, default=None,
                       help=f"ZMQ port; defaults to {SERVER_PORT_BASE} + "
                       f"{SERVER_PORT_STRIDE} * (slot from the address or lane).")
    group.add_argument('--gui', '-g', action='store_true',
                       help="Open rogue's GUI on the server, and stop when it closes.")
    group.add_argument('--windows-title', '-w', default=None, help="GUI window title.")
    group.add_argument('--log-level', type=str.upper, default='WARNING',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    return parser


def run(root: Any, args: argparse.Namespace) -> None:
    """Hold a started server open until it is told to stop.

    Parameters
    ----------
    root : ReadoutRoot
        A started root (inside its ``with``).
    args : argparse.Namespace
        As parsed from `add_arguments`. With ``--gui`` rogue's GUI is opened
        on the server and the call returns when it closes; otherwise the call
        returns on Ctrl-C or SIGTERM.
    """
    if args.gui:
        _run_gui(root, args.windows_title)
    else:
        import pyrogue
        pyrogue.waitCntrlC()


def _run_gui(root: Any, title: Optional[str]) -> None:
    # Imported here, not at the top: pydm pulls in Qt, which a headless server
    # does not have.
    import pyrogue.pydm
    pyrogue.pydm.runPyDM(serverList=root.zmqServer.address, title=title)


def configure_logging(args: argparse.Namespace) -> None:
    """Root logger at the level the command line asked for."""
    logging.basicConfig(level=args.log_level,
                        format="[%(asctime)s] %(levelname)s:%(name)s: %(message)s")


def server_port_for(args: argparse.Namespace) -> int:
    """The ZMQ port: as given, else by slot from the address's last digit or the lane."""
    if args.server_port is not None:
        return args.server_port
    if args.transport == 'pcie':
        return SERVER_PORT_BASE + SERVER_PORT_STRIDE * (args.pcie_rssi_lane + 2)
    if args.ip_addr:
        return SERVER_PORT_BASE + SERVER_PORT_STRIDE * int(args.ip_addr[-1:])
    return SERVER_PORT_BASE


def transport_from(args: argparse.Namespace, emulation_source: Any = None):
    """Build the transport the arguments name.

    Parameters
    ----------
    args : argparse.Namespace
        As parsed from `add_arguments`.
    emulation_source : rogue stream master, optional
        For the emulation transport only: what stands in for the streaming
        interface, so an application can emulate its own data. The hardware
        transports have a real one and take nothing.
    """
    from cryodaq.server import _transport
    if args.transport == 'eth':
        if not args.ip_addr:
            raise SystemExit("--addr is required for the eth transport")
        try:
            socket.inet_pton(socket.AF_INET, args.ip_addr)
        except OSError:
            raise SystemExit(f"invalid IP address {args.ip_addr!r}") from None
        return _transport.eth(args.ip_addr, dev_rssi=args.pcie_dev_rssi,
                              offload_card=not args.no_pcie_card)
    if args.transport == 'pcie':
        return _transport.pcie(lane=args.pcie_rssi_lane, ip_addr=args.ip_addr,
                               dev_rssi=args.pcie_dev_rssi, dev_data=args.pcie_dev_data,
                               offload_card=not args.no_pcie_card)
    if args.transport == 'tcp-bridge':
        for name, value in (('--ps-addr', args.ip_ps), ('--addr', args.ip_addr)):
            if not value:
                raise SystemExit(f"{name} is required for the tcp-bridge transport")
            try:
                socket.inet_pton(socket.AF_INET, value)
            except OSError:
                raise SystemExit(f"invalid IP address {value!r}") from None
        return _transport.tcp_bridge(args.ip_ps, args.ip_addr, data=args.bridge_data,
                                     lane=args.pcie_rssi_lane, dev_rssi=args.pcie_dev_rssi,
                                     dev_data=args.pcie_dev_data,
                                     offload_card=not args.no_pcie_card)
    return _transport.emulation(emulation_source)


def composition_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The `compose()` keyword arguments the parsed options name, less the transport."""
    return dict(firmware=args.firmware, layers=args.layers, platform_name=args.platform_name,
                server_port=server_port_for(args), polling=args.polling,
                configure=args.configure, capture_points=args.capture_points,
                capture_type=args.capture_type,
                top_level_options=top_level_options(args))


def top_level_options(args: argparse.Namespace) -> Dict[str, Any]:
    """The top-level class's keyword arguments for the flags that were given.

    Only a flag that was set is passed on: a platform whose class does not
    take it refuses it by name, which is right when it was asked for and
    wrong when it was the default.
    """
    flags = {'disableBay0': args.disable_bay0, 'disableBay1': args.disable_bay1,
             'enablePwrI2C': args.enable_pwri2c}
    return {name: True for name, given in flags.items() if given}
