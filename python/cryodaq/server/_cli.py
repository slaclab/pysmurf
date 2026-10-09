#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Command Line
#-----------------------------------------------------------------------------
# File       : _cli.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    The command-line options a readout server takes, and how they become the
#    arguments of `compose()`. Shared by the reduced server here and by any
#    application entry point, which adds its own options to the same parser.
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
from typing import Any, Dict

from cryodaq import platform

__all__ = ['add_arguments', 'configure_logging', 'transport_from', 'composition_kwargs',
           'server_port_for']

# The ZMQ port convention: a base, three ports per server, chosen by the slot.
SERVER_PORT_BASE = 9000
SERVER_PORT_STRIDE = 3


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add the readout server's options to ``parser``."""
    group = parser.add_argument_group('readout server')
    group.add_argument('--transport', '-t', choices=['eth', 'pcie', 'emulation'], default='eth',
                       help="How the FPGA is reached.")
    group.add_argument('--addr', '-a', dest='ip_addr', default='',
                       help="FPGA IP address; required for eth.")
    group.add_argument('--pcie-rssi-lane', '-l', type=int, choices=range(6), default=0,
                       help="PCIe RSSI lane, for pcie.")
    group.add_argument('--pcie-dev-rssi', default='/dev/datadev_0',
                       help="PCIe device node for registers.")
    group.add_argument('--pcie-dev-data', default='/dev/datadev_1',
                       help="PCIe device node for data.")
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
    group.add_argument('--disable-bay0', action='store_true',
                       help="Leave bay 0's devices out of the tree.")
    group.add_argument('--disable-bay1', action='store_true',
                       help="Leave bay 1's devices out of the tree.")
    group.add_argument('--is-prespectra', action='store_true', help="Pre-SPECTRA firmware line.")
    group.add_argument('--enable-em22xx', action='store_true', dest='enable_pwri2c',
                       help="Enable the EM22xx power monitor.")
    group.add_argument('--server-port', type=int, default=None,
                       help=f"ZMQ port; defaults to {SERVER_PORT_BASE} + "
                       f"{SERVER_PORT_STRIDE} * (slot from the address or lane).")
    group.add_argument('--log-level', default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    return parser


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


def transport_from(args: argparse.Namespace, **extra: Any):
    """Build the transport the arguments name; ``extra`` goes to its constructor."""
    from cryodaq.server import _transport
    if args.transport == 'eth':
        if not args.ip_addr:
            raise SystemExit("--addr is required for the eth transport")
        try:
            socket.inet_pton(socket.AF_INET, args.ip_addr)
        except OSError:
            raise SystemExit(f"invalid IP address {args.ip_addr!r}") from None
        return _transport.eth(args.ip_addr, **extra)
    if args.transport == 'pcie':
        return _transport.pcie(lane=args.pcie_rssi_lane, dev_rssi=args.pcie_dev_rssi,
                               dev_data=args.pcie_dev_data)
    return _transport.emulation(**extra)


def composition_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    """The `compose()` keyword arguments the parsed options name, less the transport."""
    return dict(firmware=args.firmware, layers=args.layers, platform_name=args.platform_name,
                server_port=server_port_for(args), polling=args.polling,
                configure=args.configure,
                top_level_options=dict(disableBay0=args.disable_bay0, disableBay1=args.disable_bay1,
                                       isPreSpectra=args.is_prespectra,
                                       enablePwrI2C=args.enable_pwri2c))
