#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Transports
#-----------------------------------------------------------------------------
# File       : _transport.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    How a server reaches its FPGA. A transport is two things the platform
#    chooses together: a register path (the SRP master the register tree is
#    built over) and a data path (the DDR capture streams and the streaming
#    interface the data processing attaches to). Three are provided -- Ethernet
#    with RSSI, a PCIe offload card, and an in-memory emulation -- and each is a
#    function returning one `Transport`, so the composition that builds the tree
#    is the same whichever carries the bytes.
#
#    Nothing here knows what is in the tree; a transport is built before it.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
from dataclasses import dataclass, field
from typing import Any, List

import pyrogue
import pyrogue.interfaces.simulation
import pyrogue.protocols
import rogue.hardware.axi
import rogue.interfaces.stream
import rogue.protocols.srp

__all__ = ['Transport', 'eth', 'pcie', 'emulation',
           'RSSI_PORT', 'STREAMING_PORT', 'DDR_TDEST', 'STREAMING_TDEST']

# The UDP port the RSSI register link uses, and the one the streaming interface
# arrives on without RSSI.
RSSI_PORT = 8198
STREAMING_PORT = 8195
# The DDR capture streams: a TDEST per channel, the first two channels of each of
# two converter boards.
DDR_TDEST = tuple(0x7F + 1 + channel for channel in (0, 1, 4, 5))
# The streaming interface's TDEST.
STREAMING_TDEST = 0xC1


@dataclass
class Transport:
    """What a composition needs from the hardware link.

    Parameters
    ----------
    name : str
        ``eth``, ``pcie`` or ``emulation``.
    srp : rogue memory master
        The register path the tree is built over.
    ddr_streams : list
        Four stream masters, the DDR capture channels in `DDR_TDEST` order.
    streaming_stream : rogue stream master
        The streaming interface the data processing is connected to.
    nodes : list of pyrogue nodes
        Anything the link needs added to the tree so that it starts and stops
        with it -- the RSSI link, a receiver with its own thread.
    probes_firmware : bool
        Whether reading a register before the tree exists reaches hardware. An
        emulated memory holds no build stamp, so the platform must be named.
    """

    name: str
    srp: Any
    ddr_streams: List[Any]
    streaming_stream: Any
    nodes: List[Any] = field(default_factory=list)
    probes_firmware: bool = True


def eth(ip_addr: str, *, streaming_receiver: Any = None) -> Transport:
    """Ethernet: registers over interleaved RSSI, data over UDP.

    Parameters
    ----------
    ip_addr : str
        The FPGA's address.
    streaming_receiver : rogue stream master, optional
        What receives the streaming interface's UDP packets; it is placed behind
        a FIFO. The application supplies one with a keep-alive; without it the
        streaming path is left unconnected, which a register-only server can do.
    """
    rssi = pyrogue.protocols.UdpRssiPack(name='rudp', host=ip_addr, port=RSSI_PORT,
                                         packVer=2, jumbo=True)
    srp = rogue.protocols.srp.SrpV3()
    pyrogue.streamConnectBiDir(srp, rssi.application(dest=0x0))
    ddr = [rssi.application(dest) for dest in DDR_TDEST]
    fifo = rogue.interfaces.stream.Fifo(100000, 0, True)
    nodes = [rssi]
    if streaming_receiver is not None:
        pyrogue.streamConnect(streaming_receiver, fifo)
        nodes.append(streaming_receiver)
    return Transport('eth', srp, ddr, fifo, nodes)


def pcie(*, lane: int = 0, dev_rssi: str = '/dev/datadev_0',
         dev_data: str = '/dev/datadev_1') -> Transport:
    """A PCIe offload card: registers and data over its DMA channels."""
    rogue.hardware.axi.AxiStreamDma.zeroCopyDisable(dev_rssi)
    base = lane * 0x100
    srp_stream = rogue.hardware.axi.AxiStreamDma(dev_rssi, base + 0, True)
    srp = rogue.protocols.srp.SrpV3()
    pyrogue.streamConnectBiDir(srp, srp_stream)
    ddr = [rogue.hardware.axi.AxiStreamDma(dev_rssi, base + dest, True)
           for dest in DDR_TDEST]
    streaming = rogue.hardware.axi.AxiStreamDma(dev_data, base + STREAMING_TDEST, True)
    return Transport('pcie', srp, ddr, streaming)


def emulation(streaming_source: Any = None) -> Transport:
    """No hardware: registers in memory, data from whatever source is given.

    Parameters
    ----------
    streaming_source : rogue stream master, optional
        Stands in for the streaming interface; added to the tree if it is a node.
        Without one the data path is an idle master.
    """
    srp = pyrogue.interfaces.simulation.MemEmulate()
    ddr = [rogue.interfaces.stream.Master() for _ in DDR_TDEST]
    source = streaming_source if streaming_source is not None \
        else rogue.interfaces.stream.Master()
    nodes = [source] if isinstance(source, pyrogue.Node) else []
    return Transport('emulation', srp, ddr, source, nodes, probes_firmware=False)
