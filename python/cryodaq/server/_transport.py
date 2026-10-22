#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Transports
#-----------------------------------------------------------------------------
# File       : _transport.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    How a server reaches its FPGA. A transport is two things the platform
#    chooses together: a register path (the memory master the register tree is
#    built over) and a data path (the DDR capture streams and the streaming
#    interface the data processing attaches to). Four are provided -- Ethernet
#    with RSSI, a PCIe offload card, a TCP bridge to an RFSoC's processor
#    system, and an in-memory emulation -- and each is a function returning one
#    `Transport`, so the composition that builds the tree is the same whichever
#    carries the bytes. The data path is one of two, Ethernet UDP or the card's
#    DMA, built by one function each; the register path is what differs.
#
#    A transport that runs through the PCIe offload card also carries the card
#    (`_offload_card`), whose RSSI lane to the carrier is opened around the
#    server; the Ethernet transport carries it too when one is present, to keep
#    it from intercepting the carrier's traffic. An RFSoC carrier's RF data
#    converter is driven over a register path of its own, to a daemon on the
#    processor system, so a transport to one carries a second memory master.
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
import os
from typing import Any, List, Optional

import pyrogue
import pyrogue.interfaces.simulation
import pyrogue.protocols
import rogue.hardware.axi
import rogue.interfaces.memory
import rogue.interfaces.stream
import rogue.protocols.srp

from cryodaq.server._offload_card import OffloadCard
from cryodaq.server._udp_receiver import UdpReceiver

__all__ = ['Transport', 'eth', 'pcie', 'tcp_bridge', 'emulation',
           'RSSI_PORT', 'STREAMING_PORT', 'DDR_TDEST', 'STREAMING_TDEST',
           'BRIDGE_REGISTER_PORT', 'BRIDGE_RFDC_PORT', 'BRIDGE_REGISTER_OFFSET']

# The UDP port the RSSI register link uses, and the one the streaming interface
# arrives on without RSSI.
RSSI_PORT = 8198
STREAMING_PORT = 8195
# The TCP ports of the bridge on an RFSoC's processor system: the FPGA's registers,
# and the RF data converter's driver; and the offset of the FPGA's register window
# in the processor's address space, which the bridge does not apply itself.
BRIDGE_REGISTER_PORT = 9000
BRIDGE_RFDC_PORT = 9002
BRIDGE_REGISTER_OFFSET = 0x04_0000_0000
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
        ``eth``, ``pcie``, ``tcp-bridge`` or ``emulation``.
    srp : rogue memory master
        The register path the tree is built over.
    ddr_streams : list
        Four stream masters, the DDR capture channels in `DDR_TDEST` order.
    streaming_stream : rogue stream master
        The streaming interface the data processing is connected to.
    nodes : list of pyrogue nodes
        Anything the link needs added to the tree so that it starts and stops
        with it -- the RSSI link, a receiver with its own thread.
    register_nodes : list of pyrogue nodes
        The subset of ``nodes`` the register path needs before it answers: the
        RSSI link, which opens when the tree it is in starts. A probe that reads
        a register before the real tree exists adds these to its own tree.
    probes_firmware : bool
        Whether reading a register before the tree exists reaches hardware. An
        emulated memory holds no build stamp, so the platform must be named.
    card : OffloadCard, optional
        The PCIe offload card to configure around the server: opened before
        the firmware is probed, closed after the tree stops. The pcie transport
        has one with its lane; eth has one, with no lane, when a card is in
        the machine.
    srp_rfdc : rogue memory master, optional
        A second register path, to an RFSoC carrier's RF data converter driver;
        the composition builds the converter's device over it. None where the
        platform has no such device, or it is not reachable.
    register_offset : int
        What to add to every address of the register tree: the FPGA's window in
        a processor system's address space, for a transport that goes through
        one. 0 for a link to the FPGA itself.
    """

    name: str
    srp: Any
    ddr_streams: List[Any]
    streaming_stream: Any
    nodes: List[Any] = field(default_factory=list)
    register_nodes: List[Any] = field(default_factory=list)
    probes_firmware: bool = True
    card: Optional[OffloadCard] = None
    srp_rfdc: Any = None
    register_offset: int = 0


def _udp_data_path(ip_addr: str):
    """The data path over Ethernet: the DDR streams behind RSSI, the streaming interface plain.

    Returns the RSSI link (which carries the register path too, where Ethernet
    carries registers), the four DDR masters, the streaming FIFO, and the nodes
    the link needs in the tree.
    """
    rssi = pyrogue.protocols.UdpRssiPack(name='rudp', host=ip_addr, port=RSSI_PORT,
                                         packVer=2, jumbo=True)
    ddr = [rssi.application(dest) for dest in DDR_TDEST]
    receiver = UdpReceiver(ip_addr=ip_addr, port=STREAMING_PORT)
    fifo = rogue.interfaces.stream.Fifo(100000, 0, True)
    pyrogue.streamConnect(receiver, fifo)
    return rssi, ddr, fifo, [rssi, receiver]


def _dma_data_path(lane: int, dev_rssi: str, dev_data: str):
    """The data path over the offload card's DMA channels for ``lane``."""
    base = lane * 0x100
    ddr = [rogue.hardware.axi.AxiStreamDma(dev_rssi, base + dest, True)
           for dest in DDR_TDEST]
    streaming = rogue.hardware.axi.AxiStreamDma(dev_data, base + STREAMING_TDEST, True)
    return ddr, streaming


def eth(ip_addr: str, dev_rssi: str = '/dev/datadev_0', offload_card: bool = True) -> Transport:
    """Ethernet: registers over interleaved RSSI, data over UDP.

    The streaming interface arrives on a plain UDP port, without RSSI, and is
    received behind a FIFO by a client that keeps the connection alive.

    Parameters
    ----------
    ip_addr : str
        The FPGA's address.
    dev_rssi : str
        The offload card's register device, if a card is in the machine: its
        lanes pointing at ``ip_addr`` are closed around the server, since an
        open one intercepts the traffic this link needs.
    offload_card : bool
        False leaves a card alone even when present.
    """
    rssi, ddr, fifo, nodes = _udp_data_path(ip_addr)
    srp = rogue.protocols.srp.SrpV3()
    pyrogue.streamConnectBiDir(srp, rssi.application(dest=0x0))
    card = OffloadCard(None, ip_addr, dev_rssi=dev_rssi) \
        if offload_card and os.path.exists(dev_rssi) else None
    return Transport('eth', srp, ddr, fifo, nodes, register_nodes=[rssi], card=card)


def pcie(*, lane: int = 0, ip_addr: str = '', dev_rssi: str = '/dev/datadev_0',
         dev_data: str = '/dev/datadev_1', offload_card: bool = True) -> Transport:
    """A PCIe offload card: registers and data over its DMA channels.

    Parameters
    ----------
    lane : int
        The card's lane to the carrier; its RSSI link is opened around the
        server (`OffloadCard`).
    ip_addr : str
        The carrier's address; '' takes the one the lane was last pointed at.
    dev_rssi, dev_data : str
        The card's register and data devices.
    offload_card : bool
        False does not configure the card: its lane is taken to be open
        already, by whoever opened it.
    """
    # Once, before any DMA channel on the device is opened.
    rogue.hardware.axi.AxiStreamDma.zeroCopyDisable(dev_rssi)
    srp_stream = rogue.hardware.axi.AxiStreamDma(dev_rssi, lane * 0x100, True)
    srp = rogue.protocols.srp.SrpV3()
    pyrogue.streamConnectBiDir(srp, srp_stream)
    ddr, streaming = _dma_data_path(lane, dev_rssi, dev_data)
    card = OffloadCard(lane, ip_addr, dev_rssi=dev_rssi, dev_data=dev_data) if offload_card else None
    return Transport('pcie', srp, ddr, streaming, card=card)


def tcp_bridge(ip_ps: str, ip_pl: str, *, data: str = 'udp', lane: int = 0,
               dev_rssi: str = '/dev/datadev_0', dev_data: str = '/dev/datadev_1',
               offload_card: bool = True) -> Transport:
    """An RFSoC carrier: registers over a TCP bridge on its processor system, data as chosen.

    The FPGA's registers and its RF data converter's driver are each served by
    a daemon on the carrier's processor system (``ip_ps``), reached over TCP;
    no RSSI link carries them. The data leaves the FPGA's own Ethernet port
    (``ip_pl``) exactly as a carrier's does, so the data path is Ethernet's --
    or the offload card's DMA, when the card is in the machine with a lane
    pointed at the FPGA.

    Parameters
    ----------
    ip_ps : str
        The processor system's address: the bridge.
    ip_pl : str
        The FPGA's address: where the data comes from.
    data : {'udp', 'pcie'}
        Receive the data on the host's Ethernet port, or through the offload
        card's DMA channels.
    lane, dev_rssi, dev_data, offload_card
        As `pcie`, for ``data='pcie'``; with ``'udp'`` the card, if present, is
        kept from intercepting the FPGA's traffic as `eth` does.
    """
    if data not in ('udp', 'pcie'):
        raise ValueError(f"data must be 'udp' or 'pcie', not {data!r}")
    srp = rogue.interfaces.memory.TcpClient(ip_ps, BRIDGE_REGISTER_PORT)
    srp_rfdc = rogue.interfaces.memory.TcpClient(ip_ps, BRIDGE_RFDC_PORT)
    if data == 'udp':
        _rssi, ddr, streaming, nodes = _udp_data_path(ip_pl)
        card = OffloadCard(None, ip_pl, dev_rssi=dev_rssi) \
            if offload_card and os.path.exists(dev_rssi) else None
    else:
        rogue.hardware.axi.AxiStreamDma.zeroCopyDisable(dev_rssi)
        ddr, streaming = _dma_data_path(lane, dev_rssi, dev_data)
        nodes = []
        card = OffloadCard(lane, ip_pl, dev_rssi=dev_rssi, dev_data=dev_data) if offload_card else None
    return Transport('tcp-bridge', srp, ddr, streaming, nodes, card=card,
                     srp_rfdc=srp_rfdc, register_offset=BRIDGE_REGISTER_OFFSET)


def emulation(streaming_source: Any = None, rfdc: bool = True) -> Transport:
    """No hardware: registers in memory, data from whatever source is given.

    Parameters
    ----------
    streaming_source : rogue stream master, optional
        Stands in for the streaming interface; added to the tree if it is a node.
        Without one the data path is an idle master.
    rfdc : bool
        Give the RF data converter a memory of its own too, so a platform that
        has one builds it; a platform that does not leaves it unused.
    """
    srp = pyrogue.interfaces.simulation.MemEmulate()
    ddr = [rogue.interfaces.stream.Master() for _ in DDR_TDEST]
    source = streaming_source if streaming_source is not None \
        else rogue.interfaces.stream.Master()
    nodes = [source] if isinstance(source, pyrogue.Node) else []
    return Transport('emulation', srp, ddr, source, nodes, probes_firmware=False,
                     srp_rfdc=pyrogue.interfaces.simulation.MemEmulate() if rfdc else None)
