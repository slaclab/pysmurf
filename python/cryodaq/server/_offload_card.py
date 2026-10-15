#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server PCIe Offload Card
#-----------------------------------------------------------------------------
# File       : _offload_card.py
# Created    : 2026-10-15
#-----------------------------------------------------------------------------
# Description:
#    The PCIe card that carries the RSSI link to a carrier in hardware. The
#    card has its own register space, reached over a memory map rather than
#    the server's tree, and its lanes have to be configured before anything
#    can be read from the carrier through them: an RSSI client per lane,
#    pointed at the carrier's address, opened before the server starts and
#    closed after it stops. The same card, with no lane open, intercepts the
#    UDP traffic a software RSSI link over the NIC would carry, so a server
#    on the Ethernet transport closes the lanes pointing at its carrier first.
#
#    The card's register tree is the `SmurfPcie` package's, which the
#    deployment puts on the path; it is imported when a card is opened.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import logging
import os
import socket
import time
from typing import Any, Optional

import pyrogue
import rogue.hardware.axi

from cryodaq._errors import ConnectError

__all__ = ['OffloadCard', 'LANES', 'RSSI_DEVICE_ID', 'DATA_DEVICE_ID']

_log = logging.getLogger(__name__)

# The card has six Ethernet lanes, each with its own RSSI client.
LANES = range(6)
# The two PCIe devices the card presents, told apart by the DeviceId their
# firmware reports: registers (and the RSSI engines) on one, data on the other.
RSSI_DEVICE_ID = 0
DATA_DEVICE_ID = 1
# The carrier's two UDP servers an RSSI lane talks to: registers, then data.
REMOTE_PORTS = (8198, 8195)
# Keep-alive period register value, 5 s.
KEEP_ALIVE_CONFIG = 0x2E90EDD0
# How long an opened lane gets to report its connection active before the
# card gives up: the RSSI handshake with the carrier takes well under a second
# when the carrier answers, and this bounds the wait when it does not.
CONNECT_TIMEOUT_S = 10.0
CONNECT_POLL_S = 0.05


class OffloadCard:
    """The PCIe offload card, configured around a server.

    Parameters
    ----------
    lane : int or None
        The lane whose RSSI link to open on entry and close on exit. None
        opens nothing: the card is only made to stop intercepting traffic to
        ``ip_addr`` (the Ethernet transport's case).
    ip_addr : str
        The carrier's address. '' reads the address the lane was last pointed
        at from the card itself, which is how a deployment that configured
        the card once keeps running without repeating it.
    dev_rssi, dev_data : str
        The two device nodes.
    memory : mapping of str to rogue memory slave, optional
        Stands in for the device nodes' memory maps, keyed by device node, for
        a card that is not there.

    Raises
    ------
    ConnectError
        If a device is missing or is not the device its role needs, or the
        address is unusable.
    """

    def __init__(self, lane: Optional[int], ip_addr: str = '',
                 dev_rssi: str = '/dev/datadev_0', dev_data: str = '/dev/datadev_1',
                 memory: Any = None):
        if lane is not None and lane not in LANES:
            raise ConnectError(f"RSSI lane {lane} is not one of the card's "
                               f"{LANES.start}..{LANES.stop - 1}")
        self.lane = lane
        self.ip_addr = ip_addr
        self._dev_rssi = dev_rssi
        self._dev_data = dev_data
        self._memory = memory
        self._card: Optional[pyrogue.Root] = None

    # -- the card's register tree --------------------------------------------

    def _tree(self, dev: str, role: str) -> pyrogue.Root:
        """The card's register tree over one device node, started."""
        if self._memory is None and not os.path.exists(dev):
            raise ConnectError(f"PCIe device {dev} ({role}) is not present")
        if self._memory is not None and dev not in self._memory:
            raise ConnectError(f"PCIe device {dev} ({role}) is not present (no stand-in)")
        try:
            from SmurfPcie import SmurfKcu1500RssiOffload10GbE as card
        except ImportError as e:
            raise ConnectError(f"the offload card's register tree needs the SmurfPcie "
                               f"package, which is not importable: {e}") from e
        memory = self._memory[dev] if self._memory is not None else rogue.hardware.axi.AxiMemMap(dev)
        root = pyrogue.Root(name='card', description=f'PCIe card ({role})', pollEn=False,
                            initRead=True)
        root.add(card.Core(memBase=memory))
        root.start()
        return root

    def _expect_device(self, dev: str, role: str, device_id: int) -> pyrogue.Root:
        root = self._tree(dev, role)
        try:
            found = int(root.Core.AxiPcieCore.AxiVersion.DeviceId.get())
        except Exception:
            root.stop()
            raise
        if found != device_id:
            root.stop()
            raise ConnectError(f"PCIe device {dev} reports DeviceId {found}, not the "
                               f"{device_id} of the {role} device")
        return root

    # -- the lanes -----------------------------------------------------------

    def _open_lane(self, lane: int) -> None:
        core = self._card.Core
        _log.info("opening the offload card's RSSI lane %d to %s", lane, self.ip_addr)
        core.UdpGrp.UdpConfig[lane].EnKeepAlive.set(1)
        core.UdpGrp.UdpConfig[lane].KeepAliveConfig.set(KEEP_ALIVE_CONFIG)
        core.UdpGrp.RssiClient[lane].OpenConn.set(1)
        core.UdpGrp.RssiClient[lane].CloseConn.set(0)
        for client, port in enumerate(REMOTE_PORTS):
            core.UdpGrp.UdpEngine[lane].ClientRemoteIp[client].setDisp(self.ip_addr)
            core.UdpGrp.UdpEngine[lane].ClientRemotePort[client].set(port)
        self._wait_connected(lane)

    def _wait_connected(self, lane: int) -> None:
        """Wait for the lane's RSSI connection to be up, within a bound.

        Opening the connection starts a handshake with the carrier; a register
        read sent before it completes times out. The connection's state is the
        card's to report, so it is polled rather than slept for.
        """
        active = self._card.Core.UdpGrp.RssiClient[lane].ConnectionActive
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while True:
            if active.get():
                _log.info("lane %d connected to %s", lane, self.ip_addr)
                return
            if time.monotonic() > deadline:
                raise ConnectError(f"the offload card's lane {lane} did not connect to "
                                   f"{self.ip_addr} within {CONNECT_TIMEOUT_S:g} s -- is the "
                                   f"carrier up at that address?")
            time.sleep(CONNECT_POLL_S)

    def _close_lane(self, lane: int) -> None:
        core = self._card.Core
        _log.info("closing the offload card's RSSI lane %d", lane)
        core.UdpGrp.UdpConfig[lane].KeepAliveConfig.set(0)
        core.UdpGrp.RssiClient[lane].OpenConn.set(0)
        core.UdpGrp.RssiClient[lane].CloseConn.set(1)
        for client in range(len(REMOTE_PORTS)):
            core.UdpGrp.UdpEngine[lane].ClientRemotePort[client].set(0)

    def _remote_ip(self, lane: int) -> str:
        return self._card.Core.UdpGrp.UdpEngine[lane].ClientRemoteIp[0].getDisp()

    def _close_lanes_to(self, ip_addr: str) -> None:
        """Close every lane pointing at ``ip_addr``, whichever it is."""
        for lane in LANES:
            if self._remote_ip(lane) == ip_addr:
                _log.info("lane %d points at %s; closing it", lane, ip_addr)
                self._close_lane(lane)

    def _default_addresses(self, lane: int) -> None:
        """A lane never configured has no MAC or IP; give it the conventional ones."""
        config = self._card.Core.EthPhyGrp.EthConfig[lane]
        if config.LocalMac.getDisp() == '00:00:00:00:00:00':
            config.LocalMac.setDisp(f"08:00:56:00:45:5{lane}")
            _log.info("lane %d had no MAC address; set %s", lane, config.LocalMac.getDisp())
        if config.LocalIp.getDisp() == '0.0.0.0':
            config.LocalIp.setDisp(f"10.0.1.20{lane}")
            _log.info("lane %d had no IP address; set %s", lane, config.LocalIp.getDisp())

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> 'OffloadCard':
        """Check the devices, point the lane at the carrier and open it."""
        if self.lane is not None:
            # The data device is only checked for being the right one; the
            # server reaches it through its DMA channels, not this tree.
            self._expect_device(self._dev_data, 'data', DATA_DEVICE_ID).stop()
        self._card = self._expect_device(self._dev_rssi, 'registers', RSSI_DEVICE_ID)
        try:
            if self.lane is not None:
                self._default_addresses(self.lane)
                if not self.ip_addr:
                    self.ip_addr = self._remote_ip(self.lane)
                try:
                    socket.inet_pton(socket.AF_INET, self.ip_addr)
                except OSError:
                    raise ConnectError(f"the carrier address {self.ip_addr!r} (lane {self.lane}'s "
                                       f"last target) is not an address; give one") from None
                if self.ip_addr == '0.0.0.0':
                    raise ConnectError(f"lane {self.lane} has never been pointed at a carrier "
                                       f"and no address was given")
            if self.ip_addr:
                self._close_lanes_to(self.ip_addr)
            if self.lane is not None:
                try:
                    self._open_lane(self.lane)
                except BaseException:
                    # A lane opened but never connected is closed again, so the
                    # card is left as it was found and not half-way.
                    self._close_lane(self.lane)
                    raise
        except BaseException:
            self._card.stop()
            self._card = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        """Close the lane opened on entry."""
        if self._card is None:
            return
        try:
            if self.lane is not None:
                self._close_lane(self.lane)
        finally:
            self._card.stop()
            self._card = None

    def restart(self) -> None:
        """Close and reopen the lane, for a link that has stopped answering."""
        if self._card is None or self.lane is None:
            raise ConnectError("no RSSI lane is open to restart")
        self._close_lane(self.lane)
        self._open_lane(self.lane)
