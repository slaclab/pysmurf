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
#    The card's register map is its firmware package's (`SmurfPcie`), which the
#    deployment puts on the path; it is bound to the device's memory map and
#    read and written by register name, without starting a tree of the card.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import ipaddress
import logging
import os
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
# A register transaction's deadline, seconds.
TRANSACTION_TIMEOUT_S = 1.0


class Registers:
    """The card firmware's register map over one device node, bound but not running.

    The map is the firmware package's own (``SmurfPcie``), so no offset is
    spelled here. Rogue binds a device's registers to a memory slave when the
    device is attached to a root; this attaches one and goes no further -- the
    root is never started, so it runs no update, heartbeat or poll thread and
    serves nothing -- and a register is read or written by name, synchronously.
    (The firmware package itself starts one idle helper thread when its tree is
    built; it ends with the device.)

    Parameters
    ----------
    slave : rogue memory slave
        The device's memory map, or a stand-in.
    role : str
        For the log.
    """

    def __init__(self, slave: Any, role: str):
        try:
            from SmurfPcie import SmurfKcu1500RssiOffload10GbE as card
        except ImportError as e:
            raise ConnectError(f"the offload card's register map needs the SmurfPcie "
                               f"package, which is not importable: {e}") from e
        self._root = pyrogue.Root(name='card', description=f'PCIe card ({role})',
                                  pollEn=False, initRead=False)
        # The slave is an interface of the root so that stopping the root
        # closes it; an unstarted root would otherwise never touch it.
        self._root.addInterface(slave)
        self._root.add(card.Core(memBase=slave))
        try:
            self._root._rootAttached()
            self._root._finishInit()
            self.core = self._root.Core
            # Only a started root hands its deadline to the blocks.
            self.core._setTimeout(TRANSACTION_TIMEOUT_S)
            # A field is written from the word its block holds, so the words
            # are read once first; a lane's control word shares bits with
            # fields the lane does not own. (A started root does this read.)
            self.core.readBlocks(recurse=True)
            self.core.waitBlocks(recurse=True)
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        """Release the device and whatever the map's package started."""
        pyrogue.Device._stop(self._root)


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
        If a device is missing or is not the device its role needs, the
        address is unusable, or the lane does not connect.
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
        self._regs: Optional[Registers] = None

    # -- the devices ---------------------------------------------------------

    def _open_device(self, dev: str, role: str, device_id: int) -> Registers:
        """The register map over ``dev``, checked to be the ``role`` device."""
        if self._memory is None and not os.path.exists(dev):
            raise ConnectError(f"PCIe device {dev} ({role}) is not present")
        if self._memory is not None and dev not in self._memory:
            raise ConnectError(f"PCIe device {dev} ({role}) is not present (no stand-in)")
        slave = self._memory[dev] if self._memory is not None else rogue.hardware.axi.AxiMemMap(dev)
        try:
            regs = Registers(slave, role)
        except ConnectError:
            raise
        except Exception as e:
            # Binding reads every register once; a card that does not answer
            # fails there, before anything of it is looked at.
            raise ConnectError(f"could not read PCIe device {dev} ({role}): {e}") from e
        try:
            found = int(regs.core.AxiPcieCore.AxiVersion.DeviceId.get())
        except Exception as e:
            regs.stop()
            raise ConnectError(f"could not read PCIe device {dev} ({role}): {e}") from e
        if found != device_id:
            regs.stop()
            raise ConnectError(f"PCIe device {dev} reports DeviceId {found}, not the "
                               f"{device_id} of the {role} device")
        return regs

    # -- the lanes -----------------------------------------------------------

    def _open_lane(self, lane: int) -> None:
        core = self._regs.core
        _log.info("opening the offload card's RSSI lane %d to %s", lane, self.ip_addr)
        core.UdpGrp.UdpConfig[lane].EnKeepAlive.set(1)
        core.UdpGrp.UdpConfig[lane].KeepAliveConfig.set(KEEP_ALIVE_CONFIG)
        core.UdpGrp.RssiClient[lane].OpenConn.set(1)
        core.UdpGrp.RssiClient[lane].CloseConn.set(0)
        for client, port in enumerate(REMOTE_PORTS):
            core.UdpGrp.UdpEngine[lane].ClientRemoteIp[client].setDisp(self.ip_addr)
            core.UdpGrp.UdpEngine[lane].ClientRemotePort[client].set(port)
        self._wait_connected(lane)

    def _close_lane(self, lane: int) -> None:
        core = self._regs.core
        _log.info("closing the offload card's RSSI lane %d", lane)
        core.UdpGrp.UdpConfig[lane].KeepAliveConfig.set(0)
        core.UdpGrp.RssiClient[lane].OpenConn.set(0)
        core.UdpGrp.RssiClient[lane].CloseConn.set(1)
        for client in range(len(REMOTE_PORTS)):
            core.UdpGrp.UdpEngine[lane].ClientRemotePort[client].set(0)

    def _connected(self, lane: int) -> bool:
        return bool(self._regs.core.UdpGrp.RssiClient[lane].ConnectionActive.get())

    def _wait_connected(self, lane: int) -> None:
        """Wait for the lane's RSSI connection to be up, within a bound.

        Opening the connection starts a handshake with the carrier; a register
        read sent before it completes times out. The connection's state is the
        card's to report, so it is polled rather than slept for.
        """
        deadline = time.monotonic() + CONNECT_TIMEOUT_S
        while True:
            if self._connected(lane):
                _log.info("lane %d connected to %s", lane, self.ip_addr)
                return
            if time.monotonic() > deadline:
                raise ConnectError(f"the offload card's lane {lane} did not connect to "
                                   f"{self.ip_addr} within {CONNECT_TIMEOUT_S:g} s -- is the "
                                   f"carrier up at that address?")
            time.sleep(CONNECT_POLL_S)

    def _remote_ip(self, lane: int) -> str:
        return self._regs.core.UdpGrp.UdpEngine[lane].ClientRemoteIp[0].getDisp()

    def _close_lanes_to(self, ip_addr: str) -> None:
        """Close every lane pointing at ``ip_addr``, whichever it is."""
        for lane in LANES:
            if self._remote_ip(lane) == ip_addr:
                _log.info("lane %d points at %s; closing it", lane, ip_addr)
                self._close_lane(lane)

    def _default_addresses(self, lane: int) -> None:
        """A lane never configured has no MAC or IP; give it the conventional ones."""
        config = self._regs.core.EthPhyGrp.EthConfig[lane]
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
            # server reaches it through its DMA channels.
            self._open_device(self._dev_data, 'data', DATA_DEVICE_ID).stop()
        self._regs = self._open_device(self._dev_rssi, 'registers', RSSI_DEVICE_ID)
        try:
            if self.lane is not None:
                self._default_addresses(self.lane)
                if not self.ip_addr:
                    self.ip_addr = self._remote_ip(self.lane)
                if self.ip_addr == '0.0.0.0':
                    raise ConnectError(f"lane {self.lane} has never been pointed at a carrier "
                                       f"and no address was given")
                try:
                    ipaddress.IPv4Address(self.ip_addr)
                except ValueError:
                    raise ConnectError(f"the carrier address {self.ip_addr!r} is not an "
                                       f"address") from None
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
            self._regs.stop()
            self._regs = None
            raise
        return self

    def __exit__(self, *exc) -> None:
        """Close the lane opened on entry."""
        if self._regs is None:
            return
        try:
            if self.lane is not None:
                self._close_lane(self.lane)
        finally:
            self._regs.stop()
            self._regs = None

    def restart(self) -> None:
        """Close and reopen the lane, for a link that has stopped answering."""
        if self._regs is None or self.lane is None:
            raise ConnectError("no RSSI lane is open to restart")
        self._close_lane(self.lane)
        self._open_lane(self.lane)
