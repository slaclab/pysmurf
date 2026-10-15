#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Offload Card Check
#-----------------------------------------------------------------------------
# File       : check_offload_card.py
# Created    : 2026-10-15
#-----------------------------------------------------------------------------
# Description:
# Drives the PCIe offload card's RSSI lane through `cryodaq.server.OffloadCard`
# over an emulated memory, with the card's register tree built from a smurf-pcie
# package -- a release ZIP or a checkout's firmware/python -- so that a release
# the card module cannot drive is found before a server tries to start behind it.
#
# Asserts what an opened lane's registers say, that exit closes what entry
# opened, that the Ethernet case closes only the lanes at the carrier, and that
# a lane never pointed at a carrier is refused when no address is given. No
# hardware: the device nodes are emulated memories. Needs rogue, so it runs in
# the server container; `validate_client_emulated.py` runs it too.
#
# Usage:
#     check_offload_card.py --smurf-pcie <rogue_vX.Y.Z.zip | firmware/python>
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import argparse
import sys
import zipfile

import pyrogue.interfaces.simulation

import cryodaq
import cryodaq.server._offload_card as oc

# The address of the version block's DeviceId register in the card's tree, where
# the emulated data device is told which device it is.
DEVICE_ID_ADDRESS = 0x20500
# `UdpGrp.RssiClient[lane].ConnectionActive`: the lane's RSSI connection is up.
# An emulated card never raises it, so the memory here does, for the lanes a
# check opens -- a real one is raised by the handshake with the carrier.
CONNECTION_ACTIVE_ADDRESS = {lane: 0x800000 + lane * 0x10000 + 0x2000 + 0x40 for lane in oc.LANES}
CARRIER = '10.0.1.104'
LANE = 2


def card_memories(connected=(LANE,)):
    """One emulated memory per device node; the data device reports its DeviceId.

    ``connected`` names the lanes whose connection the card reports active.
    """
    rssi, data = (pyrogue.interfaces.simulation.MemEmulate() for _ in range(2))
    data._data[DEVICE_ID_ADDRESS] = oc.DATA_DEVICE_ID
    for lane in connected:
        rssi._data[CONNECTION_ACTIVE_ADDRESS[lane]] = 1
    return {'/dev/datadev_0': rssi, '/dev/datadev_1': data}


def recording(events):
    """An OffloadCard that records which lanes it opens and closes."""
    class Card(oc.OffloadCard):
        def _open_lane(self, lane):
            events.append(f'open {lane}')
            super()._open_lane(lane)

        def _close_lane(self, lane):
            events.append(f'close {lane}')
            super()._close_lane(lane)
    return Card


def check_a_lane_never_pointed_at_a_carrier_is_refused(events, Card, memory):
    try:
        with Card(LANE, '', memory=memory):
            raise AssertionError("a lane with no target address was opened")
    except cryodaq.ConnectError as e:
        assert f'lane {LANE}' in str(e) and 'address' in str(e), e
    assert events == [], f"a refused entry touched the lanes: {events}"


def check_the_lane_is_opened_as_the_carrier_needs(events, Card, memory):
    card = Card(LANE, CARRIER, memory=memory)
    card.__enter__()
    try:
        assert events == [f'open {LANE}'], f"entry did not open the lane once: {events}"
        core = card._card.Core
        assert core.UdpGrp.UdpConfig[LANE].EnKeepAlive.get() == 1
        assert core.UdpGrp.UdpConfig[LANE].KeepAliveConfig.get() == oc.KEEP_ALIVE_CONFIG
        assert (core.UdpGrp.RssiClient[LANE].OpenConn.get(),
                core.UdpGrp.RssiClient[LANE].CloseConn.get()) == (1, 0)
        for client, port in enumerate(oc.REMOTE_PORTS):
            assert core.UdpGrp.UdpEngine[LANE].ClientRemoteIp[client].getDisp() == CARRIER
            assert core.UdpGrp.UdpEngine[LANE].ClientRemotePort[client].get() == port
        assert core.EthPhyGrp.EthConfig[LANE].LocalMac.getDisp() == f'08:00:56:00:45:5{LANE}'
        assert core.EthPhyGrp.EthConfig[LANE].LocalIp.getDisp() == f'10.0.1.20{LANE}'
        card.restart()
        assert events == [f'open {LANE}', f'close {LANE}', f'open {LANE}'], \
            f"restart did not close and reopen: {events}"
    finally:
        # A failed assertion must not leave the card's root running: the
        # process would hang on its threads instead of reporting.
        card.__exit__(None, None, None)
    assert events[-1] == f'close {LANE}', f"exit did not close the lane: {events}"
    assert card._card is None, "the card's tree was left running"
    assert (core.UdpGrp.RssiClient[LANE].OpenConn.get(),
            core.UdpGrp.RssiClient[LANE].CloseConn.get()) == (0, 1)
    # The address stays: it is how the next server finds the carrier with no --addr.
    assert core.UdpGrp.UdpEngine[LANE].ClientRemoteIp[0].getDisp() == CARRIER


def check_the_ethernet_case_closes_the_lanes_at_the_carrier_only(events, Card, memory):
    events.clear()
    with Card(None, CARRIER, memory=memory):
        pass
    assert events == [f'close {LANE}'], \
        f"eth entry did not close the lane at the carrier, and only it: {events}"
    events.clear()
    with Card(None, '10.0.1.105', memory=memory):
        pass
    assert events == [], f"eth entry closed a lane pointing elsewhere: {events}"


def check_a_lane_remembered_by_the_card_is_reused(events, Card, memory):
    """No address given, but the lane was pointed at a carrier before: that one."""
    events.clear()
    with Card(LANE, '', memory=memory) as card:
        assert card.ip_addr == CARRIER, f"the lane's remembered target was not taken: {card.ip_addr!r}"
    assert events == [f'close {LANE}', f'open {LANE}', f'close {LANE}'], events


def check_a_lane_that_never_connects_is_refused_within_the_bound(events, Card, memory):
    """A carrier that does not answer the handshake is reported, not waited on for good."""
    import time
    events.clear()
    saved = oc.CONNECT_TIMEOUT_S
    oc.CONNECT_TIMEOUT_S = 0.5
    try:
        started = time.monotonic()
        with Card(3, CARRIER, memory=card_memories(connected=())):
            raise AssertionError("a lane whose connection never came up was taken as open")
    except cryodaq.ConnectError as e:
        assert 'lane 3' in str(e) and 'connect' in str(e), e
        assert time.monotonic() - started < 5, "the refusal took far longer than the bound"
    finally:
        oc.CONNECT_TIMEOUT_S = saved
    assert events == ['open 3', 'close 3'], f"a lane that never connected was not closed again: {events}"
    events.clear()


CHECKS = (check_a_lane_never_pointed_at_a_carrier_is_refused,
          check_a_lane_that_never_connects_is_refused_within_the_bound,
          check_the_lane_is_opened_as_the_carrier_needs,
          check_the_ethernet_case_closes_the_lanes_at_the_carrier_only,
          check_a_lane_remembered_by_the_card_is_reused)


def run_all(smurf_pcie, on_path=False):
    """Run every check in order on one emulated card; returns the number that failed.

    Run alone, the package goes to the *front* of the path, so that the tree
    driven is the given package's and not one the environment already carries.
    ``on_path`` says a caller has placed it already -- at the end, when a
    firmware tree is built in the same process: a smurf-pcie release carries
    its own ``surf``, and a carrier tree built over that revision instead of
    its own has overlapping blocks. The card's tree builds over either.
    """
    if not on_path:
        sys.path.insert(0, f"{smurf_pcie}/python" if zipfile.is_zipfile(smurf_pcie) else smurf_pcie)
    import SmurfPcie
    print(f"  SmurfPcie from {SmurfPcie.__file__}")
    events = []
    Card = recording(events)
    memory = card_memories()
    failed = 0
    for check in CHECKS:
        name = check.__name__[len('check_'):]
        try:
            check(events, Card, memory)
        except Exception as e:
            failed += 1
            print(f"  FAIL  {name}")
            print(f"          {type(e).__name__}: {e}")
        else:
            print(f"  ok    {name}")
    return failed


def main():
    ap = argparse.ArgumentParser(description="Drive the offload card's lane over an emulated memory.")
    ap.add_argument('--smurf-pcie', required=True,
                    help="a smurf-pcie release ZIP, or a checkout's firmware/python")
    args = ap.parse_args()
    failed = run_all(args.smurf_pcie)
    print("All checks passed." if not failed else f"{failed} check(s) FAILED.")
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
