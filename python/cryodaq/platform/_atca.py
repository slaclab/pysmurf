#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform: the ATCA shelf manager
#-----------------------------------------------------------------------------
# File       : _atca.py
# Created    : 2026-10-30
#-----------------------------------------------------------------------------
# Description:
#    What a readout server learns about an ATCA carrier before it has a register
#    tree, and how it acts on the carrier without one: through the crate's shelf
#    manager over IPMI. The carrier's IPMC answers for the crate id, the firmware
#    git hash the FPGA is running, the boot state, and the FRU records of the
#    carrier, its AMC bays and its rear transition module; and it takes the
#    reboot command. Everything here wraps SLAC's command-line tools --
#    `ipmitool` for the raw IPMI commands and FRU reads, `cba_fru_init` and
#    `cba_amc_init` for the board inventory -- and parses only what they print.
#
#    Every tool call goes through one injected `run` function, so the module can
#    be exercised against recorded tool output with no crate, and so there is one
#    place that knows a subprocess is involved. A board whose IPMC does not
#    answer -- a bench board with no management controller -- is reported as
#    such, by the call that failed, and never guessed about. Nothing here reads
#    or writes a register, and nothing here imports rogue.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from cryodaq._errors import CryodaqError

__all__ = ['ShelfManager', 'ShelfManagerError', 'FruRecord', 'Runner', 'run_tool',
           'ipmb_address', 'fpga_address', 'SLOTS', 'AMC_BAYS', 'RTM_FRU_ID', 'BOOTED']

# A function that runs a tool: argv in, (exit status, standard output) out.
# The default runs it as a subprocess; a check hands in one that replays
# recorded output.
Runner = Callable[[Sequence[str]], Tuple[int, str]]

# The slots a front board can sit in: the two hub slots are 1 and 2 on a
# 7-slot shelf, but SLAC's convention numbers payload carriers 2 through 7.
SLOTS = range(2, 8)
# The FRU device ids on a front board's IPMC: 0 is the board itself, 1 its
# rear transition module (PICMG 3.0: the RTM is FRU 1 of the front board).
CARRIER_FRU_ID = 0
RTM_FRU_ID = 1
# The AMC bays as the inventory tool numbers them: two bays, at 0 and 2.
AMC_BAYS = (0, 2)
# How long one tool call may take; the shelf manager answers in well under a
# second, and a controller that does not answer is reported by the tool itself
# ('No Response') within a few seconds.
TOOL_TIMEOUT_S = 60
# The boot state the carrier reports once its FPGA is up.
BOOTED = 3


class ShelfManagerError(CryodaqError):
    """A shelf-manager tool failed or answered something that cannot be read.

    Parameters
    ----------
    what : str
        Which read or action failed, in the words of this module.
    argv : sequence of str
        The tool invocation.
    detail : str
        What the tool printed, or why its output could not be parsed.
    """

    def __init__(self, what: str, argv: Sequence[str], detail: str = ''):
        self.what = what
        self.argv = list(argv)
        self.detail = detail
        why = f": {detail.strip()}" if detail.strip() else ''
        super().__init__(f"{what} failed ({' '.join(argv)}){why}")


def run_tool(argv: Sequence[str]) -> Tuple[int, str]:
    """Run a tool and return its exit status and what it printed.

    Standard error is folded into the output: `ipmitool` reports an absent or
    silent controller there, and the message is what a reader needs.
    """
    if shutil.which(argv[0]) is None:
        raise ShelfManagerError('finding the tool', argv, f"{argv[0]} is not installed")
    done = subprocess.run(list(argv), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, timeout=TOOL_TIMEOUT_S, check=False)
    return done.returncode, done.stdout


def ipmb_address(slot: int) -> str:
    """The IPMB address of the front board in a slot, as `ipmitool -t` takes it.

    ``0x80 + 2 * slot``, written in hex: `ipmitool` reads a decimal target as
    well, but a hex one is what every hand-written note and log shows.
    """
    if slot not in SLOTS:
        raise ValueError(f"slot {slot} is not in {SLOTS.start}..{SLOTS.stop - 1}")
    return f"0x{0x80 + 2 * slot:02x}"


def fpga_address(crate_id: int, slot: int, offset: int = 100) -> str:
    """The IP address convention: ``10.<crate id high byte>.<low byte>.<offset + slot>``.

    The FPGA's own port is at offset 100; a carrier with a processor system
    beside the FPGA keeps that at 200.
    """
    return f"10.{(crate_id >> 8) & 0xff}.{crate_id & 0xff}.{offset + slot}"


@dataclass(frozen=True)
class FruRecord:
    """A board's FRU record, as the shelf manager reports it.

    Only the fields the bring-up decides on are named; the whole record is kept
    as the tool printed it.

    Parameters
    ----------
    present : bool
        Whether a board answered at this FRU id at all.
    part_number : str
        ``Board Part Number``; empty when absent.
    product_name : str
        ``Board Product Name`` (or ``Board Product``); empty when absent.
    serial : str
        ``Board Serial No.`` (or ``Board Serial``); empty when absent.
    fields : dict
        Every ``name : value`` line the tool printed.
    """

    present: bool
    part_number: str = ''
    product_name: str = ''
    serial: str = ''
    fields: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def absent(cls) -> 'FruRecord':
        """The record of a board that is not there."""
        return cls(present=False)

    @classmethod
    def parse(cls, text: str) -> 'FruRecord':
        """Read a FRU dump. No ``Board Part Number`` line means no board."""
        fields = {}
        for line in text.splitlines():
            m = re.match(r'\s*([A-Za-z][A-Za-z0-9 ./_-]*?)\s*:\s*(.*?)\s*$', line)
            if m and m.group(2):
                fields[m.group(1)] = m.group(2)
        part = fields.get('Board Part Number', '')
        if not part:
            return cls.absent()
        return cls(present=True, part_number=part,
                   product_name=fields.get('Board Product Name', fields.get('Board Product', '')),
                   serial=fields.get('Board Serial No.', fields.get('Board Serial', '')),
                   fields=fields)


class ShelfManager:
    """One carrier slot of one crate, reached through the shelf manager.

    Parameters
    ----------
    host : str
        The shelf manager's host name or address.
    slot : int
        The carrier's slot.
    run : callable, optional
        How tools are run; the default runs them as subprocesses.
    sleep : callable, optional
        How a wait sleeps; a check hands in a no-op.
    """

    def __init__(self, host: str, slot: int, run: Runner = run_tool,
                 sleep: Callable[[float], None] = time.sleep):
        self.host = host
        self.slot = slot
        self.ipmb = ipmb_address(slot)
        self._run = run
        self._sleep = sleep

    # ------------------------------------------------------------------
    # the tools
    # ------------------------------------------------------------------

    def _ipmitool(self, what: str, *args: str) -> str:
        argv = ['ipmitool', '-I', 'lan', '-H', self.host, '-t', self.ipmb, '-b', '0',
                '-A', 'NONE', *args]
        status, out = self._run(argv)
        if status != 0:
            raise ShelfManagerError(what, argv, out)
        return out

    def _raw(self, what: str, *command: str) -> List[int]:
        """An OEM raw command; the answer is a line of hex bytes."""
        out = self._ipmitool(what, 'raw', *command)
        try:
            return [int(tok, 16) for tok in out.split()]
        except ValueError:
            raise ShelfManagerError(what, ['ipmitool', 'raw', *command],
                                    f"not hex bytes: {out.strip()!r}") from None

    def _cba(self, what: str, tool: str, target: str) -> str:
        """An inventory dump; the tool prints nothing for an absent board."""
        argv = [tool, '--dump', target]
        status, out = self._run(argv)
        if status != 0:
            raise ShelfManagerError(what, argv, out)
        return out

    # ------------------------------------------------------------------
    # reads
    # ------------------------------------------------------------------

    def crate_id(self) -> int:
        """The crate id the carrier's IPMC holds, from which its addresses follow.

        Two bytes, low first: ``01 00`` is crate 1.
        """
        args = ['0x34', '0x04', '0xFD', '0x02']
        data = self._raw('reading the crate id', *args)
        if len(data) < 2:
            raise ShelfManagerError('reading the crate id', ['ipmitool'] + args,
                                    f"expected two bytes, got {data}")
        return data[1] << 8 | data[0]

    def firmware_git_hash(self) -> str:
        """The short git hash of the firmware the FPGA is running, 7 hex digits.

        The IPMC returns the hash bytes in reverse order; the tool output is
        reversed byte-wise before the first seven digits are taken.
        """
        args = ['0x34', '0x04', '0xe0', '0x04']
        gh_inv = self._raw('reading fw git hash', *args)
        if len(gh_inv) == 0:
            raise ShelfManagerError('reading fw git hash', ['ipmitool'] + args,
                                    'empty response')

        return ''.join([f'{i:02x}' for i in gh_inv[::-1]])[:7]

    def boot_state(self) -> int:
        """The BSI boot state byte; `BOOTED` once the FPGA is up."""
        data = self._raw('reading the boot state', '0x34', '0xF4')
        if not data:
            raise ShelfManagerError('reading the boot state', ['ipmitool', 'raw', '0x34', '0xF4'],
                                    'empty response')
        return data[0]

    def reboot(self, address: Optional[str] = None, retry_max: int = 10, retry_delay: float = 10) -> None:
        """Power-cycle the FPGA through the IPMC and wait until it has booted and answers at ``address``.

        ``address`` defaults to the one the crate id implies. ``retry_max`` polls
        of the boot state and then of the address, ``retry_delay`` apart.
        """
        # cycle power to the FPGA
        self._raw('sending power down command to FPGA', '0x2C', '0x0A', '0', '0', '2', '0')
        self._sleep(5)
        self._raw('sending power up command to FPGA', '0x2C', '0x0A', '0', '0', '1', '0')

        # wait for FPGA to boot
        fpga_ready = False
        fpga_state = None
        for i in range(retry_max):
            self._sleep(retry_delay)
            try:
                fpga_state = self.boot_state()
            except ShelfManagerError:
                continue                        # not answering yet
            if fpga_state == BOOTED:
                fpga_ready = True
                break

        if not fpga_ready:
            raise ShelfManagerError(
                'rebooting FPGA', ['ipmitool', '0x34', '0xF4'],
                f"FPGA didn't boot after {retry_max * retry_delay} seconds. Status={fpga_state}. Aborting..."
            )

        # wait for FPGA ETH
        fpga_ip = address if address is not None else fpga_address(self.crate_id(), self.slot)
        eth_ready = False
        for i in range(retry_max):
            status, out = self._run(['ping', '-c', '2', fpga_ip])
            if status == 0:
                eth_ready = True
                break
            self._sleep(retry_delay)

        if not eth_ready:
            raise ShelfManagerError(
                'waiting for FPGA eth', ['ping', fpga_ip],
                f"FPGA's ETH didn't come up after {retry_max * retry_delay} seconds. Aborting..."
            )

    def program(self, image: 'os.PathLike[str] | str') -> None:
        """Load an FPGA image through the shelf manager; the FPGA reboots into it."""
        argv = ['ProgramFPGA.bash', '-s', self.host, '-n', str(self.slot), '-m', str(image)]
        status, out = self._run(argv)
        if status != 0:
            raise ShelfManagerError('loading the FPGA image', argv, out)

    def carrier(self) -> FruRecord:
        """The carrier board's FRU record (`cba_fru_init --dump host/slot`).

        The tool prints nothing for an absent board, which parses as absent;
        what the part number means is the launcher's to decide.
        """
        out = self._cba('reading the carrier record', 'cba_fru_init', f'{self.host}/{self.slot}')
        return FruRecord.parse(out)

    def amc(self, bay: int) -> FruRecord:
        """One AMC bay's FRU record (`cba_amc_init --dump host/slot/bay`, bay in AMC_BAYS).

        The tool prints nothing for an empty bay, which parses as absent.
        """
        if bay not in AMC_BAYS:
            raise ValueError(f"bay {bay} is not one of {AMC_BAYS}")
        out = self._cba(f'reading AMC bay {bay}', 'cba_amc_init', f'{self.host}/{self.slot}/{bay}')
        return FruRecord.parse(out)

    def rtm(self) -> FruRecord:
        """The rear transition module's FRU record (`ipmitool fru print 1`); absent when the IPMC says so."""
        argv = ['ipmitool', '-I', 'lan', '-H', self.host, '-t', self.ipmb, '-b', '0', '-A', 'NONE',
                'fru', 'print', str(RTM_FRU_ID)]
        status, out = self._run(argv)
        if status != 0:
            # 'Destination unavailable' (completion code 0xD3, as read on a live
            # carrier) is the controller's answer that nothing is fitted.
            if 'Destination unavailable' in out:
                return FruRecord.absent()
            raise ShelfManagerError('reading the rear module record', argv, out)
        return FruRecord.parse(out)

    # ------------------------------------------------------------------
    # the inventory, as one record
    # ------------------------------------------------------------------

    def inventory(self) -> Dict[str, object]:
        """Everything the shelf manager says about this slot, as one record.

        The record is what a server publishes beside its startup arguments so a
        client can see what the launcher saw. A read that fails because the
        IPMC does not answer is recorded as ``None`` for that entry with the
        error's text under ``errors``; one that answers "absent" is recorded as
        absent. Nothing is inferred.
        """
        record: Dict[str, object] = {'shelfmanager': self.host, 'slot': self.slot, 'ipmb': self.ipmb,
                                     'errors': {}}
        reads = (('crate_id', self.crate_id), ('firmware_git_hash', self.firmware_git_hash),
                 ('boot_state', self.boot_state), ('carrier', self.carrier), ('rtm', self.rtm),
                 *((f'amc{bay}', (lambda b=bay: self.amc(b))) for bay in AMC_BAYS))
        for name, read in reads:
            try:
                value = read()
            except ShelfManagerError as e:
                record[name] = None
                record['errors'][name] = str(e)
                continue
            record[name] = value.fields if isinstance(value, FruRecord) and value.present else (
                False if isinstance(value, FruRecord) else value)
        record['rtm_present'] = None if record.get('rtm') is None and 'rtm' in record['errors'] \
            else bool(record.get('rtm'))
        return record
