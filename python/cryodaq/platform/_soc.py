#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform: the processor system that boots an RFSoC carrier
#-----------------------------------------------------------------------------
# File       : _soc.py
# Created    : 2026-11-16
#-----------------------------------------------------------------------------
# Description:
#    How a readout server acts on a carrier whose FPGA is booted by a processor
#    system beside it -- the RFMC carrier, the ZCU208 -- before it has a register
#    tree: over SSH to that processor's Linux. The image is the set of boot
#    files on the processor's SD card, so loading one is copying a release's
#    boot files into place, checking that they arrived whole, and rebooting.
#    Everything here wraps the standard tools (`scp`, `ssh`, `ping`) through
#    the seam the shelf manager's tools go through (`_atca.run_tool`) and
#    parses only what they print. Nothing here reads or writes a register, and
#    nothing here imports rogue.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import contextlib
import hashlib
import logging
import os
import socket
import tarfile
import tempfile
import re
import time
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from cryodaq._errors import ConnectError, CryodaqError
from cryodaq.platform import _atca

_log = logging.getLogger(__name__)

__all__ = ['ProcessorSystem', 'ProcessorSystemError', 'port_open', 'BOOT_FILES', 'BOOT_DIR', 'IMAGE_SUFFIX',
           'IMAGE_HASH_RE', 'REGISTER_BRIDGE_PORT', 'USER', 'COPY_TIMEOUT_S', 'SHUTDOWN_TIMEOUT_S',
           'REBOOT_TIMEOUT_S', 'POLL_S', 'VERSION_RETRIES', 'VERSION_RETRY_DELAY_S']

# The boot files a release ships, under one directory in its image archive
# (``<image>-0x<version>-<date>-<user>-<hash>.linux.tar.gz``): the bitstream,
# the boot loader and its script, the kernel with the root filesystem, and the
# first-stage boot loader. All five go to the card: a kernel built for one
# bitstream is not run with another.
BOOT_FILES = ('system.bit', 'BOOT.BIN', 'boot.scr', 'image.ub', 'zynqmp_fsbl.elf')
IMAGE_DIR = 'linux'
IMAGE_SUFFIX = '.linux.tar.gz'
IMAGE_HASH_RE = re.compile(r'-([0-9a-f]{7})[0-9a-f]*\.linux\.tar\.gz$')
# Where the running system mounts the SD card's boot partition, read-write.
BOOT_DIR = '/boot'
# The processor's root account, which the firmware's Linux ships without a password.
USER = 'root'
# The TCP port of the register bridge on the processor system, which answers
# once the system is up and the FPGA is programmed; the server reads the
# firmware's version block through it.
REGISTER_BRIDGE_PORT = 9000
# Non-interactive: a password prompt is a failure that says so, never a hang.
# No host key is kept: a new image brings a new root filesystem, and with it
# may come a new key. A connection the far end drops without closing (a reboot)
# is given up after two missed keep-alives.
SSH_OPTIONS = ('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no', '-o', 'UserKnownHostsFile=/dev/null',
               '-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=5',
               '-o', 'ServerAliveCountMax=2')
# A copy of some 140 MB onto an SD card over the crate's base channel; the
# shelf manager's one-minute tool timeout is not enough for it.
COPY_TIMEOUT_S = 600
# How long the reboot command is given to return (it need not: the reboot cuts
# the connection); then how long the system is given to stop answering, and to
# answer again with its bridge up. The boards take about a minute to boot.
REBOOT_COMMAND_TIMEOUT_S = 30
SHUTDOWN_TIMEOUT_S = 60
REBOOT_TIMEOUT_S = 180
POLL_S = 2
# How many times, and how often, the firmware's version is asked for after a
# reboot: the register bridge accepts connections before it serves.
VERSION_RETRIES = 15
VERSION_RETRY_DELAY_S = 2
# Files are checksummed in pieces of this size: the kernel image is over 100 MB.
_CHUNK = 1 << 20
# What a copy that did not finish, or did not check out, leaves behind.
_MIXED_CARD = (f"{BOOT_DIR} on the processor system may now hold a mix of old and new boot files, which its "
               f"next boot would run; start the server again to copy them all")


class ProcessorSystemError(CryodaqError):
    """A tool run against the processor system failed, or its answer cannot be read.

    Parameters
    ----------
    what : str
        Which action failed, in the words of this module.
    argv : sequence of str
        The tool invocation.
    detail : str
        What the tool printed, or why its output could not be used.
    timed_out : bool
        Whether the tool was still running when its time was up, so that what
        it did is unknown -- a copy may be partial, a reboot may have begun.
    """

    def __init__(self, what: str, argv: Sequence[str], detail: str = '', timed_out: bool = False):
        self.what = what
        self.argv = list(argv)
        self.detail = detail
        self.timed_out = timed_out
        why = f": {detail.strip()}" if detail.strip() else ''
        super().__init__(f"{what} failed ({' '.join(argv)}){why}")


def port_open(address: str, port: int, timeout: float = 2.0) -> bool:
    """Whether a TCP connection to ``address:port`` is accepted now.

    Parameters
    ----------
    address, port
        Where to connect.
    timeout : float
        Seconds to wait for the connection.
    """
    try:
        with contextlib.closing(socket.create_connection((address, port), timeout=timeout)):
            return True
    except OSError:
        return False


def _run(what: str, argv: Sequence[str], timeout: Optional[float] = None) -> Tuple[int, str]:
    """A tool through the shared seam; a tool that is missing or does not return is this module's error."""
    try:
        return _atca.run_tool(argv) if timeout is None else _atca.run_tool(argv, timeout)
    except _atca.ShelfManagerError as e:
        raise ProcessorSystemError(what, argv, e.detail, timed_out=isinstance(e, _atca.ToolTimeout)) from None


class ProcessorSystem:
    """One carrier's processor system, reached over SSH.

    Parameters
    ----------
    address : str
        The processor system's address: where its Linux answers, and where the
        register bridge runs once it is up.
    read_version : callable, optional
        ``read_version(address)`` returns the firmware's version block as read
        through the register bridge -- an object with a ``short_hash`` --
        raising `ConnectError` when the bridge does not answer. The read needs
        the firmware's own tree, which this package does not import, so the
        reader is handed in; without one the firmware's hash cannot be read.

    Loading an image, rebooting and reading the hash of the firmware the FPGA
    runs are the verbs this object shares with a crate's shelf manager, so a
    launcher needs no word on which it holds.
    """

    IMAGE_SUFFIX = IMAGE_SUFFIX

    def __init__(self, address: str, read_version: Optional[Callable[[str], object]] = None):
        self.address = address
        self.target = f"{USER}@{address}"
        self.read_version = read_version

    def __str__(self) -> str:
        return f"the processor system at {self.address}"

    @staticmethod
    def image_hash(image: 'os.PathLike[str] | str') -> Optional[str]:
        """The short git hash an image's file name carries, or None when it carries none."""
        m = IMAGE_HASH_RE.search(os.path.basename(str(image)))
        return m.group(1) if m else None

    def firmware_git_hash(self, attempts: int = 1) -> str:
        """The short git hash of the firmware the FPGA runs, from its version block through the register bridge.

        Asked up to ``attempts`` times, `VERSION_RETRY_DELAY_S` apart, for a
        bridge that is coming back after a reboot.
        """
        argv = ['read_version', self.address]
        if self.read_version is None:
            raise ProcessorSystemError('reading the firmware version', argv, 'no reader of the version block was given')
        for attempt in range(attempts):
            try:
                return self.read_version(self.address).short_hash
            except ConnectError as e:
                if attempt + 1 == attempts:
                    raise ProcessorSystemError('reading the firmware version', argv, str(e)) from e
                time.sleep(VERSION_RETRY_DELAY_S)

    # ------------------------------------------------------------------
    # the tools
    # ------------------------------------------------------------------

    def _ssh(self, what: str, command: str, timeout: Optional[float] = None) -> str:
        argv = ['ssh', *SSH_OPTIONS, self.target, command]
        status, out = _run(what, argv, timeout)
        if status != 0:
            raise ProcessorSystemError(what, argv, out or f"exit status {status}")
        return out

    def answers(self) -> bool:
        """Whether the processor system answers a ping now."""
        status, _ = _run('pinging the processor system', ['ping', '-c', '1', '-W', '1', self.address])
        return status == 0

    # ------------------------------------------------------------------
    # the image
    # ------------------------------------------------------------------

    @staticmethod
    @contextlib.contextmanager
    def boot_files(image: 'os.PathLike[str] | str') -> Iterator[List[str]]:
        """The release's boot files, taken out of its image archive for as long as the block runs.

        The archive must hold exactly the files a release ships (`BOOT_FILES`,
        under `IMAGE_DIR`) and nothing else: a member with another name, or
        one that is not a plain file, is refused before anything is extracted.

        Parameters
        ----------
        image : path-like
            The release's ``.linux.tar.gz``.

        Yields
        ------
        list of str
            The five files' paths, in `BOOT_FILES` order.
        """
        image = os.fspath(image)
        what, argv = 'reading the boot image', ['tar', 'tzf', image]
        expected = {f"{IMAGE_DIR}/{name}" for name in BOOT_FILES}
        with tempfile.TemporaryDirectory(prefix='boot_image_') as tmp:
            try:
                with tarfile.open(image) as tar:
                    members = [m for m in tar.getmembers() if not m.isdir()]
                    names = {m.name for m in members}
                    missing, stray = sorted(expected - names), sorted(names - expected)
                    if missing or stray:
                        raise ProcessorSystemError(what, argv, f"missing {missing}, unexpected {stray}; a release "
                                                   f"holds exactly {IMAGE_DIR}/{{{','.join(BOOT_FILES)}}}")
                    odd = [m.name for m in members if not m.isfile()]
                    if odd:
                        raise ProcessorSystemError(what, argv, f"not plain files: {odd}")
                    tar.extractall(tmp, members=members, filter='data')
            except (tarfile.TarError, OSError) as e:
                raise ProcessorSystemError(what, argv, str(e)) from e
            yield [os.path.join(tmp, IMAGE_DIR, name) for name in BOOT_FILES]

    def program(self, image: 'os.PathLike[str] | str') -> None:
        """Put a release's boot files on the SD card, reboot into them, and read back that the FPGA runs them.

        The copy is checked file by file against the card before the reboot:
        a file that did not arrive whole is the failure to catch, and it is
        cheap to catch here and dear after the reboot. The reboot returns
        with the hash the FPGA reports once the system is back; one still
        differing from the image's is an error, not a warning -- a server
        would come up on firmware other than the one on disk. Without a
        reader of the version block nothing is written: the load could not
        be read back.

        Parameters
        ----------
        image : path-like
            The release's ``.linux.tar.gz``, its name carrying the hash.
        """
        expected = self.image_hash(image)
        if self.read_version is None:
            raise ProcessorSystemError('loading the boot image', ['read_version', self.address],
                                       'no reader of the version block was given, so the load could not be read back')
        with self.boot_files(image) as files:
            self.copy_boot_files(files)
            self.verify_boot_files(files)
        copied = f"the boot files of {os.path.basename(str(image))} were copied to the processor system at {self.address}"
        try:
            running = self.reboot()
        except ProcessorSystemError as e:
            raise ProcessorSystemError('rebooting into the copied boot files', e.argv, f"{copied}, but {e.detail}") from e
        if running != expected:
            raise ProcessorSystemError('reading back the firmware after the load', ['read_version', self.address],
                                       f"{copied} and it was rebooted, but the FPGA reports firmware {running}, not "
                                       f"{expected}; the boot files did not take -- check {BOOT_DIR} there")
        _log.info("FPGA rebooted into %s", running)

    def copy_boot_files(self, files: Sequence[str]) -> None:
        """Copy the boot files onto the card's boot partition, over the ones there.

        `scp` of OpenSSH 9 copies over SFTP, which a processor system without an
        SFTP server refuses before any write; the copy is then tried once more
        over the original protocol (``-O``). A copy that fails may leave the
        card with some new files and some old, which the next boot would run;
        the error says so, and running the launcher again copies them all.

        Parameters
        ----------
        files : sequence of str
            The five boot files, as `boot_files` yields them.
        """
        what = 'copying the boot files'
        argv = ['scp', *SSH_OPTIONS, *files, f"{self.target}:{BOOT_DIR}/."]
        try:
            status, out = _run(what, argv, COPY_TIMEOUT_S)
            if status != 0:
                _log.info("scp over SFTP failed (%s); copying over the original scp protocol instead", out.strip())
                argv = ['scp', '-O', *SSH_OPTIONS, *files, f"{self.target}:{BOOT_DIR}/."]
                status, out = _run(what, argv, COPY_TIMEOUT_S)
        except ProcessorSystemError as e:
            raise ProcessorSystemError(what, e.argv, f"{e.detail} -- {_MIXED_CARD}", timed_out=e.timed_out) from None
        if status != 0:
            raise ProcessorSystemError(what, argv, f"{out or f'exit status {status}'} -- {_MIXED_CARD}")

    def verify_boot_files(self, files: Sequence[str]) -> None:
        """Check that every boot file on the card is byte for byte the one sent.

        Parameters
        ----------
        files : sequence of str
            The five boot files, as `boot_files` yields them.
        """
        remote = ' '.join(f"{BOOT_DIR}/{os.path.basename(f)}" for f in files)
        try:
            out = self._ssh('checking the copied boot files', f"sha256sum {remote}")
        except ProcessorSystemError as e:
            raise ProcessorSystemError(e.what, e.argv, f"{e.detail} -- {_MIXED_CARD}", timed_out=e.timed_out) from None
        on_card = _checksums(out)
        wrong = []
        for path in files:
            name = os.path.basename(path)
            local = _sha256(path)
            if on_card.get(name) != local:
                wrong.append(f"{name}: sent {local[:12]}, on the card {str(on_card.get(name, 'absent'))[:12]}")
        if wrong:
            raise ProcessorSystemError('checking the copied boot files', ['sha256sum', remote],
                                       f"{'; '.join(wrong)} -- {_MIXED_CARD}")

    # ------------------------------------------------------------------
    # the reboot
    # ------------------------------------------------------------------

    def reboot(self) -> Optional[str]:
        """Flush the card and reboot; return once the system is back and the FPGA's version block reads again.

        The flush before the reboot matters: a reboot during a write to the
        card can corrupt the files just copied. The reboot cuts the connection,
        so neither the command's exit status nor its failing to return says
        anything; the system going away does.

        Returns
        -------
        str or None
            The short hash the FPGA reports once the system is back, or None
            when no reader of the version block was given and the bridge's
            port opening was all there was to wait for.
        """
        argv = ['ssh', *SSH_OPTIONS, self.target, '/bin/sync; /sbin/reboot']
        said = ''
        try:
            status, out = _run('rebooting', argv, REBOOT_COMMAND_TIMEOUT_S)
            if status != 0:
                # Either the connection was cut by the reboot, or ssh refused;
                # the system going away tells which, and what ssh said is kept
                # for the error if it does not.
                said = out.strip() or f"exit status {status}"
        except ProcessorSystemError as e:
            if not e.timed_out:
                raise
        return self.wait_for_reboot(said)

    def wait_for_reboot(self, said: str = '') -> Optional[str]:
        """Wait for the system to stop answering, then to answer with its register bridge serving again.

        The bridge accepts connections before it serves registers, so once
        its port is open the version block is asked for, up to
        `VERSION_RETRIES` times, and the hash it carries is returned -- the
        same wait whether the reboot was a load's or asked for on its own.

        Parameters
        ----------
        said : str
            What the reboot command printed, if it failed; named in the error
            when the system never goes away.

        Returns
        -------
        str or None
            The firmware's short hash, or None when no reader was given.
        """
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        while self.answers():
            if time.monotonic() >= deadline:
                why = f" (the reboot command said: {said})" if said else ''
                raise ProcessorSystemError('rebooting', ['ssh', self.target, '/sbin/reboot'],
                                           f"the processor system at {self.address} still answers after "
                                           f"{SHUTDOWN_TIMEOUT_S} s; it did not reboot{why}")
            time.sleep(POLL_S)
        deadline = time.monotonic() + REBOOT_TIMEOUT_S
        while not (self.answers() and port_open(self.address, REGISTER_BRIDGE_PORT)):
            if time.monotonic() >= deadline:
                raise ProcessorSystemError('waiting for the reboot', ['ping', self.address],
                                           f"the processor system at {self.address} did not come back with its "
                                           f"register bridge on port {REGISTER_BRIDGE_PORT} in {REBOOT_TIMEOUT_S} s")
            time.sleep(POLL_S)
        if self.read_version is None:
            return None
        try:
            return self.firmware_git_hash(VERSION_RETRIES)
        except ProcessorSystemError as e:
            raise ProcessorSystemError('waiting for the reboot', e.argv,
                                       f"the processor system at {self.address} came back with its register bridge "
                                       f"on port {REGISTER_BRIDGE_PORT}, but the FPGA's version block still could not "
                                       f"be read: {e.detail}") from e


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(_CHUNK), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checksums(text: str) -> Dict[str, str]:
    """``{file name: hex digest}`` from `sha256sum`'s ``<digest>  <path>`` lines."""
    sums: Dict[str, str] = {}
    for line in text.splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[os.path.basename(parts[1].strip())] = parts[0]
    return sums
