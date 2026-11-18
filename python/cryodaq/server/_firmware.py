#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Firmware Package and Platform Probe
#-----------------------------------------------------------------------------
# File       : _firmware.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    What a server does with its firmware before it has a register tree: put the
#    firmware's Python package on the path (a release archive or a checkout),
#    read the build stamp off the hardware to learn which platform this is,
#    import the class that builds that platform's tree, and find the register
#    configuration files it was named -- in the package, or in the deployment's
#    own directory.
#
#    The probe is a tree holding only the version block, at the address every
#    platform here places it; it is started, read and stopped before the real
#    tree is built over the same link. So the platform comes from the hardware in
#    hand, as the tree's identification does once a client connects, and nothing
#    is told which hardware it is driving. The same probe, over a link of its
#    own to a processor system's register bridge, tells the launcher what
#    firmware such a board is running before it decides whether to load the
#    image it has on disk.
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
import sys
import zipfile
from dataclasses import dataclass
from typing import Any, List, Optional

import pyrogue

from cryodaq import platform
from cryodaq._errors import ConfigError, ConnectError
from cryodaq.platform import _soc

__all__ = ['load_package', 'checkout_python_dirs', 'resolve_layers', 'probe_platform', 'read_version',
           'bridge_version', 'FirmwareVersion', 'ROOT_NAME', 'DEFAULTS_SUBDIR']

_log = logging.getLogger(__name__)

# What the root of every tree here is called; the register paths a client
# resolves start with it.
ROOT_NAME = platform.ROOT_PATH
# Where, under the firmware directory, a deployment keeps its defaults: the
# register files a site chooses from, the AMC carrier's per-population defaults
# among them. They are SMuRF operations' files, from the smurf_cfg repository
# the server image pins, on a release cycle of their own -- which is why they
# are not looked for in the firmware package, and the package's own files are
# not looked for here.
DEFAULTS_SUBDIR = 'smurf_cfg/defaults'


def load_package(path: Optional[os.PathLike]) -> None:
    """Put a firmware package's Python on the import path.

    Parameters
    ----------
    path : path-like or None
        A release archive (its ``python/`` directory is added), a firmware
        checkout (its ``firmware/python`` and each submodule's are added, the
        top level's found first), or any other directory (added as given).
        None leaves the path alone, for a package already importable.

    Raises
    ------
    FileNotFoundError
        If the path names nothing.
    """
    if path is None:
        return
    path = os.fspath(path)
    if zipfile.is_zipfile(path):
        entries = [f"{path}/python"]
    elif os.path.isdir(path):
        entries = checkout_python_dirs(path) or [path]
    else:
        raise FileNotFoundError(f"no firmware package at {path}")
    # Idempotent: an entry point that needs the package before compose() does
    # (to open an offload card) loads it first, and compose() loads it again.
    # addLibraryPath puts each entry at the front, so the last one added is
    # found first: the submodules go in before the top level.
    for entry in reversed(entries):
        if entry not in sys.path:
            pyrogue.addLibraryPath(entry)


def checkout_python_dirs(path: os.PathLike) -> List[str]:
    """A firmware checkout's Python directories, the top level's first; empty if ``path`` is not one.

    The repositories keep the package at ``firmware/python`` and each
    submodule's at ``firmware/submodules/<sub>/python`` or
    ``firmware/submodules/<sub>/firmware/python``.
    """
    fw = os.path.join(os.fspath(path), 'firmware')
    top = os.path.join(fw, 'python')
    if not os.path.isdir(top):
        return []
    found = [top]
    subs = os.path.join(fw, 'submodules')
    for sub in sorted(os.listdir(subs)) if os.path.isdir(subs) else []:
        for candidate in (os.path.join(subs, sub, 'python'), os.path.join(subs, sub, 'firmware', 'python')):
            if os.path.isdir(candidate):
                found.append(candidate)
    return found


def resolve_layers(layers, firmware: Optional[os.PathLike], pmap: Optional[platform.PlatformMap],
                   defaults_dir: Optional[os.PathLike] = None) -> List[str]:
    """The register layers a server loads, as rogue's loader will take them.

    The package's own defaults come first, where the platform declares them
    (``pmap.package_defaults``), found in the package and nowhere else -- so
    only where the package is to hand; the names given follow, in order, so
    each is an override of what the package ships. A relative name is looked for in two places and nowhere else --
    never the working directory: the firmware package's configuration
    directory (``pmap.package_config_dir``, inside a release archive or under
    a checkout's ``firmware/``), which is where the files a firmware ships are,
    and the deployment's defaults directory (``defaults_dir``), which is where
    a site keeps the files it chooses from, the population defaults among them.
    Found in one of them, that is the file; in both, the name is refused
    rather than guessed; in neither, the error names both places. An absolute
    name is used as given, and so is a relative one when there is nowhere to
    look.

    Parameters
    ----------
    layers : sequence of path-like
        The names as given on the command line or by the platform.
    firmware : path-like or None
        The firmware package, or None when it is not to hand.
    pmap : PlatformMap or None
        The platform, for its package's configuration directory and the
        defaults it declares; None not to look in the package.
    defaults_dir : path-like or None
        The deployment's defaults directory; None not to look there.

    Raises
    ------
    FileNotFoundError
        If a relative name is in neither place, or the package lacks the
        defaults its platform declares.
    ConfigError
        If a relative name is in both.
    """
    archive = checkout = None
    config_dir = pmap.package_config_dir if pmap is not None else None
    if firmware is not None and config_dir is not None:
        if zipfile.is_zipfile(firmware):
            archive = os.fspath(firmware)
        elif checkout_python_dirs(firmware):
            checkout = os.path.join(os.fspath(firmware), 'firmware')
    defaults_dir = os.fspath(defaults_dir) if defaults_dir is not None else None

    def in_package(layer):
        if archive is not None:
            member = f"{config_dir}/{layer}"
            with zipfile.ZipFile(archive) as zf:
                return f"{archive}/{member}" if member in zf.namelist() else None
        if checkout is not None:
            path = os.path.join(checkout, config_dir, layer)
            return path if os.path.isfile(path) else None
        return None

    def in_deployment(layer):
        if defaults_dir is None:
            return None
        path = os.path.join(defaults_dir, layer)
        return path if os.path.isfile(path) else None

    places = [p for p in (archive and f"{archive}/{config_dir}", checkout and os.path.join(checkout, config_dir),
                          defaults_dir) if p]
    resolved = []
    if pmap is not None and pmap.package_defaults is not None and (archive or checkout):
        # The package's defaults exist only where the package is to hand; with
        # none there is no base, and a server with no layer at all reaches the
        # procedure's refusal rather than a file of that name beside it.
        shipped = in_package(pmap.package_defaults)
        if shipped is None:
            raise FileNotFoundError(f"no {pmap.package_defaults} in {places[0]}, which the {pmap.name} package ships")
        resolved.append(shipped)
    for layer in layers:
        layer = os.fspath(layer)
        if os.path.isabs(layer) or not places:
            # As given: rogue's loader takes a path into an archive as well as
            # a file, and reports what it cannot open.
            resolved.append(layer)
            continue
        found = [p for p in (in_package(layer), in_deployment(layer)) if p]
        if len(found) > 1:
            raise ConfigError(f"{layer} is in both {found[0]} and {found[1]}; name one with its path")
        if not found:
            raise FileNotFoundError(f"no {layer} in {' or '.join(places)}")
        resolved.append(found[0])
    return resolved


@dataclass(frozen=True)
class FirmwareVersion:
    """What the firmware's version block says about the image the FPGA is running.

    Attributes
    ----------
    fpga_version : int
        The image's version number, ``0xMMmmpppp`` (major, minor, patch).
    git_hash : str
        The commit the image was built from, 40 hex digits.
    build_stamp : str
        The image name and build details, as the firmware reports them.
    """
    fpga_version: int
    git_hash: str
    build_stamp: str

    @property
    def short_hash(self) -> str:
        """The first seven hex digits of the commit, as the release file names carry them."""
        return self.git_hash[:7]

    def record(self) -> dict:
        """The version as a plain record, for the inventory a server publishes."""
        return {'fpga_version': self.fpga_version, 'git_hash': self.git_hash, 'build_stamp': self.build_stamp}

    def __str__(self) -> str:
        return f"0x{self.fpga_version:08x} {self.short_hash} ({self.build_stamp})"


class _Probe(pyrogue.Root):
    """The version block alone, read before the real tree exists.

    The link the registers travel over opens when the tree holding it starts.
    The link belongs to the real tree, which does not exist yet, and a node
    cannot be in two trees, so the probe opens and closes it directly around
    its one read; the real tree opens it again when it starts.

    Every platform keeps its version block at the same address, so one block
    serves them all: the platform is named by the stamp's text, not by where
    the block sits.
    """

    def __init__(self, srp, link_nodes=(), offset=0):
        # The version block is the firmware package's, so it imports once the
        # package is on the path -- which is when a probe is built.
        import surf.axi
        pyrogue.Root.__init__(self, name=ROOT_NAME, initRead=True, pollEn=False, timeout=5.0)
        self._link_nodes = list(link_nodes)
        # At the start of the FPGA's register window, wherever the transport
        # puts that window.
        self.add(surf.axi.AxiVersion(offset=offset, memBase=srp))

    def start(self):
        started = []
        try:
            for node in self._link_nodes:
                # Recorded before it starts: a link interrupted while it waits
                # for the far end to answer has its engine running already.
                started.append(node)
                node._start()
            pyrogue.Root.start(self)
        except BaseException:
            # A link that would not open, a tree that would not start, or an
            # interrupt; the context manager's exit will not run for a start
            # that raised, so what was started is stopped here -- rogue's
            # threads are not daemons and would hold the process open. (A
            # read that finds no hardware does not raise: rogue logs it, and
            # the blank stamp is refused by identify() instead.)
            if self.running:
                pyrogue.Root.stop(self)
            for node in reversed(started):
                node._stop()
            raise

    def stop(self):
        pyrogue.Root.stop(self)
        for node in self._link_nodes:
            node._stop()

    def version(self) -> FirmwareVersion:
        """The version block's three facts, read now; a blank stamp is no hardware, and refused."""
        block = self.AxiVersion
        stamp = block.BuildStamp.get()
        if not stamp:
            raise ConnectError("the firmware's version block read blank -- is the FPGA reachable over this link?")
        return FirmwareVersion(int(block.FpgaVersion.get()), f"{int(block.GitHash.get()):040x}", str(stamp))


def read_version(srp: Any, link_nodes: Any = (), offset: int = 0, where: str = 'this transport') -> FirmwareVersion:
    """Read the version block over a register path, before any tree exists over it.

    Parameters
    ----------
    srp : rogue memory master
        The register path.
    link_nodes : sequence of pyrogue nodes
        What the path needs started before it answers; started and stopped
        around the one read.
    offset : int
        Where the FPGA's register window starts on the path.
    where : str
        How to name the path in the error.

    Raises
    ------
    ConnectError
        If the block could not be read, or read blank.
    """
    try:
        with _Probe(srp, link_nodes, offset) as probe:
            return probe.version()
    except ConnectError:
        raise
    except Exception as e:
        # The first thing a server reads from the hardware, so an address that
        # is wrong or a carrier that is down surfaces here.
        raise ConnectError(f"could not read the firmware's version block -- is the FPGA "
                           f"reachable over {where}? ({e})") from e


def probe_platform(transport: Any) -> platform.PlatformMap:
    """Read the build stamp over ``transport`` and name the platform it belongs to.

    Parameters
    ----------
    transport : Transport
        Its register path (``srp``), whatever that path needs started before it
        answers (``register_nodes``), and where the FPGA's register window starts
        on it (``register_offset``); the probe starts and stops the nodes around
        its one read.

    Returns
    -------
    PlatformMap

    Raises
    ------
    ConnectError
        If the stamp is empty or names firmware no platform claims.
    """
    version = read_version(transport.srp, transport.register_nodes, transport.register_offset)
    pmap = platform.from_stamp(version.build_stamp)
    _log.info("firmware %s identified as platform %s", pmap.tags, pmap.name)
    return pmap


def bridge_version(address: str, port: int = _soc.REGISTER_BRIDGE_PORT) -> FirmwareVersion:
    """What the FPGA behind a processor system's register bridge is running, over a link of its own.

    For the launcher, which has no transport yet and may be about to reboot
    the board: the TCP client is made for this one read and closed after it.

    Parameters
    ----------
    address : str
        The processor system's address.
    port : int
        The bridge's port for the FPGA's registers.

    Raises
    ------
    ConnectError
        If the bridge does not answer or the block read blank.
    """
    import rogue.interfaces.memory
    from cryodaq.server import _transport
    srp = rogue.interfaces.memory.TcpClient(address, port)
    try:
        return read_version(srp, (), _transport.BRIDGE_REGISTER_OFFSET,
                            where=f"the register bridge at {address}:{port}")
    finally:
        srp._stop()
