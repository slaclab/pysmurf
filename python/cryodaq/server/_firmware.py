#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Firmware Package and Platform Probe
#-----------------------------------------------------------------------------
# File       : _firmware.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    Three steps a server takes before it has a register tree: put the firmware's
#    Python package on the path (a release archive or a checkout), read the build
#    stamp off the hardware to learn which platform this is, and import the class
#    that builds that platform's tree.
#
#    The probe is a tree holding only the version block -- at the address every
#    platform here places it, under each path a platform keeps it at, since the
#    carrier core is named for its board; it is started, read and stopped before
#    the real tree is built over the same link. So the platform comes from the
#    hardware in hand, as the tree's identification does once a client connects,
#    and nothing is told which hardware it is driving.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import importlib
import logging
import os
import sys
import zipfile
from typing import Any, List, Optional

import pyrogue

from cryodaq import platform
from cryodaq._errors import ConnectError

__all__ = ['load_package', 'resolve_layers', 'probe_platform', 'top_level', 'ROOT_NAME']

_log = logging.getLogger(__name__)

# What the root of every tree here is called; the register paths a client
# resolves start with it.
ROOT_NAME = platform.VERSION_BLOCK_PATHS[0].split('.')[0]
# Where a release archive keeps its register configuration files.
PACKAGE_CONFIG_DIR = 'python/CryoDet/config'


def load_package(path: Optional[os.PathLike]) -> None:
    """Put a firmware package's Python on the import path.

    Parameters
    ----------
    path : path-like or None
        A release archive (its ``python/`` directory is added) or a directory
        (added as given). None leaves the path alone, for a package already
        importable.

    Raises
    ------
    FileNotFoundError
        If the path names nothing.
    """
    if path is None:
        return
    path = os.fspath(path)
    if zipfile.is_zipfile(path):
        entry = f"{path}/python"
    elif os.path.isdir(path):
        entry = path
    else:
        raise FileNotFoundError(f"no firmware package at {path}")
    # Idempotent: an entry point that needs the package before compose() does
    # (to open an offload card) loads it first, and compose() loads it again.
    if entry not in sys.path:
        pyrogue.addLibraryPath(entry)


def resolve_layers(layers, firmware: Optional[os.PathLike]) -> List[str]:
    """Register configuration files as rogue's loader will take them.

    With a release archive, a relative name means a file in the archive's
    configuration directory -- how a deployment names the defaults a package
    ships -- and nothing else, so a file of the same name in the working
    directory is never taken for it. An absolute name is used as given, and
    so is any name when there is no archive to look inside.

    Raises
    ------
    FileNotFoundError
        If a relative name is not in the archive.
    """
    resolved = []
    archive = os.fspath(firmware) if firmware is not None and zipfile.is_zipfile(firmware) else None
    for layer in layers:
        layer = os.fspath(layer)
        if os.path.isabs(layer) or archive is None:
            # As given: rogue's loader takes a path into an archive as well as
            # a file, and reports what it cannot open.
            resolved.append(layer)
            continue
        member = f"{PACKAGE_CONFIG_DIR}/{layer}"
        with zipfile.ZipFile(archive) as zf:
            if member not in zf.namelist():
                raise FileNotFoundError(f"{archive} holds no {member}")
        resolved.append(f"{archive}/{member}")
    return resolved


class _Probe(pyrogue.Root):
    """The version block alone, at each path a platform's full tree gives it.

    The link the registers travel over opens when the tree holding it starts.
    The link belongs to the real tree, which does not exist yet, and a node
    cannot be in two trees, so the probe opens and closes it directly around
    its one read; the real tree opens it again when it starts.

    Every platform keeps its version block at the same address, inside a
    carrier core named for its board. One block is read, and the tree answers
    at every path it is asked for: two devices over one address would be an
    overlap rogue refuses, so the paths other than the first are link nodes
    reading the same block.
    """

    def __init__(self, srp, link_nodes=()):
        # The version block is the firmware package's, so it imports once the
        # package is on the path -- which is when a probe is built.
        import surf.axi
        pyrogue.Root.__init__(self, name=ROOT_NAME, initRead=True, pollEn=False, timeout=5.0)
        self._link_nodes = list(link_nodes)
        first, *others = platform.VERSION_BLOCK_PATHS
        block = surf.axi.AxiVersion(offset=0, memBase=srp)
        self._device_at(first).add(block)
        for path in others:
            alias = pyrogue.Device(name=path.rsplit('.', 1)[1])
            self._device_at(path).add(alias)
            # Only the stamp: it is the one register identification reads, and
            # a device of its own already has an `enable`.
            alias.add(pyrogue.LinkVariable(name='BuildStamp', mode='RO',
                                           linkedGet=block.BuildStamp.get,
                                           dependencies=[block.BuildStamp]))

    def _device_at(self, path):
        """Empty devices down to the parent of ``path``, so a stamp has the path its map says."""
        *segments, _ = path.split('.')[1:]
        parent = self
        for segment in segments:
            device = parent.node(segment)
            if device is None:
                device = pyrogue.Device(name=segment)
                parent.add(device)
            parent = device
        return parent

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


def probe_platform(srp: Any, link_nodes: Any = ()) -> platform.PlatformMap:
    """Read the build stamp over ``srp`` and name the platform running it.

    Parameters
    ----------
    srp : rogue memory master
        The register path.
    link_nodes : sequence of pyrogue nodes
        What the register path needs started before it answers (a transport's
        ``register_nodes``); the probe starts and stops them around the read.

    Returns
    -------
    PlatformMap

    Raises
    ------
    ConnectError
        If the stamp is empty or names firmware no platform claims.
    """
    try:
        with _Probe(srp, link_nodes) as probe:
            pmap = platform.identify(probe)
    except ConnectError:
        raise
    except Exception as e:
        # The first thing a server reads from the hardware, so an address that
        # is wrong or a carrier that is down surfaces here.
        raise ConnectError(f"could not read the firmware's build stamp -- is the FPGA "
                           f"reachable over this transport? ({e})") from e
    _log.info("firmware %s identified as platform %s", pmap.tags, pmap.name)
    return pmap


def top_level(pmap: platform.PlatformMap) -> Any:
    """Import the class that builds ``pmap``'s register tree."""
    module_name, _, attribute = pmap.top_level.partition(':')
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ConnectError(f"platform {pmap.name} needs {module_name}, which is not "
                           f"importable: {e}. Load the firmware package first.") from e
    return getattr(module, attribute)
