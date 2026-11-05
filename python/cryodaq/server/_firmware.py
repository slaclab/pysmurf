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
#    The probe is a tree holding only the version block, at the address every
#    platform here places it; it is started, read and stopped before the real
#    tree is built over the same link. So the platform comes from the hardware in
#    hand, as the tree's identification does once a client connects, and nothing
#    is told which hardware it is driving.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import importlib.resources
import logging
import os
import sys
import zipfile
from typing import Any, List, Optional

import pyrogue

from cryodaq import platform
from cryodaq._errors import ConnectError

__all__ = ['load_package', 'checkout_python_dirs', 'resolve_layers', 'package_defaults_layer',
           'probe_platform', 'ROOT_NAME']

_log = logging.getLogger(__name__)

# What the root of every tree here is called; the register paths a client
# resolves start with it.
ROOT_NAME = platform.ROOT_PATH


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


def resolve_layers(layers, firmware: Optional[os.PathLike], config_dir: str) -> List[str]:
    """Configuration files as rogue's loader will take them.

    With a firmware package, a relative name means a file in the package's
    configuration directory (``config_dir``, the platform's) -- inside a
    release archive, or under a checkout's ``firmware/`` -- and nothing else,
    so a file of the same name in the working directory is never taken for
    it. That is how a deployment names the defaults a package ships. An
    absolute name is used as given, and so is any name when there is no
    package to look inside.

    Raises
    ------
    FileNotFoundError
        If a relative name is not in the package.
    """
    resolved = []
    archive = checkout = None
    if firmware is not None:
        if zipfile.is_zipfile(firmware):
            archive = os.fspath(firmware)
        elif checkout_python_dirs(firmware):
            checkout = os.path.join(os.fspath(firmware), 'firmware')
    for layer in layers:
        layer = os.fspath(layer)
        if os.path.isabs(layer) or (archive is None and checkout is None):
            # As given: rogue's loader takes a path into an archive as well as
            # a file, and reports what it cannot open.
            resolved.append(layer)
        elif archive is not None:
            member = f"{config_dir}/{layer}"
            with zipfile.ZipFile(archive) as zf:
                if member not in zf.namelist():
                    raise FileNotFoundError(f"{archive} holds no {member}")
            resolved.append(f"{archive}/{member}")
        else:
            path = os.path.join(checkout, config_dir, layer)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"{checkout} holds no {config_dir}/{layer}")
            resolved.append(path)
    return resolved


def package_defaults_layer(pmap: platform.PlatformMap, firmware: Optional[os.PathLike]) -> str:
    """The register defaults the platform's package ships, as rogue's loader will take them.

    Where the package was given (an archive or a checkout) the file is looked
    up in it as any relative layer is; where it is already importable, through
    the package itself -- the archive's ``python/`` is the import root, so the
    package is the first name under it in the platform's configuration
    directory.

    Raises
    ------
    ConnectError
        If the platform declares no defaults, or the package does not ship them.
    """
    name = pmap.package_defaults
    if name is None:
        raise ConnectError(f"platform {pmap.name!r} declares no package defaults; give a layer")
    try:
        if firmware is not None and (zipfile.is_zipfile(firmware) or checkout_python_dirs(firmware)):
            return resolve_layers([name], firmware, pmap.package_config_dir)[0]
        top, package, *subdir = pmap.package_config_dir.split('/')
        if top != 'python' or not package:
            raise FileNotFoundError(f"{pmap.package_config_dir!r} does not name a package under python/")
        resource = importlib.resources.files(package).joinpath('/'.join(subdir), name)
        if not resource.is_file():
            raise FileNotFoundError(f"the {package} package ships no {'/'.join(subdir)}/{name}")
        return str(resource)
    except (FileNotFoundError, ModuleNotFoundError) as e:
        raise ConnectError(f"the {pmap.name} package's register defaults were not found: {e}") from e


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
    try:
        with _Probe(transport.srp, transport.register_nodes, transport.register_offset) as probe:
            pmap = platform.from_stamp(probe.AxiVersion.BuildStamp.get())
    except ConnectError:
        raise
    except Exception as e:
        # The first thing a server reads from the hardware, so an address that
        # is wrong or a carrier that is down surfaces here.
        raise ConnectError(f"could not read the firmware's build stamp -- is the FPGA "
                           f"reachable over this transport? ({e})") from e
    _log.info("firmware %s identified as platform %s", pmap.tags, pmap.name)
    return pmap
