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
#    The probe is a two-device tree holding only the version block, which every
#    platform here places at the same address; it is started, read and stopped
#    before the real tree is built over the same link. So the platform comes from
#    the hardware in hand, as the tree's identification does once a client
#    connects, and nothing is told which hardware it is driving.
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
import zipfile
from typing import Any, List, Optional

import pyrogue

from cryodaq import platform
from cryodaq._errors import ConnectError

__all__ = ['load_package', 'resolve_layers', 'probe_platform', 'top_level', 'ROOT_NAME']

_log = logging.getLogger(__name__)

# What the root of every tree here is called; the register paths a client
# resolves start with it.
ROOT_NAME = platform.VERSION_BLOCK_PATH.split('.')[0]
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
        pyrogue.addLibraryPath(f"{path}/python")
    elif os.path.isdir(path):
        pyrogue.addLibraryPath(path)
    else:
        raise FileNotFoundError(f"no firmware package at {path}")


def resolve_layers(layers, firmware: Optional[os.PathLike]) -> List[str]:
    """Register configuration files as rogue's loader will take them.

    A relative name is looked up inside the release archive's configuration
    directory, which is how a deployment names the defaults a package ships;
    anything else is used as given.

    Raises
    ------
    FileNotFoundError
        If a relative name is given with no archive, or is not in it.
    """
    resolved = []
    archive = os.fspath(firmware) if firmware is not None and zipfile.is_zipfile(firmware) else None
    for layer in layers:
        layer = os.fspath(layer)
        if os.path.isabs(layer) or os.path.exists(layer):
            resolved.append(layer)
            continue
        if archive is None:
            raise FileNotFoundError(f"no register configuration file {layer!r}, and no "
                                    f"archive to look inside")
        member = f"{PACKAGE_CONFIG_DIR}/{layer}"
        with zipfile.ZipFile(archive) as zf:
            if member not in zf.namelist():
                raise FileNotFoundError(f"{archive} holds no {member}")
        resolved.append(f"{archive}/{member}")
    return resolved


class _Probe(pyrogue.Root):
    """The version block alone, at the path the full tree gives it.

    The link the registers travel over opens when the tree holding it starts.
    The link belongs to the real tree, which does not exist yet, and a node
    cannot be in two trees, so the probe opens and closes it directly around
    its one read; the real tree opens it again when it starts.
    """

    def __init__(self, srp, link_nodes=()):
        # The version block is the firmware package's, so it imports once the
        # package is on the path -- which is when a probe is built.
        import surf.axi
        pyrogue.Root.__init__(self, name=ROOT_NAME, initRead=True, pollEn=False, timeout=5.0)
        self._link_nodes = list(link_nodes)
        # Empty devices down to the version block, so the stamp has the path the
        # map says it has.
        *segments, _ = platform.VERSION_BLOCK_PATH.split('.')[1:]
        parent = self
        for segment in segments:
            device = pyrogue.Device(name=segment)
            parent.add(device)
            parent = device
        parent.add(surf.axi.AxiVersion(offset=0, memBase=srp))

    def start(self):
        for node in self._link_nodes:
            node._start()
        try:
            pyrogue.Root.start(self)
        except Exception:
            # The initial read failed -- no hardware answered -- and the
            # context manager's exit will not run for a start that raised.
            self.stop()
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
        raise ConnectError(f"could not read the firmware's build stamp: {e}") from e
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
