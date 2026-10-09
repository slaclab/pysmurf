#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Operation Providers
#-----------------------------------------------------------------------------
# File       : _provider.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    How operations reach the server's tree. A provider declares a name, the
#    semantic name of the device it attaches to (its anchor, a map pattern such
#    as `band[*].ops`), the node names it adds there, and a function that adds
#    them to one such device. The composition expands the anchor through the
#    platform map, so a provider never spells a register path, and attaches
#    every provider after the FPGA is in the tree and before the root starts.
#
#    An attach is atomic per provider: every (device, node) pair is checked
#    against pyrogue's own collision predicate first, and one that is already
#    present refuses the whole provider before anything is added. That is how a
#    firmware package that still carries its own copy of an operation is
#    caught -- by name, at startup, rather than halfway through an add.
#
#    The readout core supplies its providers; an application built on it
#    supplies its own through the same call.
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
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

from cryodaq import platform

__all__ = ['Provider', 'attach_providers', 'exists']

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Provider:
    """What one source of operations declares to the composition.

    Parameters
    ----------
    name : str
        Who the operations belong to, e.g. ``cryodaq.umux``; appears in the
        refusal message and the log.
    anchor : str
        The map pattern of the device each copy attaches to, e.g.
        ``band[*].ops``. Expanded against the tree in hand, so a platform with
        fewer bands gets fewer copies and one with none gets none.
    nodes : tuple of str
        Every node name ``attach`` adds under an anchor: the collision surface.
        A node added and left out of this tuple is invisible to the pre-attach
        check and would fail inside pyrogue's ``add()`` instead, halfway through.
    attach : callable
        ``attach(device)`` adds the nodes to one anchor device. Called once per
        anchor, after every anchor has been checked.
    """

    name: str
    anchor: str
    nodes: Tuple[str, ...]
    attach: Callable[[Any], None]


def exists(device: Any, name: str) -> bool:
    """Would adding a node called ``name`` to ``device`` collide?

    This is deliberately the exact predicate pyrogue's ``Device.add()`` uses
    (``pyrogue/_Node.py``, the 'Name collision' check), so that "would this
    collide?" and "did this collide?" can never disagree.
    """
    return name in device.__dir__() or name in getattr(device, '_anodes', {})


def attach_providers(root: Any, pmap: platform.PlatformMap,
                     providers: Sequence[Provider]) -> Dict[str, List[str]]:
    """Attach every provider's operations to the tree, in the order given.

    Must run after the FPGA device is in the tree and before ``Root.start()``,
    which seals it. For each provider the anchor pattern is expanded against the
    tree through the platform map; every (anchor, node) pair is checked first,
    and any one already present refuses that provider before anything is added.

    Parameters
    ----------
    root : pyrogue.Root
        The tree under construction; ``getNode(path)`` is all that is asked of it.
    pmap : PlatformMap
        The map the anchors are resolved through.
    providers : sequence of Provider
        Attached in this order.

    Returns
    -------
    dict
        Provider name to the list of anchor names it was attached to.

    Raises
    ------
    RuntimeError
        If an anchor already carries one of the provider's nodes -- the loaded
        firmware package still defines the operation itself -- or if the map
        names an anchor the tree does not have.
    """
    def has(path):
        return root.getNode(path) is not None

    attached = {}
    for provider in providers:
        anchors = []
        for name in platform.expand(pmap, has, provider.anchor):
            device = root.getNode(pmap.path(name))
            if device is None:
                raise RuntimeError(f"{provider.name}: the map names {name} but the "
                                   f"tree has no {pmap.path(name)}")
            anchors.append((name, device))
        present = [f"{name}.{node}" for name, device in anchors
                   for node in provider.nodes if exists(device, node)]
        if present:
            raise RuntimeError(
                f"{provider.name}: {len(present)} of its {len(provider.nodes)} node(s) "
                f"already exist on {len(anchors)} anchor(s), so the loaded firmware "
                f"package still provides its own copy and a second one must not be "
                f"added on top.\n  already present: {', '.join(present)}\n"
                f"Update the firmware package to one from which these operations "
                f"have been removed.")
        for name, device in anchors:
            provider.attach(device)
            device._log.info("Attached %d %s operations to %s.",
                             len(provider.nodes), provider.name, name)
        if not anchors:
            _log.warning("%s: the tree has no %s, so nothing was attached.",
                         provider.name, provider.anchor)
        attached[provider.name] = [name for name, _ in anchors]
    return attached
