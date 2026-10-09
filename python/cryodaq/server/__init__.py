#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Composition
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2026-10-08
#-----------------------------------------------------------------------------
# Description:
#    How a readout server is put together, as one call. `compose()` takes a
#    transport, a firmware package and the register configuration layers, learns
#    the platform from the hardware, builds the tree, attaches the operation
#    providers and connects the data sinks, and returns a composition that runs
#    as a context manager. With no sinks and the core's own providers it is a
#    register server that can bring a carrier up from a plain Python environment;
#    an application passes its providers and its data processing to the same
#    call and gets the full server.
#
#    This subpackage is the server side of cryodaq and the only part of it that
#    imports rogue at module level. The package does not import it; a server
#    entry point does.
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
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from cryodaq import platform
from cryodaq._errors import ConnectError
from cryodaq.server._application_config import ApplicationConfig, EXCLUDED_FROM
from cryodaq.server._application_status import ApplicationStatus
from cryodaq.server._firmware import load_package, probe_platform, resolve_layers, top_level
from cryodaq.server._provider import Provider, attach_providers
from cryodaq.server._root import ReadoutRoot
from cryodaq.server._transport import Transport, emulation, eth, pcie
from cryodaq.server._umux_ops import UMUX_OPERATIONS
from cryodaq.server import _cli

__all__ = ['compose', 'Composition', 'CORE_PROVIDERS', 'Provider', 'attach_providers',
           'Transport', 'eth', 'pcie', 'emulation', 'ReadoutRoot',
           'ApplicationConfig', 'ApplicationStatus', 'EXCLUDED_FROM', 'UMUX_OPERATIONS',
           'add_arguments', 'configure_logging', 'transport_from', 'composition_kwargs']

# The command line a server entry point shares.
add_arguments = _cli.add_arguments
configure_logging = _cli.configure_logging
transport_from = _cli.transport_from
composition_kwargs = _cli.composition_kwargs

_log = logging.getLogger(__name__)

# The operations the readout core itself provides.
CORE_PROVIDERS = (UMUX_OPERATIONS,)


@dataclass
class Composition:
    """A built server, not yet started.

    ``with compose(...) as root:`` starts the tree on entry and stops it on exit.
    ``root`` is the tree; ``pmap`` the platform it was identified as.
    """

    root: ReadoutRoot
    pmap: platform.PlatformMap

    def __enter__(self):
        self.root.start()
        return self.root

    def __exit__(self, *exc):
        self.root.stop()
        return False


def compose(*, transport: Transport, firmware: Optional[os.PathLike] = None,
            layers: Sequence[os.PathLike] = (), providers: Sequence[Provider] = CORE_PROVIDERS,
            sinks: Sequence[Any] = (), platform_name: Optional[str] = None,
            server_port: int = 0, polling: bool = True, configure: bool = False,
            variable_groups: Optional[Mapping] = None,
            top_level_options: Optional[Mapping[str, Any]] = None,
            on_start: Optional[Callable[[ReadoutRoot], None]] = None,
            top_level_class: Any = None) -> Composition:
    """Build a readout server.

    Parameters
    ----------
    transport : Transport
        From `eth`, `pcie` or `emulation`.
    firmware : path-like, optional
        The firmware's Python package: a release archive or a checkout. None if
        it is already importable.
    layers : sequence of path-like
        Register configuration files, applied in order by rogue's own loader:
        the population's defaults first, then any site or slot overrides. A
        relative name is looked up in the release archive's configuration
        directory.
    providers : sequence of Provider
        What attaches operations to the tree; the core's own by default. An
        application appends its providers.
    sinks : sequence
        The data processing: rogue stream slaves for the streaming interface,
        or callables taking the root and returning one. None for a register-only
        server.
    platform_name : str, optional
        Required when the transport cannot read a build stamp (emulation);
        otherwise the platform is read from the hardware and a name given here
        must agree with it.
    server_port : int
        ZMQ port, 0 for any free one.
    polling, configure, variable_groups
        As `ReadoutRoot`.
    top_level_options : mapping, optional
        Keyword arguments for the firmware's top-level class beyond ``memBase``
        -- which bays to leave out, firmware-line flags.
    on_start : callable, optional
        Called with the root once started, before ``Ready`` rises.
    top_level_class : class, optional
        Build the tree with this class instead of the one the platform names;
        for an emulated tree of a platform whose own package is not to hand.

    Returns
    -------
    Composition

    Raises
    ------
    ConnectError
        If the platform cannot be identified, or disagrees with ``platform_name``.
    """
    load_package(firmware)
    layers = resolve_layers(layers, firmware)

    if transport.probes_firmware:
        pmap = probe_platform(transport.srp, transport.register_nodes)
        if platform_name is not None and platform_name != pmap.name:
            raise ConnectError(f"the hardware reports platform {pmap.name!r}, "
                               f"not {platform_name!r}")
    elif platform_name is None:
        raise ConnectError(f"the {transport.name} transport reads no build stamp; "
                           f"name the platform: one of "
                           f"{', '.join(repr(m.name) for m in platform.MAPS)}")
    else:
        pmap = platform.by_name(platform_name)

    options = dict(top_level_options or {})
    fpga = _build_top_level(top_level_class or top_level(pmap), transport.srp, options)
    enabled_bays = [bay for bay, disabled in enumerate(
        (options.get('disableBay0', False), options.get('disableBay1', False))) if not disabled]

    root = ReadoutRoot(fpga=fpga, transport=transport, pmap=pmap, providers=providers,
                       sinks=sinks, layers=layers, server_port=server_port, polling=polling,
                       configure=configure, variable_groups=variable_groups,
                       enabled_bays=enabled_bays, on_start=on_start)
    _log.info("composed a %s server on platform %s with %d provider(s) and %d sink(s)",
              transport.name, pmap.name, len(providers), len(sinks))
    return Composition(root=root, pmap=pmap)


def _build_top_level(cls, srp, options):
    """Instantiate the firmware top level, dropping options a release predates."""
    try:
        return cls(memBase=srp, **options)
    except TypeError as e:
        unknown = [k for k in options if k in str(e)]
        if not unknown:
            raise
        _log.warning("this firmware package does not take %s; building without it",
                     ', '.join(unknown))
        return cls(memBase=srp, **{k: v for k, v in options.items() if k not in unknown})
