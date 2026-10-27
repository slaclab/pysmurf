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
#    providers and connects the data sinks, and returns the tree, not yet
#    started; it runs as a context manager. With no sinks and the core's own providers it is a
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
from typing import Any, Callable, Mapping, Optional, Sequence

from cryodaq import platform
from cryodaq._errors import ConnectError
from cryodaq.server._application_config import ApplicationConfig, EXCLUDED_FROM
from cryodaq.server._application_status import ApplicationStatus
from cryodaq.server._capture import CAPTURE_POINTS, add_capture_receivers
from cryodaq.server._firmware import load_package, probe_platform, resolve_layers
from cryodaq.server._provider import OperationsProvider, attach_operations
from cryodaq.server._root import ReadoutRoot
from cryodaq.server._setup import PlatformSupport, Setup
from cryodaq.server._offload_card import OffloadCard
from cryodaq.server._transport import Transport, emulation, eth, pcie, tcp_bridge
from cryodaq.server.operations import UMUX_OPERATIONS
from cryodaq.server import _cli

__all__ = ['compose', 'CORE_PROVIDERS', 'OperationsProvider', 'attach_operations', 'load_package',
           'PlatformSupport', 'Setup',
           'Transport', 'eth', 'pcie', 'tcp_bridge', 'emulation', 'OffloadCard', 'ReadoutRoot',
           'ApplicationConfig', 'ApplicationStatus', 'EXCLUDED_FROM', 'UMUX_OPERATIONS',
           'add_capture_receivers', 'CAPTURE_POINTS', 'add_arguments', 'configure_logging', 'transport_from',
           'composition_kwargs', 'run']

# The command line a server entry point shares.
add_arguments = _cli.add_arguments
configure_logging = _cli.configure_logging
transport_from = _cli.transport_from
composition_kwargs = _cli.composition_kwargs
run = _cli.run

_log = logging.getLogger(__name__)

# The operations the readout core itself provides.
CORE_PROVIDERS = (UMUX_OPERATIONS,)


def compose(*, transport: Transport, firmware: Optional[os.PathLike] = None,
            layers: Sequence[os.PathLike] = (), providers: Sequence[OperationsProvider] = CORE_PROVIDERS,
            sinks: Sequence[Any] = (), platform_name: Optional[str] = None,
            server_port: int = 0, polling: bool = True, configure: bool = False,
            variable_groups: Optional[Mapping] = None,
            top_level_options: Optional[Mapping[str, Any]] = None,
            capture_points: int = 0, capture_type: str = 'Int16',
            on_start: Optional[Callable[[ReadoutRoot], None]] = None,
            support: Optional[PlatformSupport] = None, application: str = '') -> ReadoutRoot:
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
    providers : sequence of OperationsProvider
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
        Keyword arguments for the firmware's top-level class beyond ``memBase``:
        what the deployment knows and the FPGA cannot report before its tree is
        built, such as which bays to leave out. The platform's builder
        (`PlatformSupport.top_level`) passes them on; one its class does not
        take is refused, by name.
    capture_points, capture_type
        Size and sample type of the debug capture receivers, one per DDR
        stream; 0 points for none.
    on_start : callable, optional
        Called with the root once started, before ``Ready`` rises.
    support : PlatformSupport, optional
        Stands in for the platform's own support code, which is otherwise found
        beside its map; for an emulated tree of a platform whose firmware
        package is not to hand. It builds the top level over the transport as
        the platform's would, so what the transport says about its register
        path -- where the FPGA's window starts -- reaches the tree either way.
    application : str, optional
        The application composing this server, published as
        ``application.name``; the readout core alone leaves it ''.

    Returns
    -------
    ReadoutRoot
        The tree, not started. ``with compose(...) as root:`` starts it on entry
        and stops it on exit; ``root.pmap`` is the platform it was identified as.
        A transport's offload card is configured here, since the firmware is
        probed through it, and released when the root stops -- so a root that
        is composed must be stopped (or released, if never started); the
        transport's own threads (the bridge's TCP clients) end the same way.

    Raises
    ------
    ConnectError
        If the platform cannot be identified, or disagrees with ``platform_name``.
    """
    # Everything from here raises into one handler, so a transport whose threads
    # are already running (the bridge's TCP clients) is released however the
    # composition fails -- a firmware path that does not exist included.
    try:
        load_package(firmware)
        # The offload card's lane is the register path's first link: it is
        # opened here, before the firmware is probed through it, and the root
        # closes it when it stops.
        if transport.card is not None:
            transport.card.open()
        if transport.probes_firmware:
            pmap = probe_platform(transport)
            if platform_name is not None and platform_name != pmap.name:
                raise ConnectError(f"the hardware reports platform {pmap.name!r}, "
                                   f"not {platform_name!r}")
        elif platform_name is None:
            raise ConnectError(f"the {transport.name} transport reads no build stamp; "
                               f"name the platform: one of "
                               f"{', '.join(repr(m.name) for m in platform.MAPS)}")
        else:
            pmap = platform.by_name(platform_name)

        # A relative file name means one in the archive's configuration
        # directory, which is where the platform's package keeps them.
        layers = resolve_layers(layers, firmware, pmap.package_config_dir)
        # The platform's own rogue code builds its top level for this transport
        # and whatever it adds beside it; there is no other way to build one,
        # so a stand-in is a PlatformSupport too.
        if support is None:
            support = PlatformSupport.for_map(pmap)
        fpga = support.top_level(transport, **(top_level_options or {}))

        root = ReadoutRoot(fpga=fpga, transport=transport, pmap=pmap, providers=providers,
                           sinks=sinks, layers=layers, server_port=server_port, polling=polling,
                           configure=configure, variable_groups=variable_groups, on_start=on_start,
                           application=application, devices=support.devices(transport),
                           setup=support.setup())
        if capture_points:
            add_capture_receivers(root, transport, capture_points, capture_type)
    except BaseException:
        # What a root would have released, released here: no root took it.
        for interface in transport.interfaces:
            interface._stop()
        if transport.card is not None:
            transport.card.close()
        raise
    _log.info("composed a %s server on platform %s with %d provider(s) and %d sink(s)",
              transport.name, pmap.name, len(providers), len(sinks))
    return root


