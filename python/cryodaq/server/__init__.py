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
import atexit
import contextlib
import importlib.resources
import logging
import os
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from cryodaq import platform
from cryodaq._errors import ConnectError
from cryodaq.server._application_config import ApplicationConfig, EXCLUDED_FROM
from cryodaq.server._application_status import ApplicationStatus
from cryodaq.server._capture import CAPTURE_POINTS, add_capture_receivers
from cryodaq.server._firmware import load_package, probe_platform, resolve_layers, top_level
from cryodaq.server._provider import OperationsProvider, attach_operations
from cryodaq.server._root import ReadoutRoot
from cryodaq.server._offload_card import OffloadCard
from cryodaq.server._transport import Transport, emulation, eth, pcie, tcp_bridge
from cryodaq.server.operations import UMUX_OPERATIONS
from cryodaq.server import _cli

__all__ = ['compose', 'CORE_PROVIDERS', 'OperationsProvider', 'attach_operations', 'load_package',
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
            top_level_class: Any = None, application: str = '') -> ReadoutRoot:
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
        built, such as which bays to leave out.
    capture_points, capture_type
        Size and sample type of the debug capture receivers, one per DDR
        stream; 0 points for none.
    on_start : callable, optional
        Called with the root once started, before ``Ready`` rises.
    top_level_class : class, optional
        Build the tree with this class instead of the one the platform names;
        for an emulated tree of a platform whose own package is not to hand.
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
        is composed must be stopped.

    Raises
    ------
    ConnectError
        If the platform cannot be identified, or disagrees with ``platform_name``.
    """
    load_package(firmware)

    # The offload card's lane is the register path's first link: it is opened
    # here, before the firmware is probed through it, and the root closes it
    # when it stops.
    if transport.card is not None:
        transport.card.open()
    try:
        if transport.probes_firmware:
            pmap = probe_platform(transport.srp, transport.register_nodes,
                                  offset=transport.register_offset)
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
        # The FPGA's register window starts where the transport says; a link
        # to the FPGA itself starts at 0, which its classes take for granted.
        options = dict(top_level_options or {})
        if transport.register_offset:
            options['offset'] = transport.register_offset
        fpga = _build_top_level(top_level_class or top_level(pmap), transport.srp, options)
        extras = _platform_devices(pmap, transport)

        root = ReadoutRoot(fpga=fpga, transport=transport, pmap=pmap, providers=providers,
                           sinks=sinks, layers=layers, server_port=server_port, polling=polling,
                           configure=configure, variable_groups=variable_groups, on_start=on_start,
                           application=application, **extras)
        if capture_points:
            add_capture_receivers(root, transport, capture_points, capture_type)
    except BaseException:
        if transport.card is not None:
            transport.card.close()
        raise
    _log.info("composed a %s server on platform %s with %d provider(s) and %d sink(s)",
              transport.name, pmap.name, len(providers), len(sinks))
    return root


def _build_top_level(cls, srp, options):
    """Instantiate the firmware top level with the options it takes.

    The options are what a deployment knows (which bays to leave out, a power
    monitor to enable) and are the AMC carrier's: a release that predates one,
    or another carrier's class, does not take it. A class's signature does not
    say -- the RFMC's forwards ``**kwargs`` to its base -- so the class is
    asked: an option it refuses by name is dropped, with a warning, and the
    build retried without it, until it builds or refuses for another reason.
    One command line thereby serves every platform and release.
    """
    options = dict(options)
    while True:
        try:
            return cls(memBase=srp, **options)
        except TypeError as e:
            unknown = [k for k in options if f"'{k}'" in str(e)]
            if not unknown:
                raise
            _log.warning("%s does not take %s; building without it", cls.__name__, ', '.join(unknown))
            for k in unknown:
                del options[k]


def _platform_devices(pmap: platform.PlatformMap, transport: Transport) -> Dict[str, Any]:
    """What a platform adds to the root beside the FPGA's tree, from its package.

    A platform whose map places an RF data converter on the root (the RFMC
    carrier) drives it over a register path of its own, the transport's
    ``srp_rfdc``, and its configuration procedure programs the carrier's
    clock chips from a register file the firmware package ships beside its
    code. Both are found here, where the package is importable. The file is
    read by the clock chip's driver as a plain file, so one inside a release
    archive is extracted beside the server for as long as it runs.

    Parameters
    ----------
    pmap : PlatformMap
        The platform identified; its ``rfdc_path`` says whether it has one.
    transport : Transport
        Where the converter's register path comes from.

    Returns
    -------
    dict
        Keyword arguments for `ReadoutRoot`: ``devices`` to add, and
        ``setup_options`` for the platform's procedure. Empty for a platform
        that adds nothing.

    Raises
    ------
    ConnectError
        If the platform has a converter and the transport no path to it, or
        the package ships no clock file.
    """
    if not pmap.rfdc_path:
        return {}
    if transport.srp_rfdc is None:
        raise ConnectError(f"platform {pmap.name} drives an RF data converter over a second "
                           f"register path, which the {transport.name} transport does not have")
    import axi_soc_ultra_plus_core.rfsoc_utility as rfsoc_utility
    rfdc = rfsoc_utility.Rfdc(name=pmap.rfdc_path.rsplit('.', 1)[1], memBase=transport.srp_rfdc)
    package = pmap.top_level.partition(':')[0]
    resource = importlib.resources.files(package).joinpath(platform.RFMC_LMK_CONFIG)
    if not resource.is_file():
        raise ConnectError(f"the {package} package ships no {platform.RFMC_LMK_CONFIG}, which the "
                           f"{pmap.name} configuration procedure programs the clock chips from")
    lmk_config = _EXTRACTED.enter_context(importlib.resources.as_file(resource))
    return dict(devices=[rfdc], setup_options=dict(lmk_config=str(lmk_config)))


# Files taken out of a release archive for a driver that reads plain files; kept
# for the life of the process, since a server composes once and runs.
_EXTRACTED = contextlib.ExitStack()
atexit.register(_EXTRACTED.close)
