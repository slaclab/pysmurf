#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Configuration Procedure
#-----------------------------------------------------------------------------
# File       : _setup.py
# Created    : 2026-10-20
#-----------------------------------------------------------------------------
# Description:
#    The procedure that brings a readout system from "server running" to
#    "configured": the register layers loaded and, where the platform has
#    serial converter links, the converters reset before the load and the
#    links checked after it. It takes no argument -- the layers are the
#    server's, given at start -- and runs on the server as a rogue Process a
#    client starts and waits on, so that a client with nothing but a
#    channel-access put can configure a system, and a register-only server
#    runs the same procedure.
#
#    What the procedure leaves behind is the register layers' state, and on a
#    carrier whose clocks and converters are brought up by software (the RFMC)
#    those brought up; no more -- the flux ramp, the triggers and the data
#    stream are the application's to turn on afterwards. Each platform has its
#    own small subclass, in the platform's package beside its register map,
#    naming its steps in order and working on the rogue tree directly.
#
#    `PlatformSupport` is the rest of what a platform contributes to the server's
#    tree: the firmware's top-level device, built explicitly for the transport
#    in hand, any device the server adds beside it, and that procedure. One
#    subclass per platform, found from the platform's map.
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
import time
from typing import Any, Callable, List, Optional, Sequence, Tuple, Type

import pyrogue

from cryodaq import platform

__all__ = ['Setup', 'PlatformSupport', 'StepError', 'load_layers', 'check_layer_names_this_root',
           'LOAD_TIMEOUT_PER_LAYER_S', 'LOAD_RETRIES']

_log = logging.getLogger(__name__)

# How long one register layer is given to load before a try is judged stuck. A
# full layer takes about a minute over a crate's link; the bound is well clear
# of that and scales with the number of layers, since each is a full write.
LOAD_TIMEOUT_PER_LAYER_S = 300.0
LOAD_RETRIES = 4


class StepError(RuntimeError):
    """A bring-up step could not complete; the message says which and why."""


def load_layers(root: Any, layers: Sequence[str]) -> None:
    """Load the register layers, in order, in one staged write; retry a load that fails.

    Each try is judged by the message rogue's loader leaves, cleared before it
    starts, so no try is credited with an earlier load's result. The bound
    scales with the number of layers; a load that outruns it is given as long
    again to end -- and credited if it ends well -- before a retry, and one
    that still has not ended is left alone and the step gives up: nothing can
    stop rogue's loader from outside, and starting a second over it would
    interleave two full writes.
    """
    if not layers:
        raise StepError('no register layers were given')
    for layer in layers:
        check_layer_names_this_root(root, layer)
    loader = root.LoadConfigProcess
    # The update thread can interfere with some steps of initialisation.
    loader.UpdatePeriod.set(0.0)
    files = ','.join(layers)
    timeout = LOAD_TIMEOUT_PER_LAYER_S * len(layers)

    def ended():
        # Start() returns before the worker thread raises Running, so an empty
        # or 'Running' message is the only sign a load is still due.
        return not loader.Running.value() and loader.Message.value() not in ('', 'Running')

    def wait_until(deadline):
        while not ended() and time.time() < deadline:
            time.sleep(0.1)
        return ended()

    message = ''
    for i in range(LOAD_RETRIES):
        _log.info("loading register layers %s (try %d)", files, i)
        loader.LoadMode.setDisp('File')
        loader.ConfigFile.set(files)
        loader.Message.setDisp('')
        loader.Start()
        start = time.time()
        if not wait_until(start + timeout):
            _log.warning("register layers try %d: still loading after %.0f s", i, timeout)
            if not wait_until(start + 2 * timeout):
                raise StepError(f"the layer load has not ended after {2 * timeout:.0f} s")
        message = loader.Message.value()
        if message == 'Done':
            _log.info("register layers loaded")
            return
        _log.warning("register layers try %d ended with %r", i, message)
    raise StepError(f"the register layers did not load in {LOAD_RETRIES} tries; "
                    f"the loader's last message: {message!r}")


def check_layer_names_this_root(root: Any, layer: str) -> None:
    """Refuse a layer whose top-level keys are not this root's name.

    Rogue applies a layer by looking each top-level key up as a node of the
    root, and a key that is not the root's own name (or ``root``) is logged as
    ``Entry … not found`` while the load still ends ``Done``. A file written
    for a tree whose root is called otherwise -- a firmware package's own
    defaults, say -- would then be credited as loaded and change nothing. Only
    the keys are read here, with rogue's own reader, so a file it cannot open
    fails the same way it would in the load.

    Parameters
    ----------
    root : pyrogue.Root
        The tree the layer is for; only its name is used.
    layer : str
        The file, as rogue's loader will take it (a path into an archive too).

    Raises
    ------
    StepError
        If the file cannot be read, or a top-level key is not the root's name.
    """
    try:
        keys = list(pyrogue.yamlToData(fName=layer) or {})
    except Exception as e:                                           # noqa: BLE001
        raise StepError(f"{layer}: cannot be read: {e}") from e
    strangers = [k for k in keys if k not in (root.name, 'root')]
    if strangers:
        raise StepError(f"{layer}: written for a root called {strangers[0]!r}, not "
                        f"{root.name!r}; nothing in it would be applied")


class Setup(pyrogue.Process):
    """The configuration procedure, as a process on the tree.

    It takes no parameter: it loads the register layers the server was given
    and leaves the system in the state they describe. Its outcome is the
    application status block's ``SystemConfigured``; ``Message`` names the
    step that failed when that is false. Each platform subclasses it with
    its own ``steps``, and wraps ``run_steps`` where the whole run needs
    something around it. The layers and the status block are the root's, read
    once the process is on the tree.
    """

    def __init__(self, **kwargs):
        pyrogue.Process.__init__(self, name='setDefaults',
                                 description='Configure the system: load the register layers '
                                             'and check the links',
                                 **kwargs)

    @property
    def layers(self) -> Tuple[str, ...]:
        """The register layers the server was started with, in order."""
        return tuple(self.root.layers)

    @property
    def status(self) -> Any:
        """The application status block the outcome is reported in."""
        return self.root.status

    def steps(self) -> Sequence[Tuple[str, Callable[[], None]]]:
        """The platform's steps on the started tree, in order, each named for ``Message``."""
        raise NotImplementedError('a platform subclass names its steps')

    def refuse_unloadable_layers(self) -> None:
        """Refuse, before any step, layers the load could not apply: none, or ones for another root.

        The carrier's converter resets, or the RFMC's clock programming, would
        otherwise disturb the hardware for a load that was never going to
        configure it.
        """
        if not self.layers:
            raise StepError('load layers: no register layers were given')
        for layer in self.layers:
            try:
                check_layer_names_this_root(self.root, layer)
            except StepError as e:
                raise StepError(f"load layers: {e}") from e

    def run(self) -> None:
        """Run the procedure: refuse layers the load could not apply, then the steps.

        The refusal comes before anything is touched, on every platform; what
        a platform does around its steps -- a verdict to report, polling to
        hold off -- goes in its `run_steps`.
        """
        self.refuse_unloadable_layers()
        self.run_steps()

    def run_steps(self) -> None:
        """Run the steps in order; a step that cannot complete raises, naming itself."""
        steps = self.steps()
        self.setTotalSteps(len(steps))
        for name, step in steps:
            try:
                step()
            except Exception as e:
                raise StepError(f"{name}: {e}") from e
            self.incrementSteps()

    def _process(self) -> None:
        status = self.status
        # Every write here is inside the try, so a failure anywhere -- a step,
        # or the bookkeeping around one -- still ends with the system reported
        # unconfigured and no longer configuring. The cause goes to the log and
        # to Message; rogue's own runner overwrites Message with its generic
        # text when an exception escapes, so none is allowed to. Message is
        # published before the in-progress flag drops: a client waits on the
        # flag and then reads the message, so the order is what it sees.
        failed: Optional[str] = 'did not run to the end'
        try:
            status.ConfiguringInProgress.set(True)
            status.SystemConfigured.set(False)
            self.Message.setDisp('Running')
            self.run()
            failed = None
        except Exception as e:                                       # noqa: BLE001
            failed = str(e)
            _log.error("configuration procedure failed: %s", e)
        finally:
            success = failed is None
            self.Message.setDisp('Done' if success else f"Error: {failed}")
            status.SystemConfigured.set(success)
            self.setProgress(1.0)
            status.ConfiguringInProgress.set(False)




class PlatformSupport:
    """A platform's support code: its devices instantiated, its initialisation sequence.

    In the sense of a board support package -- what a server needs to run on
    a given board: the firmware's top-level device, built for the transport in
    hand; any device the server adds beside it, driven over a register path of
    the transport's; and the configuration procedure. One subclass per
    platform, in ``cryodaq.platform.<name>.support`` beside the platform's
    register map, which is where it takes its paths from. A map is data the
    client reads without rogue; this is the server's half, and imports rogue.
    """

    #: The configuration procedure's class; `ReadoutRoot` adds one instance.
    setup: Type[Setup] = Setup

    def top_level(self, transport: Any, **options: Any) -> Any:
        """The firmware's top-level device over ``transport``'s register path.

        ``options`` are what a deployment knows and the FPGA cannot report
        before its tree is built (which bays to leave out); a platform whose
        class takes none ignores them, saying so in the log.
        """
        raise NotImplementedError('a platform builds its own top level')

    def devices(self, transport: Any) -> List[Any]:
        """Devices the server adds beside the FPGA's tree; none by default."""
        return []

    @classmethod
    def for_map(cls, pmap: platform.PlatformMap) -> 'PlatformSupport':
        """The tree contribution of the platform ``pmap`` describes.

        Found by convention beside the map: module ``<map module>.support``, its
        one `PlatformSupport` subclass.
        """
        module = importlib.import_module(f'{pmap.module}.support')
        found = [v for v in vars(module).values()
                 if isinstance(v, type) and issubclass(v, PlatformSupport) and v is not PlatformSupport]
        if len(found) != 1:
            raise ValueError(f"{module.__name__} declares {len(found)} PlatformSupport subclasses, not one")
        return found[0]()
