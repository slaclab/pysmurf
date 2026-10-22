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
#    What the procedure leaves behind is the register layers' state and no
#    more: the flux ramp, the triggers and the data stream are the
#    application's to turn on afterwards. Each platform has its own small
#    subclass naming its steps in order and working on the rogue tree directly; where a device is on the
#    tree comes from the platform's map module, which is the one place a
#    register path is written down.
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
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import pyrogue

from cryodaq import platform

__all__ = ['Setup', 'AtcaSetup', 'RfsocSetup', 'RfmcSetup', 'setup_for', 'SETUPS', 'StepError',
           'LOAD_TIMEOUT_PER_LAYER_S', 'LOAD_RETRIES', 'ELASTIC_BUFFER_RETRIES',
           'RESET_RECOVERY_TIMEOUT_S']

_log = logging.getLogger(__name__)

# How long one register layer is given to load before a try is judged stuck. A
# full layer takes about a minute over a crate's link; the bound is well clear
# of that and scales with the number of layers, since each is a full write.
LOAD_TIMEOUT_PER_LAYER_S = 300.0
LOAD_RETRIES = 4
ELASTIC_BUFFER_RETRIES = 10
# How long the RFMC's logic is given to come out of a reset before the step
# gives up; the firmware's own helpers wait for ever by default.
RESET_RECOVERY_TIMEOUT_S = 10.0
# The status block's JESD verdict, as its enumeration numbers them.
JESD_UNLOCKED, JESD_LOCKED, JESD_NOT_FOUND = 0, 1, 3


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


class Setup(pyrogue.Process):
    """The configuration procedure, as a process on the tree.

    It takes no parameter: it loads the register layers the server was given
    and leaves the system in the state they describe. Its outcome is the
    application status block's ``SystemConfigured``; ``Message`` names the
    step that failed when that is false. Each platform subclasses it with
    its own ``run``.

    Parameters
    ----------
    layers : sequence of str
        The register layers to load, in order.
    status : ApplicationStatus
        Where the outcome is reported.
    """

    def __init__(self, layers: Sequence[str], status: Any, **kwargs):
        pyrogue.Process.__init__(self, name='setDefaults',
                                 description='Configure the system: load the register layers '
                                             'and check the links',
                                 **kwargs)
        self.layers = tuple(str(layer) for layer in layers)
        self.status = status

    def steps(self, root: Any) -> Sequence[Tuple[str, Callable[[], None]]]:
        """The platform's steps on the started tree, in order, each named for ``Message``."""
        raise NotImplementedError('a platform subclass names its steps')

    def run(self, root: Any) -> None:
        """Run the steps in order; a step that cannot complete raises, naming itself.

        A server with no layers is refused before any step runs: the carrier's
        converter resets would otherwise disturb the hardware for a load that
        was never going to happen.
        """
        if not self.layers:
            raise StepError('load layers: no register layers were given')
        steps = self.steps(root)
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
            self.run(self.root)
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


class AtcaSetup(Setup):
    """The ATCA carrier's procedure: converters reset, layers loaded, links checked.

    The carrier's converters sit on AMC cards behind JESD links. A bay is
    populated when its front end (``MicrowaveMuxCore[bay]``) is on the tree;
    the composition leaves an empty bay's out.
    """

    # What each receive lane's elastic-buffer latency reads once the converter
    # links have come up correctly: lanes 2 and 3 carry no data on this AMC and
    # read the idle value, the rest settle one of two ticks apart.
    ELASTIC_BUFFER_LATENCY: Dict[int, Tuple[int, ...]] = {
        lane: ((255,) if lane in (2, 3) else (13, 14)) for lane in range(10)}

    def steps(self, root: Any) -> Sequence[Tuple[str, Callable[[], None]]]:
        # The load itself runs the firmware's link bring-up; the converters
        # are reset before it and the links judged after it.
        return (('read all', root.ReadAll),
                ('reset DACs', lambda: self.reset_dacs(root)),
                ('load layers', lambda: load_layers(root, self.layers)),
                ('check elastic buffers', lambda: self.check_elastic_buffers(root)))

    def run(self, root: Any) -> None:
        # The load runs the firmware's link check too, so the procedure's
        # outcome is also the links' verdict, reported where the status block
        # shows it -- unless the firmware predates the check, which the status
        # says ('Not found') and the outcome must not overwrite. Every step is
        # inside, so a failed read also reports the links unlocked rather than
        # leaving an earlier run's verdict.
        verdict = self.status.JesdStatus
        try:
            Setup.run(self, root)
        except Exception:
            if verdict.value() != JESD_NOT_FOUND:
                verdict.set(JESD_UNLOCKED)
            raise
        if verdict.value() != JESD_NOT_FOUND:
            verdict.set(JESD_LOCKED)

    @staticmethod
    def populated_bays(root: Any) -> List[Tuple[int, Any, Any]]:
        """``(bay, front end, JESD block)`` for each bay with a front end on the tree."""
        bays = []
        for bay in range(2):
            core = root.getNode(platform.CARRIER_FRONT_END_PATH.format(bay=bay))
            if core is not None:
                bays.append((bay, core, root.getNode(platform.CARRIER_LINKS_PATH.format(bay=bay))))
        return bays

    def reset_dacs(self, root: Any) -> None:
        """Pulse the reset of every RF DAC on every populated bay.

        A loaded configuration leaves the converter debug block disabled, which
        is where the reset bits live; it is enabled first so the writes land.
        """
        bays = self.populated_bays(root)
        for _, core, _ in bays:
            core.DBG.enable.set(True)
        for value in (1, 0):
            for _, core, _ in bays:
                for reset in core.DBG.dacReset.values():
                    reset.set(value)

    def check_elastic_buffers(self, root: Any) -> None:
        """Check each receive lane's elastic-buffer latency against the values expected.

        A lane outside its expected values has a link that did not align, and
        the layers are loaded again -- the load runs the firmware's own link
        bring-up -- before the lanes are read again.
        """
        for k in range(ELASTIC_BUFFER_RETRIES):
            _log.info("check elastic buffers (try %d)", k)
            bad = []
            for bay, _, jesd in self.populated_bays(root):
                # Reading the individual registers does not work; read the device.
                jesd.JesdRx.ReadDevice()
                for lane, latency in jesd.JesdRx.ElBuffLatency.items():
                    if latency.value() not in self.ELASTIC_BUFFER_LATENCY.get(lane, ()):
                        bad.append((bay, lane, latency.value()))
            if not bad:
                _log.info("elastic buffer check passed")
                return
            for bay, lane, latency in bad:
                _log.warning("  bay %d lane %d elastic buffer latency %d", bay, lane, latency)
            if k < ELASTIC_BUFFER_RETRIES - 1:
                _log.warning("check failed; loading the configuration again")
                load_layers(root, self.layers)
        raise StepError(f"the elastic buffer check failed {ELASTIC_BUFFER_RETRIES} times")


class RfsocSetup(Setup):
    """The RFSoC's procedure: no converter cards, so the layers alone."""

    def steps(self, root: Any) -> Sequence[Tuple[str, Callable[[], None]]]:
        return (('read all', root.ReadAll),
                ('load layers', lambda: load_layers(root, self.layers)))


class RfmcSetup(Setup):
    """The RFMC carrier's procedure: clocks and converters up, then the layers.

    The carrier's data converters are on the RFSoC and its clocks come from
    chips on the board, so bringing it up is the order the firmware's own root
    uses: reset the user logic, route a timing source to the receiver, program
    the clock chips from the vendor's register file, wait for the DSP clock,
    initialise the RF data converter, load the layers, then synchronise the
    converter tiles. The converter is a device of its own on the root, reached
    over a second register path to the processor system, not a child of the
    FPGA's tree.
    """

    # The clock crossbar output feeding the timing receiver, and the source
    # routed to it: the backplane, as the firmware's own start-up routes.
    TIMING_SOURCE = 'backplane'
    # The multi-tile synchronisation groups and their reference tiles: every
    # tile of each converter, with the tile that distributes the clock as the
    # reference (DAC228 for the DACs, ADC226 for the ADCs).
    MTS_DAC_TILES, MTS_DAC_REF_TILE = 0xF, 0x0
    MTS_ADC_TILES, MTS_ADC_REF_TILE = 0xF, 0x2

    def __init__(self, layers: Sequence[str], status: Any, lmk_config: str = ''):
        Setup.__init__(self, layers, status)
        # The LMK's register values as TI's tool exports them; the firmware
        # package ships one beside its register defaults.
        self.lmk_config = lmk_config

    def steps(self, root: Any) -> Sequence[Tuple[str, Callable[[], None]]]:
        carrier = root.getNode(platform.RFMC_CARRIER_PATH)
        rfdc = root.getNode(platform.RFMC_RFDC_PATH)
        return (('read all', root.ReadAll),
                ('reset user logic', lambda: self.reset_user_logic(carrier)),
                ('route timing', lambda: self.route_timing(carrier)),
                ('init clocks', lambda: carrier.InitClock(lmkConfig=self.lmk_config)),
                ('wait for DSP clock', lambda: self.wait_for_dsp_clock(carrier)),
                ('init RF data converter', rfdc.Init),
                ('load layers', lambda: load_layers(root, self.layers)),
                ('sync converter tiles', lambda: self.sync_converter_tiles(rfdc)))

    @staticmethod
    def reset_user_logic(carrier: Any) -> None:
        """Reset the user logic and wait for it to come back."""
        if carrier.UserRst(timeout=RESET_RECOVERY_TIMEOUT_S):
            raise StepError(f"the user logic did not come out of reset in "
                            f"{RESET_RECOVERY_TIMEOUT_S:.0f} s")

    def route_timing(self, carrier: Any) -> None:
        """Route the start-up timing source to the FPGA's timing receiver."""
        output = carrier.AxiSy56040.OutputConfig[platform.RFMC_TIMING_RECEIVER_CROSSBAR_OUTPUT]
        output.set(platform.RFMC_STARTUP_TIMING_SOURCE[self.TIMING_SOURCE])

    @staticmethod
    def wait_for_dsp_clock(carrier: Any) -> None:
        """Wait for the DSP clock domain to come out of reset."""
        if carrier.DspRstWait(timeout=RESET_RECOVERY_TIMEOUT_S):
            raise StepError(f"the DSP clock did not come out of reset in "
                            f"{RESET_RECOVERY_TIMEOUT_S:.0f} s")

    def sync_converter_tiles(self, rfdc: Any) -> None:
        """Multi-tile synchronise the DAC tiles, then the ADC tiles."""
        mts = rfdc.Mts
        mts.DacTiles.set(self.MTS_DAC_TILES)
        mts.AdcTiles.set(self.MTS_ADC_TILES)
        mts.DacRefTile.set(self.MTS_DAC_REF_TILE)
        mts.AdcRefTile.set(self.MTS_ADC_REF_TILE)
        mts.SyncDacTiles()
        mts.SyncAdcTiles()


# The procedure each platform runs, by the platform's name.
SETUPS: Dict[str, Type[Setup]] = {
    'umux-atca': AtcaSetup,
    'umux-rfsoc': RfsocSetup,
    'umux-rfmc': RfmcSetup,
}


def setup_for(pmap: platform.PlatformMap, layers: Sequence[str], status: Any, **options: Any) -> Setup:
    """The configuration procedure for a platform.

    Parameters
    ----------
    options
        What the platform's procedure needs beyond the layers: the RFMC's
        clock chip register file (``lmk_config``).

    Raises
    ------
    ValueError
        If no procedure is declared for the platform.
    """
    try:
        cls = SETUPS[pmap.name]
    except KeyError:
        raise ValueError(f"no configuration procedure is declared for platform {pmap.name!r}; "
                         f"known: {', '.join(sorted(SETUPS))}") from None
    return cls(layers, status, **options)
