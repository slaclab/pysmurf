#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Support: microwave-multiplexed readout on an ATCA carrier
#-----------------------------------------------------------------------------
# File       : atca/support.py
# Created    : 2026-10-24
#-----------------------------------------------------------------------------
# Description:
#    What this platform contributes to the server's rogue tree: the firmware's
#    top level built for the transport in hand, and the configuration procedure
#    -- converters reset, register layers loaded, serial links checked -- as a
#    process on the tree. The register map beside this module is data a client
#    reads without rogue; this is the server's half of the platform, and the one
#    place its rogue code lives.
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
from typing import Any, Callable, Dict, List, Sequence, Tuple

from cryodaq._errors import ConnectError
from cryodaq.platform import atca
from cryodaq.server._setup import PlatformSupport, Setup, StepError, load_layers

__all__ = ['AtcaSupport', 'AtcaSetup']

_log = logging.getLogger(__name__)


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
    # How many times the layers are loaded again for lanes that did not align.
    ELASTIC_BUFFER_RETRIES = 10
    # The status block's JESD verdict, as its enumeration numbers them.
    JESD_UNLOCKED, JESD_LOCKED, JESD_NOT_FOUND = 0, 1, 3

    def steps(self) -> Sequence[Tuple[str, Callable[[], None]]]:
        # The load itself runs the firmware's link bring-up; the converters
        # are reset before it and the links judged after it.
        root = self.root
        return (('read all', root.ReadAll),
                ('reset DACs', self.reset_dacs),
                ('load layers', lambda: load_layers(root, self.layers)),
                ('check elastic buffers', self.check_elastic_buffers))

    def run_steps(self) -> None:
        # The load runs the firmware's link check too, so the procedure's
        # outcome is also the links' verdict, reported where the status block
        # shows it -- unless the firmware predates the check, which the status
        # says ('Not found') and the outcome must not overwrite. Every step is
        # inside, so a failed read also reports the links unlocked rather than
        # leaving an earlier run's verdict.
        verdict = self.status.JesdStatus
        try:
            Setup.run_steps(self)
        except Exception:
            if verdict.value() != self.JESD_NOT_FOUND:
                verdict.set(self.JESD_UNLOCKED)
            raise
        if verdict.value() != self.JESD_NOT_FOUND:
            verdict.set(self.JESD_LOCKED)

    def populated_bays(self) -> List[Tuple[int, Any, Any]]:
        """``(bay, front end, JESD block)`` for each bay with a front end on the tree."""
        bays = []
        for bay in range(2):
            core = self.root.getNode(atca.MUX_CORE.format(bay=bay))
            if core is not None:
                bays.append((bay, core, self.root.getNode(atca.JESD_BAY.format(bay=bay))))
        return bays

    def reset_dacs(self) -> None:
        """Pulse the reset of every RF DAC on every populated bay.

        A loaded configuration leaves the converter debug block disabled, which
        is where the reset bits live; it is enabled first so the writes land.
        """
        bays = self.populated_bays()
        for _, core, _ in bays:
            core.DBG.enable.set(True)
        for value in (1, 0):
            for _, core, _ in bays:
                for reset in core.DBG.dacReset.values():
                    reset.set(value)

    def check_elastic_buffers(self) -> None:
        """Check each receive lane's elastic-buffer latency against the values expected.

        A lane outside its expected values has a link that did not align, and
        the layers are loaded again -- the load runs the firmware's own link
        bring-up -- before the lanes are read again.
        """
        for k in range(self.ELASTIC_BUFFER_RETRIES):
            _log.info("check elastic buffers (try %d)", k)
            bad = []
            for bay, _, jesd in self.populated_bays():
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
            if k < self.ELASTIC_BUFFER_RETRIES - 1:
                _log.warning("check failed; loading the configuration again")
                load_layers(self.root, self.layers)
        raise StepError(f"the elastic buffer check failed {self.ELASTIC_BUFFER_RETRIES} times")


class AtcaSupport(PlatformSupport):
    """The ATCA carrier's tree: the carrier's top level, and the procedure above."""

    setup = AtcaSetup

    def top_level(self, transport: Any, **options: Any) -> Any:
        """The carrier's top level over the transport's register path.

        ``options`` are the class's own keyword arguments -- ``disableBay0``,
        ``disableBay1``, ``enablePwrI2C`` -- as a deployment gives them. The
        class is the firmware package's, importable once the package is on the
        path, which it is by the time a platform has been identified.

        The class takes the FPGA's registers at 0 -- its constructor has no
        ``offset`` and a rogue device's is fixed when it is built -- so a
        transport that puts them elsewhere (a processor system's window) is
        refused: a tree built at 0 over it would address nothing the probe
        identified. Passing the window on is a change to the firmware's class.
        """
        if transport.register_offset:
            raise ConnectError(f"platform {atca.NAME}'s top level takes the FPGA's registers at 0, and cannot "
                               f"be placed at the {transport.name} transport's window {transport.register_offset:#x}")
        from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel
        return FpgaTopLevel(memBase=transport.srp, **options)
