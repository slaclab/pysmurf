#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Support: microwave-multiplexed readout on the RFMC carrier
#-----------------------------------------------------------------------------
# File       : rfmc/support.py
# Created    : 2026-10-24
#-----------------------------------------------------------------------------
# Description:
#    What this platform contributes to the server's rogue tree: the firmware's
#    top level in the processor system's register window, the RF data converter
#    as a device beside it over its own register path, and the configuration
#    procedure that brings the board's clocks and converter up before loading
#    the register layers. The register map beside this module is data a client
#    reads without rogue; this is the server's half of the platform.
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
from typing import Any, Callable, List, Sequence, Tuple

from cryodaq._errors import ConnectError
from cryodaq.platform import rfmc
from cryodaq.server._setup import PlatformSupport, Setup, StepError, load_layers

__all__ = ['RfmcSupport', 'RfmcSetup']

# The firmware's Python package, which ships the clock chips' register file
# beside its top level.
FIRMWARE_PACKAGE = 'rfmc_cryo_det'

# A file taken out of a release archive for a driver that reads plain files; kept
# for the life of the process, since a server composes once and runs.
_EXTRACTED = contextlib.ExitStack()
atexit.register(_EXTRACTED.close)


class RfmcSetup(Setup):
    """The RFMC carrier's procedure: clocks and converters up, then the layers.

    The carrier's data converters are on the RFSoC and its clocks come from
    chips on the board, so bringing it up is the order the firmware's own root
    uses at start: reset the user logic, route a timing source to the receiver,
    program the clock chips from the vendor's register file, wait for the DSP
    clock, initialise the RF data converter, read the tree, synchronise the
    converter tiles, load the layers, reset the counters. The converter is a
    device of its own on the root, reached over a second register path to the
    processor system, not a child of the FPGA's tree.

    Polling is held off for the whole run, as the firmware's root holds it off
    through its start: a poll landing on the user reset, the clock programming,
    the converter init or the tile sync reads registers whose clock domain is
    still held down.
    """

    # The clock crossbar output feeding the timing receiver, and the source
    # routed to it: the backplane, as the firmware's own start-up routes.
    TIMING_SOURCE = 'backplane'
    # The multi-tile synchronisation groups and their reference tiles: every
    # tile of each converter, with the tile that distributes the clock as the
    # reference (DAC228 for the DACs, ADC226 for the ADCs).
    MTS_DAC_TILES, MTS_DAC_REF_TILE = 0xF, 0x0
    MTS_ADC_TILES, MTS_ADC_REF_TILE = 0xF, 0x2
    # How long the logic is given to come out of a reset before the step gives
    # up; the firmware's own helpers wait for ever by default.
    RESET_RECOVERY_TIMEOUT_S = 10.0

    def steps(self) -> Sequence[Tuple[str, Callable[[], None]]]:
        root = self.root
        carrier = root.getNode(rfmc.CARRIER.CARRIER)
        rfdc = root.getNode(rfmc.RFDC)
        return (('reset user logic', lambda: self.reset_user_logic(carrier)),
                ('route timing', lambda: self.route_timing(carrier)),
                ('init clocks', lambda: self.init_clocks(carrier)),
                ('wait for DSP clock', lambda: self.wait_for_dsp_clock(carrier)),
                ('init RF data converter', rfdc.Init),
                ('read all', root.ReadAll),
                ('sync converter tiles', lambda: self.sync_converter_tiles(rfdc)),
                ('load layers', lambda: load_layers(root, self.layers)),
                ('reset counters', root.CountReset))

    def run_steps(self) -> None:
        # Polling held off: a poll mid-reset or mid-sync reads registers whose
        # clock is not yet running. Restored to what it was, whatever happens.
        polling = self.root.PollEn.value()
        self.root.PollEn.set(False)
        try:
            Setup.run_steps(self)
        finally:
            self.root.PollEn.set(polling)

    def reset_user_logic(self, carrier: Any) -> None:
        """Reset the user logic and wait for it to come back."""
        if carrier.UserRst(timeout=self.RESET_RECOVERY_TIMEOUT_S):
            raise StepError(f"the user logic did not come out of reset in "
                            f"{self.RESET_RECOVERY_TIMEOUT_S:.0f} s")

    def route_timing(self, carrier: Any) -> None:
        """Route the start-up timing source to the FPGA's timing receiver."""
        output = carrier.AxiSy56040.OutputConfig[rfmc.TIMING_RECEIVER_CROSSBAR_OUTPUT]
        output.set(rfmc.STARTUP_TIMING_SOURCE[self.TIMING_SOURCE])

    @staticmethod
    def lmk_config() -> str:
        """The clock chips' register file, as a plain file the chip's driver can open.

        The firmware package ships it as a resource; one inside a release
        archive is extracted beside the server for as long as it runs.
        """
        resource = importlib.resources.files(FIRMWARE_PACKAGE).joinpath(rfmc.LMK_CONFIG)
        if not resource.is_file():
            raise StepError(f"the {FIRMWARE_PACKAGE} package ships no {rfmc.LMK_CONFIG}, "
                            f"which the clock chips are programmed from")
        return str(_EXTRACTED.enter_context(importlib.resources.as_file(resource)))

    def init_clocks(self, carrier: Any) -> None:
        """Program the LMK and LMX clock chips from the vendor's register file."""
        carrier.InitClock(lmkConfig=self.lmk_config())

    def wait_for_dsp_clock(self, carrier: Any) -> None:
        """Wait for the DSP clock domain to come out of reset."""
        if carrier.DspRstWait(timeout=self.RESET_RECOVERY_TIMEOUT_S):
            raise StepError(f"the DSP clock did not come out of reset in "
                            f"{self.RESET_RECOVERY_TIMEOUT_S:.0f} s")

    def sync_converter_tiles(self, rfdc: Any) -> None:
        """Multi-tile synchronise the DAC tiles, then the ADC tiles."""
        mts = rfdc.Mts
        mts.DacTiles.set(self.MTS_DAC_TILES)
        mts.AdcTiles.set(self.MTS_ADC_TILES)
        mts.DacRefTile.set(self.MTS_DAC_REF_TILE)
        mts.AdcRefTile.set(self.MTS_ADC_REF_TILE)
        mts.SyncDacTiles()
        mts.SyncAdcTiles()


class RfmcSupport(PlatformSupport):
    """The RFMC carrier's tree: the top level in the processor's window, the converter beside it."""

    setup = RfmcSetup

    def top_level(self, transport: Any, **options: Any) -> Any:
        """The carrier's top level, at the FPGA's window in the processor system's address space.

        This class takes no deployment option -- the board has no bays to leave
        out -- and its own constructor would pass one on to rogue's `Device`
        rather than refuse it, so it is refused here.
        """
        if options:
            raise TypeError(f"the RFMC top level takes no option; given {', '.join(sorted(options))}")
        from rfmc_cryo_det import FpgaTopLevel
        return FpgaTopLevel(memBase=transport.srp, offset=transport.register_offset)

    def devices(self, transport: Any) -> List[Any]:
        """The RF data converter, over the transport's second register path."""
        if transport.srp_rfdc is None:
            raise ConnectError(f"platform {rfmc.NAME} drives its RF data converter over a second "
                               f"register path (srp_rfdc), which the {transport.name} transport does not have")
        from axi_soc_ultra_plus_core.rfsoc_utility import Rfdc
        return [Rfdc(name=rfmc.RFDC.rsplit('.', 1)[1], memBase=transport.srp_rfdc)]
