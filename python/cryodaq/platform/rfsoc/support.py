#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Support: microwave-multiplexed readout on an RFSoC
#-----------------------------------------------------------------------------
# File       : rfsoc/support.py
# Created    : 2026-10-24
#-----------------------------------------------------------------------------
# Description:
#    What this platform contributes to the server's rogue tree: the firmware's
#    top level built for the transport in hand, and the configuration procedure
#    -- the register layers alone, there being no converter cards to reset or
#    serial links to check. The register map beside this module is data a client
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
from typing import Any, Callable, Sequence, Tuple

from cryodaq.server._setup import PlatformSupport, Setup, load_layers

__all__ = ['RfsocSupport', 'RfsocSetup']


class RfsocSetup(Setup):
    """The RFSoC's procedure: no converter cards, so the layers alone."""

    def steps(self) -> Sequence[Tuple[str, Callable[[], None]]]:
        root = self.root
        return (('read all', root.ReadAll),
                ('load layers', lambda: load_layers(root, self.layers)))


class RfsocSupport(PlatformSupport):
    """The RFSoC's tree: the carrier's package built without its bays, and the procedure above."""

    setup = RfsocSetup

    def top_level(self, transport: Any, **options: Any) -> Any:
        """The RFSoC's top level: the carrier's class with the RFSoC flag defaulted on.

        ``options`` are the class's own keyword arguments (the bay and
        power-monitor flags it shares with the carrier), as a deployment gives
        them; the RFSoC build has no bays to leave out.
        """
        from CryoDet._MicrowaveMuxZcu208 import FpgaTopLevel
        return FpgaTopLevel(memBase=transport.srp, **options)
