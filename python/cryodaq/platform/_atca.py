#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Map: microwave-multiplexed readout on an ATCA carrier
#-----------------------------------------------------------------------------
# File       : _atca.py
# Created    : 2026-09-14
#-----------------------------------------------------------------------------
# Description:
#    The platform map for microwave-multiplexed readout carried on an ATCA
#    board: two AMC bays behind JESD data links, an RTM, and the per-band signal
#    processing of the generation.
#
#    The registers come from _umux, which this platform shares with the others
#    of its generation; what belongs here is the firmware this platform runs and,
#    as the bring-up procedures move into cryodaq, whatever of them is specific
#    to a carrier. Nothing here reads or writes a register.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

from cryodaq.platform._umux import REGISTERS, SCOPES, WITNESS

__all__ = ['NAME', 'TAGS', 'REGISTERS', 'WITNESS', 'SCOPES']

NAME = 'umux-atca'

# The firmware image names this platform runs, as the build stamp reports them.
# Read off a slot-4 carrier running v2.5.1:
#
#   MicrowaveMuxBpEthGen2: Vivado v2020.2, rdsrv403 (Ubuntu 22.04.5 LTS), ...
#
# which is also the name of the one firmware build target this platform has, so
# an image reports the target it was built from.
#
# Only images confirmed to share these registers and this bring-up belong here.
# The earlier image lines are not listed: whether they do has not been measured,
# and an unlisted image is refused by name, which is a better answer than a map
# that may be wrong about it.
TAGS = ('MicrowaveMuxBpEthGen2',)
