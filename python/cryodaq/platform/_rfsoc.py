#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Map: microwave-multiplexed readout on an RFSoC
#-----------------------------------------------------------------------------
# File       : _rfsoc.py
# Created    : 2026-09-14
#-----------------------------------------------------------------------------
# Description:
#    The platform map for microwave-multiplexed readout on an RFSoC, whose data
#    converters are on the chip: it has no per-bay JESD links and no AMC front
#    end, so the names those registers back do not resolve on it and its bay
#    scope is empty. A caller asks what a system offers rather than assuming.
#
#    The registers come from _umux, which this platform shares with the others of
#    its generation. It is a platform of its own because it is configured
#    differently -- bring-up ordering and what has to be set up part company with
#    a carrier even where the paths agree -- and that is the difference the
#    bring-up procedures will need once they move into cryodaq. Nothing here
#    reads or writes a register.
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

NAME = 'umux-rfsoc'

# The firmware image names this platform runs, as the build stamp reports them.
# Unlike the carrier's, none of these has been read off a board: they are derived.
# An image is named after the build target it comes from, and these are the target
# names of the two RFSoC firmware lines at the releases docker/server's
# definitions.sh pins. The derivation is corroborated on the carrier, whose one
# target name is exactly what a carrier reports.
#
# The pre-SPECTRA line is on this platform rather than one of its own: its
# firmware is the same base with another flag defaulted on, so it reads the same
# registers and is brought up the same way.
#
# Not to be confused with the release archives, which are named for the family
# (rogue_MicrowaveMuxZcu208_v3.2.1.zip) and not for the image inside them; the
# family name is not what a build stamp reports and is deliberately absent here.
#
# Until an RFSoC is available to read a build stamp from, these names are the
# derivation and not a measurement: they are to be confirmed against a live
# system, and corrected here if what it reports differs. A name that is wrong
# fails safe -- identification refuses a system it does not recognise rather
# than mapping it to the wrong platform -- and until then `platform_name=` names
# the platform for a system whose firmware is not listed.
TAGS = (
    'MicrowaveMuxZcu208_BaseBand',
    'MicrowaveMuxZcu208_HighOrderNyquist',
    'MicrowaveMuxZcu208_PreSpectra',
)
