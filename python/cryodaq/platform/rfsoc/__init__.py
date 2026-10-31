#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Map: microwave-multiplexed readout on an RFSoC
#-----------------------------------------------------------------------------
# File       : rfsoc/__init__.py
# Created    : 2026-09-14
#-----------------------------------------------------------------------------
# Description:
#    The platform map for microwave-multiplexed readout on an RFSoC, whose data
#    converters are on the chip: it has no per-bay JESD links and no AMC front
#    end, so it does not offer the names those registers back at all. Asking for
#    one is an unresolved name rather than a path that reaches nothing, and the
#    difference matters -- a map is a statement of what a platform has.
#
#    The registers come from _umux, which is every register of this generation
#    that does not depend on a separate converter board. It is a platform of its
#    own because it is configured differently -- bring-up ordering and what has
#    to be set up part company with a carrier even where the paths agree -- and
#    that is the difference the bring-up procedures will need once they move into
#    cryodaq. Nothing here reads or writes a register.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

from cryodaq.platform import _umux, amc

__all__ = ['NAME', 'TAGS', 'PACKAGE_CONFIG_DIR', 'BRING_UP', 'CARRIER', 'REGISTERS', 'LEGACY',
           'WITNESS', 'SCOPES']

# The carrier core is the AMC carrier's: this firmware is the carrier's package built
# without its bays, and the core, the regulator beside it and the shared registers are
# as the carrier has them, so the board's tables are taken from the carrier's module.
CARRIER = amc.CARRIER
REGISTERS = {**_umux.REGISTERS, **amc.AMC_CARRIER_REGISTERS}
LEGACY = {**_umux.LEGACY, **amc.AMC_CARRIER_LEGACY}
SCOPES = {**_umux.SCOPES, **amc.AMC_CARRIER_SCOPES}
WITNESS = _umux.WITNESS

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

# Where the firmware's release archive keeps its configuration files, so a relative
# name a server is given means a file there.
PACKAGE_CONFIG_DIR = 'python/CryoDet/config'

# An evaluation board: no shelf manager, no FRU record, no bays; one defaults
# layer, and a processor system beside the FPGA at the +200 address.
BRING_UP = _umux.BringUp(firmware_image='MicrowaveMuxZcu208', processor_offset=200,
                         defaults_layer='defaults_rfsoc_zcu208.yml')
