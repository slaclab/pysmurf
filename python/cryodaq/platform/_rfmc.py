#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Map: microwave-multiplexed readout on the RFMC carrier
#-----------------------------------------------------------------------------
# File       : _rfmc.py
# Created    : 2026-10-22
#-----------------------------------------------------------------------------
# Description:
#    The platform map for microwave-multiplexed readout on the RFMC carrier: an
#    RFSoC whose data converters are on the chip, on a carrier board of its own
#    with two RF front-end quads -- each with the same attenuator block an AMC
#    bay carries, and four synthesiser PLLs the AMC does not. It has no JESD
#    links, so it does not offer the names those registers back.
#
#    The application registers come from _umux, as every platform's do. The
#    carrier core is this board's own device, with the same leaves inside as the
#    AMC carrier's, so its paths are the shared template bound to this core; the
#    uMux application sits under the same AppTop.AppCore as on the other two.
#    Nothing here reads or writes a register.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

from cryodaq.platform import _umux

__all__ = ['NAME', 'TAGS', 'TOP_LEVEL', 'PACKAGE_CONFIG_DIR', 'CARRIER', 'REGISTERS', 'LEGACY',
           'WITNESS', 'SCOPES', 'QUAD', 'RFDC', 'SERVER_DEVICES', 'SETUP_FILES', 'STARTUP_TIMING_SOURCE',
           'TIMING_RECEIVER_CROSSBAR_OUTPUT']

NAME = 'umux-rfmc'

# The firmware image name this platform runs, as the build stamp reports it. Read
# off a board running v1.1.0:
#
#   MicrowaveMuxRfmcCarrier: Vivado v2026.1, rdsrv403 (Ubuntu 22.04.5 LTS), ...
#
# which is the one build target of the rfmc-cryo-det firmware, as the carrier's
# image reports its target.
TAGS = ('MicrowaveMuxRfmcCarrier',)

# The class in the firmware's Python package that builds this platform's register
# tree, as `module:attribute`. The RFMC firmware has a package of its own, which
# carries the carrier's and the uMux application's packages as submodules.
TOP_LEVEL = 'rfmc_cryo_det:FpgaTopLevel'
# Where the firmware's release archive keeps its configuration files -- the register
# defaults, and the clock chip's register file under lmk/ -- so a relative name a
# server is given means a file there.
PACKAGE_CONFIG_DIR = 'python/rfmc_cryo_det/config'

# --------------------------------------------------------------------------
# the carrier core
# --------------------------------------------------------------------------

# This board's carrier core, with its board-support and timing devices named for it.
# No regulator is attached beside it: the AMC carrier's EM22xx is that board's.
CARRIER = _umux.Carrier('RfmcCarrierCore', bsi='RfmcCarrierBsi', timing='RfmcCarrierTiming')

# The output of the clock crossbar that feeds the FPGA's timing receiver, and the
# source routed to it when the system is brought up. The firmware's own start-up
# routes the backplane; the alternatives are the two RTM inputs and standalone.
STARTUP_TIMING_SOURCE = {'backplane': 0x2, 'rtm0': 0x0, 'rtm1': 0x3, 'standalone': 0x1}
TIMING_RECEIVER_CROSSBAR_OUTPUT = 1

# The RF data converter: a device the *server* adds beside the FPGA's tree, driven by
# the firmware package's class over its own register path to the processor system
# rather than through the FPGA's memory, so it is not in a dump of the package. The
# configuration procedure initialises it and synchronises its tiles.
RFDC = f'{_umux.ROOT}.Rfdc'
SERVER_DEVICES = (
    _umux.ServerDevice(path=RFDC, driver='axi_soc_ultra_plus_core.rfsoc_utility:Rfdc', link='srp_rfdc'),
)
# The clock chips' register file, as TI's tool exports it, where the firmware's Python
# package ships it (relative to the package); the configuration procedure programs the
# LMK and LMX from it.
SETUP_FILES = {'lmk_config': 'config/lmk/HexRegisterValues.txt'}

# --------------------------------------------------------------------------
# what only this platform has
# --------------------------------------------------------------------------

# The RF front end: one quad per bay, carrying the attenuators and the synthesisers.
# Public, re-exported by the package: a configuration procedure for this platform
# works on the quads directly and takes where they are from here.
QUAD = f'{_umux.APP_CORE}.RfmcQuad[{{bay}}]'

# The attenuators are the same block as an AMC bay's, so the same names resolve: an
# attenuator is one RF path through the quad, numbered from 1, with an up-converter
# and a down-converter setting that are the two directions of one path.
_ATTENUATORS = f'{QUAD}.ATT'
_ATTENUATOR_UC = f'{_ATTENUATORS}.UC[{{attenuator}}]'
_ATTENUATOR_DC = f'{_ATTENUATORS}.DC[{{attenuator}}]'
# The four synthesiser PLLs of a quad (ADF5356), and whether each reports lock.
_PLL_LOCKED = f'{QUAD}.PllLocked[{{pll}}]'

_V = 'value'

REGISTERS = {**_umux.REGISTERS, **CARRIER.registers()}
REGISTERS.update({
    'bay[*].attenuator[*].uc': (_ATTENUATOR_UC, _V),
    'bay[*].attenuator[*].dc': (_ATTENUATOR_DC, _V),
    'bay[*].pll[*].locked': (_PLL_LOCKED, _V),
})

LEGACY = {**_umux.LEGACY, **CARRIER.legacy()}

SCOPES = {**_umux.SCOPES, **CARRIER.scopes()}
SCOPES.update({
    'attenuator': ((_ATTENUATOR_UC, _ATTENUATOR_DC), ('bay',)),
    'pll': ((_PLL_LOCKED,), ('bay',)),
})

# Worth recording how this carrier was left: its attenuator settings and whether
# each synthesiser locked.
WITNESS = _umux.WITNESS + (
    'bay[*].attenuator[*].uc',
    'bay[*].attenuator[*].dc',
    'bay[*].pll[*].locked',
)
