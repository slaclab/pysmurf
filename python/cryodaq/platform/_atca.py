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
#    Most of the registers come from _umux, which this platform shares with the
#    others of its generation. What this file adds are the registers of the
#    hardware only this platform carries -- the RF front end on each AMC bay, its
#    converters and attenuators, and the serial links that carry data back from
#    them. A platform whose converters share the FPGA's die has none of those, so
#    it does not declare them: a map is a statement of what a platform has, and a
#    name it offers should be one that reaches something. Nothing here reads or
#    writes a register.
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

# --------------------------------------------------------------------------
# what only this platform has
# --------------------------------------------------------------------------

# The RF front end on an AMC bay: the attenuators, the digital-to-analogue converters
# and the debug block beside them. Absent where converter and FPGA share a die.
#
# An attenuator is one RF path through the bay, numbered from 1, with an up-converter
# and a down-converter setting: UC[n] and DC[n] are the two directions of the same
# path, so `n` is one index and not two. Which 500 MHz band a path carries is not the
# firmware's to say -- it is wiring, and the application supplies that mapping.
_ATTENUATORS = f'{_umux.MUX_CORE}.ATT'
_ATTENUATOR_UC = f'{_ATTENUATORS}.UC[{{attenuator}}]'
_ATTENUATOR_DC = f'{_ATTENUATORS}.DC[{{attenuator}}]'
_DAC = f'{_umux.MUX_CORE}.DAC[{{dac}}]'
_DAC_TEMPERATURE = f'{_DAC}.Temperature'
_DAC_JESD_RESET_N = f'{_DAC}.JesdRstN'
_DAC_ENABLE = f'{_DAC}.enable'
# The clock chip on the AMC, which the converters are timed from: a TI LMK04828.
_LMK = f'{_umux.MUX_CORE}.LMK'
_LMK_ENABLE = f'{_LMK}.enable'
_LMK_POWER_UP_SYS_REF = f'{_LMK}.PwrUpSysRef'
# Which of the chip's clock inputs is used, and how. The firmware names these two by
# their register number because that is how the datasheet addresses them; the fields
# they carry are what makes them a pair. From the chip's register map, as the firmware
# records it:
#   0x0146  CLKin2_EN, CLKin1_EN, CLKin0_EN, CLKin2_TYPE, CLKin1_TYPE, CLKin0_TYPE
#   0x0147  CLKin_SEL_POL, CLKin_SEL_MODE, CLKin1_OUT_MUX, CLKin0_OUT_MUX
# So one says which input is on and what kind it is, the other how the input is selected
# and routed. Together they are how a bay's timing reference is chosen.
_LMK_CLOCK_INPUT_ENABLE = f'{_LMK}.LmkReg_0x0146'
_LMK_CLOCK_INPUT_SELECT = f'{_LMK}.LmkReg_0x0147'
# Select the external reference clock for this bay's converters.
_SELECT_EXTERNAL_REFERENCE = f'{_umux.MUX_CORE}.SelExtRef'

_BAY_DEBUG = f'{_umux.MUX_CORE}.DBG'
_BAY_DEBUG_ENABLE = f'{_BAY_DEBUG}.enable'
_DAC_RESET = f'{_BAY_DEBUG}.dacReset[{{dac}}]'

# The serial links between the FPGA and those converters, one block per bay: whether
# each direction has locked, whether it is enabled, and what each transmit lane drives.
_JESD_RX_DATA_VALID = f'{_umux.JESD_BAY}.JesdRx.DataValid'
_JESD_TX_DATA_VALID = f'{_umux.JESD_BAY}.JesdTx.DataValid'
_JESD_RX_ENABLE = f'{_umux.JESD_BAY}.JesdRx.Enable'
_JESD_TX_ENABLE = f'{_umux.JESD_BAY}.JesdTx.Enable'
_JESD_TX_DATA_OUT_MUX = f'{_umux.JESD_BAY}.JesdTx.dataOutMux[{{tx_lane}}]'
# One status counter per link in each direction, counting how often that link has
# reported itself valid. Indexed by link, which is not the transmit lane above: the lanes
# are what a link is made of.
_JESD_RX_STATUS_VALID_COUNT = (f'{_umux.JESD_BAY}.JesdRx.StatusValidCnt'
                               f'[{{link}}]')
_JESD_TX_STATUS_VALID_COUNT = (f'{_umux.JESD_BAY}.JesdTx.StatusValidCnt'
                               f'[{{link}}]')

_V, _C = 'value', 'command'

REGISTERS = dict(_umux.REGISTERS)
REGISTERS.update({
    # the RF front end
    'bay[*].attenuator[*].uc': (_ATTENUATOR_UC, _V),
    'bay[*].attenuator[*].dc': (_ATTENUATOR_DC, _V),
    'bay[*].dac[*].temperature': (_DAC_TEMPERATURE, _V),
    'bay[*].dac[*].enable': (_DAC_ENABLE, _V),
    'bay[*].dac[*].jesd_reset_n': (_DAC_JESD_RESET_N, _V),
    'bay[*].dac[*].reset': (_DAC_RESET, _V),
    'bay[*].debug.enable': (_BAY_DEBUG_ENABLE, _V),
    'bay[*].clock.enable': (_LMK_ENABLE, _V),
    'bay[*].clock.power_up_sys_ref': (_LMK_POWER_UP_SYS_REF, _C),
    'bay[*].clock.select_external_reference': (_SELECT_EXTERNAL_REFERENCE, _C),
    'bay[*].clock_input.enable': (_LMK_CLOCK_INPUT_ENABLE, _V),
    'bay[*].clock_input.select': (_LMK_CLOCK_INPUT_SELECT, _V),
    # the serial links back from it
    'bay[*].jesd.rx_data_valid': (_JESD_RX_DATA_VALID, _V),
    'bay[*].jesd.tx_data_valid': (_JESD_TX_DATA_VALID, _V),
    'bay[*].jesd.rx_enable': (_JESD_RX_ENABLE, _V),
    'bay[*].jesd.tx_enable': (_JESD_TX_ENABLE, _V),
    'bay[*].jesd.tx_lane[*].data_out_mux': (_JESD_TX_DATA_OUT_MUX, _V),
    'bay[*].jesd.link[*].rx_status_valid_count': (_JESD_RX_STATUS_VALID_COUNT, _V),
    'bay[*].jesd.link[*].tx_status_valid_count': (_JESD_TX_STATUS_VALID_COUNT, _V),
})

# The scopes those registers are indexed by. They hang off `bay`, which the shared map
# declares, because a bay is a bay on every platform of the generation; what differs is
# what is inside one.
SCOPES = dict(_umux.SCOPES)
SCOPES.update({
    # Either direction proves a path: a firmware with one and not the other has wired up
    # fewer of one, which is a fact about the tree, as with the event receiver's scopes.
    'attenuator': ((_ATTENUATOR_UC, _ATTENUATOR_DC), ('bay',)),
    'link': ((_JESD_RX_STATUS_VALID_COUNT, _JESD_TX_STATUS_VALID_COUNT), ('bay',)),
    'dac': ((_DAC,), ('bay',)),
    'tx_lane': ((_JESD_TX_DATA_OUT_MUX,), ('bay',)),
})

# Worth recording how a carrier was left: whether each bay's data links had locked, and
# what its attenuators were set to. Both are this platform's hardware, so both are here
# rather than in the shared list.
WITNESS = _umux.WITNESS + (
    'bay[*].jesd.rx_data_valid',
    'bay[*].jesd.tx_data_valid',
    'bay[*].attenuator[*].uc',
    'bay[*].attenuator[*].dc',
)
