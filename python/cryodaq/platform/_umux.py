#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Map: microwave-multiplexed readout
#-----------------------------------------------------------------------------
# File       : _umux.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
#    The map between the register tree of a microwave-multiplexed readout and
#    cryodaq's semantic names. It is data only: register path templates, the
#    table of names built from them, the registers worth recording as a witness
#    of how a system was left, and how to enumerate the indexed scopes.
#
#    This module and its siblings are the only places in cryodaq where a
#    register path appears. Nothing here reads or writes a register, and nothing
#    here imports rogue; a client resolves a name to a path through
#    cryodaq.platform and does the reading itself.
#
#    The ATCA carrier and the RFSoC both use this map. Their trees differ by
#    omission -- an RFSoC has no per-bay data links and no RF front end -- so
#    the names those registers back simply do not resolve there, which a caller
#    sees as an index-free scope rather than as a broken name.
#
#    That the two share a map is a fact about their register paths and not a
#    claim that they are one platform. Where they diverge is in procedure --
#    bring-up ordering, what has to be configured and in what sequence -- so a
#    generation whose procedures differ, and not only its paths, belongs in a
#    module of its own here rather than behind a branch in this one.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

__all__ = ['NAME', 'PROBE', 'REGISTERS', 'WITNESS', 'SCOPES']

# --------------------------------------------------------------------------
# register paths, as templates over the indexed scopes
# --------------------------------------------------------------------------

# The root device; the server names its Root "AMCc", so every path starts here.
ROOT = 'AMCc'
READY = f'{ROOT}.Ready'                       # set by the server once its start() completes
SET_DEFAULTS = f'{ROOT}.setDefaults'          # the server's own configuration procedure

# The application block the server adds beside the FPGA.
APPLICATION = f'{ROOT}.SmurfApplication'
SYSTEM_CONFIGURED = f'{APPLICATION}.SystemConfigured'
CONFIGURING_IN_PROGRESS = f'{APPLICATION}.ConfiguringInProgress'
ENABLED_BAYS = f'{APPLICATION}.EnabledBays'
STARTUP_ARGUMENTS = f'{APPLICATION}.StartupArguments'
SMURF_VERSION = f'{APPLICATION}.SmurfVersion'
JESD_STATUS = f'{APPLICATION}.JesdStatus'

# The data processor.
PROCESSOR = f'{ROOT}.SmurfProcessor'
DOWNSAMPLE_FACTOR = f'{PROCESSOR}.Downsampler.InternalFactor'
FILTER_DISABLE = f'{PROCESSOR}.Filter.Disable'
FRAME_COUNT = f'{PROCESSOR}.FrameRxStats.FrameCnt'

# The FPGA.
FPGA = f'{ROOT}.FpgaTopLevel'
CARRIER = f'{FPGA}.AmcCarrierCore'
AXI_VERSION = f'{CARRIER}.AxiVersion'
FPGA_VERSION = f'{AXI_VERSION}.FpgaVersion'
BUILD_STAMP = f'{AXI_VERSION}.BuildStamp'
GIT_HASH = f'{AXI_VERSION}.GitHash'
UPTIME = f'{AXI_VERSION}.UpTimeCnt'
TIMING_RX_LINK_UP = f'{CARRIER}.AmcCarrierTiming.TimingFrameRx.RxLinkUp'

APP_TOP = f'{FPGA}.AppTop'
APP_CORE = f'{APP_TOP}.AppCore'
STREAM_ENABLE = f'{APP_CORE}.enableStreaming'

# Serial links between the FPGA and the data converters, one block per bay.
# Absent where converter and FPGA share a die and there are no lanes to lock.
JESD_BAY = f'{APP_TOP}.AppTopJesd[{{bay}}]'
JESD_RX_DATA_VALID = f'{JESD_BAY}.JesdRx.DataValid'
JESD_TX_DATA_VALID = f'{JESD_BAY}.JesdTx.DataValid'

# Per-bay RF front end, with its programmable attenuators.
MUX_CORE = f'{APP_CORE}.MicrowaveMuxCore[{{bay}}]'
ATTENUATORS = f'{MUX_CORE}.ATT'
ATTENUATOR_UC = f'{ATTENUATORS}.UC[{{uc}}]'
ATTENUATOR_DC = f'{ATTENUATORS}.DC[{{dc}}]'

# The RTM: flux-ramp generator, slow DACs (bias lines) and the cryocard SPI.
RTM = f'{APP_CORE}.RtmCryoDet'
RAMP_MAX_CNT = f'{RTM}.RampMaxCnt'            # ramp period in 307.2 MHz ticks
ENABLE_RAMP_TRIGGER = f'{RTM}.EnableRampTrigger'
RAMP_START_MODE = f'{RTM}.RampStartMode'      # 0 internal, 1 timing system, 2 external
FLUX_RAMP_GENERATOR = f'{RTM}.RtmSpiSr'
FLUX_RAMP_ENABLE = f'{FLUX_RAMP_GENERATOR}.CfgRegEnaBit'
FAST_SLOW_RST_VALUE = f'{FLUX_RAMP_GENERATOR}.FastSlowRstValue'

# Per-band DSP.
BAND = f'{APP_CORE}.SysgenCryo.Base[{{band}}]'
DIGITIZER_FREQUENCY_MHZ = f'{BAND}.digitizerFrequencyMHz'
NUMBER_SUB_BANDS = f'{BAND}.numberSubBands'
NUMBER_CHANNELS = f'{BAND}.numberChannels'
BAND_CENTER_MHZ = f'{BAND}.bandCenterMHz'
CHANNEL_FREQUENCY_MHZ = f'{BAND}.channelFrequencyMHz'   # processing bandwidth per channel
TONE_FREQUENCY_OFFSET_MHZ = f'{BAND}.toneFrequencyOffsetMHz'
BAND_DELAY_US = f'{BAND}.bandDelayUs'
DSP_ENABLE = f'{BAND}.dspEnable'
FEEDBACK_ENABLE = f'{BAND}.feedbackEnable'
SYNTHESIS_SCALE = f'{BAND}.synthesisScale'
REF_PHASE_DELAY = f'{BAND}.refPhaseDelay'

# Per-band channel arrays. One element per channel, so a single channel is
# reached by index rather than by a name of its own.
CRYO_CHANNELS = f'{BAND}.CryoChannels'
AMPLITUDE_SCALE_ARRAY = f'{CRYO_CHANNELS}.amplitudeScale'
CENTER_FREQUENCY_ARRAY = f'{CRYO_CHANNELS}.centerFrequencyMHz'
FEEDBACK_ENABLE_ARRAY = f'{CRYO_CHANNELS}.feedbackEnable'
ETA_MAG_ARRAY = f'{CRYO_CHANNELS}.etaMag'
ETA_PHASE_ARRAY = f'{CRYO_CHANNELS}.etaPhase'

# The tuning operations attached beside the channel arrays: their parameters,
# the flag that says one is running, the processes that do the work and the
# commands that dispatch them.
OPS = CRYO_CHANNELS
OPS_IN_PROGRESS = f'{OPS}.etaScanInProgress'
OPS_DEBUG = f'{OPS}.debug'
OPS_USE_NEW_GRADIENT_DESCENT = f'{OPS}.UseNewSerialGradientDescent'
OPS_PARAMETERS = {
    'eta_scan.channel': f'{OPS}.etaScanChannel',
    'eta_scan.frequencies': f'{OPS}.etaScanFreqs',
    'eta_scan.results_real': f'{OPS}.etaScanResultsReal',
    'eta_scan.results_imag': f'{OPS}.etaScanResultsImag',
    'eta_scan.delta_f': f'{OPS}.etaScanDelF',
    'eta_scan.max_mag': f'{OPS}.etaScanMaxMag',
    'eta_scan.dwell': f'{OPS}.etaScanDwell',
    'eta_scan.amplitude': f'{OPS}.etaScanAmplitude',
    'eta_scan.averages': f'{OPS}.etaScanAverages',
    'gradient_descent.max_iters': f'{OPS}.gradientDescentMaxIters',
    'gradient_descent.averages': f'{OPS}.gradientDescentAverages',
    'gradient_descent.gain': f'{OPS}.gradientDescentGain',
    'gradient_descent.converge_hz': f'{OPS}.gradientDescentConvergeHz',
    'gradient_descent.step_hz': f'{OPS}.gradientDescentStepHz',
    'gradient_descent.momentum': f'{OPS}.gradientDescentMomentum',
    'gradient_descent.beta': f'{OPS}.gradientDescentBeta',
}
# Both gradient-descent implementations are named, because the command that
# dispatches one reads UseNewSerialGradientDescent to choose between them: a
# caller that polls the wrong one would watch a process that never ran.
OPS_PROCESSES = {
    'gradient_descent': f'{OPS}.SerialGradientDescent',
    'new_gradient_descent': f'{OPS}.NewSerialGradientDescent',
    'eta_scan': f'{OPS}.SerialEtaScan',
    'find_freq': f'{OPS}.SerialFindFreq',
}
OPS_COMMANDS = {
    'load_tune_file': f'{OPS}.loadTuneFile',
    'run_eta_scan': f'{OPS}.runEtaScan',
    'set_amplitude_scales': f'{OPS}.setAmplitudeScales',
    'start_gradient_descent': f'{OPS}.runSerialGradientDescent',
    'start_eta_scan': f'{OPS}.runSerialEtaScan',
    'start_find_freq': f'{OPS}.runSerialFindFreq',
}


# --------------------------------------------------------------------------
# the map
# --------------------------------------------------------------------------

# What this map is called, and the one path whose presence identifies a tree as
# belonging to it: the per-band signal processing every such readout has.
NAME = 'umux'
PROBE = BAND.format(band=0)

# Which indices an indexed scope has is a property of the tree, not of this
# file: each scope lists the path templates whose presence proves an index, and
# the scopes its own templates are nested inside.
SCOPES = {
    'band': ((BAND,), ()),
    'bay': ((MUX_CORE, JESD_BAY), ()),
    'uc': ((ATTENUATOR_UC,), ('bay',)),
    'dc': ((ATTENUATOR_DC,), ('bay',)),
}

# Semantic name pattern -> (register path template, kind). 'v' is read and
# written, 'c' is called, 'p' is started and polled. Kept small on purpose: a
# name is here because a core operation or the compatibility layer reaches it,
# and everything else in the tree is still there under the session's root.
_V, _C, _P = 'value', 'command', 'process'

REGISTERS = {
    # the server and its application block
    'server.ready': (READY, _V),
    'application.configured': (SYSTEM_CONFIGURED, _V),
    'application.configuring': (CONFIGURING_IN_PROGRESS, _V),
    'application.enabled_bays': (ENABLED_BAYS, _V),
    'application.startup_arguments': (STARTUP_ARGUMENTS, _V),
    'application.version': (SMURF_VERSION, _V),
    'application.jesd_status': (JESD_STATUS, _V),
    'ops.setup': (SET_DEFAULTS, _P),
    # firmware identity
    'firmware.version': (FPGA_VERSION, _V),
    'firmware.build_stamp': (BUILD_STAMP, _V),
    'firmware.git_hash': (GIT_HASH, _V),
    'firmware.uptime': (UPTIME, _V),
    # timing and streaming
    'timing.rx_link_up': (TIMING_RX_LINK_UP, _V),
    'stream.enable': (STREAM_ENABLE, _V),
    'stream.downsample.factor': (DOWNSAMPLE_FACTOR, _V),
    'stream.filter.disable': (FILTER_DISABLE, _V),
    'stream.frame_count': (FRAME_COUNT, _V),
    # flux ramp
    'flux_ramp.ramp_max_cnt': (RAMP_MAX_CNT, _V),
    'flux_ramp.enable_trigger': (ENABLE_RAMP_TRIGGER, _V),
    'flux_ramp.start_mode': (RAMP_START_MODE, _V),
    'flux_ramp.enable': (FLUX_RAMP_ENABLE, _V),
    'flux_ramp.fast_slow_rst_value': (FAST_SLOW_RST_VALUE, _V),
    # per band: channelisation, as the firmware reports it
    'band[*].center_mhz': (BAND_CENTER_MHZ, _V),
    'band[*].digitizer_rate_mhz': (DIGITIZER_FREQUENCY_MHZ, _V),
    'band[*].n_subbands': (NUMBER_SUB_BANDS, _V),
    'band[*].n_channels': (NUMBER_CHANNELS, _V),
    'band[*].channel_bandwidth_mhz': (CHANNEL_FREQUENCY_MHZ, _V),
    # per band: signal processing and the tones
    'band[*].delay_us': (BAND_DELAY_US, _V),
    'band[*].dsp.enable': (DSP_ENABLE, _V),
    'band[*].ref_phase_delay': (REF_PHASE_DELAY, _V),
    'band[*].feedback.enable': (FEEDBACK_ENABLE, _V),
    'band[*].feedback.enable_array': (FEEDBACK_ENABLE_ARRAY, _V),
    'band[*].tone.amplitude': (AMPLITUDE_SCALE_ARRAY, _V),
    'band[*].tone.frequency': (CENTER_FREQUENCY_ARRAY, _V),
    'band[*].tone.frequency_offset': (TONE_FREQUENCY_OFFSET_MHZ, _V),
    'band[*].tone.synthesis_scale': (SYNTHESIS_SCALE, _V),
    'band[*].eta.mag': (ETA_MAG_ARRAY, _V),
    'band[*].eta.phase': (ETA_PHASE_ARRAY, _V),
    # per bay
    'bay[*].jesd.rx_data_valid': (JESD_RX_DATA_VALID, _V),
    'bay[*].jesd.tx_data_valid': (JESD_TX_DATA_VALID, _V),
    'bay[*].attenuator.uc[*]': (ATTENUATOR_UC, _V),
    'bay[*].attenuator.dc[*]': (ATTENUATOR_DC, _V),
    # the tuning operations
    'band[*].ops.in_progress': (OPS_IN_PROGRESS, _V),
    'band[*].ops.debug': (OPS_DEBUG, _V),
    'band[*].ops.use_new_gradient_descent': (OPS_USE_NEW_GRADIENT_DESCENT, _V),
}
REGISTERS.update({f'band[*].ops.{name}': (path, _V)
                  for name, path in OPS_PARAMETERS.items()})
REGISTERS.update({f'band[*].ops.{name}': (path, _P)
                  for name, path in OPS_PROCESSES.items()})
REGISTERS.update({f'band[*].ops.{name}': (path, _C)
                  for name, path in OPS_COMMANDS.items()})

# The registers worth reading back to record what a system was left in: the
# data links' lock state, the ramp and its trigger, streaming, and per band the
# delay and whether its signal processing is on. Patterns, expanded over the
# indices the tree turns out to have.
WITNESS = (
    'application.jesd_status',
    'timing.rx_link_up',
    'flux_ramp.ramp_max_cnt',
    'flux_ramp.enable_trigger',
    'flux_ramp.start_mode',
    'stream.enable',
    'bay[*].jesd.rx_data_valid',
    'bay[*].jesd.tx_data_valid',
    'bay[*].attenuator.uc[*]',
    'bay[*].attenuator.dc[*]',
    'band[*].delay_us',
    'band[*].dsp.enable',
)
