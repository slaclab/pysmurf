#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Registers: microwave-multiplexed readout
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
#    The platforms of this generation share these registers and take them from
#    here; each of them names the firmware it runs and is a module of its own,
#    because what separates them is procedure -- bring-up ordering, what has to
#    be configured and in what sequence -- rather than the paths below. Their
#    trees also differ by omission, one having per-bay data links and an RF front
#    end the other does not, so a name those registers back does not resolve
#    everywhere; a caller sees that as an index-free scope, not a broken name.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

__all__ = ['REGISTERS', 'WITNESS', 'SCOPES', 'BUILD_STAMP']

# --------------------------------------------------------------------------
# register paths, as templates over the indexed scopes
# --------------------------------------------------------------------------

# The root device; the server names its Root "AMCc", so every path starts here.
ROOT = 'AMCc'
READY = f'{ROOT}.Ready'                       # set by the server once its start() completes
SET_DEFAULTS = f'{ROOT}.setDefaults'          # the server's own configuration procedure
SET_DEFAULTS_START = f'{SET_DEFAULTS}.Start'  # the same procedure, as a rogue Process
POLL_ENABLE = f'{ROOT}.enable'                # the tree-wide polling switch
READ_ALL = f'{ROOT}.ReadAll'                  # force one read of every node
SAVE_CONFIG = f'{ROOT}.SaveConfigProcess'
SAVE_CONFIG_MODE = f'{SAVE_CONFIG}.SaveMode'
SAVE_CONFIG_FILE = f'{SAVE_CONFIG}.ConfigFile'
SAVE_CONFIG_DATA_TYPE = f'{SAVE_CONFIG}.DataType'
SAVE_CONFIG_START = f'{SAVE_CONFIG}.Start'
SAVE_CONFIG_RUNNING = f'{SAVE_CONFIG}.Running'
SAVE_CONFIG_MESSAGE = f'{SAVE_CONFIG}.Message'

# The application block the server adds beside the FPGA.
APPLICATION = f'{ROOT}.SmurfApplication'
SYSTEM_CONFIGURED = f'{APPLICATION}.SystemConfigured'
CONFIGURING_IN_PROGRESS = f'{APPLICATION}.ConfiguringInProgress'
ENABLED_BAYS = f'{APPLICATION}.EnabledBays'
STARTUP_ARGUMENTS = f'{APPLICATION}.StartupArguments'
SMURF_VERSION = f'{APPLICATION}.SmurfVersion'
JESD_STATUS = f'{APPLICATION}.JesdStatus'
CHECK_JESD = f'{APPLICATION}.CheckJesd'

ROGUE_VERSION = f'{ROOT}.RogueVersion'

# The data processor: the channel map, the downsampler, the low-pass filter and the
# file writer, in the order a sample passes through them.
PROCESSOR = f'{ROOT}.SmurfProcessor'
CHANNEL_MASK = f'{PROCESSOR}.ChannelMapper.Mask'
PAYLOAD_SIZE = f'{PROCESSOR}.ChannelMapper.PayloadSize'
DOWNSAMPLE_FACTOR = f'{PROCESSOR}.Downsampler.InternalFactor'
DOWNSAMPLE_EXTERNAL_BITMASK = f'{PROCESSOR}.Downsampler.ExternalBitmask'
FILTER_DISABLE = f'{PROCESSOR}.Filter.Disable'
FILTER_A = f'{PROCESSOR}.Filter.A'
FILTER_B = f'{PROCESSOR}.Filter.B'
FILTER_GAIN = f'{PROCESSOR}.Filter.Gain'
FILTER_ORDER = f'{PROCESSOR}.Filter.Order'
FRAME_COUNT = f'{PROCESSOR}.FrameRxStats.FrameCnt'
FRAME_LOSS_COUNT = f'{PROCESSOR}.FrameRxStats.FrameLossCnt'
DOWNSAMPLE_MODE = f'{PROCESSOR}.Downsampler.DownsamplerMode'
FILTER_RESET = f'{PROCESSOR}.Filter.reset'
UNWRAPPER_RESET = f'{PROCESSOR}.Unwrapper.reset'
# The processor's own file writer, distinct from the one beside it below: two writers
# exist on a carrier and each has its own Open and Close.
DATA_FILE_NAME = f'{PROCESSOR}.FileWriter.DataFile'
DATA_FILE_OPEN = f'{PROCESSOR}.FileWriter.Open'
DATA_FILE_CLOSE = f'{PROCESSOR}.FileWriter.Close'

# The stream source, and the second writer beside the processor's own.
STREAM_DATA_SOURCE_ENABLE = f'{ROOT}.StreamDataSource.SourceEnable'

# Where the server deposits a debug capture. Two per bay, and they exist only when the
# server was built with a buffer size for them -- so a tree may legitimately have none.
# Each is a rogue DataReceiver: the frame handler writes Data and then sets Updated, so a
# reader clears Updated, triggers, waits for it to rise, and only then reads Data.
CAPTURE = f'{ROOT}.Stream{{capture}}'
CAPTURE_DATA = f'{CAPTURE}.Data'
CAPTURE_UPDATED = f'{CAPTURE}.Updated'
STREAM_WRITER = f'{ROOT}.streamDataWriter'
STREAM_WRITER_OPEN = f'{STREAM_WRITER}.Open'
STREAM_WRITER_CLOSE = f'{STREAM_WRITER}.Close'
STREAM_WRITER_DATA_FILE = f'{STREAM_WRITER}.DataFile'

# The FPGA.
FPGA = f'{ROOT}.FpgaTopLevel'
CARRIER = f'{FPGA}.AmcCarrierCore'
AXI_VERSION = f'{CARRIER}.AxiVersion'
FPGA_VERSION = f'{AXI_VERSION}.FpgaVersion'
BUILD_STAMP = f'{AXI_VERSION}.BuildStamp'
GIT_HASH = f'{AXI_VERSION}.GitHash'
GIT_HASH_SHORT = f'{AXI_VERSION}.GitHashShort'
UPTIME = f'{AXI_VERSION}.UpTimeCnt'

# The carrier's system monitor: die temperature and the three supply rails.
SYSTEM_MONITOR = f'{CARRIER}.AxiSysMonUltraScale'
FPGA_TEMPERATURE = f'{SYSTEM_MONITOR}.Temperature'
FPGA_VCC_INT = f'{SYSTEM_MONITOR}.VccInt'
FPGA_VCC_AUX = f'{SYSTEM_MONITOR}.VccAux'
FPGA_VCC_BRAM = f'{SYSTEM_MONITOR}.VccBram'

# Where the carrier reports its place in the crate.
CARRIER_BSI = f'{CARRIER}.AmcCarrierBsi'
CRATE_ID = f'{CARRIER_BSI}.CrateId'
SLOT_NUMBER = f'{CARRIER_BSI}.SlotNumber'

# The timing system: the receive link, and the event receiver's two index spaces. A
# channel selects what to match, a trigger shapes the pulse that results, and they are
# counted separately -- so they are separate scopes, and neither is the per-band channel.
TIMING = f'{CARRIER}.AmcCarrierTiming'
TIMING_RX_LINK_UP = f'{TIMING}.TimingFrameRx.RxLinkUp'
EVR = f'{TIMING}.EvrV2CoreTriggers'
EVR_CHANNEL = f'{EVR}.EvrV2ChannelReg[{{evr_channel}}]'
EVR_CHANNEL_ENABLE = f'{EVR_CHANNEL}.EnableReg'
EVR_CHANNEL_DEST_SEL = f'{EVR_CHANNEL}.DestSel'
EVR_CHANNEL_DEST_TYPE = f'{EVR_CHANNEL}.DestType'
EVR_CHANNEL_RATE_SEL = f'{EVR_CHANNEL}.RateSel'
EVR_TRIGGER = f'{EVR}.EvrV2TriggerReg[{{evr_trigger}}]'
EVR_TRIGGER_ENABLE = f'{EVR_TRIGGER}.EnableTrig'
EVR_TRIGGER_WIDTH = f'{EVR_TRIGGER}.Width'

# The RF crossbar that routes the carrier's clock and timing outputs.
CROSSBAR_OUTPUT_CONFIG = f'{CARRIER}.AxiSy56040.OutputConfig[{{output}}]'

# The beam-synchronous acquisition engines and their capture buffers.
BSA_ENGINE = (f'{CARRIER}.AmcCarrierBsa.BsaWaveformEngine[{{engine}}]'
              f'.WaveformEngineBuffers')
BSA_BUFFER_EMPTY = f'{BSA_ENGINE}.Empty[{{buffer}}]'
BSA_BUFFER_START_ADDR = f'{BSA_ENGINE}.StartAddr[{{buffer}}]'
BSA_BUFFER_END_ADDR = f'{BSA_ENGINE}.EndAddr[{{buffer}}]'
BSA_BUFFER_WRITE_ADDR = f'{BSA_ENGINE}.WrAddr[{{buffer}}]'

APP_TOP = f'{FPGA}.AppTop'
APP_CORE = f'{APP_TOP}.AppCore'
STREAM_ENABLE = f'{APP_CORE}.enableStreaming'
FIRMWARE_BAND_MASK = f'{APP_CORE}.BUILD_DSP_G'   # which bands this build was made for
DEBUG_SELECT = f'{APP_CORE}.DebugSelect[{{select}}]'
TUNE_FILE_PATH = f'{APP_CORE}.SysgenCryo.tuneFilePath'

# A word of configuration the timing header carries downstream with each frame; its
# bits are read by the data processor rather than by the firmware.
USER_CONFIG = f'{APP_CORE}.TimingHeader.userConfig[{{user_config}}]'

# The per-bay data acquisition mux, which taps the signal path for a capture, and the
# waveform source that plays a tone file back. Both are indexed by bay on every platform
# of this generation -- unlike the RF front end below.
DAQ_MUX = f'{APP_TOP}.DaqMuxV2[{{bay}}]'
DAQ_ARM_HW_TRIGGER = f'{DAQ_MUX}.ArmHwTrigger'
DAQ_TRIGGER = f'{DAQ_MUX}.TriggerDaq'
DAQ_TRIGGER_HW_ARM = f'{DAQ_MUX}.TriggerHwArm'
DAQ_DATA_BUFFER_SIZE = f'{DAQ_MUX}.DataBufferSize'
DAQ_INPUT_MUX_SEL = f'{DAQ_MUX}.InputMuxSel[{{input}}]'
DAC_SIG_GEN = f'{APP_TOP}.DacSigGen[{{bay}}]'
TONE_FILE_PATH = f'{DAC_SIG_GEN}.CsvFilePath'
LOAD_TONE_FILE = f'{DAC_SIG_GEN}.LoadCsvFile'

# The two per-bay device trees that only a platform with a separate converter board
# carries: the serial links to the converters, and the RF front end itself. Named here
# because the scopes below are enumerated by probing for them, and declared as registers
# by the platform that has them -- a map is a statement of what a platform has, so a
# platform whose converters share the FPGA's die should not claim these at all.
JESD_BAY = f'{APP_TOP}.AppTopJesd[{{bay}}]'
MUX_CORE = f'{APP_CORE}.MicrowaveMuxCore[{{bay}}]'

# The RTM: flux-ramp generator, slow DACs (bias lines) and the cryocard SPI.
RTM = f'{APP_CORE}.RtmCryoDet'
RAMP_MAX_CNT = f'{RTM}.RampMaxCnt'            # ramp period in 307.2 MHz ticks
ENABLE_RAMP_TRIGGER = f'{RTM}.EnableRampTrigger'
RAMP_START_MODE = f'{RTM}.RampStartMode'      # 0 internal, 1 timing system, 2 external
CPLD_RESET = f'{RTM}.CpldReset'
DEBOUNCE_WIDTH = f'{RTM}.DebounceWidth'
TRIGGER_HIGH_CYCLE = f'{RTM}.HighCycle'
TRIGGER_LOW_CYCLE = f'{RTM}.LowCycle'
TRIGGER_PULSE_WIDTH = f'{RTM}.PulseWidth'
FLUX_RAMP_GENERATOR = f'{RTM}.RtmSpiSr'
FLUX_RAMP_ENABLE = f'{FLUX_RAMP_GENERATOR}.CfgRegEnaBit'
FAST_SLOW_RST_VALUE = f'{FLUX_RAMP_GENERATOR}.FastSlowRstValue'
FAST_SLOW_STEP_SIZE = f'{FLUX_RAMP_GENERATOR}.FastSlowStepSize'
FLUX_RAMP_DAC_RAW = f'{FLUX_RAMP_GENERATOR}.LTC1668RawDacData'
FLUX_RAMP_MODE_CONTROL = f'{FLUX_RAMP_GENERATOR}.ModeControl'
FLUX_RAMP_SLOPE = f'{FLUX_RAMP_GENERATOR}.RampSlope'

# The serial link to the cryostat card's microcontroller. Not a register the readout
# reads: a pair of mailboxes, one written with a command word and one read for the
# reply, through which a separate processor is addressed. What travels over it is that
# card's own protocol -- addresses, retries, scalings -- which belongs to the code
# driving it and not here; the map's part is where the two mailboxes are.
SPI_CRYO = f'{RTM}.SpiCryo'
CRYOCARD_WRITE = f'{SPI_CRYO}.write'
CRYOCARD_READ = f'{SPI_CRYO}.read'

# The slow DACs that drive the TES bias lines, as whole-array registers: one element per
# DAC, written together because writing them one at a time heats the cryostat.
RTM_SPI_MAX = f'{RTM}.RtmSpiMax'
RTM_SLOW_DAC_DATA_ARRAY = f'{RTM_SPI_MAX}.TesBiasDacDataRegCh'
RTM_SLOW_DAC_ENABLE_ARRAY = f'{RTM_SPI_MAX}.TesBiasDacCtrlRegCh'

# The same kind of DAC on the same chip, driving the amplifier gate voltages rather than
# the bias lines. Whole-array registers again, and a separate pair because what they
# drive is separate.
AMP_GATE_DAC_DATA_ARRAY = f'{RTM_SPI_MAX}.HemtBiasDacDataRegCh'
AMP_GATE_DAC_ENABLE_ARRAY = f'{RTM_SPI_MAX}.HemtBiasDacCtrlRegCh'

# The lookup-table controller that plays a stored waveform out of the slow DACs, for a
# ramp the flux-ramp generator cannot shape.
RTM_LUT = f'{RTM}.LutCtrl'
RTM_LUT_CONTROL = f'{RTM_LUT}.Ctrl'
RTM_LUT_CONTINUOUS = f'{RTM_LUT_CONTROL}.Continuous'
RTM_LUT_ENABLE = f'{RTM_LUT_CONTROL}.EnableCh'
RTM_LUT_TIMER_SIZE = f'{RTM_LUT_CONTROL}.TimerSize'
RTM_LUT_DAC_ADDRESS = f'{RTM_LUT_CONTROL}.DacAxilAddr[{{lut_dac}}]'
RTM_LUT_TABLE = f'{RTM_LUT}.Lut[{{lut}}].MemArray'

RESET_RTM = f'{RTM}.resetRtm'

# Per-band DSP.
BAND = f'{APP_CORE}.SysgenCryo.Base[{{band}}]'
DIGITIZER_FREQUENCY_MHZ = f'{BAND}.digitizerFrequencyMHz'
NUMBER_SUB_BANDS = f'{BAND}.numberSubBands'
NUMBER_CHANNELS = f'{BAND}.numberChannels'
BAND_CENTER_MHZ = f'{BAND}.bandCenterMHz'
CHANNEL_FREQUENCY_MHZ = f'{BAND}.channelFrequencyMHz'   # processing bandwidth per channel
TONE_FREQUENCY_OFFSET_MHZ = f'{BAND}.toneFrequencyOffsetMHz'
DECIMATION = f'{BAND}.decimation'
ANALYSIS_SCALE = f'{BAND}.analysisScale'
TONE_SCALE = f'{BAND}.toneScale'
FREQUENCY_ERROR_ARRAY = f'{BAND}.CryoChannels.frequencyError'
BAND_DELAY_US = f'{BAND}.bandDelayUs'
DSP_ENABLE = f'{BAND}.dspEnable'
FEEDBACK_ENABLE = f'{BAND}.feedbackEnable'
SYNTHESIS_SCALE = f'{BAND}.synthesisScale'
REF_PHASE_DELAY = f'{BAND}.refPhaseDelay'
REF_PHASE_DELAY_FINE = f'{BAND}.refPhaseDelayFine'
TRIGGER_RESET_DELAY = f'{BAND}.trigRstDly'

# The feedback loop that holds a tone on its resonator: where in the ramp it acts, how
# hard, and how far it is allowed to move.
FEEDBACK_START = f'{BAND}.feedbackStart'
FEEDBACK_END = f'{BAND}.feedbackEnd'
FEEDBACK_GAIN = f'{BAND}.feedbackGain'
FEEDBACK_LIMIT = f'{BAND}.feedbackLimit'
FEEDBACK_POLARITY = f'{BAND}.feedbackPolarity'

# The tracking loop that follows a resonator as the flux ramp moves it.
LMS_DELAY = f'{BAND}.lmsDelay'
LMS_ENABLE1 = f'{BAND}.lmsEnable1'
LMS_ENABLE2 = f'{BAND}.lmsEnable2'
LMS_ENABLE3 = f'{BAND}.lmsEnable3'
LMS_FREQ_HZ = f'{BAND}.lmsFreqHz'
LMS_GAIN = f'{BAND}.lmsGain'

# Per-band signal processing, and what the DAC is told to drive.
FILTER_ALPHA = f'{BAND}.filterAlpha'
IQ_SWAP_IN = f'{BAND}.iqSwapIn'
IQ_SWAP_OUT = f'{BAND}.iqSwapOut'
NOISE_SELECT = f'{BAND}.noiseSelect'
WAVEFORM_SELECT = f'{BAND}.waveformSelect'

# What a band streams out, and the single-channel debug readout.
IQ_STREAM_ENABLE = f'{BAND}.iqStreamEnable'
RF_IQ_STREAM_ENABLE = f'{BAND}.rfIQStreamEnable'
READOUT_CHANNEL_SELECT = f'{BAND}.readoutChannelSelect'
SINGLE_CHANNEL_READOUT = f'{BAND}.singleChannelReadout'
SINGLE_CHANNEL_READOUT_OPT2 = f'{BAND}.singleChannelReadoutOpt2'

# Per-band channel arrays. One element per channel, so a single channel is
# reached by index rather than by a name of its own.
CRYO_CHANNELS = f'{BAND}.CryoChannels'
AMPLITUDE_SCALE_ARRAY = f'{CRYO_CHANNELS}.amplitudeScale'
CENTER_FREQUENCY_ARRAY = f'{CRYO_CHANNELS}.centerFrequencyMHz'
FEEDBACK_ENABLE_ARRAY = f'{CRYO_CHANNELS}.feedbackEnable'
ETA_MAG_ARRAY = f'{CRYO_CHANNELS}.etaMag'
ETA_PHASE_ARRAY = f'{CRYO_CHANNELS}.etaPhase'
LOOP_FILTER_OUTPUT_ARRAY = f'{CRYO_CHANNELS}.loopFilterOutput'

# One channel of a band, reached by index. The same quantities as the arrays above, for
# the cases that touch a single tone rather than all of them at once.
CHANNEL = f'{CRYO_CHANNELS}.CryoChannel[{{channel}}]'
CHANNEL_AMPLITUDE_SCALE = f'{CHANNEL}.amplitudeScale'
CHANNEL_CENTER_FREQUENCY = f'{CHANNEL}.centerFrequencyMHz'
CHANNEL_ETA_MAG_SCALED = f'{CHANNEL}.etaMagScaled'
CHANNEL_ETA_PHASE_DEGREE = f'{CHANNEL}.etaPhaseDegree'
CHANNEL_FEEDBACK_ENABLE = f'{CHANNEL}.feedbackEnable'
CHANNEL_FREQUENCY_ERROR = f'{CHANNEL}.frequencyErrorMHz'

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

# Which indices an indexed scope has is a property of the tree, not of this
# file: each scope lists the path templates whose presence proves an index, and
# the scopes its own templates are nested inside.
SCOPES = {
    'band': ((BAND,), ()),
    # A bay is proved by any of the per-bay device trees, because which of them a
    # platform has differs: the acquisition mux is per-bay everywhere, while the front
    # end and the data links belong to a platform with a separate converter board. So
    # the scope is shared and the registers under it are not.
    'bay': ((MUX_CORE, JESD_BAY, DAQ_MUX), ()),
    'channel': ((CHANNEL,), ('band',)),
    'input': ((DAQ_INPUT_MUX_SEL,), ('bay',)),
    # The event receiver counts what it matches and what it emits separately, so these
    # are two scopes and neither is the per-band channel above. A tree with one and not
    # the other is a firmware that wired up fewer of one, which is a fact about the
    # tree rather than a contradiction.
    'evr_channel': ((EVR_CHANNEL_ENABLE,), ()),
    'evr_trigger': ((EVR_TRIGGER_ENABLE,), ()),
    'output': ((CROSSBAR_OUTPUT_CONFIG,), ()),
    'select': ((DEBUG_SELECT,), ()),
    'capture': ((CAPTURE_DATA,), ()),
    'user_config': ((USER_CONFIG,), ()),
    'engine': ((BSA_ENGINE,), ()),
    'buffer': ((BSA_BUFFER_EMPTY,), ('engine',)),
    # The RTM waveform controller's lookup tables, and the DACs it addresses them
    # through. Its DACs are on the RTM and are not the converters a bay carries, so they
    # are a scope of their own: a scope name has to mean one thing.
    'lut': ((RTM_LUT_TABLE,), ()),
    'lut_dac': ((RTM_LUT_DAC_ADDRESS,), ()),
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
    'ops.setup.start': (SET_DEFAULTS_START, _C),
    'poll_enable': (POLL_ENABLE, _V),
    'read_all': (READ_ALL, _C),
    'save_config.mode': (SAVE_CONFIG_MODE, _V),
    'save_config.file': (SAVE_CONFIG_FILE, _V),
    'save_config.data_type': (SAVE_CONFIG_DATA_TYPE, _V),
    'save_config.start': (SAVE_CONFIG_START, _C),
    'save_config.running': (SAVE_CONFIG_RUNNING, _V),
    'save_config.message': (SAVE_CONFIG_MESSAGE, _V),
    'application.check_jesd': (CHECK_JESD, _C),
    'carrier.fpga.temperature': (FPGA_TEMPERATURE, _V),
    'carrier.fpga.vcc_int': (FPGA_VCC_INT, _V),
    'carrier.fpga.vcc_aux': (FPGA_VCC_AUX, _V),
    'carrier.fpga.vcc_bram': (FPGA_VCC_BRAM, _V),
    'carrier.bsa.engine[*].buffer[*].start_address': (BSA_BUFFER_START_ADDR, _V),
    'carrier.bsa.engine[*].buffer[*].end_address': (BSA_BUFFER_END_ADDR, _V),
    'carrier.bsa.engine[*].buffer[*].write_address': (BSA_BUFFER_WRITE_ADDR, _V),
    'band[*].decimation': (DECIMATION, _V),
    'band[*].analysis_scale': (ANALYSIS_SCALE, _V),
    'band[*].tone.scale': (TONE_SCALE, _V),
    'band[*].frequency_error': (FREQUENCY_ERROR_ARRAY, _V),
    'band[*].channel[*].frequency_error': (CHANNEL_FREQUENCY_ERROR, _V),
    'bay[*].tone_file.load': (LOAD_TONE_FILE, _C),
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
    # the tuning operations
    'band[*].ops.in_progress': (OPS_IN_PROGRESS, _V),
    'band[*].ops.debug': (OPS_DEBUG, _V),
    'band[*].ops.use_new_gradient_descent': (OPS_USE_NEW_GRADIENT_DESCENT, _V),
    # the server's own nodes, and what it writes captured data to
    'server.rogue_version': (ROGUE_VERSION, _V),
    'stream.data_source_enable': (STREAM_DATA_SOURCE_ENABLE, _V),
    'timing.user_config[*]': (USER_CONFIG, _V),
    'stream.capture[*].data': (CAPTURE_DATA, _V),
    'stream.capture[*].updated': (CAPTURE_UPDATED, _V),
    'stream.data_file.open': (DATA_FILE_OPEN, _C),
    'stream.data_file.close': (DATA_FILE_CLOSE, _C),
    'stream.frame_loss_count': (FRAME_LOSS_COUNT, _V),
    'stream.downsample.mode': (DOWNSAMPLE_MODE, _V),
    'stream.filter.reset': (FILTER_RESET, _C),
    'stream.unwrapper.reset': (UNWRAPPER_RESET, _C),
    'stream.writer.open': (STREAM_WRITER_OPEN, _C),
    'stream.writer.close': (STREAM_WRITER_CLOSE, _C),
    'stream.writer.data_file': (STREAM_WRITER_DATA_FILE, _V),
    # the data processor
    'stream.channel_mask': (CHANNEL_MASK, _V),
    'stream.payload_size': (PAYLOAD_SIZE, _V),
    'stream.downsample.external_bitmask': (DOWNSAMPLE_EXTERNAL_BITMASK, _V),
    'stream.file.name': (DATA_FILE_NAME, _V),
    'stream.filter.a': (FILTER_A, _V),
    'stream.filter.b': (FILTER_B, _V),
    'stream.filter.gain': (FILTER_GAIN, _V),
    'stream.filter.order': (FILTER_ORDER, _V),
    # the carrier's place in the crate, and the rest of the firmware's identity
    'carrier.crate_id': (CRATE_ID, _V),
    'carrier.slot_number': (SLOT_NUMBER, _V),
    'firmware.git_hash_short': (GIT_HASH_SHORT, _V),
    'firmware.band_mask': (FIRMWARE_BAND_MASK, _V),
    # the timing system's event receiver
    'timing.evr_channel[*].enable': (EVR_CHANNEL_ENABLE, _V),
    'timing.evr_channel[*].dest_sel': (EVR_CHANNEL_DEST_SEL, _V),
    'timing.evr_channel[*].dest_type': (EVR_CHANNEL_DEST_TYPE, _V),
    'timing.evr_channel[*].rate_select': (EVR_CHANNEL_RATE_SEL, _V),
    'timing.evr_trigger[*].enable': (EVR_TRIGGER_ENABLE, _V),
    'timing.evr_trigger[*].width': (EVR_TRIGGER_WIDTH, _V),
    # the clock and timing crossbar, the debug mux, and the capture engines
    'crossbar.output[*].config': (CROSSBAR_OUTPUT_CONFIG, _V),
    'debug.select[*]': (DEBUG_SELECT, _V),
    'bsa.engine[*].buffer[*].empty': (BSA_BUFFER_EMPTY, _V),
    # per bay: the acquisition mux and the waveform source
    'bay[*].daq.arm_hw_trigger': (DAQ_ARM_HW_TRIGGER, _C),
    'bay[*].daq.trigger': (DAQ_TRIGGER, _C),
    'bay[*].daq.trigger_hw_arm': (DAQ_TRIGGER_HW_ARM, _V),
    'bay[*].daq.data_buffer_size': (DAQ_DATA_BUFFER_SIZE, _V),
    'bay[*].daq.input[*].mux_sel': (DAQ_INPUT_MUX_SEL, _V),
    'bay[*].tone_file_path': (TONE_FILE_PATH, _V),
    # the RTM: the ramp's shape, the trigger it is driven by, and the slow DACs
    'flux_ramp.dac_raw': (FLUX_RAMP_DAC_RAW, _V),
    'flux_ramp.fast_slow_step_size': (FAST_SLOW_STEP_SIZE, _V),
    'flux_ramp.mode_control': (FLUX_RAMP_MODE_CONTROL, _V),
    'flux_ramp.ramp_slope': (FLUX_RAMP_SLOPE, _V),
    'rtm.slow_dac.data_array': (RTM_SLOW_DAC_DATA_ARRAY, _V),
    'rtm.slow_dac.enable_array': (RTM_SLOW_DAC_ENABLE_ARRAY, _V),
    'rtm.cryocard.write': (CRYOCARD_WRITE, _V),
    'rtm.cryocard.read': (CRYOCARD_READ, _V),
    'rtm.reset': (RESET_RTM, _C),
    'rtm.amp_gate_dac.data_array': (AMP_GATE_DAC_DATA_ARRAY, _V),
    'rtm.amp_gate_dac.enable_array': (AMP_GATE_DAC_ENABLE_ARRAY, _V),
    'rtm.waveform.continuous': (RTM_LUT_CONTINUOUS, _V),
    'rtm.waveform.enable': (RTM_LUT_ENABLE, _V),
    'rtm.waveform.timer_size': (RTM_LUT_TIMER_SIZE, _V),
    'rtm.waveform.lut_dac[*].address': (RTM_LUT_DAC_ADDRESS, _V),
    'rtm.waveform.lut[*].table': (RTM_LUT_TABLE, _V),
    'rtm.cpld_reset': (CPLD_RESET, _V),
    'rtm.debounce_width': (DEBOUNCE_WIDTH, _V),
    'rtm.trigger.high_cycle': (TRIGGER_HIGH_CYCLE, _V),
    'rtm.trigger.low_cycle': (TRIGGER_LOW_CYCLE, _V),
    'rtm.trigger.pulse_width': (TRIGGER_PULSE_WIDTH, _V),
    # per band: the feedback loop
    'band[*].feedback.start': (FEEDBACK_START, _V),
    'band[*].feedback.end': (FEEDBACK_END, _V),
    'band[*].feedback.gain': (FEEDBACK_GAIN, _V),
    'band[*].feedback.limit': (FEEDBACK_LIMIT, _V),
    'band[*].feedback.polarity': (FEEDBACK_POLARITY, _V),
    'band[*].feedback.loop_filter_output': (LOOP_FILTER_OUTPUT_ARRAY, _V),
    # per band: the tracking loop
    'band[*].lms.delay': (LMS_DELAY, _V),
    'band[*].lms.enable1': (LMS_ENABLE1, _V),
    'band[*].lms.enable2': (LMS_ENABLE2, _V),
    'band[*].lms.enable3': (LMS_ENABLE3, _V),
    'band[*].lms.freq_hz': (LMS_FREQ_HZ, _V),
    'band[*].lms.gain': (LMS_GAIN, _V),
    # per band: signal processing, the delays, and what is streamed
    'band[*].dsp.filter_alpha': (FILTER_ALPHA, _V),
    'band[*].dsp.iq_swap_in': (IQ_SWAP_IN, _V),
    'band[*].dsp.iq_swap_out': (IQ_SWAP_OUT, _V),
    'band[*].dsp.noise_select': (NOISE_SELECT, _V),
    'band[*].dsp.waveform_select': (WAVEFORM_SELECT, _V),
    'band[*].ref_phase_delay_fine': (REF_PHASE_DELAY_FINE, _V),
    'band[*].trigger_reset_delay': (TRIGGER_RESET_DELAY, _V),
    'band[*].stream.iq_enable': (IQ_STREAM_ENABLE, _V),
    'band[*].stream.rf_iq_enable': (RF_IQ_STREAM_ENABLE, _V),
    'band[*].readout.channel_select': (READOUT_CHANNEL_SELECT, _V),
    'band[*].readout.single_channel': (SINGLE_CHANNEL_READOUT, _V),
    'band[*].readout.single_channel_opt2': (SINGLE_CHANNEL_READOUT_OPT2, _V),
    'tune_file_path': (TUNE_FILE_PATH, _V),
    # per band per channel
    'band[*].channel[*].tone.amplitude': (CHANNEL_AMPLITUDE_SCALE, _V),
    'band[*].channel[*].tone.frequency': (CHANNEL_CENTER_FREQUENCY, _V),
    'band[*].channel[*].eta.mag_scaled': (CHANNEL_ETA_MAG_SCALED, _V),
    'band[*].channel[*].eta.phase_degree': (CHANNEL_ETA_PHASE_DEGREE, _V),
    'band[*].channel[*].feedback.enable': (CHANNEL_FEEDBACK_ENABLE, _V),
}
REGISTERS.update({f'band[*].ops.{name}': (path, _V)
                  for name, path in OPS_PARAMETERS.items()})
REGISTERS.update({f'band[*].ops.{name}': (path, _P)
                  for name, path in OPS_PROCESSES.items()})
REGISTERS.update({f'band[*].ops.{name}': (path, _C)
                  for name, path in OPS_COMMANDS.items()})

# The registers worth reading back to record what a system was left in: the ramp and
# its trigger, streaming, and per band the delay and whether its signal processing is
# on. Patterns, expanded over the indices the tree turns out to have. A platform adds
# the witnesses of the hardware only it has -- the data links' lock state and the
# attenuator settings are a carrier's, and are in that platform's own map.
WITNESS = (
    'application.jesd_status',
    'timing.rx_link_up',
    'flux_ramp.ramp_max_cnt',
    'flux_ramp.enable_trigger',
    'flux_ramp.start_mode',
    'stream.enable',
    'band[*].delay_us',
    'band[*].dsp.enable',
)
