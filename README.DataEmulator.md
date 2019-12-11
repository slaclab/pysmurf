# Data Emulator

The data emulator processing block present in the [SmurfProcessor](README.SmurfProcessor.md) allows to replace the raw data in the incoming frame with emulated data.

The user can choose different types of emulated signals to be generated, as well as the following parameters:
- Signal offset, with valid range [−32768, +32767].
- Signal amplitude, with valid range [0, +65535].
- Signal period, expressed as multiples of the flux ramp period.

The meaning of each of these parameters for each type of signal is described in the following section.

This module can be disabled; the incoming frame will just pass through to the next block.

## Type of emulated signals

The following type of signals can be generated:

### Zeros

All channel in the incoming frame are set to zero.

### ChannelNumber

All channels in the incoming frame are set to its channel number.

### Random

Each channel is set to an uniformly distributed random number on the interval [`-amplitude + offset`, `amplitude + offset`). Each channel will have a different random number.

### Square

A square signal is generated withing the values `-amplitude + offset` and `amplitude + offset`, and defined period. The period is expressed as multiples of the flux ramp period.

All channels in the incoming frame are set to the same square signal.

### Sawtooth

A sawtooth signal is generated withing the values `offset` and `amplitude + offset`, and defined period. The period is expressed as multiples of the flux ramp period.

All channels in the incoming frame are set to the same sawtooth signal.

### Triangle

A triangle signal is generated withing the values `-amplitude + offset` and `amplitude + offset`, and defined period. The period is expressed as multiples of the flux ramp period.

All channels in the incoming frame are set to the same triangle signal.

### Sine

A sine signal is generated with defined peak amplitude and period. The period is expressed as multiples of the flux ramp period.

All channels in the incoming frame are set to the same sine signal.

### Drop Frame

When this mode is select, one frame will be dropped on each defined period. The period is expressed as multiples of the flux ramp period.