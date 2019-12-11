# Data Emulator

The data emulator processing block present in the [SmurfProcessor](README.SmurfProcessor.md) allows to replace the raw data in the incoming frame with emulated data.

There are different version of the data emulator, depending on the output data type:
- To emulate the data coming from the firmware application, which generates `Int16` data, the emulator `StreamDataEmulatorI16` is used,
- To emulate the processed data after the SmurfProcessor blocks, which generates `Int32` data, the emulator `StreamDataEmulatorI32` is used,

The user can choose different types of emulated signals to be generated, as well as the following parameters:
- **Signal offset**: it is of the same type as the output data, that is `Int16` or `Int32`.
- **Signal amplitude**: it is of the same type as the output data, but on unsigned version, that is `UInt16` or `UInt32`.
- **Signal period**: it is expressed as the number of incoming frames. It must be greater that `2`. For the the emulator that is placed at the output of the firmware application, this period will be expressed in term of the period of the received frames, which in turn is related to the flux ramp period. On the other hand, for the emulator that is placed at the output of the SmurfProcessor, this period will be expressed in terms of the downsampler periods.

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

A square signal, with a 50% duty cycle, is generated withing the values `-amplitude + offset` and `amplitude + offset`, and defined period.

All channels in the incoming frame are set to the same square signal.

### Sawtooth

A sawtooth signal is generated withing the values `offset` and `amplitude + offset`, and defined period.

All channels in the incoming frame are set to the same sawtooth signal.

### Triangle

A triangle signal is generated withing the values `-amplitude + offset` and `amplitude + offset`, and defined period.

All channels in the incoming frame are set to the same triangle signal.

### Sine

A sine signal is generated with defined peak amplitude and period.

All channels in the incoming frame are set to the same sine signal.

### Drop Frame

When this mode is select, one frame will be dropped on each defined period.
