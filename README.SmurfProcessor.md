# SMuRF Processor

The SMuRF Processor is a pyrogue device which receives and processed the raw streamed data from the firmware application running in the SMuRF ATCA carrier's FPGA.

This device is formed by a series of processing blocks forming a processing pipeline

## Processing blocks

The processing pipeline is describe in the following diagram:


```
 +-----------------+
 | PreDataEmulator |
 +--------+--------+
          |
  +-------+-------+
  | FrameRxStats  |
  +-------+-------+
          |
 +--------+--------+
 | ChannelMapper   |
 +--------+--------+
          |
   +------+------+
   |  Unwrapper  |
   +-----+-------+
         |
    +----+----+
    |  Filter |
    +----+----+
         |
  +------+------+
  | Downsampler |
  +------+------+
         |
 +-------+-------+
 | Header2Smurf  |
 +-------+-------+
         |
+--------+---------+
| PostDataEmulator |
+--------+---------+
         |
         +------------------------+-----------------------+
         |                        |                       |
   +-----+------+      +----------+---------+       +-----+-------+
   | FileWriter |      | BandPhaseFeedback  |       | Transmitter |
   +------------+      | (x8, one per band) |       | (optional)  |
                       +--------------------+       +-------------+

```

Each module in the diagram perform the following operations:

### DataEmulators

Allows to replace the raw data in the incoming frame with emulated data.

There are two data emulator block ins the processing chain:
- **PreDataEmulator**: it is placed at the beginning of the chain, and it is used to replace the raw data coming from the firmware application before it goes into the processor blocks. It generates data of type `Int16`, to match the data type the firmware application generates.
- **PostDataEmulator**: it is placed at the end of the chain, and it is used to replace the processed data at the output of the processor, before it goes to the FileWriter and the Transmitter. It generates data of type `Int32`, to match the data type the processor generates.

Both emulator blocks provide the same functionality; the only difference is the type of data they produced

This module can be disabled; the incoming frame will just pass through to the next block.

For more details see [here](README.DataEmulator.md).

### FrameRxStats

Get statistics about the received frames from the firmware application, including number of frame received, number of lost frames, and number of received frames out of order.

This module can be disabled; the incoming frame will just pass through to the next block.

### ChannelMapper

Maps data channels from the incoming frame to channels in the out frame, using the `Mask` variable.

The `PayloadSize` variable allows to modify the output frame size:
- If it is set to zero, the the output frame will contain only the number of channels defined in the `Mask` list,
- If it is set to a number greater than the number of channels defined in the `Mask` list, then the output frame will be padded with random data,
- If it is set to a number lower that the number of channels defined in the `Mask` lits, then it will be ignored.


### Unwrapper

Thread the data in the incoming frame as phase, and unwraps it.

This module can be disabled; the incoming frame will just pass through to the next block.

### Filter

Applies a general low pass filter to the data, as described in the following equation:

```
y(n) = gain / a(0) * [ b(0) * x(n) + b(1) * x(n -1) + ... + b(order) * x(n - order)
                                   - a(1) * y(n -1) - ... - a(order) * y(n - order) ]
```

This module can be disabled; the incoming frame will just pass through to the next block.

The default coefficients were generated using this python code:

```
import scipy.signal as signal
b, a = signal.butter(4, 2*63 / 4000.)
```

### Downsampler

Perform a downsampling of the data in the incoming frame, by letting pass only factor of incoming frames.

The modules provides two triggers mode, selectable by the variable `TriggerMode`:
- `Internal`:  It uses an internal frame counter, and it releases a frame each `Factor` number of received frames.
- `Timing (BICEP)`: It uses the information from an external timing system to decide when to release a frame. A new frame is release every time the "External real time clock from timing system" (word 96 in the [Smurf header](https://github.com/slaclab/pysmurf/blob/main/README.SmurfPacket.md)) changes. This is the mode used in by BICEP.

This module can be disabled; the incoming frame will just pass through to the next block.

### Header2Smurf

Adds some SMuRF header information to the incoming frame (see [here](README.SmurfPacket.md) for details). At the moment, the only value inserted is the Unix time.

This module is not accessible from the pyrogue tree, as it doesn't have any configuration or status variables.

### FileWriter

Write the processed data to disk. See [here](README.DataFile.md) for details.

### BandPhaseFeedbck

There are 8 of these devices, one for each 500MHz band. Each device estimates and corrects changes in phase parameters for each band.

The estimator uses the following phase time delay (`tau`) and phase offset (`theta`) model:

```
phase(f) = tau * (2 * pi * f) + theta
```

The estimator assumes that the `tau` and `theta` parameters are constant for all tones in the same 500MHz band, and uses the "least square solution" to find them, by reading the phase of 2 or more tones:

```
tau = sum_i(f_i - f_mean)(p_i - p_mean)/sum_i(f_i - f_mean)^2

theta = y_mean - m * x_mean

where:
    f_i    : frequency points.
    f_mean : mean frequency.
    p_i    : phase points.
    p_mean : mean phase.
```

In order to use these devices, you need to, per each band:
- configure at least 2 fixed tones, and assign them 2 two difference channels in the SMuRF packet,
- give the a list with the channel number and a list with the frequency (in GHz) of the fixed tone tone to this module using the variables `toneChannels` and `toneFrequencies`, and
- enable the device by setting `Disable` to `False`.

If all is configured correctly, you should see the variable `DataValid` set to `True`, and the estimated values of `tau` and `theta` will be presented in the variables `Tau` and `Theta`, respectively.

***Notes:***
- The variable `DataValid` indicates if all the device settings are correct. The conditions are:
  - Both lists `toneChannels` and `toneFrequencies` must have the same size,
  - The incoming SMuRF packets must have a number of channel equal or greater to the maximum channel defined in the `toneChannels` list.
- Each device start disabled by default. It must be manually enabled by setting `Disable` to `False`.
- The variable `Band` indicates which 500MHz band the device is working on, with and index going from 0 to 7.
- The `toneFrequencies` frequency list only accepts frequencies inside the corresponding 500MHz band. The frequencies are expressed in GHz.
- The variable `NumChannels` indicates how many channels the incoming SMuRF packets have.
- The variable `FrameCnt` indicates how many valid packets has been received by the device; and the variable `BadFrameCnt` indicates how many bad frames has been rejected. Both counters can be clear using the `clearCnt` command.

### Transmitter

This is an optional block. It is intended for adding a custom block which will take the processed data and set it to a third party system. See [here](README.CustomDataTransmitter.md) for details.