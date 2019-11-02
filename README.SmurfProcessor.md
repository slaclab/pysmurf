# SMuRF Processor

The SMuRF Processor is a pyrogue device which receives and processed the raw streamed data from the firmware application running in the SMuRF ATCA carrier's FPGA.

This device is formed by a series of processing blocks forming a processing pipeline

## Processing blocks

The processing pipeline is describe in the following diagram:


```
 +---------------+
 | FrameRxStats  |
 +-------+-------+
         |
 +-------+-------+
 | DataEmulator  |
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
        |
        |
        +-------------------+
        |                   |
  +-----+------+      +-----+-------+
  | FileWriter |      | Transmitter |
  +------------+      | (optional)  |
                      +-------------+
```

Each module in the diagram perform the following operations:

### FrameRxStats

Get statistics about the received frames from the firmware application, including number of frame received, number of lost frames, and number of received frames out of order.

This module can be disabled; the incoming frame will just pass through to the next block.

### DataEmulator

Allows to replace the raw data in the incoming frame with emulated data.

This module can be disabled; the incoming frame will just pass through to the next block.

### ChannelMapper

Maps data channels from the incoming frame to channels in the out frame, using the `Mask` variable.

The `PayloadSize` variable allows set the output frame size:
- If it is set to zero, the the output frame will contain only the number of channels defined in the `Mask` list,
- If it is set to a number greater than the number of channels defined in the `Mask` list, then the output frame will be padded with random data,
- If it is set to a number lower that the number of channels defined in the `Mask` lits, then the output frame will only contain the first `PayloadSize` channel of the `Mask` list.


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

### Downsampler

Perform a downsampling of the data in the incoming frame, by letting pass only 1 data point each `Factor` number of point.

This module can be disabled; the incoming frame will just pass through to the next block.

### Header2Smurf

Adds the SMuRF header information to the incoming frame (see [here](README.SmurfPacket.md) for details).

This module is not accessible from the pyrogue tree, as it doesn't have any configuration or status variables.

### FileWriter

Write the processed data to disk. See [here](README.DataFile.md) for details.

### Transmitter

This is an optional block. It is intended for adding a custom block which will take the processed data and set it to a third party system. See [here](README.DataTransmitter.md) for details.