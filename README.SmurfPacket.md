# The SMuRF Packet

## Description

The SMuRF packet is formed by a header and a payload:

### SMuRF packet header

The header is 128-byte long and contains the following information:

| Starting word  | Offset [byte]     | Length [bytes]    | Function                                      | Description
|----------------|-------------------|-------------------|-----------------------------------------------|---------------------------
|        0       |       0           |         1         | Protocol version                              | Number to describe protocol version.  0x01 to start
|                |       1           |         1         | ATCA crate                                    | Crate number starting from 1 (0 reserved for testing)
|                |       2           |         1         | ATCA card                                     | Slot Number starting from 1 (0 reserved for testing).
|                |       3           |         1         | Smurf timing Configuration                    | (clock lock / unlock), (external ref, fiber timing ref)
|                |       4           |         4         | Number of channels of data                    | 32 bit word giving number channels of following data
|       1        |       8           |        40         | TES DAC 0-15                                  | TES DAC values. 16X 20 bit in 10X32 bit words, or 5X 65 bit
|       6        |      48           |         8         | 64 bit unix time                              | 64 bit unix time nanoseconds
|       7        |      56           |         4         | Flux ramp increment                           | signed 32 bit integer for increment
|                |      60           |         4         | Flux ramp offset                              | signed 32 it integer for offset
|       8        |      64           |         4         | Counter 0, since last 1Hz marker              | 32 bit counter since last 1Hz marker
|                |      68           |         4         | Counter 1, since last external input          | 32 bit counter since last external input
|                |      72           |         8         | Counter 2,  64 bit timing system counter      | 64 bit timestamp
|      10        |      80           |         4         | Averaging reset bits                          | up to 32 bits of average reset from timing system
|                |      84           |         4         | Frame counter                                 | Locally genreate frame counter 32 bit
|      11        |      88           |         4         | TES relay settings                            | TES and flux ramp relays, 17bits in use now
|      12        |      96           |         5         | External real time clock from timing system   | Syncword from mce for mce based systems (40 bit including header)
|      13        |     104           |         1         | Control field                                 | SEE TABLE BELOW WITH BIT DESCRIPTIONS...
|                |     105           |         1         | Test parameters                               |
|      14        |     112           |         2         | Number of rows                                | MCE header value (max 255)  (defaluts to 33 if 0)
|                |     114           |         2         | Number of rows reported                       | MCE header value (defaults to numb rows if 0)
|      15        |     120           |         2         | Row length                                    | MCE header value
|                |     122           |         2         | Data rate                                     | MCE header value

#### Control Field (byte offset 104)

The bits in the control filed (byte offset 104) are:

|  bit  | Description
|-------|-------------------------------------------
|   0   | Clear average and unwrap
|   1   | Disable stream to MCE
|   2   | Disable file write
|   3   | Set to read configuration file each cycle
|  4-7  | Test mode:  0=normal, 1-15=test modes (see below)

The test mode selected by bits 4-7 are

| Mode   | Description
|--------|---------------------
|    0   | normal data
|    1   | All zeros for smurf data
|    2   | Smurf channels value = channel #
|    3   | smurf data channels toggle between -20000, and 20000 on syncword = multiple of 1000
|    4   | Each smurf channel independantly uniformly spaced random numbers, peak to peak 1000 (before filter)
|    5   | Sine waves, frequency = (param+1) * flux_ramp_rate / 2^16
|    6   |
|    7   |
|    8   | MCE output data all set to zero
|    9   | MCE output data = mce channel number
|   10   | MCE output ramps with syncbox signal * param
|   11   |
|   12   |
|   13   |
|   14   | Break checksum on 1/1000 mce frames
|   15   | Force framedrops  on 1/1000 mcde frames


### SMuRF packet payload

The payload contains signed 32-bit averaged data points. The number of points is defined in the header's `Number of channels of data` word.
