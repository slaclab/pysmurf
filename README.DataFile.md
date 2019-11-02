# Data file

The SMuRF processor device includes a Rogue's DataWriter device which allows to write the streamed data to disk after processing (that is, after channel mapping, unwrapping, filtering and downsamplig).

Configuration data (a.k.a. metadata) from the pyrogue tree is also written to the same data file, interleaved with the processed data.

## How to configure the SmtremWriter device

The DataWriter device, can be controlled with the following variables (or its respective EPICS PV):

Pyrogue variable                               | Mode | Description
-----------------------------------------------|-------------
AMCc:StreamProcessor:FileWriter:DataFile       | RW   | Full path of the data file
AMCc:StreamProcessor:FileWriter:Open           | WO   | Open data file
AMCc:StreamProcessor:FileWriter:Close          | WO   | Close data file
AMCc:StreamProcessor:FileWriter:IsOpen         | RO   | Data file is open
AMCc:StreamProcessor:FileWriter:BufferSize     | RO   | File buffering size. Enables caching of data before call to file system
AMCc:StreamProcessor:FileWriter:MaxFileSize    | RW   | Maximum size for an individual file. Setting to a non zero splits the run data into multiple files
AMCc:StreamProcessor:FileWriter:CurrentSize    | RO   | Size of current data files(s) for current open session in bytes
AMCc:StreamProcessor:FileWriter:TotalSize      | RO   | Size of all data sub-files(s) for current open session in bytes
AMCc:StreamProcessor:FileWriter:FrameCount     | RO   | Frame in data file(s) for current open session in bytes
AMCc:StreamProcessor:FileWriter:AutoName       | WO   | Auto create data file name using data and time

## Data file format

The data file is a series of banks. Each bank is preceded by 2 32-bit word header to indicate bank information:

Header bits | Description
------------|-----------
[31:0]      | Length of the data block (including the next header 32-bit word)
[31:24]     | Chanel ID
[23:16]     | Frame error
[15:0]      | Frame flags

## DataWriter channels

Two different channels are used for data and metadata:
- Channel 0 : Processed data
- Channel 1 : Metadata

This allows the reader code to filter the data in the datafile.

## Processed data structure

Each bank which correspond to a processed data frame (that is, bank with Channel ID = 0) contains (see [here](README.SmurfPacket.md) for details):
- A 128-bytes SMuRF header
- A payload with the processed data (as signed 32-bit words)

The payload size can be controlled using the variable **AMCc:StreamProcessor:ChannelMapper:PayloadSize**. Is this value is set to 0, then the payload size will be dynamic, adjusting to contained only the number of mapped channels.

The number of channel in the SMuRF header will indicate to the reader code the number of valid processed data word in the payload. While, the DataWriter header's length value, will allow the reader code know the payload size. When the **PayloadSize** is set to 0, then the payload size will be equal to the number of channels.

## Example

The following diagram shows an example of the structure of a file with 2 banks, the first one for channel ID `0` (i.e. SMuRF data) of length `L0`, followed by a bank for channel ID `1` (i.e. medatada) of length `L1`:

```
|<-------- DataWriter header --------->|<---------------------------- SMURF PACKET --------------------------->|<-------- DataWriter header --------->|<--- METADATA PACKET ---->|
|                                      |<---- SMURF HEADER ---->|<---------------- SMURF DATA ---------------->|                                      |                          |
+--------------+-----------------------+------------------------+----------------------------------------------+--------------+-----------------------+--------------------------+
| Bank Length  | CH ID | Error | Flags |  .... | # ch | ....    | Word 0 | Word 1 | ... | Word N |   PADDING   | Bank Length  | CH ID | Error | Flags |         METADATA         |
+--------------+-----------------------+------------------------+----------------------------------------------+--------------+-----------------------+--------------------------+
     = L0      | = 0                              = N           |<----------- N words ---------->|             |    = L1      |  = 1                                             |
               |<------------------------------------------ L0 bytes ----------------------------------------->|              |<------------------- L1 Bytes ------------------->|
```