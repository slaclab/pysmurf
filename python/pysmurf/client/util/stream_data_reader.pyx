# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False

"""
Cython-optimized stream data reader for pysmurf
This module provides high-performance data reading for SMuRF stream data.
All file I/O and binary parsing is done in C for maximum performance.
"""

import numpy as np
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from libc.math cimport pi
from libc.stdio cimport (FILE, fopen, fclose, fread, fseek, ftell, 
                         SEEK_END, SEEK_SET, SEEK_CUR, feof)
from libc.string cimport memcpy
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t, int32_t

cnp.import_array()

# Frame Format Constants
cdef int SMURF_HEADER_SIZE = 128
cdef int ROGUE_HEADER_SIZE = 8
cdef int SMURF_CHANNEL_SIZE = 4
SMURF_DATA_DTYPE = np.dtype(np.int32)

# SMuRF Header as NumPy structured dtype
SMURF_HEADER_DTYPE = np.dtype([
    ('protocol_version', np.uint8),       # Offset 0
    ('crate_id', np.uint8),               # Offset 1
    ('slot_number', np.uint8),            # Offset 2
    ('timing_cond', np.uint8),            # Offset 3
    ('number_of_channels', np.uint32),    # Offset 4
    ('tes_bias', np.uint8, 40),           # Offset 8-47
    ('timestamp', np.uint64),             # Offset 48
    ('flux_ramp_increment', np.int32),    # Offset 56
    ('flux_ramp_offset', np.int32),       # Offset 60
    ('counter_0', np.uint32),             # Offset 64
    ('counter_1', np.uint32),             # Offset 68
    ('counter_2', np.uint64),             # Offset 72
    ('reset_bits', np.uint32),            # Offset 80
    ('frame_counter', np.uint32),         # Offset 84
    ('tes_relays_config', np.uint32),     # Offset 88
    ('_padding_92_95', np.uint8, 4),      # Offset 92 (unused)
    ('external_time_raw', np.uint64),     # Offset 96 (only 5 bytes used)
    ('control_field', np.uint8),          # Offset 104
    ('test_params', np.uint8),            # Offset 105
    ('_padding_106_111', np.uint8, 6),    # Offset 106 (unused)
    ('num_rows', np.uint16),              # Offset 112
    ('num_rows_reported', np.uint16),     # Offset 114
    ('_padding_116_119', np.uint8, 4),    # Offset 116 (unused)
    ('row_length', np.uint16),            # Offset 120
    ('data_rate', np.uint16),             # Offset 122
    ('_padding_124_127', np.uint8, 4),    # Offset 124 (unused)
])

# Validate dtype size
assert SMURF_HEADER_DTYPE.itemsize == 128, \
    f"SMURF_HEADER_DTYPE must be 128 bytes, got {SMURF_HEADER_DTYPE.itemsize}"

# C structures for binary parsing
cdef struct RogueHeader:
    uint32_t size
    uint16_t flags
    uint8_t error
    uint8_t channel


cdef inline void parse_rogue_header(uint8_t* data, RogueHeader* header) noexcept nogil:
    """Parse Rogue header from raw bytes - inline for speed"""
    header[0] = (<RogueHeader*>data)[0]

cdef class FastSmurfReader:
    """
    Fast Cython-based SMURF stream data reader
    Performs all file I/O and parsing in C for maximum speed
    """
    cdef FILE* file_ptr
    cdef bytes filename
    cdef long file_size
    cdef bint is_rogue
    cdef long records_read
    cdef uint8_t header_buffer[128]
    cdef uint8_t rogue_buffer[8]

    def __cinit__(self, str filename, bint is_rogue=True):
        self.filename = filename.encode('utf-8')
        self.is_rogue = is_rogue
        self.records_read = 0
        self.file_ptr = NULL

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    cdef void open(self) except *:
        """Open the file and get size"""
        self.file_ptr = fopen(self.filename, "rb")
        if self.file_ptr == NULL:
            raise IOError(f"Cannot open file: {self.filename.decode('utf-8')}")

        # Get file size
        fseek(self.file_ptr, 0, SEEK_END)
        self.file_size = ftell(self.file_ptr)
        fseek(self.file_ptr, 0, SEEK_SET)

    cdef void close(self) noexcept nogil:
        """Close the file"""
        if self.file_ptr != NULL:
            fclose(self.file_ptr)
            self.file_ptr = NULL

    def read_all_records(self, channel=None):
        """
        Read all records from file and return as numpy arrays
        This is the main high-performance function

        Arguments
        ---------
        channel : array-like or None
            Channels to read (if None, reads all)

        Returns
        -------
        timestamps : ndarray (uint64)
            Array of timestamps extracted from headers
        data : ndarray (int32, shape=[n_records, n_channels])
            2D array of channel data
        headers : ndarray (structured array)
            Structured array of SMuRF headers with dtype SMURF_HEADER_DTYPE
            Access fields like: headers['timestamp'], headers['frame_counter']
            Individual headers: headers[i] returns a single header record
        """
        # Pre-allocate with estimates
        cdef bytearray header_bytes = bytearray()
        cdef list data_list = []

        cdef RogueHeader rogue_hdr
        cdef cnp.ndarray[cnp.int32_t, ndim=1] channel_data
        cdef size_t bytes_read
        cdef long rec_end
        cdef uint32_t rogue_payload
        cdef int n_channels = -1
        cdef uint32_t current_num_channels
        cdef bint found_data

        # only save requested channels to memory
        if channel is not None:
            channel = np.atleast_1d(np.asarray(channel, dtype=np.int64))

        # Read records
        while True:
            # Check if at end of file
            if ftell(self.file_ptr) >= self.file_size:
                break

            # Process Rogue header if needed
            if self.is_rogue:
                found_data = False
                while not found_data:
                    # Check for EOF
                    if ftell(self.file_ptr) >= self.file_size:
                        break

                    # Read Rogue header
                    bytes_read = fread(self.rogue_buffer, 1, ROGUE_HEADER_SIZE, self.file_ptr)
                    if bytes_read != ROGUE_HEADER_SIZE:
                        break

                    # Parse header
                    with nogil:
                        parse_rogue_header(self.rogue_buffer, &rogue_hdr)

                    rogue_payload = rogue_hdr.size - 4
                    rec_end = ftell(self.file_ptr) + rogue_payload

                    # If data channel, process it
                    if rogue_hdr.channel == 0:
                        found_data = True
                        break
                    else:
                        # TODO support reading metadata channels
                        # Skip non-data channels
                        fseek(self.file_ptr, rec_end, SEEK_SET)

                if not found_data:
                    break

            # Read SMURF header
            bytes_read = fread(self.header_buffer, 1, SMURF_HEADER_SIZE, self.file_ptr)
            if bytes_read != SMURF_HEADER_SIZE:
                break

            # Store raw header bytes for later parsing
            header_bytes.extend(self.header_buffer[:SMURF_HEADER_SIZE])

            # Read number_of_channels directly from buffer for validation
            current_num_channels = (<uint32_t*>&self.header_buffer[4])[0]

            if n_channels == -1:
                n_channels = <int>current_num_channels
            if n_channels != current_num_channels:
                raise ValueError(
                    f"Channel count mismatch: previous frame had {n_channels}, got {current_num_channels}"
                )

            # Read channel data using numpy fromfile for efficiency
            channel_data = np.fromfile(self.filename,
                                      dtype=SMURF_DATA_DTYPE,
                                      count=n_channels,
                                      offset=ftell(self.file_ptr))

            # subset if requested
            if channel is not None:
                channel_data = channel_data[channel]

            # Advance file pointer
            fseek(self.file_ptr, n_channels * SMURF_DATA_DTYPE.itemsize, SEEK_CUR)

            # Store data
            data_list.append(channel_data)

            self.records_read += 1

        # Parse all headers at once from collected bytes
        headers_array = np.frombuffer(header_bytes, dtype=SMURF_HEADER_DTYPE)

        # Mask external_time_raw to 5 bytes (40 bits)
        headers_array['external_time_raw'] &= 0xFFFFFFFFFF

        # Convert other data to numpy arrays
        timestamps_array = headers_array['timestamp']
        data_array = np.array(data_list, dtype=np.int32)

        return timestamps_array, data_array, headers_array


def read_stream_data_cython(str datafile, channel=None, bint IQ_mode=False):
    """
    Ultra-fast Cython implementation that reads entire file in C.
    This completely replaces the Python loop over SmurfStreamReader.records()

    Parameters
    ----------
    datafile : str
        Path to the data file
    channel : int or array-like or None
        Channels to read (if None, reads all)
    IQ_mode : bool
        Whether data is in IQ streaming mode

    Returns
    -------
    t : ndarray
        Timestamps
    phase : ndarray
        Phase data (or IQ data if in IQ mode)
    headers : list
        SMuRF headers.
    """
    cdef:
        FastSmurfReader reader
        cnp.ndarray[cnp.uint64_t, ndim=1] timestamps
        cnp.ndarray[cnp.int32_t, ndim=2] data_array
        cnp.ndarray[cnp.float64_t, ndim=2] phase
        cnp.ndarray[cnp.complex128_t, ndim=2] iq_data
        int n_channels, n_records
        int i, j, chan_idx
        cnp.ndarray[cnp.int64_t, ndim=1] channel_indices

    # Open file and read all records in C
    with FastSmurfReader(datafile, is_rogue=True) as reader:
        timestamps, data_array, headers = reader.read_all_records(channel=channel)

    n_records = len(timestamps)
    n_chan = data_array.shape[1]

    # Extract selected channels and convert to phase
    if IQ_mode:
        # IQ mode: pair consecutive channels
        if n_chan % 2 != 0:
            raise ValueError(f"In IQ mode, number of channels should be even. Found {n_chan}.")
        iq_data = np.empty((n_chan // 2, n_records), dtype=np.complex128)
        iq_data = (data_array[:, ::2] + 1j * data_array[:, 1::2]).T

        return timestamps, iq_data, headers
    else:
        # Normal mode: convert to phase
        phase = np.empty((n_chan, n_records), dtype=np.float64)

        phase = data_array.T / (2.0**15) * pi

        return timestamps, phase, headers

def parse_tes_bias_from_headers(headers):
    """
    Extract TES bias data from headers and parse into 16 TES bias values.

    The tes_bias field contains 40 bytes encoding 16 TES bias values as 20-bit
    signed integers. Each pair of values fits in 5 bytes:
    - Even index (0, 2, 4, ...): bytes 0-2, lower 20 bits
    - Odd index (1, 3, 5, ...): bytes 2-4, upper 20 bits (shifted right 4)

    Parameters
    ----------
    headers : ndarray (structured array)
        Headers array with 'tes_bias' field (40 bytes per header)

    Returns
    -------
    tes_bias_array : ndarray (int32, shape=[16, n_headers])
        Parsed TES bias values, 16 values per header
    """
    cdef int n_headers = len(headers)
    cdef int i, j, b
    cdef cnp.ndarray[cnp.int32_t, ndim=2] tes_bias_array = np.empty((16, n_headers), dtype=np.int32)
    cdef cnp.ndarray[cnp.uint8_t, ndim=1] raw_bytes
    cdef uint32_t tmp

    # Process each header
    for j in range(n_headers):
        raw_bytes = headers['tes_bias'][j]

        # Parse 16 TES bias values from 40 bytes
        for i in range(16):
            b = i // 2  # Which 5-byte group (0-7)

            # 2 TES values fit in 5 bytes
            # Each pair (bytes 0-4): 00 00 01 11 11
            # Even (i%2==0): bytes 0-2, mask to 20 bits: 00 00 0x
            # Odd  (i%2==1): bytes 2-4, shift right 4, mask to 20 bits: x1 11 11

            if i % 2 == 0:  # Even index
                # Read bytes 0-2 of the 5-byte group
                tmp = ((<uint32_t>raw_bytes[b*5]) |
                       (<uint32_t>raw_bytes[b*5 + 1] << 8) |
                       (<uint32_t>raw_bytes[b*5 + 2] << 16)) & 0xFFFFF
            else:  # Odd index
                # Read bytes 2-4 of the 5-byte group, shift right 4 bits
                tmp = (((<uint32_t>raw_bytes[b*5 + 2]) |
                        (<uint32_t>raw_bytes[b*5 + 3] << 8) |
                        (<uint32_t>raw_bytes[b*5 + 4] << 16)) >> 4) & 0xFFFFF

            # Convert to signed 20-bit value
            # If bit 19 is set, it's negative (two's complement)
            if tmp >= 0x80000:
                tmp -= 0x100000

            tes_bias_array[i, j] = <int32_t>tmp

    return tes_bias_array

