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

import yaml
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
    cdef long records_read
    cdef uint8_t header_buffer[128]
    cdef uint8_t rogue_buffer[8]

    def __cinit__(self, str filename):
        self.filename = filename.encode('utf-8')
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

    cdef void _get_dimensions(self, int32_t* n_chan, int32_t* n_record):
        cdef uint32_t rogue_payload, bytes_read
        cdef RogueHeader rogue_hdr
        n_chan[0] = -1
        n_record[0] = 0

        # set file pointer back to beginning
        fseek(self.file_ptr, 0, SEEK_SET)

        # Read records
        while True:
            # Check if at end of file
            if ftell(self.file_ptr) >= self.file_size:
                break

            # Read Rogue header
            bytes_read = fread(self.rogue_buffer, 1, ROGUE_HEADER_SIZE, self.file_ptr)
            if bytes_read != ROGUE_HEADER_SIZE:
                break

            # Parse header
            parse_rogue_header(self.rogue_buffer, &rogue_hdr)

            rogue_payload = rogue_hdr.size - 4

            # If data channel, increment counter
            if rogue_hdr.channel == 0:
                n_record[0] += 1

                # read number of channels once
                if n_chan[0] == -1:
                    bytes_read = fread(self.header_buffer, 1, SMURF_HEADER_SIZE, self.file_ptr)
                    if bytes_read != SMURF_HEADER_SIZE:
                        break
                    n_chan[0] = (<uint32_t*>&self.header_buffer[4])[0]
                    # Adjust seek since we read the header
                    fseek(self.file_ptr, rogue_payload - SMURF_HEADER_SIZE, SEEK_CUR)
                    continue  # Skip the normal seek below

            # advance file pointer
            fseek(self.file_ptr, rogue_payload, SEEK_CUR)

    def read_all_records(self, channel=None, skip_meta=True):
        """
        Read all records from file and return as numpy arrays
        This is the main high-performance function

        Arguments
        ---------
        channel : array-like or None
            Channels to read (if None, reads all)
        skip_meta : bool, optional, default True
            Whether to skip metadata parsing

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
        metadata : dict
            Dictionary of metadata read from Rogue headers (if skip_meta=False)
        """
        # declare memory views for storing data
        cdef int32_t[:, :] data_view
        cdef int32_t[:] channel_data
        cdef char[:] headers_view
        # just use a list for yaml metadata
        cdef list meta_list = []

        # variables for controlling read loop
        cdef RogueHeader rogue_hdr
        cdef size_t bytes_read
        cdef uint32_t rogue_payload
        cdef int32_t n_channels, n_records
        cdef uint32_t current_num_channels
        cdef bint found_data
        cdef int32_t i, j, ch

        # first pass over file to get dimensions
        self._get_dimensions(&n_channels, &n_records)

        # allocate data buffer for reading from disk
        channel_data_array = np.zeros(n_channels, dtype=np.int32)
        channel_data = channel_data_array

        # allocate buffer for headers
        headers_array = np.zeros(n_records, dtype=SMURF_HEADER_DTYPE)
        headers_view = headers_array.view(np.int8)
        cdef int header_offset = 0

        # only save requested channels to memory
        if channel is not None:
            channel = np.atleast_1d(np.asarray(channel, dtype=np.int32))
        else:
            channel = np.arange(n_channels, dtype=np.int32)

        # Cache size and create memory view
        cdef int32_t n_sel_channels = channel.size
        cdef int32_t[:] channel_view = channel

        # allocate data array for storing in memory
        data_array = np.empty((n_records, n_sel_channels), dtype=np.int32)
        data_view = data_array

        # set file pointer back to beginning
        fseek(self.file_ptr, 0, SEEK_SET)

        # Read records
        i = 0
        while True:
            # Check if at end of file
            if ftell(self.file_ptr) >= self.file_size:
                break

            # Read Rogue headers to find data channel
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

                # If data channel, process it
                if rogue_hdr.channel == 0:
                    found_data = True
                    break
                else:
                    # read into metadata buffer
                    if skip_meta:
                        fseek(self.file_ptr, rogue_payload, SEEK_CUR)
                    else:
                        meta_buffer = bytearray(rogue_payload)
                        bytes_read = fread(<char*>meta_buffer, 1, rogue_payload, self.file_ptr)
                        if bytes_read != rogue_payload:
                            break
                        meta_list.append((self.records_read, meta_buffer.decode('utf-8')))

            if not found_data:
                break

            # Read SMURF header
            bytes_read = fread(self.header_buffer, 1, SMURF_HEADER_SIZE, self.file_ptr)
            if bytes_read != SMURF_HEADER_SIZE:
                break

            # Store raw header bytes for later parsing
            memcpy(&headers_view[header_offset], self.header_buffer, SMURF_HEADER_SIZE)
            header_offset += SMURF_HEADER_SIZE

            # Read number_of_channels directly from buffer for validation
            current_num_channels = (<uint32_t*>&self.header_buffer[4])[0]

            if n_channels == -1:
                n_channels = <int>current_num_channels
            if n_channels != current_num_channels:
                raise ValueError(
                    f"Channel count mismatch: previous frame had {n_channels}, got {current_num_channels}"
                )

            # read all channels into buffer
            bytes_read = fread(&channel_data[0], 4, n_channels, self.file_ptr)
            if bytes_read != n_channels:
                break

            # copy into array with subsetting
            for j in range(n_sel_channels):
                ch = channel_view[j]
                data_view[i, j] = channel_data[ch]

            self.records_read += 1
            i += 1

        # Mask external_time_raw to 5 bytes (40 bits)
        headers_array['external_time_raw'] &= 0xFFFFFFFFFF

        timestamps_array = headers_array['timestamp']

        # parse metadata stream from YAML
        metadata = {i: yaml.safe_load(m) for i, m in meta_list}

        return timestamps_array, data_array, headers_array, metadata


def read_stream_data_cython(str datafile, channel=None, bint IQ_mode=False, bint skip_meta=True):
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
    skip_meta : bool, optional, default True
        Whether to skip metadata parsing

    Returns
    -------
    t : ndarray
        Timestamps
    phase : ndarray
        Phase data (or IQ data if in IQ mode)
    headers : list
        SMuRF headers.
    meta : dict
        Metadata read from file. dict keys are the corresponding data index.
    """
    cdef:
        FastSmurfReader reader
        cnp.ndarray[cnp.uint64_t, ndim=1] timestamps
        cnp.int32_t[:, :] data_view
        cnp.float64_t[:, :] phase_view
        cnp.complex128_t[:, :] iq_data_view
        int n_chan, n_records
        int i, j

    # Open file and read all records in C
    with FastSmurfReader(datafile) as reader:
        timestamps, data_array, headers, meta = reader.read_all_records(
            skip_meta=skip_meta, channel=channel
        )
    data_view = data_array  # point memory view to array

    n_records = len(timestamps)
    n_chan = data_array.shape[1]

    # Extract selected channels and convert to phase
    if IQ_mode:
        # IQ mode: pair consecutive channels
        if n_chan % 2 != 0:
            raise ValueError(f"In IQ mode, number of channels should be even. Found {n_chan}.")

        iq_data = np.empty((n_chan // 2, n_records), dtype=np.complex128)
        iq_data_view = iq_data  # point memory view to array

        for i in range(n_records):
            for j in range(n_chan // 2):
                iq_data_view[j, i] = (data_view[i, 2 * j] + 1j * data_view[i, 2 * j + 1])

        return timestamps, iq_data, headers, meta
    else:
        # Normal mode: convert to phase
        phase = np.empty((n_chan, n_records), dtype=np.float64)
        phase_view = phase  # point memory view to array

        for i in range(n_records):
            for j in range(n_chan):
                phase_view[j, i] =  data_view[i, j] / (2.0**15) * pi

        return timestamps, phase, headers, meta

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

