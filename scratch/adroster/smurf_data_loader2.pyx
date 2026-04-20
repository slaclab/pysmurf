#From Tristan PM, April 2026. Fix to reading in headers.

#uncomment the following line if you want to run in a Jupyter notebook
#%%cython

import numpy as np
cimport numpy as np
from libc.stdint cimport uint8_t, uint16_t, uint32_t, int32_t
from libc.stdio cimport FILE, fopen, fread, fclose, feof, SEEK_SET, SEEK_END, SEEK_CUR, fseek, ftell, rewind
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy

# SMuRF header size in bytes
DEF SMURF_HEADER_SIZE = 128

cdef struct DataWriterHeader:
    uint32_t bank_length
    uint8_t channel_id
    uint8_t frame_error
    uint16_t frame_flags

cdef struct SmurfPacket:
    uint8_t header[128]
    int32_t* data
    uint32_t num_channels
    uint32_t payload_size

cdef struct FileStats:
    uint32_t num_data_frames
    uint32_t num_metadata_frames
    uint32_t max_channels
    uint32_t max_metadata_size

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

cdef class SmurfFileReader:
    """
    Reader for SMuRF data files with DataWriter format.
    
    File format consists of banks with 2-word headers:
    - Word 0 [31:0]: Bank length (bytes, including second header word)
    - Word 1 [31:24]: Channel ID
    - Word 1 [23:16]: Frame error
    - Word 1 [15:0]: Frame flags
    
    Channel 0: Processed SMuRF data (128-byte header + payload)
    Channel 1: Metadata
    """
    
    cdef FILE* fp
    cdef object filename
    cdef public long file_position
    cdef public long total_frames
    
    def __init__(self, str filename):
        """Initialize the reader with a filename."""
        self.filename = filename.encode('utf-8')
        self.fp = NULL
        self.file_position = 0
        self.total_frames = 0
        
    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
    
    cdef int open(self) except -1:
        """Open the file for reading."""
        if self.fp != NULL:
            raise IOError("File already open")

        self.fp = fopen(self.filename, "rb")
        if self.fp == NULL:
            raise IOError(f"Could not open file: {self.filename.decode('utf-8')}")
        
        self.file_position = 0
        return 0
    
    cdef int close(self) except -1:
        """Close the file."""
        if self.fp != NULL:
            fclose(self.fp)
            self.fp = NULL
        return 0
    
    cdef int rewind_file(self) except -1:
        """Rewind file to beginning."""
        if self.fp == NULL:
            raise IOError("File not open")
        rewind(self.fp)
        self.file_position = 0
        return 0

    cdef int get_file_size(self):
        # Save current position
        original_pos = ftell(self.fp)
        # Seek to the end of the file
        fseek(self.fp, 0, SEEK_END)
        # Get the current position (size)
        size = ftell(self.fp)
        # Restore original position
        fseek(self.fp, original_pos, SEEK_SET)
        return size

    def py_get_file_size(self):
        return self.get_file_size()
    
    cdef int read_bank_header(self, DataWriterHeader* header) except -1:
        """
        Read the DataWriter bank header (2 32-bit words).
        
        Returns:
            0 on success, -1 on error or EOF
        """
        cdef uint32_t word0, word1
        cdef size_t bytes_read
        
        # Read first word (bank length)
        # TODO: I think this fails to detect EOF
        bytes_read = fread(&word0, sizeof(uint32_t), 1, self.fp)
        if bytes_read != 1:
            if feof(self.fp):
                return -1  # End of file
            raise IOError("Error reading bank length")
        
        # Read second word (channel ID, error, flags)
        bytes_read = fread(&word1, sizeof(uint32_t), 1, self.fp)
        if bytes_read != 1:
            if feof(self.fp):
                return -1  # End of file
            raise IOError("Error reading bank header word 2")
        
        # Parse header
        header.bank_length = word0
        header.channel_id = (word1 >> 24) & 0xFF
        header.frame_error = (word1 >> 16) & 0xFF
        header.frame_flags = word1 & 0xFFFF
        
        return 0
    
    cdef int read_smurf_header(self, uint8_t* header_buffer) except -1:
        """
        Read the 128-byte SMuRF header.
        
        Returns:
            Number of channels from the header
        """
        cdef size_t bytes_read
        cdef uint32_t num_channels
        
        bytes_read = fread(header_buffer, 1, SMURF_HEADER_SIZE, self.fp)
        if bytes_read != SMURF_HEADER_SIZE:
            raise IOError(f"Error reading SMuRF header, got {bytes_read} bytes")
        
        # Extract number of channels from header
        # Assuming number of channels is stored at a specific offset in the header
        # This may need adjustment based on actual SMuRF header format
        # For now, reading it from bytes 8-11 (typical location)
        num_channels = (<uint32_t*>(header_buffer + 4))[0]
        
        return num_channels
    
    def read_next_frame(self):
        """
        Read the next frame from the file.
        
        Returns:
            dict with keys:
                - 'channel_id': Channel ID (0=data, 1=metadata)
                - 'frame_error': Frame error flags
                - 'frame_flags': Frame flags
                - 'header': SMuRF header (bytes, only for channel 0)
                - 'data': Numpy array of int32 (only for channel 0)
                - 'metadata': Raw metadata bytes (only for channel 1)
                - 'num_channels': Number of valid channels (only for channel 0)
            
            Returns None on EOF
        """
        cdef DataWriterHeader dw_header
        cdef uint8_t smurf_header[SMURF_HEADER_SIZE]
        cdef int32_t* data_buffer
        cdef uint32_t num_channels, payload_size_bytes, payload_size_words
        cdef size_t bytes_read
        cdef int result
        
        if self.fp == NULL:
            raise IOError("File not open")
        
        # Read DataWriter header
        result = self.read_bank_header(&dw_header)
        if result == -1:
            return None  # EOF
        
        # Calculate payload size (bank_length includes the second header word)
        payload_size_bytes = dw_header.bank_length - sizeof(uint32_t)
        
        frame_dict = {
            'channel_id': dw_header.channel_id,
            'frame_error': dw_header.frame_error,
            'frame_flags': dw_header.frame_flags,
        }
        
        if dw_header.channel_id == 0:
            # SMuRF data channel
            num_channels = self.read_smurf_header(smurf_header)
            
            
            # Copy header to Python bytes
            frame_dict['header'] = bytes(smurf_header[:SMURF_HEADER_SIZE])
            frame_dict['num_channels'] = num_channels
            
            # Read payload data
            data_payload_bytes = payload_size_bytes - SMURF_HEADER_SIZE
            payload_size_words = data_payload_bytes // sizeof(int32_t)
            
            # Create numpy array for data
            data_array = np.zeros(payload_size_words, dtype=np.int32)
            data_buffer = <int32_t*>np.PyArray_DATA(data_array)
            
            bytes_read = fread(data_buffer, 1, data_payload_bytes, self.fp)
            if bytes_read != data_payload_bytes:
                raise IOError(f"Error reading SMuRF data, expected {data_payload_bytes}, got {bytes_read}")
            
            # Only return valid channels
            frame_dict['data'] = data_array[:num_channels]
            
            self.total_frames += 1
            
        elif dw_header.channel_id == 1:
            # Metadata channel
            metadata_buffer = np.zeros(payload_size_bytes, dtype=np.uint8)
            bytes_read = fread(np.PyArray_DATA(metadata_buffer), 1, payload_size_bytes, self.fp)
            if bytes_read != payload_size_bytes:
                raise IOError(f"Error reading metadata, expected {payload_size_bytes}, got {bytes_read}")
            
            frame_dict['metadata'] = bytes(metadata_buffer)
        else:
            # Unknown channel, skip it
            fseek(self.fp, payload_size_bytes, SEEK_SET)
        
        self.file_position += 2 * sizeof(uint32_t) + payload_size_bytes
        
        return frame_dict
    
    cdef FileStats count_frames(self) except *:
        """
        Count frames and get statistics by doing a quick pass through the file.
        Returns to original file position when done.
        """
        cdef FileStats stats
        cdef DataWriterHeader dw_header
        cdef long original_pos
        cdef uint32_t payload_size_bytes
        cdef uint32_t file_loc
        cdef uint8_t smurf_header[SMURF_HEADER_SIZE]
        cdef uint32_t num_channels
        cdef size_t bytes_read
        cdef int result
        
        stats.num_data_frames = 0
        stats.num_metadata_frames = 0
        stats.max_channels = 0
        stats.max_metadata_size = 0
        
        if self.fp == NULL:
            raise IOError("File not open")

        file_size = self.get_file_size()
        
        # Save current position
        original_pos = ftell(self.fp)
        
        # Rewind to beginning
        self.rewind_file()
        
        # Scan through file
        print("starting to scan file")
        while True:
            file_loc = ftell(self.fp)
            if file_loc + 1 >= file_size:
                print("at last byte")
                print(file_loc, file_size)
                break
            if feof(self.fp):
                print("EOF")
                break
            result = self.read_bank_header(&dw_header)
            if result == -1:
                print("failed to read header")
                break  # EOF
            
            payload_size_bytes = dw_header.bank_length - sizeof(uint32_t)
            
            if dw_header.channel_id == 0:
                # Data frame - read header to get channel count
                num_channels = self.read_smurf_header(smurf_header)
                
                if num_channels > stats.max_channels:
                    stats.max_channels = num_channels
                
                stats.num_data_frames += 1
                
                # Skip remaining payload
                fseek(self.fp, payload_size_bytes - SMURF_HEADER_SIZE, SEEK_CUR)  # SEEK_CUR = 1
            elif dw_header.channel_id == 1:
                # Metadata frame
                if payload_size_bytes > stats.max_metadata_size:
                    stats.max_metadata_size = payload_size_bytes
                
                stats.num_metadata_frames += 1
                
                # Skip payload
                fseek(self.fp, payload_size_bytes, SEEK_CUR)  # SEEK_CUR = 1
            else:
                # Unknown channel - skip
                fseek(self.fp, payload_size_bytes, SEEK_CUR)  # SEEK_CUR = 1
        print("done scanning file")

        # Restore original position
        fseek(self.fp, original_pos, SEEK_SET)
        
        return stats

    def py_count_frames(self):
        return self.count_frames()
    
    def read_all_data_to_array(self):
        """
        Read all SMuRF data frames into a 2D NumPy array.
        
        Returns:
            tuple: (data_array, metadata_dict)
                - data_array: 2D array of shape (num_frames, num_channels) of int32
                - metadata_dict: dict with 'frame_errors', 'frame_flags' arrays
        """
        cdef FileStats stats
        cdef np.ndarray[int32_t, ndim=2] data_array
        cdef np.ndarray[uint8_t, ndim=1] frame_errors
        cdef np.ndarray[uint16_t, ndim=1] frame_flags
        cdef char[:] headers_view
        cdef DataWriterHeader dw_header
        cdef uint8_t smurf_header[SMURF_HEADER_SIZE]
        cdef uint32_t num_channels, payload_size_bytes, data_payload_bytes
        cdef size_t bytes_read, header_offset
        cdef int result
        cdef uint32_t frame_idx = 0
        cdef int32_t* row_ptr
        
        if self.fp == NULL:
            raise IOError("File not open")
        
        # First pass: count frames and get dimensions
        print("first pass read to count frames")
        stats = self.count_frames()
        print(f"found {stats} in file")
        
        if stats.num_data_frames == 0:
            # Return empty arrays
            return np.zeros((0, 0), dtype=np.int32), {
                'frame_errors': np.zeros(0, dtype=np.uint8),
                'frame_flags': np.zeros(0, dtype=np.uint16)
            }
        
        # Allocate arrays
        data_array = np.zeros((stats.num_data_frames, stats.max_channels), dtype=np.int32)
        frame_errors = np.zeros(stats.num_data_frames, dtype=np.uint8)
        frame_flags = np.zeros(stats.num_data_frames, dtype=np.uint16)
        headers_array = np.zeros(stats.num_data_frames, dtype=SMURF_HEADER_DTYPE)
        headers_view = headers_array.view(np.int8)
        header_offset = 0
        
        # Rewind for second pass
        self.rewind_file()
        
        # Second pass: read data into arrays
        while frame_idx < stats.num_data_frames:
            result = self.read_bank_header(&dw_header)
            if result == -1:
                break  # EOF
            
            payload_size_bytes = dw_header.bank_length - sizeof(uint32_t)
            
            if dw_header.channel_id == 0:
                # Data frame
                num_channels = self.read_smurf_header(smurf_header)

                # copy header
                memcpy(&headers_view[header_offset], smurf_header, SMURF_HEADER_SIZE)
                header_offset += SMURF_HEADER_SIZE
                
                # Store metadata
                frame_errors[frame_idx] = dw_header.frame_error
                frame_flags[frame_idx] = dw_header.frame_flags
                
                # Read data directly into array
                data_payload_bytes = payload_size_bytes - SMURF_HEADER_SIZE
                row_ptr = <int32_t*>(np.PyArray_DATA(data_array) + frame_idx * stats.max_channels * sizeof(int32_t))
                
                bytes_read = fread(row_ptr, 1, min(data_payload_bytes, num_channels * sizeof(int32_t)), self.fp)
                
                # Skip any remaining padding
                if data_payload_bytes > num_channels * sizeof(int32_t):
                    fseek(self.fp, data_payload_bytes - num_channels * sizeof(int32_t), 1)  # SEEK_CUR = 1
                
                frame_idx += 1
                
            elif dw_header.channel_id == 1:
                # Skip metadata frames
                fseek(self.fp, payload_size_bytes, 1)  # SEEK_CUR = 1
            else:
                # Skip unknown channels
                fseek(self.fp, payload_size_bytes, 1)  # SEEK_CUR = 1
        
        #return data_array, {
        return data_array, headers_array, {
            'frame_errors': frame_errors,
            'frame_flags': frame_flags
        }
    
    def read_all_data_to_array_variable_channels(self):
        """
        Read all SMuRF data frames into a list of 1D arrays (for variable channel counts).
        
        Returns:
            tuple: (data_list, num_channels_array, metadata_dict)
                - data_list: list of 1D arrays, each containing data for one frame
                - num_channels_array: 1D array with number of channels per frame
                - metadata_dict: dict with 'frame_errors', 'frame_flags' arrays
        """
        cdef FileStats stats
        cdef np.ndarray[uint32_t, ndim=1] num_channels_array
        cdef np.ndarray[uint8_t, ndim=1] frame_errors
        cdef np.ndarray[uint16_t, ndim=1] frame_flags
        cdef DataWriterHeader dw_header
        cdef uint8_t smurf_header[SMURF_HEADER_SIZE]
        cdef uint32_t num_channels, payload_size_bytes, data_payload_bytes, payload_size_words
        cdef size_t bytes_read
        cdef int result
        cdef uint32_t frame_idx = 0
        cdef int32_t* data_ptr
        
        if self.fp == NULL:
            raise IOError("File not open")
        
        # First pass: count frames
        stats = self.count_frames()
        
        if stats.num_data_frames == 0:
            return [], np.zeros(0, dtype=np.uint32), {
                'frame_errors': np.zeros(0, dtype=np.uint8),
                'frame_flags': np.zeros(0, dtype=np.uint16)
            }
        
        # Allocate metadata arrays
        num_channels_array = np.zeros(stats.num_data_frames, dtype=np.uint32)
        frame_errors = np.zeros(stats.num_data_frames, dtype=np.uint8)
        frame_flags = np.zeros(stats.num_data_frames, dtype=np.uint16)
        
        # List for data arrays
        data_list = []
        
        # Rewind for second pass
        self.rewind_file()
        
        # Second pass: read data
        while frame_idx < stats.num_data_frames:
            result = self.read_bank_header(&dw_header)
            if result == -1:
                break  # EOF
            
            payload_size_bytes = dw_header.bank_length - sizeof(uint32_t)
            
            if dw_header.channel_id == 0:
                # Data frame
                num_channels = self.read_smurf_header(smurf_header)
                
                # Store metadata
                num_channels_array[frame_idx] = num_channels
                frame_errors[frame_idx] = dw_header.frame_error
                frame_flags[frame_idx] = dw_header.frame_flags
                
                # Read data
                data_payload_bytes = payload_size_bytes - SMURF_HEADER_SIZE
                
                # Create array for this frame
                data_array = np.zeros(num_channels, dtype=np.int32)
                data_ptr = <int32_t*>np.PyArray_DATA(data_array)
                
                bytes_read = fread(data_ptr, 1, num_channels * sizeof(int32_t), self.fp)
                if bytes_read != num_channels * sizeof(int32_t):
                    raise IOError(f"Error reading data frame {frame_idx}")
                
                data_list.append(data_array)
                
                # Skip any remaining padding
                if data_payload_bytes > num_channels * sizeof(int32_t):
                    fseek(self.fp, data_payload_bytes - num_channels * sizeof(int32_t), 1)
                
                frame_idx += 1
                
            elif dw_header.channel_id == 1:
                # Skip metadata frames
                fseek(self.fp, payload_size_bytes, 1)
            else:
                # Skip unknown channels
                fseek(self.fp, payload_size_bytes, 1)
        
        return data_list, num_channels_array, {
            'frame_errors': frame_errors,
            'frame_flags': frame_flags
        }
    
    def read_all_frames(self, int channel_filter=-1):
        """
        Read all frames from the file into a list of dictionaries.
        
        Note: For better performance when reading many data frames,
        use read_all_data_to_array() instead.
        
        Args:
            channel_filter: If >= 0, only return frames from this channel
        
        Returns:
            List of frame dictionaries
        """
        frames = []
        
        while True:
            frame = self.read_next_frame()
            if frame is None:
                break
            
            if channel_filter < 0 or frame['channel_id'] == channel_filter:
                frames.append(frame)
        
        return frames
    
    def read_data_frames(self):
        """
        Read all SMuRF data frames (channel 0 only) into a NumPy array.
        
        Returns:
            tuple: (data_array, metadata_dict)
        """
        return self.read_all_data_to_array()
    
    def read_metadata_frames(self):
        """
        Read all metadata frames (channel 1 only).
        
        Returns:
            List of frame dictionaries containing only metadata frames
        """
        return self.read_all_frames(channel_filter=1)
    
    def __dealloc__(self):
        """Cleanup when object is destroyed."""
        if self.fp != NULL:
            fclose(self.fp)


def read_smurf_file(str filename, int channel_filter=-1):
    """
    Convenience function to read a SMuRF data file.
    
    Args:
        filename: Path to the data file
        channel_filter: If >= 0, only return frames from this channel
    
    Returns:
        List of frame dictionaries
    
    Note: For reading data frames, use read_smurf_data_array() for better performance
    """
    with SmurfFileReader(filename) as reader:
        return reader.read_all_frames(channel_filter=channel_filter)


def read_smurf_data_array(str filename):
    """
    Convenience function to read all SMuRF data into a NumPy array.
    
    Args:
        filename: Path to the data file
    
    Returns:
        tuple: (data_array, metadata_dict)
            - data_array: 2D array of shape (num_frames, num_channels) of int32
            - metadata_dict: dict with 'frame_errors', 'frame_flags' arrays
    """
    with SmurfFileReader(filename) as reader:
        return reader.read_all_data_to_array()