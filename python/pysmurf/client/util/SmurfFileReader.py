#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf file reader class
#-----------------------------------------------------------------------------
# File       : pysmurf/util/SmurfFileReader.py
# Created    : 2019-11-06
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import struct
import numpy
import os
import yaml
from collections import namedtuple
from collections import OrderedDict as odict

# Frame Format Constants
SmurfHeaderSize  = 128
RogueHeaderSize  = 8
SmurfChannelSize = 4
SmurfHeaderPack  = '4B 1I 40x 1Q 4I 1Q 3I 4x 1Q 2B 6x 2H 4x 2H 4x'
RogueHeaderPack  = 'IHBB'

# Code derived from existing code copied from Edward Young, Jesus Vasquez
# https://github.com/slaclab/pysmurf/blob/pre-release/python/pysmurf/client/util/smurf_util.py#L768
# This is the structure of the header (see README.SmurfPacket.md for a details)
# Note: This assumes that the header version is 1 (currently the only version available),
# which has a length of 128 bytes. In the future, we should check first the version,
# and then unpack the data base on the version number.
# TO DO: Extract the TES BIAS values

# Default header as a named tuple
SmurfHeaderTuple = namedtuple( 'SmurfHeader',
                              [ 'protocol_version'    ,  # 1  Byte, B
                                'crate_id'            ,  # 1  Byte, B
                                'slot_number'         ,  # 1  Byte, B
                                'timing_cond'         ,  # 1  Byte, B
                                'number_of_channels'  ,  # 4  Bytes, uint32, I
                                                         # 40 Bytes, TesBias, 40x
                                'timestamp'           ,  # 8  Bytes, uint64, Q
                                'flux_ramp_increment' ,  # 4  Bytes, int32,  I
                                'flux_ramp_offset'    ,  # 4  Bytes, int32,  I
                                'counter_0'           ,  # 4  Bytes, uint32, I
                                'counter_1'           ,  # 4  Bytes, uint32, I
                                'counter_2'           ,  # 8  Bytes, uint64, Q
                                'reset_bits'          ,  # 4  Bytes, uint32, I
                                'frame_counter'       ,  # 4  Bytes, uint32, I
                                'tes_relays_config'   ,  # 4  Bytes, bit mask (uint32), I
                                                         # 4  Bytes, unused, 4x
                                'external_time_raw'   ,  # 5  Bytes, uint64, Q (3 extra bytes)
                                'control_field'       ,  # 1  Byte, B
                                'test_params'         ,  # 1  Byte, B
                                                         # 6  Bytes, unused, 6x
                                'num_rows'            ,  # 2  Bytes, uint16, H
                                'num_rows_reported'   ,  # 2  Bytes, uint16, H
                                                         # 4  Bytes, unused, 4x
                                'row_length'          ,  # 2  Bytes, uint16, H
                                'data_rate' ] )          # 2  Bytes, uint16, H
                                                         # 4  Bytes, unused, 4x


# Default header as a named tuple
RogueHeader = namedtuple( 'RogueHeader',
                         [ 'size'                ,  # 4 Bytes, uint32, I
                           'flags'               ,  # 2 bytes, uint16, H
                           'error'               ,  # 1 bytes, uint8,  B
                           'channel' ])             # 1 bytes, uint8,  B


class SmurfHeader(SmurfHeaderTuple):

    def initialize(self, rawData):
        self.external_time = self.external_time_raw & 0xFFFFFFFFFF # Only lower 5 bytes
        self.tesBias = []

        # 24 bit bias values
        #self.tesBias = [int.from_bytes(rawData[8+i*3:8+i*3+3], 'little', signed=True) for i in range(16)]

        # 20 bit bias values
        for i in range(16):

            # 2 TES value fit in 5 bytes, starting from byte 8
            # Each pair (byte 0 - 4): 00 00 01 11 11
            # Even: bytes 0 - 2: 00 00 0x
            # Odd: bytes 2 - 4: x1 11 11
            if i % 2 == 0: # Even
                tmp = int.from_bytes(rawData[8+i*5:8+i*5+3],'little',signed=False) & 0xFFFFF
            else: # Odd
                tmp = (int.from_bytes(rawData[8+i*5+2:8+i*5+5],'little',signed=False) >> 4) & 0xFFFFF

            # Adjust negative values
            if tmp >= 0x80000:
                tmp -= 0x100000

            self.tesBias.append(tmp)

class SmurfStreamReader(object):

    def __init__(self, files, *, isRogue=True, metaEnable=False, chanCount=None):
        self._isRogue    = isRogue
        self._metaEnable = metaEnable
        self._chanCount  = chanCount
        self._currFile   = None
        self._currFName  = ''
        self._fileSize   = 0
        self._header     = None
        self._data       = None
        self._config     = {}
        self._currCount  = 0
        self._totCount   = 0

        if isinstance(files,list):
            self._fileList = files
        else:
            self._fileList = [files]

        # Check to make sure all the files are readable
        for fn in self._fileList:
            if not os.access(fn,os.R_OK):
                raise Exception(f"Unable to read file {fn}")

    def _parseRogueHeader(self):
        return RogueHeader._make(struct.Struct(RogueHeaderPack).unpack(self._currFile.read(RogueHeaderSize)))

    def _parseSmurfHeader(self):
        data = self._currFile.read(SmurfHeaderSize)
        ret = SmurfHeader._make(struct.Struct(SmurfHeaderPack).unpack(data))
        ret.initialize(data)
        return ret

    def _readPayload(self,chanCount):
        return numpy.fromfile(self._currFile, dtype=numpy.int32, count=chanCount)

    def _nextRecord(self):
        """
        Process next record, return true on success
        """
        recEnd = 0

        # We are at the end of the file
        if self._currFile.tell() == self._fileSize:
            return False

        # Use Rogue format
        if self._isRogue:

            # keep reading until we get a data channel
            while True:

                # Hit end of file
                if self._currFile.tell() == self._fileSize:
                    return False

                # Not enough data left in the file
                if (self._fileSize - self._currFile.tell()) < RogueHeaderSize:
                    print(f"Waring: File under run reading {self._currFName}")
                    return False

                # Read in Rogue header data
                rogueHeader  = self._parseRogueHeader()
                roguePayload = rogueHeader.size - 4

                # Set next frame position
                recEnd = self._currFile.tell() + roguePayload

                # Sanity check
                if recEnd > self._fileSize:
                    print(f"Waring: File under run reading {self._currFName}")
                    return False

                # If this is a data channel, break
                if rogueHeader.channel == 0:
                    break

                # Process meta data
                elif self._metaEnable and rogueHeader.channel == 1:
                    try:
                        yamlUpdate(self._config, self._currFile.read(roguePayload).decode('utf-8'))
                    except Exception as e:
                        print(f"Waring: Error processing meta data in {self._currFName}: {e}")

                # Skip over meta data
                else:
                    self._currFile.seek(recEnd)

            # Check if there is enough room for the Smurf header
            if SmurfHeaderSize > roguePayload:
                print(f"Waring: SMURF header overruns remaining record size in {self._currFName}")
                return False

        # Non Rogue file, verify there is enough room in the file for the smurf header
        elif SmurfHeaderSize + self._currFile.tell() > self._fileSize:
            print(f"Warning: SMURF header overruns remaining file size in {self._currFName}")
            return None

        # Unpack header into named tuple
        self._header = self._parseSmurfHeader()

        # Number of data channels is taken from the header
        chanCount = self._header.number_of_channels
        dataSize  = chanCount * SmurfChannelSize

        # Rogue raw data size is computed from rogue headers
        if self._isRogue:
            rawDataSize = recEnd - self._currFile.tell()

        # Legacy File
        else:

            # Use defined size if exists.
            if chanCount is not None:
                rawDataSize = chanCount * SmurfChannelSize

            # Otherwise use header value
            else:
                rawDataSize = self._header.number_of_channels * SmurfChannelSize

            # Compute legacy record end
            recEnd = self._currFile.tell() + rawDataSize

            # File overrun
            if recEnd > self._fileSize:
                print(f"SMURF data overruns remaining file size in {self._currFName}")
                return False

        # Verify sizing
        if dataSize > rawDataSize:
            print(f"Warning: SMURF read data size overruns raw data size in {self._currFName}")
            return False

        # Read record data
        self._data = self._readPayload(chanCount)
        self._currCount += 1
        self._totCount += 1

        # Jump forward if neccessary
        if ( self._currFile.tell() != recEnd ):
            self._currFile.seek(recEnd)

        return True

    def records(self):
        """
        Generator which returns (header, data) tuples
        """
        self._config = {}
        self._currCount = 0
        self._totCount  = 0

        for fn in self._fileList:
            self._fileSize = os.path.getsize(fn)
            self._currFName = fn
            self._currCount = 0

            print(f"Processing data records from {self._currFName}")
            with open(fn,'rb') as f:
                self._currFile = f

                while self._nextRecord():
                    yield (self._header, self._data)

            print(f"Processed {self._currCount} data records from {self._currFName}")

        print(f"Processed a total of {self._totCount} data records")

    @property
    def currCount(self):
        return self._currCount

    @property
    def totCount(self):
        return self._totCount

    @property
    def configDict(self):
        return self._config

    def configValue(self, path):
        obj = self._config

        if '.' in path:
            lst = path.split('.')
        else:
            lst = [path]

        for a in lst:
            if a in obj:
                obj = obj[a]
            else:
                return None

        return obj

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        pass


def keyValueUpdate(old, key, value):
    d = old
    parts = key.split('.')
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d.get(part)
    d[parts[-1]] = value

def dictUpdate(old, new):
    for k,v in new.items():
        if '.' in k:
            keyValueUpdate(old, k, v)
        elif k in old:
            old[k].update(v)
        else:
            old[k] = v

def yamlUpdate(old, new):
    dictUpdate(old, yamlToData(new))

def yamlToData(stream):
    """Load yaml to data structure"""

    class PyrogueLoader(yaml.Loader):
        pass

    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return odict(loader.construct_pairs(node))

    PyrogueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,construct_mapping)

    return yaml.load(stream, Loader=PyrogueLoader)
