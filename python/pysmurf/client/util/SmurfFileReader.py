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
from collections import namedtuple

# Frame Format Constants
SmurfHeaderSize  = 128
RogueHeaderSize  = 8
SmurfChannelSize = 4
SmurfHeaderPack  = 'BBBBI40xQIIIIQIII4xQBB6xHH4xHH4x'
RogueHeaderPack  = 'IHBB'

# Code derived from existing code copied from Edward Young, Jesus Vasquez
# https://github.com/slaclab/pysmurf/blob/pre-release/python/pysmurf/client/util/smurf_util.py#L768
# This is the structure of the header (see README.SmurfPacket.md for a details)
# Note: This assumes that the header version is 1 (currently the only version available),
# which has a length of 128 bytes. In the future, we should check first the version,
# and then unpack the data base on the version number.
# TO DO: Extract the TES BIAS values

# Default header as a named tuple
SmurfHeader = namedtuple( 'SmurfHeader',
                         [ 'protocol_version'    ,  # 1 Byte, B
                           'crate_id'            ,  # 1 Byte, B
                           'slot_number'         ,  # 1 Byte, B
                           'timing_cond'         ,  # 1 Byte, B
                           'number_of_channels'  ,  # 4 Bytes, uint32, I
                            #'tes_bias', < TO DO, include the TES bias values, 40 bytes, 40x
                           'timestamp'           ,  # 8 Bytes, uint64, Q
                           'flux_ramp_increment' ,  # 4 Bytes, int32,  I
                           'flux_ramp_offset'    ,  # 4 bytes, int32,  I
                           'counter_0'           ,  # 4 bytes, uint32, I
                           'counter_1'           ,  # 4 bytes, uint32, I
                           'counter_2'           ,  # 8 bytes, uint64, Q
                           'reset_bits'          ,  # 4 bytes, uint32, I
                           'frame_counter'       ,  # 4 bytes, uint32, I
                           'tes_relays_config'   ,  # 4 bytes, bit mask (uint32), I
                                                    # 4 bytes, unused, 4x
                           'external_time'       ,  # 5 bytes, uint64, Q (3 extra bytes)
                           'control_field'       ,  # 1 byte, B
                           'test_params'         ,  # 1 byte, B
                                                    # 6 bytes, unused, 6x
                           'num_rows'            ,  # 2 bytes, uint16, H
                           'num_rows_reported'   ,  # 2 bytes, uint16, H
                                                    # 4 bytes, unused, 4x
                           'row_length'          ,  # 2 bytes, uint16, H
                           'data_rate'              # 2 bytes, uint16, H
                                                    # 4 bytes, unused, 4x
                         ] )

# Default header as a named tuple
RogueHeader = namedtuple( 'RogueHeader',
                         [ 'size'                ,  # 4 Bytes, uint32, I
                           'flags'               ,  # 2 bytes, uint16, H
                           'error'               ,  # 1 bytes, uint8,  B
                           'channel'                # 1 bytes, uint8,  B
                         ] )



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
                    printf(f"Waring: File under run reading {self._currFName}")
                    return False

                # Read in Rogue header data
                rogueHeader = RogueHeader._make(struct.Struct(RogueHeaderPack).unpack(self._currFile.read(RogueHeaderSize)))
                payload = rogueHeader.size - 4
            
                # Set next frame position
                recEnd = self._currFile.tell() + payload

                # Sanity check
                if recEnd > self._fileSize:
                    printf(f"Waring: File under run reading {self._currFName}")
                    return False

                # If this is a data channel, break
                if rogueHeader.channel == 0: break

                # Process meta data
                elif self._metaData and rogueHeader.channel == 1:
                    try:
                        yamlUpdate(self._config, self._currFile.read(payload).decode('utf-8'))
                    except:
                        print(f"Waring: Error processing meta data in {self._currFName}")

                # Skip over meta data
                else: self._currFile.seek(recEnd)

            # Check if there is enough room for the Smurf header
            if SmurfHeaderSize > payload:
                print(f"Waring: SMURF header overruns remaining record size in {self._currFName}")
                return False

        # Non Rogue file
        elif SmurfHeaderSize + self._currFile.tell() > self._fileSize:
            print(f"Warning: SMURF header overruns remaining file size in {self._currFName}")
            return None

        # Unpack header into named tuple
        self._header = SmurfHeader._make(struct.Struct(SmurfHeaderPack).unpack(self._currFile.read(SmurfHeaderSize)))

        # Use the forced channel count if it is provided, otherwise use the header count
        if self._chanCount is not None:
            chanCount = self._chanCount
        else:
            chanCount = self._header.number_of_channels

        # Compute payload size of frame based upon channel count
        expSize = chanCount * SmurfChannelSize

        # Processing rogue frame, verify container 
        if self._isRogue:
            if expSize + self._currFile.tell() != recEnd:
                print(f"Warning: SMURF data does not align to frame record in {self._currFName}")
                return False

        # Non rogue frame
        elif expSize + self._currFile.tell() > self._fileSize:
            print(f"SMURF data overruns remaining file size in {self._currFName}")
            return False

        # Read record data
        self._data = numpy.fromfile(self._currFile, dtype=numpy.int32, count=chanCount)
        self._currCount += 1
        self._totCount += 1

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


