#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF DataToFile
#-----------------------------------------------------------------------------
# File       : _StreamDataEmulator.py
# Created    : 2019-11-15
#-----------------------------------------------------------------------------
# Description:
#    Receive a stream and write the data to disk
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import sys

import pyrogue
import rogue.interfaces.stream

class DataToFile(pyrogue.Device):
    """
    Class to write data to a file.

    Extract data from a SMuRF frame and writes it to a file.

    Args
    ----
    name : str
        Device name.
    description : str
        Device description.
    dataSize : int, optional, default 32
        Data size in bits to use when reading the data from the frame.
    numSamples : int, optional, default 1
        Number of samples to extract from the frame.
    """
    def __init__(self,
                 name="DataToFile",
                 description="Data to file writer",
                 dataSize=32,
                 numSamples=1,
                 **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)
        self._data_slave = DataSlave(dataSize=dataSize, numSamples=numSamples)
        self._meta_slave = MetaSlave()

        self.add(pyrogue.LocalVariable(
            name='FileName',
            description='Path to the data file',
            mode='RW',
            value='/tmp/fw/y.dat'))

        self.add(pyrogue.LocalCommand(
            name='WriteData',
            description='Write data to disk',
            function=self._write_data))

    def _write_data(self):
        """
        Method to write the data to the specified text file.
        """
        file_name = self.FileName.get()
        self._data_slave.write_data(file_name=file_name)

    def getDataChannel(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access slave.
        This is method is called to request the data channel.
        """
        return self._data_slave

    def getMetaChannel(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access slave.
        This is method is called to request the metadata channel.
        """
        return self._meta_slave

    def _getStreamSlave(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access master.
        """
        return self._data_slave

class DataSlave(rogue.interfaces.stream.Slave):
    """
    A Rogue slave device, used receive a stream of data and write it to disk.

    Args
    ----
    dataSize : int
        Data size in bits to use when reading the data from the frame.
    numSamples : int
        Number of samples to extract from the frame.
    """
    def __init__(self, dataSize, numSamples):
        super().__init__()
        self._data = []
        self._data_size = dataSize
        self._num_samples = numSamples

    def write_data(self, file_name):
        """
        Method to write the data buffer to a text file. Writes the
        content of the data buffer (self._data) to the output file,
        one value on each line as text.
        Each value in the data buffer can contain multiple values,
        separated by spaces.

        Args
        ----
        file_name : str
            Path to the output data file.
        """
        if not file_name:
            print("ERROR: Must define a data file first!")
            return

        try:
            with open(file_name, 'w') as f:
                for datum in self._data:
                    f.write(f'{str(datum)}\n')

        except IOError:
            print("Error trying to open {file_name}")

    def _acceptFrame(self, frame):
        """
        Receive a frame with SMuRF data. The first channel is appended
        to the data buffer.

        Args
        ----
        frame : rogue.interfaces.stream.Frame
            A frame with SMuRF data.
        """

        # Data byte in bytes
        data_size_bytes = int(self._data_size / 8)

        with frame.lock():
            data = bytearray(data_size_bytes)

            index = 128 # This is the start of the data area in a SMuRF frame
            datum = ''  # This string will hold all the values extracted from the frame

            # Extract all the values from the frame
            for i in range(self._num_samples):
                # Read a value
                frame.read(data, index + i*data_size_bytes)

                # Convert it to int
                data_int = int.from_bytes(bytes(data), byteorder=sys.byteorder, signed=True)

                # Append the value to the sting
                datum = f'{datum} {data_int}'

            # Write the resulting string to the data buffer
            self._data.append(datum)

            #try:
            #    with open(self._file_name, 'a+') as f:
            #        f.write(f'{str(data_int)}\n')
            #except IOError:
            #    pass


class MetaSlave(rogue.interfaces.stream.Slave):
    """
    A Rogue slave device, used to connect to the metadata channel.
    """
    def __init__(self):
        super().__init__()

    def _acceptFrame(self, frame):
        """
        Receive a frame with metadata. The frame is discarded.

        Args
        ----
        frame : rogue.interfaces.stream.Frame
            A frame with metadata.
        """
        pass
