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
import rogue.interfaces.stream
import pyrogue

class DataToFile(pyrogue.Device):
    """
    Class to write data to a file
    """
    def __init__(self, name="DataToFile", description="Data to file writer", **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)
        self._data_slave = DataSlave()
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

class DataSlave(rogue.interfaces.stream.Slave):
    """
    A Rogue slave device, used receive a stream of data and write it to disk.
    """
    def __init__(self):
        super().__init__()
        self._data = []

    def write_data(self, file_name):
        """
        Method to write the data buffer to a text file. Writes the
        content of the data buffer (self._data) to the output file,
        one data point on each line as text.

        Args:
        -----
        - file_name (str) : path to the output data file.
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
        Args:
        Receive a frame with SMuRF data. The first channel is appended
        to the data buffer.

        -----
        frame (rogue.interfaces.stream.Frame) : a frame with SMuRF data.
        """
        with frame.lock():
            data = bytearray(4)

            frame.read(data, 128)
            self._data.append(int.from_bytes(bytes(data), byteorder=sys.byteorder, signed=True))

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

        Args:
        -----
        frame (rogue.interfaces.stream.Frame) : a frame with metadata
        """
        pass
