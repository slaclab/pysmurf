#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF DataFromFile
#-----------------------------------------------------------------------------
# File       : _DataFromFile.py
# Created    : 2019-11-15
#-----------------------------------------------------------------------------
# Description:
#    Stream data from a file
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
import time

import pyrogue
import rogue.interfaces.stream

class DataFromFile(pyrogue.Device):
    """
    Class to stream data from a test file.

    Read data points from a file, and send them in SMuRF frames.

    Args
    ----
    name : str
        Device name.
    description : str
        Device description.
    dataSize : int, optional, default 16
        Data size in bits to use in the frame
    """

    def __init__(self,
                 name="DataFromFile",
                 description="Data from file source",
                 dataSize=16,
                 **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)
        self._data_master = DataMaster(dataSize=dataSize)

        self.add(pyrogue.LocalVariable(
            name='FileName',
            description='Path to the data file',
            mode='RW',
            value='/tmp/fw/x.dat'))

        self.add(pyrogue.LocalVariable(
            name='FrameCnt',
            description='Number of sent frames',
            mode='RO',
            value=0,
            localGet = self._data_master.get_frame_cnt))

        self.add(pyrogue.LocalCommand(
            name='SendData',
            description='Send data',
            function=self._send_data))

    def _send_data(self):
        """
        Method to send data from the specified text file.
        """
        file_name = self.FileName.get()
        self._data_master.send_data(file_name=file_name)

    def _getStreamMaster(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access master.
        """
        return self._data_master



class DataMaster(rogue.interfaces.stream.Master):
    """
    A Rogue master device, used to stream the data.

    Args
    ----
    dataSize : int
        Data size in bits to use in the frame
    """

    def __init__(self, dataSize):
        super().__init__()
        self._frame_cnt=0
        self._data_size = dataSize

    def get_frame_cnt(self):
        """
        Get the number of sent frames
        """
        return self._frame_cnt

    def send_data(self, file_name):
        """
        Send all the data from a text file. The input data file,
        must be a text file with data points on each line. The data
        must be of size 'dataSize'.
        Each line can have multiple values, separated by spaces. All
        the values in each line are send on a frame with the SMuRF
        header, with each values on a channel.

        Args
        ----
        file_name : str
            Path to the input data file.
        """
        if not file_name:
            print("ERROR: Must define a data file first!")
            return

        try:
            with open(file_name, 'r') as f:
                for data in f:
                    self.sendData(data=list(map(int, data.split())))
                    time.sleep(0.01)

        except IOError:
            print("Error trying to open {file_name}")

    # Method for generating a frame
    def sendData(self, data):
        """
        Send a Rogue Frame. The frame contains the SMuRF header and the
        input data points in contiguous channels. The SMuRF header will
        be only partially filled, containing only the number of channels,
        and a the frame counter words.

        Args
        ----
        data : list
            Input data (must be of size 'dataSize').
        """

        # Data size in bytes
        data_size_bytes = int(self._data_size / 8)

        # Request a frame to hold an SMuRF frame
        frame = self._reqFrame(128+data_size_bytes*4096, True)

        # Fill the frame with zeros
        frame.write( bytearray([0]*(128+data_size_bytes*4096)), 0 )

        # Write the number of channels
        frame.write( bytearray((4096).to_bytes(4, sys.byteorder)), 4)

        # Write the frame counter into the header
        frame.write( bytearray(self._frame_cnt.to_bytes(4, sys.byteorder)), 84)

        # Write the data into the first channel
        index = 128 # This is the start of the data area in a SMuRF frame
        for d in data:
            frame.write( bytearray(d.to_bytes(data_size_bytes, sys.byteorder, signed=True)), index)
            index += data_size_bytes

        # Send the frame
        self._sendFrame(frame)

        # Update the frame counter
        self._frame_cnt =  self._frame_cnt + 1
