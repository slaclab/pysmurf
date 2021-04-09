#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Frame Generator
#-----------------------------------------------------------------------------
# File       : _FrameGenerator.py
# Created    : 2021-04-09
#-----------------------------------------------------------------------------
# Description:
#    This class defines a master device that generates single SMuRF frames
#     containing data passed by the user.
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

class FrameGenerator(pyrogue.Device):
    """
    Class to generate single SMuRF frame containing data passed by the user.

    Once the device is instantiated and started on a Rogue root, the user
    can call the 'SendData' command, passing as an argument a list of data
    points to generate and send a SMuRF frame. The frame will contain the
    passed data, in contiguous channels starting from the first one. The
    frame however, will be of fixed size containing all 4096 maximum
    channels; unused channel will contain zeros.

    Args
    ----
    name : str, option, default "FrameGenerator"
        Device name.
    description : str, option, default "Frame generator data source"
        Device description.
    dataSize : int, optional, default 16
        Data size in bits to use in the frame
    """
    def __init__(self,
         name="FrameGenerator",
         description="Frame generator data source",
         dataSize=16,
         **kwargs):
        pyrogue.Device.__init__(self, name=name, description=description, **kwargs)
        self._data_master = self._DataMaster(dataSize=dataSize)

        self.add(pyrogue.LocalVariable(
            name='FrameCnt',
            description='Number of sent frames',
            mode='RO',
            value=0,
            localGet = self._data_master.get_frame_cnt))

        self.add(pyrogue.LocalCommand(
            name='SendData',
            description='Send data pass as an argument',
            value='',
            function=lambda arg: self._data_master.sendData(data=arg)))

    def _getStreamMaster(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access
        the master device.
        """
        return self._data_master

    class _DataMaster(rogue.interfaces.stream.Master):
        """
        A Rogue master device, used to stream the data.

        Args
        ----
        dataSize : int
            Data size in bits to use in the frame
        """
        def __init__(self, dataSize):
            super().__init__()
            self._frame_cnt = 0
            self._data_size = dataSize

        def get_frame_cnt(self):
            """
            Get the number of sent frames
            """
            return self._frame_cnt

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
            for d in list(map(int, data.split())):
                frame.write( bytearray(d.to_bytes(data_size_bytes, sys.byteorder, signed=True)), index)
                index += data_size_bytes

            # Send the frame
            self._sendFrame(frame)

            # Update the frame counter
            self._frame_cnt =  self._frame_cnt + 1
