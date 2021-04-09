#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Data Master
#-----------------------------------------------------------------------------
# File       : _DataMaster.py
# Created    : 2021-04-09
#-----------------------------------------------------------------------------
# Description:
#    This class defines a Rogue stream master that generates single SMuRF frames
#     containing data passed by the user. It is used by other emulator devices
#     defined in this package.
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


class DataMaster(rogue.interfaces.stream.Master):
    """
    A Rogue stream master to generate SMuRF frames.

    It will generate and send a SMuRF frame when the 'sendData' method is
    called. The frame will contain the list of data points passed by the
    user in contiguous channels starting from the first one. The
    frame however, will be of fixed size containing all 4096 maximum
    channels; unused channel will contain zeros.

    Args
    ----
    dataSize : int, optional, default 16
        Data size in bits to use in the frame
    """
    def __init__(self, dataSize=16):
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
        for d in data:
            frame.write( bytearray(d.to_bytes(data_size_bytes, sys.byteorder, signed=True)), index)
            index += data_size_bytes

        # Send the frame
        self._sendFrame(frame)

        # Update the frame counter
        self._frame_cnt =  self._frame_cnt + 1
