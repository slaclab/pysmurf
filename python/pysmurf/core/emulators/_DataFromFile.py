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

import time

import pyrogue

from pysmurf.core.emulators._DataMaster import DataMaster

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
        Send all the data from a text file. The input data file,
        must be a text file with data points on each line. The data
        must be of size 'dataSize'.
        Each line can have multiple values, separated by spaces. All
        the values in each line are send on a frame with the SMuRF
        header, with each values on a channel.
        """
        file_name = self.FileName.get()

        if not file_name:
            print("ERROR: Must define a data file first!")
            return

        try:
            with open(file_name, 'r') as f:
                for l in f:
                    self._data_master.sendData(data=list(map(int, l.split())))
                    time.sleep(0.01)

        except IOError:
            print("Error trying to open {file_name}")

    def _getStreamMaster(self):
        """
        Method called by streamConnect, streamTap and streamConnectBiDir to access
        the master device.
        """
        return self._data_master
