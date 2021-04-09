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

import pyrogue

from pysmurf.core.emulators._DataMaster import DataMaster


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
        self._data_master = DataMaster(dataSize=dataSize)

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
