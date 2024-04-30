#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Data Receiver
#-----------------------------------------------------------------------------
# File       : _SmurfDataReceiver.py
# Created    : 2024-04-30
#-----------------------------------------------------------------------------
# Description:
#    SMuRF Data Receiver
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import pyrogue as pr
import numpy as np

class SmurfDataReceiver(py.DataReceiver):

    def __init__(self, rxSize, rxType, **kwargs):

        if rxType == 'Int16':
            typeStr='Int16[np]'
            value=numpy.zeros(shape=rxSize, dtype=numpy.int16, order='C'),
            self._func = self._process15

        elif rxType == 'Int8':
            typeStr='Int8[np]'
            value=numpy.zeros(shape=rxSize, dtype=numpy.int8, order='C'),
            self._func = self._process8

        else:
            raise pr.VariableError(f"Invalid rxType: {rxType} passed")

        pr.DataReceiver(self, typeStr=typeStr, hideData=True, value=value, enableOnStart=True, **kwargs)

    def _process16(self,frame):

        # Get data from frame
        fl = frame.getPayload()
        datRaw = frame.getNumpy(0,fl)  # uint8

        # Convert to 16-bit signed numpy
        dat16 = np.array(datRaw, np.int16)

        # Update data
        self.Data.set(dat,write=True)
        self.Updated.set(True,write=True)


    def _process8(self,frame):

        # Get data from frame
        fl = frame.getPayload()
        datRaw = frame.getNumpy(0,fl)  # uint8

        # Convert to 8-bit signed numpy
        dat = np.array(datRaw, np.int8)

        # Update data
        self.Data.set(dat,write=True)
        self.Updated.set(True,write=True)

    def process(self,frame):
        self._func(frame)

