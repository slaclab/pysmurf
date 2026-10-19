#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Debug Capture Receivers
#-----------------------------------------------------------------------------
# File       : _capture.py
# Created    : 2024-04-30
#-----------------------------------------------------------------------------
# Description:
#    Where a debug capture lands: one rogue DataReceiver per DDR capture stream,
#    publishing the last frame as an array a client reads after triggering the
#    firmware's acquisition mux. `add_capture_receivers` puts one per stream on
#    the tree, each behind a FIFO sized to its buffer; a server built without them
#    has no capture nodes at all, which the platform map allows for.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import logging

import numpy as np
import pyrogue as pr
import rogue.interfaces.stream

__all__ = ['SmurfDataReceiver', 'add_capture_receivers', 'CAPTURE_POINTS']

_log = logging.getLogger(__name__)

# Points each capture receiver buffers in a server built with them: the DDR
# capture depth a client's ADC and DAC reads expect.
CAPTURE_POINTS = 2**19


def add_capture_receivers(root, transport, size: int, dtype: str) -> None:
    """One receiver per DDR stream, behind a FIFO sized to it.

    Parameters
    ----------
    root : ReadoutRoot
        The tree, not yet started; its map names the receivers.
    transport : Transport
        Whose ``ddr_streams`` are received.
    size : int
        Points each receiver buffers.
    dtype : str
        ``'Int16'`` or ``'Int8'``, the sample type.
    """
    _log.info("enabling capture receivers (buffer size = %d points, data type = %s)", size, dtype)
    bytes_per_point = 2 if '16' in dtype else 1
    root._stream_fifos, root._stream_slaves = [], []
    for i, ddr in enumerate(transport.ddr_streams):
        name = root.pmap.path(f'stream.capture[{i}].data').split('.')[-2]
        receiver = SmurfDataReceiver(name=name, rxSize=size, rxType=dtype)
        fifo = rogue.interfaces.stream.Fifo(1000, size * bytes_per_point, True)
        root.add(receiver)
        pr.streamConnect(fifo, receiver)
        pr.streamConnect(ddr, fifo)
        root._stream_slaves.append(receiver)
        root._stream_fifos.append(fifo)


class SmurfDataReceiver(pr.DataReceiver):

    def __init__(self, rxSize, rxType, **kwargs):

        if rxType == 'Int16':
            typeStr='Int16[np]'
            value=np.zeros(shape=rxSize, dtype=np.int16, order='C')
            self._func = self._process16

        elif rxType == 'Int8':
            typeStr='Int8[np]'
            value=np.zeros(shape=rxSize, dtype=np.int8, order='C')
            self._func = self._process8

        else:
            raise pr.VariableError(f"Invalid rxType: {rxType} passed")

        super().__init__(typeStr=typeStr, hideData=True, value=value, enableOnStart=True, **kwargs)

    def _process16(self,frame):

        # Get data from frame
        fl = frame.getPayload()
        datRaw = frame.getNumpy(0,fl)  # uint8

        # Convert to 16-bit signed numpy
        dat = datRaw.view(np.int16)

        # Update data
        self.Data.set(dat,write=True)
        self.Updated.set(True,write=True)


    def _process8(self,frame):

        # Get data from frame
        fl = frame.getPayload()
        datRaw = frame.getNumpy(0,fl)  # uint8

        # Convert to 8-bit signed numpy
        dat = datRaw.view(np.int8)

        # Update data
        self.Data.set(dat,write=True)
        self.Updated.set(True,write=True)

    def process(self,frame):
        self._func(frame)
