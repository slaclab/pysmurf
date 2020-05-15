#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : SmurfProcessor's Profile Script
#-----------------------------------------------------------------------------
# File       : profile_smurf_processor.py
# Created    : 2020-05-15
#-----------------------------------------------------------------------------
# Description:
# Script to profile the SmurfProcessor.
# In order to get profile data, the SMuRF processor needs to be modified adding
# TimerWithStats object in the appropriated places.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import sys
import time
import rogue.interfaces.stream
import pyrogue
import smurf
import pysmurf
import pysmurf.core
import pysmurf.core.devices

class LocalRoot(pyrogue.Root):
    """
    Local root device. It contains the SmurfProcessor, connected to
    a FrameStatistics and a StreamDataSource.
    """
    def __init__(self, **kwargs):
        pyrogue.Root.__init__(self, name="AMCc", initRead=True, pollEn=True, **kwargs)

        # Use the StreamDataSource as a data source
        self._streaming_stream = pysmurf.core.emulators.StreamDataSource()
        self.add(self._streaming_stream)

        # Add a frame statistics module to count the number of generated frames
        self._smurf_frame_stats = pysmurf.core.counters.FrameStatistics(name="FrameRxStats")
        self.add(self._smurf_frame_stats)

        # Add the SmurfProcessor
        # The SmurfProcessor needs con container 'TimerWithStats' objects in it
        # in order to get output profile data.
        self._smurf_processor = smurf.core.processors.SmurfProcessor()

        # Configure the SmurfProcessor
        self._smurf_processor.setMask([0]*2000)
        self._smurf_processor.setOrder(4)
        self._smurf_processor.setA([  1.0,
                     -3.74145562,
                      5.25726624,
                     -3.28776591,
                      0.77203984 ] + [0] * 11,)
        self._smurf_processor.setB([ 5.28396689e-06,
                     2.11358676e-05,
                     3.17038014e-05,
                     2.11358676e-05,
                     5.28396689e-06 ] + [0] * 11)
        #self._smurf_processor.setUnwrapperDisable(True);

        print(f"Number of channels = {self._smurf_processor.getNumCh()}")
        print(f"Filter order       = {self._smurf_processor.getOrder()}")
        print(f"Filter gain        = {self._smurf_processor.getGain()}")
        print(f"Filter A           = {self._smurf_processor.getA()}")
        print(f"Filter B           = {self._smurf_processor.getB()}")

        # Connect the StreamDataSource data source to the FrameRxStats
        # and the FrameRxStats to the SmurfProcessor
        pyrogue.streamConnect(self._streaming_stream, self._smurf_frame_stats)
        pyrogue.streamConnect(self._smurf_frame_stats, self._smurf_processor)


if __name__ == "__main__":

    with LocalRoot() as root:
        # Start the generation of frames
        root.StreamDataSource.Period.set(1e-4)
        root.StreamDataSource.SourceEnable.set(True)

        # Wait until we have received 100k frames. This assumes the
        # 'TimerWithStats' objects where created to measure 100k samples.
        while root.FrameRxStats.FrameCnt.get() < 100000:
            time.sleep(5.0)

        print(f"Final number of received frames = {self.FrameRxStats.FrameCnt.get()}")
        print(f"Number of lost frames           = {self.FrameRxStats.FrameLossCnt.get()}")