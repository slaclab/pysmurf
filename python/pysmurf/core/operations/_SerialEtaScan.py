#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Serial Eta Scan Process
#-----------------------------------------------------------------------------
# File       : _SerialEtaScan.py
# Created    : 2019-10-09
#-----------------------------------------------------------------------------
# Description:
#    Moved verbatim from cryo-det, where it lived at
#    firmware/python/CryoDet/DspCoreLib/CryoDetCmbHcd/_SerialEtaScan.py
#    as of commit 31b6fbfe (== tag MicrowaveMuxBpEthGen2_v2.5.1).
#
#    The class body below is byte-identical to that file. Only this header and
#    the comment above __all__ differ. Do not "clean up" this module: the
#    byte-for-byte match with cryo-det is what proves the move is behaviour-
#    preserving. See docs/stage1_ops_out_of_cryo_det.md.
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

import numpy as np
import pyrogue as pr

# Restrict 'from ... import *' to the class, so the module imports above
# do not leak into the pysmurf.core.operations package namespace
__all__ = ['SerialEtaScan']

class SerialEtaScan(pr.Process):
    def __init__(self, **kwargs):
        pr.Process.__init__(self)

    def _process(self):
        self.parent.etaScanInProgress.set( 1 )
        self.Message.setDisp("Started")

        # defer update callbacks
        with self.root.updateGroup():
            self.Message.setDisp("Init")

            delF            = self.parent.etaScanDelF.get()*1e-6
            numAverages     = self.parent.etaScanAverages.get()
            amplitudeScale  = self.parent.amplitudeScale.get()
            freq            = self.parent.centerFrequencyMHz.get()
            eta_max         = self.parent.etaScanMaxMag.get()

            self.parent.feedbackEnable.set(np.zeros(512, dtype=np.uint))
            self.parent.etaMag.set( np.ones(512) )
            self.parent.etaPhase.set( np.zeros(512) )
            self.parent.amplitudeScale.set( np.zeros(512, dtype=np.uint) )

            channels = np.where( amplitudeScale != 0 )

            tot = len(channels[0])
            i = 0
            failed_ch = []  # keep track of channels where eta estimation fails
            for channel in channels[0]:
                self.Message.setDisp(f"Channel {channel}")
                self.Progress.set(i/tot)
                i += 1

                self.parent.CryoChannel[channel].amplitudeScale.set( amplitudeScale[channel] )

                imag = np.mean([self.parent.CryoChannel[channel].frequencyErrorMHz.get(read=True) for _ in range(numAverages)])

                self.parent.CryoChannel[channel].etaPhaseDegree.set( -90 )
                real = np.mean([self.parent.CryoChannel[channel].frequencyErrorMHz.get(read=True) for _ in range(numAverages)])
                etaPhaseDegree = np.arctan2(imag, real)*180/np.pi

                if etaPhaseDegree > 180:
                    etaPhaseDegree = etaPhaseDegree - 360
                elif etaPhaseDegree < -180:
                    etaPhaseDegree = etaPhaseDegree + 360

                self.parent.CryoChannel[channel].etaPhaseDegree.set( etaPhaseDegree )

                self.parent.CryoChannel[channel].centerFrequencyMHz.set( freq[channel] + delF )
                posError = np.mean([self.parent.CryoChannel[channel].frequencyErrorMHz.get(read=True) for _ in range(numAverages)])
                self.parent.CryoChannel[channel].centerFrequencyMHz.set( freq[channel] - delF )
                negError = np.mean([self.parent.CryoChannel[channel].frequencyErrorMHz.get(read=True) for _ in range(numAverages)])
                self.parent.CryoChannel[channel].centerFrequencyMHz.set( freq[channel] )

                try:
                    etaMagScaled = 2*delF / ( posError - negError )
                    if etaMagScaled < 0:
                        etaMagScaled = np.abs( etaMagScaled )
                        if etaPhaseDegree > 0:
                            etaPhaseDegree = etaPhaseDegree - 180
                        else:
                            etaPhaseDegree = etaPhaseDegree + 180

                except Exception as e:
                    self._log.warning(f"Channel {channel} raised {type(e).__name__}: {e}. Clipping to 1 and turning off tone power")
                    etaMagScaled = 1.0
                    failed_ch.append(channel)

                if not np.isfinite(etaMagScaled):
                    self._log.warning(f"Channel {channel}: eta not a number. Clipping to 1 and turning off tone power")
                    etaMagScaled = 1.0
                    failed_ch.append(channel)
                elif etaMagScaled > eta_max:
                    self._log.warning(f"Channel {channel}: eta = {etaMagScaled}. Clipping to {eta_max}.")
                    etaMagScaled = eta_max

                self.parent.CryoChannel[channel].etaPhaseDegree.set( etaPhaseDegree )
                self.parent.CryoChannel[channel].etaMagScaled.set( etaMagScaled )

            amplitudeScale[failed_ch] = 0
            self.parent.amplitudeScale.set( amplitudeScale )
            time.sleep(0.1)

            for channel in channels[0]:
                if channel not in failed_ch:
                    self.parent.CryoChannel[channel].feedbackEnable.set( 1 )

        self.Progress.set(1.0)
        self.Message.setDisp(f"Done")
        self.parent.etaScanInProgress.set( 0 )
