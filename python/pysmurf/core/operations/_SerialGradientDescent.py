#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Serial Gradient Descent Process
#-----------------------------------------------------------------------------
# File       : _SerialGradientDescent.py
# Created    : 2019-10-09
#-----------------------------------------------------------------------------
# Description:
#    Moved verbatim from cryo-det, where it lived at
#    firmware/python/CryoDet/DspCoreLib/CryoDetCmbHcd/_SerialGradientDescent.py
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
import numpy as np
import pyrogue as pr

# Restrict 'from ... import *' to the class, so the module imports above
# do not leak into the pysmurf.core.operations package namespace
__all__ = ['SerialGradientDescent']

class SerialGradientDescent(pr.Process):
    def __init__(self, **kwargs):
        pr.Process.__init__(self)

    def _process(self):
        self.parent.etaScanInProgress.set( 1 )
        self.Message.setDisp("Started")

        def calcGrad(centerFreqVar, freqErrorVar, etaPhaseVar, centerFrequencyMHz, df, numAverages):
            centerFreqVar.set( centerFrequencyMHz + df )
            posResp = np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])
            etaPhaseVar.set( 90 )
            posResp += 1j*np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])

            centerFreqVar.set( centerFrequencyMHz - df )
            negResp = np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])
            etaPhaseVar.set( 0 )
            negResp += 1j*np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])

            grad = ( np.abs(posResp) - np.abs(negResp) ) / ( 2*df )

            return grad

        # defer update callbacks
        with self.root.updateGroup():
            self.Message.setDisp("Init")
            iters       = 0
            maxIters    = self.parent.gradientDescentMaxIters.get()
            gain        = self.parent.gradientDescentGain.get()
            numAverages = self.parent.gradientDescentAverages.get()
            momentum    = self.parent.gradientDescentMomentum.get()
            initialStep = self.parent.gradientDescentStepHz.get()*1e-6
            converge    = self.parent.gradientDescentConvergeHz.get()*1e-6
            beta        = self.parent.gradientDescentBeta.get()
            debug       = self.parent.debug.get()

            amplitudeScale = self.parent.amplitudeScale.get()
            freq           = self.parent.centerFrequencyMHz.get()

            self.parent.feedbackEnable.set(np.zeros(512, dtype=np.uint))
            self.parent.amplitudeScale.set(np.zeros(512, dtype=np.uint))

            self.parent.etaMag.set( np.ones(512) )
            self.parent.etaPhase.set( np.zeros(512) )

            channels = np.where( np.asarray(amplitudeScale) != 0 )

            tot = len(channels[0])
            i = 0
            for channel in channels[0]:
                self.Message.setDisp(f"Tuning Channel {channel}")
                self.Progress.set(i / tot)
                i += 1

                if debug > 0:
                    self._log.info(" ")
                    self._log.info("Tuning channel " + str(channel))
                    self._log.info(" ")

                prevDf = 0
                currDf = 0
                self.parent.CryoChannel[channel].amplitudeScale.set( amplitudeScale[channel] )

                v     = 0
                iters = 0
                cache = 0
                complete = False
                while (iters < maxIters):
                    prevDf = currDf
                    dx     = calcGrad( self.parent.CryoChannel[channel].centerFrequencyMHz,
                                       self.parent.CryoChannel[channel].frequencyErrorMHz,
                                       self.parent.CryoChannel[channel].etaPhaseDegree,
                                       freq[channel] + currDf,
                                       initialStep,
                                       numAverages ) # center difference

                    if momentum == 1:
                        v = beta*v + (1-beta)* gain * dx
                        currDf -= v
                    else:
                        #cache   += dx**2
                        cache   = beta * cache + (1 - beta) * dx**2
                        currDf -= gain * dx / np.sqrt( cache + 1e-8 )
                    step   = currDf - prevDf
                    if debug > 0:
                        self._log.info("Grad is " + str(dx) + ", step is " + str(step) + " currDf is " + str(currDf))
                    if np.abs(step) < converge:
                        complete = True
                        self.parent.CryoChannel[channel].centerFrequencyMHz.set( freq[channel] + currDf )
                        break
                    iters += 1

                    if ( ( ( freq[channel] + currDf ) > 1.2 ) | ( ( freq[channel] + currDf ) < -1.2 ) ):
                        self.parent.CryoChannel[channel].centerFrequencyMHz.set( freq[channel] )
                        self._log.info("Channel " + str(channel) + " out of range.")
                        break

                if debug > 0:
                    if not complete:
                        self._log.warning("Channel " + str(channel) + " failed to converge")
                    else:
                        self._log.info("Channel " + str(channel) + " converged after " + str(iters) + " iterations")

                self.parent.CryoChannel[channel].amplitudeScale.set( 0 )

            self.parent.amplitudeScale.set( amplitudeScale )

        self.Progress.set(1.0)
        self.Message.setDisp(f"Done")
        self.parent.etaScanInProgress.set( 0 )
