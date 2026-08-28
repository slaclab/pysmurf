#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Serial Gradient Descent Process
#-----------------------------------------------------------------------------
# File       : _NewSerialGradientDescent.py
# Created    : 2019-10-09
#-----------------------------------------------------------------------------
# Description:
#    Moved verbatim from cryo-det, where it lived at
#    firmware/python/CryoDet/DspCoreLib/CryoDetCmbHcd/_NewSerialGradientDescent.py
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
__all__ = ['NewSerialGradientDescent']

class NewSerialGradientDescent(pr.Process):
    def __init__(self, **kwargs):
        pr.Process.__init__(self)

    def _process(self):
        self.parent.etaScanInProgress.set( 1 )
        self.Message.setDisp("Started")

        # measure the gradient from samples on either side of current frequency
        def calcGrad(channel, centerFrequencyMHz, df, numAverages):
            # references to variables for this channel
            centerFreqVar = self.parent.CryoChannel[channel].centerFrequencyMHz
            freqErrorVar = self.parent.CryoChannel[channel].frequencyErrorMHz
            etaPhaseVar = self.parent.CryoChannel[channel].etaPhaseDegree

            # measure at positive offset
            centerFreqVar.set(centerFrequencyMHz + df)
            posResp = np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])
            etaPhaseVar.set(90)
            posResp += 1j * np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])

            # measure at negative offset
            centerFreqVar.set( centerFrequencyMHz - df )
            negResp = np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])
            etaPhaseVar.set(0)
            negResp += 1j * np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])

            grad = (np.abs(posResp) - np.abs(negResp)) / (2 * df)

            return grad

        # defer update callbacks
        with self.root.updateGroup():
            self.Message.setDisp("Init")

            # optimisation parameters
            maxIters    = self.parent.gradientDescentMaxIters.get()
            numAverages = self.parent.gradientDescentAverages.get()
            initialStep = self.parent.gradientDescentStepHz.get()*1e-6
            converge    = self.parent.gradientDescentConvergeHz.get()*1e-6
            debug       = self.parent.debug.get()

            # current state after running a coarse resonator search
            amplitudeScale = self.parent.amplitudeScale.get()
            freq           = self.parent.centerFrequencyMHz.get()

            # turn all channels off
            self.parent.feedbackEnable.set(np.zeros(512, dtype=np.uint))
            self.parent.amplitudeScale.set(np.zeros(512, dtype=np.uint))

            # measure real and imaginary
            self.parent.etaMag.set(np.ones(512))
            self.parent.etaPhase.set(np.zeros(512))

            # only refine channels with a resonatory identified
            channels = np.where(np.asarray(amplitudeScale) != 0)[0]

            # gradient descent using the Barzilai–Borwein method
            def gradDescent(channel):
                # start the optimisation from current state
                start_freq = freq[channel]

                # use a fixed step to start
                step = initialStep
                f_last, f = start_freq, start_freq + step
                g_last = calcGrad(channel, f_last, step, numAverages)
                g = calcGrad(channel, f, step, numAverages)

                i = 0
                complete = False
                while i < maxIters:
                    # scale learning rate and compute next step
                    df, dg = (f - f_last), (g - g_last)
                    if (df == 0) or (dg == 0):
                        step = 0
                    else:
                        alpha = np.abs(df / dg)
                        step = -1 * alpha * g
                    # check for convergence
                    if np.abs(step) < converge:
                        complete = True
                        break
                    # update position
                    f_last, g_last = f, g
                    f += step
                    g = calcGrad(channel, f, step, numAverages)
                    if debug > 0:
                        self._log.info(f"Iter {i}: step={step}")
                        self._log.info(f"          grad={g}")
                    i += 1

                if not complete:
                    self._log.warning(f"Gradient descent failed to converge on channel {channel}.")
                elif debug > 0:
                    self._log.info(f"Converged after {i} iterations.")

                return complete, f

            tot = len(channels)
            i = 0
            for channel in channels:
                self.Message.setDisp(f"Tuning Channel {channel}")
                self.Progress.set(i / tot)
                i += 1

                if debug > 0:
                    self._log.info(" ")
                    self._log.info("Tuning channel " + str(channel))
                    self._log.info(" ")

                # turn on this channel
                self.parent.CryoChannel[channel].amplitudeScale.set( amplitudeScale[channel] )

                # run the gradient descent on this channel
                complete, f = gradDescent(channel)

                # check result is in range
                if (f > 1.2) or (f < -1.2):
                    # reset to its starting point
                    self.parent.CryoChannel[channel].centerFrequencyMHz.set(freq[channel])
                    self._log.warning(f"Channel {channel} is out of range. Skipping.")
                    continue

                if complete:
                    self.parent.CryoChannel[channel].centerFrequencyMHz.set(f)
                else:
                    # reset to its starting point
                    self.parent.CryoChannel[channel].centerFrequencyMHz.set(freq[channel])
                    self._log.warning(f"Center frequency for channel {channel} not updated.")

                self.parent.CryoChannel[channel].amplitudeScale.set( 0 )

            # reset tone power to initial state
            self.parent.amplitudeScale.set( amplitudeScale )

        self.Progress.set(1.0)
        self.Message.setDisp(f"Done")
        self.parent.etaScanInProgress.set( 0 )
