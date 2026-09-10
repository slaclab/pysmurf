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
#    as of commit e3dc359c (main after PR #80). The move was taken from
#    31b6fbfe (== tag MicrowaveMuxBpEthGen2_v2.5.1) and #80's changes applied
#    on top as a separate commit, so the two can be reviewed apart.
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

# Trust region for the Barzilai-Borwein step, in units of the probe half-width
# (gradientDescentStepHz). See maxStep below for why the step needs bounding at
# all. Larger values converge in fewer iterations but tolerate bigger overshoots;
# 4 keeps the search inside a resonance a few probe widths wide.
_MAX_STEP_PROBES = 4

class _GradientUnmeasurable(Exception):
    """The gradient could not be measured at the requested frequency."""

class NewSerialGradientDescent(pr.Process):
    def __init__(self, **kwargs):
        pr.Process.__init__(self)

    def _process(self):
        self.parent.etaScanInProgress.set( 1 )
        self.Message.setDisp("Started")

        # Representable range of the centre frequency, in MHz.
        #
        # centerFrequency is Fixed(24,23), so the normalised range is
        # [-1.0, 1 - 2**-23], and the centerFrequencyMHz setter divides by
        # _freqSpanMHz before writing. Do NOT clip to the exact boundary: that
        # divide rounds the largest representable value back up by one ULP
        # (0.9999998807907104 -> 0.9999998807907106), which the register then
        # rejects as out of range. Backing off a whole LSB is ~1e9 ULPs of
        # margin and costs only 0.14 Hz of tuning range.
        _base  = self.parent.centerFrequency.base
        _lsb   = 2 ** -_base.binPoint
        _span  = self.parent._freqSpanMHz
        # Fixed.minValue()/maxValue() only exist in rogue >= v6.10.0; derive the
        # range from bitSize/binPoint, which every version exposes.
        _full  = 2 ** (_base.bitSize - 1 - _base.binPoint)
        min_f  = (-_full + _lsb) * _span
        max_f  = (_full - 2 * _lsb) * _span

        def _clip_freq(fMHz):
            """Clip an absolute centre frequency (MHz) to the representable range."""
            if not np.isfinite(fMHz):
                raise ValueError(f"Frequency {fMHz} MHz is invalid")
            return min(max(fMHz, min_f), max_f)

        # measure the gradient from samples on either side of current frequency
        def calcGrad(channel, centerFrequencyMHz, df, numAverages):
            # references to variables for this channel
            centerFreqVar = self.parent.CryoChannel[channel].centerFrequencyMHz
            freqErrorVar = self.parent.CryoChannel[channel].frequencyErrorMHz
            etaPhaseVar = self.parent.CryoChannel[channel].etaPhaseDegree

            # measure at positive offset
            pos = _clip_freq(centerFrequencyMHz + df)
            centerFreqVar.set(pos)
            etaPhaseVar.set(0)
            posResp = np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])
            etaPhaseVar.set(90)
            posResp += 1j * np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])

            # measure at negative offset
            neg = _clip_freq(centerFrequencyMHz - df)
            centerFreqVar.set(neg)
            negResp = 1j * np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])
            etaPhaseVar.set(0)
            negResp += np.mean([freqErrorVar.get(read=True) for _ in range(numAverages)])

            # Both probe points clipped to the same frequency, so there is no
            # baseline to difference. Raise rather than returning a sentinel:
            # an inf/nan here launders through alpha = |df/dg| into a 0*inf
            # step and can be mistaken for convergence.
            if pos == neg:
                raise _GradientUnmeasurable(f"probe points both saturated at {pos} MHz")

            # divide by the separation actually achieved, which clipping may
            # have narrowed to less than 2*df
            return (np.abs(posResp) - np.abs(negResp)) / (pos - neg)

        # defer update callbacks
        with self.root.updateGroup():
            self.Message.setDisp("Init")

            # optimisation parameters
            maxIters    = self.parent.gradientDescentMaxIters.get()
            numAverages = self.parent.gradientDescentAverages.get()
            initialStep = self.parent.gradientDescentStepHz.get()*1e-6
            converge    = self.parent.gradientDescentConvergeHz.get()*1e-6
            debug       = self.parent.debug.get()

            # Barzilai-Borwein extrapolates a step from two gradients and has no
            # notion of how wide the resonance is. On a dip narrower than the
            # extrapolation it steps clean over the feature and lands on the flat
            # tail, where the gradient - and so the next step - is small enough to
            # satisfy the convergence test hundreds of kHz off resonance. Bounding
            # the step keeps the search inside the feature it is measuring, at the
            # cost of needing more iterations to cover a given distance.
            maxStep     = _MAX_STEP_PROBES * initialStep

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
                        # The gradient did not change between two distinct
                        # frequencies, so there is no curvature to extrapolate
                        # from. That is a flat region, not a minimum: a constant
                        # slope, or a channel returning a constant. Reporting it
                        # as converged would commit a frequency one initial step
                        # away from the coarse search result on no evidence.
                        self._log.warning(
                            f"Channel {channel} gradient stopped changing at {f} MHz "
                            f"after {i} iterations; no minimum bracketed."
                        )
                        return False, f

                    alpha = np.abs(df / dg)
                    # bound the step to the trust region
                    step = float(np.clip(-1 * alpha * g, -maxStep, maxStep))

                    # check for convergence. maxStep is much larger than converge,
                    # so clipping above can never fake a converged step.
                    if np.abs(step) < converge:
                        complete = True
                        break
                    # update position, bailing out if the step leaves the
                    # representable range rather than waiting for the probe
                    # points to saturate
                    f_next = f + step
                    if not (min_f <= f_next <= max_f):
                        self._log.warning(
                            f"Channel {channel} left representable range at "
                            f"{f_next} MHz after {i} iterations."
                        )
                        return False, f
                    f_last, g_last = f, g
                    f = f_next
                    g = calcGrad(channel, f, step, numAverages)
                    if debug > 0:
                        self._log.info(f"Iter {i}: f={f} step={step}")
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

                try:
                    # run the gradient descent on this channel. Only catch the
                    # errors this routine raises for itself; anything else (a
                    # failed register transaction, say) is a systemic problem
                    # and must not be reported once per channel.
                    try:
                        complete, f = gradDescent(channel)
                    except (_GradientUnmeasurable, ValueError) as e:
                        self._log.error(f"Gradient descent on channel {channel} raised {e}.")
                        complete, f = False, freq[channel]

                    # check result is in range
                    if not (min_f <= f <= max_f):
                        # reset to its starting point
                        self.parent.CryoChannel[channel].centerFrequencyMHz.set(freq[channel])
                        self._log.warning(f"Channel {channel} is out of range. Skipping.")
                    elif complete:
                        self.parent.CryoChannel[channel].centerFrequencyMHz.set(f)
                    else:
                        # reset to its starting point
                        self.parent.CryoChannel[channel].centerFrequencyMHz.set(freq[channel])
                        self._log.warning(f"Center frequency for channel {channel} not updated.")
                finally:
                    # always drop this channel's tone before moving to the next,
                    # otherwise a skipped channel contaminates every later one
                    self.parent.CryoChannel[channel].amplitudeScale.set( 0 )

            # reset tone power to initial state
            self.parent.amplitudeScale.set( amplitudeScale )

        self.Progress.set(1.0)
        self.Message.setDisp(f"Done")
        self.parent.etaScanInProgress.set( 0 )
