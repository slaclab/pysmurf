#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Serial Find Freq Process
#-----------------------------------------------------------------------------
# File       : _SerialFindFreq.py
# Created    : 2019-10-09
#-----------------------------------------------------------------------------
# Description:
#    Moved verbatim from cryo-det, where it lived at
#    firmware/python/CryoDet/DspCoreLib/CryoDetCmbHcd/_SerialFindFreq.py
#    as of commit 31b6fbfe (== tag MicrowaveMuxBpEthGen2_v2.5.1).
#
#    The class body below is byte-identical to that file. Only this header and
#    the comment above __all__ differ. Do not "clean up" this module: the
#    byte-for-byte match with cryo-det is what proves the move is behaviour-
#    preserving. See docs/stage1_ops_out_of_cryo_det.md.
#
#    Note this is the one Process that is already parameterised: it takes
#    n_channels and freq_span_mhz from the CryoChannels device that owns it,
#    rather than hardcoding 512 / 1.2 like the others.
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
__all__ = ['SerialFindFreq']

class SerialFindFreq(pr.Process):
    def __init__(self, n_channels, freq_span_mhz, **kwargs):
        pr.Process.__init__(self)
        self._n_channels = n_channels
        self._freq_span_mhz = freq_span_mhz

    def _process(self):
        self.parent.etaScanInProgress.set( 1 )
        self.Message.setDisp("Started")

        # defer update callbacks
        with self.root.updateGroup():
            self.Message.setDisp("Init")
            ampl    = self.parent.etaScanAmplitude.get()
            freqs   = self.parent.etaScanFreqs.get()
            scan_l  = int(len(freqs)/self._n_channels)
            freqs_r = np.asarray(freqs).reshape(self._n_channels, scan_l)  # asarray may be redundant
            resultReal  = np.zeros_like( freqs_r )
            resultImag  = np.zeros_like( freqs_r )

            self.parent.amplitudeScale.set(np.zeros(self._n_channels, dtype=np.uint))
            self.parent.feedbackEnable.set(np.zeros(self._n_channels, dtype=np.uint))
            self.parent.etaMag.set( np.ones(self._n_channels) )
            self.parent.etaPhase.set( np.zeros(self._n_channels) )
            time.sleep(0.1)

            for subchan in range(self._n_channels):
                if np.all( freqs_r[subchan, :] == freqs_r[subchan, 0]):
                    # all frequencies set equal for this resonator, continue to next
                    continue

                self.parent.CryoChannel[subchan].amplitudeScale.set( ampl )

                f = 0.0
                for idx, freqMHz in enumerate(freqs_r[subchan, :]):
                    if (freqMHz > self._freq_span_mhz) or (freqMHz < -self._freq_span_mhz):
                        self._log.warning(f'Channel {subchan}: Freq {freqMHz} out of range')
                        resultReal[subchan, idx] = 0
                    else:
                        if f != freqMHz:
                            f = freqMHz
                            self.parent.CryoChannel[subchan].centerFrequencyMHz.set( f )
                        freqError = self.parent.CryoChannel[subchan].frequencyError.get()
                        resultReal[subchan, idx] = freqError

                self.parent.CryoChannel[subchan].etaPhaseDegree.set(-90)

                for idx, freqMHz in enumerate(freqs_r[subchan, :]):
                    if (freqMHz > self._freq_span_mhz) or (freqMHz < -self._freq_span_mhz):
                        self._log.warning(f'Channel {subchan}: Freq {freqMHz} out of range')
                        resultImag[subchan, idx] = 0
                    else:
                        if f != freqMHz:
                            f = freqMHz
                            self.parent.CryoChannel[subchan].centerFrequencyMHz.set( f )
                        freqError = self.parent.CryoChannel[subchan].frequencyError.get()
                        resultImag[subchan, idx] = freqError

                self.parent.CryoChannel[subchan].amplitudeScale.set( 0 )

            self.parent.amplitudeScale.set( np.zeros(self._n_channels, dtype=np.uint) )
            self.parent.etaScanResultsReal.set( resultReal.flatten() )
            self.parent.etaScanResultsImag.set( resultImag.flatten() )

        self.Progress.set(1.0)
        self.Message.setDisp(f"Done")
        self.parent.etaScanInProgress.set( 0 )
