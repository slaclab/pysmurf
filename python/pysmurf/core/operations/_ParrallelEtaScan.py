#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Parrallel Eta Scan Process
#-----------------------------------------------------------------------------
# File       : _ParrallelEtaScan.py
# Created    : 2019-10-09
#-----------------------------------------------------------------------------
# Description:
#    Moved verbatim from cryo-det, where it lived at
#    firmware/python/CryoDet/DspCoreLib/CryoDetCmbHcd/_ParrallelEtaScan.py
#    as of commit 31b6fbfe (== tag MicrowaveMuxBpEthGen2_v2.5.1).
#
#    The class body below is byte-identical to that file. Only this header and
#    the comment above __all__ differ. Do not "clean up" this module: the
#    byte-for-byte match with cryo-det is what proves the move is behaviour-
#    preserving. See docs/stage1_ops_out_of_cryo_det.md.
#
#    KNOWN BROKEN, and deliberately left broken by the move: the write to
#    'self.parent.etaPhaseArray' below raises AttributeError -- no such node
#    exists on CryoChannels (the array variable is 'etaPhase'). This is a
#    pre-existing defect, not something the move introduced, and there is no
#    client accessor for runParallelEtaScan, so nothing calls it. Do not
#    "fix" it by retargeting the write: 'etaPhase' is documented in radians
#    (smurf_command.py: _eta_phase_array_reg) while the value here is in
#    degrees, so that swap trades a loud crash for a silent unit error. The
#    follow-up must settle units AND add a client accessor so it is testable.
#
#    Also note the class name keeps its original misspelling. The class name is
#    the rogue node name; renaming it is a breaking API change, not cleanup.
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
__all__ = ['ParrallelEtaScan']

class ParrallelEtaScan(pr.Process):
    def __init__(self, **kwargs):
        pr.Process.__init__(self)

    def _process(self):
        self.parent.etaScanInProgress.set( 1 )
        self.Message.setDisp("Started")

        # defer update callbacks
        with self.root.updateGroup():
            self.Message.setDisp("Init")
            delF            = self.parent.etaScanDelF.get()*1e-6
            numAverages     = 10

            amplitudeScaleArray = self.parent.amplitudeScale.get()
            amplIdx = np.where( amplitudeScaleArray == 0 )

            self.parent.feedbackEnable.set(np.zeros(512, dtype=np.uint))  #TODO: should not use a hardcoded value here

            self.parent.etaMag.set( np.ones(512) )

            self.parent.etaPhase.set( np.zeros(512) )
            time.sleep(0.1)
            imag = np.mean( [ self.parent.frequencyError.get(read=True) for _ in range(numAverages) ], axis=0 )

            self.parent.etaPhase.set( -90 * np.ones(512) )
            time.sleep(0.1)
            real = np.mean( [ self.parent.frequencyError.get(read=True) for _ in range(numAverages) ], axis=0 )

            etaPhaseDegree = np.arctan2(imag, real)*180/np.pi

            idx = np.where( etaPhaseDegree > 180 )
            etaPhaseDegree[idx] = etaPhaseDegree[idx] - 360

            idx = np.where( etaPhaseDegree < -180 )
            etaPhaseDegree[idx] = etaPhaseDegree[idx] + 360

            etaPhaseDegree[amplIdx] = 0
            self.parent.etaPhaseArray.set( etaPhaseDegree )

            freq = self.parent.centerFrequencyMHz.get()

            self.parent.centerFrequencyMHz.set( freq + delF )
            time.sleep(0.1)
            posError = np.mean( [ self.parent.frequencyError.get(read=True) for _ in range(numAverages) ], axis=0 )

            self.parent.centerFrequencyMHz.set( freq - delF )
            time.sleep(0.1)
            negError = np.mean( [ self.parent.frequencyError.get(read=True) for _ in range(numAverages) ], axis=0 )

            self.parent.centerFrequencyMHz.set( freq )

            etaMagScaled = 2*delF / ( posError - negError )
            idx = np.where( etaMagScaled < 0 )
            etaMagScaled[idx] = np.abs( etaMagScaled[idx] )
            etaPhaseDegree[idx] = etaPhaseDegree[idx] + 180

            idx = np.where( etaPhaseDegree > 180 )
            etaPhaseDegree[idx] = etaPhaseDegree[idx] - 360

            idx = np.where( etaPhaseDegree < -180 )
            etaPhaseDegree[idx] = etaPhaseDegree[idx] + 360

            etaPhaseDegree[amplIdx] = 0
            etaMagScaled[amplIdx] = 0

            self.parent.etaPhase.set( etaPhaseDegree )

            self.parent.etaMag.set( etaMagScaled )

            feedbackEnable = np.zeros(512, dtype=np.uint)
            feedbackEnable[amplIdx] = 0
            self.parent.feedbackEnable.set( feedbackEnable )

        self.Progress.set(1.0)
        self.Message.setDisp(f"Done")
        self.parent.etaScanInProgress.set( 0 )
