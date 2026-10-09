#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq uMUX Operations Provider
#-----------------------------------------------------------------------------
# File       : _umux_ops.py
# Created    : 2026-08-28
#-----------------------------------------------------------------------------
# Description:
#    The resonator-tuning operations of microwave-multiplexed readout, as the
#    provider the server composition attaches to each band's CryoChannels
#    device: 19 tuning-parameter variables, 4 background processes and 6
#    commands that dispatch them. The node names are rogue paths clients reach,
#    so they are declared once, in the order they are added, and that tuple is
#    the collision surface the provider mechanism checks before adding any.
#
#    Every process takes the channel count and frequency span of the device it
#    is added to, so the geometry stays the hardware map's.
#
#    Originally cryo-det's; moved here so the algorithms no longer ship with the
#    firmware.
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

from cryodaq.server._new_serial_gradient_descent import NewSerialGradientDescent
from cryodaq.server._provider import Provider
from cryodaq.server._serial_eta_scan import SerialEtaScan
from cryodaq.server._serial_find_freq import SerialFindFreq
from cryodaq.server._serial_gradient_descent import SerialGradientDescent

__all__ = [
    'OPERATION_COMMANDS',
    'OPERATION_NODES',
    'OPERATION_PROCESSES',
    'OPERATION_VARIABLES',
    'UMUX_OPERATIONS',
]

# The 29 node names this provider owns, in the order they are added. These are
# the collision surface, so a node added by _attach() and left out of these
# tuples would be invisible to the pre-attach check.
OPERATION_VARIABLES = (
    'etaScanChannel',
    'etaScanInProgress',
    'etaScanFreqs',
    'etaScanResultsImag',
    'etaScanResultsReal',
    'etaScanDelF',
    'etaScanMaxMag',
    'etaScanDwell',
    'etaScanAmplitude',
    'gradientDescentMaxIters',
    'gradientDescentAverages',
    'gradientDescentGain',
    'gradientDescentConvergeHz',
    'gradientDescentStepHz',
    'gradientDescentMomentum',
    'gradientDescentBeta',
    'etaScanAverages',
    'debug',
    'UseNewSerialGradientDescent',
)

# Node names come from the class names, so they are documented rogue paths
# and not something to rename here.
OPERATION_PROCESSES = (
    'SerialGradientDescent',
    'NewSerialGradientDescent',
    'SerialEtaScan',
    'SerialFindFreq',
)

OPERATION_COMMANDS = (
    'loadTuneFile',
    'runEtaScan',
    'setAmplitudeScales',
    'runSerialGradientDescent',
    'runSerialEtaScan',
    'runSerialFindFreq',
)

OPERATION_NODES = OPERATION_VARIABLES + OPERATION_PROCESSES + OPERATION_COMMANDS



def _attach(cryo_channels):
    """Add all 29 operation nodes, in their original order."""
    _add_local_variables(cryo_channels)
    _add_processes(cryo_channels)
    _add_commands(cryo_channels)


# The provider the composition attaches: one instance of these nodes under every
# band's operations device the platform map names.
UMUX_OPERATIONS = Provider(name='cryodaq.umux', anchor='band[*].ops',
                           nodes=OPERATION_NODES, attach=_attach)


def _add_local_variables(ch):
    """Add the 19 tuning-parameter LocalVariables.

    Descriptions are as they have always been in the tree; three of them read
    "etaScan frequencies" for a variable that is not one. Correcting them is a
    visible change to the tree and is left for one.
    """
    ch.add(pr.LocalVariable(
        name        = "etaScanChannel",
        description = "etaScan frequency band",
        mode        = "RW",
        value       = 0,
    ))

    # keeps track of whether or not an etaScan is currently
    # in progress.  default zero, meaning scan isn't running
    # currently.  runEtaScan sets it to one while it's scanning
    ch.add(pr.LocalVariable(
        name        = "etaScanInProgress",
        description = "etaScan in progress",
        mode        = "RW",
        value       = 0,
    ))

    # make waveform for etaScanFreqs, 1000 will be our max number
    # make sure to initialize with type we want in EPICS (float)
    ch.add(pr.LocalVariable(
        name        = "etaScanFreqs",
        hidden      = True,
        description = "etaScan frequencies",
        mode        = "RW",
        value       = np.zeros(256000), # TODO what determines this size?
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanResultsImag",
        hidden      = True,
        description = "etaScan frequencies",
        mode        = "RW",
        value       = np.zeros(256000),
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanResultsReal",
        hidden      = True,
        description = "etaScan frequencies",
        mode        = "RW",
        value       = np.zeros(256000),
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanDelF",
        description = "etaScan frequencies",
        mode        = "RW",
        value       = 5000,
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanMaxMag",
        description = "Clip eta magnitude at this level.",
        mode        = "RW",
        value       = 1.0,
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanDwell",
        description = "etaScan frequencies",
        mode        = "RW",
        value       = 0.0,
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanAmplitude",
        description = "number of points to average for etaScan",
        mode        = "RW",
        value       = 0,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentMaxIters",
        description = "gradient descent max iterations",
        mode        = "RW",
        value       = 15,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentAverages",
        description = "gradient descent number of averages",
        mode        = "RW",
        value       = 2,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentGain",
        description = "Gradient descent gain",
        mode        = "RW",
        value       = 0.001,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentConvergeHz",
        description = "Use gradient convergence threshold",
        mode        = "RW",
        value       = 500.0,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentStepHz",
        description = "Use gradient descent initial step",
        mode        = "RW",
        value       = 5000.0,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentMomentum",
        description = "Use gradient descent momentum",
        mode        = "RW",
        value       = 1,
    ))

    ch.add(pr.LocalVariable(
        name        = "gradientDescentBeta",
        description = "Use gradient descent initial step",
        mode        = "RW",
        value       = 0.1,
    ))

    ch.add(pr.LocalVariable(
        name        = "etaScanAverages",
        description = "eta scan number of averages",
        mode        = "RW",
        value       = 2,
    ))

    ch.add(pr.LocalVariable(
        name        = "debug",
        description = "print debug messages",
        mode        = "RW",
        value       = 0,
    ))

    ch.add(pr.LocalVariable(
        name        = "UseNewSerialGradientDescent",
        description = "Use a new implementation of SerialGradientDescent",
        mode        = "RW",
        value       = False,
    ))


def _add_processes(ch):
    """Add the 4 background pr.Process devices.

    Each reads its sweep geometry from the device that owns it, so the
    constants stay where the hardware map defines them.
    """
    geometry = (ch._n_channels, ch._freqSpanMHz)
    ch.add(SerialGradientDescent(*geometry))
    ch.add(NewSerialGradientDescent(*geometry))
    ch.add(SerialEtaScan(*geometry))
    ch.add(SerialFindFreq(*geometry))


def _add_commands(ch):
    """Add the 6 operation dispatch commands.

    In cryo-det these were closures decorated with ``@self.command(...)``.
    That decorator is exactly ``self.add(pr.LocalCommand(function=func,
    name=func.__name__, **kwargs))`` (``pyrogue/_Device.py``), so registering
    them explicitly is behaviour-identical.

    Pass each function object straight through, undecorated and unwrapped:
    ``BaseCommand`` decides whether the command takes an argument by
    inspecting the signature (``'arg' in getfullargspec(function).args``), so
    wrapping ``setAmplitudeScales`` in a zero-argument lambda would quietly
    turn it into a command that ignores its argument.
    """
    def loadTuneFile():

        band     = ch.parent.band.get()
        tuneFile = ch.parent.parent.tuneFilePath.get()
        ch._log.info("Tuning band " + str(band) + " with tune file: " + tuneFile)

        try:

            tune = np.load( tuneFile ).item()
            tuneBand = tune[band]

            with ch.root.updateGroup():

                drive = tuneBand['drive']

                # load parameters, don't write to HW yet
                for r in tuneBand['resonances']:
                    channel      = tuneBand['resonances'][r]['channel']
                    etaPhase     = tuneBand['resonances'][r]['eta_phase']
                    etaMagScaled = tuneBand['resonances'][r]['eta_scaled']
                    centerFreq   = tuneBand['resonances'][r]['offset']

                    if channel == -1:
                        continue

                    ch.CryoChannel[channel].amplitudeScale.set( value=drive, write=False )
                    ch.CryoChannel[channel].centerFrequencyMHz.set( value=centerFreq, write=False )
                    # magnitude before phase: etaMag and etaPhase are a
                    # polar view of one Cartesian etaI/etaQ pair, and each
                    # setter recomputes from the other's current value. On a
                    # channel whose etaI/etaQ are still zero - a band that
                    # has not been tuned since reset, which is exactly when
                    # a tune gets loaded - setting the phase first writes
                    # (0, 0) and the phase is silently discarded.
                    ch.CryoChannel[channel].etaMagScaled.set( value=etaMagScaled, write=False )
                    ch.CryoChannel[channel].etaPhaseDegree.set( value=etaPhase, write=False )
                    ch.CryoChannel[channel].feedbackEnable.set( value=1, write=False )

                # write to HW, block transaction
                ch.writeBlocks()
                ch.verifyBlocks()
                ch.checkBlocks()

        except Exception as e:
            ch._log.error(
                f"Failed to load tune for band {band}. {type(e).__name__}: {e}."
            )

    ch.add(pr.LocalCommand(
        name        = "loadTuneFile",
        description = "Load ETA params",
        function    = loadTuneFile,
    ))

    def runEtaScan():
        ch.etaScanInProgress.set( 1 )

        # defer update callbacks
        with ch.root.updateGroup():
            subChan = ch.etaScanChannel.get()
            ampl    = ch.etaScanAmplitude.get()
            freqs   = ch.etaScanFreqs.get()

            # workaround for rogue local variables, RTH not sure if still needed
            # list objects get written as string, not list of float when set by GUI
            if isinstance(freqs, str):
                import ast
                freqs = ast.literal_eval(freqs)

            ch.CryoChannel[subChan].amplitudeScale.set(ampl)
            ch.CryoChannel[subChan].etaMagScaled.set(value=1)
            ch.CryoChannel[subChan].feedbackEnable.set(value=0)

            # run scan in phase
            ch.CryoChannel[subChan].etaPhaseDegree.set(value=90)
            resultsReal = []
            f           = []
            for freqMHz in freqs:

                # is there overhead of setting freqMHz if prevFreqMHz == freqMHz
                # out list of freqs may do several measurements at a single freq
                # dont' want to write the same value again
                if f != freqMHz:
                    f = freqMHz
                    ch.CryoChannel[subChan].centerFrequencyMHz.set(value=f)
                freqError = ch.CryoChannel[subChan].frequencyError.get()
                resultsReal.append( freqError )

            # run scan in quadrature
            ch.CryoChannel[subChan].etaPhaseDegree.set(value=0)
            resultsImag = []
            f           = []
            for freqMHz in freqs:
                if f != freqMHz:
                    f = freqMHz
                    ch.CryoChannel[subChan].centerFrequencyMHz.set(value=f)
                freqError = ch.CryoChannel[subChan].frequencyError.get()
                resultsImag.append( freqError )

            ch.etaScanResultsReal.set(value=resultsReal)
            ch.etaScanResultsImag.set(value=resultsImag)

        ch.etaScanInProgress.set( 0 )

    ch.add(pr.LocalCommand(
        name        = "runEtaScan",
        description = "Run etaScan",
        function    = runEtaScan,
    ))

    def setAmplitudeScales(arg):
        for c in ch.CryoChannel.values():
            c.amplitudeScale.setDisp(arg, write=False)

        ch.writeAndVerifyBlocks()

    ch.add(pr.LocalCommand(
        name        = "setAmplitudeScales",
        description = "Set all amplitudeScale values",
        value       = 0,
        function    = setAmplitudeScales,
    ))

    def runSerialGradientDescent():
        if ch.etaScanInProgress.get() != 1:
            if ch.UseNewSerialGradientDescent.get():
                ch._log.warning("Using new version of SerialGradientDescent")
                ch.NewSerialGradientDescent.Start()
            else:
                ch.SerialGradientDescent.Start()

    ch.add(pr.LocalCommand(
        name        = "runSerialGradientDescent",
        description = "Run serial gradient descent",
        function    = runSerialGradientDescent,
    ))

    def runSerialEtaScan():
        ch.SerialEtaScan.Start()

    ch.add(pr.LocalCommand(
        name        = "runSerialEtaScan",
        description = "Run serial eta scan",
        function    = runSerialEtaScan,
    ))

    def runSerialFindFreq():
        ch.SerialFindFreq.Start()

    ch.add(pr.LocalCommand(
        name        = "runSerialFindFreq",
        description = "Run parallel find freq",
        function    = runSerialFindFreq,
    ))
