#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryo channel operations attachment
#-----------------------------------------------------------------------------
# File       : _CryoOperations.py
# Created    : 2026-08-28
#-----------------------------------------------------------------------------
# Description:
#    Attaches the resonator-tuning operations to each band's CryoChannels
#    device at server startup: 19 tuning-parameter LocalVariables, 4
#    pr.Process devices and 6 dispatch commands, at
#        AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.Base[i].CryoChannels.<name>
#    with the names, types, modes and descriptions cryo-det's CryoChannels gave
#    them up to release MicrowaveMuxBpEthGen2_v2.6.0 (commit e86e81bb), from
#    which they are transcribed ('self' -> 'ch'; runEtaScan's eval() of a
#    stored string replaced by ast.literal_eval()).
#
#    A firmware package that defines the operations itself (v2.6.0 and every
#    earlier release) keeps them: nothing is added to a device that already
#    has any of these nodes, and the names it lacks are logged. A package
#    without them gets all 29 from here. The check comes before anything is
#    added, so a device is either left as the firmware built it or carries
#    pysmurf's complete set, never a mixture of the two.
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

from ._NewSerialGradientDescent import NewSerialGradientDescent
from ._SerialEtaScan import SerialEtaScan
from ._SerialFindFreq import SerialFindFreq
from ._SerialGradientDescent import SerialGradientDescent

__all__ = [
    'OPERATION_COMMANDS',
    'OPERATION_NODES',
    'OPERATION_PROCESSES',
    'OPERATION_VARIABLES',
    'attach_all_cryo_operations',
    'attach_cryo_operations',
]

# The 29 node names this module owns, in the order cryo-det added them. These
# are the collision surface, so a node added by _attach() and left out of these
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


def _exists(dev, name):
    """Would adding a node called ``name`` to ``dev`` collide?

    This is deliberately the exact predicate pyrogue's ``Device.add()`` uses
    (``pyrogue/_Node.py``, the 'Name collision' check), so that "would this
    collide?" and "did this collide?" can never disagree.
    """
    return name in dev.__dir__() or name in getattr(dev, '_anodes', {})


def attach_cryo_operations(cryo_channels: pr.Device, *,
                           label: str | None = None) -> bool:
    """Add the operations to one band's ``CryoChannels`` device, unless the firmware defines them.

    Must be called after the device is in the tree but before ``Root.start()``:
    ``add()`` refuses to touch a started tree.

    A device that already has any of ``OPERATION_NODES`` belongs to a CryoDet
    package that defines the operations itself; it is left as built, and the
    names it lacks are logged at WARNING. A device with none of them gets all
    29. Nothing is ever added to a device that has some.

    ``label`` is how the device is named in log messages. It has to be passed
    in: a pyrogue node does not know where it lives until the root is started,
    and ``Node.path`` is still just its own name at this point, so every band
    would look identical. ``attach_all_cryo_operations`` passes the band.

    Returns True if pysmurf's operations were attached, False if the
    firmware's were kept.
    """
    label = label or cryo_channels.name

    present = [n for n in OPERATION_NODES if _exists(cryo_channels, n)]
    if present:
        missing = [n for n in OPERATION_NODES if n not in present]
        cryo_channels._log.info(
            "%s: the firmware package defines %d of the %d cryo tuning "
            "operation nodes itself; keeping them, attaching none.",
            label, len(present), len(OPERATION_NODES))
        if missing:
            cryo_channels._log.warning(
                "%s: the firmware package's cryo tuning operations lack %d "
                "node(s) this pysmurf knows, which a client that uses them "
                "will not find: %s. Releases v2.5.1 and v2.6.0 of cryo-det "
                "define them all, and so does pysmurf for a package that "
                "defines none.",
                label, len(missing), ', '.join(missing))
        return False

    _attach(cryo_channels)
    cryo_channels._log.info(
        "Attached %d pysmurf cryo operations to %s.",
        len(OPERATION_NODES), label)
    return True


def attach_all_cryo_operations(fpga: pr.Device, *, n_bands: int = 8) -> list[int]:
    """Attach the operations to every band present under ``fpga`` whose firmware lacks them.

    Returns the sorted list of band indices that were attached, for logging;
    a band whose firmware package defines the operations itself is not in it.

    Bands absent from the tree are skipped: not every platform builds all
    eight, and a carrier with only one AMC populated genuinely has fewer. Note
    the node lookup, and only the node lookup, is guarded -- an error raised by
    the attach itself must propagate, or a half-attached band would start up
    and misbehave later.
    """
    bands = []
    found = 0

    for i in range(n_bands):
        try:
            cryo_channels = fpga.AppTop.AppCore.SysgenCryo.Base[i].CryoChannels
        except (AttributeError, KeyError, IndexError):
            continue
        found += 1
        if attach_cryo_operations(cryo_channels, label=f'Base[{i}].CryoChannels'):
            bands.append(i)

    if not found:
        fpga._log.warning(
            "Found no CryoChannels devices under %s, so no cryo operations "
            "were attached. Expected AppTop.AppCore.SysgenCryo.Base[i]."
            "CryoChannels for i in 0..%d.", fpga.name, n_bands - 1)

    return bands


def _attach(cryo_channels):
    """Add all 29 operation nodes, in cryo-det's original order."""
    _add_local_variables(cryo_channels)
    _add_processes(cryo_channels)
    _add_commands(cryo_channels)


def _add_local_variables(ch):
    """Add the 19 tuning-parameter LocalVariables.

    Transcribed from cryo-det, descriptions included. Three of them carry the
    copy-pasted "etaScan frequencies" description; it stays as cryo-det has
    it, since descriptions are part of the published tree that clients and
    saved variable lists see, and a firmware package that defines these
    nodes itself must present the same tree. Correcting them is a separate,
    visible change.
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

    SerialFindFreq reads its sweep geometry from the device that owns it, so
    the constants stay where the hardware map defines them. The other three
    hardcode 512 and 1.2, as they did in cryo-det.
    """
    ch.add(SerialGradientDescent())
    ch.add(NewSerialGradientDescent())
    ch.add(SerialEtaScan())
    ch.add(SerialFindFreq(ch._n_channels, ch._freqSpanMHz))


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
