#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryo channel operations attachment
#-----------------------------------------------------------------------------
# File       : _CryoOperations.py
# Created    : 2026-08-28
#-----------------------------------------------------------------------------
# Description:
#    Attaches the resonator-tuning operations to each band's CryoChannels
#    device at server startup.
#
#    These 29 nodes -- 19 tuning-parameter LocalVariables, 4 pr.Process
#    devices, and 6 dispatch commands -- used to be added by cryo-det's
#    CryoChannels.__init__ (firmware/python/CryoDet/DspCoreLib/CryoDetCmbHcd/
#    _CryoChannels.py). Because cryo-det ships as firmware, every algorithm
#    tweak cost a firmware release cycle. Moving them here decouples the two.
#
#    The rogue paths are unchanged. Every node reappears at
#        AMCc.FpgaTopLevel.AppTop.AppCore.SysgenCryo.Base[i].CryoChannels.<name>
#    with the same name, type, mode and description, so SmurfControl and
#    sodetlib keep working with no client-side change. The variable
#    definitions and command bodies below are transcribed verbatim from
#    cryo-det commit e3dc359c (main after PR #80, which removed the unused
#    ParrallelEtaScan and fixed loadTuneFile's eta write order); the only
#    edit is the mechanical rewrite of 'self' -> 'ch', since these are now
#    functions taking the device rather than methods of it. The move itself
#    was taken from 31b6fbfe (== tag MicrowaveMuxBpEthGen2_v2.5.1); #80 was
#    applied on top as its own commit so the two can be reviewed apart.
#
#    scripts/stage1_source_identity.py checks that mechanically, and it must
#    stay passing. See docs/stage1_ops_out_of_cryo_det.md.
#
#    Version coupling. pysmurf and the CryoDet package must be updated
#    together. A package that still defines these nodes is one that predates
#    the strip, and pysmurf's operations must not be layered on top of it: the
#    two copies would collide, and any subset that did attach would let one
#    package's algorithms run against the other's parameters. So attach()
#    checks every name first and raises if any of them is already there.
#
#    That check is deliberately kept rather than left to pyrogue, which would
#    also refuse the duplicate but only as a bare 'Name collision' on whichever
#    node it reached first -- after some of the others had already been added.
#    Checking up front keeps the attach atomic and lets the error say which
#    package is at fault.
#
#    This covers the RFSoC build too, even though stage 1 does not touch the
#    zcu208 repositories: their CryoDet._MicrowaveMuxZcu208.FpgaTopLevel
#    subclasses CryoDet._MicrowaveMuxBpEthGen2.FpgaTopLevel from their
#    firmware/submodules/cryo-det submodule, so their CryoChannels is this
#    cryo-det's CryoChannels. Bumping that submodule pin is what keeps them in
#    step.
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
                           label: str | None = None) -> None:
    """Add the operations to one band's ``CryoChannels`` device.

    Must be called after the device is in the tree but before ``Root.start()``:
    ``add()`` refuses to touch a started tree.

    Raises ``RuntimeError`` if the device already has any of ``OPERATION_NODES``,
    which means the loaded CryoDet package predates the move and still defines
    them itself. pysmurf and the CryoDet package are a matched pair; see the
    module header.

    ``label`` is how the device is named in log messages and errors. It has to
    be passed in: a pyrogue node does not know where it lives until the root is
    started, and ``Node.path`` is still just its own name at this point, so
    every band would look identical. ``attach_all_cryo_operations`` passes the
    band.
    """
    label = label or cryo_channels.name

    present = [n for n in OPERATION_NODES if _exists(cryo_channels, n)]
    if present:
        raise RuntimeError(
            f"{label} already defines {len(present)} of the "
            f"{len(OPERATION_NODES)} cryo tuning operation nodes, so the "
            f"loaded CryoDet package still provides its own copy of the "
            f"tuning code and pysmurf must not add a second one.\n"
            f"  already present ({len(present)}): {', '.join(present)}\n"
            f"This pysmurf requires a CryoDet package from which the "
            f"operations have been removed. Update the firmware ZIP (or the "
            f"cryo-det checkout under /tmp/fw) to a version that no longer "
            f"defines them -- for the RFSoC builds, bump the "
            f"firmware/submodules/cryo-det submodule pin.")

    _attach(cryo_channels)
    cryo_channels._log.info(
        "Attached %d pysmurf cryo operations to %s.",
        len(OPERATION_NODES), label)


def attach_all_cryo_operations(fpga: pr.Device, *, n_bands: int = 8) -> list[int]:
    """Attach the operations to every band present under ``fpga``.

    Returns the sorted list of band indices that were attached, for logging.

    Bands absent from the tree are skipped: not every platform builds all
    eight, and a carrier with only one AMC populated genuinely has fewer. Note
    the node lookup, and only the node lookup, is guarded -- an error raised by
    the attach itself must propagate, or a half-attached band would start up
    and misbehave later.
    """
    bands = []

    for i in range(n_bands):
        try:
            cryo_channels = fpga.AppTop.AppCore.SysgenCryo.Base[i].CryoChannels
        except (AttributeError, KeyError, IndexError):
            continue
        attach_cryo_operations(
            cryo_channels, label=f'Base[{i}].CryoChannels')
        bands.append(i)

    if not bands:
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

    Transcribed verbatim from cryo-det, descriptions included. Three of them
    are the copy-pasted-wrong "etaScan frequencies"; they stay wrong here,
    because saveVariableList dumps descriptions and the baseline tree diff has
    to come out empty. Correcting them is a separate, visible change.
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
    """Add the 5 background pr.Process devices.

    SerialFindFreq reads its sweep geometry from the device that owns it, so
    the constants stay where the hardware map defines them. The other four
    hardcode 512 and 1.2; that is a pre-existing wart, listed as a follow-up
    in docs/stage1_ops_out_of_cryo_det.md, not something to fix inside a
    verbatim move.
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
                freqs = eval(freqs)

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
