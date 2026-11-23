#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Support: microwave-multiplexed readout on the AMC carrier
#-----------------------------------------------------------------------------
# File       : amc/support.py
# Created    : 2026-10-24
#-----------------------------------------------------------------------------
# Description:
#    What this platform contributes to the server's rogue tree: the firmware's
#    top level built for the transport in hand, and the configuration procedure
#    -- converters reset, register layers loaded, serial links checked -- as a
#    process on the tree -- and a passive eye scan of each bay's receive lanes.
#    The register map beside this module is data a client reads without rogue;
#    this is the server's half of the platform, and the one place its rogue code
#    lives.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import argparse
import logging
import os
import shlex
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pyrogue as pr

from cryodaq._errors import ConnectError
from cryodaq.platform import amc
from cryodaq.server._setup import PlatformSupport, Setup, StepError, load_layers

__all__ = ['AmcSupport', 'AmcSetup', 'RxEyeScan']

_log = logging.getLogger(__name__)


class AmcSetup(Setup):
    """The AMC carrier's procedure: converters reset, layers loaded, links checked.

    The carrier's converters sit on AMC cards behind JESD links. A bay is
    populated when its front end (``MicrowaveMuxCore[bay]``) is on the tree;
    the composition leaves an empty bay's out.
    """

    # What each receive lane's elastic-buffer latency reads once the converter
    # links have come up correctly: lanes 2 and 3 carry no data on this AMC and
    # read the idle value, the rest settle one of two ticks apart.
    ELASTIC_BUFFER_LATENCY: Dict[int, Tuple[int, ...]] = {
        lane: ((255,) if lane in (2, 3) else (13, 14)) for lane in range(10)}
    # How many times the layers are loaded again for lanes that did not align.
    ELASTIC_BUFFER_RETRIES = 10
    # The status block's JESD verdict, as its enumeration numbers them.
    JESD_UNLOCKED, JESD_LOCKED, JESD_NOT_FOUND = 0, 1, 3

    def steps(self) -> Sequence[Tuple[str, Callable[[], None]]]:
        # The load itself runs the firmware's link bring-up; the converters
        # are reset before it and the links judged after it.
        root = self.root
        return (('read all', root.ReadAll),
                ('reset DACs', self.reset_dacs),
                ('arm eye scan', self.arm_eye_scan),
                ('load layers', lambda: load_layers(root, self.layers)),
                ('check elastic buffers', self.check_elastic_buffers))

    def run_steps(self) -> None:
        # The load runs the firmware's link check too, so the procedure's
        # outcome is also the links' verdict, reported where the status block
        # shows it -- unless the firmware predates the check, which the status
        # says ('Not found') and the outcome must not overwrite. Every step is
        # inside, so a failed read also reports the links unlocked rather than
        # leaving an earlier run's verdict.
        verdict = self.status.JesdStatus
        try:
            Setup.run_steps(self)
        except Exception:
            if verdict.value() != self.JESD_NOT_FOUND:
                verdict.set(self.JESD_UNLOCKED)
            raise
        if verdict.value() != self.JESD_NOT_FOUND:
            verdict.set(self.JESD_LOCKED)

    def populated_bays(self) -> List[Tuple[int, Any, Any]]:
        """``(bay, front end, JESD block)`` for each bay with a front end on the tree."""
        bays = []
        for bay in range(2):
            core = self.root.getNode(amc.MUX_CORE.format(bay=bay))
            if core is not None:
                bays.append((bay, core, self.root.getNode(amc.JESD_BAY.format(bay=bay))))
        return bays

    def reset_dacs(self) -> None:
        """Pulse the reset of every RF DAC on every populated bay.

        A loaded configuration leaves the converter debug block disabled, which
        is where the reset bits live; it is enabled first so the writes land.
        """
        bays = self.populated_bays()
        for _, core, _ in bays:
            core.DBG.enable.set(True)
        for value in (1, 0):
            for _, core, _ in bays:
                for reset in core.DBG.dacReset.values():
                    reset.set(value)

    def rx_gts(self, bay: int) -> List[Any]:
        """The GT channel devices of a bay's lanes, in lane order; none on a tree built without them."""
        gts = []
        while True:
            gt = self.root.getNode(amc.JESD_RX_GT.format(bay=bay, rx_lane=len(gts)))
            if gt is None:
                return gts
            gts.append(gt)

    def arm_eye_scan(self) -> None:
        """Set both eye-scan enables on every GT channel of every populated bay.

        The eye-scan sampler of these GTs only comes alive through an RX reset
        taken with its enables already set, and the only RX reset of a bay is
        ``JesdRx.ResetGTs``, which resets every GT of the bay, RX and TX. The
        load that follows holds those GTs in reset while it writes the layers,
        so enables set here are in place for that reset, and a later eye scan
        (`RxEyeScan`) finds a live sampler without resetting anything. Neither
        enable changes the data path. A tree built without the GT devices has
        nothing to arm, and says so.
        """
        for bay, _, _ in self.populated_bays():
            gts = self.rx_gts(bay)
            if not gts:
                _log.warning("bay %d has no GT DRP devices in the tree; its eye scan is not armed", bay)
            for gt in gts:
                gt.ES_EYE_SCAN_EN.set(1)
                gt.ES_ERRDET_EN.set(1)

    def check_elastic_buffers(self) -> None:
        """Check each receive lane's elastic-buffer latency against the values expected.

        A lane outside its expected values has a link that did not align, and
        the layers are loaded again -- the load runs the firmware's own link
        bring-up -- before the lanes are read again.
        """
        for k in range(self.ELASTIC_BUFFER_RETRIES):
            _log.info("check elastic buffers (try %d)", k)
            bad = []
            for bay, _, jesd in self.populated_bays():
                # Reading the individual registers does not work; read the device.
                jesd.JesdRx.ReadDevice()
                for lane, latency in jesd.JesdRx.ElBuffLatency.items():
                    if latency.value() not in self.ELASTIC_BUFFER_LATENCY.get(lane, ()):
                        bad.append((bay, lane, latency.value()))
            if not bad:
                _log.info("elastic buffer check passed")
                return
            for bay, lane, latency in bad:
                _log.warning("  bay %d lane %d elastic buffer latency %d", bay, lane, latency)
            if k < self.ELASTIC_BUFFER_RETRIES - 1:
                _log.warning("check failed; loading the configuration again")
                load_layers(self.root, self.layers)
        raise StepError(f"the elastic buffer check failed {self.ELASTIC_BUFFER_RETRIES} times")


class AmcSupport(PlatformSupport):
    """The AMC carrier's tree: the carrier's top level, and the procedure above."""

    setup = AmcSetup

    def top_level(self, transport: Any, **options: Any) -> Any:
        """The carrier's top level over the transport's register path.

        ``options`` are the class's own keyword arguments -- ``disableBay0``,
        ``disableBay1``, ``enablePwrI2C`` -- as a deployment gives them. The
        class is the firmware package's, importable once the package is on the
        path, which it is by the time a platform has been identified.

        The class takes the FPGA's registers at 0 -- its constructor has no
        ``offset`` and a rogue device's is fixed when it is built -- so a
        transport that puts them elsewhere (a processor system's window) is
        refused: a tree built at 0 over it would address nothing the probe
        identified. Passing the window on is a change to the firmware's class.
        """
        if transport.register_offset:
            raise ConnectError(f"platform {amc.NAME}'s top level takes the FPGA's registers at 0, and cannot "
                               f"be placed at the {transport.name} transport's window {transport.register_offset:#x}")
        from CryoDet._MicrowaveMuxBpEthGen2 import FpgaTopLevel
        # With the GT DRP devices (images 0x02070000 and later): the eye scan's
        # sampler is armed through them, and each bay with them gets its eye scan.
        fpga = FpgaTopLevel(memBase=transport.srp, enJesdDrp=True, **options)
        for bay, jesd in getattr(fpga.AppTop, 'AppTopJesd', {}).items():
            if jesd.eyeScanLanes():
                jesd.add(RxEyeScan(bay=bay))
        return fpga


class _Refused(Exception):
    """An eye scan refused before any lane was written."""


class _ArgumentParser(argparse.ArgumentParser):
    """argparse that raises rather than exits, for options a client sends a server."""

    def error(self, message):
        raise _Refused(message)


class RxEyeScan(pr.Process):
    """A passive statistical eye scan of the JESD receive lanes of the bay it is added to.

    Runs surf's eye-scan runner (``surf.xilinx.eyescan.runEyeScan``) over the
    bay's GT channels with ``--resetPolicy Never``: it never pulses
    ``JesdRx.ResetGTs`` -- which resets every GT of the bay, RX and TX, and so
    drops the DAC links too -- and never re-runs the link bring-up, so the
    links stay up through a scan. The sampler must therefore already be alive,
    which the configuration procedure's ``arm eye scan`` step sees to; a lane
    whose sampler is not ends ``NOT_ENABLED`` rather than being reset.

    Set the parameters, then start it. ``Mode`` is the runner's ``--mode``
    (Status, Probe, Bathtub, Eye, Margin), ``Lanes`` its ``--lanes`` (empty for
    every enabled lane), ``OutDir`` the server-side directory a run folder is
    made in (every mode but Status needs one), and ``Options`` any further
    runner options as a command line (``--targetBer 1e-10 --tag hot``). The
    options that would reset a GT, or scan lanes of the one reset in
    parallel, are refused. A run is also refused, with reads only, when the
    bay's link is not up (``DataValid`` short of ``Enable``) unless
    ``--force`` is among the options.

    What it found: ``Summary`` (Status mode's per-lane lines, or how many lanes
    were scanned and where), ``RunFolder`` (the run's files on the server) and
    ``ExitCode`` (the runner's: 0 when every lane ended OK or CLOSED, 2
    otherwise; -1 before a run). A run cannot be stopped once it is scanning:
    the runner takes its stop request from Ctrl-C only, so ``Stop`` waits for
    the run to end.
    """

    # The serial rate of the MicrowaveMuxBpEthGen2 GT IP (307.125 MHz reference,
    # 8B10B). The carrier has no JESD clock counter, so the rate cannot be
    # measured and is recorded as given.
    LINE_RATE_GBPS = 12.285

    # Set by the parameters of their own, so refused inside Options.
    _OWN_OPTIONS = ('--mode', '--lanes', '--outDir', '--lineRate')

    def __init__(self, bay, **kwargs):
        pr.Process.__init__(self, name=amc.RX_EYE_SCAN.rsplit('.', 1)[1],
                            description='Passive JESD RX eye scan of this bay (no GT reset)', **kwargs)
        self._bay = bay
        names = amc.RX_EYE_SCAN_PARAMETERS
        for name, value, mode, description in (
                ('mode', 'Status', 'RW', 'Status, Probe, Bathtub, Eye or Margin'),
                ('lanes', '', 'RW', 'Lanes to scan, e.g. 0-3,8; empty for every enabled lane'),
                ('options', '', 'RW', 'Further eye-scan runner options, as a command line'),
                ('line_rate', self.LINE_RATE_GBPS, 'RW', 'Line rate of the lanes in Gb/s, recorded as given'),
                ('out_dir', '', 'RW', 'Server-side directory the run folder is made in'),
                ('run_folder', '', 'RO', 'Run folder of the last scan; empty in Status mode'),
                ('summary', '', 'RO', 'What the last run found'),
                ('exit_code', -1, 'RO', 'Exit code of the last run: 0 every lane OK or CLOSED, 2 otherwise')):
            self.add(pr.LocalVariable(name=names[name], value=value, mode=mode, description=description))

    def _startProcess(self):
        # Cleared here, before the worker starts, so a client that has started a run
        # can wait on ExitCode leaving -1: Running rises only once the worker runs.
        if not self.Running.value():
            self.RunFolder.set('')
            self.Summary.set('')
            self.ExitCode.set(-1)
        pr.Process._startProcess(self)

    def _arguments(self, es):
        """The runner's arguments from the parameters, refusing what would not be passive."""
        options = shlex.split(self.Options.value())
        own = [o for o in options if o.split('=', 1)[0] in self._OWN_OPTIONS]
        if own:
            raise _Refused(f"{', '.join(own)} in Options: set the parameter of its own instead")
        parser = _ArgumentParser(prog=self.path, add_help=False, allow_abbrev=False)
        es.addEyeScanArguments(parser)
        parser.add_argument('--force', action='store_true')
        # The receivers run in LPM mode and carry JESD traffic; the reset is the bay's.
        parser.set_defaults(equalizer='lpm', traffic='jesd', parallel=False, resetPolicy='Never')
        # The parameters go in as their options, so argparse judges them as it does the rest.
        own = ['--mode', self.Mode.value(), '--outDir', self.OutDir.value(),
               '--lineRate', str(self.LineRateGbps.value())]
        if self.Lanes.value():
            own += ['--lanes', self.Lanes.value()]
        args = parser.parse_args(options + own)
        if args.resetPolicy != 'Never' or args.softCold or args.noResetHook or args.parallel:
            raise _Refused('this scan is passive: --resetPolicy must stay Never, and --softCold, '
                           '--noResetHook and --parallel false')
        if args.mode != 'Status' and not args.outDir:
            raise _Refused(f'{args.mode} writes a run folder: set OutDir')
        return args

    @staticmethod
    def _status_line(target):
        """One lane's sampler and link state, read only: what the runner's Status mode prints."""
        state = target.gt.EyeScan.readState()
        width = state.bitsPerSample if state.widthError is None else f'refused ({state.widthError})'
        fields = {} if target.link is None else target.link.statusFields()
        return (f'Lane[{target.lane}] EyeScanEn={state.eyeScanEn} ErrDetEn={state.errDetEn} '
                f'Status={state.controlStatusName} Samples={state.sampleCount} Errors={state.errorCount} '
                f'W={width} ' + ' '.join(f'{key}={value}' for key, value in fields.items()))

    def _process(self):
        import surf.xilinx.eyescan as es

        jesd = self.parent
        self.Message.setDisp('Running')
        folder = None
        try:
            args = self._arguments(es)
            plan = es.prepareEyeScan(args, jesd._numRxLanes)
            folder = plan.folder

            # Reads only: the link must be up, and a lane that is not enabled can never be.
            enable = int(jesd.JesdRx.Enable.get())
            valid = int(jesd.JesdRx.DataValid.get())
            if args.mode != 'Status' and valid != enable and not args.force:
                raise _Refused(f'the link is not up (DataValid 0x{valid:X}, Enable 0x{enable:X}) and this scan '
                               f'resets nothing to bring it up; configure the system or pass --force')
            targets = [t for t in jesd.eyeScanLanes() if t.lane in plan.lanes and (enable >> t.lane) & 1]
            if not targets:
                raise _Refused(f'no enabled lane among {plan.lanes} (Enable 0x{enable:X})')
            run = es.EyeScanPlan(lanes=[t.lane for t in targets], folder=folder)
            skipped = sorted(set(plan.lanes) - set(run.lanes))

            if args.mode == 'Status':
                lines = [self._status_line(t) for t in targets]
                code = 0
            else:
                version = self.root.getNode(amc.CARRIER.AXI_VERSION)
                code = es.runEyeScan(args, run, targets, version=version, extraMeta={
                    'bay': self._bay,
                    'jesdRxEnable': f'0x{enable:X}',
                    'resetScope': 'none: passive scan, JesdRx.ResetGTs never pulsed',
                })
                lines = [f'{args.mode} of lane(s) {run.lanes}: exit code {code}; files in {folder}']
                self.RunFolder.set(folder)
            if skipped:
                lines.append(f'lane(s) {skipped} not enabled (Enable 0x{enable:X}), skipped')
            self.Summary.set('\n'.join(lines))
            self.ExitCode.set(code)
            self.Message.setDisp('Done' if code == 0 else 'Done: a lane did not end OK or CLOSED')
        except (_Refused, ValueError) as e:
            # Refused before a lane was written: the run folder is still empty.
            if folder is not None and os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
            self.ExitCode.set(2)
            self.Message.setDisp(f'Error: refused: {e}')
        finally:
            # A run that raised anything else still ends with a code a client can wait on.
            if self.ExitCode.value() == -1:
                self.ExitCode.set(2)
                self.Message.setDisp('Error: the eye scan raised; see the server log')
