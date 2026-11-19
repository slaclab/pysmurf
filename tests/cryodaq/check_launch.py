#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Launch Checks
#-----------------------------------------------------------------------------
# File       : check_launch.py
# Created    : 2026-10-30
#-----------------------------------------------------------------------------
# Description:
# Checks what the server launcher decides from a crate slot, with no crate: the
# shelf-manager tools are replayed from one recorded transcript per slot
# (fixtures/ipmi/), and the launcher's decisions -- the FPGA's address, the PCIe
# lane, which bays are populated, the register defaults, the firmware package,
# whether to reprogram or reboot -- are checked against what those answers
# imply. It imports the server package, so it runs where the server's
# dependencies are.
#
# Two slots were recorded: a carrier whose controller answers everything, and a
# bench board with no controller, so every read fails. Answers the crate has not
# given -- an empty bay, a firmware mismatch -- are made up here and say so. So
# is what a processor system answers during a load (ssh, scp, ping, the version
# block read through its register bridge), no load having been recorded.
#
# What this guards: the parsing of each tool's output, the rules that turn an
# inventory into a command line (the platform from the carrier's record or
# --platform, never from a file name), and that nothing is written to a
# carrier -- no image load, no power cycle, no boot files copied, no reboot
# over SSH -- except when asked for and possible, and then in the order that
# keeps a half-copied image from being booted.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import argparse
import contextlib
import dataclasses
import hashlib
import io
import os
import pathlib
import sys
import tarfile
import tempfile
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'python'))
sys.path.insert(0, str(HERE / 'fixtures' / 'ipmi'))

from replay import Replay                                       # noqa: E402

from cryodaq import platform                                    # noqa: E402
from cryodaq._errors import ConnectError                        # noqa: E402
from cryodaq.platform import _atca, _soc, _umux                 # noqa: E402
from cryodaq.server import _cli, _firmware                      # noqa: E402

SLOT_2 = ['-S', 'shm-smrf-sp01', '-N', '2', '--transport', 'pcie']
SLOT_4 = ['-S', 'shm-smrf-sp01', '-N', '4', '--transport', 'tcp-bridge']
# What the launcher looks for in the firmware directory, by name: the carrier's
# image, the release's package archive and the shipped defaults.
IMAGE = 'MicrowaveMuxBpEthGen2-0x02050000-20260616105622-ruckman-af249f83.mcs'
ARCHIVE = 'rogue_MicrowaveMuxBpEthGen2_v2.5.1.zip'
DEFAULTS = ('defaults_c03_lb_lb.yml', 'defaults_c03_lb_none.yml', 'defaults_c03_none_lb.yml')
# The commands that write to a carrier: a power cycle and an image load through
# the shelf manager, a copy of boot files and a reboot over SSH.
WRITES = (' raw 0x2C 0x0A ', 'ProgramFPGA.bash', 'scp ', '/sbin/reboot')
# A processor-booted carrier's release: its boot image and package archive, and
# what the version block says before (v1.3.0) and after (v1.4.0) the load. The
# names are rfmc-cryo-det's; the stamps and the long hashes are made up.
BOOT_IMAGE = 'MicrowaveMuxRfmcCarrier-0x01040000-20261113191646-ruckman-54dc168.linux.tar.gz'
BOOT_ARCHIVE = 'rogue_MicrowaveMuxRfmcCarrier_v1.4.0.zip'
PS = '10.0.1.204'
RUNNING_OLD = _firmware.FirmwareVersion(0x01030000, 'd86406d' + '0' * 33, 'MicrowaveMuxRfmcCarrier: made up')
RUNNING_NEW = _firmware.FirmwareVersion(0x01040000, '54dc168' + '1' * 33, 'MicrowaveMuxRfmcCarrier: made up')
# What a read of the version block gets while the system is down, or its bridge accepts connections and does not serve.
DOWN = ConnectError("could not read the firmware's version block -- made up")


def firmware_dir(tmp, archive=ARCHIVE):
    root = pathlib.Path(tmp) / 'fw'
    (root / 'smurf_cfg' / 'defaults').mkdir(parents=True)
    for name in (IMAGE,) + tuple(f'smurf_cfg/defaults/{d}' for d in DEFAULTS):
        (root / name).touch()
    if archive:
        # A real archive with a python/ directory, so the launcher can put it on the path.
        with zipfile.ZipFile(root / archive, 'w') as zf:
            zf.writestr('python/.keep', '')
    return root


@contextlib.contextmanager
def crate(replay):
    """The shelf-manager tools answered by ``replay`` instead of run, and the boot waits skipped."""
    real_run, real_sleep = _atca.run_tool, _atca.time.sleep
    _atca.run_tool, _atca.time.sleep = replay, lambda s: None
    try:
        yield replay
    finally:
        _atca.run_tool, _atca.time.sleep = real_run, real_sleep


def launch(argv, replay, fw):
    parser = _cli.add_arguments(argparse.ArgumentParser())
    args = parser.parse_args(list(argv) + ['--firmware-dir', str(fw)])
    with crate(replay):
        return args, _cli.launch(args, parser)


def launched(argv, run=None, archive=ARCHIVE):
    """Parse ``argv`` over a fresh firmware directory and run the launcher; the replay's calls come back too."""
    run = Replay() if run is None else run
    with tempfile.TemporaryDirectory() as tmp:
        args, done = launch(argv, run, firmware_dir(tmp, archive))
        return args, done, run


def refused(argv, run=None, *, naming):
    """The launcher must refuse ``argv`` with a message that carries every word in ``naming``."""
    try:
        launched(argv, run)
    except SystemExit as e:
        missing = [w for w in naming if w not in str(e)]
        assert not missing, f"{' '.join(argv)}: the refusal does not name {missing}: {e}"
    else:
        raise AssertionError(f"{' '.join(argv)} was launched")


def wrote(replay):
    return [c for c in replay.calls if any(w in c for w in WRITES)]


def boot_image(fw, name=BOOT_IMAGE):
    """A boot image in ``fw`` with made-up contents; returns ``{file name: sha256}`` of what it holds."""
    sums = {}
    with tarfile.open(fw / name, 'w:gz') as tar:
        for member in _soc.BOOT_FILES:
            data = f"{member} of {name}".encode()
            info = tarfile.TarInfo(f"{_soc.IMAGE_DIR}/{member}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
            sums[member] = hashlib.sha256(data).hexdigest()
    return sums


class Board:
    """A processor system's tool answers, made up here, over a `Replay` for the crate's.

    ``sha256sum`` reports what was sent; after a reboot command ``ping`` fails
    ``down_for`` times, then answers again. Every call is logged on the replay.
    """

    def __init__(self, replay, sums, down_for=2):
        self.replay, self.sums, self.down_for, self.down = replay, sums, down_for, 0

    @property
    def calls(self):
        return self.replay.calls

    def __call__(self, argv, timeout=None):
        tool = os.path.basename(str(argv[0]))
        command = ' '.join([tool] + [str(a) for a in argv[1:]])
        if tool == 'scp':
            assert timeout == _soc.COPY_TIMEOUT_S, f"the copy must be given its own timeout, not {timeout}"
        if tool == 'ssh' and 'sha256sum' in command:
            self.replay.calls.append(command)
            return 0, ''.join(f"{d}  {_soc.BOOT_DIR}/{n}\n" for n, d in self.sums.items())
        if tool == 'ssh' and '/sbin/reboot' in command:
            self.down = self.down_for
        if tool == 'ping' and self.down > 0:
            self.down -= 1
            self.replay.calls.append(command)
            return 1, ''
        return self.replay(argv)


@contextlib.contextmanager
def processor(versions):
    """The version block answered from ``versions`` in turn (an exception instance is raised), waits skipped."""
    answers = list(versions)

    def bridge_version(address, port=_soc.REGISTER_BRIDGE_PORT):
        assert answers, f"the version block at {address} was read more often than prepared for"
        # The probe imports the firmware package's version block: the package must be on the path by now.
        assert any(p.endswith('.zip/python') for p in sys.path), \
            'the firmware package was not put on the path before the version block was probed'
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer
    real = _firmware.bridge_version, _soc.time.sleep, _soc.port_open
    _firmware.bridge_version, _soc.time.sleep, _soc.port_open = \
        bridge_version, lambda s: None, lambda a, p, timeout=2.0: True
    try:
        yield answers
    finally:
        _firmware.bridge_version, _soc.time.sleep, _soc.port_open = real


def soc_launch(argv, versions, run=None, image=BOOT_IMAGE, archive=BOOT_ARCHIVE):
    """Launch over a firmware directory holding a processor-booted release, the board made up."""
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, archive)
        (fw / IMAGE).unlink()
        sums = boot_image(fw, image) if image else {}
        board = Board(run or Replay(), sums)
        with processor(versions) as left:
            args, done = launch(argv, board, fw)
        return args, done, board, left

# --------------------------------------------------------------------------
# a slot whose controller answers
# --------------------------------------------------------------------------


def check_the_inventory_is_what_the_crate_said():
    """Every tool's output parsed: byte order, FRU fields, and absent against unknown."""
    _, done, _ = launched(SLOT_2)
    inv = done.inventory
    assert inv['crate_id'] == 1 and inv['ipmb'] == '0x84' and inv['slot'] == 2, inv
    assert inv['firmware_git_hash'] == 'af249f8', f"the IPMC's bytes are reversed: {inv['firmware_git_hash']}"
    assert inv['boot_state'] == 3, inv['boot_state']
    assert inv['carrier']['Board Part Number'] == 'PC_379_396_38_C03_A04', inv['carrier']
    assert inv['carrier']['Board Product'] == 'AMCC-KU15P', inv['carrier']
    assert inv['amc0']['Board Part Number'] == inv['amc2']['Board Part Number'] == 'PC_379_396_30_C03_A01', inv
    assert inv['rtm'] is False and inv['rtm_present'] is False and inv['errors'] == {}, \
        f"no rear module is fitted; the record must say absent, not unknown: {inv['rtm']!r} {inv['errors']}"
    # The same slot after a rear module was fitted (recorded).
    _, done, _ = launched(SLOT_2, Replay(variant='rtm-fitted'))
    rtm = done.inventory['rtm']
    assert done.inventory['rtm_present'] is True and rtm['Board Product'] == 'LCLS-II CRYO RTM', rtm


def check_a_slot_that_answers_decides_the_address_lane_population_and_package():
    args, done, replay = launched(SLOT_2)
    assert args.ip_addr == '10.0.1.102', f"10.0.<crate>.<100 + slot>: {args.ip_addr}"
    assert args.pcie_rssi_lane == 0, f"slot - 2: {args.pcie_rssi_lane}"
    assert not args.disable_bay0 and not args.disable_bay1, 'both bays are populated on this slot'
    assert args.enable_pwri2c, 'a C03 carrier carries the power monitor'
    assert [os.path.basename(l) for l in args.layers] == ['defaults_c03_lb_lb.yml'], args.layers
    assert args.firmware and args.firmware.endswith(ARCHIVE), args.firmware
    assert not args.configure and wrote(replay) == [], \
        f"the FPGA runs the image on disk and no boot was asked for: {wrote(replay)}"


def check_an_empty_bay_is_left_out_and_names_the_other_bays_defaults():
    """Synthetic: both bays are populated in the crate. The inventory tool prints nothing for an empty one."""
    args, _, _ = launched(SLOT_2, Replay(overrides={'cba_amc_init --dump shm-smrf-sp01/2/2': (0, '')}))
    assert args.disable_bay1 and not args.disable_bay0, (args.disable_bay0, args.disable_bay1)
    assert [os.path.basename(l) for l in args.layers] == ['defaults_c03_lb_none.yml'], args.layers


def check_a_bay_that_could_not_be_read_is_refused_not_left_out():
    """Synthetic: the inventory tool fails on one bay. Unknown is not empty; the user overrides with -E."""
    failing = Replay(overrides={'cba_amc_init --dump shm-smrf-sp01/2/2': (1, 'IPMB timeout')})
    refused(SLOT_2, failing, naming=('AMC bay 2', 'IPMB timeout', '-E'))
    args, _, _ = launched(SLOT_2 + ['-E', '--disable-bay1', '-d', '/site/defaults.yml'], failing)
    assert args.disable_bay1 and args.layers == ['/site/defaults.yml'], (args.disable_bay1, args.layers)


def check_a_carrier_record_that_could_not_be_read_is_refused_unless_nothing_is_decided_from_it():
    """Synthetic: the crate id reads but the carrier's FRU record does not. Unknown is not an unknown carrier."""
    failing = Replay(overrides={'fru print 0': (1, 'Unable to read FRU')})
    refused(SLOT_2, failing, naming=('carrier', 'Unable to read FRU', '-E and -D'))
    refused(SLOT_2 + ['-E'], failing, naming=('-E and -D',))
    args, done, replay = launched(SLOT_2 + ['-E', '-D', '--disable-bay1', '-d', '/site/defaults.yml'], failing)
    assert done.inventory['carrier'] is None and args.ip_addr == '10.0.1.102' and wrote(replay) == [], \
        (done.inventory['carrier'], args.ip_addr, wrote(replay))
    # With no record to choose the package by, --platform does, as on the other paths.
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp)
        (fw / 'rogue_MicrowaveMuxZcu208_v3.2.1.zip').touch()
        args, _ = launch(SLOT_2 + ['-E', '-D', '--platform', 'umux-amc'], failing, fw)
        assert args.firmware.endswith(ARCHIVE), args.firmware


def check_two_archives_are_refused_before_anything_reaches_the_board():
    """A directory with two releases is refused before a hard boot or a load touches the FPGA."""
    stale = Replay(overrides={'raw 0x34 0x04 0xe0 0x04': (0, ' 00 11 22 33\n')})
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp)
        (fw / 'rogue_MicrowaveMuxBpEthGen2_v2.6.0.zip').touch()
        try:
            launch(SLOT_2 + ['-H'], stale, fw)
        except SystemExit as e:
            assert '2 firmware packages' in str(e), str(e)
        else:
            raise AssertionError('two archives were not refused')
    assert wrote(stale) == [], f"the board was touched before the refusal: {wrote(stale)}"


def check_a_tool_that_does_not_answer_is_a_failed_read():
    """Synthetic: a tool hangs past its timeout; the read is recorded as failed, not raised through."""
    import subprocess

    def hang(*a, **kw):
        raise subprocess.TimeoutExpired(a[0], kw.get('timeout'))
    real_run, real_which = _atca.subprocess.run, _atca.shutil.which
    _atca.subprocess.run, _atca.shutil.which = hang, lambda name: '/usr/bin/' + name
    try:
        try:
            _atca.run_tool(['ipmitool', 'raw', '0x34', '0xF4'])
        except _atca.ShelfManagerError as e:
            assert 'no answer' in str(e), str(e)
        else:
            raise AssertionError('a timeout was not a ShelfManagerError')
        inv = _atca.ShelfManager('shm', 4).inventory()
        assert inv['crate_id'] is None and set(inv['errors']) >= {'crate_id', 'carrier', 'amc0'}, inv
    finally:
        _atca.subprocess.run, _atca.shutil.which = real_run, real_which


def check_what_the_command_line_gives_is_kept():
    """An option given beats the decided one, and a user's layer follows the population's so it overrides it."""
    args, _, _ = launched(SLOT_2 + ['-a', '10.0.9.9', '-l', '3', '--disable-bay1', '-d', '/site/override.yml'])
    assert args.ip_addr == '10.0.9.9' and args.pcie_rssi_lane == 3 and args.disable_bay1, \
        'what the command line gave was overwritten by what the launcher decided'
    assert [os.path.basename(l) for l in args.layers] == ['defaults_c03_lb_lb.yml', 'override.yml'], args.layers


def check_a_hard_boot_power_cycles_through_the_shelf_manager_or_reboots_the_processor_over_ssh_then_configures():
    """The AMC's FPGA is power-cycled and its boot state polled; the RFMC's processor system is rebooted over SSH. Either way the firmware is checked on the booted FPGA, and the register configuration is applied at start-up."""
    args, _, replay = launched(SLOT_2 + ['-H'])
    cycles = [c for c in replay.calls if ' raw 0x2C 0x0A ' in c]
    assert len(cycles) == 2 and cycles[0].endswith('0 0 2 0') and cycles[1].endswith('0 0 1 0'), \
        f"a reboot is the power-down then power-up pair: {cycles}"
    polled = [i for i, c in enumerate(replay.calls) if c.endswith('raw 0x34 0xF4')]
    hashed = [i for i, c in enumerate(replay.calls) if c.endswith('raw 0x34 0x04 0xe0 0x04')]
    assert polled and hashed and max(polled) < max(hashed), \
        'the boot state is polled after the power cycle and the firmware is checked on the booted FPGA'
    assert args.configure, 'a hard boot applies the register configuration at start-up'
    # The version block is read once the system is back (the reboot's own wait), then for the check.
    args, done, board, left = soc_launch(BENCH + ['-H'], [RUNNING_NEW, RUNNING_NEW])
    assert wrote(board) == [f"ssh {' '.join(_soc.SSH_OPTIONS)} root@{PS} /bin/sync && /sbin/reboot"], wrote(board)
    assert left == [] and args.configure, 'the image is checked after the reboot, and the configuration applied'
    # The bridge accepts connections before it serves: reads that fail while the system comes back are retried.
    args, done, board, left = soc_launch(BENCH + ['-H'], [DOWN, DOWN, RUNNING_NEW, RUNNING_NEW])
    assert left == [] and args.configure and 'programmed' not in done.decided, (left, done.decided)
    # Without the crate the processor's address and its platform must be given.
    # Over PCIe the transport asks for no address, so the refusal is the reboot's own: what would let it happen.
    soc_refused(['-a', '10.0.1.104', '--transport', 'pcie', '-H'], [], image=None,
                naming=('--ps-addr', '--shelfmanager', '--platform'))
    soc_refused(['-a', '10.0.1.104', '--transport', 'pcie', '--platform', 'umux-rfmc', '-H'], [], image=None,
                naming=('--ps-addr',))


def check_a_firmware_mismatch_loads_the_image_and_a_match_does_not():
    """Synthetic: the FPGA runs the image on disk. A different hash from the controller means a load."""
    stale = {'raw 0x34 0x04 0xe0 0x04': (0, ' 00 11 22 33\n')}
    args, done, replay = launched(SLOT_2, Replay(overrides=stale))
    assert done.decided.get('programmed') == IMAGE, done.decided
    assert [c for c in wrote(replay) if 'ProgramFPGA.bash' in c], replay.calls
    assert args.configure, 'a load reboots the FPGA, so the register configuration is applied at start-up'
    args, done, replay = launched(SLOT_2 + ['-D'], Replay(overrides=stale))
    assert 'programmed' not in done.decided and wrote(replay) == [] and not args.configure, \
        '-D must skip the firmware check'


def check_a_carrier_whose_processor_boots_it_is_checked_through_its_bridge_not_the_shelf_manager():
    """Synthetic: slot 2's controller reports an RFMC carrier's part number. Its image is a boot image its processor loads: no .mcs is looked for, the running version comes from the bridge at the derived processor address, and a boot image that is not there is a refusal."""
    rfmc_record = (0, ' Board Mfg             : SLAC\n Board Product         : RFMC\n'
                      ' Board Part Number     : PC_379_396_55_C01\n')
    args, done, board, _ = soc_launch(SLOT_2, [RUNNING_NEW], Replay(overrides={'fru print 0': rfmc_record}))
    assert args.firmware and args.firmware.endswith(BOOT_ARCHIVE), args.firmware
    assert args.ip_ps == '10.0.1.202', f"the processor system at +200: {args.ip_ps}"
    assert done.inventory['firmware_running'] == '54dc168', done.inventory
    assert 'programmed' not in done.decided and wrote(board) == [] and not args.configure, \
        f"a matching boot image was loaded: {done.decided}, {wrote(board)}"
    assert not [c for c in board.calls if 'ProgramFPGA' in c or '.mcs' in c], board.calls
    soc_refused(SLOT_2, [RUNNING_NEW], Replay(overrides={'fru print 0': rfmc_record}), image=None,
                naming=('no image', 'MicrowaveMuxRfmcCarrier', '-D'))


# --------------------------------------------------------------------------
# a slot whose controller does not answer
# --------------------------------------------------------------------------


def check_a_board_with_no_controller_is_refused_without_addresses_and_tolerated_with_them_and_its_platform():
    """With no record the addresses and the platform come from the command line. Here the AMC's: its loader is the shelf manager that did not answer, so nothing is checked and nothing written."""
    refused(SLOT_4, naming=('crate id', '--addr'))
    refused(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204'], naming=('--platform', '-D'))
    args, done, replay = launched(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204', '--platform', 'umux-amc'])
    assert args.ip_addr == '10.0.1.104' and args.ip_ps == '10.0.1.204', 'the given addresses were changed'
    assert args.pcie_rssi_lane == 2, 'the lane still follows from the slot'
    inv = done.inventory
    assert inv['crate_id'] is None and 'crate_id' in inv['errors'], inv
    assert inv['carrier'] is None and inv['rtm_present'] is None and 'rtm' in inv['errors'], \
        'the IPMC did not answer, so the carrier and the rear module are unknown, not absent'
    assert inv['amc0'] is False, 'the inventory tool said no board in the bay: absent'
    assert wrote(replay) == [] and not args.configure, f"written through a controller that does not answer: {wrote(replay)}"
    refused(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204', '--platform', 'umux-amc', '-H'],
            naming=('hard-boot', '--shelfmanager'))

# --------------------------------------------------------------------------
# a carrier its processor system boots: the boot image over SSH (all made up)
# --------------------------------------------------------------------------

BENCH = SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', PS, '--platform', 'umux-rfmc']


def soc_refused(argv, versions, run=None, *, naming, **kwargs):
    """`soc_launch` must refuse ``argv`` with a message that carries every word in ``naming``."""
    try:
        soc_launch(argv, versions, run, **kwargs)
    except SystemExit as e:
        missing = [w for w in naming if w not in str(e)]
        assert not missing, f"{' '.join(argv)}: the refusal does not name {missing}: {e}"
    else:
        raise AssertionError(f"{' '.join(argv)} was launched")


def check_a_boot_image_the_fpga_does_not_run_is_copied_and_rebooted_into_and_one_it_runs_is_left_alone():
    """Synthetic: the bench board runs v1.3.0 and v1.4.0 is on disk. Copy, check the copy, reboot, read back -- in that order -- then configure. A read-back still on v1.3.0 is a refusal. Running v1.4.0, nothing is written, and -D reads nothing."""
    args, done, board, left = soc_launch(BENCH, [RUNNING_OLD, RUNNING_NEW])
    assert done.decided.get('programmed') == BOOT_IMAGE and args.configure, done.decided
    assert left == [], 'the version block was to be read twice: before and after'
    writes = wrote(board)
    assert len(writes) == 2 and writes[0].startswith('scp ') and writes[1].endswith('/bin/sync && /sbin/reboot'), \
        f"the flush must succeed for the reboot to run: {writes}"
    assert writes[0].endswith(f"root@{PS}:{_soc.BOOT_DIR}/.") and all(f"/{_soc.IMAGE_DIR}/{f} " in writes[0]
                                                                     for f in _soc.BOOT_FILES), writes[0]
    kinds = [('copy' if c.startswith('scp ') else 'check' if 'sha256sum' in c else 'reboot')
             for c in board.calls if c.startswith('scp ') or 'sha256sum' in c or '/sbin/reboot' in c]
    assert kinds == ['copy', 'check', 'reboot'], f"copy, check, reboot, in that order: {board.calls}"
    assert done.inventory['firmware_image_on_disk'] == BOOT_IMAGE
    assert done.inventory['firmware_running'] == '54dc168', done.inventory
    try:
        soc_launch(BENCH, [RUNNING_OLD, RUNNING_OLD])
    except SystemExit as e:
        assert 'did not take' in str(e) and _soc.BOOT_DIR in str(e), str(e)
    else:
        raise AssertionError('the old firmware after the reboot was taken as a load')
    args, done, board, _ = soc_launch(BENCH, [RUNNING_NEW])
    assert 'programmed' not in done.decided and wrote(board) == [] and not args.configure, (done.decided, wrote(board))
    assert not [c for c in board.calls if c.startswith('ssh')], f"ssh was run to read the version: {board.calls}"
    args, done, board, _ = soc_launch(BENCH + ['-D'], [])
    assert 'firmware_running' not in done.inventory and wrote(board) == [] and not args.configure, \
        '-D must neither read the version block nor write'


def check_a_boot_image_that_cannot_be_checked_is_a_refusal_not_a_skip():
    """No processor address, no platform named, a silent bridge, a named bench with no image: each names what would let the start go on (-D among them, except after a load). No FPGA address for the transport is refused before the board is written to. An emulated server has no board and is asked for none of it."""
    # Over PCIe the transport does not need the processor's address; the image check does, or -D.
    soc_refused(['-S', 'shm-smrf-sp01', '-N', '4', '--transport', 'pcie', '-a', '10.0.1.104', '--platform', 'umux-rfmc'],
                [], naming=('--ps-addr', '-D'))
    # Without a slot nothing derives the FPGA's address either; the tcp-bridge transport needs it, so the start is
    # refused at the addresses -- not after a load and a reboot the transport would then refuse to use.
    run = Replay()
    soc_refused(['--transport', 'tcp-bridge', '--platform', 'umux-rfmc', '--ps-addr', PS, '-H'],
                [RUNNING_OLD, RUNNING_NEW], run, naming=('--addr', 'tcp-bridge'))
    assert wrote(run) == [], f"written to a board the transport could not reach: {wrote(run)}"
    # The image on disk does not name the platform: with no record and no --platform it is a refusal, not a guess.
    soc_refused(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', PS], [], naming=('--platform', '-D', BOOT_IMAGE))
    soc_refused(BENCH, [], image=None, naming=('no image', 'MicrowaveMuxRfmcCarrier', '-D'))
    soc_refused(BENCH, [DOWN], naming=('version block', '-D'))
    # After a reboot the bridge accepts connections before it serves: the read is retried, then refused without -D.
    try:
        soc_launch(BENCH, [RUNNING_OLD] + [DOWN] * _soc.VERSION_RETRIES)
    except SystemExit as e:
        assert 'copied' in str(e) and 'version block' in str(e) and '-D' not in str(e), f"after a load, -D is no advice: {e}"
    else:
        raise AssertionError('the bridge never answered after the reboot, and the launch went on')

    def no_tool(argv, timeout=None):
        raise AssertionError(f"a tool was run: {argv}")
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, BOOT_ARCHIVE)
        (fw / IMAGE).unlink()
        boot_image(fw)
        with processor([]):
            args, done = launch(['--transport', 'emulation', '--platform', 'umux-rfmc'], no_tool, fw)
            assert args.firmware.endswith(BOOT_ARCHIVE) and done.inventory == {} and not args.configure, (args, done)
            refused(['--transport', 'emulation', '--platform', 'umux-rfmc', '-H'], no_tool, naming=('emulated',))


def check_the_platform_comes_from_the_record_or_the_option_never_from_a_file_name():
    # Two images of the named platform are refused by count, a stranger's image is no image of the platform's.
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, BOOT_ARCHIVE)
        (fw / IMAGE).unlink()
        boot_image(fw)
        boot_image(fw, 'MicrowaveMuxRfmcCarrier-0x01030000-20261112164613-ruckman-d86406d.linux.tar.gz')
        with processor([]):
            try:
                launch(BENCH, Board(Replay(), {}), fw)
            except SystemExit as e:
                assert '2 images' in str(e), str(e)
            else:
                raise AssertionError('two boot images were not refused')
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, BOOT_ARCHIVE)
        (fw / IMAGE).unlink()
        boot_image(fw, 'SomethingElse-0x01000000-20261101000000-nobody-0123abc.linux.tar.gz')
        with processor([]):
            try:
                launch(BENCH, Board(Replay(), {}), fw)
            except SystemExit as e:
                assert 'no image' in str(e) and 'MicrowaveMuxRfmcCarrier' in str(e), str(e)
            else:
                raise AssertionError("a stranger's image was taken for the platform's")
            # Un-named, the image on disk is a refusal naming the option; named AMC with no crate, the AMC's
            # loader (the shelf manager) is out of reach and the stranger's image is nobody's: nothing is checked.
            try:
                launch(['-a', '10.0.1.102', '--transport', 'pcie'], Board(Replay(), {}), fw)
            except SystemExit as e:
                assert '--platform' in str(e) and 'SomethingElse' in str(e), str(e)
            else:
                raise AssertionError('an image on disk with no platform named was not refused')
            args, _ = launch(['-a', '10.0.1.102', '--transport', 'pcie', '--platform', 'umux-amc'],
                             Board(Replay(), {}), fw)
            assert not args.configure, args
    # The declaration refuses what the launcher could not act on.
    for kwargs in ({'image_loader': 'sd-card'}, {'image_loader': _umux.PROCESSOR_SYSTEM}):
        try:
            _umux.BringUp(firmware_image='X', **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"BringUp accepted {kwargs}")
    # With a record, the record wins: --platform agreeing changes nothing, disagreeing is a refusal.
    args, _, _ = launched(SLOT_2 + ['--platform', 'umux-amc'])
    assert args.firmware.endswith(ARCHIVE) and args.enable_pwri2c, (args.firmware, args.enable_pwri2c)
    refused(SLOT_2 + ['--platform', 'umux-rfmc'], naming=('PC_379_396_38', 'umux-rfmc'))
    # The crate id is a read of its own: a record read when the id was not still names the platform and still
    # refuses a --platform that disagrees; what follows from the id (the addresses, the shelf manager's reach) is lost.
    no_id = Replay(overrides={'raw 0x34 0x04 0xFD 0x02': (1, 'Unable to send RAW command')})
    refused(SLOT_2 + ['-a', '10.0.1.102', '--platform', 'umux-rfmc'], no_id, naming=('PC_379_396_38', 'umux-rfmc'))
    args, done, replay = launched(SLOT_2 + ['-a', '10.0.1.102'], no_id)
    assert done.inventory['crate_id'] is None and args.firmware.endswith(ARCHIVE) and wrote(replay) == [], \
        (done.inventory['crate_id'], args.firmware, wrote(replay))
    # A part number no platform declares: --platform says which; without it, an image on disk is a refusal.
    unknown = Replay(overrides={'fru print 0': (0, ' Board Mfg             : SLAC\n Board Part Number     : PC_999_000_01\n')})
    refused(SLOT_2, unknown, naming=('--platform', '-D', IMAGE))
    args, done, replay = launched(SLOT_2 + ['-D'], unknown)
    assert args.ip_addr == '10.0.1.102' and args.firmware.endswith(ARCHIVE) and wrote(replay) == [], (args, wrote(replay))
    args, done, replay = launched(SLOT_2 + ['--platform', 'umux-amc', '-E'], unknown)
    assert args.firmware.endswith(ARCHIVE) and 'layers' not in done.decided and wrote(replay) == [] and not args.configure, \
        (args.firmware, done.decided, wrote(replay))
    assert done.inventory['firmware_image_on_disk'] == IMAGE, 'the named platform has its .mcs checked through the shelf manager'

# --------------------------------------------------------------------------
# no slot, and the package
# --------------------------------------------------------------------------


def check_without_a_slot_nothing_is_read_and_nothing_reaches_the_crate():
    def no_tool(argv):
        raise AssertionError(f"a tool was run: {argv}")
    addressed = ['-a', '10.0.1.102', '--transport', 'pcie']
    # An image on disk and no platform named is a refusal; named, or with the check called off, the start goes on.
    refused(addressed, no_tool, naming=('--platform', '-D', IMAGE))
    for argv in (addressed + ['--platform', 'umux-amc'], addressed + ['-D']):
        args, done, _ = launched(argv, no_tool)
        assert done.inventory == {} and args.pcie_rssi_lane == 0, (done.inventory, args.pcie_rssi_lane)
        assert args.firmware and args.firmware.endswith(ARCHIVE), 'the package is still found beside the image'
        assert not args.configure, args
    refused(addressed + ['--platform', 'umux-amc', '-H'], no_tool, naming=('--shelfmanager', '--slot'))
    refused(['-S', 'shm-smrf-sp01'], no_tool, naming=('--slot',))
    refused(['-N', '2'], no_tool, naming=('--shelfmanager',))
    refused(['-S', 'shm-smrf-sp01', '-N', '9'], no_tool, naming=('slot',))
    # The old spellings of the two switches are still taken.
    parser = _cli.add_arguments(argparse.ArgumentParser())
    args = parser.parse_args(['--disable-hw-detect', '--no-check-fw'])
    assert args.no_hw_detect and not args.check_firmware, args


def check_the_platform_name_chooses_among_several_platforms_archives_without_a_record():
    """A deployment ships every platform's archive; with no carrier record, --platform says whose."""
    others = ('rogue_MicrowaveMuxZcu208_v3.2.1.zip', 'rogue_MicrowaveMuxRfmcCarrier_v2.0.0.zip')
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp)
        for name in others:
            (fw / name).touch()
        addressed = ['-a', '10.0.1.102', '--transport', 'pcie']
        try:
            launch(addressed, Replay(), fw)
        except SystemExit as e:
            assert '3 firmware packages' in str(e), str(e)
        else:
            raise AssertionError('three archives were not refused')
        args, _ = launch(addressed + ['--platform', 'umux-amc'], Replay(), fw)
        assert args.firmware.endswith(ARCHIVE), args.firmware
        # The platform named, its image is expected: -D, since this directory holds the AMC's only.
        args, _ = launch(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204', '--platform', 'umux-rfmc', '-D'],
                         Replay(), fw)
        assert args.firmware.endswith(others[1]), args.firmware
        # The detected defaults path is absolute whatever the directory was given as.
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            args, _ = launch(SLOT_2, Replay(), pathlib.Path('fw'))
        finally:
            os.chdir(cwd)
        assert os.path.isabs(args.layers[0]) and os.path.isabs(args.firmware), (args.layers, args.firmware)


def check_the_firmware_package_is_the_one_archive_or_else_the_one_checkout():
    """The firmware directory is the one place a package is looked for; two candidates of a kind is a refusal."""
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, archive='')
        _, done = launch(SLOT_2, Replay(), fw)
        assert 'firmware' not in done.decided, done.decided
        # A checkout beside the image is the package; a directory that is not a
        # checkout is not.
        (fw / 'notes' / 'firmware').mkdir(parents=True)
        for d in ('cryo-det/firmware/python', 'cryo-det/firmware/submodules/surf/python'):
            (fw / d).mkdir(parents=True)
        args, done = launch(SLOT_2, Replay(), fw)
        assert os.path.relpath(args.firmware, fw) == 'cryo-det', args.firmware
        (fw / 'other/firmware/python').mkdir(parents=True)
        try:
            launch(SLOT_2, Replay(), fw)
        except SystemExit as e:
            assert '2 firmware checkouts' in str(e), str(e)
        else:
            raise AssertionError('two checkouts were not refused')
        # An archive wins over a checkout; two archives are refused by name.
        (fw / ARCHIVE).touch()
        args, done = launch(SLOT_2, Replay(), fw)
        assert args.firmware.endswith(ARCHIVE), args.firmware
        (fw / 'rogue_MicrowaveMuxBpEthGen2_v2.6.0.zip').touch()
        try:
            launch(SLOT_2, Replay(), fw)
        except SystemExit as e:
            assert '2 firmware packages' in str(e) and 'v2.6.0' in str(e), str(e)
        else:
            raise AssertionError('two archives were not refused')
        # The directory itself may be the checkout (then it holds no image or
        # defaults, so neither is looked for).
        _, done = launch(SLOT_2 + ['-E', '-D'], Replay(), fw / 'cryo-det')
        assert done.decided.get('firmware') == str(fw / 'cryo-det'), done.decided


def check_a_relative_layer_is_found_in_the_package_or_the_configuration_directory_not_both():
    """One resolver for every layer: the package's configuration directory and the deployment's are the
    two places a relative name may be, the working directory never; a name in both is refused, not guessed."""
    import zipfile
    from cryodaq._errors import ConfigError
    from cryodaq.server._firmware import resolve_layers
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp)
        with zipfile.ZipFile(fw / ARCHIVE, 'w') as zf:
            zf.writestr('CryoDet/config/defaults.yml', 'in: package\n')
            zf.writestr('CryoDet/config/defaults_c03_lb_lb.yml', 'in: package too\n')
        site = fw / 'smurf_cfg' / 'defaults'
        (pathlib.Path(tmp) / 'defaults.yml').touch()                # beside the cwd: never taken
        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            # The deployment's defaults directory is <firmware-dir>/smurf_cfg/defaults; --defaults-dir moves it.
            args, done = launch(SLOT_2, Replay(), fw)
            assert args.defaults_dir == str(site), args.defaults_dir
            assert args.layers == [str(site / 'defaults_c03_lb_lb.yml')], args.layers
            other = pathlib.Path(tmp) / 'site'
            other.mkdir()
            (other / 'defaults_c03_lb_lb.yml').touch()
            args, done = launch(SLOT_2 + ['--defaults-dir', str(other)], Replay(), fw)
            assert args.layers == [str(other / 'defaults_c03_lb_lb.yml')], args.layers
            refused(SLOT_2 + ['--defaults-dir', str(pathlib.Path(tmp) / 'nowhere')],
                    naming=['defaults_c03_lb_lb.yml', 'nowhere', '--layer'])
            # A relative --defaults-dir is made absolute, as the firmware directory is, so the
            # composition takes the resolved file as given instead of resolving it again.
            args, done = launch(SLOT_2 + ['--defaults-dir', 'site'], Replay(), fw)
            assert args.defaults_dir == str(other) and args.layers == [str(other / 'defaults_c03_lb_lb.yml')], \
                (args.defaults_dir, args.layers)
            # The map's package directory moved to where this archive keeps its files; it ships no defaults.
            amc = dataclasses.replace(platform.by_name('umux-amc'), package_config_dir='CryoDet/config')
            assert resolve_layers(args.layers, fw / ARCHIVE, amc, defaults_dir=args.defaults_dir) == args.layers

            # The composition's resolution, one function: a package member, a deployment file, both, neither.
            def resolve(names, **kw):
                return resolve_layers(names, fw / ARCHIVE, amc, **kw)
            assert resolve(['defaults.yml'], defaults_dir=site) == [f"{fw / ARCHIVE}/CryoDet/config/defaults.yml"]
            assert resolve(['defaults_c03_lb_none.yml'], defaults_dir=site) == [str(site / 'defaults_c03_lb_none.yml')]
            assert resolve(['/abs/x.yml'], defaults_dir=site) == ['/abs/x.yml']
            try:
                resolve(['defaults_c03_lb_lb.yml'], defaults_dir=site)
            except ConfigError as e:
                assert 'both' in str(e) and str(site) in str(e) and ARCHIVE in str(e), str(e)
            else:
                raise AssertionError('a name in both places was not refused')
            try:
                resolve(['missing.yml'], defaults_dir=site)
            except FileNotFoundError as e:
                assert str(site) in str(e) and ARCHIVE in str(e), str(e)
            else:
                raise AssertionError('a name in neither place was accepted')
            # Nowhere to look: as given (rogue's loader reports what it cannot open).
            assert resolve_layers(['defaults.yml'], None, None) == ['defaults.yml']
        finally:
            os.chdir(cwd)

# --------------------------------------------------------------------------


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    checks = sorted((name[len('check_'):], fn) for name, fn in globals().items() if name.startswith('check_'))
    failed = []
    print(f"Checking the server launcher against recorded crate answers ({len(checks)} checks)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            failed.append(label)
            print(f"  FAIL  {label}")
            print(f"          {type(e).__name__}: {str(e)[:300]}")
        else:
            print(f"  ok    {label}")
    print("")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
