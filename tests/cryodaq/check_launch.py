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
# given -- an empty bay, a firmware mismatch -- are made up here and say so.
#
# What this guards: the parsing of each tool's output, the rules that turn an
# inventory into a command line, and that nothing is written to a carrier --
# no image load, no power cycle -- except when asked for and possible.
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
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'python'))
sys.path.insert(0, str(HERE / 'fixtures' / 'ipmi'))

from replay import Replay                                       # noqa: E402

from cryodaq import platform                                    # noqa: E402
from cryodaq.platform import _atca                              # noqa: E402
from cryodaq.server import _cli                                 # noqa: E402

SLOT_2 = ['-S', 'shm-smrf-sp01', '-N', '2', '--transport', 'pcie']
SLOT_4 = ['-S', 'shm-smrf-sp01', '-N', '4', '--transport', 'tcp-bridge']
# What the launcher looks for in the firmware directory, by name: the carrier's
# image, the release's package archive and the shipped defaults.
IMAGE = 'MicrowaveMuxBpEthGen2-0x02050000-20260616105622-ruckman-af249f83.mcs'
ARCHIVE = 'rogue_MicrowaveMuxBpEthGen2_v2.5.1.zip'
DEFAULTS = ('defaults_c03_lb_lb.yml', 'defaults_c03_lb_none.yml', 'defaults_c03_none_lb.yml')
# The ipmitool commands that write to a carrier: a power cycle and an image load.
WRITES = (' raw 0x2C 0x0A ', 'ProgramFPGA.bash')


def firmware_dir(tmp, archive=ARCHIVE):
    root = pathlib.Path(tmp) / 'fw'
    (root / 'smurf_cfg' / 'defaults').mkdir(parents=True)
    for name in (IMAGE, archive) + tuple(f'smurf_cfg/defaults/{d}' for d in DEFAULTS):
        if name:
            (root / name).touch()
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


def check_a_hard_boot_power_cycles_polls_the_boot_state_and_configures():
    args, _, replay = launched(SLOT_2 + ['-H'])
    cycles = [c for c in replay.calls if ' raw 0x2C 0x0A ' in c]
    assert len(cycles) == 2 and cycles[0].endswith('0 0 2 0') and cycles[1].endswith('0 0 1 0'), \
        f"a reboot is the power-down then power-up pair: {cycles}"
    polled = [i for i, c in enumerate(replay.calls) if c.endswith('raw 0x34 0xF4')]
    hashed = [i for i, c in enumerate(replay.calls) if c.endswith('raw 0x34 0x04 0xe0 0x04')]
    assert polled and hashed and max(polled) < max(hashed), \
        'the boot state is polled after the power cycle and the firmware is checked on the booted FPGA'
    assert args.configure, 'a hard boot applies the register configuration at start-up'


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


def check_a_carrier_whose_processor_loads_its_image_is_not_firmware_checked():
    """Synthetic: slot 2's controller reports an RFMC carrier's part number. Its image is a linux.tar.gz its processor loads; no .mcs is looked for and nothing is loaded."""
    rfmc_record = (0, ' Board Mfg             : SLAC\n Board Product         : RFMC\n'
                      ' Board Part Number     : PC_379_396_55_C01\n')
    replay = Replay(overrides={'fru print 0': rfmc_record})
    args, done, replay = launched(SLOT_2, replay, archive='rogue_MicrowaveMuxRfmcCarrier_v2.0.0.zip')
    assert args.firmware and args.firmware.endswith('rogue_MicrowaveMuxRfmcCarrier_v2.0.0.zip'), args.firmware
    assert args.ip_ps == '10.0.1.202', f"the processor system at +200: {args.ip_ps}"
    assert 'programmed' not in done.decided and wrote(replay) == [] and not args.configure, \
        f"an image the processor loads was checked or loaded: {done.decided}, {wrote(replay)}"


# --------------------------------------------------------------------------
# a slot whose controller does not answer
# --------------------------------------------------------------------------


def check_a_board_with_no_controller_is_refused_without_addresses_and_tolerated_with_them():
    refused(SLOT_4, naming=('crate id', '--addr'))
    args, done, replay = launched(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204'])
    assert args.ip_addr == '10.0.1.104' and args.ip_ps == '10.0.1.204', 'the given addresses were changed'
    assert args.pcie_rssi_lane == 2, 'the lane still follows from the slot'
    inv = done.inventory
    assert inv['crate_id'] is None and 'crate_id' in inv['errors'], inv
    assert inv['carrier'] is None and inv['rtm_present'] is None and 'rtm' in inv['errors'], \
        'the IPMC did not answer, so the carrier and the rear module are unknown, not absent'
    assert inv['amc0'] is False, 'the inventory tool said no board in the bay: absent'
    assert wrote(replay) == [], f"written through a controller that does not answer: {wrote(replay)}"
    refused(SLOT_4 + ['-a', '10.0.1.104', '-H'], naming=('hard-boot',))

# --------------------------------------------------------------------------
# no slot, and the package
# --------------------------------------------------------------------------


def check_without_a_slot_nothing_is_read_and_nothing_reaches_the_crate():
    def no_tool(argv):
        raise AssertionError(f"a tool was run: {argv}")
    args, done, _ = launched(['-a', '10.0.1.102', '--transport', 'pcie'], no_tool)
    assert done.inventory == {} and args.pcie_rssi_lane == 0, (done.inventory, args.pcie_rssi_lane)
    assert args.firmware and args.firmware.endswith(ARCHIVE), 'the package is still found beside the image'
    refused(['-a', '10.0.1.102', '--transport', 'pcie', '-H'], no_tool, naming=('--shelfmanager', '--slot'))
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
        args, _ = launch(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204', '--platform', 'umux-rfmc'],
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
