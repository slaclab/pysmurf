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
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / 'python'))
sys.path.insert(0, str(HERE / 'fixtures' / 'ipmi'))

from replay import Replay                                       # noqa: E402

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


def launched(argv, run=None, archive=ARCHIVE):
    """Parse ``argv`` over a fresh firmware directory and run the launcher; the replay's calls come back too."""
    run = Replay() if run is None else run
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, archive)
        args = _cli.add_arguments(argparse.ArgumentParser()).parse_args(list(argv) + ['--firmware-dir', str(fw)])
        done = _cli.launch(args, run=run, sleep=lambda s: None)
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
    assert any(c.endswith('raw 0x34 0xF4') for c in replay.calls), 'the boot state was not polled'
    assert args.configure, 'a hard boot applies the register configuration at start-up'


def check_a_firmware_mismatch_loads_the_image_and_a_match_does_not():
    """Synthetic: the FPGA runs the image on disk. A different hash from the controller means a load."""
    _, done, replay = launched(SLOT_2, Replay(overrides={'raw 0x34 0x04 0xe0 0x04': (0, ' 00 11 22 33\n')}))
    assert done.decided.get('programmed') == IMAGE, done.decided
    assert [c for c in wrote(replay) if 'ProgramFPGA.bash' in c], replay.calls
    _, done, replay = launched(SLOT_2 + ['-D'], Replay(overrides={'raw 0x34 0x04 0xe0 0x04': (0, ' 00 11 22 33\n')}))
    assert 'programmed' not in done.decided and wrote(replay) == [], '-D must skip the firmware check'

# --------------------------------------------------------------------------
# a slot whose controller does not answer
# --------------------------------------------------------------------------


def check_a_board_with_no_controller_is_refused_without_addresses_and_tolerated_with_them():
    refused(SLOT_4, naming=('crate id', '--addr'))
    args, done, replay = launched(SLOT_4 + ['-a', '10.0.1.104', '--ps-addr', '10.0.1.204', '-z', 'x.zip'])
    assert args.ip_addr == '10.0.1.104' and args.ip_ps == '10.0.1.204', 'the given addresses were changed'
    assert args.pcie_rssi_lane == 2, 'the lane still follows from the slot'
    inv = done.inventory
    assert inv['crate_id'] is None and 'crate_id' in inv['errors'], inv
    assert inv['carrier'] is False and inv['rtm_present'] is None, \
        'the inventory tool said no board; the IPMC did not answer, so presence is unknown, not absent'
    assert wrote(replay) == [], f"written through a controller that does not answer: {wrote(replay)}"
    refused(SLOT_4 + ['-a', '10.0.1.104', '-z', 'x.zip', '-H'], naming=('hard-boot',))

# --------------------------------------------------------------------------
# no slot, and the package
# --------------------------------------------------------------------------


def check_without_a_slot_nothing_is_read_and_nothing_reaches_the_crate():
    def no_tool(argv):
        raise AssertionError(f"a tool was run: {argv}")
    args, done, _ = launched(['-a', '10.0.1.102', '--transport', 'pcie'], no_tool)
    assert done.inventory == {} and args.pcie_rssi_lane == 0, (done.inventory, args.pcie_rssi_lane)
    assert args.firmware and args.firmware.endswith(ARCHIVE), 'the package is still found beside the image'
    for flag in ('-H', '--program'):
        refused(['-a', '10.0.1.102', '--transport', 'pcie', flag], no_tool, naming=('--shelfmanager', '--slot'))
    refused(['-S', 'shm-smrf-sp01'], no_tool, naming=('--slot',))
    refused(['-N', '2'], no_tool, naming=('--shelfmanager',))
    refused(['-S', 'shm-smrf-sp01', '-N', '9'], no_tool, naming=('slot',))


def check_the_firmware_package_is_the_one_archive_or_else_the_checkout():
    with tempfile.TemporaryDirectory() as tmp:
        fw = firmware_dir(tmp, archive='')
        parse = lambda: _cli.add_arguments(argparse.ArgumentParser()).parse_args(SLOT_2 + ['--firmware-dir', str(fw)])  # noqa: E731
        done = _cli.launch(parse(), run=Replay(), sleep=lambda s: None)
        assert 'firmware' not in done.decided and 'import_path' not in done.decided, done.decided
        # A checkout beside the image: its firmware/python and each submodule's go on the path.
        for d in ('cryo-det/firmware/python', 'cryo-det/firmware/submodules/surf/python',
                  'cryo-det/firmware/submodules/x/firmware/python', 'cryo-det/firmware/submodules/y/notpython'):
            (fw / d).mkdir(parents=True)
        done = _cli.launch(parse(), run=Replay(), sleep=lambda s: None)
        put = [os.path.relpath(p, fw) for p in done.decided.get('import_path', [])]
        assert put == ['cryo-det/firmware/python', 'cryo-det/firmware/submodules/surf/python',
                       'cryo-det/firmware/submodules/x/firmware/python'], put
        assert all(p in sys.path for p in done.decided['import_path']), 'the directories are not on the import path'
        for p in done.decided['import_path']:
            sys.path.remove(p)
        (fw / ARCHIVE).touch()
        (fw / 'rogue_MicrowaveMuxBpEthGen2_v2.6.0.zip').touch()
        try:
            _cli.launch(parse(), run=Replay(), sleep=lambda s: None)
        except SystemExit as e:
            assert '2 firmware packages' in str(e), str(e)
        else:
            raise AssertionError('two archives were not refused')

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
