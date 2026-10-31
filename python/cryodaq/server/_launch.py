#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Launch
#-----------------------------------------------------------------------------
# File       : _launch.py
# Created    : 2026-10-30
#-----------------------------------------------------------------------------
# Description:
#    What a readout server works out about its board before it builds a tree,
#    when it is told which crate slot to serve rather than which address: the
#    FPGA's address, the PCIe lane, which bays are populated, the register
#    defaults the population calls for, the firmware package on disk and whether
#    the FPGA is running it, and -- if asked -- a reboot. The facts come from the
#    platform's bring-up declaration and from the crate's shelf manager
#    (`cryodaq.platform._atca`); this module only decides, and every decision
#    is a default an explicit option overrides.
#
#    What was read is kept as one inventory record and published on the status
#    device beside the startup arguments, so a client can see what the launcher
#    saw. A board the shelf manager cannot reach is tolerated when the addresses
#    are given on the command line and refused, naming the call that failed,
#    when they are not.
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
import importlib
import logging
import pathlib
import re
import sys
import time
from typing import Any, Dict

from cryodaq import platform
from cryodaq._errors import CryodaqError
from cryodaq.platform import _atca

__all__ = ['launch', 'LaunchError', 'Launched', 'DEFAULT_FIRMWARE_DIR', 'BOOT_RETRIES',
           'BOOT_RETRY_DELAY_S']

_log = logging.getLogger(__name__)

# Where the container keeps the firmware release: the Python package archive,
# the FPGA image and the register defaults.
DEFAULT_FIRMWARE_DIR = '/tmp/fw'
DEFAULTS_SUBDIR = 'smurf_cfg/defaults'
# How long to wait for the FPGA after a reboot: ten tries, ten seconds apart,
# as the deployment has always waited.
BOOT_RETRIES = 10
BOOT_RETRY_DELAY_S = 10
# A short git hash as the firmware file names carry it.
SHORT_HASH_RE = re.compile(r'([0-9a-f]{7})[0-9a-f]*(?=\.(mcs|linux\.tar)\b)')


class LaunchError(CryodaqError):
    """The launcher could not decide how to start: what it needed and what it would take instead."""


class Launched:
    """What the launcher decided, kept beside the arguments it filled in.

    Parameters
    ----------
    inventory : dict
        What the shelf manager said about the slot; empty when none was asked.
    decided : dict
        Each argument the launcher set, by destination name, with its value.
    """

    def __init__(self):
        self.inventory: Dict[str, Any] = {}
        self.decided: Dict[str, Any] = {}

    def set_default(self, args: argparse.Namespace, name: str, value: Any, given: Any = None) -> None:
        """Fill ``name`` in unless the command line already gave it (``given`` is its unset value)."""
        if getattr(args, name) == given:
            setattr(args, name, value)
            self.decided[name] = value
        else:
            _log.info("--%s given as %r; the launcher's %r is not used",
                      name.replace('_', '-'), getattr(args, name), value)


def launch(args: argparse.Namespace, run: _atca.Runner = _atca.run_tool,
           sleep=time.sleep) -> Launched:
    """Fill the arguments a slot implies in, reading the crate where it has to.

    Called between parsing and transport selection by every entry point. With
    no ``--shelfmanager`` this does nothing but find the firmware package in
    ``--firmware-dir`` when none was named; with one, it works out what the
    bash launcher used to.

    Parameters
    ----------
    args : argparse.Namespace
        As parsed from `add_arguments`; modified in place.
    run : callable, optional
        How the shelf-manager tools are run; a check hands in a replay.
    sleep : callable, optional
        How a boot wait sleeps; a check hands in a no-op.

    Raises
    ------
    LaunchError
        If a slot was named and a fact it needs could not be read, with no
        option on the command line to stand in for it.
    """
    done = Launched()
    firmware_dir = pathlib.Path(args.firmware_dir)

    if args.shelfmanager is None and args.slot is None:
        asked = [flag for flag, on in (('--hard-boot', args.hard_boot), ('--program', args.program)) if on]
        if asked:
            raise LaunchError(f"{' and '.join(asked)} reach the FPGA through the crate's shelf manager; "
                              f"pass --shelfmanager and --slot")
        # Addressed directly, or emulated: nothing to read from a crate. The
        # firmware package is still looked for where the image keeps it, as the
        # launcher always did; a package on the import path needs none.
        _find_package(args, done, firmware_dir, image=None)
        if args.pcie_rssi_lane is None:
            args.pcie_rssi_lane = 0
        return done
    if args.shelfmanager is None or args.slot is None:
        raise LaunchError("--shelfmanager and --slot go together: the crate names the board "
                          "and the slot says which")
    if args.slot not in _atca.SLOTS:
        raise LaunchError(f"--slot {args.slot}: a carrier sits in slots "
                          f"{_atca.SLOTS.start} to {_atca.SLOTS.stop - 1}")

    shm = _atca.ShelfManager(args.shelfmanager, args.slot, run=run, sleep=sleep)
    # The lane follows from the slot on every PCIe deployment.
    done.set_default(args, 'pcie_rssi_lane', args.slot - 2, given=None)

    # What the shelf manager says. One record, every read attempted; a read the
    # controller does not answer is a None with the error beside it, so a bench
    # board without a controller leaves a complete, honest record.
    done.inventory = shm.inventory()
    errors = done.inventory['errors']
    reachable = done.inventory.get('crate_id') is not None

    # The addresses. From the crate id when it answered; otherwise the command
    # line has to carry them, and the refusal names the read that failed.
    if reachable:
        crate_id = done.inventory['crate_id']
        done.set_default(args, 'ip_addr', _atca.fpga_address(crate_id, args.slot), given='')
        _log.info("crate id 0x%04x, slot %d: FPGA at %s, PCIe RSSI lane %d", crate_id, args.slot,
                  args.ip_addr, args.pcie_rssi_lane)
    elif not args.ip_addr:
        raise LaunchError(f"the FPGA's address could not be derived: {errors.get('crate_id')}; "
                          f"pass --addr (and --ps-addr for a board with a processor system), "
                          f"or fix the crate")
    else:
        _log.warning("the shelf manager did not answer for slot %d (%s); taking the addresses "
                     "as given and skipping the inventory", args.slot, errors.get('crate_id'))

    # Which platform this is, by what the board says it is. Identification
    # proper still reads the build stamp once there is a tree; this only picks
    # the bring-up to run before one exists.
    carrier = done.inventory.get('carrier')
    bring_up, pmap = _bring_up_for(carrier, args.platform_name)
    if pmap is not None and reachable:
        processor_offset = getattr(bring_up, 'processor_offset', None)
        if processor_offset is not None:
            done.set_default(args, 'ip_ps', _atca.fpga_address(crate_id, args.slot, processor_offset),
                             given='')

    if bring_up is not None and reachable and not args.no_hw_detect:
        _population(args, done, bring_up, firmware_dir)

    image = getattr(bring_up, 'firmware_image', None) if bring_up is not None else None
    _find_package(args, done, firmware_dir, image)

    if reachable and args.check_firmware and bring_up is not None:
        _check_firmware(args, done, shm, bring_up, firmware_dir)

    if args.hard_boot:
        if not reachable:
            raise LaunchError("--hard-boot needs the shelf manager, which did not answer: "
                              f"{errors.get('crate_id')}")
        if not done.decided.get('programmed'):
            _reboot(shm, args.ip_addr)
        done.set_default(args, 'configure', True, given=False)

    _log.info("launcher decided: %s", _spelled(done.decided) or 'nothing beyond what was given')
    return done


def _spelled(decided):
    """The launcher's decisions as the command-line options they stand for."""
    spelling = {'ip_addr': '--addr {}', 'ip_ps': '--ps-addr {}', 'pcie_rssi_lane': '--pcie-rssi-lane {}',
                'disable_bay0': '--disable-bay0', 'disable_bay1': '--disable-bay1',
                'enable_pwri2c': '--enable-em22xx', 'firmware': '--firmware {}', 'configure': '--configure'}
    out = []
    for name, value in decided.items():
        if name in spelling:
            out.append(spelling[name].format(value))
        elif name == 'layers':
            out += [f"--layer {layer}" for layer in value]
        elif name == 'programmed':
            out.append(f"(loaded {value})")
        elif name == 'import_path':
            out.append(f"({len(value)} checkout directories on the import path)")
    return ' '.join(out)


# --------------------------------------------------------------------------
# the decisions
# --------------------------------------------------------------------------

def _bring_up_for(carrier, platform_name):
    """The platform whose carrier this is, from its FRU part number, and its bring-up data.

    An explicit ``--platform`` wins, and is checked against what the board says
    when the board said anything. With no record and no name there is nothing to
    decide on: ``(None, None)``, and the launcher takes the addresses it is given.
    Identification proper still reads the build stamp once there is a tree.
    """
    part = (carrier or {}).get('Board Part Number', '')
    matches = [m for m in platform.MAPS
               if any(part.startswith(prefix) for prefix in _module(m).BRING_UP.carrier_part_numbers)]
    if platform_name is not None:
        pmap = platform.by_name(platform_name)
        if matches and pmap not in matches:
            raise LaunchError(f"--platform {platform_name} but the carrier's part number {part!r} "
                              f"is a {matches[0].name} board")
        return _module(pmap).BRING_UP, pmap
    if len(matches) > 1:
        raise LaunchError(f"carrier part number {part!r} is claimed by "
                          f"{', '.join(m.name for m in matches)}; pass --platform")
    if not matches:
        if part:
            _log.warning("carrier part number %r is not one any platform declares; taking the "
                         "addresses as given", part)
        return None, None
    _log.info("carrier %s (%s): platform %s", part, (carrier or {}).get('Board Product Name', '?'), matches[0].name)
    return _module(matches[0]).BRING_UP, matches[0]


def _module(pmap):
    """The platform module a map was built from."""
    return importlib.import_module(pmap.module)


def _population(args, done, bring_up, firmware_dir):
    """What the bays hold, and what follows: which bays to leave out, which defaults layer, the power monitor.

    One record per bay from the inventory. An empty bay is left out of the tree;
    a populated one contributes its board type to the defaults layer's name,
    ``defaults_<version>_<bay 0>_<bay 1>.yml``, with ``none`` for an empty bay.
    Boards of two versions in one carrier have no single defaults file and are
    refused. The user's own layers stay after the population's, so they override
    it. Nothing here is decided when there is no bay to read.
    """
    inventory = done.inventory
    if bring_up.amc_part_number is None:
        if bring_up.defaults_layer:
            _prepend_layer(args, done, firmware_dir / DEFAULTS_SUBDIR / bring_up.defaults_layer)
        return
    words, versions = [], []
    for index, bay in enumerate(_atca.AMC_BAYS):
        record = inventory.get(f'amc{bay}')
        if not record:
            done.set_default(args, f'disable_bay{index}', True, given=False)
            words.append('none')
            _log.info("bay %d: empty; left out of the tree", index)
            continue
        part = record.get('Board Part Number', '')
        m = re.match(re.escape(bring_up.amc_part_number) + r'_C(\d{2})_(A\d{2})$', part)
        if not m:
            raise LaunchError(f"AMC bay {bay}: part number {part!r} is not a "
                              f"{bring_up.amc_part_number}_C<version>_<type> board; pass -E and the "
                              f"--disable-bay*/--layer options yourself")
        version, kind = m.groups()
        if kind not in bring_up.amc_types:
            raise LaunchError(f"AMC bay {bay}: board type {kind!r} is not one of "
                              f"{sorted(bring_up.amc_types)}")
        words.append(bring_up.amc_types[kind])
        versions.append(version)
        _log.info("bay %d: %s, version C%s, type %s (%s)", index, part, version, kind, bring_up.amc_types[kind])
    if len(set(versions)) > 1:
        raise LaunchError(f"the AMC boards are of different versions ({', '.join(versions)}); "
                          f"no one defaults file serves both. Pass -E and --layer yourself")
    if versions:
        _prepend_layer(args, done, firmware_dir / DEFAULTS_SUBDIR / f"defaults_c{versions[0]}_{'_'.join(words)}.yml")

    carrier = inventory.get('carrier') or {}
    if bring_up.power_i2c_from is not None:
        prefix, since = bring_up.power_i2c_from
        m = re.match(re.escape(prefix) + r'_C(\d{2})', carrier.get('Board Part Number', ''))
        if m and int(m.group(1)) >= since:
            done.set_default(args, 'enable_pwri2c', True, given=False)
            _log.info("carrier version C%s is C%02d or later: power monitor enabled", m.group(1), since)


def _prepend_layer(args, done, path):
    """The population's defaults go first, so a layer the user passed overrides them."""
    if not path.is_file():
        raise LaunchError(f"the defaults file this population calls for is not there: {path}; "
                          f"pass -E and --layer yourself")
    args.layers.insert(0, str(path))
    done.decided['layers'] = [str(path)]
    _log.info("register defaults for this population: %s", path.name)


def _find_package(args, done, firmware_dir, image):
    """The firmware's Python package: the one release archive in the firmware directory.

    ``image`` narrows the match to the platform's own archive when the directory
    holds several platforms' releases. None or several matches is a refusal
    naming the directory: guessing between two releases is how a server comes
    up on the wrong register map.
    """
    if args.firmware:
        return                                   # as given
    pattern = f"*{image}*.zip" if image else "*.zip"
    found = sorted(firmware_dir.glob(pattern))
    if len(found) > 1:
        raise LaunchError(f"{len(found)} firmware packages match {pattern} in {firmware_dir}; pass --firmware")
    if not found:
        # No archive: a deployment running from a checkout keeps it under the
        # firmware directory, as `<name>/firmware/python` plus each submodule's
        # `python`. Those go on the import path, as the launcher has always put
        # them; with none of those either, the package has to be importable.
        checkout = _checkout_python_dirs(firmware_dir)
        if checkout:
            for directory in checkout:
                if directory not in sys.path:
                    sys.path.insert(0, directory)
            done.decided['import_path'] = checkout
            _log.info("no firmware package matching %s in %s; on the import path from the checkout: %s",
                      pattern, firmware_dir, ', '.join(checkout))
        else:
            _log.info("no firmware package matching %s in %s; the package must be importable", pattern, firmware_dir)
        return
    done.set_default(args, 'firmware', str(found[0]), given=None)


def _checkout_python_dirs(firmware_dir):
    """A firmware checkout's Python directories under the firmware directory, innermost first.

    ``<name>/firmware/python``, ``<name>/firmware/submodules/<sub>/python`` and
    ``<name>/firmware/submodules/<sub>/firmware/python``: the layout of the
    firmware repositories and their submodules.
    """
    found = []
    for name in sorted(firmware_dir.iterdir()) if firmware_dir.is_dir() else []:
        fw = name / 'firmware'
        if not fw.is_dir():
            continue
        for candidate in [fw / 'python'] + sorted(fw.glob('submodules/*/python')) + sorted(fw.glob('submodules/*/firmware/python')):
            if candidate.is_dir():
                found.append(str(candidate))
    return found


def _check_firmware(args, done, shm, bring_up, firmware_dir):
    """Whether the FPGA runs the image on disk, and loading it when it does not.

    The image's file name carries its short git hash; the carrier's controller
    reports the hash of what the FPGA is running. Equal means nothing to do;
    different, or ``--program``, means the image is loaded, which reboots the
    FPGA -- so a hard boot after this is not needed and is skipped.
    """
    images = sorted(firmware_dir.glob(f"*{bring_up.firmware_image}*.mcs*"))
    if len(images) != 1:
        what = 'no FPGA image' if not images else f"{len(images)} FPGA images"
        raise LaunchError(f"{what} matching *{bring_up.firmware_image}*.mcs* in {firmware_dir}; "
                          f"pass -D to skip the firmware check")
    image = images[0]
    m = SHORT_HASH_RE.search(image.name)
    if not m:
        raise LaunchError(f"{image.name} carries no git hash in its name; pass -D to skip the firmware check")
    on_disk = m.group(1)
    try:
        running = shm.firmware_git_hash()
    except _atca.ShelfManagerError as e:
        raise LaunchError(f"{e}; pass -D to skip the firmware check") from e
    done.inventory['firmware_image_on_disk'] = image.name
    if running == on_disk and not args.program:
        _log.info("FPGA firmware %s, image on disk %s (%s): they match, nothing to load", running, on_disk, image.name)
        return
    _log.warning("FPGA firmware %s, image on disk %s%s: loading %s", running, on_disk,
                 ' (--program)' if args.program else '', image.name)
    try:
        shm.program(image)
    except _atca.ShelfManagerError as e:
        raise LaunchError(str(e)) from e
    done.decided['programmed'] = image.name


def _reboot(shm, ip_addr):
    """A hard boot: the shelf manager power-cycles the FPGA and waits for it to answer at its address."""
    _log.info("rebooting the FPGA in slot %d through %s", shm.slot, shm.host)
    try:
        shm.reboot(address=ip_addr or None, retry_max=BOOT_RETRIES, retry_delay=BOOT_RETRY_DELAY_S)
    except _atca.ShelfManagerError as e:
        raise LaunchError(str(e)) from e
    _log.info("FPGA rebooted and answering at %s", ip_addr)
