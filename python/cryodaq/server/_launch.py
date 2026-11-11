#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Launch
#-----------------------------------------------------------------------------
# File       : _launch.py
# Created    : 2026-10-30
#-----------------------------------------------------------------------------
# Description:
#    What a readout server works out about its board before it builds a tree.
#    Two ways in: told an address, it only finds the firmware package on disk;
#    told a crate slot, it reads the carrier through the crate's shelf manager
#    (`cryodaq.platform._atca`) and fills in what the bash launcher used to --
#    the FPGA's address, the PCIe lane, what the platform makes of the slot's
#    records, the firmware package, whether the FPGA is running the image on
#    disk, and, if asked, a reboot. The firmware directory (`--firmware-dir`)
#    is the one place a deployment keeps its release: the package archive or a
#    checkout and the FPGA image are found there, and the register
#    defaults a population calls for in its defaults directory
#    (`--defaults-dir`, under it by default).
#    What a board's records mean is the
#    platform's to say (its bring-up declaration); this module only applies the
#    answers, and every value it fills in is a default an explicit option
#    overrides.
#
#    What was read is kept as one inventory record and published on the status
#    device beside the startup arguments, so a client can see what the launcher
#    saw, and logged -- boards and revisions, package and release -- so the
#    log says what a server found. A
#    board the shelf manager cannot reach is tolerated when the addresses are
#    given on the command line and refused, naming the call that failed, when
#    they are not.
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
import datetime
import logging
import pathlib
import re
import zipfile
from typing import Any, Dict, Mapping

from cryodaq import platform
from cryodaq._errors import ConnectError, CryodaqError
from cryodaq.platform import _atca
from cryodaq.server._firmware import DEFAULTS_SUBDIR, checkout_python_dirs, resolve_layers

__all__ = ['launch', 'LaunchError', 'Launched', 'DEFAULT_FIRMWARE_DIR']

_log = logging.getLogger(__name__)

# Where the container keeps the firmware release: the Python package archive,
# the FPGA image and the register defaults.
DEFAULT_FIRMWARE_DIR = '/tmp/fw'
# A short git hash as the firmware file names carry it.
SHORT_HASH_RE = re.compile(r'([0-9a-f]{7})[0-9a-f]*(?=\.(mcs|linux\.tar)\b)')
# The two fields of a release archive's setup.py that name the release.
_SETUP_FIELD_RE = re.compile(r'''^\s*(name|version)\s*=\s*['"]([^'"]*)['"]''', re.MULTILINE)
# The date formats the FRU records carry, for the month a board was made; the
# last is the tool's form for a date the record leaves unspecified.
_FRU_DATE_FORMATS = ('%Y-%m-%d %H:%M:%S', '%a %b %d %H:%M:%S %Y %Z', '%a %b %d %H:%M:%S %Y')


class LaunchError(CryodaqError):
    """The launcher could not decide how to start: what it needed and what it would take instead."""


class Launched:
    """What the launcher decided, kept beside the arguments it filled in.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser the arguments came from: an option still at its default is
        one the launcher may fill in.

    Attributes
    ----------
    inventory : dict
        What the shelf manager said about the slot; empty when none was asked.
    decided : dict
        Each argument the launcher set, by destination name, with its value.
    """

    def __init__(self, parser: argparse.ArgumentParser):
        self._parser = parser
        self.inventory: Dict[str, Any] = {}
        self.decided: Dict[str, Any] = {}

    def set_default(self, args: argparse.Namespace, name: str, value: Any) -> None:
        """Fill ``name`` in unless the command line already gave it."""
        if getattr(args, name) == self._parser.get_default(name):
            setattr(args, name, value)
            self.decided[name] = value
        else:
            _log.info("--%s given as %r; the launcher's %r is not used",
                      name.replace('_', '-'), getattr(args, name), value)


def launch(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Launched:
    """Fill in what the command line leaves to the launcher.

    Called between parsing and transport selection by every entry point.

    Parameters
    ----------
    args : argparse.Namespace
        As parsed from ``parser``; modified in place.
    parser : argparse.ArgumentParser
        The parser, for the options' defaults.

    Raises
    ------
    LaunchError
        If a fact the start needs could not be read and no option on the
        command line stands in for it.
    """
    done = Launched(parser)
    # Absolute, so a defaults file found here is never taken for a package member.
    args.firmware_dir = str(pathlib.Path(args.firmware_dir).resolve())
    if args.defaults_dir is None:
        args.defaults_dir = str(pathlib.Path(args.firmware_dir) / DEFAULTS_SUBDIR)
    else:
        # Absolute too: the file names resolved against it are absolute, so the
        # composition takes them as given rather than resolving them again.
        args.defaults_dir = str(pathlib.Path(args.defaults_dir).resolve())
    if args.shelfmanager is None and args.slot is None:
        _launch_addressed(args, done)
    else:
        _launch_from_slot(args, done)
    _log.info("launcher decided: %s", _spelled(done.decided) or 'nothing beyond what was given')
    return done


def _launch_addressed(args, done):
    """Told an address, or emulated: nothing is read from a crate."""
    if args.hard_boot:
        raise LaunchError("--hard-boot reaches the FPGA through the crate's shelf manager; "
                          "pass --shelfmanager and --slot")
    if args.pcie_rssi_lane is None:
        args.pcie_rssi_lane = 0
    # The firmware package is still looked for in the firmware directory; a
    # package already on the import path needs none.
    _find_package(args, done, pathlib.Path(args.firmware_dir), image=_named_image(args))


def _launch_from_slot(args, done):
    """Told a crate slot: read the carrier through the shelf manager and act on what it says."""
    if args.shelfmanager is None or args.slot is None:
        raise LaunchError("--shelfmanager and --slot go together: the crate names the board "
                          "and the slot says which")
    if args.slot not in _atca.SLOTS:
        raise LaunchError(f"--slot {args.slot}: a carrier sits in slots "
                          f"{_atca.SLOTS.start} to {_atca.SLOTS.stop - 1}")
    shm = _atca.ShelfManager(args.shelfmanager, args.slot)
    firmware_dir = pathlib.Path(args.firmware_dir)
    # The PCIe lane follows from the slot on every crate.
    done.set_default(args, 'pcie_rssi_lane', args.slot - 2)

    # 1. What the shelf manager says about the slot: one record, every read
    #    attempted. A read the controller does not answer is None with the
    #    error beside it.
    done.inventory = shm.inventory()
    errors = done.inventory['errors']
    crate_id = done.inventory['crate_id']

    if crate_id is None:
        # A board with no management controller (a bench board). The command
        # line has to carry the addresses, nothing can be rebooted or checked,
        # and the package is found as for an addressed start.
        if not args.ip_addr:
            raise LaunchError(f"the FPGA's address could not be derived: {errors.get('crate_id')}; "
                              f"pass --addr (and --ps-addr for a board with a processor system), "
                              f"or fix the crate")
        if args.hard_boot:
            raise LaunchError("--hard-boot needs the shelf manager, which did not answer: "
                              f"{errors.get('crate_id')}")
        _log.warning("the shelf manager did not answer for slot %d (%s); taking the addresses "
                     "as given and skipping the inventory", args.slot, errors.get('crate_id'))
        # Every read was attempted on its own: a board whose record came back is logged as found.
        _log_population(args.slot, done.inventory)
        _find_package(args, done, firmware_dir, image=_named_image(args))
        return

    # 2. The FPGA's address follows from the crate id.
    done.set_default(args, 'ip_addr', _atca.fpga_address(crate_id, args.slot))
    _log.info("crate id 0x%04x, slot %d: FPGA at %s, PCIe RSSI lane %d", crate_id, args.slot,
              args.ip_addr, args.pcie_rssi_lane)
    _log_population(args.slot, done.inventory)

    # 3. Which platform's board this is, by its carrier's FRU part number. That
    #    chooses the image, the package, and how to read the bays; identifying
    #    the platform proper waits for the tree and its build stamp. A record
    #    that could not be read is not an unknown carrier: it is refused unless
    #    the command line gives up both detection and the firmware check.
    carrier = done.inventory.get('carrier')
    if carrier is None and not (args.no_hw_detect and not args.check_firmware):
        raise LaunchError(f"the carrier's record could not be read: {errors.get('carrier')}; "
                          f"pass -E and -D and the options yourself")
    bring_up = _bring_up_for(carrier or {})
    if bring_up is not None:
        if bring_up.processor_offset is not None:
            done.set_default(args, 'ip_ps', _atca.fpga_address(crate_id, args.slot, bring_up.processor_offset))
        # 4. What else the platform reads off the slot's records (for the AMC
        #    carrier: its bays, their defaults file, the power monitor).
        if not args.no_hw_detect:
            _apply_options(args, done, bring_up)

    # 5. The firmware's Python package, by the platform's image name -- the
    #    carrier's, or the named platform's when no record identified one.
    #    Before anything touches the board, so a directory with two releases
    #    in it is refused without a reboot.
    _find_package(args, done, firmware_dir,
                  image=bring_up.firmware_image if bring_up else _named_image(args))

    # 6. A hard boot, before the firmware check so the check sees the booted FPGA.
    if args.hard_boot:
        _reboot(shm, args.ip_addr)
        done.set_default(args, 'configure', True)

    # 7. Is the FPGA running the image on disk? If not, load it (which reboots).
    #    Only where the shelf manager loads the image; a board whose processor
    #    system loads its own is left alone.
    if bring_up is not None and args.check_firmware:
        if bring_up.shelf_manager_loads_image:
            _check_firmware(args, done, shm, bring_up, firmware_dir)
        else:
            _log.info("the %s image is loaded by the board's processor system, not through the "
                      "shelf manager; not checked here", bring_up.firmware_image)


def _spelled(decided):
    """The launcher's decisions as the command-line options they stand for."""
    spelling = {'ip_addr': '--addr {}', 'ip_ps': '--ps-addr {}', 'pcie_rssi_lane': '--pcie-rssi-lane {}',
                'disable_bay0': '--disable-bay0', 'disable_bay1': '--disable-bay1',
                'enable_pwri2c': '--enable-em22xx', 'configure': '--configure'}
    out = []
    for name, value in decided.items():
        if name in spelling:
            out.append(spelling[name].format(value))
        elif name == 'layers':
            out += [f"--layer {layer}" for layer in value]
        elif name == 'firmware':
            out.append(f"(package {value})")
        elif name == 'programmed':
            out.append(f"(loaded {value})")
    return ' '.join(out)


# --------------------------------------------------------------------------
# the steps
# --------------------------------------------------------------------------

def _bring_up_for(carrier):
    """The bring-up declaration of the platform whose carrier this is, or None with a warning."""
    part = carrier.get('Board Part Number', '')
    try:
        bring_up = platform.bring_up_for(part)
    except ConnectError as e:
        raise LaunchError(f"{e}; pass -E and -D and the options yourself") from e
    if bring_up is None:
        _log.warning("carrier part number %r is not one any platform declares; the bays and the "
                     "firmware are not checked, the addresses are taken as given", part)
    return bring_up


def _named_image(args):
    """The image name of the platform ``--platform`` names, when no carrier record can say.

    A deployment's firmware directory holds every platform's release, so with
    no record to choose by the name on the command line does; without it the
    search is over every archive. The build stamp still identifies the platform.
    """
    if args.platform_name is None:
        return None
    return platform.bring_up_by_name(args.platform_name).firmware_image


def _apply_options(args, done, bring_up):
    """Fill in the options the platform derives from the slot's records.

    The platform decides (`BringUp.options`); what is the command line's is
    done here: a defaults file it names is resolved in the deployment's
    configuration directory -- by the resolver the composition uses for every
    layer, here so a missing file is refused before anything touches the
    board -- and goes before any layer the user gave, so the user's overrides
    it; every other option is a default the command line beats.
    """
    try:
        options = bring_up.options(done.inventory)
    except ValueError as e:
        raise LaunchError(f"{e}; pass -E and the options yourself") from e
    _log.info("platform options from the slot's records: %s", options or 'none')
    for name, value in options.items():
        if name == 'layers':
            try:
                path = resolve_layers([value], None, None, defaults_dir=args.defaults_dir)[0]
            except FileNotFoundError as e:
                raise LaunchError(f"the defaults file the platform calls for is not there: {e}; "
                                  f"pass -E and --layer yourself") from e
            args.layers.insert(0, path)
            done.decided['layers'] = [path]
        else:
            done.set_default(args, name, value)


def _find_package(args, done, firmware_dir, image):
    """The firmware's Python package in the firmware directory: one release archive, or else one checkout.

    ``image`` narrows the archives to the platform's own when the directory
    holds several platforms' releases. Several candidates is a refusal naming
    them: guessing between two releases is how a server comes up on the wrong
    register map. None at all is allowed, for a package already importable.
    """
    pattern = f"*{image}*.zip" if image else "*.zip"
    archives = sorted(firmware_dir.glob(pattern))
    if len(archives) > 1:
        raise LaunchError(f"{len(archives)} firmware packages match {pattern} in {firmware_dir}: "
                          f"{', '.join(a.name for a in archives)}; keep one there")
    if archives:
        _log.info("firmware package: release archive %s, %s", archives[0], _archive_release(archives[0]))
        done.set_default(args, 'firmware', str(archives[0]))
        return
    checkouts = [d for d in ([firmware_dir] + sorted(firmware_dir.iterdir()) if firmware_dir.is_dir() else [])
                 if (d / 'firmware' / 'python').is_dir()]
    if len(checkouts) > 1:
        raise LaunchError(f"{len(checkouts)} firmware checkouts in {firmware_dir}: "
                          f"{', '.join(c.name for c in checkouts)}; keep one there")
    if checkouts:
        _log.info("firmware package: checkout %s %s, %d python directories", checkouts[0],
                  _git_describe(checkouts[0]), len(checkout_python_dirs(checkouts[0])))
        done.set_default(args, 'firmware', str(checkouts[0]))
        return
    _log.info("firmware package: none in %s (no %s, no checkout); the package must be importable",
              firmware_dir, pattern)


def _archive_release(archive: pathlib.Path) -> str:
    """``<name> <version>`` as a release archive's ``setup.py`` states them, read as text, not run; else why not."""
    try:
        with zipfile.ZipFile(archive) as zf:
            text = zf.read('setup.py').decode('utf-8', 'replace')
    except (KeyError, OSError, zipfile.BadZipFile):
        return "release unknown (no setup.py in the archive)"
    fields = dict(_SETUP_FIELD_RE.findall(text))
    if 'version' not in fields:
        return "release unknown (its setup.py names no version)"
    return f"{fields.get('name', archive.stem)} {fields['version']}"


def _git_describe(checkout: pathlib.Path) -> str:
    """``at <tag or hash>[-dirty] (<branch>)`` from the checkout's own git metadata, or why there is none.

    Only a directory with its own ``.git`` is asked, so a copy inside some
    other repository is never described as that repository. The checkout is
    named safe for this one call: in a container the server runs as another
    user than the checkout's owner, which git otherwise refuses. git runs
    through the same seam as the shelf-manager tools, so a check can answer
    for it.
    """
    if not (checkout / '.git').exists():
        return "version unknown (no git metadata)"

    def git(*words):
        status, text = _atca.run_tool(['git', '-c', f'safe.directory={checkout}', '-C', str(checkout), *words])
        if status != 0:
            raise _atca.ShelfManagerError('describing the checkout', ['git', *words], text)
        return text.strip()
    try:
        return f"at {git('describe', '--tags', '--dirty', '--always')} ({git('rev-parse', '--abbrev-ref', 'HEAD')})"
    except _atca.ShelfManagerError as e:
        return f"version unknown ({e})"


def _log_population(slot: int, inventory: Mapping[str, Any]) -> None:
    """One line per board the slot's records name: the carrier, each bay, the rear module.

    A board that could not be read is said to be unread, with the error; one the
    controller reported absent is said to be absent. The two are never confused.
    """
    _log.info("slot %d carrier: %s", slot, _board(inventory, 'carrier', 'carrier'))
    for index, bay in enumerate(_atca.AMC_BAYS):
        _log.info("slot %d bay %d: %s", slot, index, _board(inventory, f'amc{bay}', 'AMC'))
    _log.info("slot %d RTM: %s", slot, _board(inventory, 'rtm', 'RTM'))


def _board(inventory: Mapping[str, Any], key: str, kind: str) -> str:
    """A board's FRU record in one phrase: part, product, version, BOM, serial, month made."""
    record = inventory.get(key)
    if record is None:
        return f"not read ({inventory.get('errors', {}).get(key, 'no record')})"
    if not record:
        return f"no {kind}"
    phrase = [f"{kind} {record.get('Board Part Number', '?')}"]
    product = record.get('Board Product Name') or record.get('Board Product')
    if product:
        phrase[0] += f" {product}"
    for label, field in (('version', 'Product Version'), ('BOM', 'Internal BOM')):
        if record.get(field):
            phrase.append(f"{label} {record[field]}")
    serial = record.get('Board Serial No.') or record.get('Board Serial')
    if serial:
        phrase.append(f"serial {serial}")
    if record.get('Board Mfg Date'):
        phrase.append(f"made {_month(record['Board Mfg Date'])}")
    return ', '.join(phrase)


def _month(date: str) -> str:
    """``YYYY-MM`` from a FRU date in either form the records use; the text as given otherwise."""
    for fmt in _FRU_DATE_FORMATS:
        try:
            return datetime.datetime.strptime(date, fmt).strftime('%Y-%m')
        except ValueError:
            continue
    return date


def _check_firmware(args, done, shm, bring_up, firmware_dir):
    """Whether the FPGA runs the image on disk, and loading it when it does not.

    The image's file name carries its short git hash; the carrier's controller
    reports the hash of what the FPGA is running. Equal means nothing to do.
    Different means the image is loaded, which reboots the FPGA into it, so the
    register configuration is applied at start-up as after a hard boot.
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
    if running == on_disk:
        _log.info("FPGA firmware %s, image on disk %s (%s): they match, nothing to load", running, on_disk, image.name)
        return
    _log.warning("FPGA firmware %s, image on disk %s: loading %s", running, on_disk, image.name)
    try:
        shm.program(image)
    except _atca.ShelfManagerError as e:
        raise LaunchError(str(e)) from e
    done.decided['programmed'] = image.name
    done.set_default(args, 'configure', True)


def _reboot(shm, ip_addr):
    """A hard boot: the shelf manager power-cycles the FPGA and waits for it to answer at its address."""
    _log.info("rebooting the FPGA in slot %d through %s", shm.slot, shm.host)
    try:
        shm.reboot(address=ip_addr or None)
    except _atca.ShelfManagerError as e:
        raise LaunchError(str(e)) from e
    _log.info("FPGA rebooted and answering at %s", ip_addr)
