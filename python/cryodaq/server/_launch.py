#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Server Launch
#-----------------------------------------------------------------------------
# File       : _launch.py
# Created    : 2026-10-30
#-----------------------------------------------------------------------------
# Description:
#    What a readout server works out about its board before it builds a tree,
#    as one sequence that `launch` follows step for step:
#
#      1. The crate's facts. Given a slot (--shelfmanager, --slot), the carrier
#         is read through the crate's shelf manager into one inventory record,
#         a read the controller does not answer recorded as such. Without a
#         slot, or when the controller does not answer, nothing is known about
#         the board but what the command line says.
#      2. The platform. Named by the carrier's FRU part number when the record
#         was read, by --platform when it was not; the two disagreeing is a
#         refusal. With no record and no name the platform is unknown: the
#         package is still looked for, nothing is checked or rebooted, and an
#         image on disk that could have been checked is a refusal, not a skip.
#      3. The addresses. The FPGA's follows from the crate id and the slot, the
#         processor system's from the platform's offset, the PCIe lane from the
#         slot. Each is a default an explicit option overrides, and a board
#         whose crate said nothing must have them on the command line. The
#         ones the transport reaches the board by are required here, before
#         any step writes to the board.
#      4. What else the platform makes of the records (the AMC carrier's bays
#         and their defaults file; --no-hw-detect skips it).
#      5. The firmware's Python package in the firmware directory
#         (--firmware-dir, the one place a deployment keeps its release),
#         narrowed to the platform's when one is known. Before anything
#         touches the board, so two releases in one directory is a refusal
#         without a reboot.
#      6. The image loader: how this FPGA comes to run an image. The crate's
#         shelf manager programs an AMC carrier's flash (an .mcs file); a
#         processor system beside the FPGA (the RFMC carrier, the ZCU208)
#         boots it from the boot files on its SD card (a .linux.tar.gz),
#         reached over SSH. Which is the platform's declaration; the loader
#         is the platform object itself (`ShelfManager`, `ProcessorSystem`),
#         and exists only when what it needs -- a crate that answers, a
#         processor address -- does. Both answer the same verbs, so the two
#         steps after this one are the same whichever it is.
#      7. A hard boot, when asked (--hard-boot): the loader reboots the FPGA.
#      8. The firmware check (--no-check-firmware skips it): the short git
#         hash in the image file's name against the one the loader reads from
#         the FPGA. When they differ the loader programs the image, which
#         reboots the FPGA into it, and the register configuration is applied
#         at start-up.
#
#    What was read is kept as one inventory record and published on the status
#    device beside the startup arguments, so a client can see what the launcher
#    saw, and logged -- boards and revisions, package and release, what was
#    decided -- so the log says what a server found and did.
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
from typing import Any, Dict, List, Mapping, Optional, Tuple

from cryodaq import platform
from cryodaq._errors import ConnectError, CryodaqError
from cryodaq.platform import _atca, _soc, _umux
from cryodaq.server import _firmware
from cryodaq.server._firmware import DEFAULTS_SUBDIR, checkout_python_dirs, resolve_layers

__all__ = ['launch', 'LaunchError', 'Launched', 'DEFAULT_FIRMWARE_DIR']

_log = logging.getLogger(__name__)

# Where the container keeps the firmware release: the Python package archive,
# the FPGA image and the register defaults.
DEFAULT_FIRMWARE_DIR = '/tmp/fw'
# Every image a release ships, for the start that knows no platform to match
# one to.
IMAGE_SUFFIXES = (_atca.IMAGE_SUFFIX, _soc.IMAGE_SUFFIX)
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
    """Fill in what the command line leaves to the launcher, in the eight steps the module header lists.

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
    # An emulated server has no board: nothing to check, nothing to reboot.
    hardware = args.transport != 'emulation'
    # Absolute, so a defaults file found here is never taken for a package
    # member, and the file names resolved against the defaults directory are
    # absolute too, so the composition takes them as given.
    args.firmware_dir = str(pathlib.Path(args.firmware_dir).resolve())
    args.defaults_dir = str(pathlib.Path(args.firmware_dir) / DEFAULTS_SUBDIR if args.defaults_dir is None
                            else pathlib.Path(args.defaults_dir).resolve())
    firmware_dir = pathlib.Path(args.firmware_dir)

    # 1. The crate's facts.
    shm, crate_id = _read_crate(args, done)

    # 2. The platform.
    bring_up = _decide_platform(args, done, crate_id)

    # 3. The addresses.
    _decide_addresses(args, done, crate_id, bring_up)

    # 4. What else the platform makes of the records.
    if bring_up is not None and crate_id is not None and not args.no_hw_detect:
        _apply_options(args, done, bring_up)

    # 5. The firmware's Python package.
    _find_package(args, done, firmware_dir, bring_up.firmware_image if bring_up else None)

    # 6. The image loader.
    loader = _image_loader(args, shm, crate_id, bring_up) if hardware else None

    try:
        # 7. A hard boot, before the firmware check so the check sees the booted FPGA.
        if args.hard_boot:
            if loader is None:
                raise LaunchError(_no_way_to_reboot(args, hardware, bring_up))
            _log.info("rebooting the FPGA through %s", loader)
            loader.reboot()
            _log.info("FPGA rebooted")
            done.set_default(args, 'configure', True)

        # 8. The firmware check.
        if hardware and args.check_firmware:
            if loader is None:
                _not_checked(args, firmware_dir, bring_up)
            else:
                _check_firmware(args, done, loader, bring_up, firmware_dir)
    except (_atca.ShelfManagerError, _soc.ProcessorSystemError) as e:
        # A tool that failed on the board is why the server does not start.
        raise LaunchError(str(e)) from e

    _log.info("launcher decided: %s", _spelled(done.decided) or 'nothing beyond what was given')
    return done


# --------------------------------------------------------------------------
# the steps
# --------------------------------------------------------------------------

def _read_crate(args, done) -> Tuple[Optional[_atca.ShelfManager], Optional[int]]:
    """Step 1: ``(shelf manager, crate id)`` -- ``(None, None)`` with no slot, ``(shm, None)`` for a controller that does not answer.

    The inventory is one record with every read attempted: a read the
    controller does not answer is None with the error beside it, never
    confused with a board the controller reported absent.
    """
    if args.shelfmanager is None and args.slot is None:
        return None, None
    if args.shelfmanager is None or args.slot is None:
        raise LaunchError("--shelfmanager and --slot go together: the crate names the board "
                          "and the slot says which")
    if args.slot not in _atca.SLOTS:
        raise LaunchError(f"--slot {args.slot}: a carrier sits in slots "
                          f"{_atca.SLOTS.start} to {_atca.SLOTS.stop - 1}")
    shm = _atca.ShelfManager(args.shelfmanager, args.slot)
    done.inventory = shm.inventory()
    crate_id = done.inventory['crate_id']
    if crate_id is None:
        _log.warning("the shelf manager did not answer for slot %d (%s); the board is known only by what "
                     "the command line says", args.slot, done.inventory['errors'].get('crate_id'))
    else:
        _log.info("crate id 0x%04x, slot %d", crate_id, args.slot)
    # Every read was attempted on its own: a board whose record came back is logged as found.
    _log_population(args.slot, done.inventory)
    return shm, crate_id


def _decide_platform(args, done, crate_id):
    """Step 2: the platform's bring-up declaration, or None for a board nothing names.

    The carrier's FRU part number names it when the record was read
    (`platform.bring_up_for`); ``--platform`` names it when there is no
    record, and is checked against the record when there is one. The record
    is a read of its own: it names the board even when the crate id could
    not be read, which costs only what follows from the id. The platform
    proper is identified later, from the build stamp, once there is a tree.
    """
    named = None
    if args.platform_name is not None:
        try:
            named = platform.bring_up_by_name(args.platform_name)
        except ConnectError as e:
            raise LaunchError(str(e)) from e
    carrier = done.inventory.get('carrier')
    if carrier is None:
        # No record: no crate, a controller that did not answer at all, or
        # one that answered the crate id and not the record. The last is not
        # an unknown carrier, and is refused unless the command line gives up
        # both detection and the firmware check.
        if crate_id is not None and not (args.no_hw_detect and not args.check_firmware):
            raise LaunchError(f"the carrier's record could not be read: {done.inventory['errors'].get('carrier')}; "
                              f"pass -E and -D and the options yourself")
        if named is None:
            _log.info("no carrier record and no --platform: the platform is known only once there is a tree")
        return named
    part = (carrier or {}).get('Board Part Number', '')
    try:
        found = platform.bring_up_for(part)
    except ConnectError as e:
        raise LaunchError(f"{e}; pass -E and -D and the options yourself") from e
    if found is None:
        if named is None:
            _log.warning("carrier part number %r is not one any platform declares; the bays and the firmware "
                         "are not checked (pass --platform to say which platform it is)", part)
        else:
            _log.info("carrier part number %r is not one any platform declares; taking --platform %s",
                      part, args.platform_name)
        return named
    if named is not None and named is not found:
        raise LaunchError(f"the carrier in slot {args.slot} (part number {part}) is not a --platform "
                          f"{args.platform_name} board; drop the option or fix it")
    return found


# The addresses each transport reaches the board by (`_cli.transport_from`
# builds it from them). The PCIe transport's FPGA address is optional.
ADDRESSES_REQUIRED = {'eth': ('ip_addr',), 'tcp-bridge': ('ip_ps', 'ip_addr')}


def missing_addresses(args: argparse.Namespace) -> List[str]:
    """The address options the chosen transport needs and ``args`` does not carry, spelled as given.

    Parameters
    ----------
    args : argparse.Namespace
        As parsed from the server's command line, after step 3 filled in what
        the crate implies.

    Returns
    -------
    list of str
        ``'--addr'`` and/or ``'--ps-addr'``; empty when the transport has
        what it needs.
    """
    spelling = {'ip_addr': '--addr', 'ip_ps': '--ps-addr'}
    return [spelling[name] for name in ADDRESSES_REQUIRED.get(args.transport, ()) if not getattr(args, name)]


def _decide_addresses(args, done, crate_id, bring_up):
    """Step 3: the PCIe lane from the slot; the FPGA's and the processor system's addresses from the crate id.

    SLAC's convention: ``10.<crate id high byte>.<crate id low byte>.<100 + slot>``
    for the FPGA, the platform's offset in place of 100 for a processor system
    beside it. A board whose crate said nothing must have them on the command
    line. The addresses the transport reaches the board by are required here:
    the steps that write to the board come after, and a start that cannot
    reach it must not load or reboot anything first.
    """
    if args.slot is not None:
        # The PCIe lane follows from the slot on every crate.
        done.set_default(args, 'pcie_rssi_lane', args.slot - 2)
    elif args.pcie_rssi_lane is None:
        args.pcie_rssi_lane = 0
    if crate_id is not None:
        done.set_default(args, 'ip_addr', _atca.fpga_address(crate_id, args.slot))
        if bring_up is not None and bring_up.processor_offset is not None:
            done.set_default(args, 'ip_ps', _atca.fpga_address(crate_id, args.slot, bring_up.processor_offset))
        _log.info("FPGA at %s, PCIe RSSI lane %d%s", args.ip_addr, args.pcie_rssi_lane,
                  f", processor system at {args.ip_ps}" if args.ip_ps else '')
    elif args.slot is not None and not args.ip_addr:
        raise LaunchError(f"the FPGA's address could not be derived: {done.inventory['errors'].get('crate_id')}; "
                          f"pass --addr (and --ps-addr for a board with a processor system), or fix the crate")
    missing = missing_addresses(args)
    if missing:
        raise LaunchError(f"the {args.transport} transport reaches the board by {' and '.join(missing)}, which "
                          f"{'neither the crate nor the command line gives' if crate_id is not None else 'no crate was asked for'}; "
                          f"pass {'them' if len(missing) > 1 else 'it'}")


def _apply_options(args, done, bring_up):
    """Step 4: the options the platform derives from the slot's records.

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
    """Step 5: the firmware's Python package in the firmware directory -- one release archive, or else one checkout.

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


def _image_loader(args, shm, crate_id, bring_up):
    """Step 6: what loads this FPGA's image -- the crate's shelf manager, or the board's processor system.

    None when the platform is unknown or the loader is out of reach. A
    processor system that boots the FPGA is reached at its address, from the
    crate or the command line; without it the image can be neither checked
    nor loaded, which is a refusal unless the check was called off. The
    shelf manager is reached only through a crate that answered. Either
    object answers the same verbs, so the steps after this one ask no
    further which it is.
    """
    if bring_up is None:
        return None
    if bring_up.image_loader == _umux.PROCESSOR_SYSTEM:
        if args.ip_ps:
            def read_version(address):
                # The probe reads the version block with the firmware's own
                # tree, so the package goes on the path first, before the
                # composition would put it there (None for one already importable).
                _firmware.load_package(args.firmware)
                return _firmware.bridge_version(address)
            return _soc.ProcessorSystem(args.ip_ps, read_version=read_version)
        if args.check_firmware or args.hard_boot:
            raise LaunchError(f"the {bring_up.firmware_image} image is booted by the board's processor system, "
                              f"whose address is not known; pass --ps-addr (or -D to leave the image alone)")
        return None
    if crate_id is None:
        return None
    shm.address = args.ip_addr or None
    return shm


def _no_way_to_reboot(args, hardware, bring_up):
    """Why step 7 cannot be done, and what would let it."""
    if not hardware:
        return "--hard-boot: an emulated server has no board to reboot"
    if bring_up is None:
        return ("--hard-boot needs to know the board: pass --shelfmanager and --slot for a crate that answers, "
                "or --platform (with --ps-addr for a carrier its processor system boots)")
    return (f"--hard-boot reaches the FPGA through the crate's shelf manager, which "
            f"{'did not answer' if args.slot is not None else 'was not given'}; pass --shelfmanager and --slot")


def _not_checked(args, firmware_dir, bring_up):
    """Step 8 with no loader: say so in the log -- or refuse, when an image on disk could have been checked.

    With no record and no ``--platform`` an image in the firmware directory
    is not matched to a platform by its name: the start is refused and told
    to name one, or to call the check off.
    """
    if bring_up is not None:
        _log.info("the %s image is loaded by the crate's shelf manager, which %s; what the FPGA is running "
                  "is not checked", bring_up.firmware_image,
                  'did not answer' if args.slot is not None else 'was not given (no --shelfmanager, --slot)')
        return
    images = sorted(p.name for suffix in IMAGE_SUFFIXES for p in firmware_dir.glob(f"*{suffix}"))
    if images:
        raise LaunchError(f"{', '.join(images)} in {firmware_dir}: an image to check the FPGA against, but no "
                          f"carrier record names the platform; pass --platform, or -D to leave the image alone")
    _log.info("no platform named and no image in %s; what the FPGA is running is not checked", firmware_dir)


def _check_firmware(args, done, loader, bring_up, firmware_dir):
    """Step 8: the FPGA runs the image on disk, or is loaded with it.

    The image's file name carries its short git hash; the loader reads the
    FPGA's. Equal means nothing to do. Different means the loader programs
    the image, which reboots the FPGA into it, so the register configuration
    is applied at start-up as after a hard boot.
    """
    pattern = f"*{bring_up.firmware_image}*{loader.IMAGE_SUFFIX}"
    images = sorted(firmware_dir.glob(pattern))
    if len(images) != 1:
        what = 'no image' if not images else f"{len(images)} images"
        raise LaunchError(f"{what} matching {pattern} in {firmware_dir}; keep one there, or pass -D to skip "
                          f"the firmware check")
    image = images[0]
    on_disk = loader.image_hash(image)
    if on_disk is None:
        raise LaunchError(f"{image.name} carries no git hash in its name; pass -D to skip the firmware check")
    done.inventory['firmware_image_on_disk'] = image.name
    try:
        running = loader.firmware_git_hash()
    except CryodaqError as e:
        raise LaunchError(f"{e}; pass -D to skip the firmware check") from e
    done.inventory['firmware_running'] = running
    if running == on_disk:
        _log.info("FPGA firmware %s, image on disk %s (%s): they match, nothing to load", running, on_disk, image.name)
        return
    _log.warning("FPGA firmware %s, image on disk %s: loading %s through %s", running, on_disk, image.name, loader)
    loader.program(image)
    done.inventory['firmware_running'] = on_disk
    done.decided['programmed'] = image.name
    done.set_default(args, 'configure', True)


# --------------------------------------------------------------------------
# what the log says
# --------------------------------------------------------------------------

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
