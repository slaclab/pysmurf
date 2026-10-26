#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Platform Maps
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
#    The map between a platform's firmware registers and cryodaq's semantic
#    names, and the lookup over it.
#
#    A semantic name is dotted, with indexed scopes: band[4].tone.amplitude,
#    bay[0].attenuator[1].uc, stream.downsample.factor. Its pattern is the same
#    string with every index replaced by '*', and a map is a table from pattern
#    to a register path template and the kind of node it is. Resolving a name is
#    a dictionary lookup and a string format; a name with no entry, or with an
#    index the tree does not have, raises UnresolvedName rather than returning
#    None.
#
#    A platform is identified by the firmware tag its FPGA reports -- the image
#    name in the build stamp -- which each map lists the tags it covers. Identity
#    is therefore taken from the system in hand and never from a caller's flag or
#    configuration file; a tree with no firmware behind it, such as a register
#    emulation, has no tag to read and its platform has to be declared.
#
#    Nothing here writes a register, and the functions that ask what is in a
#    tree take a `has(path)` predicate rather than the tree: the caller does the
#    reading. Identification is the one exception and is deliberate -- it takes
#    the tree and reads the build stamp itself, because a platform that could be
#    told what it is would not be discovering anything. It reads that one
#    register and no other, through `getNode` and `get` alone, so a tree here is
#    duck-typed and no rogue is imported.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import functools
import re
from dataclasses import dataclass, field
from typing import (Any, Callable, Dict, List, Mapping, Sequence, Tuple)

from cryodaq._errors import ConnectError, UnresolvedName
from cryodaq.platform import _umux, atca, rfmc, rfsoc

__all__ = ['PlatformMap', 'MAPS', 'identify', 'from_stamp', 'by_name',
           'tag_of', 'parse', 'expand', 'indices', 'witness_names', 'VALUE', 'COMMAND',
           'PROCESS', 'DEVICE', 'KINDS', 'MAX_SCOPE_INDEX', 'ROOT_PATH', 'STATUS_DEVICE_PATH']

# What kind of node a name reaches: a value is read and written, a command is
# called, a process is started and polled.
VALUE = 'value'
COMMAND = 'command'
PROCESS = 'process'
# A device is where operations attach: the server composition resolves such a
# name to hand a provider the device, and a client has nothing to get or set
# at it.
DEVICE = 'device'
KINDS = (VALUE, COMMAND, PROCESS, DEVICE)

# How an indexed scope is enumerated: in windows this wide, every index in a
# window probed, and the next window probed as long as the last one found
# anything. A gap inside a window does not end the scope -- a firmware mask may
# leave a band out and keep a higher one, so a gap is a fact about the tree
# rather than the end of the scope -- and a scope the hardware has hundreds of,
# the channels of a band, is enumerated to its end without this file knowing
# how many there are: how many of something a tree has is the tree's to say.
MAX_SCOPE_INDEX = 32

# The root of every tree, which is what a server names its own.
ROOT_PATH = _umux.ROOT
# The device a server reports its own state at.
STATUS_DEVICE_PATH = _umux.APPLICATION

# Matched with fullmatch, not match: `$` also matches just before a trailing
# newline, so a name carrying one -- out of a file, off a command line -- would
# otherwise parse and then be formatted into a register path.
_SEGMENT = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+|\*)\])?')


def parse(name: str) -> Tuple[str, Dict[str, int]]:
    """Split a name into its pattern and its indices.

    Parameters
    ----------
    name : str
        e.g. ``band[4].tone.amplitude``.

    Returns
    -------
    tuple
        The pattern (``band[*].tone.amplitude``) and a mapping of scope to
        index (``{'band': 4}``). An index of ``*`` is reported as -1, so a
        pattern parses but does not resolve. The mapping is a fresh dictionary
        each call, so a caller may keep or modify it.

    Raises
    ------
    UnresolvedName
        If the name is not well formed.

    Notes
    -----
    Answers are cached, because a name parses to the same pattern every time and
    the names in use are a small fixed set -- one per register, with the indices
    of one system. A tuning loop reaching a per-channel register thousands of
    times pays for the regular expressions once rather than per call.
    """
    # Checked here, before the cache hashes it: an unhashable non-string would
    # otherwise surface as the cache's TypeError rather than as UnresolvedName.
    if not isinstance(name, str) or not name:
        raise UnresolvedName(str(name), reason='empty name')
    pattern, found = _parse_cached(name)
    return pattern, dict(found)


@functools.lru_cache(maxsize=4096)
def _parse_cached(name: str) -> Tuple[str, Tuple[Tuple[str, int], ...]]:
    """``parse`` without the copy, keyed on a non-empty string. Indices as a pairs tuple.

    Separate from ``parse`` so that what is cached is immutable: handing the same
    dictionary to two callers would let one of them change what the other reads.
    A malformed name raises here too, and the exception is not cached -- so a
    caller that fixes a typo is not told the old answer.
    """
    pattern: List[str] = []
    found: Dict[str, int] = {}
    for segment in name.split('.'):
        m = _SEGMENT.fullmatch(segment)
        if m is None:
            raise UnresolvedName(name, reason=f"bad segment {segment!r}")
        ident, index = m.groups()
        if index is None:
            pattern.append(ident)
            continue
        if ident in found:
            raise UnresolvedName(name, reason=f"scope {ident!r} indexed twice")
        found[ident] = -1 if index == '*' else int(index)
        pattern.append(f"{ident}[*]")
    return '.'.join(pattern), tuple(found.items())


def _fill(template: str, values: Mapping[str, int]) -> str:
    """Substitute a path template's placeholders, saying which one is missing."""
    try:
        return template.format(**values)
    except KeyError as e:
        raise UnresolvedName(template, reason=f"no index for {e.args[0]!r}") from e


@dataclass(frozen=True)
class PlatformMap:
    """One platform generation's map from semantic names to register paths.

    Parameters
    ----------
    name : str
        Short identifier, e.g. ``umux-atca``.
    tags : tuple of str
        The firmware image names this platform runs; how a tree is recognised as
        belonging to this map. More than one where several firmware lines share
        a platform's registers and its bring-up.
    registers : mapping
        Name pattern to ``(path template, kind)``. A platform module declares
        this as two tables, core and legacy; the map holds their union and does
        not distinguish them, since a name resolves the same way from either.
    witness : tuple of str
        Name patterns worth reading back to record how a system was left.
    scopes : mapping
        Scope name to ``(path templates that prove an index, parent scopes)``.
    module : str
        The platform's own module, ``cryodaq.platform.<name>``; the server
        finds the platform's rogue code beside it (``<module>.support``).
    tag_path : str
        The build stamp register, by which a tree is recognised as this
        platform's: inside the carrier core, which is named for the board, so
        where it is depends on the platform.
    package_config_dir : str
        Where this platform's firmware release archive keeps its configuration
        files, relative to the archive's root.
    """

    name: str
    tags: Tuple[str, ...]
    registers: Mapping[str, Tuple[str, str]]
    witness: Tuple[str, ...]
    scopes: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]
    module: str = ''
    tag_path: str = ''
    package_config_dir: str = ''

    def __contains__(self, pattern: str) -> bool:
        return pattern in self.registers

    def __len__(self) -> int:
        return len(self.registers)

    @property
    def patterns(self) -> Tuple[str, ...]:
        """Every name pattern in the map, sorted."""
        return tuple(sorted(self.registers))

    def entry(self, name: str) -> Tuple[str, str]:
        """The register path and kind a concrete name resolves to.

        Raises
        ------
        UnresolvedName
            If the map has no such pattern, or the name is a pattern rather
            than a name.
        """
        pattern, found = parse(name)
        if pattern not in self:
            raise UnresolvedName(name, pattern=pattern, reason='not in this platform map')
        if any(index < 0 for index in found.values()):
            raise UnresolvedName(name, pattern=pattern,
                                 reason='a pattern, not a name: every scope needs an index')
        template, kind = self.registers[pattern]
        return _fill(template, found), kind

    def path(self, name: str) -> str:
        """The register path a name resolves to."""
        return self.entry(name)[0]

    def kind(self, name: str) -> str:
        """``'value'``, ``'command'``, ``'process'`` or ``'device'`` for what a name reaches.

        The kind is the pattern's, so a pattern (``band[*].ops``) answers as
        well as a name (``band[0].ops``).
        """
        pattern, _ = parse(name)
        if pattern not in self:
            raise UnresolvedName(name, pattern=pattern, reason='not in this platform map')
        return self.registers[pattern][1]

    def doc(self, name: str) -> str:
        """One line naming the register behind a name; the map carries no prose."""
        path, kind = self.entry(name)
        return f"{kind} at {path}"


def _from_module(module: Any) -> PlatformMap:
    """Build the map a platform module declares as data."""
    return PlatformMap(name=module.NAME, tags=tuple(module.TAGS),
                       registers={**module.REGISTERS, **module.LEGACY},
                       witness=tuple(module.WITNESS),
                       scopes=dict(module.SCOPES), module=module.__name__,
                       tag_path=module.CARRIER.BUILD_STAMP,
                       package_config_dir=module.PACKAGE_CONFIG_DIR)


# Every supported platform. One module per platform, and a platform is a set of
# firmware that shares both a register map and a bring-up procedure -- so two
# that read the same registers but have to be configured differently are two
# entries here, however much of their table they share.
MAPS = (_from_module(atca), _from_module(rfsoc), _from_module(rfmc))


def tag_of(stamp: Any) -> str:
    """The firmware image name in a build stamp.

    Parameters
    ----------
    stamp : str or None
        A build stamp as the firmware reports it, e.g.
        ``MicrowaveMuxBpEthGen2: Vivado v2020.2, host (os), Built ... by ...``.

    Returns
    -------
    str
        The image name, or ``''`` when there is nothing to read -- which is what
        an emulated register space reports, its memory being zeros.
    """
    if not stamp:
        return ''
    # The register is a fixed-width character buffer, so a stamp arrives padded
    # and an unwritten one is padding alone.
    text = str(stamp).split('\x00')[0].strip()
    return text.split(':')[0].strip()


def by_name(name: str) -> PlatformMap:
    """The map called ``name``.

    Raises
    ------
    ConnectError
        If no map goes by it.
    """
    for candidate in MAPS:
        if candidate.name == name:
            return candidate
    known = ', '.join(m.name for m in MAPS)
    raise ConnectError(f"no platform called {name!r}; cryodaq has {known}")


def from_stamp(stamp: Any) -> PlatformMap:
    """Return the map for the firmware a build stamp names.

    Parameters
    ----------
    stamp : str or None
        A build stamp as the firmware reports it (see `tag_of`).

    Raises
    ------
    ConnectError
        If the stamp names no firmware, or firmware no map claims.
    """
    tag = tag_of(stamp)
    if not tag:
        raise _no_firmware('')
    for candidate in MAPS:
        if tag in candidate.tags:
            return candidate
    raise _unknown_firmware(tag)


def _no_firmware(where: str) -> ConnectError:
    return ConnectError(
        f"this tree reports no firmware{where}, so its platform cannot be identified; "
        f"declare one of {', '.join(repr(m.name) for m in MAPS)} instead")


def _unknown_firmware(tag: str) -> ConnectError:
    known = ', '.join(f"{t} ({m.name})" for m in MAPS for t in m.tags)
    return ConnectError(f"firmware {tag!r} belongs to no platform cryodaq knows; it has {known}")


def identify(tree: Any) -> PlatformMap:
    """Return the map for the firmware a tree is running.

    Discovery only: the answer comes from the hardware every time. A caller who
    knows which platform this is, because the tree cannot say, names it and gets
    ``by_name`` instead -- the two questions are two functions.

    Parameters
    ----------
    tree : object
        The server's tree, as a session's ``root`` is. ``getNode(path)`` and
        ``get()`` on what it returns are all that is asked of it, so this needs
        no rogue import. The build stamp is read at each place a platform keeps
        one -- the carrier cores are named for their boards -- and a tree has
        exactly one of them.

    Returns
    -------
    PlatformMap

    Raises
    ------
    ConnectError
        If the tree reports no firmware, or reports firmware no map claims.
    """
    # Each stamp register read once, though two platforms may keep theirs at
    # the same path.
    read: Dict[str, Any] = {}
    for candidate in MAPS:
        if candidate.tag_path not in read:
            node = tree.getNode(candidate.tag_path)
            read[candidate.tag_path] = None if node is None else node.get()
        if tag_of(read[candidate.tag_path]) in candidate.tags:
            return candidate
    tags = [tag_of(stamp) for stamp in read.values() if tag_of(stamp)]
    if not tags:
        raise _no_firmware(f" at {' or '.join(read)}")
    # A stamp was read and no map claims it at the register it was read from.
    raise _unknown_firmware(tags[0])


def indices(pmap: PlatformMap, has: Callable[[str], bool], scope: str,
            **fixed: int) -> Tuple[int, ...]:
    """Which indices of an indexed scope a tree has.

    Parameters
    ----------
    pmap : PlatformMap
        The map that declares how the scope is enumerated.
    has : callable
        ``has(path) -> bool``, answered against the tree in hand.
    scope : str
        e.g. ``band``, ``bay``.
    ``**fixed``
        Indices of the scope's parents, where it has any.

    Returns
    -------
    tuple of int
        The indices present, in order, gaps included -- a disabled bay or a
        firmware band mask leaves one out without ending the scope. Empty when
        the tree has none, which is how a platform without the hardware behind
        them is seen. The scope ends at the first ``MAX_SCOPE_INDEX`` consecutive
        indices the tree lacks, however many it has before that.
    """
    if scope not in pmap.scopes:
        raise KeyError(f"{pmap.name} has no scope {scope!r}")
    templates, parents = pmap.scopes[scope]
    missing = [p for p in parents if p not in fixed]
    if missing:
        raise KeyError(f"scope {scope!r} is inside {parents}; give {', '.join(missing)}")
    present: List[int] = []
    start = 0
    while True:
        found_in_window = False
        for i in range(start, start + MAX_SCOPE_INDEX):
            values = dict(fixed, **{scope: i})
            if any(has(_fill(t, values)) for t in templates):
                present.append(i)
                found_in_window = True
        if not found_in_window:
            return tuple(present)
        start += MAX_SCOPE_INDEX


def _scopes_of(pattern: str) -> Tuple[str, ...]:
    """The indexed scopes a pattern carries, left to right."""
    return tuple(segment.split('[')[0] for segment in pattern.split('.')
                 if segment.endswith('[*]'))


def expand(pmap: PlatformMap, has: Callable[[str], bool], pattern: str,
           sample: Sequence[str] = ()) -> List[str]:
    """Fill a pattern's ``*`` placeholders with the indices the tree has.

    Parameters
    ----------
    pmap : PlatformMap
        The map the pattern belongs to.
    has : callable
        ``has(path) -> bool``.
    pattern : str
        e.g. ``bay[*].attenuator[*].uc``. A pattern with no scope expands to
        itself.
    sample : sequence of str, optional
        Scopes to fill with their first, middle and last index only, for a
        caller that wants a representative name per scope rather than every
        one of them.

    Returns
    -------
    list of str
        Concrete names, empty when a scope in the pattern has no indices here.
    """
    out: List[str] = []
    # A scope name, not a string to search in: 'lut' must not sample 'lut_dac'.
    sampled = {sample} if isinstance(sample, str) else set(sample)

    def walk(text: str, remaining: Sequence[str], fixed: Dict[str, int]) -> None:
        if not remaining:
            out.append(text)
            return
        scope, rest = remaining[0], remaining[1:]
        parents = pmap.scopes[scope][1] if scope in pmap.scopes else ()
        found = indices(pmap, has, scope, **{p: fixed[p] for p in parents})
        if scope in sampled and found:
            found = tuple(dict.fromkeys((found[0], found[len(found) // 2], found[-1])))
        for index in found:
            walk(text.replace(f"{scope}[*]", f"{scope}[{index}]", 1), rest,
                 dict(fixed, **{scope: index}))

    walk(pattern, _scopes_of(pattern), {})
    return out


def witness_names(pmap: PlatformMap, has: Callable[[str], bool]) -> Tuple[str, ...]:
    """The map's witness patterns, expanded over the indices this tree has."""
    names: List[str] = []
    for pattern in pmap.witness:
        names.extend(expand(pmap, has, pattern))
    return tuple(names)
