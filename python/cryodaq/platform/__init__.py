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
#    bay[0].attenuator.uc[1], stream.downsample.factor. Its pattern is the same
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
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Mapping, Sequence, Tuple)

from cryodaq._errors import ConnectError, UnresolvedName
from cryodaq.platform import _atca, _rfsoc, _umux

__all__ = ['PlatformMap', 'MAPS', 'identify', 'by_name', 'tag_of', 'parse',
           'expand', 'indices', 'witness_names', 'VALUE', 'COMMAND', 'PROCESS',
           'KINDS', 'MAX_SCOPE_INDEX', 'TAG_PATH']

# What kind of node a name reaches: a value is read and written, a command is
# called, a process is started and polled.
VALUE = 'value'
COMMAND = 'command'
PROCESS = 'process'
KINDS = (VALUE, COMMAND, PROCESS)

# How far an index is probed for when enumerating a scope. Every index below it
# is probed, with no stopping at the first gap: a firmware mask may leave one out
# and keep a higher one, so a gap is a fact about the tree rather than the end of
# the scope.
MAX_SCOPE_INDEX = 32

# Where a system reports the firmware it is running. Every generation supported
# here carries an AMC carrier core, so one path serves them all; a generation
# that reports its firmware elsewhere makes this a property of each map.
TAG_PATH = _umux.BUILD_STAMP

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
    pattern, found = _parse_cached(name)
    return pattern, dict(found)


@functools.lru_cache(maxsize=4096)
def _parse_cached(name: str) -> Tuple[str, Tuple[Tuple[str, int], ...]]:
    """``parse`` without the copy, keyed on the name. Indices as a pairs tuple.

    Separate from ``parse`` so that what is cached is immutable: handing the same
    dictionary to two callers would let one of them change what the other reads.
    A malformed name raises here too, and the exception is not cached -- so a
    caller that fixes a typo is not told the old answer.
    """
    if not isinstance(name, str) or not name:
        raise UnresolvedName(str(name), reason='empty name')
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
        Name pattern to ``(path template, kind)``.
    witness : tuple of str
        Name patterns worth reading back to record how a system was left.
    scopes : mapping
        Scope name to ``(path templates that prove an index, parent scopes)``.
    """

    name: str
    tags: Tuple[str, ...]
    registers: Mapping[str, Tuple[str, str]]
    witness: Tuple[str, ...]
    scopes: Mapping[str, Tuple[Tuple[str, ...], Tuple[str, ...]]]

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
        if pattern not in self.registers:
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
        """``'value'``, ``'command'`` or ``'process'`` for what the name reaches."""
        return self.entry(name)[1]

    def doc(self, name: str) -> str:
        """One line naming the register behind a name; the map carries no prose."""
        path, kind = self.entry(name)
        return f"{kind} at {path}"


def _from_module(module: Any) -> PlatformMap:
    """Build the map a platform module declares as data."""
    return PlatformMap(name=module.NAME, tags=tuple(module.TAGS),
                       registers=dict(module.REGISTERS),
                       witness=tuple(module.WITNESS),
                       scopes=dict(module.SCOPES))


# Every supported platform. One module per platform, and a platform is a set of
# firmware that shares both a register map and a bring-up procedure -- so two
# that read the same registers but have to be configured differently are two
# entries here, however much of their table they share.
MAPS = (_from_module(_atca), _from_module(_rfsoc))


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
        no rogue import; the build stamp is read once.

    Returns
    -------
    PlatformMap

    Raises
    ------
    ConnectError
        If the tree reports no firmware, or reports firmware no map claims.
    """
    node = tree.getNode(TAG_PATH)
    tag = tag_of(None if node is None else node.get())
    if not tag:
        raise ConnectError(
            f"this tree reports no firmware at {TAG_PATH}, so its platform "
            f"cannot be identified; declare one of "
            f"{', '.join(repr(m.name) for m in MAPS)} instead")
    for candidate in MAPS:
        if tag in candidate.tags:
            return candidate
    known = ', '.join(f"{t} ({m.name})" for m in MAPS for t in m.tags)
    raise ConnectError(f"firmware {tag!r} belongs to no platform cryodaq knows; "
                       f"it has {known}")


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
        them is seen.
    """
    if scope not in pmap.scopes:
        raise KeyError(f"{pmap.name} has no scope {scope!r}")
    templates, parents = pmap.scopes[scope]
    missing = [p for p in parents if p not in fixed]
    if missing:
        raise KeyError(f"scope {scope!r} is inside {parents}; give {', '.join(missing)}")
    present: List[int] = []
    for i in range(MAX_SCOPE_INDEX):
        values = dict(fixed, **{scope: i})
        if any(has(_fill(t, values)) for t in templates):
            present.append(i)
    return tuple(present)


def _scopes_of(pattern: str) -> Tuple[str, ...]:
    """The indexed scopes a pattern carries, left to right."""
    return tuple(segment.split('[')[0] for segment in pattern.split('.')
                 if segment.endswith('[*]'))


def expand(pmap: PlatformMap, has: Callable[[str], bool], pattern: str) -> List[str]:
    """Fill a pattern's ``*`` placeholders with the indices the tree has.

    Parameters
    ----------
    pmap : PlatformMap
        The map the pattern belongs to.
    has : callable
        ``has(path) -> bool``.
    pattern : str
        e.g. ``bay[*].attenuator.uc[*]``. A pattern with no scope expands to
        itself.

    Returns
    -------
    list of str
        Concrete names, empty when a scope in the pattern has no indices here.
    """
    out: List[str] = []

    def walk(text: str, remaining: Sequence[str], fixed: Dict[str, int]) -> None:
        if not remaining:
            out.append(text)
            return
        scope, rest = remaining[0], remaining[1:]
        parents = pmap.scopes[scope][1] if scope in pmap.scopes else ()
        for index in indices(pmap, has, scope, **{p: fixed[p] for p in parents}):
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
