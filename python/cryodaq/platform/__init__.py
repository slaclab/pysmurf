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
#    One map covers one generation of hardware, however many boards share it. A
#    platform is identified by the shape of its tree, never by a firmware string
#    or a command-line flag. Nothing here reads or writes a register: the
#    functions that need to know what is in a tree take a `has(path)` predicate
#    and the caller does the reading.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import re
from dataclasses import dataclass
from typing import (Any, Callable, Dict, List, Mapping, Sequence, Tuple)

from cryodaq._errors import ConnectError, UnresolvedName
from cryodaq.platform import _umux

__all__ = ['PlatformMap', 'MAPS', 'identify', 'parse', 'expand', 'indices',
           'witness_names', 'VALUE', 'COMMAND', 'PROCESS', 'KINDS',
           'MAX_SCOPE_INDEX']

# What kind of node a name reaches: a value is read and written, a command is
# called, a process is started and polled.
VALUE = 'value'
COMMAND = 'command'
PROCESS = 'process'
KINDS = (VALUE, COMMAND, PROCESS)

# How far an index is probed for when enumerating a scope. Probing stops at the
# first absent index after at least one present one, so this is a ceiling rather
# than an assumption about how the firmware is built.
MAX_SCOPE_INDEX = 32

_SEGMENT = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)(?:\[(\d+|\*)\])?$')


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
        pattern parses but does not resolve.

    Raises
    ------
    UnresolvedName
        If the name is not well formed.
    """
    if not isinstance(name, str) or not name:
        raise UnresolvedName(str(name), reason='empty name')
    pattern: List[str] = []
    found: Dict[str, int] = {}
    for segment in name.split('.'):
        m = _SEGMENT.match(segment)
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
    return '.'.join(pattern), found


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
        Short identifier, e.g. ``umux``.
    probe : str
        A register path this platform has and others do not; how a tree is
        recognised as belonging to this map.
    registers : mapping
        Name pattern to ``(path template, kind)``.
    witness : tuple of str
        Name patterns worth reading back to record how a system was left.
    scopes : mapping
        Scope name to ``(path templates that prove an index, parent scopes)``.
    """

    name: str
    probe: str
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
    return PlatformMap(name=module.NAME, probe=module.PROBE,
                       registers=dict(module.REGISTERS),
                       witness=tuple(module.WITNESS),
                       scopes=dict(module.SCOPES))


# Every supported platform generation, in the order a tree is tested against
# them. A generation whose register paths, or whose bring-up and configuration
# procedures, genuinely differ arrives as another module here -- never as a
# branch inside one. Two generations sharing a module is a statement that they
# are the same platform, so it is worth re-earning each time one is added.
MAPS = (_from_module(_umux),)


def identify(root: Any) -> PlatformMap:
    """Return the map whose probe register this tree has.

    Parameters
    ----------
    root : object
        A rogue root, local or over a client; only ``getNode`` is used.

    Raises
    ------
    ConnectError
        If no supported map recognises the tree.
    """
    for candidate in MAPS:
        try:
            if root.getNode(candidate.probe) is not None:
                return candidate
        except Exception:                                       # noqa: BLE001
            continue
    tried = ', '.join(f"{m.name} ({m.probe})" for m in MAPS)
    raise ConnectError(f"no supported platform map recognises this tree; tried {tried}")


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
        The indices present, in order; empty when the tree has none, which is
        how a platform without the hardware behind them is seen.
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
        elif present:
            break
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
