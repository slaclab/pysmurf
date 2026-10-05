#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Configuration
#-----------------------------------------------------------------------------
# File       : config.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
#    How a configuration is resolved from layers, and how the resolved result
#    is recorded beside the system it configured.
#
#    A configuration is a YAML mapping. A file may name the files it builds on
#    with an `inherit:` key -- one path or a list, relative to the file -- and
#    the application may supply a default layer under everything. Resolving
#    applies the layers in order, default first, deep-merging mappings and
#    replacing everything else, and remembers for every key which file and
#    line set it. The result is a Resolved: the flat values, that provenance,
#    a hash of the values, and the chain of layers.
#
#    What the keys mean is not decided here. The application passes a
#    `validate` callable that sees the merged values and raises its own errors;
#    this module refuses only what is wrong with the layering itself -- a file
#    that cannot be read, a chain that loops, a layer that is not a mapping.
#    Nothing here knows a key name, a register or a unit.
#
#    The configuration record is a JSON file holding a Resolved plus what the
#    system looked like when it was applied: firmware identity and the witness
#    registers. It is written atomically -- to a temporary file in the same
#    directory, synced, then renamed over the old one -- so a reader finds the
#    previous record or the new one and never half of either, and a dated copy
#    is kept beside it. The configuring operation writes it; nothing reads it
#    back into a session yet. It is the on-disk record that persisting values
#    an operation measures will build on, and it is a record, not a source of
#    truth: the server carries what it was given, and deleting the file costs
#    nothing but history.
#
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import hashlib
import json
import os
import re
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union)

import yaml

from cryodaq._errors import ConfigError

__all__ = ['Resolved', 'load', 'merge', 'flatten', 'INHERIT_KEY', 'DEFAULT_LAYER',
           'VALIDATED_LAYER', 'write_record', 'read_record', 'record_path', 'RECORD_DIR']

# The key a layer names its parents with. Consumed by the resolution; it is not
# a value and does not appear in the result.
INHERIT_KEY = 'inherit'

# What provenance calls the layer the application supplied as a mapping rather
# than a file. A default read from a file is named by its path like any other.
DEFAULT_LAYER = '<default>'

# What provenance calls a value no layer set: the application's validator
# filled it in. Its line is 0.
VALIDATED_LAYER = '<validated>'

# Where configuration records live under a status directory, and the dated-copy format.
RECORD_DIR = 'resolved'
_HISTORY_STAMP = '%Y%m%dT%H%M%SZ'
# A hostname's own characters stay; ':' becomes '_'; anything else is %XX. The
# three alphabets are disjoint, so two endpoints never share a file name.
_SLUG_KEEP = re.compile(r'[A-Za-z0-9.-]')

Layer = Union[str, Path, Mapping[str, Any]]
Validator = Callable[[Dict[str, Any]], Dict[str, Any]]


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Resolved:
    """A configuration resolved from its layers.

    Parameters
    ----------
    values : mapping
        The merged values, ``inherit`` removed.
    provenance : mapping
        Dotted key to ``(layer, line)``: which layer set the value that
        survived, and the line in it; ``(VALIDATED_LAYER, 0)`` for a value the
        validator filled in. Every leaf of ``values`` has an entry and nothing
        else does; a mapping's provenance is that of its keys.
    hash : str
        SHA-256 of the canonical JSON of ``values``, so two resolutions that
        agree on every value agree here, whatever their layering.
    layers : tuple of str
        The chain in the order applied, first is lowest.
    """

    values: Mapping[str, Any]
    provenance: Mapping[str, Tuple[str, int]]
    hash: str
    layers: Tuple[str, ...]

    def get(self, key: str, default: Any = None) -> Any:
        """The value at a dotted key.

        Parameters
        ----------
        key : str
            Dotted path into ``values``, e.g. ``'bands.4.att_uc'``.
        default : object, optional
            Returned when no value sits at ``key``.
        """
        node: Any = self.values
        for part in key.split('.'):
            if not isinstance(node, Mapping):
                return default
            if part not in node:
                # A validator may have made numeric keys int (``bands: {4: ...}``);
                # the dotted path is text either way.
                if not (part.isdigit() and int(part) in node):
                    return default
                part = int(part)  # type: ignore[assignment]
            node = node[part]
        return node

    def to_dict(self) -> Dict[str, Any]:
        """A plain, JSON-serialisable form; ``from_dict`` reverses it."""
        return {'values': _plain(self.values),
                'provenance': {k: list(v) for k, v in self.provenance.items()},
                'hash': self.hash,
                'layers': list(self.layers)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> 'Resolved':
        """Rebuild from ``to_dict``'s output; the hash is recomputed and checked.

        Parameters
        ----------
        data : mapping
            What ``to_dict`` returned, possibly after a trip through JSON:
            ``values``, ``provenance``, ``hash`` and ``layers``.

        Raises
        ------
        ConfigError
            If ``data['hash']`` is present and is not the hash of ``data['values']``.
        """
        values = data['values']
        recorded = data.get('hash')
        computed = hash_of(values)
        if recorded is not None and recorded != computed:
            raise ConfigError('<record>', reason=f"hash {recorded} does not match "
                              f"the values it records ({computed})")
        return cls(values=values,
                   provenance={k: (v[0], int(v[1])) for k, v in data.get('provenance', {}).items()},
                   hash=computed,
                   layers=tuple(data.get('layers', ())))


def hash_of(values: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a mapping.

    Parameters
    ----------
    values : mapping
        Nested configuration values; keys are stringified and sorted, so two
        mappings equal as JSON hash the same whatever their key types or order.
    """
    canonical = json.dumps(_plain(values), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _plain(value: Any, key: str = '') -> Any:
    """Copy ``value`` into what JSON takes, or refuse it by key.

    Mappings become dicts with string keys, sequences lists, a path its string,
    a NumPy scalar the Python scalar it holds. Anything else that JSON cannot
    write -- and two keys JSON would merge, ``1`` and ``'1'`` -- is refused
    here, naming the dotted key, rather than deep inside ``json.dumps`` naming
    nothing or dropping a value.
    """
    if isinstance(value, Mapping):
        out = {}
        for k, v in value.items():
            if str(k) in out:
                # 1 and '1' are one JSON key; keeping either would drop the other.
                raise ConfigError('<record>', key=f"{key}.{k}" if key else str(k),
                                  reason=f"keys {k!r} and {str(k)!r} are the same key in JSON")
            out[str(k)] = _plain(v, f"{key}.{k}" if key else str(k))
        return out
    if isinstance(value, (list, tuple)):
        return [_plain(v, f"{key}[{i}]") for i, v in enumerate(value)]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if hasattr(value, 'item') and hasattr(value, 'dtype'):       # a NumPy scalar
        return _plain(value.item(), key)
    raise ConfigError('<record>', key=key or None,
                      reason=f"not JSON-serialisable: {type(value).__name__}")


# --------------------------------------------------------------------------
# reading a layer
# --------------------------------------------------------------------------

def _read_yaml(path: Path) -> Tuple[Any, Dict[str, int]]:
    """Parse one YAML file; the mapping and the line every dotted key starts on.

    Lines come from the parser's own marks: the document is composed into its
    node graph, which carries a start mark per key, and walked once. The values
    are then constructed by the safe loader as usual. No other library.
    """
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as e:
        raise ConfigError(str(path), reason=f"cannot read: {e.strerror or e}") from e
    except UnicodeDecodeError as e:
        raise ConfigError(str(path), reason=f"not UTF-8 text: byte {e.start}") from e
    try:
        root = yaml.compose(text, Loader=yaml.SafeLoader)
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        mark = getattr(e, 'problem_mark', None)
        where = f" at line {mark.line + 1}" if mark is not None else ''
        problem = getattr(e, 'problem', None) or e
        raise ConfigError(str(path), reason=f"not valid YAML{where}: {problem}") from e
    lines: Dict[str, int] = {}
    if isinstance(root, yaml.MappingNode):
        _key_lines(root, '', lines)
    return data, lines


def _key_lines(node: Any, prefix: str, lines: Dict[str, int]) -> None:
    """Record the line every key under ``node`` starts on, by dotted path."""
    for key_node, value_node in node.value:
        dotted = f"{prefix}{key_node.value}"
        lines[dotted] = key_node.start_mark.line + 1
        if isinstance(value_node, yaml.MappingNode):
            _key_lines(value_node, f"{dotted}.", lines)


def _load_layer(layer: Layer, base: Optional[Path]) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    """A layer's name, its mapping and its key lines."""
    if isinstance(layer, Mapping):
        _refuse_dotted_keys(DEFAULT_LAYER, layer)
        return DEFAULT_LAYER, dict(layer), {}
    path = Path(layer)
    if base is not None and not path.is_absolute():
        path = (base / path)
    path = path.resolve()
    data, lines = _read_yaml(path)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise ConfigError(str(path), reason=f"a configuration layer is a mapping, not "
                          f"{type(data).__name__}")
    _refuse_dotted_keys(str(path), data)
    return str(path), dict(data), lines


def _refuse_dotted_keys(name: str, values: Mapping[str, Any], prefix: str = '') -> None:
    """A key with a ``.`` in it is refused: the dot is how a leaf is addressed.

    Provenance, ``Resolved.get`` and the record all name a leaf by its dotted
    path, so ``a.b: 1`` beside ``a: {b: 2}`` would be two leaves with one name.
    """
    for key, value in values.items():
        if '.' in str(key):
            raise ConfigError(name, key=f"{prefix}{key}",
                              reason="a key may not contain '.'; nest it instead")
        if isinstance(value, Mapping):
            _refuse_dotted_keys(name, value, f"{prefix}{key}.")


# --------------------------------------------------------------------------
# resolving
# --------------------------------------------------------------------------

def merge(lower: Mapping[str, Any], upper: Mapping[str, Any]) -> Dict[str, Any]:
    """``upper`` over ``lower``: mappings merge key by key, anything else is replaced.

    A list is replaced whole -- there is no way to say "these too" for a list,
    and appending silently would make the result depend on a layer the author
    of ``upper`` may never have seen.

    Parameters
    ----------
    lower : mapping
        The layer underneath.
    upper : mapping
        The layer on top; where both have a mapping at a key the two merge,
        otherwise ``upper``'s value wins. Neither argument is modified.
    """
    out: Dict[str, Any] = dict(lower)
    for key, value in upper.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = merge(out[key], value)
        else:
            out[key] = value
    return out


def flatten(values: Mapping[str, Any], prefix: str = '') -> Dict[str, Any]:
    """Leaves of a nested mapping as ``{'a.b.c': value}``; empty mappings are leaves.

    Parameters
    ----------
    values : mapping
        The nested mapping.
    prefix : str
        Prepended to every key; the recursion passes ``'a.b.'`` down.
    """
    out: Dict[str, Any] = {}
    for key, value in values.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, Mapping) and value:
            out.update(flatten(value, f"{dotted}."))
        else:
            out[dotted] = value
    return out


def _chain(layer: Layer, base: Optional[Path], seen: List[str]) -> List[Tuple[str, Dict[str, Any], Dict[str, int]]]:
    """The layers ``layer`` stands on, then itself; lowest first. Refuses a loop."""
    name, data, lines = _load_layer(layer, base)
    if name in seen:
        loop = ' -> '.join(seen + [name])
        raise ConfigError(name, key=INHERIT_KEY, reason=f"inheritance loops: {loop}")
    parents = data.pop(INHERIT_KEY, None)
    if parents is None:
        parents_list: Sequence[Any] = ()
    elif isinstance(parents, str):
        parents_list = (parents,)
    elif isinstance(parents, list) and all(isinstance(p, str) for p in parents):
        parents_list = parents
    else:
        raise ConfigError(name, key=INHERIT_KEY,
                          reason=f"names a path or a list of paths, not {type(parents).__name__}")
    here = Path(name).parent if name != DEFAULT_LAYER else base
    out: List[Tuple[str, Dict[str, Any], Dict[str, int]]] = []
    for parent in parents_list:
        out.extend(_chain(parent, here, seen + [name]))
    out.append((name, data, lines))
    return out


def load(path: Union[str, Path], *, default: Optional[Layer] = None,
         validate: Optional[Validator] = None) -> Resolved:
    """Resolve the configuration file at ``path`` over what it inherits.

    Parameters
    ----------
    path : str or Path
        The top layer. Its ``inherit:`` key names the layers below it, one path
        or a list, relative to the file; each of those may inherit in turn.
    default : str, Path or mapping, optional
        A layer under everything: the application's shipped defaults, as a
        file or as a mapping.
    validate : callable, optional
        ``validate(values) -> values``, the application's judgement of what the
        merged values mean. It may fill in or coerce; whatever it returns is the
        result, and a leaf it added is attributed to ``VALIDATED_LAYER``. Its
        exceptions are not caught here.

    Returns
    -------
    Resolved

    Raises
    ------
    ConfigError
        For a layer that cannot be read or is not a mapping, an ``inherit`` that
        is not a path or a list of paths, or a chain that loops. Each names the
        file, and the key where there is one.
    """
    chain: List[Tuple[str, Dict[str, Any], Dict[str, int]]] = []
    if default is not None:
        chain.extend(_chain(default, None, []))
    chain.extend(_chain(path, Path.cwd(), []))

    values: Dict[str, Any] = {}
    provenance: Dict[str, Tuple[str, int]] = {}
    for name, data, lines in chain:
        values = merge(values, data)
        for dotted in flatten(data):
            provenance[dotted] = (name, lines.get(dotted, 0))
    if validate is not None:
        values = validate(values)
    # Provenance describes the result. A key a higher layer replaced with a
    # mapping, removed by replacing its parent, or dropped by the validator
    # keeps no stale line; one the validator filled in is the validator's.
    provenance = {k: provenance.get(k, (VALIDATED_LAYER, 0)) for k in flatten(values)}
    return Resolved(values=values, provenance=provenance, hash=hash_of(values),
                    layers=tuple(name for name, _, _ in chain))


# --------------------------------------------------------------------------
# the configuration record
# --------------------------------------------------------------------------

def record_path(status_dir: Union[str, Path], endpoint: str) -> Path:
    """Where the configuration record for an endpoint lives under a status directory.

    One file per endpoint: ``<status_dir>/resolved/<endpoint-slug>.json``, the
    slug being the endpoint with ``:`` made ``_`` -- ``localhost:9012`` is
    ``localhost_9012.json`` -- and any character a hostname cannot hold
    percent-encoded, so no two endpoints name the same file.

    Parameters
    ----------
    status_dir : str or Path
        The status directory; ``RECORD_DIR`` is created under it when written.
    endpoint : str
        The server, as ``host:port``.

    Raises
    ------
    ConfigError
        If ``endpoint`` is empty: there is no file for it.
    """
    if not endpoint:
        raise ConfigError('<record>', reason='an empty endpoint has no record file')
    slug = ''.join(c if _SLUG_KEEP.match(c) else '_' if c == ':' else
                   ''.join(f"%{b:02X}" for b in c.encode()) for c in endpoint)
    return Path(status_dir) / RECORD_DIR / f"{slug}.json"


def write_record(resolved: Resolved, path: Union[str, Path], *,
                 endpoint: str = '', firmware: Optional[Mapping[str, Any]] = None,
                 witness: Optional[Mapping[str, Any]] = None,
                 extra: Optional[Mapping[str, Any]] = None,
                 history: bool = True) -> Path:
    """Record ``resolved`` and how the system looked when it was applied.

    Written to a temporary file in the same directory, synced to disk, then
    renamed over ``path``, so the file is always a whole record. A dated copy
    ``<stem>.<UTC stamp>.json`` is kept beside it unless ``history`` is False.

    Parameters
    ----------
    resolved : Resolved
        What was applied.
    path : str or Path
        Usually ``record_path(...)``. Its directory is created.
    endpoint : str, optional
        The server the configuration was applied to, as ``host:port``.
    firmware : mapping, optional
        The firmware identity fields of the session's description, so the
        record says what hardware it was written for.
    witness : mapping, optional
        Semantic name to value, the witness registers read after the
        configuration was applied.
    extra : mapping, optional
        Anything the application wants kept with the record -- its version,
        the file it started from.
    history : bool
        Keep the dated copy. False writes ``path`` alone.

    Returns
    -------
    Path
        ``path``, as written.
    """
    path = Path(path)
    now = time.gmtime()
    record = {
        'resolved': resolved.to_dict(),
        'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', now),
        'endpoint': endpoint,
        'firmware': _plain(firmware or {}, 'firmware'),
        'witness': _plain(witness or {}, 'witness'),
        'extra': _plain(extra or {}, 'extra'),
    }
    text = json.dumps(record, indent=2, sort_keys=True) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(path, text)
    if history:
        stamp = time.strftime(_HISTORY_STAMP, now)
        _replace_atomically(path.with_name(f"{path.stem}.{stamp}{path.suffix}"), text)
    return path


def _replace_atomically(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` so that a reader sees the old file or the new one."""
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix='.tmp', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_record(path: Union[str, Path]) -> Dict[str, Any]:
    """The record ``write_record`` wrote at ``path``.

    Parameters
    ----------
    path : str or Path
        The record file.

    Raises
    ------
    ConfigError
        If the file is missing, is not JSON, is not a configuration record, or
        holds a resolution whose hash is not that of its values.
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except OSError as e:
        raise ConfigError(str(path), reason=f"cannot read: {e.strerror or e}") from e
    except ValueError as e:
        raise ConfigError(str(path), reason=f"not valid JSON: {e}") from e
    if not isinstance(data, dict) or 'resolved' not in data:
        raise ConfigError(str(path), reason='not a configuration record: no "resolved" entry')
    # The resolution it holds has to be one: values whose hash is the recorded hash.
    try:
        Resolved.from_dict(data['resolved'])
    except (ConfigError, KeyError, TypeError, AttributeError) as e:
        raise ConfigError(str(path), key='resolved',
                          reason=f"not a resolution: {getattr(e, 'reason', None) or e}") from e
    return data
