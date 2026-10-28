#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf Client Configuration
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
#    The pysmurf client's configuration: a YAML file over the packaged
#    default.yaml, resolved by cryodaq.config with this package's schema as the
#    judge of what the keys mean. A file may build on others with `inherit:`.
#    The legacy JSON `.cfg` files are still accepted, converted in memory with
#    a deprecation warning; `python -m pysmurf.client.config convert` writes
#    the YAML once.
#
#        from pysmurf.client.config import load
#        cfg = load('experiment.yaml')      # a cryodaq.Resolved
#        cfg.values['wiring']['R_sh']
#        cfg.provenance['wiring.R_sh']      # (file, line) that set it
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Dict, Optional, Union

from cryodaq import config as _config

from pysmurf.client.config import legacy, schema

__all__ = ['load', 'load_mapping', 'adopt', 'DEFAULT', 'LEGACY_SUFFIXES', 'legacy', 'schema']

# The default layer shipped with the package.
DEFAULT = Path(__file__).with_name('default.yaml')

# What a legacy file is called. Anything else is read as YAML.
LEGACY_SUFFIXES = ('.cfg', '.json')


def load(path: Union[str, Path], *,
         measured_from: Optional[Union[str, Path]] = None) -> _config.Resolved:
    """Resolve a pysmurf configuration file over the packaged default.

    Parameters
    ----------
    path : str or Path
        A YAML file, possibly inheriting others, or a legacy ``.cfg``.
    measured_from : str or Path, optional
        A configuration record -- the one ``setup()`` and the measuring
        operations keep under the status directory, or a dated copy beside
        it. The values operations measured are taken out of it and laid over
        the file, so a system is configured again with yesterday's measured
        delays over today's files; every such key names the record as its
        provenance, every other keeps the file's, and the record is the last
        of ``layers``. The file itself is not changed. A record with nothing
        measured adds nothing.

    Returns
    -------
    cryodaq.Resolved
        Validated values with per-key provenance; ``values['bands']`` is keyed
        by ``int`` and has ``band_default`` applied.

    Raises
    ------
    cryodaq.ConfigError
        For a fault in the layering, or a record that will not read.
    pysmurf.client.config.schema.ConfigInvalid
        For a value pysmurf cannot use, naming its key.
    """
    path = Path(path)
    if measured_from is not None:
        record = Path(measured_from)
        layer = _config.measured_layer(_config.read_record(record))
        if path.suffix in LEGACY_SUFFIXES:
            below = legacy.convert(path)
            name = str(path.resolve())
        else:
            below, name = path, None
        # The record's measured keys are the top layer; the file (or its
        # converted mapping) is what they stand on, resolved and validated
        # first so its numbered tables are keyed as the measured layer's are
        # and every key it set keeps its file and line.
        files = _attribute(_config.load(below, default=DEFAULT, validate=schema.validate, name=name))
        return _config.load(layer, default=files, validate=schema.validate, name=str(record.resolve()))
    if path.suffix in LEGACY_SUFFIXES:
        # The converter warns once, naming the file and what it dropped.
        return load_mapping(legacy.convert(path), name=str(path.resolve()))
    return _attribute(_config.load(path, default=DEFAULT, validate=schema.validate))


def _attribute(resolved: _config.Resolved) -> _config.Resolved:
    """Credit a per-band value the validator copied from ``band_default`` to the line that set it.

    The core attributes whatever the validator filled in to ``VALIDATED_LAYER``;
    for pysmurf most of that is ``band_default`` applied under each band, and
    the file and line that set the default is the better answer. Only when the
    band gave no part of that key, though: ``band_default`` applies key by key,
    so a band's own block for a key replaces the default's whole, and what the
    schema then fills in under it is the schema's, not the default's line.
    """
    provenance = dict(resolved.provenance)
    for key, where in resolved.provenance.items():
        if where[0] != _config.VALIDATED_LAYER or not key.startswith('bands.'):
            continue
        _, band, leaf = key.split('.', 2)
        top = leaf.split('.', 1)[0]
        if any(k == f'bands.{band}.{top}' or k.startswith(f'bands.{band}.{top}.')
               for k, w in resolved.provenance.items() if w[0] != _config.VALIDATED_LAYER):
            continue
        default = provenance.get(f'band_default.{leaf}')
        if default is not None:
            provenance[key] = default
    return _config.Resolved(resolved.values, provenance, resolved.hash, resolved.layers)


def load_mapping(values: Dict[str, Any], *, name: str = 'in-memory') -> _config.Resolved:
    """Resolve an already-read mapping over the packaged default, as ``load`` would a file.

    Parameters
    ----------
    values : dict
        A mapping in the schema's shape -- a converted legacy file, or one
        built in code.
    name : str
        What the provenance and any refusal call this layer in place of a file
        path; the legacy file's own path, for a converted one.
    """
    return _attribute(_config.load(values, default=DEFAULT, validate=schema.validate, name=name))


def adopt(resolved: _config.Resolved) -> _config.Resolved:
    """A resolution read back from the server, in the shape ``load`` gives.

    Parameters
    ----------
    resolved : cryodaq.Resolved
        As ``Resolved.from_dict`` rebuilt it from the server's record.

    Raises
    ------
    cryodaq.ConfigError
        If re-validation changed the hash: the record would not be the one
        that was written.

    A record travels as JSON, which has no integer keys: the bands come back as
    ``{'4': ...}`` where ``load`` gave ``{4: ...}``, and so do the wiring tables.
    Re-validating restores the types the schema declares. The hash is of the
    canonical form and does not change; that is asserted, since a record whose
    hash moved on re-validation would not be the record that was published.
    """
    values = schema.validate(dict(resolved.values))
    again = _config.Resolved(values, resolved.provenance, _config.hash_of(values), resolved.layers)
    if again.hash != resolved.hash:
        raise _config.ConfigError('<record>', reason=f"re-validation changed the hash "
                                  f"({resolved.hash[:12]} -> {again.hash[:12]})")
    return again
