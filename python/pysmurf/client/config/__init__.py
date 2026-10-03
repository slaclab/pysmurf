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
from typing import Any, Dict, Union

from cryodaq import config as _config

from pysmurf.client.config import legacy, schema

__all__ = ['load', 'load_mapping', 'adopt', 'DEFAULT', 'LEGACY_SUFFIXES', 'legacy', 'schema']

# The default layer shipped with the package.
DEFAULT = Path(__file__).with_name('default.yaml')

# What a legacy file is called. Anything else is read as YAML.
LEGACY_SUFFIXES = ('.cfg', '.json')


def load(path: Union[str, Path]) -> _config.Resolved:
    """Resolve a pysmurf configuration file over the packaged default.

    Parameters
    ----------
    path : str or Path
        A YAML file, possibly inheriting others, or a legacy ``.cfg``.

    Returns
    -------
    cryodaq.Resolved
        Validated values with per-key provenance; ``values['bands']`` is keyed
        by ``int`` and has ``band_default`` applied.

    Raises
    ------
    cryodaq.ConfigError
        For a fault in the layering.
    pysmurf.client.config.schema.ConfigInvalid
        For a value pysmurf cannot use, naming its key.
    """
    path = Path(path)
    if path.suffix in LEGACY_SUFFIXES:
        # The converter warns once, naming the file and what it dropped.
        return load_mapping(legacy.convert(path), name=str(path.resolve()))
    return _attribute(_config.load(path, default=DEFAULT, validate=schema.validate))


def _attribute(resolved: _config.Resolved) -> _config.Resolved:
    """Credit a per-band value the validator copied from ``band_default`` to the line that set it.

    The core attributes whatever the validator filled in to ``VALIDATED_LAYER``;
    for pysmurf most of that is ``band_default`` applied under each band, and
    the file and line that set the default is the better answer.
    """
    provenance = dict(resolved.provenance)
    for key, where in resolved.provenance.items():
        if where[0] != _config.VALIDATED_LAYER or not key.startswith('bands.'):
            continue
        _, _, leaf = key.split('.', 2)
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
        What the provenance calls this layer in place of a file path.
    """
    import tempfile
    # The loader reads files, so a mapping goes through one; its provenance
    # then names `name` rather than the temporary path.
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        f.write(legacy.to_yaml(values))
    try:
        resolved = _config.load(f.name, default=DEFAULT, validate=schema.validate)
    finally:
        Path(f.name).unlink(missing_ok=True)
    # Lines in the temporary file mean nothing to a reader; a converted layer
    # is named without one.
    temp = str(Path(f.name).resolve())
    provenance = {k: ((name, 0) if v[0] == temp else v)
                  for k, v in resolved.provenance.items()}
    layers = tuple(name if layer == temp else layer for layer in resolved.layers)
    return _attribute(_config.Resolved(resolved.values, provenance, resolved.hash, layers))


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
