#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Layer Boundary Checks
#-----------------------------------------------------------------------------
# File       : check_boundaries.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
# Checks that the cryodaq package keeps to its layer boundaries. Most of them
# read source with `ast`; the last two import the package and use the part of it
# that is meant to work with nothing installed. Neither rogue nor hardware is
# needed, so this runs anywhere Python does -- and where rogue is absent it also
# shows that the package does not quietly need it.
#
# Register-path boundary: a rogue register path appears only under
# cryodaq.platform. This is the firmware/software boundary, and it is what
# "the client does not know the register map" means mechanically.
#
# Map boundary: cryodaq.platform imports neither rogue nor pyrogue. The maps are
# data and the lookup over them is arithmetic on strings; reading a tree is the
# client's job, and a map module that reached for a tree would be doing the
# client's work in the wrong layer.
#
# Import direction: nothing under cryodaq imports pysmurf, smurf or sodetlib --
# the readout core never imports an application -- and the client reaches the
# platform layer through the cryodaq.platform package, never one of its map
# modules.
#
# Runtime surface: importing the package, resolving an endpoint, building the
# capture paths and looking up a map all work with nothing installed. rogue is a
# dependency of connect() alone, and connect() judges its arguments before it
# reaches for one, so a mistyped target is reported as a mistyped target.
#
# Application boundary: the identifiers is_rfsoc, tes, bias_group and
# pA_per_phi0 do not appear anywhere under cryodaq.
#
# Geometry boundary: the channel-geometry literals of one particular firmware
# (614.4, 2.4, bare 128 and 512) do not appear anywhere under cryodaq; geometry
# is read from the hardware at run time.
#
# The last check writes a deliberately non-compliant package to a temporary
# directory and confirms every rule fires on it, so a rule that silently stops
# matching is itself a failure.
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
import ast
import importlib
import pathlib
import re
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PACKAGE = REPO / 'python' / 'cryodaq'

# Top-level modules the readout core must never import, anywhere.
APPLICATION_IMPORTS = ('pysmurf', 'smurf', 'sodetlib')
# Top-level modules the map layer must never import: it holds no tree access.
MAP_FORBIDDEN_IMPORTS = ('pyrogue', 'rogue')
# Identifiers that mark the application boundary.
APPLICATION_NAMES = ('is_rfsoc', 'tes', 'bias_group', 'pA_per_phi0')
APPLICATION_NAMES_RE = re.compile(r'\b(' + '|'.join(APPLICATION_NAMES) + r')\b')
# Channel-geometry literals of one particular firmware.
GEOMETRY_LITERALS = (614.4, 2.4, 128, 512)
# Anything that looks like a rogue register path.
REGISTER_PATH_RE = re.compile(
    r'(^|[.\s\'"])(AMCc|FpgaTopLevel|AppTop|AppCore|SysgenCryo|CryoChannels|'
    r'AmcCarrierCore|RtmCryoDet|SmurfApplication|SmurfProcessor)(\.|\[|$)')

# The package being checked; set by main() so the check_* functions take no
# arguments and can be discovered by name.
PACKAGE = DEFAULT_PACKAGE


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def modules(package):
    """Yield (relative path, parsed tree) for every module under package."""
    for path in sorted(package.rglob('*.py')):
        rel = path.relative_to(package)
        yield rel, ast.parse(path.read_text(), filename=str(path))


def in_platform(rel):
    return rel.parts[:1] == ('platform',)


def imported_modules(tree):
    """Yield (dotted module name, level) for every import in tree."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node.module or '', node.level


def imported_from(tree):
    """Yield (dotted module name, imported name, level) for every `from` import.

    The imported name matters as well as the module it came from: `from
    cryodaq.platform import _umux` reaches a map module exactly as `import
    cryodaq.platform._umux` does, and the module name alone does not show it.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                yield node.module or '', alias.name, node.level


def strings(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            yield node.lineno, node.value


def identifiers(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.lineno, node.id
        elif isinstance(node, ast.Attribute):
            yield node.lineno, node.attr
        elif isinstance(node, ast.arg):
            yield node.lineno, node.arg
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.lineno, node.arg
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.lineno, node.name


def numbers(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            yield node.lineno, node.value


def violations(package):
    """Return every boundary violation under package as (rule, 'file:line detail')."""
    found = []
    for rel, tree in modules(package):
        platform = in_platform(rel)
        for name, level in imported_modules(tree):
            top = name.split('.')[0]
            if level == 0 and top in APPLICATION_IMPORTS:
                found.append(('imports', f"{rel}: imports {name}"))
            if platform and top in MAP_FORBIDDEN_IMPORTS:
                found.append(('imports', f"{rel}: imports {name} inside cryodaq.platform"))
            # The client uses the platform package, not one of its map modules:
            # `cryodaq.platform._x` or `from .platform._x import`.
            private = (name.startswith('cryodaq.platform.') or
                       (level == 1 and name.startswith('platform.')))
            if private and not platform:
                found.append(('imports', f"{rel}: reaches into {name}"))
        for module, name, level in imported_from(tree):
            # `from cryodaq.platform import _umux`. A public name out of the
            # package is how the client is meant to reach the maps, so only a
            # private one is a violation.
            reaches_map = (module in ('cryodaq.platform', 'platform') and
                           name.startswith('_'))
            if reaches_map and not platform:
                found.append(('imports', f"{rel}: reaches into {module}.{name}"))
        for lineno, ident in identifiers(tree):
            if ident in APPLICATION_NAMES:
                found.append(('application', f"{rel}:{lineno} identifier {ident!r}"))
        for lineno, text in strings(tree):
            m = APPLICATION_NAMES_RE.search(text)
            if m:
                found.append(('application', f"{rel}:{lineno} string mentions {m.group(1)!r}"))
            if REGISTER_PATH_RE.search(text) and not platform:
                found.append(('paths', f"{rel}:{lineno} register path {text[:60]!r}"))
        for lineno, value in numbers(tree):
            if value in GEOMETRY_LITERALS:
                found.append(('geometry', f"{rel}:{lineno} literal {value}"))
    return found


def report(rule, package=None):
    bad = [detail for r, detail in violations(package or PACKAGE) if r == rule]
    if bad:
        raise AssertionError(f"{len(bad)} violation(s):\n" + '\n'.join(f"    {b}" for b in bad))


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

def check_package_present():
    files = list(PACKAGE.rglob('*.py'))
    assert (PACKAGE / '__init__.py').is_file(), f"no package at {PACKAGE}"
    assert (PACKAGE / 'platform' / '__init__.py').is_file(), "no cryodaq.platform package"
    assert files, "package has no modules"


def check_import_direction():
    report('imports')


def check_no_application_names():
    report('application')


def check_no_geometry_literals():
    report('geometry')


def check_register_paths_only_in_platform():
    report('paths')


def _imported():
    """The package under check, imported from where it is.

    Every check above reads source; the two below run it. What they are for is
    the claim the source cannot make on its own -- that the package needs
    nothing but Python until a session is opened -- and the only way to see it
    is to import the package somewhere rogue is not, which is what CI does.
    """
    if str(PACKAGE.parent) not in sys.path:
        sys.path.insert(0, str(PACKAGE.parent))
    return importlib.import_module(PACKAGE.name)


def check_the_dependency_free_surface_works():
    cryodaq = _imported()
    assert cryodaq.endpoint_of('crate:4') == ('localhost', 9012), \
        cryodaq.endpoint_of('crate:4')
    assert cryodaq.endpoint_of('a-server:9099') == ('a-server', 9099), \
        cryodaq.endpoint_of('a-server:9099')
    paths = cryodaq.Paths.under('/tmp/nowhere', name='boundaries')
    assert paths.tune.is_relative_to('/tmp/nowhere'), paths.tune
    assert cryodaq.platform.by_name('umux-atca').patterns, 'the map is empty'


def check_connect_judges_its_arguments_before_needing_rogue():
    cryodaq = _imported()
    # A named platform belongs here with the target and the deadline: it is a
    # lookup in a table this package carries, so nothing about it needs a tree.
    # The three together are every argument that can be judged without one.
    cases = (('not-an-endpoint', {}, cryodaq.ConnectError, 'not-an-endpoint'),
             ('crate:4', {'timeout': 0}, ValueError, 'timeout'),
             ('crate:4', {'platform_name': 'no-such-platform'},
              cryodaq.ConnectError, 'no-such-platform'))
    for bad, kwargs, expect, needle in cases:
        try:
            session = cryodaq.connect(bad, **kwargs)
        except expect as e:
            assert needle in str(e), f"the error does not name {needle!r}: {e}"
        except ImportError as e:                                # pragma: no cover
            raise AssertionError(
                f"rogue was needed to refuse {bad!r} {kwargs!r}: {e}") from e
        else:
            session.close()
            raise AssertionError(f"{bad!r} {kwargs!r} was accepted")


def check_rules_fire_on_a_bad_package():
    bad_client = (
        "import pysmurf\n"
        "from .platform._umux import REGISTERS\n"
        # The same reach written so that the module name alone looks public.
        "from cryodaq.platform import _atca\n"
        # Legal, and must stay legal: the public lookup is how the client is
        # meant to reach a map.
        "from cryodaq.platform import parse\n"
        "def f(is_rfsoc, x):\n"
        "    n = 512\n"
        "    return 'AMCc.FpgaTopLevel.AppTop', x.bias_group, n\n"
    )
    bad_map = (
        "import sodetlib\n"
        "import pyrogue as pr\n"
        "RATE = 614.4\n"
        "note = 'the tes bias'\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        pkg = pathlib.Path(tmp) / 'cryodaq'
        (pkg / 'platform').mkdir(parents=True)
        (pkg / '__init__.py').write_text('')
        (pkg / '_client.py').write_text(bad_client)
        (pkg / 'platform' / '__init__.py').write_text('')
        (pkg / 'platform' / '_x.py').write_text(bad_map)
        found = violations(pkg)
    rules = {r for r, _ in found}
    expected = {'imports', 'application', 'geometry', 'paths'}
    assert rules == expected, f"rules fired: {sorted(rules)}, expected {sorted(expected)}"
    details = '\n'.join(d for _, d in found)
    for needle in ('imports pysmurf', 'reaches into platform._umux',
                   'reaches into cryodaq.platform._atca', 'imports sodetlib',
                   'imports pyrogue inside cryodaq.platform',
                   "identifier 'is_rfsoc'", "identifier 'bias_group'",
                   "mentions 'tes'", 'literal 512', 'literal 614.4', 'register path'):
        assert needle in details, f"rule for {needle!r} did not fire:\n{details}"
    assert 'platform.parse' not in details, \
        f"the public map lookup was reported as a violation:\n{details}"


# --------------------------------------------------------------------------

def main():
    global PACKAGE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=pathlib.Path, default=DEFAULT_PACKAGE,
                        help='cryodaq package directory to check (default: this checkout)')
    args = parser.parse_args()
    PACKAGE = args.package.resolve()

    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []

    print(f"Checking cryodaq boundaries in {PACKAGE} ({len(checks)} checks)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                  # noqa: BLE001
            failed.append(label)
            print(f"  FAIL  {label}")
            print(f"          {type(e).__name__}: {e}")
        else:
            print(f"  ok    {label}")

    print("")
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
