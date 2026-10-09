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
# Publisher substitutability: the null publisher a session falls back to spells
# its parameters the way pysmurf's Publisher spells them, so a caller that passes
# them by keyword gets the same call on either. Compared by parsing both files,
# because importing pysmurf's would pull in its plotting dependencies.
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
# Where the rest of the package may import them: inside a function, so the
# package imports where rogue is not installed -- or in the one subpackage that
# exists only where a server runs, which the package does not import and a
# composition does. Named by package-relative directory: the exemption is for
# that place, not for a file name wherever it turns up.
SERVER_SIDE_PACKAGE = ('server',)
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

# The legacy client, which holds one cryodaq session and no rogue client of its
# own -- except the shelf manager's monitor, a different server, reached where named.
CLIENT = REPO / 'python' / 'pysmurf' / 'client'
ATCA_MONITOR_EXCEPTION = 'base/base_class.py:SmurfBase.__init__'


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


def module_level_imports(tree):
    """Yield (dotted module name, lineno) for every import not inside a function.

    A module-level `if`, `try` or class body still counts: none of them defers
    the import to a call, so the module still needs the package to import.
    """
    def walk(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.Import):
                for alias in child.names:
                    yield alias.name, child.lineno
            elif isinstance(child, ast.ImportFrom):
                yield child.module or '', child.lineno
            yield from walk(child)
    yield from walk(tree)


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
        if rel.parts[:1] != SERVER_SIDE_PACKAGE:
            for name, lineno in module_level_imports(tree):
                if name.split('.')[0] in MAP_FORBIDDEN_IMPORTS:
                    found.append(('rogue', f"{rel}:{lineno} imports {name} at module level"))
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


def check_rogue_is_imported_only_where_a_session_opens_or_a_server_runs():
    report('rogue')


def check_register_paths_only_in_platform():
    report('paths')


def check_register_paths_in_data_files_are_in_platform_too():
    """The rule above, applied to what the package carries that is not source.

    The register-path rule is enforced by reading string constants out of modules,
    so a map shipped as data rather than as code would satisfy it without being
    subject to it -- and a register map is exactly the kind of thing that is easier
    to ship as data. Every non-source file in the package is therefore read as text
    and held to the same rule, which is also what keeps the rule honest as the
    package grows a catalog, a schema or a fixture.
    """
    offenders = []
    for path in sorted(PACKAGE.rglob('*')):
        if path.suffix == '.py' or not path.is_file():
            continue
        if '__pycache__' in path.parts:
            continue                     # the interpreter's own compiled copies
        rel = path.relative_to(PACKAGE)
        if in_platform(rel):
            continue
        try:
            text = path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            # A file this cannot read is a file this cannot clear. The package ships
            # no binary data today, so one appearing outside platform is refused
            # rather than passed over: skipping it would be the one way to carry a
            # register map past this rule.
            offenders.append(f"{rel} is not UTF-8 text and cannot be checked for "
                             f"register paths")
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if REGISTER_PATH_RE.search(line):
                offenders.append(f"{rel}:{lineno} register path {line.strip()[:60]!r}")
                break                    # one report per file is enough to place it
    listed = '\n'.join(f"    {o}" for o in offenders)
    assert not offenders, (f"{len(offenders)} data file(s) outside cryodaq.platform "
                           f"carry a register path:\n{listed}")


def _parameter_names(tree, class_name, method):
    """The positional parameter names of one method of one class, from source."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method:
                    return [a.arg for a in item.args.args]
    raise AssertionError(f"{class_name}.{method} was not found")


def check_the_null_publisher_matches_the_real_one():
    """The stand-in publisher takes the same parameter names as the real one.

    A caller passing them by keyword -- sodetlib does, at ``util.py:85`` -- gets a
    ``TypeError`` from a stand-in that renamed them, so the names are part of the
    interface and not an implementation detail. Compared from source rather than
    by importing either one: this script runs where neither rogue nor pysmurf's
    plotting dependencies are installed.
    """
    pub = REPO / 'python' / 'pysmurf' / 'client' / 'util' / 'pub.py'
    real = ast.parse(pub.read_text())
    null = ast.parse((PACKAGE / '_session.py').read_text())
    for method in ('register_file', 'publish'):
        want = _parameter_names(real, 'Publisher', method)
        got = _parameter_names(null, 'NullPublisher', method)
        # The stand-in takes **kwargs for the rest, so what has to match is the
        # names it spells out, in order, from the front.
        assert want[:len(got)] == got, \
            (f"NullPublisher.{method}{tuple(got)} does not match "
             f"Publisher.{method}{tuple(want)}: a keyword call that works on one "
             f"raises TypeError on the other")


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
    # A named platform belongs here with the target: it is a lookup in a table
    # this package carries, so nothing about it needs a tree. The deadline is
    # deliberately not here -- its range is the transport's, so an unusable one is
    # refused by rogue rather than a second time by this package, and there is
    # nothing to see for it without rogue installed.
    cases = (('not-an-endpoint', {}, cryodaq.ConnectError, 'not-an-endpoint'),
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


def _imports_of(node):
    """The top-level module names an import statement names."""
    if isinstance(node, ast.Import):
        return [a.name.split('.')[0] for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or '').split('.')[0]]
    return []


def _client_aliases(tree):
    """Every bare name a module binds to ``VirtualClient``, wherever it binds it.

    ``from pyrogue.interfaces import VirtualClient as VC`` is a client constructed
    under another name; ``VC = pyrogue.interfaces.VirtualClient`` is the same by
    assignment. Both are read so that spelling the call differently is not a way
    past the rule.
    """
    names = {'VirtualClient'}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name == 'VirtualClient':
                    names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            value = node.value
            if ((isinstance(value, ast.Attribute) and value.attr == 'VirtualClient') or
                    (isinstance(value, ast.Name) and value.id in names)):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
    return names


def _is_virtual_client_call(node, aliases):
    """Whether a call constructs a ``VirtualClient``, however the name is reached."""
    if not isinstance(node, ast.Call):
        return False
    callee = node.func
    if isinstance(callee, ast.Attribute):
        return callee.attr == 'VirtualClient'
    return isinstance(callee, ast.Name) and callee.id in aliases


def rogue_reaches(client):
    """Where the legacy client reaches rogue directly, as ``file:line detail`` strings.

    The client connects through ``cryodaq.connect`` and reaches every register
    through the session, so a ``pyrogue`` or ``rogue`` import anywhere outside a
    function body -- at module level, under a module-level ``if``, in a ``try``
    or its handlers, in a class body -- or a ``VirtualClient`` constructed
    anywhere, is a second client stack beside the session's. The one place
    allowed to construct a client is named in ``ATCA_MONITOR_EXCEPTION``, as
    ``file:function``: the shelf manager's monitor is a different server with no
    platform map, and until it has one it is reached the old way, inside a
    function -- so the import there is inside the function too. The exemption is
    for one client: a second construction in the same function is a second stack.
    """
    found = []
    exempt = []
    for path in sorted(client.rglob('*.py')):
        rel = path.relative_to(client)
        tree = ast.parse(path.read_text(), filename=str(path))
        aliases = _client_aliases(tree)
        # Every node's enclosing function -- qualified by its class, or None -- so
        # that "inside a function" is decided by the tree and not by which statement
        # shapes were thought of, and the exemption names one method of one class.
        enclosing = {}

        def visit(node, scope, func):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.ClassDef):
                    inner_scope, inner_func = scope + [child.name], func
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    inner_scope, inner_func = scope, '.'.join(scope + [child.name])
                else:
                    inner_scope, inner_func = scope, func
                enclosing[child] = func
                visit(child, inner_scope, inner_func)
        visit(tree, [], None)

        for node, func in enclosing.items():
            for name in _imports_of(node):
                if name in MAP_FORBIDDEN_IMPORTS and func is None:
                    found.append(f"{rel}:{node.lineno} imports {name} outside any function")
            if _is_virtual_client_call(node, aliases):
                if f"{rel}:{func}" == ATCA_MONITOR_EXCEPTION:
                    exempt.append(f"{rel}:{node.lineno}")
                else:
                    found.append(f"{rel}:{node.lineno} constructs a VirtualClient in "
                                 f"{func if func is not None else 'the module body'}")
    if len(exempt) > 1:
        found.append(f"{ATCA_MONITOR_EXCEPTION} constructs {len(exempt)} VirtualClients "
                     f"({', '.join(exempt)}); the exemption is for one")
    return found


def check_the_client_has_one_rogue_stack():
    bad = rogue_reaches(CLIENT)
    if bad:
        raise AssertionError(f"{len(bad)} second-stack site(s):\n" +
                             '\n'.join(f"    {b}" for b in bad))


def check_the_one_stack_rule_fires():
    good = (
        "import cryodaq\n"
        "class SmurfBase:\n"
        "    def __init__(self, atca_monitor):\n"
        "        if atca_monitor:\n"
        "            import pyrogue.interfaces\n"
        "            self._atca = pyrogue.interfaces.VirtualClient(addr='a', port=1)\n"
    )
    # Every shape a second stack could take that is not inside a function: a
    # guarded import, a plain one, one under a module-level if, one in an except
    # handler, one in a class body; a VirtualClient constructed by attribute, by
    # bare name after a local import, at module level, and in a second __init__
    # of the file that holds the exemption.
    bad = (
        "try:\n"
        "    import pyrogue.interfaces\n"
        "except ModuleNotFoundError:\n"
        "    import rogue\n"
        "from pyrogue import VariableWait\n"
        "if True:\n"
        "    import pyrogue as pr\n"
        "class Base:\n"
        "    import rogue.interfaces\n"
        "    def connect(self):\n"
        "        self._client = pyrogue.interfaces.VirtualClient(addr='a', port=1)\n"
        "    def other(self):\n"
        "        from pyrogue.interfaces import VirtualClient\n"
        "        return VirtualClient('a', 1)\n"
        "class Second:\n"
        "    def __init__(self):\n"
        "        self.c = pyrogue.interfaces.VirtualClient(addr='a', port=1)\n"
        "top = pyrogue.interfaces.VirtualClient(addr='a', port=1)\n"
        "def aliased():\n"
        "    from pyrogue.interfaces import VirtualClient as VC\n"
        "    return VC('a', 1)\n"
        "def assigned():\n"
        "    Maker = pyrogue.interfaces.VirtualClient\n"
        "    return Maker('a', 1)\n"
    )
    with tempfile.TemporaryDirectory() as tmp:
        client = pathlib.Path(tmp)
        (client / 'base').mkdir()
        (client / 'base' / 'base_class.py').write_text(good)
        assert rogue_reaches(client) == [], \
            f"the allowed ATCA-monitor site was reported: {rogue_reaches(client)}"
        (client / 'base' / 'base_class.py').write_text(bad)
        found = rogue_reaches(client)
        details = '\n'.join(found)
        for needle in (':2 imports pyrogue outside any function',
                       ':4 imports rogue outside any function',
                       ':5 imports pyrogue outside any function',
                       ':7 imports pyrogue outside any function',
                       ':9 imports rogue outside any function',
                       ':11 constructs a VirtualClient in Base.connect',
                       ':14 constructs a VirtualClient in Base.other',
                       ':17 constructs a VirtualClient in Second.__init__',
                       ':18 constructs a VirtualClient in the module body',
                       ':21 constructs a VirtualClient in aliased',
                       ':24 constructs a VirtualClient in assigned'):
            assert needle in details, f"rule for {needle!r} did not fire:\n{details}"
        assert len(found) == 11, f"expected 11 findings, got {len(found)}:\n{details}"
        # The exemption admits one construction, not the method: a second client
        # beside the monitor's, in the very function that is allowed one, is caught.
        (client / 'base' / 'base_class.py').write_text(
            good + "        self._client = pyrogue.interfaces.VirtualClient(addr='b', port=2)\n")
        found = rogue_reaches(client)
        assert len(found) == 1 and 'constructs 2 VirtualClients' in found[0], found


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
        # A deferred import is how a module may reach rogue; one under a `try`
        # at module level is not deferred at all.
        "def g():\n"
        "    import pyrogue.interfaces\n"
        "try:\n"
        "    import rogue\n"
        "except ImportError:\n"
        "    rogue = None\n"
    )
    server_side = "import pyrogue\nclass D(pyrogue.Device):\n    pass\n"
    bad_map = (
        "import sodetlib\n"
        "import pyrogue as pr\n"
        "RATE = 614.4\n"
        "note = 'the tes bias'\n"
    )
    global PACKAGE
    saved = PACKAGE
    with tempfile.TemporaryDirectory() as tmp:
        pkg = pathlib.Path(tmp) / 'cryodaq'
        (pkg / 'platform').mkdir(parents=True)
        (pkg / '__init__.py').write_text('')
        (pkg / '_client.py').write_text(bad_client)
        # A server-side module where it belongs is exempt from the rogue rule --
        # and only from that one: the import direction still holds there.
        (pkg / 'server').mkdir()
        (pkg / 'server' / '__init__.py').write_text('')
        (pkg / 'server' / '_root.py').write_text(server_side)
        (pkg / 'server' / '_bad.py').write_text('import pysmurf\n')
        # The same content one directory over is not exempt.
        (pkg / 'other').mkdir()
        (pkg / 'other' / '__init__.py').write_text('')
        (pkg / 'other' / '_root.py').write_text(server_side)
        (pkg / 'platform' / '__init__.py').write_text('')
        (pkg / 'platform' / '_x.py').write_text(bad_map)
        # A map shipped as data outside the platform package, which the source
        # rules cannot see; the same thing in its legitimate home, which may not be
        # reported; and a binary file outside platform, which must be -- not for
        # what it holds but because it cannot be read, and a file this cannot
        # read is one it cannot clear.
        (pkg / 'smuggled.json').write_text('{"path": "AMCc.FpgaTopLevel.AppTop"}')
        (pkg / 'platform' / 'shipped.json').write_text(
            '{"path": "AMCc.FpgaTopLevel.AppTop"}')
        (pkg / 'logo.bin').write_bytes(b'\x00\xff\xfe AMCc.FpgaTopLevel')
        found = violations(pkg)

        PACKAGE = pkg
        try:
            check_register_paths_in_data_files_are_in_platform_too()
        except AssertionError as e:
            data_rule = str(e)
        else:
            data_rule = ''
        finally:
            PACKAGE = saved

    assert 'smuggled.json' in data_rule, \
        f"a register map shipped as data outside the platform package was not caught: {data_rule!r}"
    assert 'shipped.json' not in data_rule, \
        f"a data file in its legitimate home was reported: {data_rule!r}"
    assert 'logo.bin' in data_rule, \
        f"a binary file outside platform was passed over rather than refused: {data_rule!r}"

    rules = {r for r, _ in found}
    expected = {'imports', 'application', 'geometry', 'paths', 'rogue'}
    assert rules == expected, f"rules fired: {sorted(rules)}, expected {sorted(expected)}"
    details = '\n'.join(d for _, d in found)
    for needle in ('imports pysmurf', 'reaches into platform._umux',
                   'reaches into cryodaq.platform._atca', 'imports sodetlib',
                   'imports pyrogue inside cryodaq.platform',
                   "identifier 'is_rfsoc'", "identifier 'bias_group'",
                   "mentions 'tes'", 'literal 512', 'literal 614.4', 'register path',
                   '_client.py:11 imports rogue at module level',
                   '_x.py:2 imports pyrogue at module level',
                   'other/_root.py:1 imports pyrogue at module level',
                   'server/_bad.py: imports pysmurf'):
        assert needle in details, f"rule for {needle!r} did not fire:\n{details}"
    assert 'platform.parse' not in details, \
        f"the public map lookup was reported as a violation:\n{details}"
    assert 'pyrogue.interfaces' not in details, \
        f"an import deferred into a function was reported:\n{details}"
    assert 'server/_root.py' not in details, \
        f"the server-side package was reported for importing rogue:\n{details}"


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
