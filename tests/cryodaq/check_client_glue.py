#!/usr/bin/env python3
"""Check the glue between the legacy pysmurf client and its cryodaq session.

``SmurfBase`` holds a ``cryodaq.Session``; ``_caget``, ``_caput`` and ``_wait_for``
reach the tree through it. What those three promise their callers is checked here
against a stand-in session, with no hardware and no rogue:

* a semantic name resolves through the session and a string the map does not
  know is tried as a register path, so callers outside this repository that
  still spell a path keep working until they have moved;
* a name nothing answers to raises ``ValueError`` naming it, as it always has;
* ``_wait_for`` polls with a fresh read, raises ``TimeoutError`` when its bound
  runs out, honours the bound it was given -- and, given none, **waits forever**,
  which is the behaviour it has always had and its docstring now states;
* the log handler maps every ``logging`` level onto the SmurfLogger level that
  shows it at the same verbosity, and never sends an error somewhere quiet.

The client is not imported: its dependency stack includes a plotting library
that is not installed where this runs. The class bodies under test are compiled
on their own from source, as ``check_sodetlib_contract.py`` reads them.
"""
import ast
import logging
import pathlib
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[2]
COMMAND = REPO / 'python' / 'pysmurf' / 'client' / 'command' / 'smurf_command.py'
LOGGER = REPO / 'python' / 'pysmurf' / 'client' / 'base' / 'logger.py'

sys.path.insert(0, str(REPO / 'python'))

import cryodaq  # noqa: E402
from cryodaq import platform  # noqa: E402


# --------------------------------------------------------------------------
# stand-ins
# --------------------------------------------------------------------------

class _Node:
    def __init__(self, values):
        self._values = list(values)
        self.reads = 0

    def get(self, index=-1):
        self.reads += 1
        return self._values.pop(0) if len(self._values) > 1 else self._values[0]


class _Tree:
    def __init__(self, nodes):
        self._nodes = nodes

    def getNode(self, path):
        return self._nodes.get(path)


class _Session:
    """What the glue asks of a session: ``pmap``, ``node``, ``root``."""

    def __init__(self, nodes):
        self.pmap = platform.by_name('umux-atca')
        self.root = _Tree(nodes)

    def node(self, name):
        path = self.pmap.path(name)
        node = self.root.getNode(path)
        if node is None:
            raise cryodaq.UnresolvedName(name, pattern=path, reason='not in this tree')
        return node


def _compile(path, class_name, namespace):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    body = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    rebuilt = ast.ClassDef(name=class_name, bases=[], keywords=[], body=body.body,
                           decorator_list=[], type_params=[])
    ast.copy_location(rebuilt, body)
    module = ast.Module(body=[rebuilt], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, f'<{path.name}>', 'exec'), namespace)          # noqa: S102
    return namespace[class_name]


class _Absent:
    """Stands in for a module the class body names and these checks never reach.

    The command module's methods mention ``np`` and ``version`` in their bodies,
    which the compiled class needs bound at call time and not before; the three
    accessors under test touch neither on the paths driven here, and the job this
    runs in installs nothing. An attribute reached on one is a check that
    wandered off its path, and says so.
    """

    def __init__(self, name):
        self._name = name

    def __getattr__(self, attr):
        raise AssertionError(f'a check reached {self._name}.{attr}, which is not installed here')


def glue():
    """The three accessors, on an object that has just what they read."""
    import functools
    import typing
    ns = {'np': _Absent('numpy'), 'functools': functools, 'time': time,
          'cryodaq': cryodaq, 'warnings': __import__('warnings'),
          'os': __import__('os'), 'subprocess': __import__('subprocess'),
          'Literal': typing.Literal, 'version': _Absent('packaging.version')}
    Mixin = _compile(COMMAND, 'SmurfCommandMixin', ns)

    class Client:
        offline = False
        LOG_USER = LOG_ERROR = 0
        LOG_INFO = 1
        LOG_TASK = 2
        _node = Mixin._node
        _caget = Mixin._caget
        _caput = Mixin._caput
        _wait_for = Mixin._wait_for

        def __init__(self, nodes):
            self._session = _Session(nodes)
            self.logged = []

        def log(self, msg, level=0):
            self.logged.append((msg, level))

    return Client


def handler_class():
    # logger.py imports nothing beyond the standard library, so it is imported as is.
    import importlib.util
    spec = importlib.util.spec_from_file_location('pysmurf_client_logger', LOGGER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SmurfLogHandler


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------

NAME = 'band[4].delay_us'
PATH = platform.by_name('umux-atca').path(NAME)


def check_a_name_resolves_through_the_session():
    client = glue()({PATH: _Node([7])})
    assert client._caget(NAME) == 7


def check_a_register_path_is_still_accepted_for_now():
    client = glue()({PATH: _Node([7])})
    assert client._caget(PATH) == 7, "a raw path -- sodetlib still spells some -- was refused"


def check_a_name_nothing_answers_to_raises_value_error():
    client = glue()({})
    for bad in (NAME, 'no.such.name', 'AMCc.NoSuch.Path'):
        try:
            client._caget(bad)
        except ValueError as e:
            assert bad in str(e), f"{e} does not name {bad!r}"
        else:
            raise AssertionError(f"{bad!r} was not refused")


def check_wait_for_reads_fresh_and_returns_when_the_condition_holds():
    node = _Node([1, 1, 0])
    client = glue()({PATH: node})
    client._wait_for(NAME, lambda x: x == 0, timeout=5, poll=0.001)
    assert node.reads == 3, f"expected 3 reads, one per poll; got {node.reads}"


def check_wait_for_raises_timeout_error_at_its_bound():
    client = glue()({PATH: _Node([1])})
    started = time.monotonic()
    try:
        client._wait_for(NAME, lambda x: x == 0, timeout=0.05, poll=0.01)
    except TimeoutError as e:
        assert NAME in str(e) and '0.05' in str(e), str(e)
    else:
        raise AssertionError('a wait past its bound did not raise')
    elapsed = time.monotonic() - started
    assert 0.04 <= elapsed < 1.0, f"the bound was not honoured: {elapsed:.3f} s"


def check_wait_for_without_a_bound_is_documented_as_unbounded():
    import inspect
    doc = inspect.getdoc(glue()._wait_for) or ''
    assert 'None waits forever' in doc, \
        "the unbounded default is a hazard callers have to be told about; say so in the docstring"
    # And it is unbounded in fact: a wait with no timeout does not raise on its own.
    # Proven without waiting forever by counting polls against a value that never
    # satisfies the condition until the stand-in flips it.
    node = _Node([1] * 30 + [0])
    client = glue()({PATH: node})
    client._wait_for(NAME, lambda x: x == 0, poll=0.0)
    assert node.reads == 31, node.reads


def check_the_log_handler_maps_levels_the_right_way_round():
    Handler = handler_class()
    seen = []
    handler = Handler(lambda msg, level: seen.append((msg, level)),
                      {'user': 0, 'error': 0, 'info': 1, 'task': 2})
    expected = {logging.CRITICAL: 0, logging.ERROR: 0, logging.WARNING: 0,
                cryodaq.LOG_USER: 0, logging.INFO: 1, logging.DEBUG: 2}
    for levelno, want in expected.items():
        got = handler.translate(levelno)
        assert got == want, f"logging level {levelno} -> SmurfLogger {got}, expected {want}"
    logger = logging.getLogger('check_client_glue')
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.error('e')
    logger.log(cryodaq.LOG_USER, 'u')
    logger.info('i')
    logger.debug('d')
    assert seen == [('e', 0), ('u', 0), ('i', 1), ('d', 2)], seen


# --------------------------------------------------------------------------

def main():
    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items() if name.startswith('check_'))
    failed = []
    print(f"Checking the client glue over the session ({len(checks)} checks)")
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
