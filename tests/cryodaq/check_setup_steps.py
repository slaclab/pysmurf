#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Configuration Procedure Checks
#-----------------------------------------------------------------------------
# File       : check_setup_steps.py
# Created    : 2026-10-20
#-----------------------------------------------------------------------------
# Description:
# Checks of the server's configuration procedure without rogue or hardware:
# every platform has a procedure, every node a procedure reaches is on that
# platform's committed tree dump, the register sequence each platform runs is
# the one expected, and the link check judges lanes as the hardware has been
# witnessed to.
#
# The procedures run against a stand-in tree built from a fixture dump, shaped
# as rogue shapes one (attribute access, indexed children as dictionaries) and
# recording every read, write and call by path, with rogue's layer loader stood
# in for. So a procedure that reaches for a device a platform does not have
# fails here, on the dump, rather than on the crate.
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
import logging
import os
import pathlib
import re
import sys
import types

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'python'))

from cryodaq import platform                                        # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
PLATFORM_OF = {'atca': 'umux-atca', 'rfsoc': 'umux-rfsoc', 'rfmc': 'umux-rfmc'}
_INDEXED = re.compile(r'^(\w+)\[(\d+)\]$')


def setup_module():
    """The procedure module, imported with rogue stood in for.

    The module imports pyrogue to subclass its Process; the steps do not use
    it. Where rogue is not installed a stand-in module with the names the
    class statement needs is put in its place, and the module is loaded by
    file so that the server package's own import of the rest of rogue is not
    triggered.
    """
    if 'cryodaq.server._setup' in sys.modules:
        return sys.modules['cryodaq.server._setup']
    try:
        import pyrogue                                                # noqa: F401
    except ImportError:
        fake = types.ModuleType('pyrogue')

        class Process:
            def __init__(self, *a, **k):
                pass

        def yamlToData(stream='', fName=None):
            # Enough of rogue's reader for the top-level keys the check reads.
            import yaml
            with open(fName) as fh:
                return yaml.safe_load(fh)

        fake.Process = Process
        fake.yamlToData = yamlToData
        sys.modules['pyrogue'] = fake
    import importlib.util
    path = HERE.parent.parent / 'python' / 'cryodaq' / 'server' / '_setup.py'
    spec = importlib.util.spec_from_file_location('cryodaq.server._setup', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['cryodaq.server._setup'] = module
    spec.loader.exec_module(module)
    return module


def load_dump(stem):
    with open(FIXTURES / f'{stem}.varlist.txt', 'rt', encoding='utf-8', errors='replace') as fh:
        paths = {line.split('\t', 1)[0].strip() for line in fh if line.strip()}
    paths.discard('Path')
    return paths


class Node:
    """One node of the stand-in tree: its children by name, indexed children as dicts, a value.

    Shaped as rogue's nodes are reached from code: ``node.child``,
    ``node.array[3]``, ``node.array.items()``; a leaf is read with ``value()``
    or ``get()``, written with ``set()``/``setDisp()``, and a command is called.
    Every read, write and call is appended to the tree's log by path.
    """

    def __init__(self, tree, path):
        self.tree = tree
        self.path = path
        self.children = {}
        self.arrays = {}
        self.enum = None

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)
        if name in self.children:
            return self.children[name]
        if name in self.arrays:
            return self.arrays[name]
        raise AttributeError(f"{self.path} has no child {name}")

    def _value(self):
        return self.tree.values.get(self.path, 0)

    def get(self):
        self.tree.log.append(('get', self.path))
        return self._value()

    def value(self):
        return self._value()

    def set(self, value):
        self.tree.log.append(('set', self.path, value))
        self.tree.values[self.path] = value

    def setDisp(self, value):
        self.tree.log.append(('set', self.path, value))
        self.tree.values[self.path] = value

    def __call__(self, *args):
        self.tree.log.append(('call', self.path) + args)
        return None


class StandInRoot(Node):
    """A tree whose nodes are the paths in a fixture dump, recording what is done to them.

    Values read back as 0 unless the check sets them, which is what emulated
    memory reads as.
    """

    def __init__(self, paths, values=None, pmap=None):
        Node.__init__(self, self, 'AMCc')
        self.name = 'AMCc'
        self.pmap = pmap
        self.values = dict(values or {})
        self.log = []
        self.by_path = {'AMCc': self}
        for path in sorted(paths):
            self._grow(path)

    def _grow(self, path):
        parts = path.split('.')
        if parts[0] != 'AMCc':
            return
        node = self
        for depth in range(1, len(parts)):
            here = '.'.join(parts[:depth + 1])
            if here in self.by_path:
                node = self.by_path[here]
                continue
            child = Node(self, here)
            m = _INDEXED.match(parts[depth])
            if m:
                node.arrays.setdefault(m.group(1), {})[int(m.group(2))] = child
            else:
                node.children[parts[depth]] = child
            self.by_path[here] = child
            node = child

    def getNode(self, path):
        return self.by_path.get(path)


class RfmcCarrierCore(Node):
    """The RFMC's carrier core stood in for: the three bring-up helpers its class adds.

    They are methods of the firmware's device, not nodes, so the stand-in tree
    grows them here: each records its call and answers as the hardware helper
    does -- a reset wait returns whether it timed out, judged by a value the
    check may set; the clock init raises if told to.
    """

    def UserRst(self, timeout=0):
        self.tree.log.append(('call', f'{self.path}.UserRst', timeout))
        return bool(self.tree.values.get('user reset times out'))

    def DspRstWait(self, timeout=0):
        self.tree.log.append(('call', f'{self.path}.DspRstWait', timeout))
        return bool(self.tree.values.get('dsp reset times out'))

    def InitClock(self, lmkConfig=None):
        self.tree.log.append(('call', f'{self.path}.InitClock', lmkConfig))
        if self.tree.values.get('clocks fail'):
            raise RuntimeError('PLL not locked: Lmx[1]')


def tree(stem, values=None):
    _setup = setup_module()
    pmap = platform.by_name(PLATFORM_OF[stem])
    root = StandInRoot(load_dump(stem), values, pmap)
    if pmap.rfdc_path:
        # The RF data converter is not in a dump of the firmware package (it is a
        # device the server adds, over its own link), so the stand-in grows it:
        # its init command and the multi-tile sync block the procedure drives.
        for leaf in ('Init', 'Mts.DacTiles', 'Mts.AdcTiles', 'Mts.DacRefTile', 'Mts.AdcRefTile',
                     'Mts.SyncDacTiles', 'Mts.SyncAdcTiles'):
            root._grow(f'{pmap.rfdc_path}.{leaf}')
        # The carrier core's bring-up helpers are methods of its class.
        core = root.by_path[platform.RFMC_CARRIER_PATH]
        core.__class__ = RfmcCarrierCore
    return _setup, root


class StoodInLoad:
    """rogue's loader stood in for: records each load, and raises if told to."""

    def __init__(self, _setup, fail=False):
        self._setup = _setup
        self.fail = fail
        self.loads = []

    def __enter__(self):
        self.saved = self._setup.load_layers

        def load(root, layers):
            if not layers:
                raise self._setup.StepError('no register layers were given')
            self.loads.append(tuple(layers))
            # Into the tree's log too, so a check can see where the load fell
            # among the reads and writes around it.
            root.log.append(('load',) + tuple(layers))
            if self.fail:
                raise self._setup.StepError('stood-in load failed')
        self._setup.load_layers = load
        return self

    def __exit__(self, *exc):
        self._setup.load_layers = self.saved


class Flag:
    """A status variable stood in for: records every value set on it."""

    def __init__(self, value=0):
        self.values = []
        self._value = value

    def set(self, v):
        self.values.append(v)
        self._value = v

    def value(self):
        return self._value


class Status:
    """The application status block stood in for."""

    def __init__(self):
        self.ConfiguringInProgress = Flag()
        self.SystemConfigured = Flag()
        self.JesdStatus = Flag()


def procedure(_setup, stem, layers=('a.yml',)):
    """A platform's procedure object, built without rogue."""
    cls = _setup.SETUPS[PLATFORM_OF[stem]]
    setup = cls.__new__(cls)
    setup.layers = tuple(layers)
    setup.status = Status()
    if cls is _setup.RfmcSetup:
        setup.lmk_config = '<lmk file>'
        # Polling is this root's variable; the stand-in tree has rogue's leaves.
    # rogue's step counters, stood in for.
    setup.progress = []
    setup.setTotalSteps = lambda n: setup.progress.append(('total', n))
    setup.incrementSteps = lambda: setup.progress.append('step')
    return setup


def run_procedure(stem, layers=('a.yml',), values=None):
    """A platform's whole procedure, with the layer load stood in; the recorded log and the loads."""
    _setup, root = tree(stem, values)
    with StoodInLoad(_setup) as load:
        procedure(_setup, stem, layers).run(root)
    return root.log, load.loads


def aligned_lanes(values):
    """Every lane of both bays reading what an aligned link reads."""
    for bay in (0, 1):
        for lane in range(10):
            values[f'{JESD_BAY(bay)}.JesdRx.ElBuffLatency[{lane}]'] = 255 if lane in (2, 3) else 13
    return values


def JESD_BAY(bay):
    return platform.CARRIER_LINKS_PATH.format(bay=bay)


# --------------------------------------------------------------------------

def check_every_platform_has_a_procedure():
    _setup = setup_module()
    for pmap in platform.MAPS:
        assert pmap.name in _setup.SETUPS, f"{pmap.name} has no configuration procedure"
        assert issubclass(_setup.SETUPS[pmap.name], _setup.Setup)
    try:
        _setup.setup_for(platform.by_name('umux-atca').__class__(
            name='no-such', tags=(), registers={}, witness=(), scopes={}), (), None)
    except ValueError as e:
        assert 'no-such' in str(e) and 'umux-atca' in str(e), e
    else:
        raise AssertionError('a platform with no procedure was accepted')


def check_every_procedure_reaches_only_nodes_its_platform_has():
    """Run on each platform's dump, no procedure asks for a node the tree lacks."""
    for stem in PLATFORM_OF:
        values = aligned_lanes({}) if stem == 'atca' else None
        log, loads = run_procedure(stem, values=values)
        assert log, f"{stem}: the procedure touched nothing"
        assert loads == [('a.yml',)], loads


def check_the_rfmc_sequence_is_the_firmwares_own():
    """The RFMC's steps, in the order its firmware's root brings the board up.

    The expected side is the firmware root's `start()`, transcribed from the
    release (`rfmc_cryo_det/_Root.py`): user reset, timing route, clock chips
    from the vendor file, DSP clock wait, converter init, a read of the tree,
    tile sync, then the configuration load and a counter reset -- with every
    wait bounded, the backplane routed to the receiver's crossbar output, the
    tile groups the firmware uses, and polling off throughout. The firmware's
    final health check is left out, as the carrier's was.
    """
    _setup, root = tree('rfmc', {'AMCc.PollEn': True})
    setup = procedure(_setup, 'rfmc', layers=('d.yml',))
    setup.lmk_config = '/pkg/lmk.txt'      # over the stand-in's
    with StoodInLoad(_setup) as load:
        setup.run(root)
    core = platform.RFMC_CARRIER_PATH
    rfdc = root.pmap.rfdc_path
    crossbar = f'{core}.AxiSy56040.OutputConfig[{platform.RFMC_TIMING_RECEIVER_CROSSBAR_OUTPUT}]'
    expected = [
        ('set', 'AMCc.PollEn', False),
        ('call', f'{core}.UserRst', _setup.RESET_RECOVERY_TIMEOUT_S),
        ('set', crossbar, 0x2),                                   # backplane, STARTUP_TIMING in the firmware
        ('call', f'{core}.InitClock', '/pkg/lmk.txt'),
        ('call', f'{core}.DspRstWait', _setup.RESET_RECOVERY_TIMEOUT_S),
        ('call', f'{rfdc}.Init'),
        ('call', 'AMCc.ReadAll'),
        ('set', f'{rfdc}.Mts.DacTiles', 0xF),
        ('set', f'{rfdc}.Mts.AdcTiles', 0xF),
        ('set', f'{rfdc}.Mts.DacRefTile', 0x0),                  # DAC228, the DAC clock source
        ('set', f'{rfdc}.Mts.AdcRefTile', 0x2),                  # ADC226, the ADC clock source
        ('call', f'{rfdc}.Mts.SyncDacTiles'),
        ('call', f'{rfdc}.Mts.SyncAdcTiles'),
        ('load', 'd.yml'),
        ('call', 'AMCc.CountReset'),
        ('set', 'AMCc.PollEn', True),
    ]
    assert root.log == expected, '\n'.join(f'  {e}' for e in root.log)
    assert load.loads == [('d.yml',)]
    assert setup.progress == [('total', 9)] + ['step'] * 9, setup.progress
    # The RFMC has no JESD links and never touches a verdict.
    assert setup.status.JesdStatus.values == []
    # A reset that does not come back, a clock that does not lock, and no clock
    # file are each the named step's failure, with the firmware's reason where it
    # gives one; polling is restored on the way out of a failed run too.
    for values, step, reason in (({'user reset times out': True}, 'reset user logic', 'did not come out of reset'),
                                 ({'dsp reset times out': True}, 'wait for DSP clock', 'did not come out of reset'),
                                 ({'clocks fail': True}, 'init clocks', 'PLL not locked: Lmx[1]'),
                                 ({}, 'init clocks', 'no clock chip register file')):
        _setup, root = tree('rfmc', dict(values, **{'AMCc.PollEn': True}))
        setup = procedure(_setup, 'rfmc')
        if 'no clock' in reason:
            setup.lmk_config = None
        try:
            with StoodInLoad(_setup):
                setup.run(root)
        except _setup.StepError as e:
            assert str(e).startswith(f'{step}: ') and reason in str(e), e
        else:
            raise AssertionError(f'{step} did not fail')
        assert not [e for e in root.log if e[0] == 'load'], f'{step} failed and the layers were still loaded'
        assert root.log[-1] == ('set', 'AMCc.PollEn', True), f'polling was not restored after {step} failed'


def check_the_carrier_sequence_is_the_witnessed_one():
    """The carrier's steps, in order: read, converters reset, layers loaded, links checked."""
    log, loads = run_procedure('atca', layers=('d.yml', 's.yml'), values=aligned_lanes({}))
    calls = [e[1] for e in log if e[0] == 'call']
    assert calls[0] == 'AMCc.ReadAll', calls[:1]
    # Converter cards reset: the debug block on, then every DAC's reset 1 then 0.
    enables = [e for e in log if e[0] == 'set' and e[1].endswith('DBG.enable')]
    assert len(enables) == 2 and all(e[2] is True for e in enables), enables
    dac_resets = [e for e in log if e[0] == 'set' and 'dacReset' in e[1]]
    assert [e[2] for e in dac_resets] == [1] * 4 + [0] * 4, dac_resets
    # Both layers in one load, after the resets; the links read after the load.
    assert loads == [('d.yml', 's.yml')], loads
    sets = [e for e in log if e[0] == 'set']
    assert sets[-1] == dac_resets[-1], f"something was written after the converter resets: {sets[-1]}"
    reads = [c for c in calls if c.endswith('JesdRx.ReadDevice')]
    assert len(reads) == 2, f"{len(reads)} device reads for two bays"
    # The order, as a sequence of kinds: read, every reset write, the one load, the link reads.
    kinds = [e[0] if e[0] != 'call' else ('read_all' if e[1] == 'AMCc.ReadAll' else 'link_read')
             for e in log]
    expected = ['read_all'] + ['set'] * (len(enables) + len(dac_resets)) + ['load'] + ['link_read'] * 2
    assert kinds == expected, kinds


def check_the_carrier_skips_an_unpopulated_bay():
    """A tree the composition built with one bay's front end left out is reset and checked on the other."""
    paths = {p for p in load_dump('atca') if 'MicrowaveMuxCore[1]' not in p and 'AppTopJesd[1]' not in p}
    _setup, root = tree('atca')
    root = StandInRoot(paths, aligned_lanes({}))
    with StoodInLoad(_setup) as load:
        procedure(_setup, 'atca').run(root)
    touched = {e[1] for e in root.log if e[0] in ('set', 'call')}
    # Bay 1's devices are MicrowaveMuxCore[1] and AppTopJesd[1]; DAC 1 of bay 0 is not.
    assert not [p for p in touched if '[1].' in p], touched
    assert [e for e in root.log if e[0] == 'set' and 'dacReset' in e[1]], 'bay 0 was not reset'
    assert load.loads == [('a.yml',)]


def check_the_platform_without_links_loads_the_layers_alone():
    log, loads = run_procedure('rfsoc')
    assert loads == [('a.yml',)], loads
    calls = [e[1] for e in log if e[0] == 'call']
    assert calls == ['AMCc.ReadAll'], calls
    assert not [e for e in log if e[0] == 'set'], 'the platform without converter cards wrote a register'


def check_the_link_check_judges_lanes_as_witnessed():
    """A lane outside its values reloads and, still bad, fails; aligned lanes pass without a reload."""
    _setup, root = tree('atca', aligned_lanes({}))
    expected = _setup.AtcaSetup.ELASTIC_BUFFER_LATENCY
    assert set(expected) == set(range(10))
    assert expected[2] == expected[3] == (255,)
    assert all(expected[lane] == (13, 14) for lane in (0, 1, 4, 5, 6, 7, 8, 9))
    root.values[f'{JESD_BAY(0)}.JesdRx.ElBuffLatency[5]'] = 7
    setup = procedure(_setup, 'atca')
    saved_retries = _setup.ELASTIC_BUFFER_RETRIES
    _setup.ELASTIC_BUFFER_RETRIES = 2
    try:
        with StoodInLoad(_setup) as load:
            try:
                setup.check_elastic_buffers(root)
            except _setup.StepError as e:
                assert 'elastic buffer' in str(e), e
            else:
                raise AssertionError('a lane at 7 passed the elastic buffer check')
        assert load.loads == [('a.yml',)], f"the check reloaded {len(load.loads)} times for 2 tries"
        reads = [e for e in root.log if e[0] == 'call' and e[1].endswith('ReadDevice')]
        assert len(reads) == 4, f"{len(reads)} device reads for two bays over two tries"
    finally:
        _setup.ELASTIC_BUFFER_RETRIES = saved_retries
    _setup, root = tree('atca', aligned_lanes({}))
    with StoodInLoad(_setup) as load:
        setup.check_elastic_buffers(root)
    assert not load.loads, 'aligned lanes caused a reload'


def check_a_layer_written_for_another_root_is_refused():
    """A layer whose top-level key is not this root's name would apply nothing and be credited; it is refused first.

    Rogue logs ``Entry … not found`` for such a key and the load still ends
    ``Done``. A firmware package's own defaults are keyed for its root's name,
    which is not this server's, so the case is real.
    """
    import tempfile
    _setup, root = tree('rfsoc')
    with tempfile.TemporaryDirectory() as d:
        theirs = pathlib.Path(d, 'theirs.yml')
        theirs.write_text('Root:\n  FpgaTopLevel:\n    AppTop:\n      enable: True\n')
        ours = pathlib.Path(d, 'ours.yml')
        ours.write_text('AMCc:\n  FpgaTopLevel:\n    AppTop:\n      enable: True\n')
        try:
            _setup.check_layer_names_this_root(root, str(theirs))
        except _setup.StepError as e:
            assert "'Root'" in str(e) and "'AMCc'" in str(e), e
        else:
            raise AssertionError('a layer keyed for another root was accepted')
        _setup.check_layer_names_this_root(root, str(ours))
        try:
            _setup.check_layer_names_this_root(root, str(pathlib.Path(d, 'missing.yml')))
        except _setup.StepError as e:
            assert 'cannot be read' in str(e), e
        else:
            raise AssertionError('a missing layer was accepted')
        # The load runs the check before touching the loader.
        root.LoadConfigProcess = StoodInLoader(['Done'])
        try:
            _setup.load_layers(root, (str(theirs),))
        except _setup.StepError:
            pass
        else:
            raise AssertionError('load_layers loaded a layer keyed for another root')
        assert root.LoadConfigProcess.started == [], 'the loader was started for a refused layer'


def check_a_load_with_no_layers_is_refused():
    """No layers: the load step refuses, and the whole procedure refuses before touching the hardware."""
    _setup, root = tree('atca')
    try:
        _setup.load_layers(root, ())
    except _setup.StepError as e:
        assert 'no register layers' in str(e), e
    else:
        raise AssertionError('a load with no layers ran')
    for stem in PLATFORM_OF:
        _setup, root = tree(stem, aligned_lanes({}) if stem == 'atca' else None)
        try:
            with StoodInLoad(_setup):
                procedure(_setup, stem, layers=()).run(root)
        except _setup.StepError as e:
            assert str(e) == 'load layers: no register layers were given', e
        else:
            raise AssertionError(f'{stem}: a procedure with no layers ran')
        assert root.log == [], f"{stem}: a procedure with no layers touched the tree: {root.log[:3]}"


class StoodInLoader:
    """rogue's ``LoadConfigProcess`` stood in for: each ``Start`` ends at once with the next message given."""

    def __init__(self, messages):
        self._messages = list(messages)
        self.started = []

        class Var:
            def __init__(self, outer, name):
                self.outer, self.name, self.v = outer, name, ''

            def set(self, v):
                self.v = v

            def setDisp(self, v):
                self.v = v

            def value(self):
                return self.v

        self.UpdatePeriod, self.LoadMode, self.ConfigFile, self.Message = (Var(self, n) for n in 'UPLM')
        self.Running = Var(self, 'R')
        self.Running.v = False

    def Start(self):
        self.started.append(self.ConfigFile.v)
        self.Message.v = self._messages.pop(0)


def check_a_load_that_keeps_failing_reports_the_loaders_last_message():
    """Every try judged by the message the loader leaves; the failure carries the last one, a Done returns."""
    import tempfile
    _setup, root = tree('atca')
    with tempfile.TemporaryDirectory() as tmp:
        d, s_ = (str(pathlib.Path(tmp, n)) for n in ('d.yml', 's.yml'))
        for f in (d, s_):
            pathlib.Path(f).write_text('AMCc:\n  enable: True\n')
        root.LoadConfigProcess = StoodInLoader(['Stopped after error!'] * 3 + ['Error: [Errno 2] No such file: s.yml'])
        try:
            _setup.load_layers(root, (d, s_))
        except _setup.StepError as e:
            assert str(e).startswith(f"the register layers did not load in {_setup.LOAD_RETRIES} tries"), e
            assert "No such file: s.yml" in str(e), f"the loader's last message is not in the failure: {e}"
        else:
            raise AssertionError('four failed tries were credited')
        assert root.LoadConfigProcess.started == [f'{d},{s_}'] * _setup.LOAD_RETRIES
        root.LoadConfigProcess = StoodInLoader(['Stopped after error!', 'Done'])
        _setup.load_layers(root, (d,))
        assert root.LoadConfigProcess.started == [d, d], 'a Done on the second try did not end the load'


def check_a_failing_step_ends_the_procedure_unconfigured():
    """A step that raises leaves SystemConfigured false and its message in Message; the links' verdict follows."""
    _setup, root = tree('atca', aligned_lanes({}))

    order = []

    class Message:
        text = None

        def setDisp(self, v):
            Message.text = v
            if v.startswith('Error') or v == 'Done':
                order.append('message')

    setup = procedure(_setup, 'atca')
    status = setup.status
    status.ConfiguringInProgress.set = lambda v: (status.ConfiguringInProgress.values.append(v),
                                                  order.append('in_progress_false') if v is False else None)
    setup.Message = Message()
    setup.setProgress = lambda p: None
    # rogue's Node reads `root` from `_root`; the stand-in class has neither.
    try:
        setup.root = root
    except AttributeError:
        setup._root = root
    with StoodInLoad(_setup, fail=True):
        setup._process()
    assert status.SystemConfigured.values == [False, False], status.SystemConfigured.values
    assert status.ConfiguringInProgress.values == [True, False]
    # The carrier reports its links unlocked when a step fails, locked when the run ends well.
    assert status.JesdStatus.values == [_setup.JESD_UNLOCKED], status.JesdStatus.values
    # The message names the step that failed as well as why.
    assert Message.text == 'Error: load layers: stood-in load failed', Message.text
    # The message is published before the in-progress flag drops, which is what
    # a client waiting on the flag then reads.
    assert order.index('message') < order.index('in_progress_false'), order
    with StoodInLoad(_setup):
        setup._process()
    assert status.SystemConfigured.values[-1] is True
    assert status.JesdStatus.values == [_setup.JESD_UNLOCKED, _setup.JESD_LOCKED], status.JesdStatus.values
    assert Message.text == 'Done', Message.text
    # A read that fails on a rerun after a good one reports the links unlocked, not the earlier Locked.
    again = procedure(_setup, 'atca')
    good = tree('atca', aligned_lanes({}))[1]
    with StoodInLoad(_setup):
        again.run(good)
    bad = tree('atca', aligned_lanes({}))[1]
    bad.ReadAll = lambda: (_ for _ in ()).throw(TimeoutError('register transaction timed out'))
    try:
        with StoodInLoad(_setup):
            again.run(bad)
    except _setup.StepError as e:
        assert str(e) == 'read all: register transaction timed out', e
    else:
        raise AssertionError('a failing read was not reported')
    assert again.status.JesdStatus.values == [_setup.JESD_LOCKED, _setup.JESD_UNLOCKED], again.status.JesdStatus.values
    # Four steps declared each run; the good run stepped through all four, the bad one through none.
    assert again.progress == [('total', 4)] + ['step'] * 4 + [('total', 4)], again.progress
    # A firmware without the health check has a status of 'Not found', which the outcome leaves alone.
    old = procedure(_setup, 'atca')
    old.status.JesdStatus = Flag(_setup.JESD_NOT_FOUND)
    with StoodInLoad(_setup):
        old.run(tree('atca', aligned_lanes({}))[1])
    with StoodInLoad(_setup, fail=True):
        try:
            old.run(tree('atca', aligned_lanes({}))[1])
        except _setup.StepError:
            pass
    assert old.status.JesdStatus.values == [], f"a 'Not found' verdict was overwritten: {old.status.JesdStatus.values}"
    # The platform without links never touches a verdict.
    rf = procedure(_setup, 'rfsoc')
    with StoodInLoad(_setup):
        rf.run(tree('rfsoc')[1])
    assert rf.status.JesdStatus.values == [], rf.status.JesdStatus.values


# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    logging.getLogger('cryodaq').setLevel(logging.CRITICAL)

    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []
    print(f"Checking the configuration procedure ({len(checks)} checks)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                       # noqa: BLE001
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
