#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Cryodaq Configuration Resolution Checks
#-----------------------------------------------------------------------------
# File       : check_config_resolves.py
# Created    : 2026-09-30
#-----------------------------------------------------------------------------
# Description:
# Checks of cryodaq.config: that a configuration split over layers resolves to
# the same values as the equivalent single file, that every key can be traced to
# the layer and line that set it, that the faults the layering itself can have
# are refused by name, and that the sidecar records and re-validates a resolution
# the way its readers rely on.
#
# The fixtures under fixtures/config/ are synthetic: three layers whose keys
# mean nothing, a flat file written to equal their resolution, and the expected
# result as JSON. Nothing in them is a word an application uses, so what is
# proven here is the machinery, and an application's own schema is proven by
# the application's own check.
#
# The sidecar checks use a stand-in session that has just what the session's
# reattach path reads -- a description, a way to read a name, a status path --
# so the choice between the server's record, the sidecar and refusal is driven
# through every branch without a server.
#
# Needs PyYAML, which cryodaq.config imports where it reads a file.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
"""Check that cryodaq resolves layered configurations and records them faithfully."""
import argparse
import json
import os
import pathlib
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURES = HERE / 'fixtures' / 'config'

sys.path.insert(0, str(REPO / 'python'))

import cryodaq  # noqa: E402
from cryodaq import config  # noqa: E402
from cryodaq._errors import ConfigError, DescriptionMismatch  # noqa: E402

# The layers, lowest first, and the flat equivalent.
LAYERS = ('default.yaml', 'site.yaml', 'slot.yaml')
FLAT = 'flat.yaml'
EXPECTED = 'expected.json'
TOP = LAYERS[-1]

# Which layer each marker key was set in: the fixture's statement of itself.
MARKERS = {'marker.set_by': 'default.yaml',
           'marker.set_by_site': 'site.yaml',
           'marker.set_by_slot': 'slot.yaml'}


def resolve(name=TOP, **kwargs):
    return config.load(FIXTURES / name, **kwargs)


def layer_of(resolved, key):
    return pathlib.Path(resolved.provenance[key][0]).name


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def check_the_layers_resolve_to_the_committed_result():
    resolved = resolve()
    expected = json.loads((FIXTURES / EXPECTED).read_text())
    assert config._plain(resolved.values) == expected, \
        f"resolved {json.dumps(config._plain(resolved.values), sort_keys=True)}"
    assert config.INHERIT_KEY not in resolved.values, 'inherit is not a value'
    assert tuple(pathlib.Path(p).name for p in resolved.layers) == LAYERS, resolved.layers


def check_the_flat_equivalent_gives_the_same_values_and_hash():
    layered = resolve()
    flat = resolve(FLAT)
    differ = {k for k in set(config.flatten(flat.values)) | set(config.flatten(layered.values))
              if config.flatten(flat.values).get(k) != config.flatten(layered.values).get(k)}
    assert not differ, f"flat and layered differ at {sorted(differ)}"
    assert flat.hash == layered.hash, f"hashes differ: {flat.hash} != {layered.hash}"
    # And the hash is of the values, not of the layering or the key order.
    assert flat.hash == config.hash_of(json.loads((FIXTURES / EXPECTED).read_text()))


def check_every_key_names_the_layer_that_set_it():
    resolved = resolve()
    # Every leaf has provenance and nothing else does.
    leaves = set(config.flatten(resolved.values))
    assert set(resolved.provenance) == leaves, \
        f"provenance and leaves differ by {set(resolved.provenance) ^ leaves}"
    expected = dict(MARKERS)
    # A key set at every layer belongs to the topmost; one nobody overrides to the lowest.
    expected.update({'depth.everywhere': 'slot.yaml', 'depth.to_site': 'site.yaml',
                     'depth.to_slot': 'slot.yaml', 'depth.untouched': 'default.yaml'})
    # Inside one merged mapping, sibling leaves keep their own layers; a list is
    # replaced whole, so it belongs to whoever wrote it last.
    expected.update({'nested.a.b.c': 'default.yaml', 'nested.a.b.d': 'site.yaml',
                     'nested.a.b.e': 'slot.yaml', 'nested.a.list': 'site.yaml'})
    for key, layer in expected.items():
        assert layer_of(resolved, key) == layer, \
            f"{key} from {layer_of(resolved, key)}, expected {layer}"


def check_provenance_lines_point_at_the_key_in_its_file():
    resolved = resolve()
    for key, (file, line) in resolved.provenance.items():
        text = pathlib.Path(file).read_text().splitlines()
        assert 1 <= line <= len(text), f"{key}: line {line} outside {file}"
        leaf = key.rsplit('.', 1)[-1]
        assert text[line - 1].lstrip().startswith(f"{leaf}:"), \
            f"{key}: line {line} of {pathlib.Path(file).name} is {text[line - 1]!r}"


def check_a_default_mapping_sits_under_everything():
    resolved = resolve(default={'only_default': 1, 'scalar': -1, 'depth': {'untouched': 'x'}})
    assert resolved.values['only_default'] == 1
    assert resolved.values['scalar'] == 30, 'the top layer wins over the default'
    assert resolved.values['depth']['untouched'] == 'default', 'the file default wins over the mapping'
    assert resolved.provenance['only_default'] == (config.DEFAULT_LAYER, 0)
    assert resolved.layers[0] == config.DEFAULT_LAYER


def check_the_validator_sees_the_merged_values_and_its_answer_is_the_result():
    seen = {}

    def validate(values):
        seen.update(values)
        return {'replaced': True}

    resolved = resolve(validate=validate)
    assert seen['scalar'] == 30, 'the validator did not see the merged values'
    assert config._plain(resolved.values) == {'replaced': True}
    assert resolved.hash == config.hash_of({'replaced': True}), 'the hash is of what came back'


def check_the_result_survives_a_round_trip_through_plain_data():
    resolved = resolve()
    again = cryodaq.Resolved.from_dict(json.loads(json.dumps(resolved.to_dict())))
    assert config._plain(again.values) == config._plain(resolved.values)
    assert again.provenance == resolved.provenance
    assert again.hash == resolved.hash and again.layers == resolved.layers
    assert resolved.get('nested.a.b.d') == 20 and resolved.get('no.such', 'dflt') == 'dflt'


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------

def refused(fn, *needles):
    try:
        fn()
    except ConfigError as e:
        for needle in needles:
            assert needle in str(e), f"{e} does not name {needle!r}"
        return e
    raise AssertionError('accepted what should have been refused')


def with_files(files):
    """A temporary directory holding ``files`` ({name: text}); returns its path."""
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_config_'))
    for name, text in files.items():
        (d / name).write_text(text)
    return d


def check_an_inheritance_loop_is_refused_naming_the_chain():
    d = with_files({'a.yaml': 'inherit: b.yaml\n', 'b.yaml': 'inherit: a.yaml\n'})
    e = refused(lambda: config.load(d / 'a.yaml'), 'loops', 'a.yaml', 'b.yaml')
    assert e.key == config.INHERIT_KEY
    # A file that inherits itself is the shortest loop.
    d2 = with_files({'me.yaml': 'inherit: me.yaml\n'})
    refused(lambda: config.load(d2 / 'me.yaml'), 'loops', 'me.yaml')


def check_an_unreadable_layer_is_refused_naming_it():
    d = with_files({'top.yaml': 'inherit: missing.yaml\n'})
    refused(lambda: config.load(d / 'top.yaml'), 'missing.yaml', 'cannot read')
    refused(lambda: config.load(d / 'nowhere.yaml'), 'nowhere.yaml')


def check_a_layer_that_is_not_yaml_is_refused_with_its_line():
    d = with_files({'bad.yaml': 'ok: 1\nbroken: [1, 2\n'})
    e = refused(lambda: config.load(d / 'bad.yaml'), 'bad.yaml', 'not valid YAML')
    assert 'line' in str(e), str(e)


def check_a_layer_that_is_not_a_mapping_is_refused():
    d = with_files({'list.yaml': '- 1\n- 2\n', 'text.yaml': 'just words\n'})
    refused(lambda: config.load(d / 'list.yaml'), 'list.yaml', 'mapping', 'list')
    refused(lambda: config.load(d / 'text.yaml'), 'text.yaml', 'mapping', 'str')
    # An empty file is an empty layer, not a fault.
    d2 = with_files({'empty.yaml': '', 'top.yaml': 'inherit: empty.yaml\nx: 1\n'})
    assert config.load(d2 / 'top.yaml').values == {'x': 1}


def check_an_inherit_that_is_not_a_path_or_list_of_paths_is_refused():
    d = with_files({'m.yaml': 'inherit: {a: 1}\n', 'n.yaml': 'inherit: [1, 2]\n'})
    for name in ('m.yaml', 'n.yaml'):
        e = refused(lambda: config.load(d / name), name, config.INHERIT_KEY)
        assert e.key == config.INHERIT_KEY


def check_a_validator_refusal_passes_through_unchanged():
    class Refusal(Exception):
        pass

    def validate(values):
        raise Refusal('scalar is wrong')

    try:
        resolve(validate=validate)
    except Refusal as e:
        assert 'scalar' in str(e)
    else:
        raise AssertionError('the validator was not consulted')


# --------------------------------------------------------------------------
# the sidecar
# --------------------------------------------------------------------------

def check_a_sidecar_round_trips_and_keeps_a_dated_copy():
    resolved = resolve()
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_sidecar_'))
    path = config.sidecar_path(d, 'some-host:9012')
    assert path == d / config.SIDECAR_DIR / 'some_host_9012.json', path
    written = config.write_sidecar(resolved, path, endpoint='some-host:9012',
                                   firmware={'version': '2.5.1'}, witness={'w': 1},
                                   extra={'note': 'x'})
    assert written == path and path.is_file()
    record = config.read_sidecar(path)
    again = cryodaq.Resolved.from_dict(record['resolved'])
    assert again.hash == resolved.hash and config._plain(again.values) == config._plain(resolved.values)
    assert record['witness'] == {'w': 1} and record['firmware'] == {'version': '2.5.1'}
    assert record['extra'] == {'note': 'x'} and record['endpoint'] == 'some-host:9012'
    assert record['written_at'].endswith('Z')
    copies = sorted(p.name for p in path.parent.iterdir())
    assert len(copies) == 2 and copies[1] == path.name, copies
    assert copies[0].startswith('some_host_9012.') and copies[0].endswith('.json'), copies


def check_a_sidecar_write_is_atomic():
    resolved = resolve()
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_sidecar_'))
    path = config.sidecar_path(d, 'h:1')
    config.write_sidecar(resolved, path, history=False)
    before = path.read_bytes()

    real_replace = os.replace

    def failing_replace(src, dst):
        raise OSError('disk went away')

    os.replace = failing_replace
    try:
        try:
            config.write_sidecar(cryodaq.Resolved({'changed': 1}, {}, config.hash_of({'changed': 1}), ()),
                                 path, history=False)
        except OSError:
            pass
        else:
            raise AssertionError('the failing rename did not raise')
    finally:
        os.replace = real_replace
    assert path.read_bytes() == before, 'a failed write changed the record'
    leftovers = [p.name for p in path.parent.iterdir() if p.name != path.name]
    assert not leftovers, f"a failed write left {leftovers}"


def check_a_corrupt_sidecar_is_refused_naming_the_file():
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_sidecar_'))
    (d / 'x.json').write_text('{not json')
    refused(lambda: config.read_sidecar(d / 'x.json'), 'x.json', 'JSON')
    (d / 'y.json').write_text('{"nothing": 1}')
    refused(lambda: config.read_sidecar(d / 'y.json'), 'y.json', 'resolved')
    refused(lambda: config.read_sidecar(d / 'absent.json'), 'absent.json')
    # A record whose hash does not match its own values is refused too.
    record = resolve().to_dict()
    record['values']['scalar'] = 31
    refused(lambda: cryodaq.Resolved.from_dict(record), 'hash')


# --------------------------------------------------------------------------
# publish and reattach, over a stand-in tree
# --------------------------------------------------------------------------

class _Node:
    def __init__(self, value, mode='RW'):
        self.value, self.mode = value, mode

    def get(self, index=-1):
        return self.value

    def set(self, value, index=-1, **kwargs):
        self.value = value


class _Tree:
    """Just enough of a rogue tree for a Session: getNode over a dict of paths."""

    name = 'AMCc'

    def __init__(self, nodes):
        self.nodes = nodes

    def getNode(self, path):
        return self.nodes.get(path)


class _Client:
    def __init__(self, tree):
        self.root = tree

    def stop(self):
        pass


def system(*, configured, with_description=True, status_dir):
    """A Session on a stand-in tree with the firmware, application and witness registers."""
    from cryodaq import _session
    pmap = cryodaq.platform.by_name('umux-atca')
    values = {'firmware.version': 0x2050000, 'firmware.build_stamp': 'MicrowaveMuxBpEthGen2: x',
              'firmware.git_hash': 'abc', 'application.configured': configured,
              'application.jesd_status': 1, 'timing.rx_link_up': 0,
              'flux_ramp.ramp_max_cnt': 76799, 'flux_ramp.enable_trigger': 1,
              'flux_ramp.start_mode': 0, 'stream.enable': 1,
              'band[0].dsp.enable': 1, 'band[0].delay_us': 2.5}
    nodes = {pmap.path(n): _Node(v) for n, v in values.items()}
    # A band exists where its scope's proving node does; the stand-in has band 0.
    for template, in (pmap.scopes['band'][0],):
        nodes[template.format(band=0)] = _Node(None)
    if with_description:
        for n in (_session.RESOLVED_CONFIG, _session.RESOLVED_HASH, _session.RESOLVED_WRITTEN_AT):
            nodes[pmap.path(n)] = _Node('')
    return _session.Session(_Client(_Tree(nodes)), pmap, endpoint='stand-in:9012',
                            paths=cryodaq.Paths(data=status_dir, output=status_dir, tune=status_dir,
                                                plot=status_dir, status=status_dir))


def check_publish_writes_the_server_and_the_sidecar_and_resolved_prefers_the_server():
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_reattach_'))
    resolved = resolve()
    sess = system(configured=True, status_dir=d)
    assert sess.resolved() is None, 'nothing published yet'
    assert sess.description['resolved_config_hash'] is None
    path = sess.publish(resolved, extra={'note': 1})
    assert path == config.sidecar_path(d, 'stand-in:9012') and path.is_file()
    assert sess.get('description.hash') == resolved.hash
    assert sess.description['resolved_config_hash'] == resolved.hash
    record = config.read_sidecar(path)
    assert record['witness']['stream.enable'] == 1 and 'band[0].dsp.enable' in record['witness']
    assert 'band[0].delay_us' not in record['witness'], 'a tuning witness is not compared'
    assert record['firmware']['firmware_version'] == 0x2050000 and record['extra'] == {'note': 1}
    # The server answers while it remembers, even if the sidecar has gone.
    path.unlink()
    again = sess.resolved()
    assert again is not None and again.hash == resolved.hash


def check_resolved_falls_back_to_the_sidecar_after_a_restart():
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_reattach_'))
    resolved = resolve()
    system(configured=True, status_dir=d).publish(resolved)
    # A restarted server: description empty, configured false.
    restarted = system(configured=False, status_dir=d)
    again = restarted.resolved()
    assert again is not None and again.hash == resolved.hash
    assert config._plain(again.values) == config._plain(resolved.values)
    # A server without the description node at all still records to the sidecar.
    old = system(configured=True, with_description=False, status_dir=d)
    old.publish(resolved)
    assert old.description['resolved_config_hash'] is None
    assert old.resolved().hash == resolved.hash, 'read back from the sidecar'


def check_a_sidecar_that_disagrees_with_the_system_is_refused_naming_the_field():
    d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_reattach_'))
    resolved = resolve()
    system(configured=True, status_dir=d).publish(resolved)
    path = config.sidecar_path(d, 'stand-in:9012')

    def corrupt(section, field, value):
        record = config.read_sidecar(path)
        record[section][field] = value
        path.write_text(json.dumps(record))

    def refused_by(field, expected, actual):
        try:
            system(configured=False, status_dir=d).resolved()
        except DescriptionMismatch as e:
            assert e.field == field and e.expected == expected and e.actual == actual, \
                f"{e.field} {e.expected!r} {e.actual!r}"
            assert field in str(e) and repr(expected) in str(e) and repr(actual) in str(e), str(e)
        else:
            raise AssertionError(f"a sidecar wrong at {field} was adopted")

    corrupt('witness', 'stream.enable', 0)
    refused_by('stream.enable', 0, 1)
    corrupt('witness', 'stream.enable', 1)
    corrupt('firmware', 'firmware_version', 1)
    refused_by('firmware_version', 1, 0x2050000)
    corrupt('firmware', 'firmware_version', 0x2050000)
    # A witness the record has and this tree lacks disagrees too: None is an answer.
    corrupt('witness', 'band[7].dsp.enable', 1)
    refused_by('band[7].dsp.enable', 1, None)
    # Restored, it is adopted again.
    record = config.read_sidecar(path)
    del record['witness']['band[7].dsp.enable']
    path.write_text(json.dumps(record))
    assert system(configured=False, status_dir=d).resolved().hash == resolved.hash


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--selftest', action='store_true',
                    help='drive each check with a wrong input and require a complaint')
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items() if name.startswith('check_'))
    failed = []
    print(f"Checking configuration resolution ({len(checks)} checks)")
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


def selftest():
    """Prove the checks fail on what they should reject.

    Each check compares a resolution with something committed or computed; the
    risk is a comparison that cannot fail. So each is pointed at a fixture set
    with one thing wrong and required to complain about that thing.
    """
    global FIXTURES
    saved = FIXTURES
    failures = 0

    def expect_failure(label, fn, saying):
        nonlocal failures
        try:
            fn()
        except AssertionError as e:
            if saying in str(e):
                print(f"  ok    {label}")
                print(f"          refused with: {str(e)[:96]}")
            else:
                failures += 1
                print(f"  FAIL  {label}: refused for a different reason than the case sets up")
                print(f"          wanted: {saying}")
                print(f"          got:    {str(e)[:96]}")
        except Exception as e:                                  # noqa: BLE001
            failures += 1
            print(f"  FAIL  {label}: {type(e).__name__}: {e}")
        else:
            failures += 1
            print(f"  FAIL  {label}: accepted")

    def variant(**edits):
        """A copy of the fixtures with some files replaced."""
        d = pathlib.Path(tempfile.mkdtemp(prefix='cryodaq_config_selftest_'))
        for p in saved.iterdir():
            (d / p.name).write_text(edits.get(p.name, p.read_text()))
        return d

    print("Selftest: each check must refuse a fixture set with one thing wrong")
    try:
        FIXTURES = variant(**{EXPECTED: json.dumps({'scalar': 31})})
        expect_failure('a wrong expected result is caught',
                       check_the_layers_resolve_to_the_committed_result, 'resolved')

        FIXTURES = variant(**{FLAT: (saved / FLAT).read_text().replace('scalar: 30', 'scalar: 31')})
        expect_failure('a flat file that differs from the layers is caught',
                       check_the_flat_equivalent_gives_the_same_values_and_hash, "['scalar']")

        site = (saved / 'site.yaml').read_text().replace('  to_site: site\n', '')
        FIXTURES = variant(**{'site.yaml': site})
        expect_failure('provenance that names the wrong layer is caught',
                       check_every_key_names_the_layer_that_set_it, 'depth.to_site')

        slot = (saved / 'slot.yaml').read_text().replace('marker:\n', 'marker:\n  set_by_site: slot\n')
        FIXTURES = variant(**{'slot.yaml': slot})
        expect_failure('a marker overridden by a higher layer is caught',
                       check_every_key_names_the_layer_that_set_it, 'marker.set_by_site')

        FIXTURES = saved
        real_load = config.load

        def forgetful(path, **kwargs):
            r = real_load(path, **kwargs)
            return cryodaq.Resolved(r.values, dict(list(r.provenance.items())[:-1]), r.hash, r.layers)
        config.load = forgetful
        try:
            expect_failure('a leaf with no provenance is caught',
                           check_every_key_names_the_layer_that_set_it, 'provenance')
        finally:
            config.load = real_load

        real_load = config._read_yaml

        def shifted(path):
            data, lines = real_load(path)
            return data, {k: v + 1 for k, v in lines.items()}
        config._read_yaml = shifted
        try:
            expect_failure('a provenance line that misses its key is caught',
                           check_provenance_lines_point_at_the_key_in_its_file, 'line')
        finally:
            config._read_yaml = real_load

        real_replace = config._replace_atomically

        def non_atomic(path, text):
            # Writes in place, so an interruption leaves half a file behind.
            if 'changed' in text:
                path.write_text(text[:len(text) // 2])
                raise OSError('interrupted half way')
            path.write_text(text)
        config._replace_atomically = non_atomic
        try:
            expect_failure('a write that leaves a partial file is caught',
                           check_a_sidecar_write_is_atomic, 'changed the record')
        finally:
            config._replace_atomically = real_replace

        real_merge = config.merge
        config.merge = lambda lower, upper: {**lower, **upper}
        try:
            expect_failure('a shallow merge is caught',
                           check_the_layers_resolve_to_the_committed_result, 'resolved')
        finally:
            config.merge = real_merge

        from cryodaq import _session
        real_resolved = _session.Session.resolved

        def trusting(self):
            # Adopts the sidecar without looking at the system.
            path = self.sidecar_path()
            if path.is_file():
                return cryodaq.Resolved.from_dict(config.read_sidecar(path)['resolved'])
            return real_resolved(self)
        _session.Session.resolved = trusting
        try:
            expect_failure('a reattach that does not check the witnesses is caught',
                           check_a_sidecar_that_disagrees_with_the_system_is_refused_naming_the_field,
                           'was adopted')
        finally:
            _session.Session.resolved = real_resolved

        real_witness = _session.Session.witness
        _session.Session.witness = lambda self, configured_only=False: real_witness(self)
        try:
            expect_failure('a sidecar that records tuning witnesses is caught',
                           check_publish_writes_the_server_and_the_sidecar_and_resolved_prefers_the_server,
                           'not compared')
        finally:
            _session.Session.witness = real_witness
    finally:
        FIXTURES = saved

    print("")
    if failures:
        print(f"SELFTEST FAILED ({failures})")
        return 1
    print("Selftest passed: every check refuses what it should.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
