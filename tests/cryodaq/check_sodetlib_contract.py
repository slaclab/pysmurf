#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Sodetlib Compatibility Contract Check
#-----------------------------------------------------------------------------
# File       : check_sodetlib_contract.py
# Created    : 2026-09-21
#-----------------------------------------------------------------------------
# Description:
# Checks that the client still offers everything sodetlib calls on it.
#
# sodetlib drives this client from another repository and another organisation,
# so its use of it cannot be found by reading this one: a refactor here can
# remove a method that nothing in this repository calls and break an
# observatory's analysis code, with nothing failing until someone runs it. The
# surface it depends on is therefore frozen in sodetlib_contract.json and checked
# here on every build.
#
# What the freezing is for is that a removal becomes a *decision*. A method
# dropped from the contract shows up in a diff, with a commit message saying why;
# a method dropped without touching the contract fails this check. That asymmetry
# is the whole point -- it is easy to remove something by accident and hard to
# remove it deliberately without noticing.
#
# The private names are listed for the same reason and with a weaker promise.
# sodetlib reaches sixteen underscore-prefixed attributes and methods, which are
# not interface and are not guaranteed; they are recorded so that removing one is
# something someone chose, and so that the list of what an eventual sodetlib
# change has to stop using is written down rather than rediscovered.
#
# This reads the client as source rather than importing it: the client brings a
# plotting stack with it, and this has to run on the bare runner beside the other
# checks here.
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
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
CONTRACT = HERE / 'sodetlib_contract.json'
CLIENT = REPO / 'python' / 'pysmurf' / 'client'

# Where a SmurfControl gets its methods. It is assembled from mixins, so the
# surface is the union of their members and not any one class's.
MIXINS = (
    ('base/smurf_control.py', 'SmurfControl'),
    ('base/base_class.py', 'SmurfBase'),
    ('base/smurf_config_properties.py', 'SmurfConfigPropertiesMixin'),
    ('command/smurf_command.py', 'SmurfCommandMixin'),
    ('util/smurf_util.py', 'SmurfUtilMixin'),
    ('tune/smurf_tune.py', 'SmurfTuneMixin'),
    ('debug/smurf_iv.py', 'SmurfIVMixin'),
    ('debug/smurf_noise.py', 'SmurfNoiseMixin'),
    ('command/cryo_card.py', 'CryoCard'),
)


def contract():
    data = json.loads(CONTRACT.read_text(encoding='utf-8'))
    if not data.get('public'):
        raise AssertionError(f'{CONTRACT} freezes no public names')
    return data


def surface():
    """Every name a SmurfControl offers, by where it comes from.

    Methods and class-level assignments both count: sodetlib reads
    ``S._n_bias_groups`` as readily as it calls ``S.get_att_uc``, and from the
    caller's side there is no difference.
    """
    found = {}
    for relative, class_name in MIXINS:
        path = CLIENT / relative
        if not path.exists():
            raise AssertionError(f'{relative} is gone; the client has been '
                                 f'restructured and this list needs updating')
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        cls = next((n for n in tree.body
                    if isinstance(n, ast.ClassDef) and n.name == class_name), None)
        if cls is None:
            raise AssertionError(f'{relative} no longer defines {class_name}')
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                found.setdefault(node.name, relative)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.setdefault(target.id, relative)
        # Attributes assigned on self in __init__ are surface too.
        init = next((n for n in cls.body
                     if isinstance(n, ast.FunctionDef) and n.name == '__init__'), None)
        if init is not None:
            for node in ast.walk(init):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if not isinstance(target, ast.Attribute):
                            continue
                        if isinstance(target.value, ast.Name) \
                                and target.value.id == 'self':
                            found.setdefault(target.attr, relative)
    return found


def check_the_client_offers_every_public_name_sodetlib_calls():
    """The frozen public surface is all still there.

    A name missing here is a method sodetlib calls that this client no longer has.
    Whether that is a break or an intended removal is not for this check to
    decide -- what it insists on is that the contract was edited to say so.
    """
    frozen = contract()['public']
    have = surface()
    missing = sorted(name for name in frozen if name not in have)
    assert not missing, (
        f'{len(missing)} name(s) sodetlib calls are gone from the client. If that is '
        f'deliberate, remove them from sodetlib_contract.json in the same commit and '
        f'say why: ' + ', '.join(missing[:8]))


def check_the_private_names_sodetlib_reaches_are_still_there():
    """The private surface too, with a weaker promise.

    These are not interface and are not guaranteed. They are frozen so that
    removing one is a decision rather than an accident, and so that what an
    eventual sodetlib change has to stop using is written down.
    """
    frozen = contract().get('private', [])
    have = surface()
    missing = sorted(name for name in frozen if name not in have)
    assert not missing, (
        f'{len(missing)} private name(s) sodetlib reaches are gone. These are not '
        f'interface, so removing one is allowed -- but it needs the corresponding '
        f'sodetlib change, and this entry removed: ' + ', '.join(missing))


def check_the_contract_names_nothing_the_client_never_had():
    """A frozen name the client has never offered would make this check vacuous.

    The contract is generated from a measurement of sodetlib, so an entry the
    client does not have means the measurement read something else -- a helper
    sodetlib defines itself, most likely. Caught here rather than silently
    weakening every assertion above.
    """
    data = contract()
    have = surface()
    for name in data.get('not_ours', []):
        assert name not in have, (
            f'{name} is recorded as not ours, but the client defines it; move it into '
            f'the frozen surface')
    assert len(data['public']) >= 60, \
        f"only {len(data['public'])} public names frozen; too few to mean much"


def check_the_deliberately_dropped_names_are_recorded():
    """A name this refactor removed on purpose says so, and stays removed.

    sodetlib's private surface is modifiable by operator decision, so some of it
    is expected to go. What must not happen is a name quietly reappearing on the
    contract's dropped list while still being offered, which would mean the list
    has stopped describing anything.
    """
    dropped = contract().get('deliberately_dropped', {})
    have = surface()
    still = sorted(name for name in dropped if name in have)
    assert not still, (
        'name(s) recorded as deliberately dropped that the client still offers: ' +
        ', '.join(still))
    for name, why in dropped.items():
        assert why and len(why) > 10, f'{name} is dropped with no reason given'


def selftest():
    """Prove each check fails on input it should reject."""
    import tempfile
    global CONTRACT, CLIENT
    saved = (CONTRACT, CLIENT)
    failures = 0

    def expect_failure(label, fn):
        nonlocal failures
        try:
            fn()
        except AssertionError as e:
            print(f'  ok    {label}')
            print(f'          refused with: {str(e)[:92]}')
        except Exception as e:                                   # noqa: BLE001
            failures += 1
            print(f'  FAIL  {label}: raised {type(e).__name__} rather than refusing')
            print(f'          {e}')
        else:
            failures += 1
            print(f'  FAIL  {label}: accepted input it should have refused')

    def write(payload):
        CONTRACT.write_text(json.dumps(payload), encoding='utf-8')

    real = json.loads(saved[0].read_text(encoding='utf-8'))
    try:
        with tempfile.TemporaryDirectory() as tmp:
            CONTRACT = pathlib.Path(tmp) / 'contract.json'

            # A public name the client does not offer must fail.
            bad = json.loads(json.dumps(real))
            bad['public'] = bad['public'] + ['get_a_method_that_never_existed']
            write(bad)
            expect_failure('a frozen public name the client lacks is caught',
                           check_the_client_offers_every_public_name_sodetlib_calls)

            # And a private one.
            bad = json.loads(json.dumps(real))
            bad['private'] = bad['private'] + ['_never_existed']
            write(bad)
            expect_failure('a frozen private name the client lacks is caught',
                           check_the_private_names_sodetlib_reaches_are_still_there)

            # A contract with almost nothing in it must fail rather than pass easily.
            bad = json.loads(json.dumps(real))
            bad['public'] = bad['public'][:3]
            write(bad)
            expect_failure('a contract too small to mean anything is caught',
                           check_the_contract_names_nothing_the_client_never_had)

            # A name recorded as not ours that the client in fact defines must fail.
            bad = json.loads(json.dumps(real))
            bad['not_ours'] = ['get_att_uc']
            write(bad)
            expect_failure('a name wrongly recorded as not ours is caught',
                           check_the_contract_names_nothing_the_client_never_had)

            # A dropped name the client still offers must fail.
            bad = json.loads(json.dumps(real))
            bad['deliberately_dropped'] = {
                'get_att_uc': 'claimed dropped but still present'}
            write(bad)
            expect_failure('a dropped name the client still offers is caught',
                           check_the_deliberately_dropped_names_are_recorded)

            # A dropped name with no reason must fail.
            bad = json.loads(json.dumps(real))
            bad['deliberately_dropped'] = {'_gone_for_good': 'x'}
            write(bad)
            expect_failure('a removal with no reason given is caught',
                           check_the_deliberately_dropped_names_are_recorded)

            # A client whose mixins cannot be read must fail rather than find nothing.
            write(real)
            CLIENT = pathlib.Path(tmp) / 'no_such_client'
            expect_failure('a client this cannot read is caught',
                           check_the_client_offers_every_public_name_sodetlib_calls)
    finally:
        CONTRACT, CLIENT = saved

    print('')
    if failures:
        print(f'SELFTEST FAILED ({failures})')
        return 1
    print('SELFTEST PASS -- every check refuses the input it is meant to refuse.')
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selftest', action='store_true',
                        help="drive each check with input it must refuse")
    args = parser.parse_args()

    if args.selftest:
        return selftest()

    checks = sorted((name[len('check_'):], fn)
                    for name, fn in globals().items()
                    if name.startswith('check_'))
    failed = []

    print(f"Checking the sodetlib compatibility contract ({len(checks)} checks)")
    for label, fn in checks:
        try:
            fn()
        except Exception as e:                                   # noqa: BLE001
            failed.append(label)
            print(f"  FAIL  {label}")
            print(f"          {type(e).__name__}: {e}")
        else:
            print(f"  ok    {label}")

    print('')
    if failed:
        print(f"FAILED ({len(failed)}): {', '.join(failed)}")
        return 1
    data = contract()
    print(f"All checks passed ({len(data['public'])} public and "
          f"{len(data.get('private', []))} private name(s) frozen at sodetlib "
          f"{data['sodetlib_commit']}).")
    return 0


if __name__ == '__main__':
    sys.exit(main())
