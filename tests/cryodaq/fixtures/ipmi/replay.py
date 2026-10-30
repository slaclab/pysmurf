#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : Shelf-Manager Tool Replay
#-----------------------------------------------------------------------------
# File       : replay.py
# Created    : 2026-10-30
#-----------------------------------------------------------------------------
# Description:
# Answers the shelf-manager tools the server launcher runs (ipmitool, cba_fru_init,
# cba_amc_init, ping, ProgramFPGA.bash) from the transcripts beside this file,
# one per crate slot: `slot<N>.txt` holds each command as the launcher spells
# it, what the tool printed, and its exit status. Used two ways: imported, a
# `Replay` is a `run` for `cryodaq.server.launch`; on PATH under a tool's name
# (a symlink to this file), it is the tool, for an entry point run as a child.
# A command the transcripts do not hold is an error, not a guess; the write
# commands (reboot, program) are recorded and answered with success.
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import os
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = ('ipmitool', 'cba_fru_init', 'cba_amc_init', 'ping', 'ProgramFPGA.bash')
# `$ <command>` opens an answer, `$[tag] <command>` a variant of one; `= <n>` closes it.
COMMAND_RE = re.compile(r'^\$(?:\[([\w-]+)\])? (.+)$')


def transcript(slot):
    """``{(tag, command): (exit status, text)}`` from ``slot<N>.txt``."""
    answers, key, lines = {}, None, []
    for line in (HERE / f'slot{slot}.txt').read_text().splitlines():
        m = COMMAND_RE.match(line)
        if m:
            key, lines = (m.group(1) or '', m.group(2)), []
        elif line.startswith('= ') and key is not None:
            answers[key] = (int(line[2:]), ''.join(l + '\n' for l in lines))
            key = None
        elif key is not None:
            lines.append(line)
    return answers


class Replay:
    """A `run` answering from the transcripts, logging every call in ``calls``.

    Parameters
    ----------
    variant : str, optional
        A tag whose answers replace the untagged ones (``'rtm-fitted'``).
    overrides : mapping, optional
        ``{substring of the command: (exit status, text)}`` for answers the
        crate did not give; a check that uses one says so.
    """

    def __init__(self, variant='', overrides=None):
        self.variant = variant
        self.overrides = dict(overrides or {})
        self.calls = []
        self._slots = {}

    def __call__(self, argv):
        argv = [str(a) for a in argv]
        tool = os.path.basename(argv[0])
        command = ' '.join([tool] + argv[1:])
        self.calls.append(command)
        if tool not in TOOLS:
            raise AssertionError(f"not a shelf-manager tool: {command}")
        for needle, answer in self.overrides.items():
            if needle in command:
                return answer
        # The writes: a power cycle (NetFn 0x2C, command 0x0A) and an image load.
        # Recorded above; a check asserts on the calls, never on hardware.
        if tool in ('ping', 'ProgramFPGA.bash') or ' raw 0x2C 0x0A ' in command:
            return 0, ''
        slot = self._slot_of(tool, argv)
        answers = self._slots.setdefault(slot, transcript(slot))
        for tag in ((self.variant,) if self.variant else ()) + ('',):
            if (tag, command) in answers:
                return answers[(tag, command)]
        raise AssertionError(f"slot {slot} has no recorded answer for: {command}")

    @staticmethod
    def _slot_of(tool, argv):
        if tool == 'ipmitool':
            return (int(argv[argv.index('-t') + 1], 16) - 0x80) // 2
        return int(argv[-1].split('/')[1])


def main():
    """Run as a tool from PATH: the symlink's name says which."""
    tool = os.path.basename(sys.argv[0])
    try:
        status, text = Replay()([tool] + sys.argv[1:])
    except (AssertionError, FileNotFoundError, ValueError) as e:
        print(f"replay: {e}", file=sys.stderr)
        return 1
    sys.stdout.write(text)
    return status


if __name__ == '__main__':
    sys.exit(main())
