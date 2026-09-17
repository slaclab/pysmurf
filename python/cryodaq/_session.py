#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Session
#-----------------------------------------------------------------------------
# File       : _session.py
# Created    : 2026-09-11
#-----------------------------------------------------------------------------
# Description:
#    connect() opens a rogue client on a running server and returns a Session:
#    the client, the platform map that server's tree belongs to, and the little
#    bookkeeping a client needs -- where it writes what it captures, where it
#    logs, and what it publishes. A session ends at `close()` or on leaving a
#    `with` block, and one a program never closes is closed as it exits.
#
#    A session is thin on purpose. get(), set() and node() are a dictionary
#    lookup followed by getNode(); call() starts a command or a process and, if
#    asked, waits on the process's own Running flag. Operations are rogue
#    Process devices, handed back as they are rather than wrapped, so anything
#    rogue can do with one is still available. The whole tree is reachable
#    through `session.root` for work no semantic name covers.
#
#    There is no configuration file: the server is authoritative about its own
#    state. Connecting reads what it says it is, and the registers that record
#    how it was left, and reports them; it changes nothing.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import logging
import math
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union)

from cryodaq import platform
from cryodaq._errors import ConnectError, UnresolvedName

__all__ = ['Session', 'connect', 'Paths', 'ValidationReport', 'NullPublisher',
           'endpoint_of', 'LOG_USER', 'LOG_INFO', 'LOG_ERROR']

# Log levels. The client this replaces used verbosity thresholds of its own and
# compared them by hand; these are ordinary logging levels, so handlers, filters
# and files work as they do everywhere else. USER sits above INFO because it is
# what the operator asked for and should survive a quieter setting.
LOG_ERROR = logging.ERROR
LOG_USER = 25
LOG_INFO = logging.INFO
logging.addLevelName(LOG_USER, 'USER')

log = logging.getLogger(__name__)

# A server for ATCA slot N listens on 9000 + 3 N; the two ports above it are its
# update and notification sockets. This arithmetic exists only here.
SLOT_PORT_BASE = 9000
SLOT_PORT_STRIDE = 3
_CRATE = re.compile(r'^crate:(\d+)$')
_HOST = re.compile(r'^([A-Za-z0-9_.-]+):(\d+)$')

# Where a directory tree of captured data is rooted when the caller names none.
DATA_DIR_ENV = 'SMURF_DATA_DIR'
DEFAULT_DATA_DIR = '/data'

# How often a bounded wait asks a process whether it is still running.
POLL_INTERVAL_S = 0.2

# How long one request may stay unanswered before it is a failed link rather than
# a busy server, and how often to say so while waiting. rogue leaves a linked
# client retrying forever; the client this replaces overrode that with these two
# values, and they remain the default because a request that has gone unanswered
# for half a minute is not going to be answered.
DEFAULT_TIMEOUT_S = 30.0
WARN_INTERVAL_S = 5.0

# Child nodes every rogue Process carries.
PROCESS_START = 'Start'
PROCESS_STOP = 'Stop'
PROCESS_RUNNING = 'Running'

# Stopping a client at exit has to happen before the interpreter joins threads
# that are not daemons -- the link monitor is one -- and that join comes *before*
# atexit runs, so an atexit hook is too late to end it. This hook is early
# enough. It is CPython-internal, so it is looked up rather than assumed, and
# what to do without it is decided at connect.
_REGISTER_BEFORE_JOIN = getattr(threading, '_register_atexit', None)

# The clients this process has open, by endpoint. Keyed the way rogue keys its
# own cache because the rule is rogue's: one client per endpoint. Deliberately
# not a set -- two clients on one endpoint compare equal, so a set would keep a
# torn-down one in preference to the live one meant to replace it.
_open_clients: Dict[str, Any] = {}
_join_hook_registered = False


def _stop_open_clients() -> None:
    """Stop every client still open, on the way out of the interpreter."""
    while _open_clients:
        endpoint, client = _open_clients.popitem()
        try:
            client.stop()
        except Exception:                                       # noqa: BLE001
            # Shutdown is under way and threading does not guard these calls, so
            # a raise here would cost every hook registered before this one.
            log.debug("%s: closing at exit failed", endpoint, exc_info=True)


def _stop_when_the_interpreter_does(client: Any, endpoint: str,
                                    logger: logging.Logger) -> bool:
    """Arrange for ``client`` to be stopped at exit. False if nothing can be.

    rogue's link monitor is not a daemon thread, so the interpreter waits for it
    on the way out; leaving it running therefore means owning its end. An atexit
    handler cannot do that -- the wait for non-daemon threads comes first, so a
    session nobody closed holds the interpreter open with no way left to say
    otherwise. Registering with ``threading`` runs while stopping the client is
    still worth something.
    """
    global _join_hook_registered
    if _REGISTER_BEFORE_JOIN is None:                           # pragma: no cover
        logger.warning(
            "%s: this Python has no threading._register_atexit, so a session "
            "left open would hold the interpreter open at exit; running without "
            "the link monitor instead", endpoint)
        return False
    _open_clients[endpoint] = client
    if not _join_hook_registered:
        _REGISTER_BEFORE_JOIN(_stop_open_clients)
        _join_hook_registered = True
    return True


# --------------------------------------------------------------------------
# bookkeeping the session carries
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Paths:
    """Where the client writes what it captures. Never sent to the server."""

    data: Path
    output: Path
    tune: Path
    plot: Path
    status: Path

    @classmethod
    def under(cls, base: Union[str, Path, None] = None, *, name: Optional[str] = None,
              date: Optional[str] = None) -> 'Paths':
        """Build the usual layout under ``base``.

        Parameters
        ----------
        base : str or Path, optional
            Root of the data tree. Falls back to the ``SMURF_DATA_DIR``
            environment variable, then to ``/data``.
        name : str, optional
            Session name; the UTC start time by default.
        date : str, optional
            Date directory; today in UTC by default.
        """
        root = Path(base or os.environ.get(DATA_DIR_ENV) or DEFAULT_DATA_DIR)
        date = date or time.strftime('%Y%m%d', time.gmtime())
        name = name or time.strftime('%Y%m%d_%H%M%S', time.gmtime())
        session = root / date / name
        return cls(data=root, output=session / 'outputs', tune=root / 'tune',
                   plot=session / 'plots', status=root / 'status')

    def create(self) -> None:
        """Create the directories. Nothing in a session does this for you."""
        for directory in (self.output, self.tune, self.plot, self.status):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ValidationReport:
    """Which semantic names reached a node on this server's tree, and which did not."""

    resolved: Tuple[str, ...]
    unresolved: Tuple[Tuple[str, str], ...]

    @property
    def ok(self) -> bool:
        return not self.unresolved

    def __str__(self) -> str:
        total = len(self.resolved) + len(self.unresolved)
        return f"{len(self.resolved)} of {total} names resolved"


class NullPublisher:
    """The publisher a session uses when the caller injects none: it records nothing.

    An application that publishes supplies its own object with the same two
    methods; the session only ever calls them.
    """

    def register_file(self, path: str, kind: str, **kwargs: Any) -> None:
        """Note that a file was written."""
        log.debug("publisher: file %s (%s)", path, kind)

    def publish(self, message: Any, kind: str = 'general', **kwargs: Any) -> None:
        """Note that something happened."""
        log.debug("publisher: %s (%s)", message, kind)


# --------------------------------------------------------------------------
# endpoints
# --------------------------------------------------------------------------

def endpoint_of(target: str) -> Tuple[str, int]:
    """Resolve a target string to the host and port a server listens on.

    Parameters
    ----------
    target : str
        ``crate:4`` for an ATCA slot on this host, or ``host:port`` for any
        server anywhere. The slot form is shorthand for the port convention.

    Returns
    -------
    tuple
        ``(host, port)``.

    Raises
    ------
    ConnectError
        If the target is neither form.
    """
    if not isinstance(target, str):
        raise ConnectError(f"target must be a string, not {type(target).__name__}")
    text = target.strip()
    m = _CRATE.match(text)
    if m:
        return 'localhost', SLOT_PORT_BASE + SLOT_PORT_STRIDE * int(m.group(1))
    m = _HOST.match(text)
    if m:
        return m.group(1), int(m.group(2))
    raise ConnectError(f"unrecognised target {target!r}; expected 'crate:<slot>' "
                       f"or '<host>:<port>'")


# --------------------------------------------------------------------------
# the session
# --------------------------------------------------------------------------

class Session:
    """A rogue client on one readout system, and the names that reach into it.

    Built by ``connect``. ``root`` is the server's tree as rogue presents it;
    ``get``, ``set``, ``node`` and ``call`` reach it by semantic name through the
    platform map in ``pmap``. ``paths``, ``log`` and ``pub`` are the client-side
    bookkeeping and touch nothing on the server.
    """

    def __init__(self, client: Any, pmap: platform.PlatformMap, *,
                 endpoint: str = '', publisher: Optional[Any] = None,
                 paths: Union[Paths, str, Path, None] = None,
                 logger: Optional[logging.Logger] = None):
        self._client = client
        self.pmap = pmap
        self.endpoint = endpoint
        self.log = logger or logging.getLogger('cryodaq')
        self.pub = publisher if publisher is not None else NullPublisher()
        self.paths = paths if isinstance(paths, Paths) else Paths.under(paths)
        # Whether a path is in this tree, and what access it declares, do not
        # change while the session is open, so the answers are kept: enumerating
        # a scope asks repeatedly, and each mode costs a request of its own.
        self._present: Dict[str, bool] = {}
        self._modes: Dict[str, str] = {}
        self.description = self._read_description()
        self.log.log(LOG_INFO, "session on %s: %s platform, bands %s, %s",
                     endpoint, pmap.name, list(self.indices('band')),
                     'configured' if self.description.get('configured') else 'not configured')

    # ------------------------------------------------------------------
    # the tree
    # ------------------------------------------------------------------

    @property
    def root(self) -> Any:
        """The server's tree. Everything a semantic name does not cover is still here."""
        if self._client is None:
            raise ConnectError('this session is closed')
        return self._client.root

    def _at(self, path: str, name: str) -> Any:
        """The node at a register path, or UnresolvedName naming what was asked for."""
        node = self.root.getNode(path)
        if node is None:
            raise UnresolvedName(name, pattern=path, reason='not in this tree')
        return node

    def _has(self, path: str) -> bool:
        """Whether this tree has a node at ``path``; answers are cached.

        A path the tree does not have is reported as absent by ``getNode``
        itself. Anything raised here is the link failing rather than an answer
        about the tree, and is left to propagate: cached as an absence it would
        quietly shrink every later name listing, witness and validation.
        """
        if path not in self._present:
            self._present[path] = self.root.getNode(path) is not None
        return self._present[path]

    def _mode_of(self, path: str, node: Any) -> str:
        """The access a tree declares for a node -- ``RO``, ``WO`` or ``RW``.

        Asked because a write to a read-only node is not refused by anything
        below: a firmware register drops it where the transaction is built and a
        server-side value keeps it, so a status register can be made to disagree
        with the system it describes. A node that declares no access is taken as
        writable, which is how one was treated before this was asked. Answers
        are cached with the same warrant as ``_has`` -- what a tree declares does
        not change while a session is open -- and reading one costs a request.
        """
        if path not in self._modes:
            self._modes[path] = str(getattr(node, 'mode', 'RW'))
        return self._modes[path]

    # ------------------------------------------------------------------
    # names
    # ------------------------------------------------------------------

    def node(self, name: str) -> Any:
        """The rogue node a semantic name reaches.

        The way to anything rogue offers that this interface does not: a
        process's ``Progress`` and ``Message``, a variable's units, a device's
        children.

        Parameters
        ----------
        name : str
            e.g. ``band[4].ops.gradient_descent``.

        Raises
        ------
        UnresolvedName
            If the map has no such name, or this tree has no such node.
        """
        return self._at(self.pmap.path(name), name)

    def get(self, name: str, *, index: int = -1) -> Any:
        """Read the value a semantic name reaches.

        Parameters
        ----------
        name : str
            e.g. ``band[4].tone.amplitude``.
        index : int
            For a register that is an array, the element to read; the whole
            array by default.

        Raises
        ------
        UnresolvedName
            If the name is not a value in this tree.
        """
        path, kind = self.pmap.entry(name)
        if kind != platform.VALUE:
            raise UnresolvedName(name, pattern=path, reason=f"a {kind}; use call() or node()")
        return self._at(path, name).get(index=index)

    def set(self, name: str, value: Any, *, index: int = -1) -> None:
        """Write the value a semantic name reaches.

        The write is verified and waited for, which is what rogue does unasked;
        a caller who wants otherwise has the node itself through ``node()``.

        Parameters
        ----------
        name : str
            e.g. ``band[4].tone.amplitude``.
        value : object
            What to write. A string written to an enumerated register is taken
            as one of its labels.
        index : int
            For a register that is an array, the element to write.

        Raises
        ------
        UnresolvedName
            If the name is not a value in this tree, or is one this tree
            declares read-only.
        """
        path, kind = self.pmap.entry(name)
        if kind != platform.VALUE:
            raise UnresolvedName(name, pattern=path, reason=f"a {kind}; use call() or node()")
        node = self._at(path, name)
        if self._mode_of(path, node) == 'RO':
            raise UnresolvedName(name, pattern=path, reason='read-only in this tree')
        if isinstance(value, str) and getattr(node, 'enum', None):
            node.setDisp(value, index=index)
        else:
            node.set(value, index=index)

    def call(self, name: str, *args: Any, wait: Optional[float] = None,
             poll: float = POLL_INTERVAL_S) -> Any:
        """Run the command or process a semantic name reaches.

        A command is called and its return value returned. A process is started
        through its own ``Start`` and the process node is returned, so the caller
        can read ``Progress`` and ``Message`` from the server itself.

        Parameters
        ----------
        name : str
            e.g. ``band[4].ops.load_tune_file``.
        ``*args``
            Arguments to a command. A process takes none: set its parameter
            registers by name first.
        wait : float, optional
            For a process, seconds to wait for the server's ``Running`` flag to
            clear. Without it the call returns as soon as the process is
            started. Zero reads the flag once; ``math.inf`` waits as long as the
            process takes.
        poll : float
            Seconds between reads of ``Running`` while waiting. Positive and
            finite: a wait that never sleeps is a wait that spins on the
            transport.

        Returns
        -------
        object
            What a command returned, or the process node.

        Raises
        ------
        UnresolvedName
            If the name is not a command or process in this tree.
        ValueError
            If ``wait`` is negative or ``poll`` is not finite and positive.
            Judged before the process is started, so a refused call has left
            nothing running.
        TimeoutError
            If a process is still running when ``wait`` expires. The process is
            untouched and can be polled or stopped through its node.

        Notes
        -----
        A wait ends when the server says the process is not running, which is
        also true before it starts: a process short enough to finish inside one
        poll interval is indistinguishable here from one that did nothing. What
        happened is in the process's own ``Message``. The ambiguity is the
        server's and not a stale read -- ``Running`` is a local variable on the
        server, rogue's ``get`` reads by default, and the client proxies each
        call, so every poll crosses the wire and returns what the server holds
        at that moment. The window is elsewhere: rogue raises the flag in the
        worker thread rather than in ``Start``, which returns before it.

        ``wait`` bounds how long the process is given, and each read of
        ``Running`` is bounded separately by the session's ``timeout``. A link
        that stops answering therefore ends the wait as a transport failure
        rather than a ``TimeoutError``, and only a session opened with
        ``timeout=None`` can wait on one indefinitely.
        """
        path, kind = self.pmap.entry(name)
        node = self._at(path, name)
        if kind == platform.COMMAND:
            return node(*args)
        if kind != platform.PROCESS:
            raise UnresolvedName(name, pattern=path, reason=f"a {kind}; use get() or set()")
        if args:
            raise TypeError(f"{name!r} is a process and takes no arguments; "
                            f"set its parameters by name first")
        # Both bounds are judged before Start, so a malformed one cannot leave a
        # process running behind a call that failed on its way to waiting. A
        # negative wait fails these comparisons and so does a NaN; infinity does
        # not, and is meant to pass -- unlike the request deadline, which has to
        # reach rogue as an integer number of milliseconds, a wait never leaves
        # Python, and "as long as it takes" is a thing a caller may fairly ask.
        # A non-positive poll is refused rather than left to fail inside the
        # loop, where a zero would spin on the transport and a negative one
        # would surface as a ValueError out of time.sleep.
        if wait is not None and not wait >= 0:
            raise ValueError("wait must be a non-negative number of seconds, or "
                             f"None to return once the process starts, not {wait!r}")
        if not (poll > 0 and math.isfinite(poll)):
            raise ValueError("poll must be a finite positive number of seconds, "
                             f"not {poll!r}")
        self._at(f"{path}.{PROCESS_START}", name)()
        if wait is not None:
            self._wait_idle(name, path, wait, poll)
        return node

    def _wait_idle(self, name: str, path: str, timeout: float, poll: float) -> None:
        """Poll a process's Running flag until it clears or the bound expires."""
        running = self._at(f"{path}.{PROCESS_RUNNING}", name)
        deadline = time.monotonic() + timeout
        while running.get():
            left = deadline - time.monotonic()
            if left <= 0:
                raise TimeoutError(f"{name!r} still running after {timeout:g} s")
            # Sleeping the whole interval would carry the wait past the bound the
            # caller asked for, by up to one interval -- or by all of it, where
            # the interval is the longer of the two.
            time.sleep(min(poll, left))

    def stop(self, name: str) -> None:
        """Ask the process a name reaches to stop. It may take a moment to notice."""
        path, kind = self.pmap.entry(name)
        if kind != platform.PROCESS:
            raise UnresolvedName(name, pattern=path, reason=f"a {kind}; not a process")
        self._at(f"{path}.{PROCESS_STOP}", name)()

    # ------------------------------------------------------------------
    # what this system offers
    # ------------------------------------------------------------------

    def indices(self, scope: str, **fixed: int) -> Tuple[int, ...]:
        """The indices of an indexed scope this tree has.

        Parameters
        ----------
        scope : str
            ``band``, ``bay``, ``uc`` or ``dc`` on a microwave-multiplexed
            readout.
        ``**fixed``
            Indices of the scope's parents, where it has any: an attenuator is
            in a bay.

        Returns
        -------
        tuple of int
            Empty where this platform does not have the hardware behind the
            scope, which is how such a system differs from one that does.
        """
        return platform.indices(self.pmap, self._has, scope, **fixed)

    def _candidates(self) -> Iterator[str]:
        """Every name the map offers this tree, indices filled in from the tree.

        What the map claims, not what resolves: an index the tree has does not
        promise that every name scoped by it is in this build. Which of these
        are real is exactly what ``validate`` reports.
        """
        for pattern in self.pmap.patterns:
            for name in platform.expand(self.pmap, self._has, pattern):
                yield name

    def validate(self, names: Optional[Iterable[str]] = None) -> ValidationReport:
        """Confirm that names reach a node in this tree, one lookup each.

        An expert tool: a few hundred round trips, and nothing on the normal
        path uses it. Discovery is the map, not this.

        Parameters
        ----------
        names : iterable of str, optional
            What to check; every name the map offers this tree by default.
        """
        resolved: List[str] = []
        unresolved: List[Tuple[str, str]] = []
        for name in (self._candidates() if names is None else names):
            try:
                path, _ = self.pmap.entry(name)
                if not self._has(path):
                    raise UnresolvedName(name, pattern=path, reason='not in this tree')
            except UnresolvedName as e:
                unresolved.append((name, str(e)))
            else:
                resolved.append(name)
        return ValidationReport(tuple(resolved), tuple(unresolved))

    def witness(self) -> Dict[str, Any]:
        """Read the registers that record how this system was left.

        Returns
        -------
        dict
            Semantic name to value, in the map's order. A name whose register
            will not read is recorded as None and warned about, so a hole in the
            set is reported rather than hidden.
        """
        values: Dict[str, Any] = {}
        missing: List[str] = []
        for name in platform.witness_names(self.pmap, self._has):
            try:
                values[name] = self.get(name)
            except UnresolvedName:
                values[name] = None
                missing.append(name)
        if missing:
            self.log.warning("%d witness register(s) did not read: %s",
                             len(missing), ', '.join(missing))
        return values

    def _optional_get(self, name: str) -> Any:
        """Read ``name``, or None when this tree does not have it."""
        try:
            return self.get(name)
        except UnresolvedName as e:
            self.log.debug("description: %s", e)
            return None

    def _read_description(self) -> Dict[str, Any]:
        """What the server says it is, read once at connect."""
        return {
            'endpoint': self.endpoint,
            'platform': self.pmap.name,
            'read_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'root': self.root.name,
            'ready': self._optional_get('server.ready'),
            'configured': self._optional_get('application.configured'),
            'configuring': self._optional_get('application.configuring'),
            'enabled_bays': self._optional_get('application.enabled_bays'),
            'startup_arguments': self._optional_get('application.startup_arguments'),
            'application_version': self._optional_get('application.version'),
            'jesd_status': self._optional_get('application.jesd_status'),
            'firmware_version': self._optional_get('firmware.version'),
            'firmware_build_stamp': self._optional_get('firmware.build_stamp'),
            'firmware_git_hash': self._optional_get('firmware.git_hash'),
        }

    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the client. The session is unusable afterwards.

        rogue shares one client per endpoint, so this ends every session opened
        on the same server from this process. A session never closed is closed
        on the way out of the interpreter instead, so forgetting this costs the
        transport nothing; calling it is still how a program says when.
        """
        client, self._client = self._client, None
        if client is not None:
            _open_clients.pop(self.endpoint, None)
            client.stop()

    def __enter__(self) -> 'Session':
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = 'configured' if self.description.get('configured') else 'unconfigured'
        return f"<Session {self.endpoint} {self.pmap.name} {state}>"


def connect(target: str, *, timeout: Optional[float] = DEFAULT_TIMEOUT_S,
            monitor: bool = True,
            platform_name: Optional[str] = None,
            publisher: Optional[Any] = None,
            paths: Union[Paths, str, Path, None] = None,
            logger: Optional[logging.Logger] = None) -> Session:
    """Open a session on the server ``target`` names.

    Parameters
    ----------
    target : str
        ``crate:4`` for an ATCA slot on this host, or ``host:port`` for any
        server anywhere -- including an emulated one, which is a rogue tree
        served over the same sockets. There is no configuration file.
    timeout : float, optional
        Seconds a single request may stay unanswered before it fails and the
        link is called stalled. This bounds one request, not one operation:
        waiting for a long process polls its ``Running`` flag, so a tuning run
        may take as long as it likes. ``None`` waits forever, which is what
        rogue does when nothing asks otherwise, and is the only way to ask for
        it: a deadline has to be a finite number of milliseconds by the time it
        reaches rogue. Set on the client at connect, as the client this replaces
        set it -- see *Notes*.
    monitor : bool
        Leave rogue's link monitor running; it is what notices a dead server.
        False turns it off, which is what the client this replaces did. True
        does not turn one back on: rogue's monitor loop ends for good once the
        flag drops, so nothing can restart a monitor another client stopped.
        The monitor's thread is not a daemon, so a session left open would hold
        the interpreter at exit; leaving the monitor running therefore also
        registers the close that a caller who never called ``close`` did not --
        see *Notes*.
    platform_name : str, optional
        Which platform this is, for a system whose firmware cannot say: an
        emulated register space reports an empty build stamp, and a bench system
        may run firmware not yet listed in the maps. Given, the name is taken as
        it stands and identification never runs -- so a wrong one is a wrong
        register map, seen as names that do not resolve. The firmware registers
        are still read, for the description every session reports; what they say
        is not compared against the name. Left out, and it should be, the
        firmware is what decides.
    publisher : object, optional
        Something with ``register_file`` and ``publish``; nothing is published
        without one.
    paths : Paths or str or Path, optional
        Where the client writes what it captures.
    logger : logging.Logger, optional
        Where the session logs.

    Returns
    -------
    Session

    Raises
    ------
    ConnectError
        If the target does not parse, no platform goes by ``platform_name``, no
        server answers, or the firmware the tree reports belongs to no supported
        platform.
    ValueError
        If ``timeout`` is neither a finite positive number nor None.
    ImportError
        If rogue is not installed. Everything above is judged first, so a
        mistyped target or platform is reported as such even here; opening the
        transport is the step that cannot be reached, and it is the only one.

    Notes
    -----
    **One session per server per process.** rogue keeps one client per
    ``(addr, port)``, so a second session on the same server does not get a
    socket of its own: it gets the first one's, and ``timeout`` and ``monitor``
    are set on that shared client, so the last connect decides both for every
    session holding it -- and closing any one of them ends the transport under
    all the others. Concurrent sessions on one endpoint are not supported;
    a program that wants two needs two processes.

    **A session nobody closes is closed at exit.** rogue's link monitor runs in a
    thread that is not a daemon, so the interpreter waits for it on the way out
    and a session left open would never let go -- the case being an interactive
    session, where nothing guarantees a ``close``. What ends it is registered
    with ``threading`` rather than ``atexit``: the wait for non-daemon threads
    happens before ``atexit`` runs, so an ``atexit`` handler is already too late
    to stop the thread being waited for.
    """
    host, port = endpoint_of(target)
    endpoint = f"{host}:{port}"
    # Finite as well as positive: a deadline goes to rogue as an integer number
    # of milliseconds, and infinity does not become one -- so an unbounded wait
    # is asked for by name, with None, and never by a float that would otherwise
    # be refused only after a socket had been opened, and by the wrong exception.
    if timeout is not None and not (timeout > 0 and math.isfinite(timeout)):
        raise ValueError("timeout must be a finite positive number of seconds, "
                         f"or None for no deadline, not {timeout!r}")
    # A named platform is a lookup in a table this package carries, so it is
    # answered here with the rest: naming a platform and discovering one are two
    # questions, and only the second needs a tree.
    declared = platform.by_name(platform_name) if platform_name is not None else None
    # Every argument is judged above, before the one import that needs anything
    # outside this package: a caller who has mistyped a target or a platform
    # should be told that, whether or not the machine this runs on has rogue.
    try:
        import pyrogue.interfaces
    except ImportError as e:                                    # pragma: no cover
        raise ImportError(
            "connecting needs rogue, which arrives with the smurf server image "
            "rather than from a package index; cryodaq itself does not") from e
    try:
        client = pyrogue.interfaces.VirtualClient(addr=host, port=port)
    except Exception as e:                                      # noqa: BLE001
        raise ConnectError(f"no server at {endpoint}: {e}") from e
    try:
        # Linking leaves rogue's own request policy in place -- warn once a
        # second and then retry forever -- so the bound goes on the client here,
        # where the client this replaces put it. In milliseconds, and a zero
        # deadline reads as no deadline, so a sub-millisecond bound rounds up to
        # one instead of down to forever.
        warn = WARN_INTERVAL_S if timeout is None else min(WARN_INTERVAL_S, timeout)
        client.setTimeout(max(1, int(warn * 1000)),
                          0 if timeout is None else max(1, int(timeout * 1000)))
        # The monitor thread is what notices a dead server; a caller who turns it
        # off is asking for the historical behaviour of not being told. It is
        # also not a daemon thread, so leaving it running is a promise to end it:
        # where that promise cannot be made, a caller who forgot close() would
        # pay for it with an interpreter that will not exit, and no monitor is
        # the better of those two.
        if not monitor or not _stop_when_the_interpreter_does(
                client, endpoint, logger or log):
            client._monEnable = False
        if not client.linked:
            raise ConnectError(f"the client at {endpoint} did not link")
        if declared is not None:
            # An override worth being able to find afterwards: the map is this
            # caller's word. The firmware registers are read a moment later for
            # the description, so what is skipped is the comparison, not the
            # read -- say that, or the log claims more than it knows.
            (logger or log).info(
                "%s: platform declared as %s; the firmware it reports was not "
                "consulted", endpoint, declared.name)
        pmap = declared if declared is not None else platform.identify(client.root)
        return Session(client, pmap, endpoint=endpoint, publisher=publisher,
                       paths=paths, logger=logger)
    except BaseException:
        _open_clients.pop(endpoint, None)
        client.stop()
        raise
