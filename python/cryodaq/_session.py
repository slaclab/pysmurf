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
#    logs, and what it publishes.
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
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import (Any, Dict, Iterable, Iterator, List, Optional, Tuple, Union)

import pyrogue.interfaces

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

# Child nodes every rogue Process carries.
PROCESS_START = 'Start'
PROCESS_STOP = 'Stop'
PROCESS_RUNNING = 'Running'


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
        # Whether a path is in this tree does not change while the session is
        # open, so the answers are kept: enumerating a scope asks repeatedly.
        self._present: Dict[str, bool] = {}
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
        """Whether this tree has a node at ``path``; answers are cached."""
        if path not in self._present:
            try:
                self._present[path] = self.root.getNode(path) is not None
            except Exception:                                   # noqa: BLE001
                self._present[path] = False
        return self._present[path]

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

    def set(self, name: str, value: Any, *, index: int = -1, check: bool = True) -> None:
        """Write the value a semantic name reaches.

        Parameters
        ----------
        name : str
            e.g. ``band[4].tone.amplitude``.
        value : object
            What to write. A string written to an enumerated register is taken
            as one of its labels.
        index : int
            For a register that is an array, the element to write.
        check : bool
            Verify the write landed before returning.

        Raises
        ------
        UnresolvedName
            If the name is not a value in this tree.
        """
        path, kind = self.pmap.entry(name)
        if kind != platform.VALUE:
            raise UnresolvedName(name, pattern=path, reason=f"a {kind}; use call() or node()")
        node = self._at(path, name)
        if isinstance(value, str) and getattr(node, 'enum', None):
            node.setDisp(value, index=index)
        else:
            node.set(value, index=index, check=check)

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
            started.
        poll : float
            Seconds between reads of ``Running`` while waiting.

        Returns
        -------
        object
            What a command returned, or the process node.

        Raises
        ------
        UnresolvedName
            If the name is not a command or process in this tree.
        TimeoutError
            If a process is still running when ``wait`` expires. The process is
            untouched and can be polled or stopped through its node.

        Notes
        -----
        A wait ends when the server says the process is not running, which is
        also true before it starts: a process short enough to finish inside one
        poll interval is indistinguishable here from one that did nothing. What
        happened is in the process's own ``Message``.
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
        self._at(f"{path}.{PROCESS_START}", name)()
        if wait is not None:
            self._wait_idle(name, path, wait, poll)
        return node

    def _wait_idle(self, name: str, path: str, timeout: float, poll: float) -> None:
        """Poll a process's Running flag until it clears or the bound expires."""
        running = self._at(f"{path}.{PROCESS_RUNNING}", name)
        deadline = time.monotonic() + timeout
        while running.get():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"{name!r} still running after {timeout:g} s")
            time.sleep(poll)

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

    def names(self) -> Iterator[str]:
        """Every semantic name this tree offers, indices filled in from the tree."""
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
            What to check; everything this tree offers by default.
        """
        resolved: List[str] = []
        unresolved: List[Tuple[str, str]] = []
        for name in (self.names() if names is None else names):
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
            log.debug("description: %s", e)
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
        on the same server from this process.
        """
        client, self._client = self._client, None
        if client is not None:
            client.stop()

    def __enter__(self) -> 'Session':
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        state = 'configured' if self.description.get('configured') else 'unconfigured'
        return f"<Session {self.endpoint} {self.pmap.name} {state}>"


def connect(target: str, *, timeout: Optional[float] = None, monitor: bool = True,
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
        Seconds a single request may stay outstanding before rogue calls the
        link stalled. Off by default, as rogue has it: a long process is not a
        stall.
    monitor : bool
        Keep rogue's link monitor running; it is what notices a dead server.
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
        If the target does not parse, no server answers it, or its tree matches
        no supported platform map.
    """
    host, port = endpoint_of(target)
    endpoint = f"{host}:{port}"
    try:
        client = pyrogue.interfaces.VirtualClient(addr=host, port=port,
                                                  requestStallTimeout=timeout)
    except Exception as e:                                      # noqa: BLE001
        raise ConnectError(f"no server at {endpoint}: {e}") from e
    try:
        if not monitor:
            # The monitor thread is what notices a dead server; a caller who
            # turns it off is asking for the historical behaviour of not being
            # told.
            client._monEnable = False
        if not client.linked:
            raise ConnectError(f"the client at {endpoint} did not link")
        pmap = platform.identify(client.root)
        return Session(client, pmap, endpoint=endpoint, publisher=publisher,
                       paths=paths, logger=logger)
    except BaseException:
        client.stop()
        raise
