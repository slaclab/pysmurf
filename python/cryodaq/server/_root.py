#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : Cryodaq Readout Root
#-----------------------------------------------------------------------------
# File       : _root.py
# Created    : 2019-10-11
#-----------------------------------------------------------------------------
# Description:
#    The rogue root of a readout server. It is built over a transport and a
#    firmware top level, and carries what every server has regardless of
#    application: the ZMQ interface, the application status block, the record
#    of the configuration a client gave it, the operations the providers add to
#    the firmware's devices, file writers for the capture and streaming paths,
#    the run control, and the configuration procedure that loads the register
#    layers and checks the links came up.
#
#    Data sinks -- whatever processes the streaming interface -- are handed in
#    by the composition and connected here; the root does not know what they
#    are. A server with none is a register server, which is enough to bring a
#    carrier up.
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
from typing import Any, Callable, Mapping, Optional, Sequence

import pyrogue
import pyrogue.interfaces
import pyrogue.interfaces.stream
import pyrogue.utilities.fileio

from cryodaq import platform
from cryodaq._errors import UnresolvedName
from cryodaq.server._application_config import ApplicationConfig
from cryodaq.server._application_status import ApplicationStatus
from cryodaq.server._firmware import ROOT_NAME
from cryodaq.server._provider import OperationsProvider, attach_operations
from cryodaq.server._setup import Setup

__all__ = ['ReadoutRoot']

_log = logging.getLogger(__name__)


class ReadoutRoot(pyrogue.Root):
    """The tree of a readout server.

    Parameters
    ----------
    fpga : pyrogue.Device
        The firmware's top-level device, already built over the transport.
    transport : Transport
        Where the data streams come from, and any nodes the link needs in the tree.
    pmap : PlatformMap
        The platform the firmware was identified as; providers attach through it.
    providers : sequence of OperationsProvider
        What adds operations to the tree, in order.
    sinks : sequence
        What the streaming interface is connected to: rogue stream slaves, or
        callables taking this root and returning one, for a sink that needs
        the root (its metadata stream) to be built. A pyrogue node among the
        results is added to the tree.
    layers : sequence of path-like
        The register configuration files, in the order they are applied.
    server_port : int
        ZMQ port; 0 picks a free one.
    polling : bool
        Enable the poll thread.
    configure : bool
        Apply the layers once the root has started.
    variable_groups : mapping, optional
        ``path -> {'groups': [...], 'pollInterval': ...}``, applied at start.
    on_start : callable, optional
        Called with the root once it has started and before ``Ready`` rises;
        where an application attaches what needs a running tree.
    application : str, optional
        The application this server is composed for, published on the status
        device (``application.name``) so a client can refuse the wrong server;
        '' for the readout core alone.
    """

    def __init__(self, *, fpga: Any, transport: Any, pmap: platform.PlatformMap,
                 providers: Sequence[OperationsProvider] = (), sinks: Sequence[Any] = (),
                 layers: Sequence[Any] = (), server_port: int = 0, polling: bool = True,
                 configure: bool = False, variable_groups: Optional[Mapping] = None,
                 on_start: Optional[Callable] = None, application: str = '',
                 devices: Sequence[Any] = (), setup: Optional[Any] = None, **kwargs):
        pyrogue.Root.__init__(self, name=ROOT_NAME, initRead=True, pollEn=polling,
                              timeout=5.0, **kwargs)
        self.pmap = pmap
        self._fpga = fpga
        self._transport = transport
        self.layers = [str(layer) for layer in layers]
        self._configure = configure
        self._variable_groups = variable_groups
        self._on_start = on_start

        self.zmqServer = pyrogue.interfaces.ZmqServer(root=self, addr='*', port=server_port)
        self.addInterface(self.zmqServer)
        # The transport's threaded interfaces that are not nodes (the bridge's
        # TCP clients) stop with the tree, as the ZMQ server does.
        self.addInterface(*transport.interfaces)

        # The metadata stream: every variable in the 'stream' group, as it changes.
        self.stream = pyrogue.interfaces.stream.Variable(root=self, incGroups='stream')

        self.status = ApplicationStatus(pmap, application=application)
        self.add(self.status)
        # Where the configuration a client applies is published, so a later
        # client can read it back without the file.
        self.add(ApplicationConfig())
        self.add(self._fpga)
        # What the platform adds beside the FPGA's tree (the RFMC's RF data
        # converter, over its own register path), built by the platform's own code.
        for device in devices:
            self.add(device)

        # Which converter bays are populated: the ones with an RF front end in
        # the tree, which on every carrier that has one carries the attenuators.
        # A bay the composition left out keeps its DAQ mux but loses its front
        # end, and a platform whose converters share the FPGA's die has no front
        # end at all, so reports none. Read off the tree through the map, not
        # from a flag.
        def has(path):
            return self.getNode(path) is not None
        self._enabled_bays = [
            bay for bay in platform.indices(pmap, has, 'bay')
            if 'attenuator' in pmap.scopes and platform.indices(pmap, has, 'attenuator', bay=bay)]

        # Operations attach after the FPGA is in the tree and before start()
        # seals it; the platform map says where.
        self.attached = attach_operations(self, pmap, providers)

        # Three unrelated things are called a stream here. `self.stream` above
        # is metadata: variable updates. The transport carries two kinds of
        # data from the FPGA: the DDR capture channels, raw samples the DAQ mux
        # records on demand (`ddr_streams`, one per channel), and the streaming
        # interface, the detector data frames the firmware sends continuously
        # (`streaming_stream`). Each kind gets a file writer; only the second
        # is processed.
        self._stm_data_writer = pyrogue.utilities.fileio.StreamWriter(name='streamDataWriter')
        self.add(self._stm_data_writer)
        self._stm_interface_writer = pyrogue.utilities.fileio.StreamWriter(name='streamingInterface')
        self.add(self._stm_interface_writer)

        # The sinks are what the application attaches to the detector data
        # frames -- a processor, a transmitter. Each receives every frame; the
        # same frames are also written raw, so a file capture does not depend
        # on any sink.
        self.sinks = []
        for sink in sinks:
            if callable(sink) and not isinstance(sink, pyrogue.Node):
                sink = sink(self)
            if isinstance(sink, pyrogue.Node):
                self.add(sink)
            pyrogue.streamConnect(transport.streaming_stream, sink)
            self.sinks.append(sink)
        pyrogue.streamConnect(transport.streaming_stream,
                              self._stm_interface_writer.getChannel(0))
        # The DDR captures go to file only, one writer channel per capture
        # channel; the debug capture receivers added later read the same
        # streams into memory.
        for i, ddr in enumerate(transport.ddr_streams):
            pyrogue.streamConnect(ddr, self._stm_data_writer.getChannel(i))

        self.add(pyrogue.RunControl(
            name='streamRunControl',
            description='Run controller',
            cmd=self._node('daq.software_trigger'),
            rates={1: '1 Hz', 10: '10 Hz', 30: '30 Hz'}))

        # The configuration procedure, as the platform does it. Exposed as a
        # process so a client can start it and wait on it, and one that takes no
        # argument: the files it loads are the server's, so a client with a
        # 40-character limit on what it can send (EPICS channel access) can
        # still run it. The base procedure has no steps; a platform gives its own.
        self.setup = setup if setup is not None else Setup()
        self.add(self.setup)

        # Saving state and configuration must not read first: a read of the
        # whole tree mid-run disturbs it, and the cached values are current.
        # (The SaveConfig process rogue also offers does read first.)
        self.SaveState.replaceFunction(lambda arg: self.saveYaml(
            name=arg, readFirst=False, modes=['RW', 'RO', 'WO'], incGroups=None,
            excGroups='NoState', autoPrefix='state', autoCompress=True))
        self.SaveConfig.replaceFunction(lambda arg: self.saveYaml(
            name=arg, readFirst=False, modes=['RW', 'WO'], incGroups=None,
            excGroups='NoConfig', autoPrefix='config', autoCompress=False))

        # Whatever the link needs started and stopped with the tree.
        for node in transport.nodes:
            self.add(node)

        # Behind an offload card with a lane open, the link can be restarted
        # from the tree, for a carrier that has stopped answering.
        if transport.card is not None and transport.card.lane is not None:
            self.add(pyrogue.LocalCommand(
                name=self.pmap.path('server.restart_rssi').split('.')[-1],
                description='Restart RSSI Link',
                function=lambda: transport.card.restart()))

        self.add(pyrogue.LocalVariable(
            name="Ready",
            description="Server has finished initialisation",
            value=False))

    # -- the tree by semantic name ------------------------------------------

    def _node(self, name: str, **indices: int) -> Any:
        """The node a semantic name resolves to on this tree; None if the map or the tree lacks it."""
        try:
            return self.getNode(self.pmap.path(name.format(**indices)))
        except UnresolvedName:
            return None

    def _get(self, name: str, **indices: int) -> Any:
        return self._node(name, **indices).get()

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start the tree, then what a readout server does once it is up.

        A start that fails anywhere stops whatever it had started before
        re-raising, so that a failed server does not hold its process open.
        """
        was_running = self.running
        try:
            pyrogue.Root.start(self)
            self._after_start()
        except BaseException:
            # Rogue starts the tree's threads before it starts the interfaces
            # and reads the registers, and they are not daemons: left running
            # after a failed start they hold the process open with no server
            # on it. The context manager's exit does not run for an enter that
            # raised, so the stop is here -- unless the root was already
            # running, in which case the refusal is the only thing to report.
            # A start that failed before rogue marked the root running (the
            # tree's own checks) has no threads to stop but still holds the
            # card.
            if was_running:
                raise
            if self.running:
                self.stop()
            else:
                self.release()
            raise

    def _after_start(self):
        """What a readout server does once its tree is up, before ``Ready``."""
        self._setup_groups()

        try:
            _log.info("FPGA image: %s; version 0x%x; git 0x%x",
                      self._get('firmware.build_stamp'), self._get('firmware.version'),
                      self._get('firmware.git_hash'))
        except Exception as e:
            _log.warning("could not read the FPGA image information: %s", e)

        self.status.EnabledBays.set(self._enabled_bays)

        # The firmware's own link check. A platform without links names neither
        # the check nor a status for it; a release before the check lacks the
        # command, and then the status says 'Not found'. Looked up before any
        # configuration runs, since the configuration itself calls the check and
        # reports through its status. One predicate decides both: the status
        # node exists -- the map's say-so, the same one the status device used.
        self._jesd_status = self._node('application.jesd_status')
        self._jesd_health_cmd = self._node('jesd.health') if self._jesd_status is not None else None
        if self._jesd_status is not None and self._jesd_health_cmd is None:
            self._jesd_status.set(3)

        # Variable updates must not interleave with configuration.
        self.setup.UpdatePeriod.set(0.0)
        if self._configure:
            self.setup._process()
        if self._on_start is not None:
            self._on_start(self)
        self.Ready.set(True)

    def stop(self):
        """Stop the tree and everything started with it, then release the card."""
        _log.info("stopping the root")
        try:
            pyrogue.Root.stop(self)
        finally:
            self.release()

    def release(self) -> None:
        """Give the transport back: close the card's lane, stop the link's threads.

        The card was opened by `compose`, before the firmware was probed
        through it, and the server's end is the lane's end; the transport's
        own threaded interfaces (the bridge's TCP clients) run from the moment
        they are built. `stop` calls this, and so does anything that ends a
        composed root without starting it. Safe to call more than once.
        """
        for interface in self._transport.interfaces:
            interface._stop()
        if self._transport.card is not None:
            self._transport.card.close()

    def _setup_groups(self):
        """Apply the variable groups and poll intervals the composition gave."""
        if not self._variable_groups:
            return
        for path, spec in self._variable_groups.items():
            node = self.getNode(path)
            if node is None:
                _log.warning("variable groups: %s not found", path)
                continue
            for group in spec['groups']:
                node.addToGroup(group)
            if spec.get('pollInterval') is not None and node.isinstance(pyrogue.BaseVariable):
                node.setPollInterval(spec['pollInterval'])

    # -- the links' verdict ----------------------------------------------------

    def _check_jesd_health(self):
        """Run the firmware's JESD health check and record its verdict."""
        if self._jesd_health_cmd is not None:
            self._jesd_status.set(2)
            self._jesd_status.set(self._jesd_health_cmd.call())
