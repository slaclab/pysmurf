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
#    the run control, and the procedure that loads the register defaults and
#    checks the links came up.
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
import time
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
    """

    def __init__(self, *, fpga: Any, transport: Any, pmap: platform.PlatformMap,
                 providers: Sequence[OperationsProvider] = (), sinks: Sequence[Any] = (),
                 layers: Sequence[Any] = (), server_port: int = 0, polling: bool = True,
                 configure: bool = False, variable_groups: Optional[Mapping] = None,
                 on_start: Optional[Callable] = None, **kwargs):
        pyrogue.Root.__init__(self, name=ROOT_NAME, initRead=True, pollEn=polling,
                              timeout=5.0, **kwargs)
        self.pmap = pmap
        self._fpga = fpga
        self._transport = transport
        self._layers = [str(layer) for layer in layers]
        self._configure = configure
        self._variable_groups = variable_groups
        self._on_start = on_start

        self.zmqServer = pyrogue.interfaces.ZmqServer(root=self, addr='*', port=server_port)
        self.addInterface(self.zmqServer)

        # The metadata stream: every variable in the 'stream' group, as it changes.
        self.stream = pyrogue.interfaces.stream.Variable(root=self, incGroups='stream')

        self.status = ApplicationStatus(pmap)
        self.add(self.status)
        # Where the configuration a client applies is published, so a later
        # client can read it back without the file.
        self.add(ApplicationConfig())
        self.add(self._fpga)

        # Which converter bays are populated: the ones with a front end in the
        # tree. A bay the composition left out keeps its DAQ mux but loses its
        # front end, and a platform whose converters share the FPGA's die has no
        # front end at all, so reports none. Read off the tree through the map,
        # not from a flag.
        self._enabled_bays = [bay for bay in platform.indices(
            pmap, lambda path: self.getNode(path) is not None, 'bay')
            if self._node('bay[{bay}].debug.enable', bay=bay) is not None]

        # Operations attach after the FPGA is in the tree and before start()
        # seals it; the platform map says where.
        self.attached = attach_operations(self, pmap, providers)

        # File writers: the capture streams (TDEST 0x80..) and the streaming
        # interface (TDEST 0xC0..).
        self._stm_data_writer = pyrogue.utilities.fileio.StreamWriter(name='streamDataWriter')
        self.add(self._stm_data_writer)
        self._stm_interface_writer = pyrogue.utilities.fileio.StreamWriter(name='streamingInterface')
        self.add(self._stm_interface_writer)

        self.sinks = []
        for sink in sinks:
            if callable(sink) and not isinstance(sink, pyrogue.Node):
                sink = sink(self)
            if isinstance(sink, pyrogue.Node):
                self.add(sink)
            pyrogue.streamConnect(transport.streaming_stream, sink)
            self.sinks.append(sink)
        for i, ddr in enumerate(transport.ddr_streams):
            pyrogue.streamConnect(ddr, self._stm_data_writer.getChannel(i))
        pyrogue.streamConnect(transport.streaming_stream,
                              self._stm_interface_writer.getChannel(0))

        self.add(pyrogue.RunControl(
            name='streamRunControl',
            description='Run controller',
            cmd=self._node('daq.software_trigger'),
            rates={1: '1 Hz', 10: '10 Hz', 30: '30 Hz'}))

        # The configuration procedure: load the register layers and check the
        # links. Exposed as a process so a client can start it and wait on it,
        # and one that takes no argument: the files it loads are the server's,
        # so a client with a 40-character limit on what it can send (EPICS
        # channel access) can still run it.
        self.add(pyrogue.Process(
            name='setDefaults',
            description='Set default configuration',
            function=self._set_defaults_cmd))

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

    def start(self):
        """Start the tree, then what a readout server does once it is up."""
        pyrogue.Root.start(self)
        try:
            self._after_start()
        except BaseException:
            # A started root runs threads that are not daemons; left running
            # after a failed start it holds the process open with no server
            # on it. The context manager's exit does not run for an enter that
            # raised, so the stop is here.
            self.stop()
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
        self.setDefaults.UpdatePeriod.set(0.0)
        if self._configure:
            self._set_defaults_cmd()
        if self._on_start is not None:
            self._on_start(self)
        self.Ready.set(True)

    def stop(self):
        """Stop the tree and everything started with it."""
        _log.info("stopping the root")
        pyrogue.Root.stop(self)

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

    # -- the configuration procedure -----------------------------------------

    def _load_config(self, timeout=60, max_retries=4):
        """Load the register layers, retrying; True if rogue reported Done."""
        success = False
        # The update thread can interfere with some steps of initialisation.
        self.LoadConfigProcess.UpdatePeriod.set(0.0)
        files = ','.join(self._layers)

        for i in range(max_retries):
            _log.info("setting defaults from %s (try %d)", files, i)
            try:
                self.LoadConfigProcess.LoadMode.setDisp("File")
                self.LoadConfigProcess.ConfigFile.set(files)
                self.LoadConfigProcess.Start()
                start = time.time()
                while self.LoadConfigProcess.Running.value() and \
                        (timeout == 0 or time.time() - start <= timeout):
                    time.sleep(0.1)
                if self.LoadConfigProcess.Message.value() != "Done":
                    _log.warning("setting defaults try %d failed: LoadConfig did not finish", i)
                else:
                    success = True
                    break
            except Exception as e:
                _log.error("setting defaults try %d failed with: %s", i, e)

        if success:
            _log.info("defaults were set correctly")
        else:
            _log.error("failed to set defaults after %d retries", max_retries)
        return success

    def _check_elastic_buffers(self):
        """Check the JESD receive elastic buffers after a load; reload on failure.

        The latency must read 13 or 14 on lanes 0-1 and 4-9, and 255 on lanes 2-3.
        """
        success = False
        max_retries = 10

        for k in range(max_retries):
            _log.info("check elastic buffers (try %d)", k)
            retry_load_config = False
            # The populated bays whose links this platform's map names: a platform
            # without links checks nothing and passes.
            for i in self._enabled_bays:
                read = self._node('bay[{bay}].jesd.rx.read', bay=i)
                if read is None:
                    continue
                # Reading the individual registers does not work; read the device.
                read()
                for j in range(10):
                    latency = self._node('bay[{bay}].jesd.rx_lane[{lane}].elastic_buffer_latency',
                                         bay=i, lane=j).value()
                    latency_ok = latency == 255 if j in (2, 3) else latency in (13, 14)
                    if latency_ok:
                        _log.debug("  OK - JesdRx[%d].ElBuffLatency[%d] = %d", i, j, latency)
                    else:
                        _log.warning("  JesdRx[%d].ElBuffLatency[%d] = %d", i, j, latency)
                        retry_load_config = True
            if retry_load_config:
                if k < max_retries - 1:
                    _log.warning("check failed; reloading the configuration")
                    if not self._load_config():
                        break
            else:
                success = True
                break

        if success:
            _log.info("elastic buffer check passed")
        else:
            _log.error("elastic buffer check failed %d times", max_retries)
        return success

    def _set_defaults_cmd(self):
        """Configure the system: load the layers, then check the links."""
        status = self.status
        status.ConfiguringInProgress.set(True)
        status.SystemConfigured.set(False)

        success = bool(self._layers)
        if not success:
            _log.error("no register configuration files were given; aborting")
        if success and not self._load_config():
            success = False
        if success and not self._check_elastic_buffers():
            success = False

        status.SystemConfigured.set(success)
        # The loaded configuration runs the firmware's link check, so its
        # outcome is also the links' verdict.
        if self._jesd_health_cmd is not None:
            self._jesd_status.set(success)
        status.ConfiguringInProgress.set(False)

    def _check_jesd_health(self):
        """Run the firmware's JESD health check and record its verdict."""
        if self._jesd_health_cmd is not None:
            self._jesd_status.set(2)
            self._jesd_status.set(self._jesd_health_cmd.call())
