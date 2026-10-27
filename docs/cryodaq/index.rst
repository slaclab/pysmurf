.. _cryodaq:

cryodaq
=======

Readout for cryogenic detector systems. What is here today is the client half:
connect to a server, read and write it by semantic name, and run the operations
its tree offers.

A **semantic name** means the same thing on every platform.
``band[0].tone.amplitude`` names the tone amplitude of band 0 wherever that
generation of firmware happens to keep it, and the platform layer turns the name
into the register path for the system actually connected. Register paths appear
only in :mod:`cryodaq.platform`; nothing above it builds one.

.. code-block:: python

   import cryodaq

   with cryodaq.connect('crate:4') as sess:
       sess.set('band[0].tone.amplitude', 0, index=17)
       sess.set('band[0].ops.gradient_descent.max_iters', 15)
       proc = sess.call('band[0].ops.start_gradient_descent')

Connecting needs rogue, which arrives with the SMuRF server image rather than
from a package index. It is imported by :func:`cryodaq.connect` and nowhere
else, so the package, the platform maps and the client-side bookkeeping work
anywhere Python does, and the rogue import fails where it is actually needed
rather than at import time.

The client is a thin layer over rogue's own client: a session holds one, and
``session.root`` is the server's tree, whole and unwrapped. What the client adds
is the stable name and the small amount of bookkeeping a client needs -- where
it writes, where it logs, what it publishes.

Session
-------

.. automodule:: cryodaq
    :members: connect, Session, Paths, ValidationReport, endpoint_of
    :undoc-members:

config
------

How a configuration is resolved from layered YAML files, and how the result is
recorded beside the system it configured. A file names what it builds on with
``inherit:``; the application supplies a default under everything and a
``validate`` callable that judges what the keys mean. The result carries every
key's provenance and a hash of the values, so two layerings that agree on every
value agree on the hash.

``Session.record_config`` writes a resolution to the server's ``ApplicationConfig``
device and to a record on disk; ``Session.resolved_config`` reads it back from
the server. A server that has restarted carries nothing and is configured
again -- the file is a record of what the system was given, with the firmware
identity and witness registers of the moment, not a fallback.

.. automodule:: cryodaq.config
    :members: load, Resolved, merge, flatten, record_path, write_record,
              read_record
    :undoc-members:

Errors
------

.. automodule:: cryodaq._errors
    :members:
    :undoc-members:

platform
--------

Which registers a system has, and where. A platform is identified by the
firmware image its FPGA reports, so a session resolves names against the map for
the hardware in front of it rather than against a configured guess. Three are
mapped: the AMC carrier -- an ATCA carrier with AMC converter cards (``umux-amc``), the RFSoC
built from the same firmware without them (``umux-rfsoc``), and the RFMC
carrier, an RFSoC on a board of its own (``umux-rfmc``). The carrier core is the
one device named for the board, so where the build stamp sits is a property of
each map; the stamp's text is what names the platform.

This is the firmware/software boundary: a register path lives here or nowhere.

.. automodule:: cryodaq.platform
    :members: identify, from_stamp, by_name, PlatformMap, MAPS, KINDS,
              indices, expand, tag_of, witness_names
    :undoc-members:

Writing a platform map
^^^^^^^^^^^^^^^^^^^^^^

A platform is a set of firmware that shares both a register map and a bring-up
procedure; two that read the same registers but are configured differently are
two platforms. Each is one package under ``cryodaq/platform/`` with two halves:
``<name>/__init__.py``, the register map -- data only, no rogue, no reads,
listed in ``MAPS`` and turned into a :class:`PlatformMap` by the package -- and
``<name>/support.py``, the platform's support code in the board-support sense --
its devices instantiated and its initialisation sequence, in rogue -- which only
a server imports. The map module declares:

``NAME``
    The platform's short name (``umux-amc``), what ``--platform`` and
    :func:`by_name` take.
``TAGS``
    The firmware image names this platform runs, as the build stamp reports
    them before its first colon. Read one off a board rather than deriving it:
    an image not listed is refused by name, which is the safe failure.
``PACKAGE_CONFIG_DIR``
    Where the firmware's release archive keeps its configuration files, so a
    relative layer name a server is given means a file there.
``CARRIER``
    The carrier core's paths, bound from the shared template
    (``_umux.Carrier(core, bsi=, timing=)``) to the core device this board
    has; its ``BUILD_STAMP`` is where identification reads the stamp.
``REGISTERS``, ``LEGACY``
    Name pattern to ``(path template, kind)``. Take the shared tables from
    ``_umux`` and add the registers of the hardware only this platform
    carries; a name the platform cannot back must not be declared -- a map is
    a statement of what a tree has. Patterns index their scopes with ``[*]``
    and the template formats the same scope names (``{bay}``). ``LEGACY``
    holds names only deprecated compatibility accessors reach.
``SCOPES``
    Scope name to ``(path templates that prove an index exists, parent
    scopes)``; a scope is something the hardware has *n* of.
``WITNESS``
    Name patterns worth reading back to record how a system was left.

The support module holds one :class:`~cryodaq.server.PlatformSupport` subclass, found
by the composition beside the map, with plain rogue code in it:

``top_level(transport, **options)``
    Builds the firmware's top-level device over the transport's register path
    (``transport.srp``, at ``transport.register_offset`` where the FPGA sits in
    a processor system's window), with the deployment's options for the class;
    an option the class does not take is refused by name. This is the one
    place the top level is built -- a released-package dump builds through it too.
``devices(transport)``
    Devices the server adds beside the FPGA's tree -- hardware a class of the
    firmware package drives over a register path of its own, such as the RFMC
    carrier's RF data converter over ``transport.srp_rfdc``. A dump of the
    package has none of these; the map still names where they sit.
``setup``
    The configuration procedure: a :class:`~cryodaq.server.Setup` subclass
    whose ``steps`` are the firmware's own bring-up order, working on the tree
    directly and taking its paths from the map module beside it.

The checks that keep a platform honest are in ``tests/cryodaq``: every name
must resolve on a dump of the platform's released package, every procedure
step must reach a node its platform has, and the map's stamp register must be
the one the platform reports its own stamp at.

server
------

The server side: how a readout server is composed, and what it carries. One
call takes a transport, the firmware package and the register configuration
layers, reads the build stamp off the hardware to learn the platform, builds the
tree, attaches the operation providers and connects the data sinks. With no
sinks it is a register server from a plain Python environment -- enough to bring
a carrier up (``python -m cryodaq.server``); an application passes its providers
and data processing to the same call.

Operations reach the tree through a :class:`~cryodaq.server.OperationsProvider`: a name,
the semantic name of the device each copy attaches to, the node names it adds,
and the function that adds them. The composition expands the anchor through the
platform map, so a provider never spells a register path, and checks every node
against pyrogue's own collision rule before adding any.

A transport that runs through the PCIe offload card carries the card as well
(:class:`~cryodaq.server.OffloadCard`): the card has its own register space,
and its RSSI lane to the carrier is configured and opened by ``compose()``
before the firmware is probed through it, and closed when the root stops. The
Ethernet transport carries the card too when one is in the machine, to close
the lanes that would otherwise intercept the carrier's traffic.
``--no-pcie-card`` leaves the card alone. The card's register map is its
firmware package's (``SmurfPcie``), bound to the device and written by name;
no tree of the card is started -- the server runs one tree, the carrier's.

The RFMC carrier is reached over a TCP bridge on its processor system
(``--transport tcp-bridge --ps-addr <PS> --addr <FPGA>``): the FPGA's registers
and its RF data converter's driver are each served there, and the data leaves
the FPGA's own Ethernet port as a carrier's does -- received on the host's port
by default, or through the offload card's lane with ``--bridge-data pcie``. The
converter is built as a device beside the FPGA's tree over its own register
path, and the carrier's configuration procedure brings the board's clocks and
converter up, synchronises the converter's tiles and then loads the layers, in
the order the firmware's own start-up uses, with polling held off throughout. A
layer whose top-level key is not the server root's name is refused rather than
credited: rogue would log it as not found and still end the load well. The
procedure has run to ``Done`` on one board on a bench, over the bridge, with the
converter tiles synchronised and the register layer applied -- not with cold
resonators, so what it leaves tuned is unverified -- and runs in emulation with
the clock, reset and converter steps stood in, since an emulated memory has
none of that hardware.

What a platform contributes to the tree -- its top level, the devices beside
it, its procedure -- is the platform's own rogue code, in
``cryodaq.platform.<name>.support`` (see *Writing a platform map*); the
composition asks it for each and adds what it is given. This subpackage and
those modules are the parts of cryodaq that import rogue at module level; the
package does not import it.

.. automodule:: cryodaq.server
    :members: compose, OperationsProvider, attach_operations, PlatformSupport, Setup,
              Transport, eth, pcie, tcp_bridge, emulation, ReadoutRoot, CORE_PROVIDERS,
              add_capture_receivers, OffloadCard
    :undoc-members:
