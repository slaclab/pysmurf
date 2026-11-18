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

Some of what a configuration names is measured on the system rather than
chosen: a band's delay, a tracking frequency. An operation that measures one
records it with ``Session.record_measured``: the value replaces the configured
one in the record and its provenance is ``(operation, time)`` where a
configured value's is ``(file, line)``; a reattaching client sees the measured
value and which operation set it, and the record's dated history holds what it
was before. The user's files are never written. ``measured_layer`` takes a
record -- today's or a dated copy -- and returns its measured keys as a layer,
so a system is configured again with yesterday's measurements over today's
files; configuring from the files alone clears every measurement.

.. automodule:: cryodaq.config
    :members: load, Resolved, merge, flatten, record_path, write_record,
              read_record, measured_layer
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
    Where the firmware's release archive keeps the configuration files its
    authors ship with the firmware, so a relative layer name a server is given
    means a file there (or one in the deployment's defaults directory, which
    holds what SMuRF operations maintain; see *Register configuration files*
    below).
``PACKAGE_DEFAULTS``
    The register defaults file in that directory a server loads first, under
    any layer it is given, or None. A platform with one set of defaults for
    every board (the RFMC carrier, which has no bays) names the file its
    package ships; one whose defaults depend on the population (the AMC
    carrier) declares None and the launcher chooses the file from the slot's
    records.
``BRING_UP``
    What defines this platform's carrier to the launcher before there is a
    tree (a ``_umux.BringUp``): the image name the release's files carry, the
    carrier part numbers a shelf manager reports for it, where a processor
    system sits, and whether the shelf manager loads the FPGA image. A platform whose records imply more -- the AMC carrier's bays
    choose its defaults file -- subclasses it and overrides ``options()``,
    which turns the shelf manager's record of the slot into server options by
    name; the launcher applies them and knows nothing of bays. An empty
    part-number tuple means a shelf manager cannot recognise this platform and
    the launcher takes the addresses it is given.
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

Starting a server
^^^^^^^^^^^^^^^^^

``python -m cryodaq.server`` starts a register server and ``python -m
pysmurf.core.server`` the SMuRF server; both take the same options
(``--help`` lists them) and both pass what they parsed through
:func:`cryodaq.server.launch` before choosing a transport. The launcher fills
in what the command line left out, and every value it fills in is a default:
an option given explicitly is used as given, and the log says which of the
launcher's answers went unused. It is one sequence of eight steps, run top to
bottom whatever the board; two facts decide what each step can do -- whether
a crate answered for the board, and how the board's FPGA comes to run an
image (its *loader*, below).

1. **The crate's facts.** Given a slot (``--shelfmanager <host>`` with
   ``--slot <2..7>``), the carrier is read through the crate's shelf manager
   into one inventory record: the crate id, the carrier's and the rear
   module's FRU records and each AMC bay's, every read attempted and a read
   the controller does not answer recorded as missing with its error. The
   record is published on the status device beside the startup arguments, so
   a client sees what the launcher saw. Without a slot, or when the
   controller does not answer at all (a bench board; a carrier with no
   management controller, such as the RFMC carrier), nothing is known about
   the board but what the command line says, and the steps below take it
   from there.
2. **The platform.** From the carrier's FRU part number when the record was
   read (:func:`cryodaq.platform.bring_up_for`); from ``--platform <name>``
   when it was not. Given both, the record wins and the option is checked
   against it: a disagreement is a refusal, since one of the two is wrong
   about the board in the slot. A part number no platform declares is
   logged, and ``--platform`` may say which; a record that could not be read
   at all is refused, naming the error, unless ``--no-hw-detect`` and
   ``--no-check-firmware`` both say the launcher is to decide nothing from
   it. A file name never names the platform. The platform proper is
   identified later, from the build stamp, once there is a tree.
3. **The addresses.** By SLAC's convention from the crate id:
   ``10.<crate id high byte>.<crate id low byte>.<100 + slot>`` for the FPGA,
   and ``<200 + slot>`` for the processor system of a platform that has one.
   The PCIe RSSI lane is ``slot - 2``, or 0 without a slot. A board whose
   crate said nothing must have its addresses on the command line
   (``--addr``, and ``--ps-addr`` for a board with a processor system); the
   start is refused, naming the read that failed, when it does not.
4. **What else the platform makes of the records** (``BringUp.options``;
   skipped with ``--no-hw-detect``). For the AMC carrier: a bay without a
   board is left out of the tree (``--disable-bay0/1``), the carrier version
   and the board types choose the defaults file,
   ``defaults_<version>_<bay 0>_<bay 1>.yml`` in the configuration directory
   (below) with ``lb``, ``hb``, ``tkid`` or ``none`` per bay, and a Gen2
   carrier of version C03 or later has its power monitor enabled
   (``--enable-em22xx``). The platform's defaults file goes before any
   ``--layer`` given, so the user's overrides it.
5. **The firmware package**, from the firmware directory as above, the
   archives narrowed to those carrying the platform's image name when the
   platform is known. Before the steps that touch the board, so a directory
   holding two releases is refused without a reboot.
6. **The image loader.** The platform's bring-up declaration says how its
   FPGA comes to run an image. An AMC carrier's flash is programmed by the
   crate's shelf manager (an ``.mcs`` file), so its loader exists only when
   the crate answered. A carrier whose processor system boots the FPGA (the
   RFMC carrier, the ZCU208) runs the boot files on that processor's SD
   card, shipped as ``<image>-0x<version>-<date>-<user>-<short githash>.linux.tar.gz``
   and reached over SSH, so its loader exists once the processor's address
   is known, from the crate or from ``--ps-addr``; without the address the
   image can be neither checked nor loaded, which is a refusal unless
   ``--no-check-firmware`` calls the check off.
7. **A hard boot**, when asked (``--hard-boot``), through the loader: the
   shelf manager power-cycles the FPGA and waits for it to answer at its
   address; a processor system is rebooted over SSH, and the FPGA with it.
   The register configuration is then applied at start-up (``--configure``).
   Without a loader the reboot is refused, naming what would allow it.
8. **The firmware check** (skipped with ``--no-check-firmware``): the short
   git hash in the image file's name (``*-<short githash>.mcs[.gz]`` or
   ``.linux.tar.gz``, one of a kind in the firmware directory) against the
   one the FPGA reports -- by the carrier's controller through the shelf
   manager, or from the firmware's version block through the processor's
   register bridge, the same read the server later identifies the platform
   by. When they differ the loader loads the image, which reboots the FPGA
   into it, and ``--configure`` is set as after a hard boot (a hard boot on a
   stale image is therefore two reboots). The processor system's load copies
   the five boot files to the card over SSH (``root`` on the processor has no
   password), checks them there against what was sent, reboots, and reads
   the block back once the system answers again; a hash that still differs
   is a refusal. Without a loader the check says so in the log -- unless no
   platform is known and the firmware directory holds an image that could
   have been checked, which is a refusal: name the platform, or call the
   check off.

An emulated server (``--transport emulation --platform <name>``) has no
board: steps 6 to 8 do nothing and ``--hard-boot`` is refused.

The shelf manager is driven through SLAC's crate tools (``ipmitool``,
``cba_amc_init``, ``ProgramFPGA.bash``) from ``cryodaq.platform._atca``, and
a processor system through ``ssh``, ``scp`` and ``ping`` from
``cryodaq.platform._soc``, both by the one function that starts a subprocess;
nothing writes to a carrier -- no power cycle, no image load -- except at
steps 7 and 8, and the launcher's checks replay recorded crate answers with
no crate (a processor system's answers during a load are made up there, none
having been recorded).

The firmware directory (``--firmware-dir``, default ``/tmp/fw``) is the one
place a deployment keeps its firmware, and the one option that names it. The
server finds the firmware's Python package there: the one release archive
(``rogue_<image>_v*.zip``), or else the one checkout (a directory with
``firmware/python``, or the firmware directory being one itself). Two of a
kind is a refusal naming them, since guessing between releases is how a server
comes up on the wrong register map; none is allowed, for a package already
importable. The same directory holds the FPGA image the firmware check
compares against: the ``.mcs`` the shelf manager programs, or the
``.linux.tar.gz`` a processor system boots from.

Register configuration files come from two places and no other, because two
parties write them on two release cycles. The firmware package's own
configuration directory (``PACKAGE_CONFIG_DIR`` in the platform's map module;
``python/CryoDet/config`` in a CryoDet release, ``python/rfmc_cryo_det/config``
in an RFMC one), inside the archive or under the checkout, holds what the
firmware's authors ship with the firmware: its own register defaults and, on
the RFMC, the clock chip's register file. The deployment's defaults directory
(``--defaults-dir``, default ``<firmware-dir>/smurf_cfg/defaults``) holds what
SMuRF operations maintain in the ``smurf_cfg`` repository, which the server
image pins at its own version: the files a site chooses from, above all the
AMC carrier's per-population defaults of step 4 (``defaults_c03_lb_lb.yml``
and its kin), which no firmware release carries. Each platform draws on one of
the two in practice -- the RFMC on its package, the AMC carrier on the
deployment -- but the server does not know which in advance, so one resolver
knows both places. A relative ``--layer`` name is looked for in both and
refused if it is in both, since a server configured from the wrong file is as
wrong as one on the wrong register map; the working directory is never
searched. An absolute name is used as given.

A platform whose package ships one set of defaults (``PACKAGE_DEFAULTS`` in
its map module; the RFMC carrier's) loads that file
first, and every ``--layer`` after it as an override, so a site names only
what it changes; a platform whose defaults depend on the population has no
such base, its layers are the launcher's choice from the slot's records, and
its configuration procedure refuses to run with none.

The server's own log and rogue's are set apart: ``--log-level`` (default
``INFO``) governs what the server says about its decisions, identification and
composition, a few lines; ``--rogue-log-level`` (default ``WARNING``) governs
rogue's, which at ``INFO`` narrates the whole tree build.

Options
^^^^^^^

The command line both entry points take, generated from the parser itself
(``cryodaq.server.add_arguments``); ``python -m pysmurf.core.server`` shows the
same options under its own name. The *readout server* group is what the server
is built from; the *crate slot* group is what the launcher reads the rest from.

.. argparse::
   :module: cryodaq.server.__main__
   :func: parser
   :prog: python -m cryodaq.server
   :nodescription:

.. automodule:: cryodaq.server
    :members: compose, OperationsProvider, attach_operations, PlatformSupport, Setup,
              Transport, eth, pcie, tcp_bridge, emulation, ReadoutRoot, CORE_PROVIDERS,
              add_capture_receivers, OffloadCard, launch
    :undoc-members:
