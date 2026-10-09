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
the hardware in front of it rather than against a configured guess.

This is the firmware/software boundary: a register path lives here or nowhere.

.. automodule:: cryodaq.platform
    :members: identify, by_name, PlatformMap, MAPS, KINDS, indices, expand,
              tag_of, witness_names
    :undoc-members:

server
------

The server side: how a readout server is composed, and what it carries. One
call takes a transport, the firmware package and the register configuration
layers, reads the build stamp off the hardware to learn the platform, builds the
tree, attaches the operation providers and connects the data sinks. With no
sinks it is a register server from a plain Python environment -- enough to bring
a carrier up (``python -m cryodaq.server``); an application passes its providers
and data processing to the same call.

Operations reach the tree through a :class:`~cryodaq.server.Provider`: a name,
the semantic name of the device each copy attaches to, the node names it adds,
and the function that adds them. The composition expands the anchor through the
platform map, so a provider never spells a register path, and checks every node
against pyrogue's own collision rule before adding any.

This is the one part of cryodaq that imports rogue at module level; the package
does not import it.

.. automodule:: cryodaq.server
    :members: compose, Composition, Provider, attach_providers, Transport, eth,
              pcie, emulation, ReadoutRoot, CORE_PROVIDERS
    :undoc-members:
