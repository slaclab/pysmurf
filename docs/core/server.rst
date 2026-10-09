.. _server:

server module
=============

The SMuRF server is :func:`cryodaq.server.compose` with pysmurf's data
processing attached: the ``SmurfProcessor`` on the streaming interface, the
capture receivers on the DDR streams, the publisher and pysmurf's identity on
the status device. ``python -m pysmurf.core.server`` starts one; a streaming
application calls :func:`pysmurf.core.server.compose` with its own transmitter.

.. automodule:: pysmurf.core.server
    :members: compose, transport_for, add_arguments, main, VARIABLE_GROUPS
