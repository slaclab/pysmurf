Installation
============

Requirements
------------

- Python >= 3.10
- A running SMuRF rogue server (for hardware interaction)

Dependencies (installed automatically): numpy, scipy, matplotlib,
seaborn, pyepics, pyyaml, Cython, schema, packaging.

Install
-------

.. tab-set::

   .. tab-item:: PyPI

      .. code-block:: bash

         pip install pysmurf-slac

   .. tab-item:: Source

      .. code-block:: bash

         git clone https://github.com/slaclab/pysmurf.git
         cd pysmurf && pip install .

      For development: ``pip install -e .``

   .. tab-item:: Docker

      Production systems use Docker containers from the GitHub
      Container Registry:

      .. code-block:: bash

         docker pull ghcr.io/slaclab/pysmurf-server:<TAG>
         docker pull ghcr.io/slaclab/pysmurf-client:<TAG>

      Start the server:

      .. code-block:: bash

         # By shelfmanager + slot:
         docker run ghcr.io/slaclab/pysmurf-server:<TAG> -S <shelfmanager> -N <slot>

         # By direct IP:
         docker run ghcr.io/slaclab/pysmurf-server:<TAG> -a <FPGA_IP>

      .. note::

         Older releases may still be available on DockerHub under
         ``tidair/pysmurf-server`` and ``tidair/pysmurf-client``.

      See `README.Docker.md
      <https://github.com/slaclab/pysmurf/blob/main/README.Docker.md>`_
      for full container usage.

Client without Docker
---------------------

The client reaches a server over the server's rogue ZMQ sockets, so it runs
on any machine that can reach that port: no EPICS and no firmware package.
Control needs nothing else; operations that capture data to a file (the
debug captures, streamed data) have the server write the file and the
client read it back, so a client on another machine needs the server's data
directory mounted at the same path. It needs ``pyrogue`` beside pysmurf's
own dependencies.
``pyrogue`` is not on PyPI (the PyPI packages of that name are unrelated);
install it from `slaclab/rogue <https://github.com/slaclab/rogue>`_ -- its
conda package, or a build from source, as rogue's own documentation
describes -- into the environment you then install pysmurf into:

.. code-block:: bash

   git clone https://github.com/slaclab/pysmurf.git
   cd pysmurf && pip install -e .

Then point a ``SmurfControl`` at the server. A server serving the carrier
in slot 4 of a crate on this host listens on port 9012 (9000 plus three per
slot); a server anywhere else is reached by host and port:

.. code-block:: python

   import pysmurf.client
   S = pysmurf.client.SmurfControl(cfg_file='/path/to/slot4.yaml',
                                   server_addr='smurf-srv11', server_port=9012)

Leave ``cfg_file`` out to reattach to a server that has already been set up
(:doc:`configuration`, *Reattaching without a file*). The container's
``start_client.sh`` does exactly this; its ``-t crate:4`` is the same slot
convention.

Offline Mode
------------

For data analysis without hardware:

.. code-block:: python

   import pysmurf
   S = pysmurf.SmurfControl(offline=True)
   timestamps, phase, mask = S.read_stream_data('/path/to/data.dat')
