Installation
============

Requirements
------------

- Python >= 3.8
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

         docker pull ghcr.io/slaclab/pysmurf-server-base:<TAG>
         docker pull ghcr.io/slaclab/pysmurf-client:<TAG>

      Start the server:

      .. code-block:: bash

         # By shelfmanager + slot:
         docker run ghcr.io/slaclab/pysmurf-server-base:<TAG> -S <shelfmanager> -N <slot>

         # By direct IP:
         docker run ghcr.io/slaclab/pysmurf-server-base:<TAG> -a <FPGA_IP>

      .. note::

         Older releases may still be available on DockerHub under
         ``tidair/pysmurf-server`` and ``tidair/pysmurf-client``.

      See `README.Docker.md
      <https://github.com/slaclab/pysmurf/blob/main/README.Docker.md>`_
      for full container usage.

Offline Mode
------------

For data analysis without hardware:

.. code-block:: python

   import pysmurf
   S = pysmurf.SmurfControl(offline=True)
   timestamps, phase, mask = S.read_stream_data('/path/to/data.dat')
