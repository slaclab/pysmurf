See Also
========

Additional documentation lives in README files in the repository.
These cover specific subsystems in more detail than the user guide.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Document
     - Content
   * - `README.SmurfProcessor.md <https://github.com/slaclab/pysmurf/blob/main/README.SmurfProcessor.md>`_
     - Software data processing pipeline: block diagram, filter
       coefficients, downsampler modes, channel mapping, file writer
   * - `README.DataFile.md <https://github.com/slaclab/pysmurf/blob/main/README.DataFile.md>`_
     - Output data file format (frame structure, header fields, data
       encoding)
   * - `README.SmurfPacket.md <https://github.com/slaclab/pysmurf/blob/main/README.SmurfPacket.md>`_
     - Streaming packet format (header word definitions, timing
       fields, TES bias encoding)
   * - `README.Docker.md <https://github.com/slaclab/pysmurf/blob/main/README.Docker.md>`_
     - Docker container setup, server startup arguments, firmware
       version checking, FPGA IP auto-detection
   * - `README.config_file.md <https://github.com/slaclab/pysmurf/blob/main/README.config_file.md>`_
     - Configuration file format, all parameters, setup walkthrough
   * - `README.CustomDataTransmitter.md <https://github.com/slaclab/pysmurf/blob/main/README.CustomDataTransmitter.md>`_
     - Writing custom data transmitters for real-time output to
       external DAQ systems
   * - `README.DataEmulator.md <https://github.com/slaclab/pysmurf/blob/main/README.DataEmulator.md>`_
     - Data emulator usage for testing without hardware

Related Repositories
--------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Repository
     - Description
   * - `slaclab/pysmurf <https://github.com/slaclab/pysmurf>`_
     - This package (Python control software)
   * - `slaclab/rogue <https://github.com/slaclab/rogue>`_
     - Hardware abstraction / data streaming framework
   * - `cyndiayu/babysmurf <https://github.com/cyndiayu/babysmurf>`_
     - Python simulation of SMuRF tracking/demodulation algorithms

Publications
------------

- C. Yu et al., "SLAC Microresonator RF (SMuRF) Electronics: A
  tone-tracking readout system for superconducting microwave resonator
  arrays," Rev. Sci. Instrum. (2022).
  `arXiv:2208.10523 <https://arxiv.org/abs/2208.10523>`_
