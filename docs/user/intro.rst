============
Introduction
============

Name
====
**pysmurf** - Python control software for the SMuRF (SLAC Microresonator RF) electronics

Description
===========
**pysmurf** is a Python-based control and data acquisition software package for the SMuRF (SLAC Microresonator RF) readout system. SMuRF is designed for reading out large arrays of superconducting transition-edge sensors (TES) and microwave kinetic inductance detectors (MKIDs) used in millimeter-wave and submillimeter-wave astronomical instruments.

The SMuRF system uses frequency-domain multiplexing to read out hundreds of detectors simultaneously. **pysmurf** provides both low-level register access to FPGA firmware and high-level analysis and control functions for:

* Detector tuning and resonator characterization
* TES biasing and IV curve measurement
* Tone tracking and flux ramp control
* Noise measurement and characterization
* Data streaming and acquisition

Software Architecture
=====================

The full SMuRF software stack, of which **pysmurf** is only one part, consists of multiple layers:

Firmware Layer
**************
The FPGA firmware handles real-time signal processing, tone generation, resonator tracking, and data streaming.

Rogue Layer
***********
Rogue is a Python-based hardware abstraction layer providing structured access to FPGA registers and data streams.

EPICS Layer
***********
The Experimental Physics and Industrial Control System (EPICS) provides a standardized control system interface between firmware registers and higher-level software.

**pysmurf** Layer
*****************
**pysmurf** wraps the EPICS interface and provides user-friendly functions for common operations, automated tuning procedures, and data analysis tools.

Key Concepts
============

Bands and Channels
******************
SMuRF organizes frequency space into bands of 500 MHz each. Each band supports up to 416 channels for reading out individual resonators. Bands are numbered starting from 0:

* Band 0: 4.0-4.5 GHz
* Band 1: 4.5-5.0 GHz
* Band 2: 5.0-5.5 GHz
* Band 3: 5.5-6.0 GHz

Subbands
********
Each band is subdivided into 512 overlapping subbands of approximately 2.4 MHz width (1.2 MHz useful bandwidth). Channels are assigned to subbands in a specific interleaved pattern, with 1 channel assignable per subband.

Resonators
**********
The microwave SQUID multiplexer (μMUX) uses superconducting LC resonators coupled to TES detectors through rf-SQUIDs. Each resonator has a characteristic frequency that shifts when power is deposited in the coupled TES.

Tone Tracking
*************
SMuRF continuously monitors the complex transmission near each resonator frequency to track frequency shifts caused by TES response to incoming radiation.

Flux Ramp
*********
A periodic flux ramp applied to RF SQUIDs upconverts the detector signals in frequency space, avoiding 1/f noise from two-level systems in the resonator dielectrics.
