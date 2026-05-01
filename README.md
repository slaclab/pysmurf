# pysmurf

[DOE Code](https://www.osti.gov/doecode/biblio/75053)

[![pypi](https://img.shields.io/pypi/v/pysmurf-slac)](https://pypi.org/project/pysmurf-slac/) ![versions](https://img.shields.io/pypi/pyversions/pysmurf-slac) [![test-or-deploy](https://img.shields.io/github/actions/workflow/status/slaclab/pysmurf/test-or-deploy.yml)](https://github.com/slaclab/pysmurf/actions/workflows/test-or-deploy.yml) [![Documentation Status](https://readthedocs.org/projects/pysmurf/badge/?version=main)](https://pysmurf.readthedocs.io/en/main/?badge=main)

The python control software for SMuRF. Includes scripts to do low level
commands as well as higher level analysis.

## Installation
Install pysmurf using pip:
```
pip3 install pysmurf-slac
```

### Installing from Source
To install from source clone this repository and install using pip:

```
git clone https://github.com/slaclab/pysmurf.git
cd pysmurf/
pip3 install .
```

## Configuration

pysmurf is driven by a per-hardware-setup configuration (`.cfg`) file. Site
configs live under [`cfg_files/<site>/`](cfg_files); start from
[`cfg_files/template/template.cfg`](cfg_files/template/template.cfg) when
adding a new setup.

- See [README.config_file.md](README.config_file.md) for the cfg file format,
  the per-band `init` block, `bad_mask`, `amplifier`, `pic_to_bias_group`,
  `bias_group_to_pair`, and the other top-level fields.
- See [README.Docker.md](README.Docker.md) for running the server/client
  containers; the cfg file is passed to the client via `-c <config_file>` and
  the host data directory is mounted with `-v <local_data_dir>:/data`.

Some sites (SLAC, NIST, Princeton) use `shawnhammer`-style deploy scripts
under [`scratch/shawn/scripts/`](scratch/shawn/scripts) that symlink the
chosen cfg files into fixed locations
(`/data/smurf_startup_cfg/smurf_startup.cfg`,
`/data/pysmurf_cfg/<experiment>.cfg`) and the firmware docker into
`/home/cryo/docker/smurf/current`. These are deployment conventions, not a
required pysmurf layout — see those scripts for examples.

## Documentation
Documentation is built using Sphinx, and follows the [NumPy Style
Docstrings][1] convention. To build the documentation first install
the pysmurf package, then run:

```
cd docs/
make html
```

Output will be located in `_build/html`. You can view the compiled
documentation by opening `_build/html/index.html` in the browser of your choice.

The documentation is also updated to readthedocs here: https://pysmurf.readthedocs.io/en/main/

[1]: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_numpy.html
