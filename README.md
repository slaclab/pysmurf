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
