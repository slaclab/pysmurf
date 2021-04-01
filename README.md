# pysmurf

[![Build Status](https://github.com/slaclab/pysmurf/workflows/CI/CD/badge.svg)](https://github.com/slaclab/pysmurf/actions?query=workflow%3ACI%2FCD) [![Documentation Status](https://readthedocs.org/projects/pysmurf/badge/?version=main)](https://pysmurf.readthedocs.io/en/main/?badge=main)

The python control software for SMuRF. Includes scripts to do low level
commands as well as higher level analysis.

## Installation
To install pysmurf clone this repository and install using pip:

```
git clone https://github.com/slaclab/pysmurf.git
cd pysmurf/
pip3 install -r requirements.txt
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
