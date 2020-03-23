#!/usr/bin/env python
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to 
# the license terms in the LICENSE.txt file found in the top-level directory 
# of this distribution and at: 
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html. 
# No part of the pysmurf software package, including this file, may be 
# copied, modified, propagated, or distributed except according to the terms 
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
from setuptools import setup, find_packages

import versioneer

setup(name='pysmurf',
      description='The python control software for SMuRF',
      packages=find_packages(where='python'),
      package_dir={'': 'python'},
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass())
