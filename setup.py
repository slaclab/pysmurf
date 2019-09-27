from setuptools import setup, find_packages

import versioneer

setup(name='pysmurf',
      description='The python control software for SMuRF',
      package_dir={'pysmurf': 'pysmurf'},
      packages=find_packages(),
      version=versioneer.get_version(),
      cmdclass=versioneer.get_cmdclass())
