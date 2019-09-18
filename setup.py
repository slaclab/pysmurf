from setuptools import setup, Extension

import versioneer

setup (name = 'pysmurf',
       description='The python control software for SMuRF',
       package_dir={'pysmurf': 'pysmurf'},
       packages=['pysmurf'],
       version=versioneer.get_version(),
       cmdclass=versioneer.get_cmdclass())
