from setuptools import setup
from Cython.Build import cythonize

setup(ext_modules = cythonize("smurf_data_loader2.pyx", compiler_directives={'language_level': "3"}))