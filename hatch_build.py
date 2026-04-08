"""Hook for hatchling to build Cython extensions"""
from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from Cython.Build import cythonize
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from setuptools import Distribution
import numpy


class CythonHook(BuildHookInterface):

    def initialize(self, version, build_data):
        """Build Cython extensions"""
        print("[CythonHook] Building Cython extensions...")

        # Define extension
        ext = Extension(
            "pysmurf.client.util.stream_data_reader",
            ["python/pysmurf/client/util/stream_data_reader.pyx"],
            include_dirs=[numpy.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c++",
        )

        # Cythonize
        ext_modules = cythonize([ext], compiler_directives={
            'language_level': "3", 'boundscheck': False, 'wraparound': False,
            'cdivision': True, 'initializedcheck': False,
        })

        # Build
        dist = Distribution({'ext_modules': ext_modules})
        cmd = build_ext(dist)
        # point to root of python source
        # this will place the compiled extension where editable installs can find it
        cmd.build_lib = "python"
        cmd.build_temp = "build_cython"
        cmd.finalize_options()
        cmd.run()

        build_data['pure_python'] = False
        build_data['infer_tag'] = True
