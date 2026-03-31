#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : PySMuRF Python Package Directory File
#-----------------------------------------------------------------------------
# File       : __init__.py
# Created    : 2019-09-30
#-----------------------------------------------------------------------------
# Description:
#    Mark this directory as python package directory.
#-----------------------------------------------------------------------------
# This file is part of the smurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import os
import subprocess

try:
    # If setuptools_scm is installed (e.g. in a development environment with
    # an editable install), then use it to determine the version dynamically.
    from setuptools_scm import get_version

    # This will fail with LookupError if the package is not installed in
    # editable mode or if Git is not installed.
    __version__ = get_version(root="..", relative_to=__file__, version_scheme="no-guess-dev")
except (ImportError, LookupError):
    # As a fallback, try to get the version from Git directly
    try:
        git_describe = subprocess.check_output(['git', 'describe', '--tags', '--dirty', '--always'], universal_newlines=True).strip()
        version_parts = git_describe.split('-')
        if len(version_parts) == 1:
            __version__ = version_parts[0]
        else:
            __version__ = '-'.join(version_parts[:-2]) + '+' + version_parts[-2] + '.' + version_parts[-1]
    except (subprocess.CalledProcessError, OSError):
        # Fallback to a version file
        version_file = os.path.join(os.path.dirname(__file__), '_version.py')
        if os.path.exists(version_file):
            exec(open(version_file).read(), globals())
        else:
            __version__ = 'unknown'

# strip leading v
__version__ = __version__.lstrip('v')
