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

# Define the variable '__version__':
# This has the closest behavior to versioneer that I could find
# https://github.com/maresb/hatch-vcs-footgun-example
try:
    # If setuptools_scm is installed (e.g. in a development environment with
    # an editable install), then use it to determine the version dynamically.
    from setuptools_scm import get_version

    # This will fail with LookupError if the package is not installed in
    # editable mode or if Git is not installed.
    __version__ = get_version(root="..", relative_to=__file__, version_scheme="no-guess-dev")
except (ImportError, LookupError):
    # As a fallback, use the version that is hard-coded in the file.
    try:
        from pysmurf._version import __version__  # noqa: F401
    except ModuleNotFoundError:
        # The user is probably trying to run this without having installed
        # the package, so complain.
        raise RuntimeError(
            "pysmurf is not correctly installed. "
            "Please install it with pip."
        )
