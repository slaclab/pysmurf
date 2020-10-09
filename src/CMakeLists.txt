# ----------------------------------------------------------------------------
# Title      : ROGUE CMAKE Control
# ----------------------------------------------------------------------------
# File       : CMakeLists.txt
# Created    : 2019-09-24
# ----------------------------------------------------------------------------
# This file is part of the rogue software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
# ----------------------------------------------------------------------------

file(GLOB smurf_SRC "*.cpp")
target_sources(smurf PRIVATE ${smurf_SRC})

add_subdirectory(smurf)