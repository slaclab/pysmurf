# ----------------------------------------------------------------------------
# Title      : SMURF CMAKE Control
# ----------------------------------------------------------------------------
# File       : CMakeLists.txt
# Created    : 2019-09-24
# ----------------------------------------------------------------------------
# This file is part of the smurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the smurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
# ----------------------------------------------------------------------------

add_subdirectory(common)
add_subdirectory(counters)
add_subdirectory(conventers)
add_subdirectory(processors)
add_subdirectory(transmitters)

target_sources(smurf PRIVATE "${CMAKE_CURRENT_LIST_DIR}/module.cpp")
