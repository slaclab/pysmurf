# ----------------------------------------------------------------------------
# Title      : SMuRF CMAKE
# ----------------------------------------------------------------------------
# File       : CMakeLists.txt
# Created    : 2018-02-27
# ----------------------------------------------------------------------------
# This file is part of the rogue_example software. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue_example software, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
# ----------------------------------------------------------------------------

# Check cmake version
cmake_minimum_required(VERSION 2.8)
include(InstallRequiredSystemLibraries)

# Project name
project (smurf)

# C/C++
enable_language(CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")

# Boost Configuration
set(Boost_USE_MULTITHREADED ON)

# Boost may need help on SLAC machines
set(BOOST_ROOT:PATHNAME $ENV{BOOST_PATH})

# First try standard suffix for boost
find_package(Boost 1.58 COMPONENTS system thread python3)

# Next try Debian/Ubuntu suffix for boost
if (NOT Boost_FOUND)
   find_package(Boost 1.58 REQUIRED COMPONENTS system thread python-py35)
endif()

# Find python3
find_package(PythonInterp 3 REQUIRED)
find_package(PythonLibs 3 REQUIRED)

# Find Rogue
set(Rogue_DIR $ENV{ROGUE_DIR}/lib)
find_package(Rogue REQUIRED)

include_directories(/usr/local/include/)

include_directories(${ROGUE_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include/)

# Create rogue python library
AUX_SOURCE_DIRECTORY(src SRC_FILES)
add_library(smurf SHARED ${SRC_FILES})

# Set output to TOP/lib, remove lib prefix
set_target_properties(smurf PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set_target_properties(smurf PROPERTIES PREFIX "")

# Link to rogue core
TARGET_LINK_LIBRARIES(smurf LINK_PUBLIC ${ROGUE_LIBRARIES})

# Setup configuration file
set(CONF_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/include)
set(CONF_LIBRARIES    ${PROJECT_SOURCE_DIR}/lib/smurf.so)

# Create the config file
configure_file(smurfConfig.cmake.in ${PROJECT_SOURCE_DIR}/lib/smurfConfig.cmake @ONLY)

add_subdirectory(src)