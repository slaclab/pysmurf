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

# Add support for building in conda environment
if (DEFINED ENV{CONDA_PREFIX})
   set(CMAKE_PREFIX_PATH "$ENV{CONDA_PREFIX}")
   link_directories($ENV{CONDA_PREFIX}/lib)
endif()

# Check cmake version
cmake_minimum_required(VERSION 3.5)
include(InstallRequiredSystemLibraries)

# Project name
project (smurf)

# C/C++
enable_language(CXX)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x -Wno-deprecated")

#####################################
# Find Rogue & Support Libraries
#####################################
if (DEFINED ENV{ROGUE_DIR})
   set(Rogue_DIR $ENV{ROGUE_DIR}/lib)
else()
   set(Rogue_DIR ${CMAKE_PREFIX_PATH}/lib)
endif()
find_package(Rogue REQUIRED)

#####################################
# Setup build
#####################################
# Include files
include_directories(${ROGUE_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include/)
include_directories(/usr/local/include/)

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
