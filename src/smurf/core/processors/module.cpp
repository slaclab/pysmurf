/**
 *-----------------------------------------------------------------------------
 * Title      : Python Module for Filters
 * ----------------------------------------------------------------------------
 * File       : module.cpp
 * Created    : 2019-09-27
 * ----------------------------------------------------------------------------
 * Description:
 *   Python module setup
 * ----------------------------------------------------------------------------
 * This file is part of the smurf software platform. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of the smurf software platform, including this file, may be
 * copied, modified, propagated, or distributed except according to the terms
 * contained in the LICENSE.txt file.
 * ----------------------------------------------------------------------------
**/

#include <boost/python.hpp>
#include "smurf/core/processors/module.h"
#include "smurf/core/processors/SmurfProcessor.h"

namespace bp  = boost::python;
namespace scp = smurf::core::processors;

void scp::setup_module()
{
    // map the IO namespace to a sub-module
    bp::object module(bp::handle<>(bp::borrowed(PyImport_AddModule("smurf.core.processors"))));

    // make "from mypackage import class1" work
    bp::scope().attr("processors") = module;

    // set the current scope to the new sub-module
    bp::scope io_scope = module;

    scp::SmurfProcessor::setup_python();
}
