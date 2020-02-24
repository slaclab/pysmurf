/**
 *-----------------------------------------------------------------------------
 * Title      : Python Package
 * ----------------------------------------------------------------------------
 * File       : package.cpp
 * Created    : 2019-09-27
 * ----------------------------------------------------------------------------
 * Description:
 * Python package setup
 * ----------------------------------------------------------------------------
 * This file is part of the rogue software platform. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
 *    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of the rogue software platform, including this file, may be
 * copied, modified, propagated, or distributed except according to the terms
 * contained in the LICENSE.txt file.
 * ----------------------------------------------------------------------------
**/

#include <boost/python.hpp>
#include <smurf/module.h>

BOOST_PYTHON_MODULE(smurf)
{
   PyEval_InitThreads();

   smurf::setup_module();

   printf("smurf package imported\n");
};