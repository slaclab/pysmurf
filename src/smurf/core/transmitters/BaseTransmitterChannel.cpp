/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Base Transmitter Channel
 * ----------------------------------------------------------------------------
 * File          : BaseTransmitterChannel.cpp
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *   SMuRF Data Base Transmitter Class, channel interface
 *-----------------------------------------------------------------------------
 * This file is part of the smurf software platform. It is subject to
 * the license terms in the LICENSE.txt file found in the top-level directory
 * of this distribution and at:
    * https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
 * No part of the smurf software platform, including this file, may be
 * copied, modified, propagated, or distributed except according to the terms
 * contained in the LICENSE.txt file.
 *-----------------------------------------------------------------------------
**/

#include <boost/python.hpp>
#include "smurf/core/transmitters/BaseTransmitterChannel.h"
#include "smurf/core/transmitters/BaseTransmitter.h"

namespace bp  = boost::python;
namespace sct = smurf::core::transmitters;

sct::BaseTransmitterChannel::BaseTransmitterChannel(sct::BaseTransmitterPtr bt, uint32_t channel)
{
    channel_ = channel;
    bt_ = bt;
}

sct::BaseTransmitterChannelPtr sct::BaseTransmitterChannel::create(sct::BaseTransmitterPtr bt, uint32_t channel)
{
    return std::make_shared<BaseTransmitterChannel>(bt,channel);
}

void sct::BaseTransmitterChannel::setup_python()
{
    bp::class_< sct::BaseTransmitterChannel,
                sct::BaseTransmitterChannelPtr,
                bp::bases<ris::Slave>,
                boost::noncopyable >
                ("BaseTransmitterChannel",bp::no_init);
   bp::implicitly_convertible<sct::BaseTransmitterChannelPtr, ris::SlavePtr>();
}

void sct::BaseTransmitterChannel::acceptFrame(ris::FramePtr frame)
{
   if ( channel_ == 0 ) bt_->acceptDataFrame(frame);
   else if ( channel_ == 1 ) bt_->acceptMetaFrame(frame);
}

