/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Header2Smurf
 * ----------------------------------------------------------------------------
 * File          : Downsamlper.cpp
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data Header2Smurf Class.
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
#include "smurf/core/conventers/Header2Smurf.h"

namespace scc = smurf::core::conventers;

scc::Header2Smurf::Header2Smurf()
:
    ris::Slave(),
    ris::Master(),
    disable(false),
    eLog_(rogue::Logging::create("pysmurf.Header2Smurf"))
{
}

scc::Header2SmurfPtr scc::Header2Smurf::create()
{
    return std::make_shared<Header2Smurf>();
}

void scc::Header2Smurf::setup_python()
{
    bp::class_< scc::Header2Smurf,
                scc::Header2SmurfPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("Header2Smurf",bp::init<>())
        .def("setDisable", &Header2Smurf::setDisable)
        .def("getDisable", &Header2Smurf::getDisable)
    ;
    bp::implicitly_convertible< scc::Header2SmurfPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< scc::Header2SmurfPtr, ris::MasterPtr >();
}

void scc::Header2Smurf::setDisable(bool d)
{
    disable = d;
}

const bool scc::Header2Smurf::getDisable() const
{
    return disable;
}

void scc::Header2Smurf::acceptFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // If the processing block is disabled, do not process the frame
    if (!disable)
    {
        // Check for frames with errors or flags
        if (  frame->getError() || ( frame->getFlags() & 0x100 ) )
        {
            eLog_->error("Received frame with errors and/or flags");
            return;
        }

        // Get the frame size
        std::size_t frameSize { frame->getPayload() };

        // Check for frames with size less than at least the header size
        if ( frameSize < SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize )
        {
            eLog_->error("Received frame with size lower than the header size. Received frame size=%zu, expected header size=%zu",
                frameSize, SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize);
            return;
        }

        // Create a SmurfHeader object on the frame
        SmurfHeaderPtr<ris::FrameIterator> smurfHeaderOut(SmurfHeader<ris::FrameIterator>::create(frame));

        // Stet he protocol version
        smurfHeaderOut->setVersion(1);

        // Set the UNIX time
        smurfHeaderOut->setUnixTime(helpers::getTimeNS());
    }

    // Send the frame to the next slave.
    sendFrame(frame);
}
