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
    tesBias(reqFrame(TesBiasArray<ris::FrameIterator>::TesBiasBufferSize, true)),
    tba(TesBiasArray<ris::FrameIterator>::create(tesBias->beginWrite()))
{
    // Update the payload of the TES bias frame buffer
    tesBias->setPayload(TesBiasArray<ris::FrameIterator>::TesBiasBufferSize);

    // Update the iterator, just in case the method setPayload invalidated it
    tba->setDataIt(tesBias->beginWrite());

    // Clear the buffer
    std::fill(tesBias->beginWrite(), tesBias->endWrite(), 0);
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
        .def("setTesBias", &Header2Smurf::setTesBias)
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

void scc::Header2Smurf::setTesBias(std::size_t index, int32_t value)
{
    // Hold the mutex while the data tesBias array is being written to.
    std::lock_guard<std::mutex> lock(*tba->getMutex());

    tba->setWord(index, value);
}

void scc::Header2Smurf::acceptFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // If the processing block is disabled, do not process the frame
    if (!disable)
    {
        // Create a SmurfHeader object on the frame
        SmurfHeaderPtr<ris::FrameIterator> smurfHeaderOut(SmurfHeader<ris::FrameIterator>::create(frame));

        // Stet he protocol version
        smurfHeaderOut->setVersion(1);

        // Copy the TES Bias values
        {
            // Hold the mutex while the data tesBias array is being written to.
            std::lock_guard<std::mutex> lock(*tba->getMutex());

            smurfHeaderOut->copyTESBiasArrayFrom(tesBias->beginRead());
        }

        // Set the UNIX time
        smurfHeaderOut->setUnixTime(helpers::getTimeNS());
    }

    // Send the frame to the next slave.
    sendFrame(frame);
}
