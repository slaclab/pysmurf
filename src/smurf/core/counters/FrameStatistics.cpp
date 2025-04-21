/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Frame Statistics Module
 * ----------------------------------------------------------------------------
 * File          : FrameStatistics.cpp
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Descciption :
 *   SMuRF Frame Statistics Class.
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
#include "smurf/core/counters/FrameStatistics.h"

namespace scc = smurf::core::counters;

scc::FrameStatistics::FrameStatistics()
:
    ris::Slave(),
    ris::Master(),
    disable(false),
    frameCnt(0),
    frameSize(0),
    firstFrame(true),
    frameLossCnt(0),
    frameOutOrderCnt(0),
    badFrameCnt(0),
    frameNumber(0),
    prevFrameNumber(0),
    eLog_(rogue::Logging::create("pysmurf.FrameStatistics"))
{
}

scc::FrameStatisticsPtr scc::FrameStatistics::create()
{
    return std::make_shared<FrameStatistics>();
}

// Setup Class in python
void scc::FrameStatistics::setup_python()
{
    bp::class_< scc::FrameStatistics,
                scc::FrameStatisticsPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("FrameStatistics", bp::init<>())
        .def("setDisable",          &FrameStatistics::setDisable)
        .def("getDisable",          &FrameStatistics::getDisable)
        .def("getFrameCnt",         &FrameStatistics::getFrameCnt)
        .def("getFrameSize",        &FrameStatistics::getFrameSize)
        .def("clearCnt",            &FrameStatistics::clearCnt)
        .def("getFrameLossCnt",     &FrameStatistics::getFrameLossCnt)
        .def("getFrameOutOrderCnt", &FrameStatistics::getFrameOutOrderCnt)
        .def("getBadFrameCnt",      &FrameStatistics::getBadFrameCnt)
    ;
    bp::implicitly_convertible< scc::FrameStatisticsPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< scc::FrameStatisticsPtr, ris::MasterPtr >();
}

void scc::FrameStatistics::setDisable(bool d)
{
    disable = d;
}

const bool scc::FrameStatistics::getDisable() const
{
    return disable;
}

const std::size_t scc::FrameStatistics::getFrameCnt() const
{
    return frameCnt;
}

const std::size_t scc::FrameStatistics::getFrameSize() const
{
    return frameSize;
}

const std::size_t scc::FrameStatistics::getFrameLossCnt() const
{
    return frameLossCnt;
}

const std::size_t scc::FrameStatistics::getFrameOutOrderCnt() const
{
    return frameOutOrderCnt;
}

const std::size_t scc::FrameStatistics::getBadFrameCnt() const
{
    return badFrameCnt;
}

void scc::FrameStatistics::clearCnt()
{
    frameCnt         = 0;
    frameLossCnt     = 0;
    frameOutOrderCnt = 0;
    badFrameCnt      = 0;
}

void scc::FrameStatistics::acceptFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // Only process the frame is the block is enable.
    if (!disable)
    {
        // Acquire lock on frame.
        ris::FrameLockPtr lock{frame->lock()};

        // Check for errors in the frame:

        // - Check for frames with errors or flags
        if (  frame->getError() || ( frame->getFlags() & 0x100 ) )
        {
            // Log error
            eLog_->warning("Received frame with errors and/or flags");

            // Increase bad frame counter
            ++badFrameCnt;

            return;
        }

        // Get the frame size
        frameSize = frame->getPayload();

        // - Check for frames with size less than at least the header size
        if ( frameSize < SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize )
        {
            // Log error
            eLog_->warning("Received frame with size lower than the header size. Receive frame size=%zu, expected header size=%zu",
                frameSize, SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize);

            // Increase bad frame counter
            ++badFrameCnt;

            return;
        }


        // - The frame has at least the header, so we can construct a (smart) pointer to
        //   the SMuRF header in the input frame (Read-only)
        SmurfHeaderROPtr<ris::FrameIterator> smurfHeaderIn(SmurfHeaderRO<ris::FrameIterator>::create(frame));

        // - Read the number of channel from the header
        uint32_t numChannels { smurfHeaderIn->getNumberChannels() };

        // - Now we can get the number of channels from the header and check if the total frame size is correct.
        //   The frame should have at least enough room to hold the number of channels defined in its header.
        //   Padded frames are allowed.
        if ( ( SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize + ( numChannels * sizeof(fw_t) ) ) > frameSize )
        {
            // Log error
            eLog_->warning("Received frame does not match expected size. Received frame size=%zu. Minimum expected size: header=%zu + payload=%i",
                        frameSize, smurfHeaderIn->SmurfHeaderSize, numChannels * sizeof(fw_t));

            // Increase bad frame counter
            ++badFrameCnt;

            return;
        }

        // At this point the frame is valid

        // Update the frame counter
        ++frameCnt;

        // Store the current and last frame numbers
        // - Previous frame number
        prevFrameNumber = frameNumber;  // Previous frame number

        // - Current frame number
        frameNumber = smurfHeaderIn->getFrameCounter();

        // Check if we are missing frames, or receiving out-of-order frames
        if (firstFrame)
        {
            // Don't compare the first frame
            firstFrame = false;
        }
        else
        {
            // Discard out-of-order frames
            if ( frameNumber < prevFrameNumber )
            {
                ++frameOutOrderCnt;
                return;
            }

            // If we are missing frame, add the number of missing frames to the counter
            std::size_t frameNumberDelta = frameNumber - prevFrameNumber - 1;
            if ( frameNumberDelta )
              frameLossCnt += frameNumberDelta;
        }
    }

    // Send the frame to the next slave.
    sendFrame(frame);
}
