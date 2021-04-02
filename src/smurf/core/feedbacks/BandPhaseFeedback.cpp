/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Band Phase Feedback Module
 * ----------------------------------------------------------------------------
 * File          : BandPhaseFeedback.cpp
 * Created       : 2021-04-02
 *-----------------------------------------------------------------------------
 * Description :
 *   This module estimates a band phase parameters, based on 2 or more fixes
 *   tones, and does a slow feedback on all the resonators in the band to
 *   compensate for phase shift and other variations (like temperature drifts).
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
#include "smurf/core/feedbacks/BandPhaseFeedback.h"

namespace scf = smurf::core::feedbacks;

scf::BandPhaseFeedback::BandPhaseFeedback()
:
    ris::Slave(),
    ris::Master(),
    disable(false),
    frameCnt(0),
    badFrameCnt(0),
    eLog_(rogue::Logging::create("pysmurf.BandPhaseFeedback"))
{
}

scf::BandPhaseFeedbackPtr scf::BandPhaseFeedback::create()
{
    return std::make_shared<BandPhaseFeedback>();
}

// Setup Class in python
void scf::BandPhaseFeedback::setup_python()
{
    bp::class_< scf::BandPhaseFeedback,
                scf::BandPhaseFeedbackPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("BandPhaseFeedback", bp::init<>())
        .def("setDisable",          &BandPhaseFeedback::setDisable)
        .def("getDisable",          &BandPhaseFeedback::getDisable)
        .def("getFrameCnt",         &BandPhaseFeedback::getFrameCnt)
        .def("getBadFrameCnt",      &BandPhaseFeedback::getBadFrameCnt)
        .def("clearCnt",            &BandPhaseFeedback::clearCnt)
    ;
    bp::implicitly_convertible< scf::BandPhaseFeedbackPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< scf::BandPhaseFeedbackPtr, ris::MasterPtr >();
}

void scf::BandPhaseFeedback::setDisable(bool d)
{
    disable = d;
}

const bool scf::BandPhaseFeedback::getDisable() const
{
    return disable;
}

const std::size_t scf::BandPhaseFeedback::getFrameCnt() const
{
    return frameCnt;
}

const std::size_t scf::BandPhaseFeedback::getBadFrameCnt() const
{
    return badFrameCnt;
}

void scf::BandPhaseFeedback::clearCnt()
{
    frameCnt    = 0;
    badFrameCnt = 0;
}

void scf::BandPhaseFeedback::acceptFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // Only process the frame is the block is enable.
    if (!disable)
    {
        // Acquire lock on frame.
        ris::FrameLockPtr lock { frame->lock() };

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

        // - Check for frames with size less than at least the header size
        std::size_t frameSize { frame->getPayload() };
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

        // Update the frame counter
        ++frameCnt;
    }

    // Send the frame to the next slave.
    sendFrame(frame);
}
