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
    toneCh(maxNumTones, 0),
    toneFreq(maxNumTones, 0),
    dataValid(false),
    tau(0.0),
    theta(0.0),
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
        .def("setToneChannels",     &BandPhaseFeedback::setToneChannels)
        .def("getToneChannels",     &BandPhaseFeedback::getToneChannels)
        .def("setToneFrequencies",  &BandPhaseFeedback::setToneFrequencies)
        .def("getToneFrequencies",  &BandPhaseFeedback::getToneFrequencies)
        .def("getTau",              &BandPhaseFeedback::getTau)
        .def("getTheta",            &BandPhaseFeedback::getTheta)
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

void scf::BandPhaseFeedback::setToneChannels(bp::list m)
{
    std::size_t listSize = len(m);

    // Check if the number of tone is valid
    if ( (listSize > maxNumTones) || (listSize < minNumTones) )
    {
        eLog_->error("Invalid number of tones = %zu", listSize);

        // Do not update the 'toneCh' vector.
        return;
    }

    // We will use a temporal vector to hold the new data.
    // New data will be check as it is pushed to this vector. If there
    // are not error, this vector will be swap with 'toneCh'.
    std::vector<std::size_t> temp;

    for (std::size_t i{0}; i < listSize; ++i)
    {
        std::size_t val = bp::extract<std::size_t>(m[i]);

        // Check if the channel index is not greater than the maximum
        // allowed channel index.
        if (val > maxChIndex)
        {
            eLog_->error("Invalid channel number %zu at index %zu", val, i);

            // Do not update the 'toneCh' vector.
            return;
        }

        // A valid number was found. Add it to the temporal vector
        temp.push_back(val);
    }

    // Take the mutex before changing the 'toneCh' vector
    std::lock_guard<std::mutex> lock(mut);

    // At this point, all element in the mask list are valid.
    // Update the 'toneCh' vector
    toneCh.swap(temp);

    // Check if both input vector are valid
    checkDataValid();
}

const bp::list scf::BandPhaseFeedback::getToneChannels() const
{
    bp::list temp;

    for (auto const &v : toneCh)
        temp.append(v);

    return temp;
}


void scf::BandPhaseFeedback::setToneFrequencies(bp::list m)
{
   std::size_t listSize = len(m);

    // Check if the number of tone is valid
    if ( (listSize > maxNumTones) || (listSize < minNumTones) )
    {
        eLog_->error("Invalid number of tones = %zu", listSize);

        // Do not update the 'toneCh' vector.
        return;
    }

    // We will use a temporal vector to hold the new data.
    // New data will be check as it is pushed to this vector. If there
    // are not error, this vector will be swap with 'toneFreq'.
    std::vector<double> temp;

    for (std::size_t i{0}; i < listSize; ++i)
    {
        // No checks at the moment. Just push the data

        temp.push_back(bp::extract<std::size_t>(m[i]));
    }

    // Take the mutex before changing the 'toneCh' vector
    std::lock_guard<std::mutex> lock(mut);

    // At this point, all element in the mask list are valid.
    // Update the 'toneFreq' vector
    toneFreq.swap(temp);

    // Check if both input vector are valid
    checkDataValid();
}

const bp::list scf::BandPhaseFeedback::getToneFrequencies() const
{
    bp::list temp;

    for (auto const &v : toneFreq)
        temp.append(v);

    return temp;
}


const double scf::BandPhaseFeedback::getTau() const
{
    return tau;
}

const double scf::BandPhaseFeedback::getTheta() const
{
    return theta;
}

void scf::BandPhaseFeedback::checkDataValid()
{
    // Check if both vectors has the same size
    if ( toneCh.size() != toneFreq.size() )
        dataValid = false;
    else
        dataValid = true;

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
