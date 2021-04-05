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

scf::BandPhaseFeedback::BandPhaseFeedback(std::size_t band)
:
    ris::Slave(),
    ris::Master(),
    disable(false),
    bandNum(band),
    frameCnt(0),
    badFrameCnt(0),
    numCh(0),
    toneCh(maxNumTones, 0),
    toneFreq(maxNumTones, 0),
    maxToneCh(0),
    dataValid(false),
    tau(0.0),
    theta(0.0),
    freqMean(0.0),
    freqDiffs(maxNumTones, 0),
    freqVar(0.0),
    eLog_(rogue::Logging::create("pysmurf.BandPhaseFeedback"))
{
    // Verify that the band number is valid
    if (bandNum > maxBandNum)
        throw std::runtime_error("BandPhaseFeedback: Band number out of range");
}

scf::BandPhaseFeedbackPtr scf::BandPhaseFeedback::create(std::size_t band)
{
    return std::make_shared<BandPhaseFeedback>(band);
}

// Setup Class in python
void scf::BandPhaseFeedback::setup_python()
{
    bp::class_< scf::BandPhaseFeedback,
                scf::BandPhaseFeedbackPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("BandPhaseFeedback", bp::init<std::size_t>())
        .def("setDisable",          &BandPhaseFeedback::setDisable)
        .def("getDisable",          &BandPhaseFeedback::getDisable)
        .def("getBand",             &BandPhaseFeedback::getBand)
        .def("getFrameCnt",         &BandPhaseFeedback::getFrameCnt)
        .def("getBadFrameCnt",      &BandPhaseFeedback::getBadFrameCnt)
        .def("clearCnt",            &BandPhaseFeedback::clearCnt)
        .def("getNumCh",            &BandPhaseFeedback::getNumCh)
        .def("setToneChannels",     &BandPhaseFeedback::setToneChannels)
        .def("getToneChannels",     &BandPhaseFeedback::getToneChannels)
        .def("setToneFrequencies",  &BandPhaseFeedback::setToneFrequencies)
        .def("getToneFrequencies",  &BandPhaseFeedback::getToneFrequencies)
        .def("getTau",              &BandPhaseFeedback::getTau)
        .def("getTheta",            &BandPhaseFeedback::getTheta)
        .def("getDataValid",        &BandPhaseFeedback::getDataValid)
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

const std::size_t scf::BandPhaseFeedback::getBand() const
{
    return bandNum;
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

const std::size_t scf::BandPhaseFeedback::getNumCh() const
{
    return numCh;
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
        if (val > maxNumCh)
        {
            eLog_->error("Invalid channel number %zu at index %zu", val, i);

            // Do not update the 'toneCh' vector.
            return;
        }

        // A valid number was found. Add it to the temporal vector
        temp.push_back(val);
    }

    // Take the mutex before changing the 'toneCh' vector
    std::lock_guard<std::mutex> lock { mut };

    // At this point, all element in the mask list are valid.
    // Update the 'toneCh' vector
    toneCh.swap(temp);

    // Update the maximum tone channel
    maxToneCh = *std::max_element(toneCh.begin(), toneCh.end());

    // Check if the input parameters are valid
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

        temp.push_back(bp::extract<double>(m[i]));
    }

    // Take the mutex before changing the 'toneCh' vector
    std::lock_guard<std::mutex> lock { mut };

    // At this point, all element in the mask list are valid.
    // Update the 'toneFreq' vector
    toneFreq.swap(temp);

    // Update the frequency mean, deltas, and variance
    freqMean = 2 * M_PI * std::accumulate(toneFreq.begin(), toneFreq.end(), 0.0) / toneFreq.size();
    std::vector<double>().swap(freqDiffs);
    freqVar = 0;
    for (auto const &f : toneFreq)
    {
        double d { 2 * M_PI * f - freqMean };
        freqDiffs.push_back(d);
        freqVar += d*d;
    }

    // Check if the input parameters are valid
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

const bool scf::BandPhaseFeedback::getDataValid() const
{
    return dataValid;
}

void scf::BandPhaseFeedback::checkDataValid()
{
    // Check if the input parameters are valid, which has 2 conditions:
    // - Both the toneCh and toneFreq must have the same size, and
    // - The maximum channel in 'toneCh' is not greater that the number of
    //   channels in the input frame.
    if ( (toneCh.size() == toneFreq.size()) && (maxToneCh < numCh) )
        dataValid = true;
    else
        dataValid = false;
}

void scf::BandPhaseFeedback::acceptFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // Only process the frame is the block is enable.
    if (!disable)
    {
        // Acquire lock on frame.
        ris::FrameLockPtr frameLock { frame->lock() };

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

        // At this point the frame is valid

        // Update the frame counter
        ++frameCnt;

        // Construct a (smart) pointer to the SMuRF packet in the input frame (Read-only)
        SmurfPacketROPtr sp { SmurfPacketRO::create(frame) };

        // Read the number of channel from the header
        numCh = sp->getHeader()->getNumberChannels();

        // Take the mutex here, to avoid the tone parameters to change while they are used.
        std::lock_guard<std::mutex> lock { mut };

        // Check if the input parameters are valid
        checkDataValid();

        // If the input parameters are are not valid, we do not process the data.
        // We also set the tau and theta estimation to 0.
        if (!dataValid)
        {
            tau = 0.0;
            theta = 0.0;
            return;
        }

        // Extract the phase from the specified channels
        std::vector<int32_t> phase;
        for (auto const &c : toneCh)
            phase.push_back(sp->getData(c));

        // Estimate the band "tau" (slope) and "theta" (offset) parameters using least
        // square solution:
        //
        //     tau = sum_i(f_i - f_mean)(p_i - p_mean)/sum_i(f_i - f_mean)^2
        //
        //     theta = y_mean - m * x_mean
        //
        // where:
        //     f_i    : frequency points.
        //     f_mean : mean frequency.
        //     p_i    : phase points.
        //     p_mean : mean phase.

        // Calculate the mean phase
        double phaseMean { std::accumulate(phase.begin(), phase.end(), 0.0) / phase.size() };

        // Calculate tau (slope)
        tau = 0;
        for (std::size_t i {0}; i < phase.size(); ++i)
            tau += freqDiffs[i] * (phase[i] - phaseMean);
        tau /= freqVar;

        // Calculate theta (offset)
        theta = phaseMean - tau * freqMean;

    }

    // Send the frame to the next slave.
    sendFrame(frame);
}
