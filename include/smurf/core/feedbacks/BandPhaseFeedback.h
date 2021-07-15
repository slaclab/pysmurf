#ifndef _SMURF_CORE_FEEDBACKS_BANDPHASEFEEDBACK_H_
#define _SMURF_CORE_FEEDBACKS_BANDPHASEFEEDBACK_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Band Phase Feedback Module
 * ----------------------------------------------------------------------------
 * File          : BandPhaseFeedback.h
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

#include <iostream>
#include <numeric>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/interfaces/stream/Master.h>
#include <rogue/GilRelease.h>
#include "smurf/core/common/SmurfPacket.h"

namespace bp  = boost::python;
namespace ris = rogue::interfaces::stream;

namespace smurf
{
    namespace core
    {
        namespace feedbacks
        {
            class BandPhaseFeedback;
            typedef std::shared_ptr<BandPhaseFeedback> BandPhaseFeedbackPtr;

            class BandPhaseFeedback : public ris::Slave, public ris::Master
            {
            public:
                BandPhaseFeedback(std::size_t band);
                ~BandPhaseFeedback() {};

                static BandPhaseFeedbackPtr create(std::size_t band);

                static void setup_python();

                // Disable the processing block. The data
                // will just pass through to the next slave
                void       setDisable(bool d);
                const bool getDisable() const;

                // Get the band number
                const std::size_t getBand() const;

                // Get the frame counter
                const std::size_t getFrameCnt() const;

                // Get the number of bad frames
                const std::size_t getBadFrameCnt() const;

                // Clear all counter.
                void clearCnt();

                // Get the number of channels in the incoming frame
                const std::size_t getNumCh() const;

                // Set/Get the tone channels
                void           setToneChannels(bp::list m);
                const bp::list getToneChannels() const;

                // Set/Get the tone frequencies
                void           setToneFrequencies(bp::list m);
                const bp::list getToneFrequencies() const;

                // Get the raw tone phases
                const bp::list getTonePhase() const;

                // Get the band phase parameters estimations
                const double getTau()   const;
                const double getTheta() const;

                // Get the estimation coefficient of determination
                const double getR2() const;

                // Get the ready flag
                const bool getReady() const;

                // Check if all settings are valid
                void checkReady();

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

            private:
                // Minimum number of tones.
                static const std::size_t minNumTones = 2;

                // Maximum number of band.
                // The minimum number is 0, and as this is a unsigned variable,
                // we don't need to explicitly check it.
                static const std::size_t maxBandNum = 7;

                // This is the maximum allowed channel index
                static const std::size_t maxNumCh = 4096;

                bool                        disable;        // Disable flag
                const std::size_t           bandNum;        // Band number
                const double                minToneFreq;    // Minimum tone frequency
                const double                maxToneFreq;    // Maximum tone frequency
                std::size_t                 frameCnt;       // Frame counter
                std::size_t                 badFrameCnt;    // Number of frames with errors
                std::size_t                 numCh;          // Number of channels in the incoming frame
                std::vector<std::size_t>    toneCh;         // Tone channels
                std::vector<double>         toneFreq;       // Tone Frequencies
                std::vector<int32_t>        tonePhase;      // Tone raw phases
                std::size_t                 maxToneCh;      // Maximum channel in the 'toneCh' vector.
                bool                        ready;          // Flag to indicate that all conditions are valid
                double                      tau;            // Band phase slope estimation (tau).
                double                      theta;          // Band phase offset estimation (theta).
                double                      R2;             // Coefficient of determination of the estimation
                double                      freqMean;       // Mean frequency
                std::vector<double>         freqDiffs;      // Frequencies deltas respect to the mean value
                double                      freqVar;        // Frequency variance
                std::mutex                  mut;            // Mutex

                // Logger
                std::shared_ptr<rogue::Logging> eLog_;
            };
        }
    }
}

#endif
