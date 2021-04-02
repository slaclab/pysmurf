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
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/interfaces/stream/Master.h>
#include <rogue/GilRelease.h>
#include "smurf/core/common/SmurfHeader.h"

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
                BandPhaseFeedback();
                ~BandPhaseFeedback() {};

                static BandPhaseFeedbackPtr create();

                static void setup_python();

                // Disable the processing block. The data
                // will just pass through to the next slave
                void       setDisable(bool d);
                const bool getDisable() const;

                // Get the frame counter
                const std::size_t getFrameCnt() const;

                // Get the number of bad frames
                const std::size_t getBadFrameCnt() const;

                // Clear all counter.
                void clearCnt();

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

            private:
                bool        disable;           // Disable flag
                std::size_t frameCnt;          // Frame counter
                std::size_t badFrameCnt;       // Number of frames with errors

                // Logger
                std::shared_ptr<rogue::Logging> eLog_;
            };
        }
    }
}

#endif
