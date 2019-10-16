#ifndef _SMURF_CORE_COUNTERS_FRAMESTATISTICS_H_
#define _SMURF_CORE_COUNTERS_FRAMESTATISTICS_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Frame Statistics Module
 * ----------------------------------------------------------------------------
 * File          : FrameStatistics.h
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Frame Statistics Class.
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
        namespace counters
        {
            class FrameStatistics;
            typedef std::shared_ptr<FrameStatistics> FrameStatisticsPtr;

            class FrameStatistics : public ris::Slave, public ris::Master
            {
            public:
                FrameStatistics();
                ~FrameStatistics() {};

                static FrameStatisticsPtr create();

                static void setup_python();

                // Disable the processing block. The data
                // will just pass through to the next slave
                void       setDisable(bool d);
                const bool getDisable() const;

                // Get the frame counter
                const std::size_t getFrameCnt() const;

                // Get the last frame size (in bytes)
                const std::size_t getFrameSize() const;

                // Get number of lost frames
                const std::size_t getFrameLossCnt() const;

                // Get the number of out-of-order frames
                const std::size_t getFrameOutOrderCnt() const;

                // Clear all counter.
                void clearCnt();

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

            private:
                bool        disable;           // Disable flag
                std::size_t frameCnt;          // Frame counter
                std::size_t frameSize;         // Last frame size (bytes)
                bool        firstFrame;        // Flag to indicate we are processing the first frame
                std::size_t frameLossCnt;      // Number of frame lost
                std::size_t frameOutOrderCnt;  // Counts the number of times we received an out-of-order frame
                std::size_t frameNumber;       // Current frame number
                std::size_t prevFrameNumber;   // Last frame number
            };
        }
    }
}

#endif
