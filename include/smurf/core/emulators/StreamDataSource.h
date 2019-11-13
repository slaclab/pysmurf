#ifndef _SMURF_CORE_EMULATORS_STREAMDATASOURCE_H_
#define _SMURF_CORE_EMULATORS_STREAMDATASOURCE_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Source
 * ----------------------------------------------------------------------------
 * File          : StreamDataSource.h
 * Created       : 2019-11-12
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data StreamDataSource Class
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

#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/FrameAccessor.h>
#include <rogue/interfaces/stream/Master.h>
#include <rogue/GilRelease.h>
#include <rogue/Logging.h>
#include "smurf/core/common/SmurfHeader.h"
#include "smurf/core/common/Helpers.h"
#include <stdint.h>
#include <thread>

namespace bp  = boost::python;
namespace ris = rogue::interfaces::stream;

namespace smurf
{
    namespace core
    {
        namespace emulators
        {
            class StreamDataSource;
            typedef std::shared_ptr<StreamDataSource> StreamDataSourcePtr;

            class StreamDataSource : public ris::Master
            {
            public:
                StreamDataSource();
                ~StreamDataSource();

                static StreamDataSourcePtr create();

                static void setup_python();

                // Frame rate period in us
                void     setSourcePeriod(uint16_t value);
                uint16_t getSourcePeriod();

                void     setSourceEnable(bool enable);
                bool     getSourceEnable();

                void     setCrateId(uint8_t value);
                uint8_t  getCrateId();

                void     setSlotNum(uint8_t value);
                uint8_t  getSlotNum();

            private:

               std::shared_ptr<rogue::Logging> eLog_;

               uint16_t sourcePeriod_;
               bool     sourceEnable_;
               uint8_t  crateId_;
               uint8_t  slotNumber_;
               uint32_t frameCounter_;

               std::thread* thread_;
               bool threadEn_;

               void runThread();

            };
        }
    }
}

#endif
