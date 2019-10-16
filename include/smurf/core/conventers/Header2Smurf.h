#ifndef _SMURF_CORE_CONVENTERS_Header2Smurf_H_
#define _SMURF_CORE_CONVENTERS_Header2Smurf_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Header2Smurf
 * ----------------------------------------------------------------------------
 * File          : Header2Smurf.h
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

#include <iostream>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/interfaces/stream/Master.h>
#include <rogue/GilRelease.h>
#include "smurf/core/common/SmurfHeader.h"
#include "smurf/core/common/TesBiasArray.h"
#include "smurf/core/common/Helpers.h"

namespace bp  = boost::python;
namespace ris = rogue::interfaces::stream;

namespace smurf
{
    namespace core
    {
        namespace conventers
        {
            class Header2Smurf;
            typedef std::shared_ptr<Header2Smurf> Header2SmurfPtr;

            // This class converts the header in the frame to the Smurf Header
            class Header2Smurf : public ris::Slave, public ris::Master
            {
            public:
                Header2Smurf();
                ~Header2Smurf() {};

                static Header2SmurfPtr create();

                static void setup_python();

                // Disable the processing block. The data
                // will just pass through to the next slave
                void       setDisable(bool d);
                const bool getDisable() const;

                // Receive the TesBias from pyrogue
                void setTesBias(std::size_t index, int32_t value);

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

            private:
                bool            disable; // Disable flag
                ris::FramePtr   tesBias;
                TesBiasArrayPtr<ris::FrameIterator> tba;
            };
        }
    }
}

#endif
