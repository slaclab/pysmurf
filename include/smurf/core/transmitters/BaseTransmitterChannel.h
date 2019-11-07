#ifndef _SMURF_CORE_TRANSMITTERS_BASETRANSMITTER_CHANNEL_H_
#define _SMURF_CORE_TRANSMITTERS_BASETRANSMITTER_CHANNEL_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Base Transmitter Channel
 * ----------------------------------------------------------------------------
 * File          : BaseTransmitterChannel.h
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data Base Transmitter Class, channel ineterface
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
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/GilRelease.h>

namespace bp  = boost::python;
namespace ris = rogue::interfaces::stream;

namespace smurf
{
    namespace core
    {
        namespace transmitters
        {
            class BaseTransmitterChannel;
            typedef std::shared_ptr<BaseTransmitterChannel> BaseTransmitterChannelPtr;

            class BaseTransmitter;

            class BaseTransmitterChannel : public ris::Slave
            {
            public:
                BaseTransmitterChannel(std::shared_ptr<smurf::core::transmitters::BaseTransmitter> bt, uint32_t channel);
                ~BaseTransmitterChannel() {};

                static BaseTransmitterChannelPtr create(std::shared_ptr<smurf::core::transmitters::BaseTransmitter> bt, uint32_t channel);

                static void setup_python();

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

            private:

                uint32_t channel_;

                std::shared_ptr<smurf::core::transmitters::BaseTransmitter> bt_;

            };
        }
    }
}

#endif
