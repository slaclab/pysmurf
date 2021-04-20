#ifndef _SMURF_CORE_TRANSMITTERS_BASETRANSMITTER_H_
#define _SMURF_CORE_TRANSMITTERS_BASETRANSMITTER_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Base Transmitter
 * ----------------------------------------------------------------------------
 * File          : BaseTransmitter.h
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data Base Transmitter Class.
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

#include <functional>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/Buffer.h>
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/GilRelease.h>
#include "smurf/core/common/SmurfHeader.h"
#include "smurf/core/common/SmurfPacket.h"
#include "smurf/core/transmitters/BaseTransmitterChannel.h"

namespace bp  = boost::python;
namespace ris = rogue::interfaces::stream;

namespace smurf
{
    namespace core
    {
        namespace transmitters
        {
            class BaseTransmitter;
            typedef std::shared_ptr<BaseTransmitter> BaseTransmitterPtr;

            // TX callback function pointer.
            // The function signature must be 'void(T)'
            template <typename T>
            using tx_func_t =  std::function<void(T)>;

            class BaseTransmitter: public std::enable_shared_from_this<smurf::core::transmitters::BaseTransmitter>
            {
            public:
                BaseTransmitter();
                virtual ~BaseTransmitter() {};

                static BaseTransmitterPtr create();

                static void setup_python();

                // Disable the processing block. The data
                // will just pass through to the next slave
                void       setDisable(bool d);
                const bool getDisable() const;

                // Get data channel
                BaseTransmitterChannelPtr getDataChannel();

                // Get meta data channel
                BaseTransmitterChannelPtr getMetaChannel();

                // Clear all counter.
                void clearCnt();

                // Get the data dropped counter
                const std::size_t getDataFrameCnt() const;

                // Get the metadata dropped counter
                const std::size_t getMetaFrameCnt() const;

                // Get the data dropped counter
                const std::size_t getDataDropCnt() const;

                // Get the metadata dropped counter
                const std::size_t getMetaDropCnt() const;

                // Accept new data frames
                void acceptDataFrame(ris::FramePtr frame);

                // Accept new meta frames
                void acceptMetaFrame(ris::FramePtr frame);

                // This method is intended to be used to take SMuRF packet and send them to other
                // systems.
                // This method is called whenever a new SMuRF packet is ready, and a SmurfPacketROPtr object
                // (which is a smart pointer to a read-only interface to a Smurf packer object) is passed.
                // It must be overwritten by the user application
                virtual void dataTransmit(SmurfPacketROPtr sp) {};

                // This method is intended to be used to take SMuRF meta data and send them to other
                // system.
                // This method is called whenever new a new metadata frame is ready, which is passed as a
                // std::string object.
                // It must be overwritten by the user application
                virtual void metaTransmit(std::string cfg) {};

            private:
                bool                      disable;          // Disable flag
                BaseTransmitterChannelPtr dataChannel;      // Data channel interface
                BaseTransmitterChannelPtr metaChannel;      // Metadata channel interface
                std::size_t               dataFrameCnt;     // Number of received data frames
                std::size_t               metaFrameCnt;     // Number of received metadata frames
                std::size_t               dataDropFrameCnt; // Number of dropped data frames
                std::size_t               metaDropFrameCnt; // Number of dropped metadata frames

                // TX callback functions.
                tx_func_t<SmurfPacketROPtr> txDataFunc;
                tx_func_t<std::string>      txMetaFunc;
            };
        }
    }
}

#endif
