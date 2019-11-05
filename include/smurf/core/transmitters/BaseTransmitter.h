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

#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/Buffer.h>
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/GilRelease.h>
#include "smurf/core/common/SmurfHeader.h"
#include "smurf/core/common/SmurfPacket.h"

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

            class BaseTransmitter : public ris::Slave
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

                // Clear all counter.
                void clearCnt();

                // Get the dropped packet counter
                const std::size_t getPktDropCnt() const;

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

                // This method is intended to be used to take SMuRF packet and send them to other
                // system.
                // This method is called whenever a new SMuRF packet is ready, and a SmurfPacketROPtr object
                // (which is a smart pointer to a read-only interface to a Smurf packer object) is passed.
                // It must be overwritten by the user application
                virtual void transmit(SmurfPacketROPtr sp) {};

            private:
                bool                          disable;              // Disable flag
                std::size_t                   pktDropCnt;           // Dropped packet counter
                std::vector<SmurfPacketROPtr> pktBuffer;            // Dual buffer of Smurf packets. Can hold 2 packets.
                std::size_t                   writeIndex;           // Buffer position to be written
                std::size_t                   readIndex;            // Buffer position to be read
                std::size_t                   pktCount;             // Number of packets in the buffer
                bool                          txDataReady;          // Flag to indicate new data is ready t be sent
                std::atomic<bool>             runTxThread;          // Flag used to stop the thread
                std::thread                   pktTransmitterThread; // Thread where the SMuRF packet transmission will run
                std::condition_variable       txCV;                 // Variable to notify the thread new data is ready
                std::mutex                    txMutex;              // Mutex used for accessing the conditional variable

                // Transmit method. Will run in the pktTransmitterThread thread.
                // Here is where the method 'transmit' is called.
                void pktTansmitter();

            };
        }
    }
}

#endif
