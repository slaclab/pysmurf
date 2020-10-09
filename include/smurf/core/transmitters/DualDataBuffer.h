#ifndef _SMURF_CORE_TRANSMITTERS_DUALDATABUFFER_H_
#define _SMURF_CORE_TRANSMITTERS_DUALDATABUFFER_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Base Transmitter
 * ----------------------------------------------------------------------------
 * File          : DualDataBuffer.h
 * Created       : 2019-11-11
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Dual Data Buffer Class.
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

#include <vector>
#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <functional>
#include <memory>
#include "smurf/core/common/SmurfPacket.h"

namespace smurf
{
    namespace core
    {
        namespace transmitters
        {
            template <typename T>
            class DualDataBuffer;

            template <typename T>
            using DualDataBufferPtr = std::shared_ptr< DualDataBuffer<T> >;

            // TX callback function pointer.
            // The function signature must be 'void(T)'
            template <typename T>
            using tx_func_t =  std::function<void(T)>;

            template <typename T>
            class DualDataBuffer
            {
            public:
            	// Constructor:
            	// - callbackFunc : A pointer to a function to be called when new data is ready.
                // - trheadname   : A name to be given to the txTransmit thread. Omitted if empty.
                DualDataBuffer(std::function<void(T)> callbackFunc, const std::string& threadName);
                ~DualDataBuffer() {};

                // Factory method
                static DualDataBufferPtr<T> create(std::function<void(T)> callbackFunc, const std::string& threadName);

                // Insert a new element in the buffer
                void insertData(const T& d);

                // Get the number of dropped elements
                const std::size_t getDropCnt() const;

                // Clear the counters
                void clearCnt();

            private:
                // Prevent construction using the default or copy constructor.
                // Prevent an DualDataBuffer object to be assigned as well.
                DualDataBuffer();
                DualDataBuffer(const DualDataBuffer&);
                DualDataBuffer& operator=(const DualDataBuffer&);

                std::size_t             dropCnt;      // Dropped element counter
                std::vector<T>          buffer;       // Dual buffers. Can hold two elements
                std::size_t             readIndex;    // Buffer position to be read
                std::size_t             writeIndex;   // Buffer position to be written
                std::size_t             dataCnt;      // Number of data elements in the buffer
                bool                    txDataReady;  // Flag to indicate new data is ready t be sent
                std::atomic<bool>       runTxThread;  // Flag used to stop the thread
                std::condition_variable txCV;         // Variable to notify the thread new data is ready
                std::mutex              txMutex;      // Mutex used for accessing the conditional variable
                std::thread             txThread;     // Thread where the SMuRF packet transmission will run

                // Transmit method. Will run in the 'txThread' thread.
                // Here is where the txFunc callback function will be called.
                void txTransmitter();

                // TX callback function.
                tx_func_t<T> txFunc;
            };
        }
    }
}

#endif