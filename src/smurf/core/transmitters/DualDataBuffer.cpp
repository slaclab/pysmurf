/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Base Transmitter
 * ----------------------------------------------------------------------------
 * File          : DualDataBuffer.cpp
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

#include "smurf/core/transmitters/DualDataBuffer.h"

namespace sct = smurf::core::transmitters;

template <typename T>
sct::DualDataBuffer<T>::DualDataBuffer(std::function<void(T)> callbackFunc, const std::string& threadName)
:
    dropCnt(0),
    buffer(2),
    readIndex(0),
    writeIndex(0),
    dataCnt(0),
    txDataReady(false),
    runTxThread(true),
    txThread(std::thread( &DualDataBuffer::txTransmitter, this)),
    txFunc(callbackFunc)
{
    if (!threadName.empty())
    {
        if( pthread_setname_np( txThread.native_handle(), threadName.c_str() ) )
            perror( "pthread_setname_np failed for the SmurfPacketTx thread" );
    }
}

template <typename T>
sct::DualDataBuffer<T>::~DualDataBuffer()
{
    runTxThread = false;
    txThread.join();
}

template <typename T>
sct::DualDataBufferPtr<T> sct::DualDataBuffer<T>::create(std::function<void(T)> callbackFunc, const std::string& threadName)
{
    return std::make_shared< DualDataBuffer<T> >(callbackFunc, threadName);
}

template <typename T>
void sct::DualDataBuffer<T>::clearCnt()
{
    dropCnt = 0;
}

template <typename T>
const std::size_t sct::DualDataBuffer<T>::getDropCnt() const
{
    return dropCnt;
}

template <typename T>
void sct::DualDataBuffer<T>::insertData(const T& data)
{
	// Check if the buffer is not full.
    // If the buffer is full, the data will be dropped
    if (dataCnt < 2)
    {
        // Insert a new element into the buffer
        buffer.at(writeIndex) = data;

        // Update the write position index
        writeIndex ^= 1;

        // Increment the number of elements in the buffer
        ++dataCnt;

        // Notify the TX tread that new data is ready to be send
        txDataReady = true;
        std::unique_lock<std::mutex> lock{txMutex};
        txCV.notify_all();
    }
    else
    {
        // Increase the dropped counter
        ++dropCnt;
    }
}

template <typename T>
void sct::DualDataBuffer<T>::txTransmitter()
{
    // Infinite loop
    for(;;)
    {
        // Check if new data is ready
        if ( !txDataReady )
        {
            // Wait until data is ready, with a 1s timeout
            std::unique_lock<std::mutex> lock(txMutex);
            txCV.wait_for( lock, std::chrono::seconds(1) );
        }
        else
        {
            // Call the transmit callback function here
            txFunc(buffer.at(readIndex));

            // Update the read position index
            readIndex ^= 1;

            // Decrement the number of elements in the buffer
            --dataCnt;

            // Cleat the flag
            txDataReady = false;
        }

        // Check if we should stop the loop
        if (!runTxThread)
            return;
    }
}

template class sct::DualDataBuffer<std::string>;
template class sct::DualDataBuffer<SmurfPacketROPtr>;
