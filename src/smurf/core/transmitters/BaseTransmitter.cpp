/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Base Transmitter
 * ----------------------------------------------------------------------------
 * File          : BaseTransmitter.cpp
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *   SMuRF Data Base Transmitter Class.
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
#include <boost/python.hpp>
#include "smurf/core/transmitters/BaseTransmitter.h"

namespace bp  = boost::python;
namespace sct = smurf::core::transmitters;

sct::BaseTransmitter::BaseTransmitter()
:
    disable(false),
    pktDropCnt(0),
    pktBuffer(2),
    writeIndex(0),
    readIndex(0),
    pktCount(0),
    txDataReady(false),
    runTxThread(true),
    pktTransmitterThread(std::thread( &BaseTransmitter::pktTansmitter, this ))
{
    if( pthread_setname_np( pktTransmitterThread.native_handle(), "SmurfPacketTx" ) )
        perror( "pthread_setname_np failed for the SmurfPacketTx thread" );
}

sct::BaseTransmitterPtr sct::BaseTransmitter::create()
{
    return std::make_shared<BaseTransmitter>();
}

void sct::BaseTransmitter::setup_python()
{
    bp::class_< sct::BaseTransmitter,
                sct::BaseTransmitterPtr,
                boost::noncopyable >
                ("BaseTransmitter",bp::init<>())
        .def("setDisable",     &BaseTransmitter::setDisable)
        .def("getDisable",     &BaseTransmitter::getDisable)
        .def("clearCnt",       &BaseTransmitter::clearCnt)
        .def("getPktDropCnt",  &BaseTransmitter::getPktDropCnt)
        .def("getDataChannel", &BaseTransmitter::getDataChannel)
        .def("getMetaChannel", &BaseTransmitter::getMetaChannel)
    ;
}

// Get data channel
sct::BaseTransmitterChannelPtr sct::BaseTransmitter::getDataChannel()
{
    // Create the dataChanenl object the first time this is called
    if (!dataChannel)
        dataChannel = sct::BaseTransmitterChannel::create(shared_from_this(),0);

    return dataChannel;
}

// Get meta data channel
sct::BaseTransmitterChannelPtr sct::BaseTransmitter::getMetaChannel()
{
    // Create the metaChanenl object the first time this is called
    if (!metaChannel)
        metaChannel = sct::BaseTransmitterChannel::create(shared_from_this(),1);

    return metaChannel;
}

void sct::BaseTransmitter::setDisable(bool d)
{
    disable = d;
}

const bool sct::BaseTransmitter::getDisable() const
{
    return disable;
}

void sct::BaseTransmitter::clearCnt()
{
    pktDropCnt = 0;
}

const std::size_t sct::BaseTransmitter::getPktDropCnt() const
{
    return pktDropCnt;
}

void sct::BaseTransmitter::acceptDataFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // If the processing block is disabled, do not process the frame
    if (disable)
        return;

    // When a new packet is received, add it to the dual buffer. This will
    // allow to prepare a new packet while the previous one is still being
    // transmitted. The packet will be transmitted in a different thread.

    // Check if the buffer is not full.
    // If the buffer is full, the packet will be dropped
    if (pktCount < 2)
    {
        // Add a new packet into the buffer
        pktBuffer.at(writeIndex) = SmurfPacketRO::create(frame);

        // Update the write position index
        writeIndex ^= 1;

        // Increment the number of packet in the buffer
        ++pktCount;

        // Notify the TX tread that new data is ready to be send
        txDataReady = true;
        std::unique_lock<std::mutex> lock{txMutex};
        txCV.notify_all();
    }
    else
    {
        // Increase the dropped packet counter
        ++pktDropCnt;
    }
}

void sct::BaseTransmitter::acceptMetaFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;
    ris::FrameLockPtr fLock = frame->lock();

    if ( frame->bufferCount() != 1 ) return;

    std::string cfg(reinterpret_cast<char const*>(frame->beginRead().ptr()), frame->getPayload());
    fLock->unlock();

    metaTransmit(cfg);
}

void sct::BaseTransmitter::pktTansmitter()
{
    std::cout << "SmurfPacket Transmitter thread started..." << std::endl;

    // Infinite loop
    for(;;)
    {
        // Check if new data is ready
        if ( !txDataReady )
        {
            // Wait until data is ready, with a 10s timeout
            std::unique_lock<std::mutex> lock(txMutex);
            txCV.wait_for( lock, std::chrono::seconds(10) );
        }
        else
        {
            // Call the transmit method here
            transmit( pktBuffer.at(readIndex) );

            // Update the read position index
            readIndex ^= 1;

            // Decrement the number of packets in the buffer
            --pktCount;

            // Cleat the flag
            txDataReady = false;
        }

        // Check if we should stop the loop
        if (!runTxThread)
        {
            std::cout << "SmurfPacket Transmitter thread interrupted." << std::endl;
            return;
        }
    }
}
