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

#include <boost/python.hpp>
#include "smurf/core/common/Timer.h"
#include "smurf/core/transmitters/BaseTransmitter.h"

namespace bp  = boost::python;
namespace sct = smurf::core::transmitters;

sct::BaseTransmitter::BaseTransmitter()
:
    ris::Slave(),
    disable(false),
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
                bp::bases<ris::Slave>,
                boost::noncopyable >
                ("BaseTransmitter",bp::init<>())
        .def("setDisable", &BaseTransmitter::setDisable)
        .def("getDisable", &BaseTransmitter::getDisable)
    ;
    bp::implicitly_convertible< sct::BaseTransmitterPtr, ris::SlavePtr >();
}

void sct::BaseTransmitter::setDisable(bool d)
{
    disable = d;
}

const bool sct::BaseTransmitter::getDisable() const
{
    return disable;
}

void sct::BaseTransmitter::acceptFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // If the processing block is disabled, do not process the frame
    if (disable)
        return;

    // When a new packet is recived, add it to the dual buffer. This will
    // allow to prepare a new packet while the previous one is still being
    // transmitted. The packet will be transmitted in a different thread.

    // Check if the buffer is not full.
    // If the buffer is full, the packet will be droped
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

            // Update the read position idnex
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
