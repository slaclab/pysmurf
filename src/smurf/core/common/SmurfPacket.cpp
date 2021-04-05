/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Packet
 * ----------------------------------------------------------------------------
 * File          : SmurfPacket.cpp
 * Created       : 2019-10-29
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Packet Class.
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

#include "smurf/core/common/SmurfPacket.h"

//////////////////////////////////////////
////// + SmurfPacketRO definitions ///////
//////////////////////////////////////////

SmurfPacketRO::SmurfPacketRO(ris::FramePtr frame)
:
    dataSize(0),
    header(SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize)
{
    // Get the frame size
    std::size_t frameSize { frame->getPayload() };

    // Check if frame is at least the size of the header
    if ( SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize > frameSize )
        throw std::runtime_error("Trying to construct a SmurfPacket object on a frame with size lower that the header size.");

    ris::FrameIterator it { frame->beginRead() };

    // Copy the header
    std::copy(it, it + SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize, header.begin());

    // Create a header (smart) pointer on the buffer
    headerPtr = SmurfHeaderRO< std::vector<uint8_t>::iterator >::create(header);

    // Get the number of data channels in the packet and reserve space in the data buffer
    dataSize = headerPtr->getNumberChannels();

    // Check if the frame size is at least enough to hold the number of channels defined in this header.
    // Frame padded to increase its size are allowed.
    if ( SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize + ( dataSize * sizeof(data_t) ) > frameSize )
        throw std::runtime_error("Trying to construct a SmurfPacket object on a frame with bad size.");

    // Allocate space in the data vector to hold the data
    data.reserve(dataSize);

    // Move the iterator to the data area
    it += SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize;

    // Copy the data
    for (std::size_t i{0}; i < dataSize; ++i)
        data.push_back( *(reinterpret_cast<data_t*>( &(*(it + i * sizeof(data_t)) ) ) ) );
}

SmurfPacketROPtr SmurfPacketRO::create(ris::FramePtr frame)
{
    return std::make_shared<SmurfPacketRO>(frame);
}

SmurfPacketRO::HeaderPtr SmurfPacketRO::getHeader() const
{
    return headerPtr;
}

const SmurfPacketRO::data_t SmurfPacketRO::getData(std::size_t index) const
{
    return data.at(index);
}

//////////////////////////////////////////////////
////// + SmurfPacketZeroCopyRO definitions ///////
//////////////////////////////////////////////////

SmurfPacketZeroCopyRO::SmurfPacketZeroCopyRO(ris::FramePtr frame)
:
    _frame(frame),
    headerPtr(SmurfHeaderRO<ris::FrameIterator>::create(frame)),
    dataSize(dataSize = headerPtr->getNumberChannels())
{
    // Verify that the frame data area is contained in a single buffer
    ris::FrameIterator frameIt { _frame->beginRead() };
    frameIt += SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize;
    if ( dataSize*sizeof(data_t) > frameIt.remBuffer() )
        throw std::runtime_error("Trying to create a SmurfPacket object on a multi-buffer frame");

    // Point the data pointer to the beginning of the frame data area
    data = reinterpret_cast<data_t*>(frameIt.ptr());
}

SmurfPacketZeroCopyROPtr SmurfPacketZeroCopyRO::create(ris::FramePtr frame)
{
    return std::make_shared<SmurfPacketZeroCopyRO>(frame);
}

SmurfPacketZeroCopyRO::HeaderPtr SmurfPacketZeroCopyRO::getHeader() const
{
    return headerPtr;
}

const SmurfPacketZeroCopyRO::data_t SmurfPacketZeroCopyRO::getData(std::size_t index) const
{
    // Verify that the index in not out of the packet range
    if (index > dataSize)
        throw std::runtime_error("Trying to read data from a SmurfPacket with an index out of range.");

    return data[index];
}