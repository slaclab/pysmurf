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

    // Check if the frame size is correct
    if ( SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize + ( dataSize * sizeof(data_t) ) != frameSize )
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

