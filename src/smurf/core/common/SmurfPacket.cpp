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
    ris::FrameIterator it{frame->beginRead()};

    // Copy the header
    std::copy(it, it + SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize, header.begin());

    headerPtr = SmurfHeaderRO< std::vector<uint8_t>::iterator >::create(header);

    // Get the number of data channels in the packet and reserve space in the data buffer
    dataSize = headerPtr->getNumberChannels();

    // Allocate space in the dat vector to hold the data
    data.reserve(dataSize);

    // Move the iterator to the data area
    it += SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize;

    // Copy the data
    for (std::size_t i{0}; i < dataSize; ++i)
        data.push_back( static_cast<int32_t>( *(it + i * sizeof(int32_t)) ) );
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

