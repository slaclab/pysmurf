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

////////////////////////////////////////////
// SmurfPacketManagerRO class definitions //
////////////////////////////////////////////
template<typename CreationPolicy>
SmurfPacketManagerRO<CreationPolicy>::SmurfPacketManagerRO(ris::FramePtr frame)
:
    CreationPolicy(frame)
{
}

template<typename CreationPolicy>
SmurfPacketManagerROPtr<CreationPolicy> SmurfPacketManagerRO<CreationPolicy>::create(ris::FramePtr frame)
{
    return std::make_shared<SmurfPacketManagerRO>(frame);
}

///////////////////////////////////
// CopyCreator class definitions //
///////////////////////////////////
CopyCreator::CopyCreator(ris::FramePtr frame)
:
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
    std::size_t dataSize { headerPtr->getNumberChannels() };

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

CopyCreator::HeaderPtr CopyCreator::getHeader() const
{
    return headerPtr;
}

const CopyCreator::data_t CopyCreator::getData(std::size_t index) const
{
    return data.at(index);
}

///////////////////////////////////////
// ZeroCopyCreator class definitions //
///////////////////////////////////////
ZeroCopyCreator::ZeroCopyCreator(ris::FramePtr frame)
:
    framePtr(frame),
    headerPtr(SmurfHeaderRO<ris::FrameIterator>::create(frame)),
    dataSize(dataSize = headerPtr->getNumberChannels())
{
    // Verify that the frame data area is contained in a single buffer
    ris::FrameIterator frameIt { framePtr->beginRead() };
    frameIt += SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize;
    if ( dataSize*sizeof(data_t) > frameIt.remBuffer() )
        throw std::runtime_error("Trying to create a SmurfPacket object on a multi-buffer frame");

    // Point the data pointer to the beginning of the frame data area
    data = reinterpret_cast<data_t*>(frameIt.ptr());
}

ZeroCopyCreator::HeaderPtr ZeroCopyCreator::getHeader() const
{
    return headerPtr;
}

const ZeroCopyCreator::data_t ZeroCopyCreator::getData(std::size_t index) const
{
    // Verify that the index in not out of the packet range
    if (index >= dataSize)
        throw std::runtime_error("Trying to read data from a SmurfPacket with an index out of range.");

    return data[index];
}

std::vector<ZeroCopyCreator::data_t> ZeroCopyCreator::getAllData() const
{
    return std::vector<data_t>(data, data+dataSize);
}

// Explicit template instantiations
template class SmurfPacketManagerRO<CopyCreator>;
template class SmurfPacketManagerRO<ZeroCopyCreator>;
