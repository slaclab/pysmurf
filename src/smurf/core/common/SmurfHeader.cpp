/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Header (on a frame)
 * ----------------------------------------------------------------------------
 * File          : SmurfHeader.cpp
 * Created       : 2019-10-01
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Header Class.
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

#include "smurf/core/common/SmurfHeader.h"

//////////////////////////////////////////
////// + SmurfHeaderRO definitions ///////
//////////////////////////////////////////

template<>
SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderRO(ris::FramePtr frame)
:
    headerIt(frame->beginRead()),
    tba(TesBiasArray<ris::FrameIterator>::create(headerIt + this->headerTESDACOffset))
{
}

template<>
SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderRO(std::vector<uint8_t>& buffer)
:
    headerIt(buffer.begin()),
    tba(TesBiasArray<std::vector<uint8_t>::iterator>::create(headerIt + this->headerTESDACOffset))
{
}

template<>
SmurfHeaderROPtr<ris::FrameIterator> SmurfHeaderRO<ris::FrameIterator>::create(ris::FramePtr frame)
{
    return std::make_shared<SmurfHeaderRO<ris::FrameIterator>>(frame);
}

template<>
SmurfHeaderROPtr<std::vector<uint8_t>::iterator> SmurfHeaderRO<std::vector<uint8_t>::iterator>::create(std::vector<uint8_t>& buffer)
{
    return std::make_shared< SmurfHeaderRO<std::vector<uint8_t>::iterator> >(buffer);
}

// Function to get header words
template<typename T>
const uint8_t SmurfHeaderRO<T>::getVersion() const
{
    return getU8Word(headerVersionOffset);
}

template<typename T>
const uint8_t SmurfHeaderRO<T>::getCrateID() const
{
    return getU8Word(headerCrateIDOffset);
}

template<typename T>
const uint8_t SmurfHeaderRO<T>::getSlotNumber() const
{
    return getU8Word(headerSlotNumberOffset);
}

template<typename T>
const uint8_t SmurfHeaderRO<T>::getTimingConfiguration() const
{
    return getU8Word(headerTimingConfigurationOffset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getNumberChannels() const
{
    return getU32Word(headerNumberChannelOffset);
}

template<typename T>
const int32_t SmurfHeaderRO<T>::getTESBias(std::size_t index) const
{
    return tba->getWord(index);
}

template<typename T>
void SmurfHeaderRO<T>::copyTESBiasArrayTo(T it) const
{
    std::copy(it + headerTESDACOffset,
        it + headerTESDACOffset + TesBiasArray<T>::TesBiasBufferSize,
        headerIt);
}

template<typename T>
const uint64_t SmurfHeaderRO<T>::getUnixTime() const
{
    return getU64Word(headerUnixTimeOffset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getFluxRampIncrement() const
{
    return getU32Word(headerFluxRampIncrementOffset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getFluxRampOffset() const
{
    return getU32Word(headerFluxRampOffsetOffset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getCounter0() const
{
    return getU32Word(headerCounter0Offset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getCounter1() const
{
    return getU32Word(headerCounter1Offset);
}

template<typename T>
const uint64_t SmurfHeaderRO<T>::getCounter2() const
{
    return getU64Word(headerCounter2Offset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getAveragingResetBits() const
{
    return getU32Word(headerAveragingResetBitsOffset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getFrameCounter() const
{
    return getU32Word(headerFrameCounterOffset);
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getTESRelaySetting() const
{
    return getU32Word(headerTESRelaySettingOffset);
}

template<typename T>
const uint64_t SmurfHeaderRO<T>::getExternalTimeClock() const
{
    return getU64Word(headerExternalTimeClockOffset);
}

template<typename T>
const uint8_t SmurfHeaderRO<T>::getControlField() const
{
    return getU8Word(headerControlFieldOffset);
}

template<typename T>
const bool SmurfHeaderRO<T>::getClearAverageBit() const
{
    return getWordBit(headerControlFieldOffset, clearAvergaveBitOffset);
}

template<typename T>
const bool SmurfHeaderRO<T>::getDisableStreamBit() const
{
    return getWordBit(headerControlFieldOffset, disableStreamBitOffset);
}

template<typename T>
const bool SmurfHeaderRO<T>::getDisableFileWriteBit() const
{
    return getWordBit(headerControlFieldOffset, disableFileWriteBitOffset);
}

template<typename T>
const bool SmurfHeaderRO<T>::getReadConfigEachCycleBit() const
{
    return getWordBit(headerControlFieldOffset, readConfigEachCycleBitOffset);
}

template<typename T>
const uint8_t SmurfHeaderRO<T>::getTestMode() const
{
    return ( ( getU8Word(headerControlFieldOffset) >> 4 ) & 0x0f );
}

template<typename T>
const uint8_t SmurfHeaderRO<T>::getTestParameters() const
{
    return getU8Word(headerTestParametersOffset);
}

template<typename T>
const uint16_t SmurfHeaderRO<T>::getNumberRows() const
{
    return getU16Word(headerNumberRowsOffset);
}

template<typename T>
const uint16_t SmurfHeaderRO<T>::getNumberRowsReported() const
{
    return getU16Word(headerNumberRowsReportedOffset);
}

template<typename T>
const uint16_t SmurfHeaderRO<T>::getRowLength() const
{
    return getU16Word(headerRowLengthOffset);
}

template<typename T>
const uint16_t SmurfHeaderRO<T>::getDataRate() const
{
    return getU16Word(headerDataRateOffset);
}

// Helper functions
template<typename T>
const uint8_t SmurfHeaderRO<T>::getU8Word(std::size_t offset) const
{
    return *(headerIt+offset);
}

template<typename T>
const uint16_t SmurfHeaderRO<T>::getU16Word(std::size_t offset) const
{
    union
    {
        uint16_t w;
        uint8_t  b[2];
    } aux;

    for (std::size_t i{0}; i < 2; ++i)
        aux.b[i] = *(headerIt+offset+i);

    return aux.w;
}

template<typename T>
const uint32_t SmurfHeaderRO<T>::getU32Word(std::size_t offset) const
{
    union
    {
        uint32_t w;
        uint8_t  b[4];
    } aux;

    for (std::size_t i{0}; i < 4; ++i)
        aux.b[i] = *(headerIt+offset+i);

    return aux.w;
}

template<typename T>
const uint64_t SmurfHeaderRO<T>::getU64Word(std::size_t offset) const
{
    union
    {
        uint64_t w;
        uint8_t  b[8];
    } aux;

    for (std::size_t i{0}; i < 8; ++i)
        aux.b[i] = *(headerIt+offset+i);

    return aux.w;
}

template<typename T>
const bool SmurfHeaderRO<T>::getWordBit(std::size_t offset, std::size_t index) const
{
    if (index >= 8)
        throw std::runtime_error("Trying to get a byte's bit with index > 8");

    return ( (*(headerIt+offset) >> index ) & 0x01 );
}


//////////////////////////////////////////
////// - SmurfHeaderRO definitions ///////
//////////////////////////////////////////


////////////////////////////////////////
////// + SmurfHeader definitions ///////
////////////////////////////////////////

template<>
SmurfHeader<ris::FrameIterator>::SmurfHeader(ris::FramePtr frame)
:
    SmurfHeaderRO<ris::FrameIterator>(frame),
    headerIt(frame->beginWrite()),
    tba(TesBiasArray<ris::FrameIterator>::create(headerIt + this->headerTESDACOffset))
{
}

template<>
SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeader(std::vector<uint8_t>& buffer)
:
    SmurfHeaderRO<std::vector<uint8_t>::iterator>(buffer),
    headerIt(buffer.begin()),
    tba(TesBiasArray<std::vector<uint8_t>::iterator>::create(headerIt + this->headerTESDACOffset))
{
}

template<>
SmurfHeaderPtr<ris::FrameIterator> SmurfHeader<ris::FrameIterator>::create(ris::FramePtr frame)
{
    return std::make_shared<SmurfHeader<ris::FrameIterator>>(frame);
}

template<>
SmurfHeaderPtr<std::vector<uint8_t>::iterator> SmurfHeader<std::vector<uint8_t>::iterator>::create(std::vector<uint8_t>& buffer)
{
    return std::make_shared< SmurfHeader<std::vector<uint8_t>::iterator> >(buffer);
}

// Function to get header words
template<typename T>
void SmurfHeader<T>::setVersion(uint8_t value) const
{
    setU8Word(this->headerVersionOffset, value);
}

template<typename T>
void SmurfHeader<T>::setCrateID(uint8_t value) const
{
    setU8Word(this->headerCrateIDOffset, value);
}

template<typename T>
void SmurfHeader<T>::setSlotNumber(uint8_t value) const
{
    setU8Word(this->headerSlotNumberOffset, value);
}

template<typename T>
void SmurfHeader<T>::setTimingConfiguration(uint8_t value) const
{
    setU8Word(this->headerTimingConfigurationOffset, value);
}

template<typename T>
void SmurfHeader<T>::setNumberChannels(uint32_t value) const
{
    setU32Word(this->headerNumberChannelOffset, value);
}

template<typename T>
void SmurfHeader<T>::setTESBias(std::size_t index, int32_t value) const
{
    tba->setWord(index, value);
}

template<typename T>
void SmurfHeader<T>::copyTESBiasArrayFrom(T it) const
{
    std::copy(headerIt + this->headerTESDACOffset,
        headerIt + this->headerTESDACOffset + TesBiasArray<T>::TesBiasBufferSize,
        it);
}

template<typename T>
void SmurfHeader<T>::setUnixTime(uint64_t value) const
{
    setU64Word(this->headerUnixTimeOffset, value);
}

template<typename T>
void SmurfHeader<T>::setFluxRampIncrement(uint32_t value) const
{
    setU32Word(this->headerFluxRampIncrementOffset, value);
}

template<typename T>
void SmurfHeader<T>::setFluxRampOffset(uint32_t value) const
{
    setU32Word(this->headerFluxRampOffsetOffset, value);
}

template<typename T>
void SmurfHeader<T>::setCounter0(uint32_t value) const
{
    setU32Word(this->headerCounter0Offset, value);
}

template<typename T>
void SmurfHeader<T>::setCounter1(uint32_t value) const
{
    setU32Word(this->headerCounter1Offset, value);
}

template<typename T>
void SmurfHeader<T>::setCounter2(uint64_t value) const
{
    setU64Word(this->headerCounter2Offset, value);
}

template<typename T>
void SmurfHeader<T>::setAveragingResetBits(uint32_t value) const
{
    setU32Word(this->headerAveragingResetBitsOffset, value);
}

template<typename T>
void SmurfHeader<T>::setFrameCounter(uint32_t value) const
{
    setU32Word(this->headerFrameCounterOffset, value);
}

template<typename T>
void SmurfHeader<T>::setTESRelaySetting(uint32_t value) const
{
    setU32Word(this->headerTESRelaySettingOffset, value);
}

template<typename T>
void SmurfHeader<T>::setExternalTimeClock(uint64_t value) const
{
    setU64Word(this->headerExternalTimeClockOffset, value);
}

template<typename T>
void SmurfHeader<T>::setControlField(uint8_t value) const
{
    setU8Word(this->headerControlFieldOffset, value);
}

template<typename T>
void SmurfHeader<T>::setClearAverageBit(bool value) const
{
    setWordBit(this->headerControlFieldOffset, value, this->clearAvergaveBitOffset);
}

template<typename T>
void SmurfHeader<T>::setDisableStreamBit(bool value) const
{
    setWordBit(this->headerControlFieldOffset, value, this->disableStreamBitOffset);
}

template<typename T>
void SmurfHeader<T>::setDisableFileWriteBit(bool value) const
{
    setWordBit(this->headerControlFieldOffset, value, this->disableFileWriteBitOffset);
}

template<typename T>
void SmurfHeader<T>::setReadConfigEachCycleBit(bool value) const
{
    setWordBit(this->headerControlFieldOffset, value, this->readConfigEachCycleBitOffset);
}

template<typename T>
void SmurfHeader<T>::setTestMode(uint8_t value) const
{
    uint8_t u8 = this->getControlField();

    u8 &= 0x0f;
    u8 |= ( (value << 4 ) & 0xf0 );

    setU8Word(this->headerControlFieldOffset, u8);
}

template<typename T>
void SmurfHeader<T>::setTestParameters(uint8_t value) const
{
    setU8Word(this->headerTestParametersOffset, value);
}

template<typename T>
void SmurfHeader<T>::setNumberRows(uint16_t value) const
{
    setU16Word(this->headerNumberRowsOffset, value);
}

template<typename T>
void SmurfHeader<T>::setNumberRowsReported(uint16_t value) const
{
    setU16Word(this->headerNumberRowsReportedOffset, value);
}

template<typename T>
void SmurfHeader<T>::setRowLength(uint16_t value) const
{
    setU16Word(this->headerRowLengthOffset, value);
}

template<typename T>
void SmurfHeader<T>::setDataRate(uint16_t value) const
{
    setU16Word(this->headerDataRateOffset, value);
}

// Helper functions
template<typename T>
void SmurfHeader<T>::setU8Word(std::size_t offset, uint8_t value) const
{
    *(headerIt+offset) = value;
}

template<typename T>
void SmurfHeader<T>::setU16Word(std::size_t offset, uint16_t value) const
{
    union
    {
        uint16_t w;
        uint8_t  b[2];
    } aux;

    aux.w  = value;

    for (std::size_t i{0}; i < 2; ++i)
        *(headerIt+offset+i) = aux.b[i];
}

template<typename T>
void SmurfHeader<T>::setU32Word(std::size_t offset, uint32_t value) const
{
    union
    {
        uint32_t w;
        uint8_t  b[4];
    } aux;

    aux.w = value;

    for (std::size_t i{0}; i < 4; ++i)
        *(headerIt+offset+i) = aux.b[i];
}

template<typename T>
void SmurfHeader<T>::setU64Word(std::size_t offset, uint64_t value) const
{
    union
    {
        uint64_t w;
        uint8_t  b[8];
    } aux;

    aux.w = value;

    for (std::size_t i{0}; i < 8; ++i)
        *(headerIt+offset+i) = aux.b[i];
}

template<typename T>
void SmurfHeader<T>::setWordBit(std::size_t offset, std::size_t index, bool value) const
{
    if (index >= 8)
        throw std::runtime_error("Trying to set a byte's bit with index > 8");

    uint8_t aux = *(headerIt+offset);

    if (value)
        aux |= (0x01 << index);
    else
        aux &= ~(0x01 << index);

    *(headerIt+offset) = aux;
}

////////////////////////////////////////
////// - SmurfHeader definitions ///////
////////////////////////////////////////

template class SmurfHeaderRO<ris::FrameIterator>;
template class SmurfHeader<ris::FrameIterator>;
template class SmurfHeaderRO<std::vector<uint8_t>::iterator>;
template class SmurfHeader<std::vector<uint8_t>::iterator>;
