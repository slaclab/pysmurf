#ifndef _SMURF_CORE_COMMON_SMURFHEADER_H_
#define _SMURF_CORE_COMMON_SMURFHEADER_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Header (on a frame)
 * ----------------------------------------------------------------------------
 * File          : SmurfHeader.h
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

#include <memory>
#include <stdexcept>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include "smurf/core/common/TesBiasArray.h"

namespace ris = rogue::interfaces::stream;

template<typename T>
class SmurfHeaderRO;
template<typename T>
class SmurfHeader;

template<typename T>
using  SmurfHeaderROPtr = std::shared_ptr<SmurfHeaderRO<T>>;

template<typename T>
using SmurfHeaderPtr = std::shared_ptr<SmurfHeader<T>>;

// SMuRF header class. This class give a read-only access
template<typename T>
class SmurfHeaderRO
{
public:
    SmurfHeaderRO(ris::FramePtr frame);
    SmurfHeaderRO(std::vector<uint8_t>& buffer);
    virtual ~SmurfHeaderRO() {};

    static SmurfHeaderROPtr<T> create(ris::FramePtr frame);
    static SmurfHeaderROPtr<T> create(std::vector<uint8_t>& buffer);

    const uint8_t  getVersion()                   const;  // Get protocol version
    const uint8_t  getCrateID()                   const;  // Get ATCA crate ID
    const uint8_t  getSlotNumber()                const;  // Get ATCA slot number
    const uint8_t  getTimingConfiguration()       const;  // Get timing configuration
    const uint32_t getNumberChannels()            const;  // Get number of channel in this packet
    const int32_t  getTESBias(std::size_t index)  const;  // Get TES DAC values 16X 20 bit
    void           copyTESBiasArrayTo(T it) const;   // Copy the TES DAC full array to a destination iterator
    const uint64_t getUnixTime()                  const;  // Get 64 bit unix time nanoseconds
    const uint32_t getFluxRampIncrement()         const;  // Get signed 32 bit integer for increment
    const uint32_t getFluxRampOffset()            const;  // Get signed 32 it integer for offset
    const uint32_t getCounter0()                  const;  // Get 32 bit counter since last 1Hz marker
    const uint32_t getCounter1()                  const;  // Get 32 bit counter since last external input
    const uint64_t getCounter2()                  const;  // Get 64 bit timestamp
    const uint32_t getAveragingResetBits()        const;  // Get up to 32 bits of average reset from timing system
    const uint32_t getFrameCounter()              const;  // Get locally genreate frame counter 32 bit
    const uint32_t getTESRelaySetting()           const;  // Get TES and flux ramp relays, 17bits in use now
    const uint64_t getExternalTimeClock()         const;  // Get Syncword from mce for mce based systems (40 bit including header)
    const uint8_t  getControlField()              const;  // Get control field word
    const bool     getClearAverageBit()           const;  // Get control field's clear average and unwrap bit (bit 0)
    const bool     getDisableStreamBit()          const;  // Get control field's disable stream to MCE bit (bit 1)
    const bool     getDisableFileWriteBit()       const;  // Get control field's disable file write (bit 2)
    const bool     getReadConfigEachCycleBit()    const;  // Get control field's set to read configuration file each cycle bit (bit 3)
    const uint8_t  getTestMode()                  const;  // Get control field's test mode (bits 4-7)
    const uint8_t  getTestParameters()            const;  // Get test parameters
    const uint16_t getNumberRows()                const;  // Get MCE header value (max 255) (defaluts to 33 if 0)
    const uint16_t getNumberRowsReported()        const;  // Get MCE header value (defaults to numb rows if 0)
    const uint16_t getRowLength()                 const;  // Get MCE header value
    const uint16_t getDataRate()                  const;  // Get MCE header value

    // The size of the smurf header (in bytes)
    // It has public access.
    static const std::size_t SmurfHeaderSize                 = 128;

protected:
    // Header word offsets (in bytes)
    static const std::size_t headerVersionOffset              = 0;
    static const std::size_t headerCrateIDOffset              = 1;
    static const std::size_t headerSlotNumberOffset           = 2;
    static const std::size_t headerTimingConfigurationOffset  = 3;
    static const std::size_t headerNumberChannelOffset        = 4;
    static const std::size_t headerTESDACOffset               = 8;
    static const std::size_t headerUnixTimeOffset             = 48;
    static const std::size_t headerFluxRampIncrementOffset    = 56;
    static const std::size_t headerFluxRampOffsetOffset       = 60;
    static const std::size_t headerCounter0Offset             = 64;
    static const std::size_t headerCounter1Offset             = 68;
    static const std::size_t headerCounter2Offset             = 72;
    static const std::size_t headerAveragingResetBitsOffset   = 80;
    static const std::size_t headerFrameCounterOffset         = 84;
    static const std::size_t headerTESRelaySettingOffset      = 88;
    static const std::size_t headerExternalTimeClockOffset    = 96;
    static const std::size_t headerControlFieldOffset         = 104;
    static const std::size_t headerTestParametersOffset       = 105;
    static const std::size_t headerNumberRowsOffset           = 112;
    static const std::size_t headerNumberRowsReportedOffset   = 114;
    static const std::size_t headerRowLengthOffset            = 120;
    static const std::size_t headerDataRateOffset             = 122;

    // Header's control field bit offset
    static const std::size_t clearAvergaveBitOffset           = 0;
    static const std::size_t disableStreamBitOffset           = 1;
    static const std::size_t disableFileWriteBitOffset        = 2;
    static const std::size_t readConfigEachCycleBitOffset     = 3;

private:
    // Prevent construction using the default or copy constructor.
    // Prevent an SmurfHeaderRO object to be assigned as well.
    SmurfHeaderRO();
    SmurfHeaderRO(const SmurfHeaderRO&);
    SmurfHeaderRO& operator=(const SmurfHeaderRO&);

    // helper functions
    const uint8_t  getU8Word(  std::size_t offset                    ) const; // Returns uint8_t word from the header, at offset 'offset'
    const uint16_t getU16Word( std::size_t offset                    ) const; // Returns uint16_t word from the header, at offset 'offset'
    const uint32_t getU32Word( std::size_t offset                    ) const; // Returns uin3t2_t word from the header, at offset 'offset'
    const uint64_t getU64Word( std::size_t offset                    ) const; // Returns uint64_t word from the header, at offset 'offset'
    const bool     getWordBit( std::size_t offset, std::size_t index ) const; // Returns bit 'index' from a header byte at offset 'offset'

    // Private variables
    T headerIt;  // Iterator to the start of the header in a Frame

    // TES Bias array object
    TesBiasArrayPtr<T> tba;
};

// SMuRF header class. This class give a read-write access
template<typename T>
class SmurfHeader : public SmurfHeaderRO<T>
{
public:
    SmurfHeader(ris::FramePtr frame);
    SmurfHeader(std::vector<uint8_t>& buffer);
    ~SmurfHeader() {};

    static SmurfHeaderPtr<T> create(ris::FramePtr frame);
    static SmurfHeaderPtr<T> create(std::vector<uint8_t>& buffer);

    void setVersion(uint8_t value) const;                     // Set protocol version
    void setCrateID(uint8_t value) const;                     // Set ATCA crate ID
    void setSlotNumber(uint8_t value) const;                  // Set ATCA slot number
    void setTimingConfiguration(uint8_t value) const;         // Set timing configuration
    void setNumberChannels(uint32_t value) const;             // Set number of channel in this packet
    void setTESBias(std::size_t index, int32_t value) const;  // Set TES DAC values 16X 20 bit
    void copyTESBiasArrayFrom(T it) const;   // Copy the TES DAC full array from a source iterator
    void setUnixTime(uint64_t value) const;                   // Set 64 bit unix time nanoseconds
    void setFluxRampIncrement(uint32_t value) const;          // Set signed 32 bit integer for increment
    void setFluxRampOffset(uint32_t value) const;             // Set signed 32 it integer for offset
    void setCounter0(uint32_t value) const;                   // Set 32 bit counter since last 1Hz marker
    void setCounter1(uint32_t value) const;                   // Set 32 bit counter since last external input
    void setCounter2(uint64_t value) const;                   // Set 64 bit timestamp
    void setAveragingResetBits(uint32_t value) const;         // Set up to 32 bits of average reset from timing system
    void setFrameCounter(uint32_t value) const;               // Set locally genreate frame counter 32 bit
    void setTESRelaySetting(uint32_t value) const;            // Set TES and flux ramp relays, 17bits in use now
    void setExternalTimeClock(uint64_t value) const;          // Set Syncword from mce for mce based systems (40 bit including header)
    void setControlField(uint8_t value) const;                // Set control field word
    void setClearAverageBit(bool value) const;                // Set control field's clear average and unwrap bit (bit 0)
    void setDisableStreamBit(bool value) const;               // Set control field's disable stream to MCE bit (bit 1)
    void setDisableFileWriteBit(bool value) const;            // Set control field's disable file write (bit 2)
    void setReadConfigEachCycleBit(bool value) const;         // Set control field's set to read configuration file each cycle bit (bit 3)
    void setTestMode(uint8_t value) const;                    // Set control field's test mode (bits 4-7)
    void setTestParameters(uint8_t value) const;              // Set test parameters
    void setNumberRows(uint16_t value) const;                 // Set MCE header value (max 255) (defaluts to 33 if 0)
    void setNumberRowsReported(uint16_t value) const;         // Set MCE header value (defaults to numb rows if 0)
    void setRowLength(uint16_t value) const;                  // Set MCE header value
    void setDataRate(uint16_t value) const;                   // Set MCE header value

private:
    // Prevent construction using the default or copy constructor.
    // Prevent an SmurfHeaderRO object to be assigned as well.
    SmurfHeader();
    SmurfHeader(const SmurfHeader&);
    SmurfHeader& operator=(const SmurfHeader&);

    // helper functions
    void setU8Word(  std::size_t offset, uint8_t value                 ) const; // Write a uint8_t word into the header, at offset 'offset'
    void setU16Word( std::size_t offset, uint16_t value                ) const; // Write a uint16_t word into the header, at offset 'offset'
    void setU32Word( std::size_t offset, uint32_t value                ) const; // Write a uin3t2_t word into the header, at offset 'offset'
    void setU64Word( std::size_t offset, uint64_t value                ) const; // Write a uint64_t word into the header, at offset 'offset'
    void setWordBit( std::size_t offset, std::size_t index, bool value ) const; // write a bit at 'index' position into the header byte at offset 'offset'

    // Private variables
    T headerIt;  // Iterator to the start of the header in a Frame

    // TES Bias array object
    TesBiasArrayPtr<T> tba;
};

#endif
