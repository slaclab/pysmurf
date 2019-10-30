#ifndef _SMURF_CORE_COMMON_SMURFPACKET_H_
#define _SMURF_CORE_COMMON_SMURFPACKET_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Packet
 * ----------------------------------------------------------------------------
 * File          : SmurfPacket.h
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

#include <memory>
#include <stdexcept>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include "smurf/core/common/SmurfHeader.h"

class SmurfPacketRO;

typedef std::shared_ptr<SmurfPacketRO> SmurfPacketROPtr;

typedef int32_t data_t;        // Data type stored in the packet

// SMuRF packet class. This class give a read-only access
class SmurfPacketRO
{
public:
    SmurfPacketRO(ris::FramePtr frame);
    virtual ~SmurfPacketRO() {};

    static SmurfPacketROPtr create(ris::FramePtr frame);

    SmurfHeaderROPtr< std::vector<uint8_t>::iterator > getHeader() const; // Get a pointer to the header object
    const data_t getData(std::size_t index) const;                        // Get a data value

private:
    // Prevent construction using the default or copy constructor.
    // Prevent an SmurfHeaderRO object to be assigned as well.
    SmurfPacketRO();
    SmurfPacketRO(const SmurfPacketRO&);
    SmurfPacketRO& operator=(const SmurfPacketRO&);

    // Data types
    //typedef int32_t data_t;        // Data type stored in the packet

    // Varariables
    std::size_t                                        dataSize;  // Number of data values in the packet
    std::vector<uint8_t>                               header;    // Bufer for the header
    std::vector<data_t>                                data;      // Buffer for the data
    SmurfHeaderROPtr< std::vector<uint8_t>::iterator > headerPtr; // SmurfHeader object (smart pointer)
};

#endif
