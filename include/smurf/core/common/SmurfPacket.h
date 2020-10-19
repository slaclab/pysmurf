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

// SMuRF packet class. This class give a read-only access
class SmurfPacketRO
{
public:
    // Data types
    typedef int32_t                                            data_t;    // Data type stored in the packet
    typedef SmurfHeaderROPtr< std::vector<uint8_t>::iterator > HeaderPtr; // SmurfHeader pointer

    // Constructor
    SmurfPacketRO(ris::FramePtr frame);

    // Destructor
    virtual ~SmurfPacketRO() {};

    // Factory method
    static SmurfPacketROPtr create(ris::FramePtr frame);

    // Get a pointer to a header object
    HeaderPtr getHeader() const;

    // Get a data value
    const data_t getData(std::size_t index) const;

    const std::vector<data_t>& getDataVector() const { return data; }

private:
    // Prevent construction using the default or copy constructor.
    // Prevent an SmurfHeaderRO object to be assigned as well.
    SmurfPacketRO();
    SmurfPacketRO(const SmurfPacketRO&);
    SmurfPacketRO& operator=(const SmurfPacketRO&);

    // Variables
    std::size_t          dataSize;  // Number of data values in the packet
    std::vector<uint8_t> header;    // Buffer for the header
    std::vector<data_t>  data;      // Buffer for the data
    HeaderPtr            headerPtr; // SmurfHeader object (smart pointer)
};

#endif
