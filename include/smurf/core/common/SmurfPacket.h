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

// SMuRF packet class. This class give a read-only access
// It is a host class with different creation policies.
template<typename CreationPolicy>
class SmurfPacketManagerRO;

// Convenient typedef to a smart pointer to a SmurfPacketManagerRO.
template<typename CreationPolicy>
using SmurfPacketManagerROPtr = std::shared_ptr< SmurfPacketManagerRO<CreationPolicy> >;

// Policy classes
class CopyCreator;
class ZeroCopyCreator;

// Host class
template<typename CreationPolicy>
class SmurfPacketManagerRO : public CreationPolicy
{
public:
    // Constructor
    SmurfPacketManagerRO(ris::FramePtr frame);

    // Destructor
    virtual ~SmurfPacketManagerRO() {};

    // Factory method
    static SmurfPacketManagerROPtr<CreationPolicy> create(ris::FramePtr frame);

private:
    // Prevent construction using the default or copy constructor.
    // Prevent an SmurfHeaderRO object to be assigned as well.
    SmurfPacketManagerRO();
    SmurfPacketManagerRO(const SmurfPacketManagerRO&);
    SmurfPacketManagerRO& operator=(const SmurfPacketManagerRO&);
};

// Policy classes to define the type of object creation.

// Copy creator policy class: this class makes a full copy of the frame information into
// local vectors.
class CopyCreator
{
public:
    // Data types
    typedef int32_t                                            data_t;    // Data type stored in the packet
    typedef SmurfHeaderROPtr< std::vector<uint8_t>::iterator > HeaderPtr; // SmurfHeader pointer

    // Constructor
    CopyCreator(ris::FramePtr frame);

    // Destructor
    virtual ~CopyCreator() {};

    // Get a pointer to a header object
    HeaderPtr getHeader() const;

    // Get a data value
    const data_t getData(std::size_t index) const;

private:
    // Variables
    std::vector<uint8_t> header;    // Buffer for the header
    std::vector<data_t>  data;      // Buffer for the data
    HeaderPtr            headerPtr; // SmurfHeader object (smart pointer)
};

// ZeroCopy creator policy: this class provides direct access to the underlaying frame,
// without make any copy.
// As the underlaying frame is accessed directly, you must create and hold a lock on the
// frame for the while lifetime of the object.
class ZeroCopyCreator
{
public:
    // Data types
    typedef int32_t                                data_t;    // Data type stored in the packet
    typedef SmurfHeaderROPtr< ris::FrameIterator > HeaderPtr; // SmurfHeader pointer

    // Constructor
    ZeroCopyCreator(ris::FramePtr frame);

    // Destructor
    virtual ~ZeroCopyCreator() {};


    // Get a pointer to a header object
    HeaderPtr getHeader() const;

    // Get a data value
    const data_t getData(std::size_t index) const;

private:
    // Variables
    ris::FramePtr framePtr;  // Frame pointer
    HeaderPtr     headerPtr; // SmurfHeader object (smart pointer)
    std::size_t   dataSize;  // Number of data values in the packet
    data_t*       data;      // Pointer to the packet data area
};

#endif
