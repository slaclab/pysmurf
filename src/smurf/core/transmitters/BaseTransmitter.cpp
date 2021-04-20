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

#include <iostream>
#include <boost/python.hpp>
#include "smurf/core/transmitters/BaseTransmitter.h"

namespace bp  = boost::python;
namespace sct = smurf::core::transmitters;

sct::BaseTransmitter::BaseTransmitter()
:
    disable(false),
    dataFrameCnt(0),
    metaFrameCnt(0),
    dataDropFrameCnt(0),
    metaDropFrameCnt(0),
    txDataFunc(std::bind(&BaseTransmitter::dataTransmit, this, std::placeholders::_1)),
    txMetaFunc(std::bind(&BaseTransmitter::metaTransmit, this, std::placeholders::_1))
{
}

sct::BaseTransmitterPtr sct::BaseTransmitter::create()
{
    return std::make_shared<BaseTransmitter>();
}

void sct::BaseTransmitter::setup_python()
{
    bp::class_< sct::BaseTransmitter,
                sct::BaseTransmitterPtr,
                boost::noncopyable >
                ("BaseTransmitter",bp::init<>())
        .def("setDisable",      &BaseTransmitter::setDisable)
        .def("getDisable",      &BaseTransmitter::getDisable)
        .def("clearCnt",        &BaseTransmitter::clearCnt)
        .def("getDataFrameCnt", &BaseTransmitter::getDataFrameCnt)
        .def("getMetaFrameCnt", &BaseTransmitter::getMetaFrameCnt)
        .def("getDataDropCnt",  &BaseTransmitter::getDataDropCnt)
        .def("getMetaDropCnt",  &BaseTransmitter::getMetaDropCnt)
        .def("getDataChannel",  &BaseTransmitter::getDataChannel)
        .def("getMetaChannel",  &BaseTransmitter::getMetaChannel)
    ;
}

// Get data channel
sct::BaseTransmitterChannelPtr sct::BaseTransmitter::getDataChannel()
{
    // Create the dataChanenl object the first time this is called
    if (!dataChannel)
        dataChannel = sct::BaseTransmitterChannel::create(shared_from_this(),0);

    return dataChannel;
}

// Get meta data channel
sct::BaseTransmitterChannelPtr sct::BaseTransmitter::getMetaChannel()
{
    // Create the metaChanenl object the first time this is called
    if (!metaChannel)
        metaChannel = sct::BaseTransmitterChannel::create(shared_from_this(),1);

    return metaChannel;
}

void sct::BaseTransmitter::setDisable(bool d)
{
    disable = d;
}

const bool sct::BaseTransmitter::getDisable() const
{
    return disable;
}

void sct::BaseTransmitter::clearCnt()
{
    dataFrameCnt = 0;
    metaFrameCnt = 0;
    dataDropFrameCnt = 0;
    metaDropFrameCnt = 0;
}

const std::size_t sct::BaseTransmitter::getDataFrameCnt() const
{
    return dataFrameCnt;
}

const std::size_t sct::BaseTransmitter::getMetaFrameCnt() const
{
    return metaFrameCnt;
}

const std::size_t sct::BaseTransmitter::getDataDropCnt() const
{
    return dataDropFrameCnt;
}

const std::size_t sct::BaseTransmitter::getMetaDropCnt() const
{
    return metaDropFrameCnt;
}

void sct::BaseTransmitter::acceptDataFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // If the processing block is disabled, do not process the frame
    if (disable)
        return;

    ris::FrameLockPtr fLock { frame->lock() };

    // Only accept frames which are single buffer
    if ( frame->bufferCount() != 1 )
    {
        ++dataDropFrameCnt;
        return;
    }

    // Reject frames with errors or flags
    if ( frame->getError() || ( frame->getFlags() & 0x100 ) )
    {
        ++dataDropFrameCnt;
        return;
    }

    // Increment the data frame counter
    ++dataFrameCnt;

    // Convert the frame to a SmurfPacketRO object
    SmurfPacketROPtr pkt { SmurfPacketRO::create(frame) };
    fLock->unlock();

    // Call the data TX callback method passing a SmurfPacketRO object
    txDataFunc(pkt);
}

void sct::BaseTransmitter::acceptMetaFrame(ris::FramePtr frame)
{
    rogue::GilRelease noGil;

    // If the processing block is disabled, do not process the frame
    if (disable)
        return;

    ris::FrameLockPtr fLock { frame->lock() };

    // Only accept frames which are single buffer
    if ( frame->bufferCount() != 1 )
    {
        ++metaDropFrameCnt;
        return;
    }

    // Reject frames with errors or flags
    if ( frame->getError() || ( frame->getFlags() & 0x100 ) )
    {
        ++metaDropFrameCnt;
        return;
    }

    // Increment the metadata frame counter
    ++metaFrameCnt;

    // Convert the frame payload to a string
    std::string cfg ( reinterpret_cast<char const*>(frame->beginRead().ptr()), frame->getPayload() );
    fLock->unlock();

    // Call the metadata TX callback function passing the string
    txMetaFunc(cfg);
}
