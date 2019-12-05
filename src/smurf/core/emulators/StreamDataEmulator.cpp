/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Emulator
 * ----------------------------------------------------------------------------
 * File          : SmurfProcessor.cpp
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data Processor Class.
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

#include <boost/python.hpp>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/FrameAccessor.h>
#include "smurf/core/emulators/StreamDataEmulator.h"
#include "smurf/core/common/SmurfHeader.h"
#include <cmath>

namespace sce = smurf::core::emulators;
namespace ris = rogue::interfaces::stream;

sce::StreamDataEmulator::StreamDataEmulator()
:
    eLog_(rogue::Logging::create("pysmurf.emulator")),
    sinAmplitude_(0),
    sinBaseline_(0),
    sinPeriod_(0),
    sinChannel_(0),
    sinEnable_(false),
    sinCount_(0),
    disabled_(true),
    type_(0),
    amplitude_(65535),
    offset_(0),
    period_(1),
{
}

sce::StreamDataEmulator::~StreamDataEmulator() { }

sce::StreamDataEmulatorPtr sce::StreamDataEmulator::create()
{
    return std::make_shared<StreamDataEmulator>();
}

void sce::StreamDataEmulator::setup_python()
{
    bp::class_< sce::StreamDataEmulator,
                sce::StreamDataEmulatorPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("StreamDataEmulator",bp::init<>())

        // Sin Generate Parameters
        .def("setSinAmplitude",   &StreamDataEmulator::setSinAmplitude)
        .def("getSinAmplitude",   &StreamDataEmulator::getSinAmplitude)
        .def("setSinBaseline",    &StreamDataEmulator::setSinBaseline)
        .def("getSinBaseline",    &StreamDataEmulator::getSinBaseline)
        .def("setSinPeriod",      &StreamDataEmulator::setSinPeriod)
        .def("getSinPeriod",      &StreamDataEmulator::getSinPeriod)
        .def("setSinChannel",     &StreamDataEmulator::setSinChannel)
        .def("getSinChannel",     &StreamDataEmulator::getSinChannel)
        .def("setSinEnable",      &StreamDataEmulator::setSinEnable)
        .def("getSinEnable",      &StreamDataEmulator::getSinEnable)

        .def("setDisable",        &StreamDataEmulator::setDisable)
        .def("getDisable",        &StreamDataEmulator::getDisable)
        .def("setTpe",            &StreamDataEmulator::setTpe)
        .def("getTpe",            &StreamDataEmulator::getTpe)
        .def("setAmplitude",      &StreamDataEmulator::setAmplitude)
        .def("getAmplitude",      &StreamDataEmulator::getAmplitude)
        .def("setOffset",         &StreamDataEmulator::setOffset)
        .def("getOffset",         &StreamDataEmulator::getOffset)
        .def("setPeriod",         &StreamDataEmulator::setPeriod)
        .def("getPeriod",         &StreamDataEmulator::getPeriod)

    ;
    bp::implicitly_convertible< sce::StreamDataEmulatorPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< sce::StreamDataEmulatorPtr, ris::MasterPtr >();
}

void sce::StreamDataEmulator::setDisable(bool d)
{
    disabled_ = d;
}

const bool sce::StreamDataEmulator::getDisable() const
{
    return disabled_;
}

void sce::StreamDataEmulator::setType(std::size_t value)
{
    type_ = value;
}

const std::size_t sce::StreamDataEmulator::getType() const
{
    return type_;
}

void sce::StreamDataEmulator::setAmplitude(uint16_t value)
{
    // The amplitude value can not be zero
    if (value)
        amplitude_  = value;
}

const uint16_t sce::StreamDataEmulator::getAmplitude() const
{
    return amplitude_;
}

void sce::StreamDataEmulator::setOffset(uint16_t value)
{
    offset_ = value;
}

const uint16_t sce::StreamDataEmulator::getOffset() const
{
    return offset_;
}

void sce::StreamDataEmulator::setPeriod(uint32_t value)
{
    // The period value can not be zero
    if (value)
        period_ = value;
}

const uint32_t sce::StreamDataEmulator::getPeriod() const
{
    return period_;
}

// Sin parameters
void sce::StreamDataEmulator::setSinAmplitude(uint16_t value) {
    std::lock_guard<std::mutex> lock(mtx_);
    sinAmplitude_ = value;
}

uint16_t sce::StreamDataEmulator::getSinAmplitude() {
    return sinAmplitude_;
}

void sce::StreamDataEmulator::setSinBaseline(uint16_t value) {
    std::lock_guard<std::mutex> lock(mtx_);
    sinBaseline_ = value;
}

uint16_t sce::StreamDataEmulator::getSinBaseline() {
    return sinBaseline_;
}

void sce::StreamDataEmulator::setSinPeriod(uint16_t value) {
    std::lock_guard<std::mutex> lock(mtx_);
    sinPeriod_ = value;
}

uint16_t sce::StreamDataEmulator::getSinPeriod() {
    return sinPeriod_;
}

void sce::StreamDataEmulator::setSinChannel(uint16_t value) {
    std::lock_guard<std::mutex> lock(mtx_);
    sinChannel_ = value;
}

uint16_t sce::StreamDataEmulator::getSinChannel() {
    return sinChannel_;
}

void sce::StreamDataEmulator::setSinEnable(bool value) {
    std::lock_guard<std::mutex> lock(mtx_);
    sinEnable_ = value;
}

bool sce::StreamDataEmulator::getSinEnable() {
    return sinEnable_;
}

void sce::StreamDataEmulator::acceptFrame(ris::FramePtr frame) {

    {
        rogue::GilRelease noGil;
        ris::FrameLockPtr fLock = frame->lock();
        std::lock_guard<std::mutex> lock(mtx_);


        // Make sure the frame is a single buffer, copy if neccessary
        if ( ! this->ensureSingleBuffer(frame,true) )
        {
            eLog_->error("Failed to copy frame to single buffer. Check downstream slave types, maybe add a FIFO?");
            return;
        }

        // Sine wave enabled
        if ( sinEnable_ ) genSinWave(frame);
    }

    // Send frame outside of lock
    this->sendFrame(frame);
}

// Generic sine wave generator
void sce::StreamDataEmulator::genSinWave(ris::FramePtr &frame) {

    SmurfHeaderROPtr<ris::FrameIterator> header = SmurfHeaderRO<ris::FrameIterator>::create(frame);

    uint32_t numChannels { header->getNumberChannels() };

    if ( sinChannel_ >= numChannels )
    {
        eLog_->error("Configured sinChannel exceeds number of rows defined in the header.");
        return;
    }

    if ( header->SmurfHeaderSize + (numChannels * 2) != frame->getPayload() )
    {
        eLog_->error("Received frame does not match expected size. Size=%i, header=%i, payload=%i",
                frame->getPayload(), header->SmurfHeaderSize, numChannels*2);
        return;
    }

    // Get frame iterator
    ris::FrameIterator fPtr = frame->beginRead();

    // Jump over the header
    fPtr += header->SmurfHeaderSize;

    // Create uint16 accessor to the data
    ris::FrameAccessor<uint16_t> dPtr(fPtr, numChannels);

    dPtr[sinChannel_] = int((float)sinBaseline_ +
                       (float)sinAmplitude_ * sin((float)sinCount_/(float)sinPeriod_));

    if ( ++sinCount_ == sinPeriod_ )
        sinCount_ = 0;
}

