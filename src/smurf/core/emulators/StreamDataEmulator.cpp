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
#include "smurf/core/emulators/StreamDataEmulator.h"
#include "smurf/core/common/SmurfHeader.h"
#include <cmath>
#include <random>

namespace sce = smurf::core::emulators;
namespace ris = rogue::interfaces::stream;

sce::StreamDataEmulator::StreamDataEmulator()
:
    eLog_(rogue::Logging::create("pysmurf.emulator")),
    disable_(true),
    type_(SignalType::Zeros),
    amplitude_(32767),
    offset_(0),
    period_(1),
    periodCounter_(0),
    genSignal_(0)
{
}

sce::StreamDataEmulatorPtr sce::StreamDataEmulator::create()
{
    return std::make_shared<StreamDataEmulator>();
}

// Setup Class in python
void sce::StreamDataEmulator::setup_python()
{
    bp::class_< sce::StreamDataEmulator,
                sce::StreamDataEmulatorPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("StreamDataEmulator",bp::init<>())
        .def("setDisable",        &StreamDataEmulator::setDisable)
        .def("getDisable",        &StreamDataEmulator::getDisable)
        .def("setType",           &StreamDataEmulator::setType)
        .def("getType",           &StreamDataEmulator::getType)
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
    disable_ = d;
}

const bool sce::StreamDataEmulator::getDisable() const
{
    return disable_;
}

void sce::StreamDataEmulator::setType(int value)
{
    // Verify that the type is in range
    if (value < static_cast<int>(SignalType::Size))
        type_ = static_cast<SignalType>(value);
}

const int sce::StreamDataEmulator::getType() const
{
    return static_cast<int>(type_);
}

void sce::StreamDataEmulator::setAmplitude(uint16_t value)
{
    // The amplitude value can not be zero, nor higher that 2^15-1
    if ((value) && (value < 32768 ) )
        amplitude_  = value;
}

const uint16_t sce::StreamDataEmulator::getAmplitude() const
{
    return amplitude_;
}

void sce::StreamDataEmulator::setOffset(int16_t value)
{
    offset_ = value;
}

const int16_t sce::StreamDataEmulator::getOffset() const
{
    return offset_;
}

void sce::StreamDataEmulator::setPeriod(std::size_t value)
{
    // The period value can not be zero
    if (value)
        period_ = value;
}

const std::size_t sce::StreamDataEmulator::getPeriod() const
{
    return period_;
}

void sce::StreamDataEmulator::acceptFrame(ris::FramePtr frame)
{
    {
        rogue::GilRelease noGil;

        // Only process the frame is the block is enable.
        if (!disable_)
        {
            ris::FrameLockPtr fLock = frame->lock();
            std::lock_guard<std::mutex> lock(mtx_);

            // Make sure the frame is a single buffer, copy if necessary
            if ( ! this->ensureSingleBuffer(frame,true) )
            {
                eLog_->error("Failed to copy frame to single buffer. Check downstream slave types, maybe add a FIFO?");
                return;
            }

            // Read the number of channel from the header header
            SmurfHeaderROPtr<ris::FrameIterator> header = SmurfHeaderRO<ris::FrameIterator>::create(frame);
            uint32_t numChannels { header->getNumberChannels() };

            // Check frame integrity.
            if ( header->SmurfHeaderSize + (numChannels * sizeof(fw_t)) != frame->getPayload() )
            {
                eLog_->error("Received frame does not match expected size. Size=%i, header=%i, payload=%i",
                        frame->getPayload(), header->SmurfHeaderSize, numChannels*2);
                return;
            }

            // Get frame iterator
            ris::FrameIterator fPtr = frame->beginRead();

            // Jump over the header
            fPtr += header->SmurfHeaderSize;

            // Create fw_t accessor to the data
            ris::FrameAccessor<fw_t> dPtr(fPtr, numChannels);

            // Generate the type of signal selected
            switch(type_)
            {
                case SignalType::Zeros:
                    genZeroWave(dPtr);
                    break;
                case SignalType::ChannelNumber:
                    genChannelNumberWave(dPtr);
                    break;
                case SignalType::Random:
                    genRandomWave(dPtr);
                    break;
                case SignalType::Square:
                    genSquareWave(dPtr);
                    break;
                case SignalType::Sawtooth:
                    getSawtoothWave(dPtr);
                    break;
                case SignalType::Triangle:
                    genTriangleWave(dPtr);
                    break;
                case SignalType::Sine:
                    genSinWave(dPtr);
                    break;
            }
        }
    }

    // Send frame outside of lock
    this->sendFrame(frame);
}

void sce::StreamDataEmulator::genZeroWave(ris::FrameAccessor<fw_t> &dPtr) const
{
    // Set all channels to zero
    std::fill(dPtr.begin(), dPtr.end(), 0);
}

void sce::StreamDataEmulator::genChannelNumberWave(ris::FrameAccessor<fw_t> &dPtr) const
{
    // Set each channel to its channel number
    for (std::size_t i{0}; i < dPtr.size(); ++i)
        dPtr.at(i) = i;
}

void sce::StreamDataEmulator::genRandomWave(ris::FrameAccessor<fw_t> &dPtr) const
{
    // Generated uniform distributed numbers for each channel
    // applying the selected amplitude and offset.
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());  // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(-amplitude_ + offset_, amplitude_ + offset_);

    for (std::size_t i{0}; i < dPtr.size(); ++i )
        // Use dis to transform the random unsigned int generated by gen into a
        // double in [-amplitude_ + offset, amplitude_ + offset).
        // Each call to dis(gen) generates a new random double.
        dPtr.at(i) = static_cast<fw_t>(dis(gen));
}

void sce::StreamDataEmulator::genSquareWave(ris::FrameAccessor<fw_t> &dPtr)
{
    // Generate a square signal between [-'amplitude_', 'amplitude_'], with an
    // offset of 'offset_' and with period (2 * 'period_').
    // Note: The frame rate is 2*(flux ramp rate). That's why we are generating
    // a signal with a period (2 * 'period_').
    if ( ( periodCounter_++ % ( 2 * period_ ) ) < period_ )
        genSignal_ = -amplitude_ + offset_;
    else
        genSignal_ = amplitude_ + offset_;

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), genSignal_);
}

void sce::StreamDataEmulator::getSawtoothWave(ris::FrameAccessor<fw_t> &dPtr)
{
    // Generate a sawtooth signal between [offset, 'amplitude_'], with a
    // period (2 * 'period_').
    // Note: The frame rate is 2*(flux ramp rate). That's why we are generating
    // a signal with a period (2 * 'period_').
    genSignal_ = offset_ + (periodCounter_++ % (2*period_)) * amplitude_ / ( 2 * period_ - 1);

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), genSignal_);
}

void sce::StreamDataEmulator::genTriangleWave(ris::FrameAccessor<fw_t> &dPtr)
{
    // Generate a triangle signal between [-'amplitude_', 'amplitude_'], with an
    // offset of 'offset_' and with period (2 * 'period_').
    // Note: The frame rate is 2*(flux ramp rate). That's why we are generating
    // a signal with a period (2 * 'period_').
    genSignal_ =
        ( std::abs<fw_t>((periodCounter_++ % (2*period_)) - period_) )
        * 2 * amplitude_ / period_ - amplitude_ + offset_;
}

void sce::StreamDataEmulator::genSinWave(ris::FrameAccessor<fw_t> &dPtr)
{
    // Generate a sine signal between [-'amplitude_', 'amplitude_'], with an
    // offset of 'offset_' and with period (2 * 'period_').
    // Note: The frame rate is 2*(flux ramp rate). That's why we are generating
    // a signal with a period (2 * 'period_').
    genSignal_ = amplitude_ * std::sin( 2 * M_PI * periodCounter_++ / ( 2 * period_ ) ) + offset_;

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), genSignal_);
}

