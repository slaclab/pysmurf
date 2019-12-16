/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Emulator
 * ----------------------------------------------------------------------------
 * File          : StreamDataEmulator.cpp
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data StreamDataEmulator Class
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

namespace sce = smurf::core::emulators;
namespace ris = rogue::interfaces::stream;

template <typename T>
sce::StreamDataEmulator<T>::StreamDataEmulator()
:
    eLog_(rogue::Logging::create("pysmurf.emulator")),
    disable_(true),
    type_(SignalType::Zeros),
    amplitude_(maxAmplitude),
    offset_(0),
    period_(2),
    halfPeriod_(1),
    periodCounter_(0),
    gen(rd()),
    dis(-amplitude_ + offset_, amplitude_ + offset_),
    dropFrame_(false)
{
}

template <typename T>
sce::StreamDataEmulatorPtr<T> sce::StreamDataEmulator<T>::create()
{
    return std::make_shared< StreamDataEmulator<T> >();
}

// Setup Class in python
template <typename T>
void sce::StreamDataEmulator<T>::setup_python(const std::string& name)
{
    bp::class_< sce::StreamDataEmulator<T>,
                sce::StreamDataEmulatorPtr<T>,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                (name.c_str(), bp::init<>())
        .def("setDisable",        &StreamDataEmulator<T>::setDisable)
        .def("getDisable",        &StreamDataEmulator<T>::getDisable)
        .def("setType",           &StreamDataEmulator<T>::setType)
        .def("getType",           &StreamDataEmulator<T>::getType)
        .def("setAmplitude",      &StreamDataEmulator<T>::setAmplitude)
        .def("getAmplitude",      &StreamDataEmulator<T>::getAmplitude)
        .def("setOffset",         &StreamDataEmulator<T>::setOffset)
        .def("getOffset",         &StreamDataEmulator<T>::getOffset)
        .def("setPeriod",         &StreamDataEmulator<T>::setPeriod)
        .def("getPeriod",         &StreamDataEmulator<T>::getPeriod)

    ;
    bp::implicitly_convertible< sce::StreamDataEmulatorPtr<T>, ris::SlavePtr  >();
    bp::implicitly_convertible< sce::StreamDataEmulatorPtr<T>, ris::MasterPtr >();
}

template <typename T>
void sce::StreamDataEmulator<T>::setDisable(bool d)
{
    disable_ = d;
}

template <typename T>
const bool sce::StreamDataEmulator<T>::getDisable() const
{
    return disable_;
}

template <typename T>
void sce::StreamDataEmulator<T>::setType(int value)
{
    // Verify that the type is in range
    if (value < static_cast<int>(SignalType::Size))
    {
        // Take th mutex before changing the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        type_ = static_cast<SignalType>(value);

        // Rest the frame period counter
        periodCounter_ = 0;
    }
}

template <typename T>
const int sce::StreamDataEmulator<T>::getType() const
{
    return static_cast<int>(type_);
}

template <typename T>
void sce::StreamDataEmulator<T>::setAmplitude(uT_t value)
{
    // The amplitude value can not be zero, nor higher that maxAmplitude
    if ((value) && (value <= maxAmplitude ) )
    {
        // Take th mutex before changing the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        amplitude_  = value;

        // Update the range of the uniform_real_distribution when
        // the amplitude changes.
        dis = std::uniform_real_distribution<double>(-amplitude_ + offset_, amplitude_ + offset_);

        // Rest the frame period counter
        periodCounter_ = 0;
    }
}

template <typename T>
const typename sce::StreamDataEmulator<T>::uT_t sce::StreamDataEmulator<T>::getAmplitude() const
{
    return amplitude_;
}

template <typename T>
void sce::StreamDataEmulator<T>::setOffset(T value)
{
    // Take th mutex before changing the parameters
    std::lock_guard<std::mutex> lock(mtx_);

    offset_ = value;

    // Update the range of the uniform_real_distribution when
    // the offset changes.
    dis = std::uniform_real_distribution<double>(-amplitude_ + offset_, amplitude_ + offset_);

    // Rest the frame period counter
    periodCounter_ = 0;
}

template <typename T>
const T sce::StreamDataEmulator<T>::getOffset() const
{
    return offset_;
}

template <typename T>
void sce::StreamDataEmulator<T>::setPeriod(std::size_t value)
{
    // The period value must at least 2
    if (value >= 2)
    {
        // Take th mutex before changing the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        // Update the period value
        period_ = value;

        // Get the half period value, for convenience
        halfPeriod_ = period_ / 2;

        // Rest the frame period counter
        periodCounter_ = 0;
    }
}

template <typename T>
const std::size_t sce::StreamDataEmulator<T>::getPeriod() const
{
    return period_;
}

template <typename T>
void sce::StreamDataEmulator<T>::acceptFrame(ris::FramePtr frame)
{
    {
        rogue::GilRelease noGil;

        // Only process the frame is the block is enable.
        if (!disable_)
        {
            // Acquire lock on frame
            ris::FrameLockPtr fLock = frame->lock();

            // Make sure the frame is a single buffer, copy if necessary
            if ( ! this->ensureSingleBuffer(frame,true) )
            {
                eLog_->error("Failed to copy frame to single buffer. Check downstream slave types, maybe add a FIFO?");
                return;
            }

            // Get the frame size
            std::size_t frameSize = frame->getPayload();

            // Check for frames with errors, flags, of size less than at least the header size
            if (  frame->getError() ||
                ( frame->getFlags() & 0x100 ) ||
                ( frameSize < SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize ) )
            {
                eLog_->error("Received frame with size lower than the header size. Frame size=%zu, Header size=%zu",
                    frameSize, frameSize < SmurfHeaderRO<ris::FrameIterator>::SmurfHeaderSize);
                return;
            }

            // Read the number of channel from the header header
            SmurfHeaderROPtr<ris::FrameIterator> header = SmurfHeaderRO<ris::FrameIterator>::create(frame);
            uint32_t numChannels { header->getNumberChannels() };

            // Check frame integrity.
            if ( header->SmurfHeaderSize + (numChannels * sizeof(T)) != frameSize )
            {
                eLog_->error("Received frame does not match expected size. Size=%zu, header=%zu, payload=%i",
                        frameSize, header->SmurfHeaderSize, numChannels*2);
                return;
            }

            // Get frame iterator
            ris::FrameIterator fPtr = frame->beginRead();

            // Jump over the header
            fPtr += header->SmurfHeaderSize;

            // Create T accessor to the data
            ris::FrameAccessor<T> dPtr(fPtr, numChannels);

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
                case SignalType::DropFrame:
                    genFrameDrop();
                    break;
            }
        }
    }


    // If the drop frame flag is set, clear it and
    // don't send the frame.
    if (dropFrame_)
    {
        dropFrame_ = false;
        return;
    }

    // Send frame outside of lock
    this->sendFrame(frame);
}

template <typename T>
void sce::StreamDataEmulator<T>::genZeroWave(ris::FrameAccessor<T> &dPtr) const
{
    // Set all channels to zero
    std::fill(dPtr.begin(), dPtr.end(), 0);
}

template <typename T>
void sce::StreamDataEmulator<T>::genChannelNumberWave(ris::FrameAccessor<T> &dPtr) const
{
    // Set each channel to its channel number
    for (std::size_t i{0}; i < dPtr.size(); ++i)
        dPtr.at(i) = i;
}

template <typename T>
void sce::StreamDataEmulator<T>::genRandomWave(ris::FrameAccessor<T> &dPtr)
{
    // Generated uniform distributed numbers for each channel
    // applying the selected amplitude and offset.
    for (std::size_t i{0}; i < dPtr.size(); ++i )
        // Use dis to transform the random unsigned int generated by gen into a
        // double in [-amplitude_ + offset, amplitude_ + offset).
        // Each call to dis(gen) generates a new random double.
        dPtr.at(i) = static_cast<T>(dis(gen));
}

template <typename T>
void sce::StreamDataEmulator<T>::genSquareWave(ris::FrameAccessor<T> &dPtr)
{
    T s;

    {
        // Take th mutex before using the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        // Generate a square signal between [-'amplitude_', 'amplitude_'], with an
        // offset of 'offset_' and with period 'period_'.
        if ( periodCounter_ < halfPeriod_ )
            s = -amplitude_ + offset_;
        else
            s = amplitude_ + offset_;

        // Reset the period counter when it reaches the define period
        if ( ( ++periodCounter_ >= period_ ) )
            periodCounter_ = 0;
    }

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), s);
}

template <typename T>
void sce::StreamDataEmulator<T>::getSawtoothWave(ris::FrameAccessor<T> &dPtr)
{
    T s;

    {
        // Take th mutex before using the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        // Generate a sawtooth signal between [offset, 'amplitude_'], with a
        // period 'period_'.
        s = offset_ + periodCounter_ * amplitude_ / ( period_ - 1);

        // Reset the period counter when it reaches the define period
        if ( ( ++periodCounter_ >= period_ ) )
            periodCounter_ = 0;
    }

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), s);
}

template <typename T>
void sce::StreamDataEmulator<T>::genTriangleWave(ris::FrameAccessor<T> &dPtr)
{
    T s;

    {
        // Take th mutex before using the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        // Generate a triangle signal between [-'amplitude_', 'amplitude_'], with an
        // offset of 'offset_' and with period 'period_'.
        s = ( std::abs<T>( periodCounter_ - halfPeriod_ ) )
            * 2 * amplitude_ / halfPeriod_ - amplitude_ + offset_;

        // Reset the period counter when it reaches the define period
        if ( ( ++periodCounter_ >= period_ ) )
            periodCounter_ = 0;
    }

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), s);
}

template <typename T>
void sce::StreamDataEmulator<T>::genSinWave(ris::FrameAccessor<T> &dPtr)
{
    T s;

    {
        // Take th mutex before using the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        // Generate a sine signal between [-'amplitude_', 'amplitude_'], with an
        // offset of 'offset_' and with period 'period_'.
        s = amplitude_ * std::sin( 2 * M_PI * ++periodCounter_ / period_ ) + offset_;
    }

    // Set all channels to the same signal
    std::fill(dPtr.begin(), dPtr.end(), s);
}

template <typename T>
void sce::StreamDataEmulator<T>::genFrameDrop()
{
    {
        // Take th mutex before using the parameters
        std::lock_guard<std::mutex> lock(mtx_);

        // Set the flag to drop a frame and reset the period counter when
        // it reaches the define period.
        if ( ( ++periodCounter_ >= period_ ) )
        {
            dropFrame_ = true;
            periodCounter_ = 0;
        }
    }
}

template class sce::StreamDataEmulator<int16_t>;
template class sce::StreamDataEmulator<int32_t>;