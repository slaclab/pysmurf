/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Processor
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
#include "smurf/core/processors/SmurfProcessor.h"
#include <iostream>
#include <string>

namespace scp = smurf::core::processors;

// This is the init function for the SmurfProcessor. These functions
// are exposed by setup_python below, and the whole class is
// instantiated by the SmurfProcessor python class which inherits the
// pyrogue Device type. Typically either CmbEth.py or CmbPcie.py will
// intantiate the SmurfProcessor object. To be clear, it's Rogue that
// starts the SMuRF C++ code.

// SmurfProcessor: https://github.com/slaclab/pysmurf/blob/c4b191d1bf49f49d137f8771dbfe70d0a50a11c0/python/pysmurf/core/devices/_SmurfProcessor.py
// Examples: https://slaclab.github.io/rogue/custom_module/wrapper.html
// CmbEth: https://github.com/slaclab/pysmurf/blob/c4b191d1bf49f49d137f8771dbfe70d0a50a11c0/python/pysmurf/core/roots/CmbEth.py
scp::SmurfProcessor::SmurfProcessor()
:
    ris::Slave(),
    ris::Master(),
    numCh(maxNumCh),
    payloadSize(0),
    mask(numCh,0),
    disableUnwrapper(false),
    currentData(numCh, 0),
    previousData(numCh, 0),
    wrapCounter(numCh, 0),
    inputData(numCh, 0),
    disableFilter(false),
    order(4),
    gain(1),
    a( order + 1 ,1 ),
    b( order + 1, 1 ),
    currentIndex(0),
    x( ( order + 1 ) * numCh ),
    y( ( order + 1 ) * numCh ),
    outData(numCh,0),
    disableDownsampler(false),
    downsamplerMode(0),
    factor(20),
    downsamplerInternalCount(0),
    externalBitmask(0),
    downsamplerOutgoingCount(0),
    headerCopy(SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize, 0),
    runTxThread(true),
    txDataReady(false),
    pktTransmitterThread(std::thread( &SmurfProcessor::pktTansmitter, this )),
    eLog_(rogue::Logging::create("pysmurf.SmurfProcessor"))
{
    if( pthread_setname_np( pktTransmitterThread.native_handle(), "pktTransmitter" ) )
        perror( "pthread_setname_np failed for pktTransmitterThread thread" );
}

scp::SmurfProcessorPtr scp::SmurfProcessor::create()
{
    return std::make_shared<SmurfProcessor>();
}

// This defines the functions exposed to SmurfProcessor.py.
// See the Rogue documentation for details.
void scp::SmurfProcessor::setup_python()
{
    bp::class_< scp::SmurfProcessor,
                scp::SmurfProcessorPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("SmurfProcessor",bp::init<>())
        // Channel mapping variables
        .def("getNumCh",                &SmurfProcessor::getNumCh)
        .def("setPayloadSize",          &SmurfProcessor::setPayloadSize)
        .def("getPayloadSize",          &SmurfProcessor::getPayloadSize)
        .def("setMask",                 &SmurfProcessor::setMask)
        .def("getMask",                 &SmurfProcessor::getMask)
        // Unwrapper variables
        .def("setUnwrapperDisable",     &SmurfProcessor::setUnwrapperDisable)
        .def("getUnwrapperDisable",     &SmurfProcessor::getUnwrapperDisable)
        .def("resetUnwrapper",          &SmurfProcessor::resetUnwrapper)
        // Filter variables
        .def("setFilterDisable",        &SmurfProcessor::setFilterDisable)
        .def("getFilterDisable",        &SmurfProcessor::getFilterDisable)
        .def("setOrder",                &SmurfProcessor::setOrder)
        .def("getOrder",                &SmurfProcessor::getOrder)
        .def("setA",                    &SmurfProcessor::setA)
        .def("getA",                    &SmurfProcessor::getA)
        .def("setB",                    &SmurfProcessor::setB)
        .def("getB",                    &SmurfProcessor::getB)
        .def("setGain",                 &SmurfProcessor::setGain)
        .def("getGain",                 &SmurfProcessor::getGain)
        .def("resetFilter",             &SmurfProcessor::resetFilterWithMutex)
        // Downsampler variables
        .def("setDownsamplerDisable",   &SmurfProcessor::setDownsamplerDisable)
        .def("getDownsamplerDisable",   &SmurfProcessor::getDownsamplerDisable)
        .def("setDownsamplerMode",      &SmurfProcessor::setDownsamplerMode)
        .def("getDownsamplerMode",      &SmurfProcessor::getDownsamplerMode)
        .def("setDownsamplerInternalFactor",    &SmurfProcessor::setDownsamplerInternalFactor)
        .def("getDownsamplerInternalFactor",    &SmurfProcessor::getDownsamplerInternalFactor)
        .def("getDownsamplerExternalBitmask", &SmurfProcessor::getDownsamplerExternalBitmask)
        .def("setDownsamplerExternalBitmask", &SmurfProcessor::setDownsamplerExternalBitmask)
        .def("getDownsamplerOutgoingCount", &SmurfProcessor::getDownsamplerOutgoingCount)
    ;
    bp::implicitly_convertible< scp::SmurfProcessorPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< scp::SmurfProcessorPtr, ris::MasterPtr >();
}

const std::size_t scp::SmurfProcessor::getNumCh() const
{
    return numCh;
}

const std::size_t scp::SmurfProcessor::getPayloadSize() const
{
    return payloadSize;
}

void scp::SmurfProcessor::setPayloadSize(std::size_t s)
{
    payloadSize = s;
}

void scp::SmurfProcessor::setMask(bp::list m)
{
    std::size_t listSize = len(m);

    // Check if the size of the list, is not greater than
    // the number of channels we can have in the output packet.
    if ( listSize > maxNumCh )
    {
        // This should go to a logger instead
        std::cerr << "ERROR: Trying to set a mask list of length = " << listSize \
                  << ", which is larger that the number of channel in a SMuRF packet = " \
                  <<  maxNumCh << std::endl;

        // Do not update the mask vector.
        return;
    }

    // We will use a temporal vector to hold the new data.
    // New data will be check as it is pushed to this vector. If there
    // are not error, this vector will be swap with 'mask'.
    std::vector<std::size_t> temp;

    for (std::size_t i{0}; i < listSize; ++i)
    {
        std::size_t val = bp::extract<std::size_t>(m[i]);

        // Check if the mask value is not greater than
        // the number of channel we received in the incoming frame
        if (val > maxNumCh)
        {
            // This should go to a logger instead
            std::cerr << "ERROR: mask value at index " << i << " is " << val \
                      << ", which is greater the maximum number of channel we expect from an input frame = " \
                      << maxNumCh << std::endl;

            // Do not update the mask vector.
            return;
        }

        // A valid number was found. Add it to the temporal vector
        temp.push_back(val);
    }

    // Take the mutex before changing the mask vector
    std::lock_guard<std::mutex> lockMap(mutChMapper);

    // At this point, all element in the mask list are valid.
    // Update the mask vector
    mask.swap(temp);

    // Update the number of mapped channels
    updateNumCh();
}

const bp::list scp::SmurfProcessor::getMask() const
{
    bp::list temp;

    for (auto const &v : mask)
        temp.append(v);

    return temp;
}

void scp::SmurfProcessor::updateNumCh()
{
    // Start with the size of the mask vector as the new size
    std::size_t newNumCh = mask.size();

    // If the new size if different from the current size, update it
    if (numCh != newNumCh)
    {
        numCh = newNumCh;

        // Reset the Unwrapper
        resetUnwrapper();

        // Reset the filter
        // Take the mutex before changing the filter parameters
        // This make sure that the new order value is not used before
        // the a and b array are resized.
        std::lock_guard<std::mutex> lockFilter(mutFilter);
        resetFilter();
    }
}

void scp::SmurfProcessor::setUnwrapperDisable(bool d)
{
    disableUnwrapper = d;

    // Reset the unwrapper when it is re-enable.
    if (!disableUnwrapper)
        resetUnwrapper();
}

const bool scp::SmurfProcessor::getUnwrapperDisable() const
{
    return disableUnwrapper;
}

void scp::SmurfProcessor::resetUnwrapper()
{
    // Take the mutex before reseting the unwrapper as the
    // This makes sure that the new vectors are not used while
    // they are potentially being resized.
    std::lock_guard<std::mutex> lockUnwrapper(mutUnwrapper);

    std::vector<fw_t>(numCh).swap(currentData);
    std::vector<fw_t>(numCh).swap(previousData);
    std::vector<unwrap_t>(numCh).swap(wrapCounter);
    std::vector<unwrap_t>(numCh).swap(inputData);
}

void scp::SmurfProcessor::setFilterDisable(bool d)
{
    // Take the mutex before changing the filter parameters
    // This make sure that the new order value is not used before
    // the a and b array are resized.
    std::lock_guard<std::mutex> lock(mutFilter);

    disableFilter = d;

    // Reset the filter when is enable flag changes.
    resetFilter();
}

const bool scp::SmurfProcessor::getFilterDisable() const
{
    return disableFilter;
}

void scp::SmurfProcessor::setOrder(std::size_t o)
{
    // Check if the new order is different from the current one
    if ( o != order )
    {
        // Take the mutex before changing the filter parameters
        // This make sure that the new order value is not used before
        // the a and b array are resized.
        std::lock_guard<std::mutex> lock(mutFilter);

        order = o;

        // When the order change, reset the filter
        resetFilter();
    }
}

const std::size_t scp::SmurfProcessor::getOrder() const
{
    return order;
}

void scp::SmurfProcessor::setA(bp::list l)
{
    std::vector<double> temp;

    // Take the mutex before changing the filter parameters
    // This make sure that the 'a' array is not used before it has
    // beem resized, if necessary.
    std::lock_guard<std::mutex> lock(mutFilter);

    std::size_t listSize = len(l);

    if (listSize == 0)
    {
        // Verify that the input list is not empty.
        // If empty, set the coefficients vector to a = [1.0].
        eLog_->error("Trying to set an empty set of a coefficients. Defaulting to 'a = [1.0]'");
        temp.push_back(1.0);
    }
    else if (l[0] == 0)
    {
        // Verify that the first coefficient is not zero.
        // if it is, set the coefficients vector to a = [1.0].
        eLog_->error("The first a coefficient can not be zero. Defaulting to 'a = [1.0]'");
        temp.push_back(1.0);
    }
    else
    {
        // Extract the coefficients coming from python into a temporal vector
        for (std::size_t i{0}; i < listSize; ++i)
            temp.push_back(bp::extract<double>(l[i]));
    }

    // Update the a vector with the new values
    a.swap(temp);

    // When the coefficients change, reset the filter
    resetFilter();
}

const bp::list scp::SmurfProcessor::getA() const
{
    bp::list temp;

    for (auto const &v : a)
        temp.append(v);

    return temp;
}

void scp::SmurfProcessor::setB(bp::list l)
{
    std::vector<double> temp;

    // Take the mutex before changing the filter parameters
    // This make sure that the 'b' array is not used before it has
    // beem resized, if necessary.
    std::lock_guard<std::mutex> lock(mutFilter);

    std::size_t listSize = len(l);

    if (listSize == 0)
    {
        // Verify that the input list is not empty.
        // If empty, set the coefficients vector to a = [0.0].
        eLog_->error("ERROR: Trying to set an empty set of a coefficients. Defaulting to 'b = [0.0]'");
        temp.push_back(0.0);
    }
    else
    {
        // Extract the coefficients coming from python into a temporal vector
        for (std::size_t i{0}; i < len(l); ++i)
            temp.push_back(bp::extract<double>(l[i]));
    }

    // Update the a vector with the new values
    b.swap(temp);

    // When the coefficients change, reset the filter
    resetFilter();
}

const bp::list scp::SmurfProcessor::getB() const
{
    bp::list temp;

    for (auto const &v : b)
        temp.append(v);

    return temp;
}

void scp::SmurfProcessor::setGain(double g)
{
    gain = g;
}

const double scp::SmurfProcessor::getGain() const
{
    return gain;
}

// Reset the filter. Resize and Zero-initialize the data buffer, and
// check if the coefficient vectors have the correct size, and expand
// if necessary, padding with zeros.
void scp::SmurfProcessor::resetFilter()
{
    // Resize and re-initialize the data buffer
    std::vector<double>( ( order + 1 ) * numCh ).swap(x);
    std::vector<double>( ( order + 1 ) * numCh ).swap(y);

    // Use the mutex to update the outData buffer
    {
        std::lock_guard<std::mutex> lock(outDataMutex);
        std::vector<filter_t>( numCh ).swap(outData);
    }

    // Check that a coefficient vector size is at least 'order + 1'.
    // If not, add expand it with zeros.
    if ( a.size() < (order + 1) )
        a.resize(order +  1, 0);

    // Check that b coefficient vector size is at least 'order + 1'.
    // If not, add expand it with zeros.
    if ( b.size() < (order + 1) )
        b.resize(order +  1, 0);

    // Reset the index of the older point in the buffer
    currentIndex = 0;
}

// Reset the filter but holding the mutex. This is need when reseting the
// filter from python
void scp::SmurfProcessor::resetFilterWithMutex()
{
    // Take the mutex before changing the filter parameters
    // This make sure that the new order value is not used before
    // the a and b array are resized.
    std::lock_guard<std::mutex> lockFilter(mutFilter);
    resetFilter();
}

void scp::SmurfProcessor::setDownsamplerDisable(bool d)
{
    disableDownsampler = d;
}

const bool scp::SmurfProcessor::getDownsamplerDisable() const
{
    return disableDownsampler;
}

void scp::SmurfProcessor::setDownsamplerMode(std::size_t mode) {
  downsamplerMode = mode;

  if (mode == downsamplerModeInternal) {
    setDownsamplerInternalCount(0);
  }
}

const std::size_t scp::SmurfProcessor::getDownsamplerMode() const
{
    return downsamplerMode;
}

void scp::SmurfProcessor::setDownsamplerInternalFactor(std::size_t f)
{
    // Check if the factor is 0
    if (0 == f)
    {
        // This should go to a logger instead
        std::cerr << "ERROR: Trying to set factor = 0."<< std::endl;
        return;
    }

    factor = f;

    // When the factor is changed, reset the internal counter.
    setDownsamplerInternalCount(0);
}

const std::size_t scp::SmurfProcessor::getDownsamplerInternalFactor() const {
    return factor;
}

void scp::SmurfProcessor::setDownsamplerInternalCount(std::size_t c)
{
    downsamplerInternalCount = c;
}

const std::size_t scp::SmurfProcessor::getDownsamplerInternalCount() const {
  return downsamplerInternalCount;
}

void scp::SmurfProcessor::setDownsamplerExternalBitmask(std::size_t m)
{
    externalBitmask = m;
}

const std::size_t scp::SmurfProcessor::getDownsamplerExternalBitmask() const
{
    return externalBitmask;
}

const std::size_t scp::SmurfProcessor::getDownsamplerOutgoingCount() const
{
    return downsamplerOutgoingCount;
}

void scp::SmurfProcessor::acceptFrame(ris::FramePtr frame)
{
    // Release the GIL
    rogue::GilRelease noGil;

    std::size_t frameSize;

    // Hold the frame lock
    ris::FrameLockPtr lockFrame{frame->lock()};

    // Check for frames with errors or flags
    if ( frame->getError() || ( frame->getFlags() & 0x100 ) )
    {
        eLog_->error("Received frame with errors and/or flags");
        return;
    }

    // Get the frame size
    frameSize = frame->getPayload();

    // Check if the frame size is lower than the header size
    if ( frameSize < SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize )
    {
        eLog_->error("Received frame with size lower than the header size. Received frame size=%zu, expected header size=%zu",
            frameSize, SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize);
        return;
    }

    // Create a FrameAccessor object to access the frame data.
    ris::FrameIterator frameIt { frame->begin() };
    ris::FrameAccessor<uint8_t> frameAccessor { frameIt, static_cast<uint32_t>(frameSize) };

    // Do sanity checks on the incoming frame
    // - The frame has at least the header, so we can construct a (smart) pointer to it
    SmurfHeaderPtr<ris::FrameIterator> header { SmurfHeader<ris::FrameIterator>::create(frame) };

    // - Read the number of channel from the header
    uint32_t numChannels { header->getNumberChannels() };

    // - The incoming frame should have at least the supported maximum number of channels
    if ( numChannels < maxNumCh )
    {
         eLog_->error("Received frame with less channels that the maximum supported. Number of channels in received frame=%zu, supported maximum number of channels=%zu",
            numChannels, maxNumCh);
        return;
    }

    // - Check if the frame size is correct. The frame should have at least enough room to
    //   hold the number of channels defined in its header. Padded frames are allowed.
    if ( header->SmurfHeaderSize + (numChannels * sizeof(fw_t)) > frameSize )
    {
        eLog_->error("Received frame does not match expected size. Received frame size=%zu. Minimum expected sizes: header=%zu, payload=%i",
            frameSize, header->SmurfHeaderSize, numChannels * sizeof(fw_t));

        return;
    }

    // Acquire the channel mapper lock. We acquired here, so that is hold during the hold frame processing chain,
    // to avoid the 'numCh' parameter to be changed during that time.
    std::lock_guard<std::mutex> lockMapper(mutChMapper);

    // Map and unwrap data at the same time
    {
        // Acquire the lock while the unwrapper vectors are used.
        // Acquire this lock outside the loop, to increase performance.
        std::lock_guard<std::mutex> lock(mutUnwrapper);

        // Move the current data to the previous data
        previousData.swap(currentData);

        // Beginning of the data area in the frame, using the FrameAccessor
        std::vector<uint8_t>::const_iterator inIt(frameAccessor.begin() + SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize);

        // Build iterator to the data vectors
        std::vector<fw_t>::iterator     currentIt  { currentData.begin()  };
        std::vector<fw_t>::iterator     previousIt { previousData.begin() };
        std::vector<unwrap_t>::iterator inputIt    { inputData.begin()    };
        std::vector<unwrap_t>::iterator wrapIt     { wrapCounter.begin()  };

        // Map and unwrap data in a single loop
        for(auto const& m : mask)
        {
            // Get the mapped value from the framweBuffer and cast it
            // Reinterpret the bytes from the frame to 'fw_t' values. And the cast that value to 'unwrap_t' values
            *currentIt = *(reinterpret_cast<const fw_t*>(&(*( inIt + m * sizeof(fw_t) ))));
            *inputIt = static_cast<unwrap_t>(*currentIt);

            // Unwrap the value is the unwrapper is not disabled.
            // If it is disabled, don't do anything to the data
            if (!disableUnwrapper)
            {
                // Check if the value wrapped
                if ((*currentIt > upperUnwrap) && (*previousIt < lowerUnwrap))
                {
                    // Decrement wrap counter
                    *wrapIt -= stepUnwrap;
                }
                else if ((*currentIt < lowerUnwrap) && (*previousIt > upperUnwrap))
                {
                    // Increment wrap counter
                    *wrapIt += stepUnwrap;
                }

                // Add the wrap counter to the value
                *inputIt += *wrapIt;
            }

            // increase output channel index
            ++currentIt;
            ++previousIt;
            ++inputIt;
            ++wrapIt;
        }

        // Update the number of channels in the header
        header->setNumberChannels(numCh);
    }

    // Filter data
    { // filter parameter lock scope
        // Filter the data, if the filter is not disabled.
        if (!disableFilter)
        {
            // Acquire the lock while the filter parameters are used.
            std::lock_guard<std::mutex> lockFilter(mutFilter);

            // Acquire the lock for the unwrapper, as we are using the 'inputData' vector.
            std::lock_guard<std::mutex> lockUnwrapper(mutUnwrapper);

            // The pass sample buffer x and y have the following structure:
            //
            //  |     .....    |     .....    |      ...     |     .....    |
            //  ^              ^              ^              ^              ^
            //  |--------------|--------------|              |--------------|
            //     (order + 1)    (order + 1)                  (order + 1)
            //    Pass samples   Pass samples                  Pass samples
            //    of channel 0   of channel 1                  of channel (numCh-1)
            //  ^                                                           ^
            //  |-----------------------------------------------------------|
            //                    ( ( order + 1 ) * numCh )
            //
            // Each block of (order + 1) values container the current values plus the pass
            // (order) values of each channel, as a circular buffer.
            // The 'currentIndex' variable contains the index of the current sample (time = 0).
            // The value at index (currentIndex - 1) will be the sample at time = -1.
            // The value at index (currentIndex + 1) will be the sample at time = -order.

            // Copy the 'order' value to a local variable. This improve the performance of this function.
            std::size_t order_copy { order };

            // Move the 'currentIndex' index to the oldest slot in the buffer
            if (currentIndex == order_copy)
                currentIndex = 0;
            else
                ++currentIndex;

            // Create iterators
            std::vector<double>::iterator         xCh    { x.begin() + currentIndex };  // Current X value, first channel
            std::vector<double>::iterator         yCh    { y.begin() + currentIndex };  // Current Y value, first channel
            std::vector<double>::const_iterator   aIt    { a.begin()                };  // A coefficients
            std::vector<double>::const_iterator   bIt    { b.begin()                };  // B coefficients
            std::vector<unwrap_t>::const_iterator dataIt { inputData.begin()        };  // Input data

            // First, update the first points in the x and y buffers for all channels.

            // On each cycle, move the iterators to the next channel, which is in the
            // next block of size (order + 1)
            for (std::size_t ch{0}; ch < numCh; ++ch, xCh += order_copy + 1, yCh += order_copy + 1)
            {
                // New input value
                *xCh = static_cast<double>(*dataIt++);

                // Stat computing the output value
                *yCh= *bIt * *xCh;
            }

            // Now calculate the output value based on previous values from time -1 to -order
            // for all channels.

            // Index to the previous sample. It will be updated inside the loop
            std::size_t previousIndex { currentIndex };

            // Create iterator
            std::vector<double>::const_iterator xChPrev; // Previous X value, first channel
            std::vector<double>::const_iterator yChPrev; // Previous Y value, first channel

            // We first iterate over all the previous samples.
            for (std::size_t t{1}; t < order + 1; ++t)
            {
                // Get the index to the previous sample
                if (previousIndex == 0)
                    previousIndex = order_copy;
                else
                    --previousIndex;

                // Update the iterators each time we move to a previous sample
                yCh     = y.begin() + currentIndex;  // Go back to first channel
                xChPrev = x.begin() + previousIndex; // Go back to first channel
                yChPrev = y.begin() + previousIndex; // Go back to first channel
                ++aIt;                               // Go the next coefficient
                ++bIt;                               // Go the next coefficient

                // Update the output value for all channels. On each cycle, move the iterators
                // to the next channel, which is in the next block of size (order + 1)
                for (std::size_t ch{0}; ch < numCh; ++ch, yCh += order_copy + 1, xChPrev += order_copy + 1, yChPrev += order_copy + 1)
                {
                    *yCh += *bIt * *xChPrev - *aIt * *yChPrev;
                }
            }

            // Finally, divide the results for all channels by the coefficient a[0].

            // On each cycle, move the iterator to the next channel, which is in
            // the next block of size (order + 1)
            yCh = y.begin() + currentIndex;     // Go back to the first channel
            for (std::size_t ch{0}; ch < numCh; ++ch, yCh += order_copy + 1)
            {
                *yCh /= a[0];
            }
        }
    } // filter parameter lock scope

    // Downsample the data, if the downsampler is not disabled.
    // Otherwise, the data will be send on each cycle.
    if (!disableDownsampler)
    {

        if (downsamplerMode == downsamplerModeExternal) {
            // Read the averaging reeset bits from the header.  The
	    // first 10 bits are fixed rate markers, the next 18 are
	    // sequence markers.
            std::size_t externalBits { header->getTimingBits() };

	    // Mask it. E.g. If the mask is 2, then externalBits is 2
	    // for only one frame, which might for example trigger at
	    // 2 kHz.
            externalBits &= externalBitmask;

	    // Ignore frames where the masked bits aren't equal to the mask.
            if (externalBits != externalBitmask)
                return;

        } else if (downsamplerMode == downsamplerModeInternal) {
            // Use the counter in this program to trigger.
	    ++downsamplerInternalCount;
	  
            if (downsamplerInternalCount < factor)
                return;

            // Set the counter to 0.
	    setDownsamplerInternalCount(0);
        }

        // Increase until the next trigger.
        ++downsamplerOutgoingCount;
    }

    // Give the data to the Tx thread to be sent to the next slave.
    {
        // Wait for the TX thread to finish sending the previous data.
        std::unique_lock<std::mutex> lock(txMutex);
        txCV.wait( lock, [this]{ return !txDataReady; } );
    }
    {
        std::lock_guard<std::mutex> lock(txMutex);

        // Copy the header
        std::copy(frameAccessor.begin(), frameAccessor.begin() + SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize, headerCopy.begin());

        // Copy the data
        {
            // Hold the mutex while we copy the data
            std::lock_guard<std::mutex> lockData(outDataMutex);

            // Acquire the lock for the unwrapper, as we are using the 'inputData' vector.
            std::lock_guard<std::mutex> lockUnwrapper(mutUnwrapper);

            // Check if the filter was disabled. If it was disabled, use the 'inputData' vector as source.
            // Otherwise, use the 'y' vector, applying the gain and casting.
            if (disableFilter)
            {
                // Just cast the data type
                std::transform( inputData.begin(), inputData.end(), outData.begin(),
                    [this](const unwrap_t& v) -> filter_t { return static_cast<filter_t>(v); });
            }
            else
            {
                // Create iterators
                std::vector<double>::const_iterator yCh(y.begin() + currentIndex); // Filtered value, first channel
                std::vector<filter_t>::iterator     outIt(outData.begin());        // Destination

                // Multiply the values by the gain, and cast the result
                // to 'filter_t' into he outData buffer.
                // The current filtered values for each channel will be located at steps of (order+1) in the 'y' buffer.
                // On each cycle, move the iterator to the next channel, which is in
                // the next block of size (order + 1)
                for (std::size_t ch{0}; ch < numCh; ++ch, yCh += order + 1)
                {
                    *outIt++ = static_cast<filter_t>(*yCh * gain);
                }
            }
        }

        txDataReady = true;
    }
    // Notify the Tx thread that new data is ready
    txCV.notify_one();
}

void scp::SmurfProcessor::pktTansmitter()
{
    std::cout << "Transmitter thread started..." << std::endl;

    // Infinite loop
    for(;;)
    {
        // Wait until data is ready, with a 10s timeout
        std::unique_lock<std::mutex> lock(txMutex);
        if(txCV.wait_for( lock, std::chrono::seconds(10), [this]{ return txDataReady; } ))
        {
            // Output frame size. Start with the size of the header
            std::size_t outFrameSize = SmurfHeaderRO<std::vector<uint8_t>::iterator>::SmurfHeaderSize;

            // Extract the number of channels from the passed header
            SmurfHeaderROPtr<std::vector<uint8_t>::iterator> header { SmurfHeaderRO<std::vector<uint8_t>::iterator>::create(headerCopy) };
            std::size_t numChannels { header->getNumberChannels() };

            if (payloadSize > numChannels)
                // If the payload size is greater that the number of channels, then reserved
                // that number of channels in the output frame.
                outFrameSize += payloadSize * sizeof(filter_t);
            else
                // Otherwise, the size of the frame will only hold the number of channels
                outFrameSize += numChannels * sizeof(filter_t);

            // Request a new frame
            ris::FramePtr outFrame = reqFrame(outFrameSize, true);
            outFrame->setPayload(outFrameSize);
            ris::FrameIterator outFrameIt = outFrame->beginWrite();

            // Copy the header from the input frame to the output frame
            outFrameIt = std::copy(headerCopy.begin(), headerCopy.end(), outFrameIt);

            // Copy the data to the output frame
            {
                std::lock_guard<std::mutex> lockData(outDataMutex);

                // Get a pointer to the underlying bytes of the outData vector. In this way,
                // we can memcopy the data to the output frame payload area, using a pointer as
                // well) directly.
                uint8_t *outDataPtr { reinterpret_cast< uint8_t* >( outData.data() ) };
                std::memcpy(outFrameIt.ptr(), outDataPtr, outData.size()*sizeof(filter_t));
            }

            // Send the frame to the next slave.
            sendFrame(outFrame);

            // Notify the main thread that we can receive new data
            // Manual unlocking is done before notifying, to avoid waking up
            // the waiting thread only to block again (see notify_one for details)
            txDataReady = false;
            lock.unlock();
            txCV.notify_one();
        }

        // Check if we should stop the loop
        if (!runTxThread)
        {
            std::cout << "pktTansmitter interrupted." << std::endl;
            return;
        }
    }
}
