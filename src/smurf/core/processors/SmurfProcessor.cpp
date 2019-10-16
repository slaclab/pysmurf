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

namespace scp = smurf::core::processors;

scp::SmurfProcessor::SmurfProcessor()
:
    ris::Slave(),
    ris::Master(),
    frameBuffer(SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize + maxNumCh * sizeof(fw_t),0),
    numCh(528),
    payloadSize(numCh),
    mask(numCh,0),
    disableUnwrapper(false),
    currentData(numCh, 0),
    previousData(numCh, 0),
    wrapCounter(numCh, 0),
    disableFilter(false),
    order(4),
    gain(1),
    a( order + 1 ,1 ),
    b( order + 1, 1 ),
    currentBlockIndex(order),
    x( (order +1) * numCh ),
    y( (order +1) * numCh ),
    outData(numCh,0),
    disableDownsampler(false),
    factor(20),
    sampleCnt(0),
    headerCopy(SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize, 0),
    dataCopy(numCh * sizeof(filter_t), 0),
    runTxThread(true),
    txDataReady(false),
    pktTransmitterThread(std::thread( &SmurfProcessor::pktTansmitter, this ))
{
    if( pthread_setname_np( pktTransmitterThread.native_handle(), "pktTransmitter" ) )
        perror( "pthread_setname_np failed for pktTransmitterThread thread" );
}

scp::SmurfProcessorPtr scp::SmurfProcessor::create()
{
    return std::make_shared<SmurfProcessor>();
}

void scp::SmurfProcessor::setup_python()
{
    bp::class_< scp::SmurfProcessor,
                scp::SmurfProcessorPtr,
                bp::bases<ris::Slave,ris::Master>,
                boost::noncopyable >
                ("SmurfProcessor",bp::init<>())
        // Channel mapping variables
        .def("setUnwrapperDisable",     &SmurfProcessor::setUnwrapperDisable)
        .def("getUnwrapperDisable",     &SmurfProcessor::getUnwrapperDisable)
        .def("getNumCh",                &SmurfProcessor::getNumCh)
        .def("setPayloadSize",          &SmurfProcessor::setPayloadSize)
        .def("getPayloadSize",          &SmurfProcessor::getPayloadSize)
        .def("setMask",                 &SmurfProcessor::setMask)
        .def("getMask",                 &SmurfProcessor::getMask)
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
        // Downsampler variables
        .def("setDownsamplerDisable",   &SmurfProcessor::setDownsamplerDisable)
        .def("getDownsamplerDisable",   &SmurfProcessor::getDownsamplerDisable)
        .def("setFactor",               &SmurfProcessor::setFactor)
        .def("getFactor",               &SmurfProcessor::getFactor)
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

    // If the payload size if defined, only process up to that number of channels
    if (payloadSize && payloadSize < listSize)
        listSize = payloadSize;

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
    numCh = listSize;

    // Reset the Unwrapper when the number of channels change
    resetUnwrapper();

    // Reset the filter
    // Take the mutex before changing the filter parameters
    // This make sure that the new order value is not used before
    // the a and b array are resized.
    std::lock_guard<std::mutex> lockFilter(mutFilter);
    resetFilter();
}

const bp::list scp::SmurfProcessor::getMask() const
{
    bp::list temp;

    for (auto const &v : mask)
        temp.append(v);

    return temp;
}

void scp::SmurfProcessor::setUnwrapperDisable(bool d)
{
    disableUnwrapper = d;
}

const bool scp::SmurfProcessor::getUnwrapperDisable() const
{
    return disableUnwrapper;
}

void scp::SmurfProcessor::resetUnwrapper()
{
    std::vector<unwrap_t>(numCh).swap(currentData);
    std::vector<unwrap_t>(numCh).swap(previousData);
    std::vector<unwrap_t>(numCh).swap(wrapCounter);
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

        // When the order it change, reset the filter
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

    // Verify that the input list is not empty.
    // If empty, set the coefficients vector to a = [1.0].
    if (listSize == 0)
    {
        // This should go to a logger instead
        std::cerr << "ERROR: Trying to set an empty set of a coefficients. Defaulting to 'a = [1.0]'"<< std::endl;
        temp.push_back(1.0);
        a.swap(temp);

        return;
    }

    // Verify that the first coefficient is not zero.
    // if it is, set the coefficients vector to a = [1.0].
    if (l[0] == 0)
    {
        // This should go to a logger instead
        std::cerr << "ERROR: The first a coefficient can not be zero. Defaulting to 'a = [1.0]'"<< std::endl;
        temp.push_back(1.0);
        a.swap(temp);
        return;
    }

    // Extract the coefficients coming from python into a temporal vector
    for (std::size_t i{0}; i < listSize; ++i)
    {
        temp.push_back(bp::extract<double>(l[i]));
    }

    // Update the a vector with the new values
    a.swap(temp);

    // Check that the a coefficient vector size is at least 'order + 1'.
    // If not, add expand it with zeros.
    if ( a.size() < (order + 1) )
        a.resize(order +  1, 0);
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

    // Verify that the input list is not empty.
    // If empty, set the coefficients vector to a = [0.0].
    if (listSize == 0)
    {
        // This should go to a logger instead
        std::cerr << "ERROR: Trying to set an empty set of a coefficients. Defaulting to 'b = [0.0]'"<< std::endl;
        temp.push_back(0.0);
        b.swap(temp);

        return;
    }

    // Extract the coefficients coming from python into a temporal vector
    for (std::size_t i{0}; i < len(l); ++i)
    {
        temp.push_back(bp::extract<double>(l[i]));
    }

    // Update the a vector with the new values
    b.swap(temp);

    // Check that the b coefficient vector size is at least 'order + 1'.
    // If not, add expand it with zeros.
    if ( b.size() < (order + 1) )
        b.resize(order +  1, 0);
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
    std::vector<double>( (order + 1) * numCh ).swap(x);
    std::vector<double>( (order + 1) * numCh ).swap(y);

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
    currentBlockIndex = 0;
}

void scp::SmurfProcessor::setDownsamplerDisable(bool d)
{
    disableDownsampler = d;
}

const bool scp::SmurfProcessor::getDownsamplerDisable() const
{
    return disableDownsampler;
}

void scp::SmurfProcessor::setFactor(std::size_t f)
{
    // Check if the factor is 0
    if (0 == f)
    {
        // This should go to a logger instead
        std::cerr << "ERROR: Trying to set factor = 0."<< std::endl;
        return;
    }

    factor = f;

    // When the factor is changed, reset the counter.
    resetDownsampler();
}

const std::size_t scp::SmurfProcessor::getFactor() const
{
    return factor;
}

void scp::SmurfProcessor::resetDownsampler()
{
    sampleCnt = 0;
}

void scp::SmurfProcessor::acceptFrame(ris::FramePtr frame)
{
    // Copy the frame into a STL container
    {
        // Hold the frame lock
        ris::FrameLockPtr lockFrame{frame->lock()};

        std::vector<uint8_t>::iterator outIt(frameBuffer.begin());

        // Copy using BufferIterators in combination with std::copy for performance reasons
        for (ris::Frame::BufferIterator inIt=frame->beginBuffer(); inIt != frame->endBuffer(); ++inIt)
            outIt = std::copy((*inIt)->begin(), (*inIt)->endPayload(), outIt);
    }

    // Map and unwrap data at the same time
    {
        // Acquire the lock while the channel map parameters are used.
        std::lock_guard<std::mutex> lockParam(mutChMapper);

        // Move the current data to the previous data
        previousData.swap(currentData);

        // Begining of the data area in the frameBuffer
        std::vector<uint8_t>::iterator inIt(frameBuffer.begin() + SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize);

        // Output channel index
        std::size_t i{0};

        // Map and unwrap data in a single loop
        for(auto const& m : mask)
        {
            // Get the mapped value from the framweBuffer and cast it
            currentData.at(i) = static_cast<unwrap_t>(*(inIt + m * sizeof(fw_t)));

            // Unwrap the value is the unwrapper is not disabled.
            // If it is disabled, don't do anything to the data
            if (!disableUnwrapper)
            {
                // Check if the value wrapped
                if ((currentData.at(i) > upperUnwrap) && (previousData.at(i) < lowerUnwrap))
                {
                    // Decrement wrap counter
                    wrapCounter.at(i) -= stepUnwrap;
                }
                else if ((currentData.at(i) < lowerUnwrap) && (previousData.at(i) > upperUnwrap))
                {
                    // Increment wrap counter
                    wrapCounter.at(i) += stepUnwrap;
                }

                // Add the wrap counter to the value
                currentData.at(i) += wrapCounter.at(i);
            }

            // increase output channel index
            ++i;
        }

        // Update the number of channels in the header
        SmurfHeaderPtr<std::vector<uint8_t>::iterator> header{ SmurfHeader<std::vector<uint8_t>::iterator>::create(frameBuffer) };
        header->setNumberChannels(numCh);
    }

    // Filter data
    { // filter parameter lock scope
        // Filter the data, if the filter is not disabled.
        if (!disableFilter)
        {
            // Acquire the lock while the filter parameters are used.
            std::lock_guard<std::mutex> lockParam(mutFilter);

            // Update the 'current' index to the oldest slot in the buffer
            currentBlockIndex = (currentBlockIndex + 1) % (order + 1);

            // Get index to the current data block
            std::size_t currentBlockPointer{currentBlockIndex * numCh};

            // Create iterators
            auto xIt(x.begin());
            auto yIt(y.begin());
            auto aIt(a.begin());
            auto bIt(b.begin());
            auto dataIt(currentData.begin());

            // Iterate over the channel samples
            for (std::size_t ch{0}; ch < numCh; ++ch)
            {
                // Cast the input value to double into the output buffer
                *(xIt + currentBlockPointer) = static_cast<double>( *dataIt );

                // Start computing the output value
                *(yIt + currentBlockPointer) = *bIt * *(xIt + currentBlockPointer);

                // Iterate over the pass samples
                for (std::size_t t{1}; t < order + 1; ++t)
                {
                    // Compute the correct index in the 'circular' buffer
                    std::size_t passBlockIndex{ ( ( order + currentBlockIndex - t + 1 ) % (order + 1) ) * numCh };

                    *(yIt + currentBlockPointer) += *(bIt + t) * *(xIt + passBlockIndex)
                        - *(aIt + t) * *(yIt + passBlockIndex);
                }

                // Divide the resulting value by the first a coefficient
                *(yIt + currentBlockPointer) /= *(aIt);

                //Move to the next channel sample
                ++xIt;
                ++yIt;
                ++dataIt;
            }
        }

    } // filter parameter lock scope

    // Downsample the data, if the downsampler is not disabled.
    // Otherwise, the data will be send on each cycle.
    if (!disableDownsampler)
    {
        // Downsampler. If we haven't reached the factor counter, we don't do anything
        // When we reach the factor counter, we send the resulting frame.
        if (++sampleCnt < factor)
            return;

        // Reset the downsampler
        resetDownsampler();
    }

    // Give the data to the Tx thread to be sent to the next slave.
    {
        // Copy the header
        std::copy(frameBuffer.begin(), frameBuffer.begin() + SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize, headerCopy.begin());

        // Copy the data
        {
            // Iterator to the current output buffer
            auto yIt(y.begin() + currentBlockIndex * numCh);

            // Hold the mutex while we copy the data
            std::lock_guard<std::mutex> lock(outDataMutex);

            // Check if the filter was disabled. If it was disabled, use the 'currentData' vector as source.
            // Otherwise, use the 'y' vector, applying the gain and casting.
            if (disableFilter)
            {
                // Just cast the data type
                std::transform( currentData.begin(), currentData.end(), outData.begin(),
                    [this](const unwrap_t& v) -> filter_t { return static_cast<filter_t>(v); });
            }
            else
            {
                // Multiply the values by the gain, and cast the result
                // to 'filter_t' into he outData buffer
                std::transform( y.begin() + currentBlockIndex * numCh,
                    y.begin() + currentBlockIndex * numCh + numCh,
                    outData.begin(),
                    [this](const double& v) -> filter_t { return static_cast<filter_t>(v * gain); });
            }
        }

        // Notify the Tx thread that new data is ready
        txDataReady = true;
        std::unique_lock<std::mutex> lock(txMutex);
        txCV.notify_all();
    }
}

void scp::SmurfProcessor::pktTansmitter()
{
    std::cout << "Transmitter thread started..." << std::endl;

    // Infinite loop
    for(;;)
    {
        // Check if new data is ready
        if ( !txDataReady )
        {
            // Wait until data is ready, with a 10s timeout
            std::unique_lock<std::mutex> lock(txMutex);
            txCV.wait_for( lock, std::chrono::seconds(10) );
        }
        else
        {
            // Request a new frame, to hold the same payload as the input frame
            std::size_t outFrameSize = SmurfHeader<std::vector<uint8_t>::iterator>::SmurfHeaderSize;

            if (payloadSize)
                // If the payload size was defined, used that number as the number of channel in the output frame.
                outFrameSize += payloadSize * sizeof(filter_t);
            else
                // Otherwise, the size of the frame will only hold the number of channels
                outFrameSize += numCh * sizeof(filter_t);

            ris::FramePtr outFrame = reqFrame(outFrameSize, true);
            outFrame->setPayload(outFrameSize);
            ris::FrameIterator outFrameIt = outFrame->beginWrite();

            // Copy the header from the input frame to the output frame
            outFrameIt = std::copy(headerCopy.begin(), headerCopy.end(), outFrameIt);

            // Copy the data to the output frame
            {
                std::lock_guard<std::mutex> lock(outDataMutex);
                std::size_t i{0};
                for(auto it = outData.begin(); it != outData.end(); ++it)
                    helpers::setWord<filter_t>(outFrameIt, i++, *it);
            }
            // Send the frame to the next slave.
            sendFrame(outFrame);

            // Clear the flag
            txDataReady = false;
        }

        // Check if we should stop the loop
        if (!runTxThread)
        {
            std::cout << "pktTansmitter interrupted." << std::endl;
            return;
        }
    }
}
