#ifndef _SMURF_CORE_EMULATORS_STREAMDATAEMULATOR_H_
#define _SMURF_CORE_EMULATORS_STREAMDATAEMULATOR_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Emulator
 * ----------------------------------------------------------------------------
 * File          : StreamDataEmulator.h
 * Created       : 2019-10-28
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

#include <type_traits>
#include <limits>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include <rogue/interfaces/stream/FrameAccessor.h>
#include <rogue/interfaces/stream/Buffer.h>
#include <rogue/interfaces/stream/Slave.h>
#include <rogue/interfaces/stream/Master.h>
#include <rogue/GilRelease.h>
#include <rogue/Logging.h>
#include "smurf/core/common/SmurfHeader.h"
#include "smurf/core/common/Helpers.h"
#include <random>

namespace bp  = boost::python;
namespace ris = rogue::interfaces::stream;

namespace smurf
{
    namespace core
    {
        namespace emulators
        {
            template<typename T>
            class StreamDataEmulator;

            template<typename T>
            using StreamDataEmulatorPtr = std::shared_ptr< StreamDataEmulator<T> >;

            template<typename T>
            class StreamDataEmulator : public ris::Slave, public ris::Master
            {
            private:
                // Data types
                // - Data type from firmware (T), in its unsigned version
                typedef typename std::make_unsigned<T>::type uT_t;

            public:
                StreamDataEmulator();
                ~StreamDataEmulator() {};

                static StreamDataEmulatorPtr<T> create();

                static void setup_python(const std::string& name);

                // Accept new frames
                void acceptFrame(ris::FramePtr frame);

                // Disable the processing block. The data
                // will just pass through to the next slave
                void       setDisable(bool d);
                const bool getDisable() const;

                // Set/Get operation mode
                void      setType(int value);
                const int getType() const;

                // Set/Get signal amplitude
                void       setAmplitude(uT_t value);
                const uT_t getAmplitude() const;

                // Set/Get signal offset
                void    setOffset(T value);
                const T getOffset() const;

                // Set/Get  signal period
                void              setPeriod(std::size_t value);
                const std::size_t getPeriod() const;

            private:
                // Types of signal
                enum class SignalType { Zeros, ChannelNumber, Random, Square, Sawtooth, Triangle, Sine, DropFrame, Size };

                // Maximum amplitude value
                const uT_t maxAmplitude = std::numeric_limits<uT_t>::max();

                // Signal generator methods
                void genZeroWave(ris::FrameAccessor<T> &dPtr)          const;
                void genChannelNumberWave(ris::FrameAccessor<T> &dPtr) const;
                void genRandomWave(ris::FrameAccessor<T> &dPtr);
                void genSquareWave(ris::FrameAccessor<T> &dPtr);
                void getSawtoothWave(ris::FrameAccessor<T> &dPtr);
                void genTriangleWave(ris::FrameAccessor<T> &dPtr);
                void genSinWave(ris::FrameAccessor<T> &dPtr);
                void genFrameDrop();

                // Logger
                std::shared_ptr<rogue::Logging> eLog_;

                // Mutex
                std::mutex  mtx_;

                // Variables
                bool        disable_;       // Disable flag
                SignalType  type_;          // signal type
                uT_t        amplitude_;     // Signal amplitude
                T           offset_;        // Signal offset
                std::size_t period_;        // Signal period
                std::size_t halfPeriod_;    // Signal half period
                std::size_t periodCounter_; // Frame period counter
                bool        dropFrame_;     // Flag to indicate if the frame should be dropped

                // Variables use to generate random numbers
                std::random_device                     rd;  // Will be used to obtain a seed for the random number engine
                std::mt19937                           gen; // Standard mersenne_twister_engine seeded with rd()
                std::uniform_real_distribution<double> dis; // Use to transform the random unsigned int generated by gen into a double

            };
        }
    }
}

#endif
