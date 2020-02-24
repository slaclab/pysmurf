#ifndef _SMURF_CORE_COMMON_HELPERS_H_
#define _SMURF_CORE_COMMON_HELPERS_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Helpers Functions
 * ----------------------------------------------------------------------------
 * File          : Helpers.h
 * Created       : 2019-09-27
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Helper Functions.
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

#include <chrono>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>

namespace ris = rogue::interfaces::stream;

namespace helpers
{
    // Function to read a data word from a data frame.
    // - 'it' iterator should point to the payload area (not to the header).
    // - 'offset' is expressed in number of words.
    template<typename T>
    const T getWord(ris::FrameIterator it, std::size_t offset)
    {
        union
        {
            T       w;
            uint8_t b[sizeof(T)];
        } temp;

        for (std::size_t i{0}; i < sizeof(T); ++i)
            temp.b[i] = *(it + offset * sizeof(T) + i);

        return temp.w;
    };

    // Function to write a data word from a data frame.
    // - 'it' iterator should point to the payload area (not to the header).
    // - 'offset' is expressed in number of words.
    template<typename T>
    void setWord(ris::FrameIterator it, std::size_t offset, T value)
    {
        union
        {
            T       w;
            uint8_t b[sizeof(T)];
        } temp;

        temp.w = value;

        for (std::size_t i{0}; i < sizeof(T); ++i)
            *(it + offset * sizeof(T) + i) = temp.b[i];
    };

    // Get the current time in nanoseconds.
    // Use the steady clock which guarantees to be monotonically increasing.
    inline uint64_t getTimeNS()
    {
        return std::chrono::time_point_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now()).time_since_epoch().count();
    };
}

#endif
