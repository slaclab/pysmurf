/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF TES Bias Array
 * ----------------------------------------------------------------------------
 * File          : SmurfHeader.h
 * Created       : 2019-10-01
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF TES Bias Array Class.
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

#include "smurf/core/common/TesBiasArray.h"

template <typename T>
TesBiasArray<T>::TesBiasArray(T it)
:
    dataIt(it)
{
}

template <typename T>
TesBiasArrayPtr<T> TesBiasArray<T>::create(T it)
{
    return std::make_shared<TesBiasArray<T>>(it);
}

template <typename T>
void TesBiasArray<T>::setDataIt(T it)
{
    dataIt = it;
}

template <typename T>
void TesBiasArray<T>::setWord(const WordIndex& index, int32_t value) const
{
    if (index >= TesBiasCount)
        throw std::runtime_error("Trying to write a TES bias value in an address of out the buffer range.");

    if (index.Word() == 0)
    {
        // Create an union with the passed value
        U v { static_cast<uint32_t>(value) };

        // Copy the value
        std::size_t i{5*index.Block()};
        *(dataIt + i) = v.b[0];
        *(dataIt + i + 1) = v.b[1];
        uint8_t temp = *(dataIt + i + 2);
        temp &= 0xf0;
        temp |= (v.b[2] & 0x0f);
        *(dataIt + i + 2) = temp;
    }
    else
    {
        // Create an union with the passed value
        U v { static_cast<uint32_t>(value) << 4 };

        // Copy the value
        std::size_t i{5*index.Block() + 2};
        uint8_t temp = *(dataIt + i);
        temp &= 0x0f;
        temp |= (v.b[0] & 0xf0);
        *(dataIt + i) = temp;
        *(dataIt + i + 1) = v.b[1];
        *(dataIt + i + 2) = v.b[2];
    }
};

template <typename T>
const int32_t TesBiasArray<T>::getWord(const WordIndex& index) const
{
    if (index >= TesBiasCount)
        throw std::runtime_error("Trying to read a TES bias value in an address of out the buffer range.");

    std::size_t offset(0), shift(0);

    if (index.Word())
    {
        offset = 2;
        shift = 4;
    }

    U v { 0 };
    std::size_t i{5*index.Block() + offset};
    for (std::size_t j{0}; j < 3; ++j)
        v.b[j] = *(dataIt + i + j);

    return ( ( static_cast<int32_t>(v.w) >> shift) & 0xfffff );
}

template <typename T>
std::mutex* TesBiasArray<T>::getMutex()
{
  return &mut;
};

template class TesBiasArray<ris::FrameIterator>;
template class TesBiasArray<std::vector<uint8_t>::iterator>;
