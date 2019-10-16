#ifndef _SMURF_CORE_COMMON_TESBIASARRAY_H_
#define _SMURF_CORE_COMMON_TESBIASARRAY_H_

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

#include <stdexcept>
#include <mutex>
#include <vector>
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameLock.h>
#include <rogue/interfaces/stream/FrameIterator.h>

namespace ris = rogue::interfaces::stream;

template <typename T>
class TesBiasArray;

template <typename T>
using TesBiasArrayPtr = std::shared_ptr<TesBiasArray<T>>;

// Class to handle TES bias array of values.
template <typename T>
class TesBiasArray
{
private:
    // Iterator to the start of the TES Bias area in the SMuRF Header
    T dataIt;

    // Mutex, to safety access the data from different threads
    std::mutex mut;

    // Helper class to handler indexes of TES bias words
    // TES bias are 20-bit words = 2.5 bytes
    // 16x TES bias occupy 40 bytes, which are divided into 8 blocks of 2 words (5 bytes) each
    // From each word index (0-15) a block number (0-7), and a word sub-index inside that block (0-1)
    // is generated. For example, TES bias 6 and 7 are both located on block 3; with 6 at the first word
    // and 7 on the second word inside that block.
    class WordIndex
    {
    public:
        WordIndex(std::size_t i) : index( i ), block( i / 2 ), word( i % 2 ) {};

        std::size_t Index() const { return index; }; // 20-bit word index
        std::size_t Block() const { return block; }; // 2-word block number
        std::size_t Word()  const { return word;  }; // Word index inside the block

        bool operator >= (std::size_t rhs) const { return index >= rhs; };

    private:
        std::size_t index; // 20-bit word index (0-15)
        std::size_t block; // 2-word block index (0-7)
        std::size_t word;  // Word index in the word block (0-1)
    };

    // Helper Union to access individual bytes of a word
    typedef union
    {
        uint32_t w;
        uint8_t  b[4];
    } U;

    // Helper Struct to sign extend 20-bit values
    typedef struct
    {
        int32_t word:20;
    } S;

public:
    TesBiasArray(T it);
    ~TesBiasArray() {};

    // Number of TES Bias values.
    static const std::size_t TesBiasCount = 16;

    // Size of each TES Bias value (in bits)
    static const std::size_t TesBiasBitSize = 20;

    // Size of the buffer needed to store the TES Bias (in bytes)
    static const std::size_t TesBiasBufferSize = TesBiasCount * TesBiasBitSize / 8;

    static TesBiasArrayPtr<T> create(T it);

    // Change the data pointer
    void setDataIt(T it);

    // Write a TES bias value
    void setWord(const WordIndex& index, int32_t value) const;

    // Read a TES bias value
    const int32_t getWord(const WordIndex& index) const;

    // Method to get the mutex
    std::mutex* getMutex();
};

#endif
