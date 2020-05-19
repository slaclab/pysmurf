#ifndef _SMURF_CORE_COMMON_TIMER_H_
#define _SMURF_CORE_COMMON_TIMER_H_

/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Timer
 * ----------------------------------------------------------------------------
 * File          : Timer.h
 * Created       : 2019-10-01
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Timer Class.
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
#include <numeric>
#include <cstdio>
#include "smurf/core/common/Helpers.h"

// This timer will measure the time (in ns) the object is in scope.
// So, it can be used like this:
//    { <---- start measuring time here
//        Timer t {"MyTimer"};
//
//      ... code ...
//
//    } <---- stop measuring time here
class Timer
{
public:
    Timer(std::string n)
    :
        name(n),
        t(helpers::getTimeNS())
    {
    };

    ~Timer()
    {
        std::cout << name << ", start = " << t << std::endl;
        std::cout << name << ", end   = " << helpers::getTimeNS() - t << std::endl;
    };

private:
    std::string name;
    uint64_t    t;
};

// This timer will take a defined number of samples (in ns), and it will print
// statistics when done (maximum, minimum, and average values). Also, if
// and output file is specified, it will write the histogram data to it. The
// histogram data will be written only once, the first time the number of
// samples is reached.
// Time samples are taken using the start() and stop() methods. So, it can
// be used like this:
//    TimerWithStats t { "MyTimer", 1000000, "/tmp/MyTimer.dat" };
//    for (std::size_t i {0}; i <= 1000000; ++i)
//    {
//        t.start();
//
//        ... code ...
//
//        t.stop();
//
//    }
class TimerWithStats
{
public:
    TimerWithStats(const std::string& n, std::size_t ns, const std::string& f = "", std::size_t hs = 100)
    :
        name(n),
        numSamples(ns),
        index(0),
        samples(ns),
        histogramStep(hs),
        fileName(f)
    {
        if ( ! fileName.empty() )
        {
            if ( ! ( file = fopen( fileName.c_str(), "w") ) )
            {
                std::cerr << "Unable to open file " << fileName << std::endl;
            }
        }

    };

    ~TimerWithStats()
    {
        if ( file != NULL )
            fclose(file);
    };

    void start()
    {
        t = helpers::getTimeNS();
    };

    void stop()
    {
        samples.at(index) = helpers::getTimeNS() - t;

        if (++index >= numSamples)
        {
            printStats();

            if ( file != NULL )
            {
                writeHistogram();
                fclose(file);
                file = NULL;
            }

            index = 0;
            std::vector<uint64_t>(numSamples).swap(samples);
        }
    };

    void printStats()
    {
        std::cout << "Timer: " << name << std::endl;
        std::cout << "-----------" << std::endl;
        std::cout << "Size    = " << samples.size() << std::endl;
        std::cout << "Maximum = " << *(std::max_element(samples.begin(), samples.end())) << std::endl;
        std::cout << "Average = " << std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size() << std::endl;
        std::cout << "Minimum = " << *(std::min_element(samples.begin(), samples.end())) << std::endl;
        std::cout << "-----------" << std::endl;
        std::cout << std::endl;
    };

    void writeHistogram()
    {
        std::vector<std::size_t> h (histogramStep, 0);

        fprintf(file, "# microsecond\tcounts\n");
        for (auto const &s : samples)
        {
            std::size_t i { static_cast<std::size_t>( s / 1000 ) };
            if ( i > (histogramStep -1) )
                i = histogramStep - 1;

            ++h.at(i);
        }

        std::size_t i {0};
        for(auto const &l : h)
            fprintf(file,  "%04zu\t\t%zu\n", ++i, l );
    };

private:
    std::string           name;
    std::size_t           numSamples;
    std::size_t           index;
    std::vector<uint64_t> samples;
    uint64_t              t;
    std::size_t           histogramStep;
    std::string           fileName;
    FILE*                 file;
};
#endif
