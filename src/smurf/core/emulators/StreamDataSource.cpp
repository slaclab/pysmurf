
/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Source
 * ----------------------------------------------------------------------------
 * File          : StreamDataSource.cpp
 * Created       : 2019-11-12
 *-----------------------------------------------------------------------------
 * Description :
 *    SMuRF Data StreamDataSource Class
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
#include "smurf/core/emulators/StreamDataSource.h"
#include "smurf/core/common/SmurfHeader.h"
#include <rogue/Logging.h>
#include <rogue/GilRelease.h>
#include <cmath>

namespace sce = smurf::core::emulators;
namespace ris = rogue::interfaces::stream;

sce::StreamDataSource::StreamDataSource() {
   sourcePeriod_ = 0;
   sourceEnable_ = false;
   crateId_      = 0;
   slotNumber_   = 0;
   frameCounter_ = 0;

   eLog_ = rogue::Logging::create("pysmurf.source");

   threadEn_ = true;
   thread_ = new std::thread(&sce::StreamDataSource::runThread, this);
}

sce::StreamDataSource::~StreamDataSource() { 
   threadEn_ = false;
   rogue::GilRelease noGil;
   thread_->join();
}

sce::StreamDataSourcePtr sce::StreamDataSource::create() {
    return std::make_shared<StreamDataSource>();
}

void sce::StreamDataSource::setup_python() {
   bp::class_< sce::StreamDataSource,
               sce::StreamDataSourcePtr,
               bp::bases<ris::Master>,
               boost::noncopyable >
               ("StreamDataSource",bp::init<>())

        // Sin Generate Parameters
        .def("setSourcePeriod",   &StreamDataSource::setSourcePeriod)
        .def("getSourcePeriod",   &StreamDataSource::getSourcePeriod)
        .def("setSourceEnable",   &StreamDataSource::setSourceEnable)
        .def("getSourceEnable",   &StreamDataSource::getSourceEnable)
        .def("setCrateId",        &StreamDataSource::setCrateId)
        .def("getCrateId",        &StreamDataSource::getCrateId)
        .def("setSlotNum",        &StreamDataSource::setSlotNum)
        .def("getSlotNum",        &StreamDataSource::getSlotNum)
    ;
    bp::implicitly_convertible< sce::StreamDataSourcePtr, ris::MasterPtr >();
}

void sce::StreamDataSource::setSourcePeriod(uint32_t value) {
   sourcePeriod_ = value;
}

uint32_t sce::StreamDataSource::getSourcePeriod() {
   return sourcePeriod_;
}

void sce::StreamDataSource::setSourceEnable(bool enable) {

   if ( enable && ! sourceEnable_ ) {
      frameCounter_ = 0;
      gettimeofday(&lastTime_,NULL);
   }

   sourceEnable_ = enable;
}

bool sce::StreamDataSource::getSourceEnable() {
   return sourceEnable_;
}

void sce::StreamDataSource::setCrateId(uint8_t value) {
   crateId_ = value;
}

uint8_t sce::StreamDataSource::getCrateId() {
   return crateId_;
}

void sce::StreamDataSource::setSlotNum(uint8_t value) {
   slotNumber_ = value;
}

uint8_t sce::StreamDataSource::getSlotNum() {
   return slotNumber_;
}

void sce::StreamDataSource::runThread() {
   ris::FramePtr frame;
   struct timeval endTime;
   struct timeval periodTime;
   struct timeval currTime;
   uint64_t unixTime;
   uint32_t size;
   uint32_t i;

   eLog_->logThreadId();

   while(threadEn_) {
      if (sourceEnable_ && sourcePeriod_) {

         // Check if period passed
         periodTime.tv_sec  = sourcePeriod_ / 1000000;
         periodTime.tv_usec = sourcePeriod_ % 1000000;
         gettimeofday(&currTime,NULL);
         timeradd(&lastTime_,&periodTime,&endTime);

         unixTime  = currTime.tv_sec * 1e9;
         unixTime += currTime.tv_usec * 1e3;

         if ( timercmp(&currTime,&endTime,>=)) {
            size = SmurfHeader<ris::FrameIterator>::SmurfHeaderSize + 2*4096;

            frame = this->reqFrame(size, true);

            // Header record
            SmurfHeaderPtr<ris::FrameIterator> header = SmurfHeader<ris::FrameIterator>::create(frame);

            // Init Header
            header->setVersion(1);                     // Set protocol version
            header->setCrateID(crateId_);              // Set ATCA crate ID
            header->setSlotNumber(slotNumber_);        // Set ATCA slot number
            header->setTimingConfiguration(0);         // Set timing configuration
            header->setNumberChannels(512);            // Set number of channel in this packet
            header->setUnixTime(unixTime);             // Set 64 bit unix time nanoseconds
            header->setFluxRampIncrement(0);           // Set signed 32 bit integer for increment
            header->setFluxRampOffset(0);              // Set signed 32 it integer for offset
            header->setCounter0(0);                    // Set 32 bit counter since last 1Hz marker
            header->setCounter1(0);                    // Set 32 bit counter since last external input
            header->setCounter2(0);                    // Set 64 bit timestamp
            header->setAveragingResetBits(0);          // Set up to 32 bits of average reset from timing system
            header->setFrameCounter(frameCounter_);    // Set locally genreate frame counter 32 bit
            header->setTESRelaySetting(0);             // Set TES and flux ramp relays, 17bits in use now
            header->setExternalTimeClock(0);           // Set Syncword from mce for mce based systems (40 bit including header)
            header->setControlField(0);                // Set control field word
            header->setClearAverageBit(0);             // Set control field's clear average and unwrap bit (bit 0)
            header->setDisableStreamBit(0);            // Set control field's disable stream to MCE bit (bit 1)
            header->setDisableFileWriteBit(0);         // Set control field's disable file write (bit 2)
            header->setReadConfigEachCycleBit(0);      // Set control field's set to read configuration file each cycle bit (bit 3)
            //header->setTestMode(0);                    // Set control field's test mode (bits 4-7)
            header->setTestParameters(0);              // Set test parameters
            header->setNumberRows(0);                  // Set MCE header value (max 255) (defaluts to 33 if 0)
            header->setNumberRowsReported(0);          // Set MCE header value (defaults to numb rows if 0)
            header->setRowLength(0);                   // Set MCE header value
            header->setDataRate(0);                    // Set MCE header value

            for (i=0; i < 16; i++) 
               header->setTESBias(i, 0);               // Set TES DAC values 16X 20 bit

            // Get frame iterator
            ris::FrameIterator fPtr = frame->beginWrite();

            // Jump over the header
            fPtr += header->SmurfHeaderSize;

            // Create uint16 accessor to the data
            ris::FrameAccessor<uint16_t> dPtr(fPtr,4096);

            // Set data to zero
            memset(dPtr.begin(),0,4096*2);

            frame->setPayload(size);
            this->sendFrame(frame);
            frame.reset();

            ++frameCounter_;
            lastTime_ = currTime;
         }
         else usleep(10);
      }
      else usleep(1000);
   }
}

