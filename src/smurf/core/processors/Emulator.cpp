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
#include <rogue/interfaces/stream/Frame.h>
#include <rogue/interfaces/stream/FrameIterator.h>
#include "smurf/core/processors/Emulator.h"
#include "smurf/core/common/SmurfHeader.h"
#include <cmath>

namespace scp = smurf::core::processors;
namespace ris = rogue::interfaces::stream;

scp::Emulator::Emulator() {
   sinAmplitude_ = 0;
   sinBaseline_  = 0;
   sinPeriod_    = 0;
   sinChannel_   = 0;
   sinEnable_    = false;
   sinCount_     = 0;

   eLog_ = rogue::Logging::create("pysmurf.emulator");
}

scp::Emulator::~Emulator() { }

scp::EmulatorPtr scp::Emulator::create() {
    return std::make_shared<Emulator>();
}

void scp::Emulator::setup_python() {
   bp::class_< scp::Emulator,
               scp::EmulatorPtr,
               bp::bases<ris::Slave,ris::Master>,
               boost::noncopyable >
               ("Emulator",bp::init<>())

        // Sin Generate Parameters
        .def("setSinAmplitude",   &Emulator::setSinAmplitude)
        .def("getSinAmplitude",   &Emulator::getSinAmplitude)
        .def("setSinBaseline",    &Emulator::setSinBaseline)
        .def("getSinBaseline",    &Emulator::getSinBaseline)
        .def("setSinPeriod",      &Emulator::setSinPeriod)
        .def("getSinPeriod",      &Emulator::getSinPeriod)
        .def("setSinChannel",     &Emulator::setSinChannel)
        .def("getSinChannel",     &Emulator::getSinChannel)
        .def("setSinEnable",      &Emulator::setSinEnable)
        .def("getSinEnable",      &Emulator::getSinEnable)
    ;
    bp::implicitly_convertible< scp::EmulatorPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< scp::EmulatorPtr, ris::MasterPtr >();
}

// Sin parameters
void scp::Emulator::setSinAmplitude(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinAmplitude_ = value;
}

uint16_t scp::Emulator::getSinAmplitude() {
   return sinAmplitude_;
}

void scp::Emulator::setSinBaseline(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinBaseline_ = value;
}

uint16_t scp::Emulator::getSinBaseline() {
   return sinBaseline_;
}

void scp::Emulator::setSinPeriod(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinPeriod_ = value;
}

uint16_t scp::Emulator::getSinPeriod() {
   return sinPeriod_;
}

void scp::Emulator::setSinChannel(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinChannel_ = value;
}

uint16_t scp::Emulator::getSinChannel() {
   return sinChannel_;
}

void scp::Emulator::setSinEnable(bool value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinEnable_ = value;
}

bool scp::Emulator::getSinEnable() {
   return sinEnable_;
}

void scp::Emulator::acceptFrame(ris::FramePtr frame) {

   {
      rogue::GilRelease noGil;
      ris::FrameLockPtr fLock = frame->lock();
      std::lock_guard<std::mutex> lock(mtx_);


      // Make sure the frame is a single buffer, copy if neccessary
      if ( ! this->ensureSingleBuffer(frame,true) ) {
         eLog_->error("Failed to copy frame to single buffer. Check downstream slave types, maybe add a FIFO?");
         return;
      }

      // Sine wave enabled
      if ( sinEnable_ ) genSinWave(frame);
   }

   // Send frame outside of lock
   this->sendFrame(frame);
}

// Generic sine wave generator
void scp::Emulator::genSinWave(ris::FramePtr &frame) {

   SmurfHeaderROPtr<ris::FrameIterator> header = SmurfHeaderRO<ris::FrameIterator>::create(frame);

   if ( sinChannel_ >= header->getNumberRows() ) {
      eLog_->error("Configured sinChannel exceeds number of rows defined in the header.");
      return;
   }

   if ( header->SmurfHeaderSize + (header->getNumberRows() * 2) != frame->getPayload() ) {
      eLog_->error("Received frame does not match expected size. Size=%i, header=%i, payload=%i",
                  frame->getPayload(), header->SmurfHeaderSize, header->getNumberRows()*2);
      return;
   }

   // Get frame iterator
   ris::FrameIterator fPtr = frame->beginRead();

   // Jump over the header
   fPtr += header->SmurfHeaderSize;

   // Create uint16 pointer to the data
   uint16_t * dPtr = (uint16_t *)fPtr.ptr();

   dPtr[sinChannel_] = int((float)sinBaseline_ + 
                       (float)sinAmplitude_ * sin((float)sinCount_/(float)sinPeriod_));

   if ( ++sinCount_ == sinPeriod_ ) sinCount_ = 0;
}

