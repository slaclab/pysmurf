/**
 *-----------------------------------------------------------------------------
 * Title         : SMuRF Data Emulator
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
#include <rogue/interfaces/stream/FrameAccessor.h>
#include "smurf/core/emulators/StreamDataEmulator.h"
#include "smurf/core/common/SmurfHeader.h"
#include <cmath>

namespace sce = smurf::core::emulators;
namespace ris = rogue::interfaces::stream;

sce::StreamDataEmulator::StreamDataEmulator() {
   sinAmplitude_ = 0;
   sinBaseline_  = 0;
   sinPeriod_    = 0;
   sinChannel_   = 0;
   sinEnable_    = false;
   sinCount_     = 0;

   eLog_ = rogue::Logging::create("pysmurf.emulator");
}

sce::StreamDataEmulator::~StreamDataEmulator() { }

sce::StreamDataEmulatorPtr sce::StreamDataEmulator::create() {
    return std::make_shared<StreamDataEmulator>();
}

void sce::StreamDataEmulator::setup_python() {
   bp::class_< sce::StreamDataEmulator,
               sce::StreamDataEmulatorPtr,
               bp::bases<ris::Slave,ris::Master>,
               boost::noncopyable >
               ("StreamDataEmulator",bp::init<>())

        // Sin Generate Parameters
        .def("setSinAmplitude",   &StreamDataEmulator::setSinAmplitude)
        .def("getSinAmplitude",   &StreamDataEmulator::getSinAmplitude)
        .def("setSinBaseline",    &StreamDataEmulator::setSinBaseline)
        .def("getSinBaseline",    &StreamDataEmulator::getSinBaseline)
        .def("setSinPeriod",      &StreamDataEmulator::setSinPeriod)
        .def("getSinPeriod",      &StreamDataEmulator::getSinPeriod)
        .def("setSinChannel",     &StreamDataEmulator::setSinChannel)
        .def("getSinChannel",     &StreamDataEmulator::getSinChannel)
        .def("setSinEnable",      &StreamDataEmulator::setSinEnable)
        .def("getSinEnable",      &StreamDataEmulator::getSinEnable)
    ;
    bp::implicitly_convertible< sce::StreamDataEmulatorPtr, ris::SlavePtr  >();
    bp::implicitly_convertible< sce::StreamDataEmulatorPtr, ris::MasterPtr >();
}

// Sin parameters
void sce::StreamDataEmulator::setSinAmplitude(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinAmplitude_ = value;
}

uint16_t sce::StreamDataEmulator::getSinAmplitude() {
   return sinAmplitude_;
}

void sce::StreamDataEmulator::setSinBaseline(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinBaseline_ = value;
}

uint16_t sce::StreamDataEmulator::getSinBaseline() {
   return sinBaseline_;
}

void sce::StreamDataEmulator::setSinPeriod(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinPeriod_ = value;
}

uint16_t sce::StreamDataEmulator::getSinPeriod() {
   return sinPeriod_;
}

void sce::StreamDataEmulator::setSinChannel(uint16_t value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinChannel_ = value;
}

uint16_t sce::StreamDataEmulator::getSinChannel() {
   return sinChannel_;
}

void sce::StreamDataEmulator::setSinEnable(bool value) {
   std::lock_guard<std::mutex> lock(mtx_);
   sinEnable_ = value;
}

bool sce::StreamDataEmulator::getSinEnable() {
   return sinEnable_;
}

void sce::StreamDataEmulator::acceptFrame(ris::FramePtr frame) {

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
void sce::StreamDataEmulator::genSinWave(ris::FramePtr &frame) {

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

   // Create uint16 accessor to the data
   ris::FrameAccessor<uint16_t> dPtr(fPtr,header->getNumberRows());

   dPtr[sinChannel_] = int((float)sinBaseline_ + 
                       (float)sinAmplitude_ * sin((float)sinCount_/(float)sinPeriod_));

   if ( ++sinCount_ == sinPeriod_ ) sinCount_ = 0;
}

