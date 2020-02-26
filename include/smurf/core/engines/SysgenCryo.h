#include <stdint.h>
#include <rogue/interfaces/memory/Constants.h>
#include <rogue/interfaces/memory/Master.h>
#include <boost/python.hpp>
#include <boost/python/module.hpp>
#include <boost/python/stl_iterator.hpp>

#ifndef _SYSGEN_CRYO_H_
#define _SYSGEN_CRYO_H_

namespace smurf {
    namespace core {
        namespace engines {

           typedef struct {
               union {
                   struct { 
                       int etaMag   : 16;
                       int etaPhase : 16;
                   };
                   uint32_t reg;
               };
           } config0;
        
           typedef struct {
               union {
                   struct {
                       int          centerFrequency : 24;
                       unsigned int amplitudeScale  : 4;
                       unsigned int pad             : 3;
                       unsigned int feedbackEnable  : 1;
                   };
                   uint32_t reg;
               };
           } config1;
        
           typedef struct {
               union {
                   struct {
                       int          loopFilterOutput  : 24;
                       unsigned int amplitudeReadback : 4;
                       unsigned int pad               : 4;
                   };
                   uint32_t reg;
               };
           } status0;
        
           typedef struct {
               union {
                   struct {
                       int          frequencyError : 24;
                       unsigned int pad            : 8;
                   };
                   uint32_t reg;
               };
           } status1;
        
        
           template<typename T>
           inline
           T clamp( T n, T lower, T upper )
           {
               return  (n <= lower ? lower : n <= upper ? n : upper);
           }
        
           template<typename T>
           inline
           std::vector< T > py_list_to_std_vector( const boost::python::object& iterable )
           {
               return std::vector< T >( boost::python::stl_input_iterator< T >( iterable ),
                                        boost::python::stl_input_iterator< T >( ) );
           }
        
        
           template <class T>
           inline
           boost::python::list std_vector_to_py_list(std::vector<T> vector) {
               typename std::vector<T>::iterator iter;
               boost::python::list list;
               for (iter = vector.begin(); iter != vector.end(); ++iter) {
                   list.append(*iter);
               }
               return list;
           }
        
           class SysgenCryo : public rogue::interfaces::memory::Master {
               private:
                   static constexpr int      NUM_CHANNELS_C      = 512;
                   static constexpr double   FREQUENCY_SPAN_C    = 9.6e6;
                   static constexpr double   MAX_FREQUENCY_C     = 4.8e6;
                   static constexpr double   MIN_FREQUENCY_C     = -4.8e6;
                   static constexpr int      MAX_AMPLITUDE_C     = 15;
                   static constexpr int      MIN_AMPLITUDE_C     = 0;
                   static constexpr double   MAX_ETA_PHASE_C     = 180.0;
                   static constexpr double   MIN_ETA_PHASE_C     = -180.0;
                   static constexpr double   MAX_ETA_MAG_C       = 100.0;
                   static constexpr double   MIN_ETA_MAG_C       = 0.0;
        
                   static constexpr uint32_t CONFIG0_BASE_ADDR_C = 0x0000;
                   static constexpr uint32_t CONFIG1_BASE_ADDR_C = 0x0800;
                   static constexpr uint32_t STATUS0_BASE_ADDR_C = 0x1000;
                   static constexpr uint32_t STATUS1_BASE_ADDR_C = 0x1800;
        
                   uint32_t offset_;
        
                   uint32_t config0_address_;
                   uint32_t config1_address_;
                   uint32_t status0_address_;
                   uint32_t status1_address_;
             
                   uint32_t band_;
        
                   std::vector<double> results_real_;
                   std::vector<double> results_imag_;
        
                   config0 config0_shadow[NUM_CHANNELS_C]; 
                   config1 config1_shadow[NUM_CHANNELS_C]; 
                   status0 status0_shadow[NUM_CHANNELS_C]; 
                   status1 status1_shadow[NUM_CHANNELS_C]; 
        
               public:
        
                   SysgenCryo() : rogue::interfaces::memory::Master() {}
        
                   static std::shared_ptr<smurf::core::engines::SysgenCryo> create() {
                       static std::shared_ptr<smurf::core::engines::SysgenCryo> ret = std::make_shared<smurf::core::engines::SysgenCryo>();
                       return ret;
                   }
        
                   void   setOffset(uint32_t offset);
        
                   void   setBand(uint32_t offset);
        
                   void   updateConfig0Shadow(bool read);
                   void   updateConfig1Shadow(bool read);
                   void   updateStatus0Shadow(bool read);
                   void   updateStatus1Shadow(bool read);
                   void   readAll();
        
                   bool   readReg(uint32_t address, size_t size, uint32_t *reg);
                   bool   writeReg(uint32_t address, size_t size, uint32_t *reg);
        
                   // config 0
                   void   setFeedbackEnable(int channel, int enable, bool write);
                   int    getFeedbackEnable(int channel, bool read);
        
                   void   setFeedbackEnableArray(std::vector<int> array, bool write);
                   void   setFeedbackEnableArrayPy(const boost::python::object &iterable, bool write);
        
                   std::vector<int> getFeedbackEnableArray( bool read );
                   boost::python::list getFeedbackEnableArrayPy( bool read );
        
                   void   setCenterFrequencyMHz(int channel, double frequencyMHz, bool write);
                   double getCenterFrequencyMHz(int channel, bool read);
        
                   void   setCenterFrequencyArray(std::vector<double> array, bool write);
                   void   setCenterFrequencyArrayPy(const boost::python::object &iterable, bool write);
        
                   std::vector<double> getCenterFrequencyArray( bool read );
                   boost::python::list getCenterFrequencyArrayPy( bool read );
        
                   void   setAmplitudeScale(int channel, int amplitude, bool write);
                   int    getAmplitudeScale(int channel, bool read);
        
                   void   setAmplitudeScaleArray(std::vector<int> amplitudeScale, bool write);
                   void   setAmplitudeScaleArrayPy(const boost::python::object &iterable, bool write);
        
                   std::vector<int> getAmplitudeScaleArray( bool read );
                   boost::python::list getAmplitudeScaleArrayPy( bool read );
        
                   // config 1
                   void   setEtaPhaseDegree(int chanenl, double etaPhase, bool write);
                   double getEtaPhaseDegree(int channel, bool read);
        
                   void   setEtaPhaseArray(std::vector<double> etaPhase, bool write);
                   void   setEtaPhaseArrayPy(const boost::python::object &iterable, bool write);
        
                   std::vector<double> getEtaPhaseArray( bool read );
                   boost::python::list getEtaPhaseArrayPy( bool read );
        
                   void   setEtaMagScaled(int channel, double etaMag, bool write);
                   double getEtaMagScaled(int channel, bool read);
        
                   void   setEtaMagArray(std::vector<double> etaMag, bool write);
                   void   setEtaMagArrayPy(const boost::python::object &iterable, bool write);
        
                   std::vector<double> getEtaMagArray( bool read );
                   boost::python::list getEtaMagArrayPy( bool read );
        
                   // status
                   int    getAmplitudeReadback(int channel, bool read);
        
                   double getLoopFilterOutput(int channel, bool read);
        
                   std::vector<double> getLoopFilterOutputArray( bool read );
                   boost::python::list getLoopFilterOutputArrayPy( bool read );
        
                   double getFrequencyErrorMHz(int channel, bool read);
                   std::vector<double> getFrequencyErrorArray( bool read );
                   boost::python::list getFrequencyErrorArrayPy( bool read );
        
                   void regReadTest(int nloops);
        
                   std::vector<double> getResultsReal();
                   boost::python::list getResultsRealPy();
                   std::vector<double> getResultsImag();
                   boost::python::list getResultsImagPy();
        
                   void runEtaScan(std::vector<int> channels, int amplitude, std::vector<double> frequencies);
                   void runEtaScanPy(const boost::python::object &channelspy, int amplitude, const boost::python::object &iterable);
        
                   static void setup_python() {
                       boost::python::class_<SysgenCryo,    std::shared_ptr<SysgenCryo>,
                                                            boost::python::bases<rogue::interfaces::memory::Master>,
                                                            boost::noncopyable>("SysgenCryo",
                                                            boost::python::init<>())
                           .def("setOffset",                &SysgenCryo::setOffset)
                           .def("setBand",                  &SysgenCryo::setBand)
                           .def("setFeedbackEnable",        &SysgenCryo::setFeedbackEnable)
                           .def("getFeedbackEnable",        &SysgenCryo::getFeedbackEnable)
                           .def("setFeedbackEnableArray",   &SysgenCryo::setFeedbackEnableArrayPy)
                           .def("getFeedbackEnableArray",   &SysgenCryo::getFeedbackEnableArrayPy)
                           .def("setCenterFrequencyMHz",    &SysgenCryo::setCenterFrequencyMHz)
                           .def("getCenterFrequencyMHz",    &SysgenCryo::getCenterFrequencyMHz)
                           .def("setCenterFrequencyArray",  &SysgenCryo::setCenterFrequencyArrayPy)
                           .def("getCenterFrequencyArray",  &SysgenCryo::getCenterFrequencyArrayPy)
                           .def("setAmplitudeScale",        &SysgenCryo::setAmplitudeScale)
                           .def("getAmplitudeScale",        &SysgenCryo::getAmplitudeScale)
                           .def("setAmplitudeScaleArray",   &SysgenCryo::setAmplitudeScaleArrayPy)
                           .def("getAmplitudeScaleArray",   &SysgenCryo::getAmplitudeScaleArrayPy)
                           .def("setEtaPhaseDegree",        &SysgenCryo::setEtaPhaseDegree)
                           .def("getEtaPhaseDegree",        &SysgenCryo::getEtaPhaseDegree)
                           .def("setEtaPhaseArray",         &SysgenCryo::setEtaPhaseArrayPy)
                           .def("getEtaPhaseArray",         &SysgenCryo::getEtaPhaseArrayPy)
                           .def("setEtaMagScaled",          &SysgenCryo::setEtaMagScaled)
                           .def("getEtaMagScaled",          &SysgenCryo::getEtaMagScaled)
                           .def("setEtaMagArray",           &SysgenCryo::setEtaMagArrayPy)
                           .def("getEtaMagArray",           &SysgenCryo::getEtaMagArrayPy)
                           .def("getAmplitudeReadback",     &SysgenCryo::getAmplitudeReadback)
                           .def("getLoopFilterOutput",      &SysgenCryo::getLoopFilterOutput)
                           .def("getLoopFilterOutputArray", &SysgenCryo::getLoopFilterOutputArrayPy)
                           .def("getFrequencyErrorMHz",     &SysgenCryo::getFrequencyErrorMHz)
                           .def("getFrequencyErrorArray",   &SysgenCryo::getFrequencyErrorArrayPy)
                           .def("getResultsReal",           &SysgenCryo::getResultsRealPy)
                           .def("getResultsImag",           &SysgenCryo::getResultsImagPy)
                           .def("runEtaScan",               &SysgenCryo::runEtaScanPy)
                           .def("regReadTest",  &SysgenCryo::regReadTest)
                           ;
                       boost::python::implicitly_convertible<std::shared_ptr<SysgenCryo>, rogue::interfaces::memory::MasterPtr>();
                   }
           };

           typedef std::shared_ptr<smurf::core::engines::SysgenCryo> SysgenCryoPtr;
      }
   }
};

#endif

