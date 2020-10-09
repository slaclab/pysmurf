#include <smurf/core/engines/SysgenCryo.h>
#include <math.h>
#include <chrono>

namespace bp  = boost::python;
namespace sce = smurf::core::engines;

void sce::SysgenCryo::setOffset(uint32_t offset) {
    offset_ = offset;

    config0_address_ = offset_ + CONFIG0_BASE_ADDR_C;
    config1_address_ = offset_ + CONFIG1_BASE_ADDR_C;
    status0_address_ = offset_ + STATUS0_BASE_ADDR_C;
    status1_address_ = offset_ + STATUS1_BASE_ADDR_C;

    return;
}

void sce::SysgenCryo::setBand(uint32_t band) {
   band_ = band;
}


void sce::SysgenCryo::updateConfig0Shadow( bool read ) {
    uint32_t ret[NUM_CHANNELS_C];
    if (read)
    {
	readReg(config0_address_, 4*NUM_CHANNELS_C, ret);
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            config0_shadow[i].reg = ret[i];
        }
    } 
    else
    {
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            ret[i] = config0_shadow[i].reg;
        }
	writeReg(config0_address_, 4*NUM_CHANNELS_C, ret);
    } 
    return;
}

void sce::SysgenCryo::updateConfig1Shadow(bool read) {
    uint32_t ret[NUM_CHANNELS_C];
    if (read)
    {
	readReg(config1_address_, 4*NUM_CHANNELS_C, ret);
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            config1_shadow[i].reg = ret[i];
        }
    } 
    else
    {
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            ret[i] = config1_shadow[i].reg;
        }
	writeReg(config1_address_, 4*NUM_CHANNELS_C, ret);
    } 
    return;
}

void sce::SysgenCryo::updateStatus0Shadow(bool read) {
    uint32_t ret[NUM_CHANNELS_C];
    if (read)
    {
	readReg(status0_address_, 4*NUM_CHANNELS_C, ret);
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            status0_shadow[i].reg = ret[i];
        }
    } 
    else
    {
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            ret[i] = status0_shadow[i].reg;
        }
	writeReg(status0_address_, 4*NUM_CHANNELS_C, ret);
    } 
    return;
}

void sce::SysgenCryo::updateStatus1Shadow(bool read) {
    uint32_t ret[NUM_CHANNELS_C];
    if (read)
    {
	readReg(status1_address_, 4*NUM_CHANNELS_C, ret);
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            status1_shadow[i].reg = ret[i];
        }
    } 
    else
    {
        for(int i = 0; i < NUM_CHANNELS_C; i++)
        {
            ret[i] = status1_shadow[i].reg;
        }
	writeReg(status1_address_, 4*NUM_CHANNELS_C, ret);
    } 
    return;
}

void sce::SysgenCryo::readAll() {

    updateConfig0Shadow(true);
    updateConfig1Shadow(true);
    updateStatus0Shadow(true);
    updateStatus1Shadow(true);

    return;
}

bool sce::SysgenCryo::readReg(uint32_t address, size_t size, uint32_t *reg) {
    uint32_t id;

    this->clearError();
    id = this->reqTransaction(address, size, reg, rogue::interfaces::memory::Read);
    this->waitTransaction(id);

    if ( this->getError() != "" ) {
        printf("got error\n");
        return false;
    }
    return true;
}


bool sce::SysgenCryo::writeReg(uint32_t address, size_t size, uint32_t *reg) {
    uint32_t id;

    this->clearError();
    id = this->reqTransaction(address, size, reg, rogue::interfaces::memory::Write);
    this->waitTransaction(id);

    if ( this->getError() != "" ) {
        printf("got error\n");
        return false;
    }
    return true;
}

void sce::SysgenCryo::setFeedbackEnable(int channel, int enable, bool write) {
    config1_shadow[channel].feedbackEnable = enable;
    if (write)
    {
        writeReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }

    return;
}

int sce::SysgenCryo::getFeedbackEnable(int channel, bool read) { 
    if (read)
    {
        readReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }
    return config1_shadow[channel].feedbackEnable;
}

void sce::SysgenCryo::setFeedbackEnableArray(std::vector<int> array, bool write) {
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        setFeedbackEnable(channel, array[channel], false);
    } 
    if (write)
    { 
        updateConfig1Shadow( false );
    } 
    return;
}

void sce::SysgenCryo::setFeedbackEnableArrayPy( const bp::object& iterable, bool write ) {
    std::vector<int> array = py_list_to_std_vector<int>( iterable );
    setFeedbackEnableArray( array, write);
    return;
}

std::vector<int> sce::SysgenCryo::getFeedbackEnableArray( bool read ) {
    std::vector<int> array(512);
    if (read)
    {
        updateConfig1Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        array[channel] = getFeedbackEnable(channel, false);
    } 
    return array;
}

bp::list sce::SysgenCryo::getFeedbackEnableArrayPy( bool read ) {
    std::vector<int> array;
    bp::list list;
    array = getFeedbackEnableArray( read );
    list = std_vector_to_py_list<int>( array );
    return list;
}

void sce::SysgenCryo::setCenterFrequencyMHz(int channel, double frequency, bool write) {
    frequency = clamp(frequency, MIN_FREQUENCY_C, MAX_FREQUENCY_C);
    config1_shadow[channel].centerFrequency = (int) round( frequency * pow(2, 24) / FREQUENCY_SPAN_C * 1e6 );
    if (write)
    {
        writeReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }
    return;
}

double sce::SysgenCryo::getCenterFrequencyMHz(int channel, bool read) {
    if (read)
    {
        readReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }
    return ( (double) config1_shadow[channel].centerFrequency ) * pow(2, -24) * FREQUENCY_SPAN_C * 1e-6;
}

void sce::SysgenCryo::setCenterFrequencyArray(std::vector<double> array, bool write) {
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        setCenterFrequencyMHz(channel, array[channel], false);
    } 
    if (write)
    { 
        updateConfig1Shadow( false );
    } 
    return;
}

void sce::SysgenCryo::setCenterFrequencyArrayPy( const bp::object& iterable, bool write ) {
    std::vector<double> array = py_list_to_std_vector<double>( iterable );
    setCenterFrequencyArray( array, write);
    return;
}

std::vector<double> sce::SysgenCryo::getCenterFrequencyArray( bool read ) {
    std::vector<double> array(512);
    if (read)
    {
        updateConfig1Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        array[channel] = getCenterFrequencyMHz(channel, false);
    } 
    return array;
}

bp::list sce::SysgenCryo::getCenterFrequencyArrayPy( bool read ) {
    std::vector<double> array;
    bp::list list;
    array = getCenterFrequencyArray( read );
    list = std_vector_to_py_list<double>( array );
    return list;
}

void sce::SysgenCryo::setAmplitudeScale(int channel, int amplitude, bool write) {
    amplitude = clamp(amplitude, MIN_AMPLITUDE_C, MAX_AMPLITUDE_C);
    config1_shadow[channel].amplitudeScale = amplitude;
    if (write)
    {
        writeReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }
    return;
}

int sce::SysgenCryo::getAmplitudeScale(int channel, bool read) {
    if (read)
    {
        readReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }
    return config1_shadow[channel].amplitudeScale;
}

void sce::SysgenCryo::setAmplitudeScaleArray(std::vector<int> array, bool write) {
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        setAmplitudeScale(channel, array[channel], false);
    } 
    if (write)
    { 
        updateConfig1Shadow( false );
    } 
    return;
}

void sce::SysgenCryo::setAmplitudeScaleArrayPy( const bp::object& iterable, bool write ) {
    std::vector<int> array = py_list_to_std_vector<int>( iterable );
    setAmplitudeScaleArray(array, write);
    return;
}

std::vector<int> sce::SysgenCryo::getAmplitudeScaleArray( bool read ) {
    std::vector<int> array(512);
    if (read)
    {
        updateConfig1Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        array[channel] = getAmplitudeScale(channel, false);
    } 
    return array;
}

bp::list sce::SysgenCryo::getAmplitudeScaleArrayPy( bool read ) {
    std::vector<int> array;
    bp::list list;
    array = getAmplitudeScaleArray( read );
    list = std_vector_to_py_list<int>( array );
    return list;
}

int sce::SysgenCryo::getAmplitudeReadback(int channel, bool read) {
    if (read)
    {
        readReg(status0_address_ + 4*channel, 4, &(status0_shadow[channel].reg));
    }
    return status0_shadow[channel].amplitudeReadback;
}

void sce::SysgenCryo::setEtaPhaseDegree(int channel, double etaPhase, bool write) {
    etaPhase = clamp(etaPhase, MIN_ETA_PHASE_C, MAX_ETA_PHASE_C);
    config0_shadow[channel].etaPhase = (int) round( etaPhase * pow(2, 15) / 180.0 );
    if (write)
    {
        writeReg(config0_address_ + 4*channel, 4, &(config0_shadow[channel].reg));
    }
    return;
}

double sce::SysgenCryo::getEtaPhaseDegree(int channel, bool read) {
    if (read)
    {
        readReg(config0_address_ + 4*channel, 4, &(config0_shadow[channel].reg));
    }
    return ( (double) config0_shadow[channel].etaPhase ) * pow(2, -15) * 180.0;
}

void sce::SysgenCryo::setEtaPhaseArray(std::vector<double> etaPhase, bool write) {
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        setEtaPhaseDegree(channel, etaPhase[channel], false);
    } 
    if (write)
    { 
        updateConfig0Shadow( false );
    } 
    return;
}

void sce::SysgenCryo::setEtaPhaseArrayPy( const bp::object& iterable, bool write ) {
    std::vector<double> etaPhaseArray= py_list_to_std_vector<double>( iterable );
    setEtaPhaseArray( etaPhaseArray, write);
    return;
}

std::vector<double> sce::SysgenCryo::getEtaPhaseArray( bool read ) {
    std::vector<double> etaPhaseArray(512);
    if (read)
    {
        updateConfig0Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        etaPhaseArray[channel] = getEtaPhaseDegree(channel, false);
    } 
    return etaPhaseArray;
}

bp::list sce::SysgenCryo::getEtaPhaseArrayPy( bool read ) {
    std::vector<double> etaPhaseArray;
    bp::list list;
    etaPhaseArray = getEtaPhaseArray( read );
    list = std_vector_to_py_list<double>( etaPhaseArray );
    return list;
}

void sce::SysgenCryo::setEtaMagScaled(int channel, double etaMag, bool write) {
    etaMag = clamp(etaMag, MIN_ETA_MAG_C, MAX_ETA_MAG_C);
    config0_shadow[channel].etaMag = (int) round( etaMag * pow(2, 10) );
    if (write)
    {
        writeReg(config0_address_ + 4*channel, 4, &(config0_shadow[channel].reg));
    }
    return;
}

double sce::SysgenCryo::getEtaMagScaled(int channel, bool read) {
    if (read)
    {
        readReg(config0_address_ + 4*channel, 4, &(config0_shadow[channel].reg));
    }
    return ( (double) config0_shadow[channel].etaMag ) * pow(2, -10);
}

void sce::SysgenCryo::setEtaMagArray(std::vector<double> etaMag, bool write) {
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        setEtaMagScaled(channel, etaMag[channel], false);
    } 
    if (write)
    {
        updateConfig0Shadow( false );
    }
    return;
}

void sce::SysgenCryo::setEtaMagArrayPy( const bp::object& iterable, bool write ) {
    std::vector<double> etaMagArray = py_list_to_std_vector<double>( iterable );
    setEtaMagArray( etaMagArray, write);
    return;
}

std::vector<double> sce::SysgenCryo::getEtaMagArray( bool read ) {
    std::vector<double> etaMagArray(512);
    if (read)
    {
        updateConfig0Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        etaMagArray[channel] = getEtaMagScaled(channel, false);
    } 
    return etaMagArray;
}

bp::list sce::SysgenCryo::getEtaMagArrayPy( bool read ) {
    std::vector<double> array;
    bp::list list;
    array = getEtaMagArray( read );
    list = std_vector_to_py_list<double>( array );
    return list;
}

void sce::SysgenCryo::regReadTest(int nloops)
{
    printf("Reg read test\n");
    auto start = std::chrono::high_resolution_clock::now();
    int channel = 0;
    for(int i = 0; i < nloops; i++)
    {
        updateConfig0Shadow(true);
       // readReg(config1_address_ + 4*channel, 4, &(config1_shadow[channel].reg));
    }
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    double hz = ((double)nloops)/(((double)duration.count())*1e-6);
    printf("Nloops: %d\n", nloops);
    printf("  Duration: %f\n", (double)duration.count());
    printf("  Hz: %f\n", hz);

}

double sce::SysgenCryo::getLoopFilterOutput(int channel, bool read) {
    if (read)
    {
        readReg(status0_address_ + 4*channel, 4, &(status0_shadow[channel].reg));
    }
    return status0_shadow[channel].loopFilterOutput;
}

std::vector<double> sce::SysgenCryo::getLoopFilterOutputArray( bool read ) {
    std::vector<double> array(512);
    if (read)
    {
        updateStatus0Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        array[channel] = sce::SysgenCryo::getLoopFilterOutput(channel, false);
    } 
    return array;
}

bp::list sce::SysgenCryo::getLoopFilterOutputArrayPy( bool read ) {
    std::vector<double> array;
    bp::list list;
    array = getLoopFilterOutputArray( read );
    list = std_vector_to_py_list<double>( array );
    return list;
}

double sce::SysgenCryo::getFrequencyErrorMHz(int channel, bool read) {
    if (read)
    {
        readReg(status1_address_ + 4*channel, 4, &(status1_shadow[channel].reg));
    }
    return ( (double) status1_shadow[channel].frequencyError ) * pow(2, -23) * FREQUENCY_SPAN_C * 1e-6;
}

std::vector<double> sce::SysgenCryo::getFrequencyErrorArray( bool read ) {
    std::vector<double> array(512);
    if (read)
    {
        updateStatus1Shadow( true );
    }
    for (int channel = 0; channel < NUM_CHANNELS_C; channel++)
    {
        array[channel] = getFrequencyErrorMHz(channel, false);
    } 
    return array;
}

bp::list sce::SysgenCryo::getFrequencyErrorArrayPy( bool read ) {
    std::vector<double> array;
    bp::list list;
    array = getFrequencyErrorArray( read );
    list = std_vector_to_py_list<double>( array );
    return list;
}

















std::vector<double> sce::SysgenCryo::getResultsReal() {
    return results_real_;
}

bp::list sce::SysgenCryo::getResultsRealPy() {
    std::vector<double> array;
    bp::list list;
    array = getResultsReal();
    list = std_vector_to_py_list<double>( array );
    return list;
}

std::vector<double> sce::SysgenCryo::getResultsImag() {
    return results_imag_;
}

bp::list sce::SysgenCryo::getResultsImagPy() {
    std::vector<double> array;
    bp::list list;
    array = getResultsImag();
    list = std_vector_to_py_list<double>( array );
    return list;
}

void sce::SysgenCryo::runEtaScan(std::vector<int> channels, int amplitude, std::vector<double> frequencies) {
    int num_channels      = channels.size();
    int freqs_per_channel = frequencies.size();
    int total_points      = num_channels*freqs_per_channel;
    std::vector<double> resultsReal(total_points);
    std::vector<double> resultsImag(total_points);

    // turn off all channels
    for (int i = 0; i < num_channels; ++i)
    { 
        setAmplitudeScale(channels[i], 0, true);
    } 

    for (int i = 0; i < num_channels; ++i)
    {
        int channel = channels[i];
	int offset  = i*freqs_per_channel;
	double centerFrequency = getCenterFrequencyMHz(channel, false);

        setAmplitudeScale(channel, amplitude, true);
        setFeedbackEnable(channel, 0, true);
        setEtaMagScaled(channel, 1.0, true);
        setEtaPhaseDegree(channel, 0.0, true);
        setCenterFrequencyMHz(channel, frequencies[0], true);
        usleep(100);
        for(int i = 0; i < frequencies.size(); i++)
        { 
            setCenterFrequencyMHz(channel, centerFrequency + frequencies[i], true);
            readReg(status1_address_ + 4*channel, 4, &(status1_shadow[channel].reg));
            resultsReal[offset + i] = status1_shadow[channel].frequencyError;
        } 

        setEtaPhaseDegree(channel, -90.0, true);
        setCenterFrequencyMHz(channel, frequencies[0], true);
        usleep(100);
        for(int i = 0; i < frequencies.size(); i++)
        { 
            setCenterFrequencyMHz(channel, centerFrequency + frequencies[i], true);
            readReg(status1_address_ + 4*channel, 4, &(status1_shadow[channel].reg));
            resultsImag[offset + i] = status1_shadow[channel].frequencyError;
        } 
	
        setCenterFrequencyMHz(channel, centerFrequency, true);
        setAmplitudeScale(channel, 0, true);
    }
    results_real_ = resultsReal;
    results_imag_ = resultsImag;
    return;
}

void sce::SysgenCryo::runEtaScanPy(const bp::object &channelspy, int amplitude, const bp::object &iterable) {
    std::vector<double> frequencies = py_list_to_std_vector<double>( iterable );
    std::vector<int> channels = py_list_to_std_vector<int>( channelspy );
    runEtaScan(channels, amplitude, frequencies);
    return;
}

