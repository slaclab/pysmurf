# Test scripts

## Description

Theses are a series of scripts that can be use to test certain functionality of the pysmurf core.

## Test scripts

### validate_filter.py

This script validates the behavior of the SmurfProcessor's filter.

The script creates local root devices which contains an instance of the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py) device, connected to a [DataFromFile](../../python/pysmurf/core/emulators/_DataFromFile.py) data source, and it uses a [DataToFile](../../python/pysmurf/core/transmitters/_DataToFile.py) transmitter.

The scripts generates the filer parameters based on input arguments from the user (filter order and frequency) and set those parameters in the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py) device.

Then, the script generates random data, and send the data through the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py). The results are saved to disk. The same data is filtered using a `scipy.signal.butter` filter. Finally, both result are compared, calculating the RMSE between them. The test fails if the resulting RMSE is not `0`.

### validate_unwrapper.py

This script validates the behavior of the SmurfProcessor's unwrapper.

The script creates local root devices which contains an instance of the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py) device, connected to a [DataFromFile](../../python/pysmurf/core/emulators/_DataFromFile.py) data source, and it uses a [DataToFile](../../python/pysmurf/core/transmitters/_DataToFile.py) transmitter.

The scripts perform two tests:
#### Unwrapper enabled

The script generates a `int32` sawtooth signal. From this signal, it generates a wrapped version, as `int16`. Then it send this wrapped version trough the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py). The results are saved to disk.

Finally, the script compares the original unwrapped signal to the result at the output of the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py), calculating the RMSE between them. The test fails if the resulting RMSE is not `0`.

#### Unwrapper disabled

In this test, the script disables the `Unwrapper` from the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py).

Then, the script generates a wrapped version of a sawtooth signal, and sends it trough the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py). The results are saved to disk.

Finally, the script compares the original wrapped signal to the result at the output of the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py), calculating the RMSE between them. The test fails if the resulting RMSE is not `0`.

### profile_smurf_processor.py

This script can be used to profile different section of the SmurfProcessor. In order to get profile data, the SMuRF processor needs to be modified adding `TimerWithStats` object (available in [Timer.h](../../include/smurf/core/common/Timer.h)) in the appropriated places.

The script creates local root devices which contains an instance of the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py) device, connected to a [StreamDataSource](../../python/pysmurf/core/emulators/_StreamDataSource.py) data source, and a [FrameStatistics](../../python/pysmurf/core/counters/_FrameStatistics.py) device. The script them enabled the data source and let it sends `100000` frames trough the [SmurfProcessor](../../python/pysmurf/core/devices/_SmurfProcessor.py).

### validate_band_estimator.py

This scripts validates the behavior of the BandParameterEstimator.