#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : SmurfProcessor's Unwrapper Validation Script
#-----------------------------------------------------------------------------
# File       : validate_unwrapper.py
# Created    : 2017-06-20
#-----------------------------------------------------------------------------
# Description:
# Script to validate the behavior of the SmurfProcessor's unwrapper
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software platform. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the rogue software platform, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------

import scipy.signal as signal
import numpy as np
import argparse

import pyrogue
import pysmurf
import pysmurf.core.devices
import pysmurf.core.transmitters

# Input arguments
parser = argparse.ArgumentParser(description='Test the SmurfProcessor Unwrapper.')
# Number of generated points
parser.add_argument('--input_size',
        type=int,
        default=1000,
        help='Number of point to generate')

# Output directory
parser.add_argument('--out_dir',
        type=str,
        default='/tmp',
        help='Directory to write the output data')


class LocalRoot(pyrogue.Root):
    """
    Local root device. It contains the SmurfProcessor, connected to
    a DataFromFile data source, and using the DataToFile transmitter.
    It will generate frame with the data from an input text file,
    send those frames through the SmurfProcessor, and write the results
    to an output text file.
    """
    def __init__(self, **kwargs):
        pyrogue.Root.__init__(self, name="AMCc", initRead=True, pollEn=True, **kwargs)

        # Use the DataFromFile as a stream data source
        self._streaming_stream =  pysmurf.core.emulators.DataFromFile()
        self.add(self._streaming_stream)

        # Add the SmurfProcessor, using the DataToFile transmitter
        # to write the results to a text file
        self._smurf_processor = pysmurf.core.devices.SmurfProcessor(
            name="SmurfProcessor",
            description="Process the SMuRF Streaming Data Stream",
            root=self,
            txDevice=pysmurf.core.transmitters.DataToFile())
        self.add(self._smurf_processor)

        # Connect the DataFromFile data source to the SmurfProcessor
        pyrogue.streamConnect(self._streaming_stream, self._smurf_processor)

# Main body
if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()
    input_size = args.input_size
    input_data_file = f'{args.out_dir}/x.dat'
    wrapped_data_file = f'{args.out_dir}/w.dat'
    smurf_unwrapped_file = f'{args.out_dir}/y.dat'
    smurf_wrapped_file = f'{args.out_dir}/y_w.dat'

    # Generate a sawtooth signal, as int32
    print(f'Generation sawtooth signal, {input_size} points... ', end='')
    x1 = 2**17 * signal.sawtooth(2 * np.pi * 2 * np.linspace(0, 1, input_size), 0.5)
    x = x1.astype('int32')
    print('Done')

    # Generate a wrapped version of the data, as int16
    print('Wrapping data...', end='')
    w = x.astype('int16')
    print('Done')

    # Save the input data to disk
    print(f'Writing sawtooth generated data to "{input_data_file}"... ', end='')
    np.savetxt(input_data_file, x, fmt='%i')
    print('Done')

    # Save the wrapped data to disk
    print(f'Writing wrapped data to "{wrapped_data_file}"... ', end='')
    np.savetxt(wrapped_data_file, w, fmt='%i')
    print('Done')

    # Send the input data through the SmurfProcessor, disabling the filter and downsampler,
    # leaving the unwrapper enabled.
    print('\nFirst test: Unwrapper enabled\n')
    print('Starting the SmurfProcessor, and unwrap the same data with it... ')
    with LocalRoot() as root:
        # Disable the filter
        print('  Disabling data filter... ', end='')
        root.SmurfProcessor.Filter.Disable.set(True)
        print('Done')

        # Disable the downsampler
        print('  Disabling data downsampling... ', end='')
        root.SmurfProcessor.Downsampler.Disable.set(True)
        print('Done')

        # Set the input data file
        print(f'  Setting input data file to "{wrapped_data_file}"... ', end='')
        root.DataFromFile.FileName.set(wrapped_data_file)
        print('Done.')

        # Set the output data file
        print(f'  Setting output data file to "{smurf_unwrapped_file}"... ', end='')
        root.SmurfProcessor.DataToFile.FileName.set(smurf_unwrapped_file)
        print('Done.')

        # Start sending the data trough the processor
        print('  Sending data through the SmurfProcessor... ', end='')
        root.DataFromFile.SendData.call()
        print('Done.')

        print(f'    Number of frame sent = {root.DataFromFile.FrameCnt.get()}')

        # Write the results
        print('  Writing results... ', end='')
        root.SmurfProcessor.DataToFile.WriteData.call()
        print('Done.')

    # Load the results obtained using the smurf processor
    print('Reading results... ', end='')
    y_smurf = np.loadtxt(smurf_unwrapped_file, dtype='int32')
    print('Done.')

    # Calculate the RMSE between the original data and the data unwrapped by SMuRF

    # The output of the unwrapper will start at zero, while the sawtooth signal will start at -MAX.
    # This will create an offset between the two signals. So, we need to remove it before calculating
    # the RMSE. So, let's add MAX (which will be the first point) to the original data
    x_no_offset = x - x[0]
    rmse = np.sqrt(np.square(np.subtract(x_no_offset,y_smurf).astype('int64')).mean())
    print(f'RMSE = {rmse}')

    # Verify that the 2 results are identical
    if rmse != 0:
        raise AssertionError(f'RMSE value {rmse} is not zero')

    print('Test passed!')

    # Now let's make sure that with the unwrapped disabled, we get the same dwrapped data
    # at its output.
    print('\nSecond test: Unwrapper disabled\n')
    print('Starting the SmurfProcessor, and send the data through it... ')
    with LocalRoot() as root:
        # Disable the unwrapper
        print('  Disabling data unwrapper... ', end='')
        root.SmurfProcessor.Unwrapper.Disable.set(True)
        print('Done')

        # Disable the filter
        print('  Disabling data filter... ', end='')
        root.SmurfProcessor.Filter.Disable.set(True)
        print('Done')

        # Disable the downsampler
        print('  Disabling data downsampling... ', end='')
        root.SmurfProcessor.Downsampler.Disable.set(True)
        print('Done')

        # Set the input data file
        print(f'  Setting input data file to "{wrapped_data_file}"... ', end='')
        root.DataFromFile.FileName.set(wrapped_data_file)
        print('Done.')

        # Set the output data file
        print(f'  Setting output data file to "{smurf_wrapped_file}"... ', end='')
        root.SmurfProcessor.DataToFile.FileName.set(smurf_wrapped_file)
        print('Done.')

        # Start sending the data trough the processor
        print('  Sending data through the SmurfProcessor... ', end='')
        root.DataFromFile.SendData.call()
        print('Done.')

        print(f'    Number of frame sent = {root.DataFromFile.FrameCnt.get()}')

        # Write the results
        print('  Writing results... ', end='')
        root.SmurfProcessor.DataToFile.WriteData.call()
        print('Done.')

    # Load the results obtained using the smurf processor
    print('Reading results... ', end='')
    y_w_smurf = np.loadtxt(smurf_wrapped_file, dtype='int32')
    print('Done.')

    # Calculate the RMSE between the input and the output data

    rmse = np.sqrt(np.square(np.subtract(w,y_w_smurf).astype('int64')).mean())
    print(f'RMSE = {rmse}')

    # Verify that the 2 results are identical
    if rmse != 0:
        raise AssertionError(f'RMSE value {rmse} is not zero')

    print('Test passed!')

    print('\nAll SmurfProcessor unwrapper tests passed!')
