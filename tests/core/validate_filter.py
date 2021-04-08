#!/usr/bin/env python3
#-----------------------------------------------------------------------------
# Title      : SmurfProcessor's Filter Validation Script
#-----------------------------------------------------------------------------
# File       : validate_filter.py
# Created    : 2017-06-20
#-----------------------------------------------------------------------------
# Description:
# Script to validate the behavior of the SmurfProcessor's filter
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
parser = argparse.ArgumentParser(description='Test the SmurfProcessor Filter.')
# Filter order
parser.add_argument('--filter_order',
        type=int,
        default=4,
        help='Filter order')

# Filter frequency
parser.add_argument('--filter_freq',
        type=int,
        default=2*63/4000,
        help='Filter order')

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
    filter_order = args.filter_order
    filter_freq = args.filter_freq
    input_size = args.input_size
    input_data_file = f'{args.out_dir}/x.dat'
    python_filtered_file = f'{args.out_dir}/y_python.dat'
    smurf_filtered_file = f'{args.out_dir}/y_smurf.dat'

    # Generate filter coefficients
    print(f'Generating filer parameters for freq {filter_freq}, order {filter_order}... ', end='')
    b,a = signal.butter(filter_order, filter_freq)
    print('Done!')
    print('Filter coefficients:')
    print(f'  a = {a}')
    print(f'  b = {b}')

    # Generate random data, as int16
    print(f'Generation random number, {input_size} points... ', end='')
    x1 =  np.random.randn(input_size)
    x2 = x1 / np.abs(x1).max() * 2**15
    x = x2.astype('int16')
    print('Done')

    # Filter the data
    print('Filtering data...', end='')
    y1, _ = signal.lfilter(b, a, x, zi=signal.lfilter_zi(b, a)*[0])
    y = y1.astype('int32')
    print('Done')

    # Save the input data to disk
    print(f'Writing random generated data to "{input_data_file}"... ', end='')
    np.savetxt(input_data_file, x, fmt='%i')
    print('Done')

    # Save the output data to disk
    print(f'Writing filtered data to "{python_filtered_file}"... ', end='')
    np.savetxt(python_filtered_file, y, fmt='%i')
    print('Done')

    # Send the input data through the SmurfProcessor, disabling the unwrapper and
    # downsampling, and setting the filer with the generated coefficients.
    print('Starting the SmurfProcessor, and filter the same data with it')
    with LocalRoot() as root:
        # Disable the unwrapper
        print('  Disabling data unwrapping... ', end='')
        root.SmurfProcessor.Unwrapper.Disable.set(True)
        print('Done')

        # Disable the downsampler
        print('  Disabling data downsampling... ', end='')
        root.SmurfProcessor.Downsampler.Disable.set(True)
        print('Done')

        # Set the filter parameters according to our simulation
        print('  Setting filter parameters... ', end='')
        root.SmurfProcessor.Filter.A.set(a.tolist())
        root.SmurfProcessor.Filter.B.set(b.tolist())
        root.SmurfProcessor.Filter.Order.set(filter_order)
        print('Done')

        # Print current filter settings
        print('  Filter set to:')
        print(f'    order = {root.SmurfProcessor.Filter.Order.get()}')
        print(f'    A     = {root.SmurfProcessor.Filter.A.get()}')
        print(f'    B     = {root.SmurfProcessor.Filter.B.get()}')

        # Set the input data file
        print(f'  Setting input data file to "{input_data_file}"... ', end='')
        root.DataFromFile.FileName.set(input_data_file)
        print('Done.')

        # Set the output data file
        print(f'  Setting output data file to "{smurf_filtered_file}"... ', end='')
        root.SmurfProcessor.DataToFile.FileName.set(smurf_filtered_file)
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
    y_smurf = np.loadtxt(smurf_filtered_file, dtype='int32')
    print('Done.')

    # Calculate the RMSE between the 2 filter's output
    rmse = np.sqrt(np.square(np.subtract(y,y_smurf)).mean())
    print(f'RMSE = {rmse}')

    # Verify that the 2 results are identical
    if rmse != 0:
        raise AssertionError(f'RMSE value {rmse} is not zero')


    print('SmurfProcessor filter test passed!')
