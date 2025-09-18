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

import argparse
import time

import pyrogue
import pysmurf
import pysmurf.core.devices
import pysmurf.core.transmitters

# Input arguments
parser = argparse.ArgumentParser(description='Test the SmurfProcessor Base Transmitter.')


# Stream data time interval
parser.add_argument('--stream-time',
    dest='stream_time',
    type=float,
    default=30.0,
    help='Time interval to stream data (in seconds). Default = 30.')

# Stream data rate
parser.add_argument('--stream-rate',
    dest='stream_rate',
    type=float,
    default=500.0,
    help='Stream data rate (in Hz). Default = 500.')

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

        # Add streamer
        self.stream = pyrogue.interfaces.stream.Variable(root=self, incGroups='stream')

        # Add a variable to produce metadata
        self.add(pyrogue.LocalVariable(name="testMeta", value=1, groups="stream"))

        # Use the DataFromFile as a stream data source
        self._streaming_stream =  pysmurf.core.emulators.StreamDataSource()
        self.add(self._streaming_stream)

        # Add the SmurfProcessor, using the DataToFile transmitter
        # to write the results to a text file
        self._smurf_processor = pysmurf.core.devices.SmurfProcessor(
            name="SmurfProcessor",
            description="SMuRF Processor",
            root=self,
            txDevice=pysmurf.core.transmitters.BaseTransmitter(name='Transmitter'))
        self.add(self._smurf_processor)

        # Connect the DataFromFile data source to the SmurfProcessor
        pyrogue.streamConnect(self._streaming_stream, self._smurf_processor)

# Main body
if __name__ == "__main__":

    # Parse input arguments
    args = parser.parse_args()
    stream_time = args.stream_time
    stream_rate = args.stream_rate

    # Send data trough a SmurfProcessor, with the base transmitter attached to it
    # and writing the data to an output file. We disable the downsample in order to send
    # all the samples to the transmitter and output file.
    print('Starting the SmurfProcessor device and sending data through it... ')
    with LocalRoot() as root:
        # Disable the downsampler
        print('  Disabling data downsampling... ', end='')
        root.SmurfProcessor.Downsampler.Disable.set(True)
        print('Done')

        # Open the output data file
        print('  Opening the FileWriter output file... ', end='')
        root.SmurfProcessor.FileWriter.DataFile.set('/dev/null')
        root.SmurfProcessor.FileWriter.Open()
        print('Done')

        # Send data for the specified amount of time
        print(f'  Streaming data for {stream_time} seconds, at {stream_rate} Hz... ', end='')
        root.StreamDataSource.SourceEnable.set(True)
        root.StreamDataSource.Period.set(1/stream_rate)
        time.sleep(stream_time)
        root.StreamDataSource.SourceEnable.set(False)
        print('Done')

        # Close the output file
        print('  Closing the FileWriter output file... ', end='')
        root.SmurfProcessor.FileWriter.Close()
        print('Done')

        # Delay to make sure all counters are up-to-date
        time.sleep(2)

        # Read all the frame counters
        print('  Reading counters... ', end='')
        # FrameRxStats
        rx_cnt = root.SmurfProcessor.FrameRxStats.FrameCnt.get()
        rx_ooo_cnt = root.SmurfProcessor.FrameRxStats.FrameOutOrderCnt.get()
        rx_loss_cnt = root.SmurfProcessor.FrameRxStats.FrameLossCnt.get()
        rx_bad_cnt = root.SmurfProcessor.FrameRxStats.BadFrameCnt.get()

        # Fifos
        data_fifo_drop_cnt = root.SmurfProcessor.DataFifo.FrameDropCnt.get()
        meta_fifo_drop_cnt = root.SmurfProcessor.MetaFifo.FrameDropCnt.get()

        # FileWriter
        file_cnt = root.SmurfProcessor.FileWriter.FrameCount.get()

        # Transmitter
        tx_data_cnt = root.SmurfProcessor.Transmitter.dataFrameCnt.get()
        tx_meta_cnt = root.SmurfProcessor.Transmitter.metaFrameCnt.get()
        tx_data_drop_cnt = root.SmurfProcessor.Transmitter.dataDropCnt.get()
        tx_meta_drop_cnt = root.SmurfProcessor.Transmitter.metaDropCnt.get()
        print('Done')

    print('Results:')
    print('  FrameRxStats:')
    print(f'    Rx frames           = {rx_cnt}')
    print(f'    Out-of-order frames = {rx_ooo_cnt}')
    print(f'    Loss frames         = {rx_loss_cnt}')
    print(f'    Bad frames          = {rx_bad_cnt}')
    print('  DataFifo:')
    print(f'    Drop frames         = {data_fifo_drop_cnt}')
    print('  MetaFifo:')
    print(f'    Drop frames         = {meta_fifo_drop_cnt}')
    print('  FileWriter:')
    print(f'    Written frames      = {file_cnt}')
    print('  Transmitter:')
    print(f'    Data rx frames      = {tx_data_cnt}')
    print(f'    Data drop frames    = {tx_data_drop_cnt}')
    print(f'    Meta rx frames      = {tx_meta_cnt}')
    print(f'    Meta drop frames    = {tx_meta_drop_cnt}')

    # Validate results
    ## There should not be drop frames in the data fifo
    if data_fifo_drop_cnt != 0:
        raise AssertionError(f'Data frames dropped in the fifo: {data_fifo_drop_cnt}')

    ## There should not be drop frames in the metadata fifo
    if meta_fifo_drop_cnt != 0:
        raise AssertionError(f'Metadata frames dropped in the fifo: {meta_fifo_drop_cnt}')

    ## There should not be drop data frames in the transmitter
    if tx_meta_drop_cnt != 0:
        raise AssertionError(f'Data frames dropped in the transmitter: {tx_data_drop_cnt}')

    ## There should not be drop meta-data frames in the transmitter
    if tx_meta_drop_cnt != 0:
        raise AssertionError(f'Meatadata frames dropped in the transmitter: {tx_meta_drop_cnt}')

    ## All Rx frames should have been sent to the transmitter
    if rx_cnt != tx_data_cnt:
        raise AssertionError(f'Missing RX frames in the transmitter: {rx_cnt - tx_data_cnt}')

    ## The transmitter should have received some metadata frames
    if tx_meta_cnt == 0:
        raise AssertionError('No metadata frames were received in the transmitter')

    ## The FileWriter should have received more than the received frames,
    ## as it writes also metadata fames
    if file_cnt < rx_cnt:
        raise AssertionError('Missing RX frames in the FileWritter')

    ## All test passed
    print('SmurfProcessor base transmitter test passed!')
