import numpy as np
from pysmurf.base import SmurfBase
from pysmurf.command.sync_group import SyncGroup as SyncGroup
import time
import os
import struct
import time
import epics

class SmurfUtilMixin(SmurfBase):

    def take_debug_data(self, band, channel=None, single_channel_readout=1):
        """
        """
        # Set proper single channel readout
        if channel is not None:
            if single_channel_readout == 1:
                self.set_single_channel_readout(band, 1)
                self.set_single_channel_readout_opt2(band, 0)
            elif single_channel-readout == 2:
                self.set_single_channel_readout(band, 0)
                self.set_single_channel_readout_opt2(band, 1)
            else:
                self.log('single_channel_readout must be 1 or 2', 
                    self.LOG_ERROR)
                raise ValueError('single_channel_readout must be 1 or 2')


    def take_stream_data(self, band, meas_time):
        """
        Takes streaming data for a given amount of time
        Args:
        -----
        band (int) : The band to stream data
        meas_time (float) : The amount of time to observe for in seconds

        Returns:
        --------
        data_filename (string): The fullpath to where the data is stored
        """
        self.log('Staring to take data.', self.LOG_USER)
        data_filename = self.stream_data_on(band)
        time.sleep(meas_time)
        self.stream_data_off(band)
        self.log('Done taking data.', self.LOG_USER)
        return data_filename

    def stream_data_on(self, band):
        """
        Turns on streaming data on specified channel

        Args:
        -----
        band (int) : The band to stream data

        Returns:
        --------
        data_filename (string): The fullpath to where the data is stored
        """
        # Check if flux ramp is non-zero
        ramp_max_cnt = self.get_ramp_max_cnt()
        if ramp_max_cnt == 0:
            self.log('Flux ramp frequency is zero. Cannot take data.', 
                self.LOG_ERROR)
        else:
            if self.get_single_channel_readout(band) and \
                self.get_single_channel_readout_opt2(band):
                self.log('Streaming all channels on band {}'.format(band), 
                    self.LOG_USER)

            # Make the data file
            timestamp = '%10i' % time.time()
            data_filename = os.path.join(self.output_dir, timestamp+'.dat')
            self.log('Writing to file : {}'.format(data_filename), 
                self.LOG_USER)
            self.set_streaming_datafile(data_filename)
            self.set_streaming_file_open(1)  # Open the file

            self.set_stream_enable(band, 1, write_log=True)

            return data_filename

    def stream_data_off(self, band):
        """
        Turns off streaming data on specified band

        Args:
        -----
        band (int) : The band to turn off stream data
        """
        self.set_stream_enable(band, 0, write_log=True)
        self.set_streaming_file_open(0)  # Close the file


    def read_stream_data(self, datafile):
        """
        Loads data taken with the fucntion stream_data_on
        """

        file_writer_header_size = 2  # 32-bit words
        smurf_header_size = 4  # 32-bit words
        header_size = file_writer_header_size + smurf_header_size
        smurf_data_size = 1024;  # 32-bit words
        nominal_frame_size = header_size + smurf_data_size;

        with open(datafile, mode='rb') as file:
            file_content = file.read()

        # Convert binary file to int array. The < indicates little-endian
        raw_dat = np.asarray(struct.unpack("<" + "i" * ((len(file_content)) // 4), 
            file_content))

        # To do : add bad frame check
        frame_start = np.ravel(np.where(1 + raw_dat/4==nominal_frame_size))
        n_frame = len(frame_start)

        I = np.zeros((512, n_frame))
        Q = np.zeros((512, n_frame))
        timestamp = np.zeros(n_frame)

        for i in np.arange(n_frame):
            timestamp[i] = raw_dat[frame_start[i]+2]
            start = frame_start[i] + header_size
            end = start + 512*2
            I[:,i] = raw_dat[start:end:2]
            Q[:,i] = raw_dat[start+1:end+1:2]

        return timestamp, I, Q

    def read_stream_data_daq(self, data_length, bay=0, hw_trigger=False):
        """
        """
        # Ask mitch why this is what it is...
        if bay == 0:
            stream0 = self.epics_root + ":AMCc:Stream0"
            stream1 = self.epics_root + ":AMCc:Stream1"
        else:
            stream0 = self.epics_root + ":AMCc:Stream4"
            stream1 = self.epics_root + ":AMCc:Stream5"

        print('camonitoring')
        epics.camonitor(stream0)
        epics.camonitor(stream1)
        print('done camonitoring')
        
        pvs = [stream0, stream1]
        sg  = SyncGroup(pvs)

        # trigger PV
        if not hw_trigger:
            self.set_trigger_daq(1, write_log=True)
        else:
            self._caput(self.epics_root + 
                ':AMCc:FpgaTopLevel:AppTop:DaqMuxV2[0]:ArmHwTrigger', 1, 
                write_log=True)
        time.sleep(.1)
        sg.wait()

        vals = sg.get_values()
        print(vals[pvs[0]])

        r0 = vals[pvs[0]]
        r1 = vals[pvs[1]]
        
        return r0, r1

    def read_adc_data(self, adc_number, data_length, hw_trigger=False):
        """
        """
        if adc_number > 3:
            bay = 1
            adc_number = adc_number - 4
        else:
            bay = 0

        self.setup_daq_mux('adc', adc_number, data_length)

        res = self.read_stream_data_daq(data_length, bay=bay,
            hw_trigger=hw_trigger)
        dat = res[1] + 1.j * res[0]

        return dat

    def read_dac_data(self, dac_number, data_length, hw_trigger=False):
        """
        Read the data directly off the DAC
        """
        if dac_number > 3:
            bay = 1
            dac_number = dac_number - 4
        else:
            bay = 0

        self.setup_daq_mux('dac', dac_number, data_length)

        res = self.read_stream_data_daq(data_length, bay=bay, 
            hw_trigger=hw_trigger)
        dat = res[1] + 1.j * res[0]

        return dat

    def setup_daq_mux(self, converter, converter_number, data_length):
        """
        Sets up for either ADC or DAC data taking.

        Args:
        -----
        converter (str) : Whether it is the ADC or DAC. choices are 'adc' and 
            'dac'
        converter_number (int) : The ADC or DAC number to take data on.
        data_length (int) : The amount of data to take.
        """
        if converter.lower() == 'adc':
            daq_mux_channel0 = (converter_number + 1)*2
            daq_mux_channel1 = daq_mux_channel0 + 1
        elif converter.lower() == 'dac':
            daq_mux_channel0 = (converter_number + 1)*2 + 10
            daq_mux_channel1 = daq_mux_channel0 + 1

        # setup buffer size
        self.set_buffer_size(data_length)

        # input mux select
        self.set_input_mux_sel(0, daq_mux_channel0, write_log=True)
        self.set_input_mux_sel(1, daq_mux_channel1, write_log=True)


    def set_buffer_size(self, size):
        """
        Sets the buffer size for reading and writing DAQs

        Args:
        -----
        size (int) : The buffer size in number of points
        """
        # Change DAQ data buffer size

        # Change waveform engine buffer size
        self.set_data_buffer_size(size, write_log=True)
        for daq_num in np.arange(4):
            s = self.get_waveform_start_addr(daq_num, convert=True)
            e = s + 4*size
            self.set_waveform_end_addr(daq_num, e, convert=True)
            self.log('DAQ number {}: start {} - end {}'.format(daq_num, s, e))


    def which_on(self, band):
        '''
        Finds all detectors that are on.

        Args:
        -----
        band (int) : The band to search.

        Returns:
        --------
        channels_on (int array) : The channels that are on
        '''
        amps = self.get_amplitude_scale_array(band)
        return np.ravel(np.where(amps != 0))

    def band_off(self, band, **kwargs):
        '''
        Turns off all tones in a band
        '''
        self.set_amplitude_scales(band, 0, **kwargs)
        self.set_feedback_enable_array(band, np.zeros(512, dtype=int), **kwargs)
        self.set_cfg_reg_ena_bit(0, wait_after=.11, **kwargs)

    def channel_off(self, band, channel, **kwargs):
        """
        Turns off tones for a single channel
        """
        self.log('Turning off band {} channel {}'.format(band, channel), 
            self.LOG_USER)
        self.set_amplitude_scale_channel(band, channel, 0, **kwargs)
        self.set_feedback_enable_channel(band, channel, 0, **kwargs)

    def set_feedback_limit_khz(self, band, feedback_limit_khz):
        '''
        '''
        digitizer_freq_mhz = self.get_digitizer_frequency_mhz(band)
        bandcenter = self.get_band_center_mhz(band)
        n_subband = self.get_number_sub_bands(band)

        subband_bandwidth = 2 * digitizer_freq_mhz / n_subband
        desired_feedback_limit_mhz = feedback_limit_khz/1000.

        if desired_feedback_limit_mhz > subband_bandwidth/2:
            desired_feedback_limit_mhz = subband_bandwidth/2

        desired_feedback_limit_dec = np.floor(desired_feedback_limit_mhz/
            (subband_bandwidth/2.))


    def get_fpga_status(self):
        '''
        Loads FPGA status checks if JESD is ok.

        Returns:
        ret (dict) : A dictionary containing uptime, fpga_version, git_hash,
            build_stamp, jesd_tx_enable, and jesd_tx_valid
        '''
        uptime = self.get_fpga_uptime()
        fpga_version = self.get_fpga_version()
        git_hash = self.get_fpga_git_hash()
        build_stamp = self.get_fpga_build_stamp()

        git_hash = ''.join([chr(y) for y in git_hash]) # convert from int to ascii
        build_stamp = ''.join([chr(y) for y in build_stamp])

        self.log("Build stamp: " + str(build_stamp) + "\n", self.LOG_USER)
        self.log("FPGA version: Ox" + str(fpga_version) + "\n", self.LOG_USER)
        self.log("FPGA uptime: " + str(uptime) + "\n", self.LOG_USER)

        jesd_tx_enable = self.get_jesd_tx_enable()
        jesd_tx_valid = self.get_jesd_tx_data_valid()
        if jesd_tx_enable != jesd_tx_valid:
            self.log("JESD Tx DOWN", self.LOG_USER)
        else:
            self.log("JESD Tx Okay", self.LOG_USER)

        # dict containing all values
        ret = {
            'uptime' : uptime,
            'fpga_version' : fpga_version,
            'git_hash' : git_hash,
            'build_stamp' : build_stamp,
            'jesd_tx_enable' : jesd_tx_enable,
            'jesd_tx_valid' : jesd_tx_valid
        }

        return ret

    def freq_to_subband(self, freq, band_center, subband_order):
        '''Look up subband number of a channel frequency

        To do: This probably should not be hard coded. If these values end
        up actually being persistent, we should move them into base class.

        Args:
        -----
        freq (float): frequency in MHz
        band_center (float): frequency in MHz of the band center
        subband_order (list): order of subbands within the band

        Returns:
        --------
        subband_no (int): subband (0..31) of the frequency within the band
        offset (float): offset from subband center
        '''
        try:
            order = [int(x) for x in subband_order] # convert it to a list
        except ValueError:
            order = [8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15,\
                    31, 0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23]

        # can we pull these hardcodes out?
        bb = floor((freq - (band_center - 307.2 - 9.6)) / 19.2)
        offset = freq - (band_center - 307.2) - bb * 19.2
        
        subband_no = order[bb]
        
        return subband_no, offset

    def get_channel_order(self, channel_orderfile=None):
        ''' produces order of channels from a user-supplied input file


        To Do : un-hardcode this.

        Args:
        -----

        Optional Args:
        --------------
        channelorderfile (str): path to a file that contains one channel per line

        Returns :
        channel_order (int array) : An array of channel orders
        '''

        # to do
        # for now this is literally just a list oops
        # sorry

        if channel_orderfile is not None:
            with open(channel_orderfile) as f:
                channel_order = f.read().splitlines()
        else:
            channel_order = [384, 416, 448, 480, 144, 176, 208, 240, 400, 432,\
                464, 496, 136, 168, 200, 232, 392, 424, 456, 488,\
                152, 184, 216, 248, 408, 440, 472, 504, 132, 164, 196, 228,\
                388, 420, 452, 484, 148, 180, 212, 244, 404, 436, 468, 500,\
                140, 172, 204, 236, 396, 428, 460, 492, 156, 188, 220, 252,\
                412, 444, 476, 508, 130, 162, 194, 226, 386, 418, 450, 482,\
                146, 178, 210, 242, 402, 434, 466, 498, 138, 170, 202, 234,\
                394, 426, 458, 490, 154, 186, 218, 250, 410, 442, 474, 506,\
                134, 166, 198, 230, 390, 422, 454, 486, 150, 182, 214, 246,\
                406, 438, 470, 502, 142, 174, 206, 238, 398, 430, 462, 494,\
                158, 190, 222, 254, 414, 446, 478, 510, 129, 161, 193, 225,\
                385, 417, 449, 481, 145, 177, 209, 241, 401, 433, 465, 497,\
                137, 169, 201, 233, 393, 425, 457, 489, 153, 185, 217, 249,\
                409, 441, 473, 505, 133, 165, 197, 229, 389, 421, 453, 485,\
                149, 181, 213, 245, 405, 437, 469, 501, 141, 173, 205, 237,\
                397, 429, 461, 493, 157, 189, 221, 253, 413, 445, 477, 509,\
                131, 163, 195, 227, 387, 419, 451, 483, 147, 179, 211, 243,\
                403, 435, 467, 499, 139, 171, 203, 235, 395, 427, 459, 491,\
                155, 187, 219, 251, 411, 443, 475, 507, 135, 167, 199, 231,\
                391, 423, 455, 487, 151, 183, 215, 247, 407, 439, 471, 503,\
                143, 175, 207, 239, 399, 431, 463, 495, 159, 191, 223, 255,\
                415, 447, 479, 511, 0, 32, 64, 96, 256, 288, 320, 352,\
                16, 48, 80, 112, 272, 304, 336, 368, 8, 40, 72, 104,\
                264, 296, 328, 360, 24, 56, 88, 120, 280, 312, 344, 376,\
                4, 36, 68, 100, 260, 292, 324, 356, 20, 52, 84, 116,\
                276, 308, 340, 372, 12, 44, 76, 108, 268, 300, 332, 364,\
                28, 60, 92, 124, 284, 316, 348, 380, 2, 34, 66, 98,\
                258, 290, 322, 354, 18, 50, 82, 114, 274, 306, 338, 370,\
                10, 42, 74, 106, 266, 298, 330, 362, 26, 58, 90, 122,\
                282, 314, 346, 378, 6, 38, 70, 102, 262, 294, 326, 358,\
                22, 54, 86, 118, 278, 310, 342, 374, 14, 46, 78, 110,\
                270, 302, 334, 366, 30, 62, 94, 126, 286, 318, 350, 382,\
                1, 33, 65, 97, 257, 289, 321, 353, 17, 49, 81, 113,\
                273, 305, 337, 369, 9, 41, 73, 105, 265, 297, 329, 361,\
                25, 57, 89, 121, 281, 313, 345, 377, 5, 37, 69, 101,\
                261, 293, 325, 357, 21, 53, 85, 117, 277, 309, 341, 373,\
                13, 45, 77, 109, 269, 301, 333, 365, 29, 61, 93, 125,\
                285, 317, 349, 381, 3, 35, 67, 99, 259, 291, 323, 355,\
                19, 51, 83, 115, 275, 307, 339, 371, 11, 43, 75, 107,\
                267, 299, 331, 363, 27, 59, 91, 123, 283, 315, 347, 379,\
                7, 39, 71, 103, 263, 195, 327, 359, 23, 55, 87, 119,\
                279, 311, 343, 375, 15, 47, 79, 111, 271, 303, 335, 367,\
                31, 63, 95, 127, 287, 319, 351, 383, 128, 160, 192, 224]

        return channel_order

    def get_subband_from_channel(self, band, channel, channelorderfile=None):
        """ returns subband number given a channel number
        Args:
        root (str): epics root (eg mitch_epics)
        band (int): which band we're working in
        channel (int): ranges 0..511, cryo channel number

        Optional Args:
        channelorderfile(str): path to file containing order of channels

        Returns:
        subband (int) : The subband the channel lives in
        """

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        #n_subbands = 128 # just for testing while not hooked up to epics server
        #n_channels = 512
        n_chanpersubband = n_channels / n_subbands

        if channel > n_channels:
            raise ValueError('channel number exceeds number of channels')

        if channel < 0:
            raise ValueError('channel number is less than zero!')

        chanOrder = getChannelOrder(channelorderfile)
        idx = chanOrder.index(channel)

        subband = idx // n_chanpersubband
        return int(subband)

    def get_subband_centers(self, band, asOffset=False):
        """ returns frequency in MHz of subband centers
        Args:
         band (int): which band
         asOffset (bool): whether to return as offset from band center \
                 (default is no, which returns absolute values)
        """

        digitizerFrequencyMHz = self.get_digitizer_frequency_mhz(band)
        bandCenterMHz = self.get_band_center_mhz(band)
        n_subbands = self.get_number_sub_bands(band)

        subband_width_MHz = 2 * digitizerFrequencyMHz / n_subbands

        subbands = list(range(n_subbands))
        subband_centers = (np.arange(1, n_subbands + 1) - n_subbands/2) * \
            subband_width_MHz/2

        return subbands, subband_centers

    def get_channels_in_subband(self, band, channelorderfile, subband):
        """ returns channels in subband
        Args:
         band (int): which band
         channelorderfile(str): path to file specifying channel order
         subband (int): subband number, ranges from 0..127
        """

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        n_chanpersubband = int(n_channels / n_subbands)

        if subband > n_subbands:
            raise ValueError("subband requested exceeds number of subbands")

        if subband < 0:
            raise ValueError("requested subband less than zero")

        chanOrder = getChannelOrder(channelorderfile)
        subband_chans = chanOrder[subband * n_chanpersubband : subband * \
            n_chanpersubband + n_chanpersubband]

        return subband_chans

    def iq_to_phase(self, i, q):
        """
        Changes IQ to phase

        Args:
        -----
        i (float array)
        q (float arry)

        Returns:
        --------
        phase (float array) : 
        """
        return np.unwrap(np.arctan2(q, i))


    def hex_string_to_int(self, s):
        """
        Converts hex string, which is an array of characters, into an int.

        Args:
        -----
        s (character array) : An array of chars to be turned into a single int.

        Returns:
        --------
        i (int64) : The 64 bit int
        """
        return np.int(''.join([chr(x) for x in s]),0)


    def int_to_hex_string(self, i):
        """
        Converts an int into a string of characters.

        Args:
        -----
        i (int64) : A 64 bit int to convert into hex.

        Returns:
        --------
        s (char array) : A character array representing the int
        """
        # Must be array length 300
        s = np.zeros(300, dtype=int)
        i_hex = hex(i)[2:]
        for j in np.arange(len(i_hex)):
            s[j] = int(i_hex[j])

        return s

    def set_tes_bias_bipolar(self, bias_num, volt, do_enable=True, **kwargs):
        """
        bias_num (int): The gate number to bias
        volt (float): The TES bias to command in voltage.

        Opt args:
        --------
        do_enable (bool) : Sets the enable bit. Default is True.
        """

        bias_order = np.array([9, 11, 13, 15, 16, 14, 12, 10, 7, 5, 3, 1, 8, 6, 
            4, 2])
        dac_positives = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 
            26, 28, 30, 32])
        dac_negatives = np.array([1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 
            25, 27, 29, 31])

        dac_idx = np.ravel(np.where(bias_order == bias_num))

        dac_positive = dac_positives[dac_idx][0]
        dac_negative = dac_negatives[dac_idx][0]

        volts_pos = volt / 2
        volts_neg = - volt / 2

        if do_enable:
            self.set_tes_bias_enable(dac_positive, 2, **kwargs)
            self.set_tes_bias_enable(dac_negative, 2, **kwargs)

        self.set_tes_bias_volt(dac_positive, volts_pos, **kwargs)
        self.set_tes_bias_volt(dac_negative, volts_neg, **kwargs)

    def set_amplifier_bias(self, bias_hemt=.54, bias_50k=-.71, **kwargs):
        """
        Sets the HEMT and 50 K amp voltages.

        Opt Args:
        ---------
        bias_hemt (float): The HEMT bias voltage in units of volts
        bias_50k (float): The 50K bias voltage in units of volts

        """
        self.set_hemt_enable(**kwargs)
        self.set_50k_amp_enable(**kwargs)

        self.set_hemt_gate_voltage(bias_hemt, **kwargs)
        self.set_50k_amp_gate_voltage(bias_50k, **kwargs)


    def get_timestamp(self):
        """
        Returns:
        timestampe (str): Timestamp as a string
        """
        return '{:10}'.format(int(time.time()))


