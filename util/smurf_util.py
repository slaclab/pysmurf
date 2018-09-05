import numpy as np
from pysmurf.base import SmurfBase

class SmurfUtilMixin(SmurfBase):

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

    def get_channel_order(self, channelorderfile=None):
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
        n_subbands = self.get_number_channels(band)

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