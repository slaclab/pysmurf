#import epics
import numpy as np
from math import floor

"""Epics wrappers for working with single cryo channel
"""

SysgenCryo = "AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:"



def config_cryo_channel(smurfCfg, bandNo, channelNo, frequencyMhz, ampl, \
        feedbackEnable, etaPhase, etaMag):
    """written to match configCryoChannel.m

       Args:
        smurfCfg (config object): configuration object (really, a dictionary)
        bandNo (int): which band (2 or 3 for 5-6GHz)
        channelNo (int): cryo channel number (0 .. 511)
        frequencyMhz (float): frequency within subband (-19.2 .. 19.2)
        amplitude (int): ADC output amplitude (0 .. 15)
        feedbackEnable (0 or 1): boolean for enabling feedback
        etaPhase (float): feedback eta phase, in degrees (-180 .. 180)
        etaMag (float): feedback eta magnitude
    """

    # construct the pvRoot
    smurfInitCfg = smurfCfg.get('init')
    root = smurfInitCfg['epics_root'] + ":"
    bandStr = 'Base[{0}]:'.format(bandNo) # this is the new Python 3 way
    CryoChannels = SysgenCryo + bandStr + "CryoChannels:"
    epicsRoot = root + CryoChannels + 'CryoChannel[{0}]:'.format(channelNo)

    n_subband = epics.caget(root + SysgenCryo + bandStr +  \
            'numberSubBands') # should be 128
    band = epics.caget(root + SysgenCryo + bandStr + \
            'digitizerFrequencyMHz') # 614.4 MHz
    sub_band = band / (n_subband/2) # width of each subband

    ## some checks to make sure we put in values within the correct ranges

    if frequencyMhz > sub_band/2:
        #print("configCryoChannel: freq too high! setting to top of subband")
        freq = sub_band/2
    elif frequencyMhz < sub_band/2:
        #print("configCryoChannel: freq too low! setting to bottom of subband")
        freq = -sub_band/2
    else:
        freq = frequencyMhz

    if ampl > 15:
        #print("configCryoChannel: amplitude too high! setting to 15")
        amp = 15
    elif ampl < 0:
        #print("configCryoChannel: amplitude too low! setting to 0"
        amp = 0
    else:
        amp = ampl

    # get phase within -180..180
    phase = etaPhase
    while etaPhase > 180:
        phase = phase - 360
    while etaPhase < -180:
        phase = phase + 360

    pv_list = ['centerFrequencyMHz', 'amplitudeScale', 'feedbackEnable', \
            'etaPhaseDegree', 'etaMagScaled']
    pv_values = [freq, amp, feedbackEnable, phase, etaMag]

    for i in range(len(pv_list)):
        epics.caput(epicsRoot + pv_list[i], pv_values[i])

def fluxramp_onoff(root, frState):
    """turn the flux ramp on or off.

    Args:
     root (str): epics root (eg mitch_epics)
     frState (bin): flux ramp state
    """
    epicsRoot = root + ":AMCc:FpgaTopLevel:AppTop:AppCore:"
    rtmRoot = epicsRoot + "RtmCryoDet:"
    rtmSpiRoot = rtmRoot + "RtmSpiSr:"

    epics.caput(rtmSpiRoot + 'CfgRegEnaBit', str(frState))



def band_off(root, bandNo):
    """turn off all the channels quickly (hooray!)

       Args:
        root (str): epics root (eg mitch_epics)
        bandNo (int): band to turn off
    """
    epicsRoot = root + ":" + SysgenCryo + 'Base[{0}]:'.format(bandNo) \
            + 'CryoChannels:'
    epics.caput(epicsRoot + 'setAmplitudeScales', 0)
    epics.caput(epicsRoot + 'feedbackEnableArray', np.zeros(512).astype(int))
    
    fluxramp_onoff(root, 0) # also turn off flux ramp

def turn_off(root, bands):
    """ turn off all bands

    Args:
     root (str): epics root
     bands (list of ints): band(s) to turn off
    """

    for band in bands:
        band_off(root, band)

def getChannelOrder(channelorderfile):
    """ produces order of channels from a user-supplied input file

    Args:
     channelorderfile (str): path to a file that contains one channel
        per line
    """

    # to do
    # for now this is literally just a list oops
    # sorry

    channelOrder = [384, 416, 448, 480, 144, 176, 208, 240, 400, 432, 464, 496,\
            136, 168, 200, 232, 392, 424, 456, 488,\
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

    with open(channelorderfile) as f:
        channelOrder = f.read().splitlines()
    

    return channelOrder

def get_subband_from_channel(root, bandNo, channelorderfile, channel):
    """ returns subband number given a channel number

    Args:
     root (str): epics root (eg mitch_epics)
     bandNo (int): which band we're working in
     channelorderfile(str): path to file containing order of channels
     channel (int): ranges 0..511, cryo channel number
    """

    base_root = root + ":" + SysgenCryo + "Base[{0}]:".format(bandNo)
    n_subbands = epics.caget(base_root, 'numberSubBands')
    n_channels = epics.caget(base_root, 'numberChannels')
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

def get_subband_centers(root, bandNo, asOffset = False):
    """ returns frequency in MHz of subband centers

    Args:
     root (str): epics root
     bandNo (int): which band
     asOffset (bool): whether to return as offset from band center \
             (default is no, which returns absolute values)
    """

    base_root = root + ":" + SysgenCryo + "Base[{0}]:".format(bandNo)
    digitizerFrequencyMHz = epics.caget(base_root + 'digitizerFrequencyMHz')
    bandCenterMHz = epics.caget(base_root + 'bandCenterMHz')
    n_subbands = epics.caget(base_root + 'numberSubBands')

    subband_width_MHz = 2 * digitizerFrequencyMHz / n_subbands

    subbands = list(range(n_subbands))
    subband_centers = (np.arange(1, n_subbands + 1) - n_subbands/2) * \
            subband_width_MHz/2

def get_channels_in_subband(root, bandNo, channelorderfile, subband):
    """ returns channels in subband

    Args:
     root (str): epics root
     bandNo (int): which band
     channelorderfile(str): path to file specifying channel order
     subband (int): subband number, ranges from 0..127
    """

    base_root = root + ":" + SysgenCryo + "Base[{0}]:".format(bandNo)
    n_subbands = epics.caget(base_root + 'numberSubBands')
    n_channels = epics.caget(base_root + 'numberChannels')
    n_chanpersubband = n_channels / n_subbands

    if subband > n_subbands:
        raise ValueError("subband requested exceeds number of subbands")

    if subband < 0:
        raise ValueError("requested subband less than zero")

    chanOrder = getChannelOrder(channelorderfile)
    subband_chans = chanOrder[subband * n_chanpersubband : subband * \
            n_chanpersubband + n_chanpersubband]

    return subband_chans


def parallel_scan(root, bandNo, scanchans, Adrive = 10, scanfreqs = None):
    """ Scan a bunch of channels in parallel for real and complex response
    Returns (n_freqs) x (n_channels) complex frequency response

    Args:
     root: epics root
     bandNo: which band to scan/read
     scanchans (np array): 1 x n_channels logical array, 1 if channel should
        be scanned
     Adrive (int): drive power to scan at
     scanfreqs (np array): n_freqs x n_channels array of frequencies to scan.
        For each channel j, will step through (:, j) frequencies
    """

    base_root = root + ":" + SysgenCryo + "Base[{0}]:".format(bandNo)
    cryochannels_root = base_root + "CryoChannels:"

    n_channels = epics.caget(base_root + 'numberChannels')

    if scanfreqs is None:
        # default to scanning -3 to 3 about channel center
        scanfreqs = np.transpose(np.broadcast_to(np.arange(-3, 3.1, 0.1), \
                (61, n_channels))) # there has to be some better way :(

    epics.caput(cryochannels_root + 'etaMagArray', np.ones((1, n_channels)))
    epics.caput(cryochannels_root + 'feedbackEnableArray', np.zeros((1, \
            n_channels)))
    epics.caput(cryochannels_root + 'amplitudeScaleArray', Adrive * scanchans)

    n_freqs = np.shape(scanfreqs)[0]
    freq_error = np.zeros((n_freqs, n_channels)) # preinitialize for speed
    realImag = [1, 1j]
    etaPhase = [0, 90]

    # this part is currently hideously slow
    for x in range(2): # two sweeps, indexed by x
        epics.capus(cryochannels_root + 'etaPhaseArray', etaPhase[x] * \
                np.ones((1, n_channels)))
        for freq in range(n_freqs): # indexing is easier than iterating directly
            epics.caput(cryochannels_root + 'centerFrequencyArray', \
                    scanfreqs[freq, :])
            freq_error[freq, :] = freq_error[freq, :] + realImag[x] * \
                    epics.caget(cryochannels_root + 'frequencyErrorArray')

    return freq_error


def freq_to_subband(freq, band_center, subband_order):
    """look up subband number of a channel frequency

       Args:
        freq (float): frequency in MHz
        band_center (float): frequency in MHz of the band center
        subband_order (list): order of subbands within the band

       Outputs:
        subband_no (int): subband (0..31) of the frequency within the band
        offset (float): offset from subband center
    """

    # subband_order = [8 24 9 25 10 26 11 27 12 28 13 29 14 30 15 31 0 16 1 17\
    #        2 18 3 19 4 20 5 21 6 22 7 23]
    # default order, but this is a PV now so it can be fed in

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

