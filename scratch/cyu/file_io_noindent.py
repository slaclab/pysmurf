import math
import numpy as np

def process_data(filename, dtype=np.uint32):
    """
    reads a file taken with take_debug_data and processes it into
       data + header

    Args:
    -----
    filename (str): path to file

    Optional:
    dtype (np dtype): datatype to cast to, defaults unsigned 32 bit int

    Returns:
    -----
    header (np array)
    data (np array)
    """
    n_chan = 2 # number of stream channels
    header_size = 4 # 8 bytes in 16-bit word

    rawdata = np.fromfile(filename, dtype='<u4').astype(dtype)

    rawdata = np.transpose(np.reshape(rawdata, (n_chan,-1))) # -1 
	# is equiv to [] in Matlab and transpose is a weird Python/Matlab
	# incompatibility

    if dtype==np.uint32:
        header = rawdata[:2, :]
        data = np.delete(rawdata, (0,1), 0).astype(dtype)
    elif dtype==np.int32:
        header = np.zeros((2,2))
        header[:,0] = rawdata[:2,0].astype(np.uint32)
        header[:,1] = rawdata[:2,1].astype(np.uint32)
        data = np.double(np.delete(rawdata, (0,1), 0))
    elif dtype==np.int16:
        header1 = np.zeros((4,2))
        header1[:,0] = rawdata[:4,0].astype(np.uint16)
        header1[:,1] = rawdata[:4,1].astype(np.uint16)
        header1 = np.double(header1)
        header = header1[::2] + header1[1::2] * (2**16) # what am I doing
    else:
        raise TypeError('Type {} not yet supported!'.format(dtype))

    if header[1,1] == 2:
        header = np.fliplr(header)
        data = np.fliplr(data)

    return header, data

def decode_data(filename, swapFdF=False):
    """
    take a dataset from take_debug_data and spit out results

    Args:
    -----
    filename (str): path to file

    Optional:
    swapFdF (bool): whether the F and dF (or I/Q) streams are flipped

    Returns:
    -----
    [f, df, sync] if iqStreamEnable = 0
    [I, Q, sync] if iqStreamEnable = 1
    """

    subband_halfwidth_MHz = 4.8 # can we remove this hardcode
    if swapFdF:
        nF = 1
        nDF = 0
    else:
        nF = 0
        nDF = 1

    header, rawdata = process_data(filename)

    # decode strobes
    strobes = np.floor(rawdata / (2**30))
    data = rawdata - (2**30)*strobes
    ch0_strobe = np.remainder(strobes, 2)
    flux_ramp_strobe = np.floor((strobes - ch0_strobe) / 2)

    # decode frequencies
    ch0_idx = np.where(ch0_strobe[:,0] == 1)[0]
    f_first = ch0_idx[0]
    f_last = ch0_idx[-1]
    freqs = data[f_first:f_last, 0]
    neg = np.where(freqs >= 2**23)[0]
    f = np.double(freqs)
    if len(neg) > 0:
        f[neg] = f[neg] - 2**24

    if np.remainder(len(f),512)==0:
        f = np.reshape(f, (-1, 512)) # -1 is [] in Matlab

    #flux_ramp_strobe_f = flux_ramp_strobe[f_first, f_last, nF] # what is this?

    # frequency errors
    ch0_idx_df = np.where(ch0_strobe[:,1] == 1)[0]
    if len(ch0_idx_df) > 0:
        d_first = ch0_idx_df[0]
        d_last = ch0_idx_df[-1]
        dfreq = data[d_first:d_last, 1]
        neg = np.where(dfreq >= 2**23)[0]
        df = np.double(dfreq)
        if len(neg) > 0:
            df[neg] = df[neg] - 2**24

        if np.remainder(len(df), 512) == 0:
            df = np.reshape(df, (-1, 512)) * subband_halfwidth_MHz/ 2**23
    else:
        df = []

    #flux_ramp_strobe_df = flux_ramp_strobe[d_first, d_last, nDF]
    return f, df, flux_ramp_strobe

def decode_single_channel(filename, swapFdF=False):
    """
    decode take_debug_data file if in singlechannel mode

    Args:
    -----
    filename (str): path to file to decode

    Optional:
    swapFdF (bool): whether to swap f and df streams

    Returns:
    [f, df, sync] if iq_stream_enable = False
    [I, Q, sync] if iq_stream_enable = True
    """

    subband_halfwidth_MHz = 4.8 # take out this hardcode

    if swapFdF:
        nF = 1
        nDF = 0
    else:
        nF = 0
        nDF = 1

    header, rawdata = process_data(filename)

    # decode strobes
    strobes = np.floor(rawdata / (2**30))
    data = rawdata - (2**30)*strobes
    ch0_strobe = np.remainder(strobes, 2)
    flux_ramp_strobe = np.floor((strobes - ch0_strobe) / 2)

    # decode frequencies
    freqs = data[:,nF]
    neg = np.where(freqs >= 2**23)[0]
    f = np.double(freqs)
    if len(neg) > 0:
        f[neg] = f[neg] - 2**24

    f = np.transpose(f) * subband_halfwidth_MHz / 2**23

    dfreqs = data[:,nDF]
    neg = np.where(dfreqs >= 2**23)[0]
    df = np.double(dfreqs)
    if len(neg) > 0:
        df[neg] = df[neg] - 2**24

    df = np.transpose(df) * subband_halfwidth_MHz / 2**23

    return f, df, flux_ramp_strobe

