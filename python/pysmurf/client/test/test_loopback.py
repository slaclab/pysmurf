import pytest
import numpy as np
import pysmurf.client
import os

@pytest.fixture(scope='session')
def smurf_control():
    epics_prefix = 'smurf_server_s5'
    config_file = os.path.join('/usr/local/src/pysmurf/',
                               'cfg_files/stanford/',
                               'experiment_fp30_cc02-03_lbOnlyBay0.cfg' )
    S = pysmurf.client.SmurfControl(epics_root=epics_prefix,
                                    cfg_file=config_file,
                                    setup=False,
                                    make_logfile=False,
                                    shelf_manager="shm-smrf-sp01")
    
    return S


def test_uc_dc_atts(smurf_control):
    accept_frac = .1
    
    band = 1
    # Turn off all attenuators on band
    smurf_control.set_att_uc(band, 0)
    smurf_control.set_att_dc(band, 0)

    # Take baseline measurement
    f, r = smurf_control.full_band_resp(band, make_plot=False,
                                 show_plot=False, correct_att=False)

    # Check UC atts
    uc_atts = np.array([5, 15, 25])
    for uc in uc_atts:
        smurf_control.set_att_uc(band, uc, wait_after=.1)
        f, resp = smurf_control.full_band_resp(band, make_plot=False,
                                               show_plot=False,
                                               correct_att=False)
        ratio = np.median(np.abs(resp)/np.abs(r))
        exp_ratio = np.sqrt(10**(-uc/10/2))
        assert ratio < exp_ratio * (1+accept_frac) and \
            ratio > exp_ratio * (1-accept_frac), \
            "UC att not within acceptable limits."

    # Turn UC att back to 0
    smurf_control.set_att_uc(band, 0)
    
    # Check DC atts
    dc_atts = np.array([5, 15, 25])
    for dc in dc_atts:
        smurf_control.set_att_dc(band, dc, wait_after=.1)
        f, resp = smurf_control.full_band_resp(band, make_plot=False,
                                               show_plot=False,
                                               correct_att=False)
        ratio = np.median(np.abs(resp)/np.abs(r))
        exp_ratio = np.sqrt(10**(-dc/10/2))
        assert ratio < exp_ratio * (1+accept_frac) and \
            ratio > exp_ratio * (1-accept_frac), \
            "DC att not within acceptable limits."

    
def test_data_write_and_read(smurf_control):
    band = 1
    
    # Define the payload size (this is the default val too)
    payload_size = 512
    smurf_control.set_payload_size(payload_size)

    # Turn on some channels
    x = (np.random.randn(512)>0)*10
    smurf_control.set_amplitude_scale_array(band, x)
    input_n_chan = np.sum(x>0)
    
    # Take 5 seconds of data
    filename = smurf_control.take_stream_data(5)

    # The mask file is set by the data streamer, so num_channels
    # is not set until then. So this check needs to happen
    # afterwards.
    assert (len(smurf_control.which_on(band)) ==
            smurf_control.get_smurf_processor_num_channels()),\
            f"The number of channels on band {band} is not the same as " + \
            "the number of channels the smurf_processor thinks are on." + \
            "You may have other channels on in another band."
    
    t, d, m = smurf_control.read_stream_data(d)
    n_chan, n_samp = np.shape(d)

    assert n_samp > 0, \
        "The data written to disk has no samples. Something is wrong. " + \
        "Check that the flux ramp is on and you are triggering in the " + \
        "correct mode. See documentation for set_ramp_start_mode."
    
    assert n_chan == 512, \
        "read_stream_data should return data with 512 channels " + \
        "by default. "

    t, d, m = smurf_control.read_stream_data(filename, array_size=0)
    n_chan, _ = np.shape(d)

    assert n_chan == input_n_chan,\
        "read_stream_data it supposed to return an array of size n_chan " +\
        "when optional arg array_size=0."
