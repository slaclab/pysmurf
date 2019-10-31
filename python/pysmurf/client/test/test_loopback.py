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

    
