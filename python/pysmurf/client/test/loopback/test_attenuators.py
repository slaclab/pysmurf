import pytest
import pysmurf.client
import os

@pytest.fixture
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


def test_method():
    smurf_control.full_band_resp(1, make_plot=False, show_plot=False)
    
