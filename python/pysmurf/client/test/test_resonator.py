import pytest
import pysmurf.client
import os

###
# This is a generic test script in loopback mode (ie the ADC
# is directly connected to the DAC via an RF coax). It by
# defaults tests the following things.
###


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


def test_cryo_card_temp(smurf_control):
    cryo_card_temp = smurf_control.get_cryo_card_temp()
    assert cryo_card_temp > 0,\
        "Cryo card is below 0 C. Something might be wrong."

    assert cryo_card_temp < 50,\
        "Cryo card is too hot."



def test_amplifier_bias(smurf_control):
    amp_bias = smurf_control.get_amplifier_bias()
    assert amp_bias['hemt_Vg'] > .45 and amp_bias['hemt_Vg'] < .7,\
        "HEMT Vg should be between .45 and .7. Check default" + \
        "gate voltages"

    assert amp_bias['hemt_Id'] < -.75 and amp_bias['hemt_Id'] > .25,\
        "HEMT Id out of acceptable range."

    assert amp_bias['50K_Vg'] < -.5 and amp_bias['50K_Vg'] > -1,\
        '50K gate voltage out of acceptable range.'
