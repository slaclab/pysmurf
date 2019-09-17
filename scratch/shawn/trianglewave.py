import pysmurf

#S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='test_epics',cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_fp28_smurfsrv04.cfg')


import numpy as np
import time

Vrange=np.linspace(0,0.195/6.,100)+S.get_tes_bias_bipolar(3)
Vrange=[Vrange,Vrange[::-1]]
Vrange=np.array(Vrange).flatten()

while True:
    for Vtes in Vrange:
        S.set_tes_bias_bipolar(7,Vtes)
        time.sleep(0.005)

