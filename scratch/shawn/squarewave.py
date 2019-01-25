import pysmurf

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='test_epics',cfg_file='/usr/local/controls/Applications/smurf/pysmurf/pysmurf/cfg_files/experiment_fp28_smurfsrv04.cfg')

import numpy as np
import time
import sys

dwell=1
count=0
while count<int(sys.argv[1]):
    print('count=%d'%count)
    S.set_tes_bias_bipolar(3,0.195/200.)
    time.sleep(dwell)
    S.set_tes_bias_bipolar(3,0.0)
    time.sleep(dwell)
    count=count+1

