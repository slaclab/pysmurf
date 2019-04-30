import pysmurf
# import matplotlib.pylab as plt
import numpy as np
import sys
slot=int(sys.argv[1])

if slot==5:
    epics_prefix = 'smurf_server_s5'
    config_file='/data/pysmurf_cfg/experiment_fp29_smurfsrv03_noExtRef_hbOnlyBay0.cfg'
elif slot==2:
    epics_prefix = 'smurf_server_s2'
    config_file='/data/pysmurf_cfg/experiment_fp29_smurfsrv03_noExtRef_lbOnlyBay0.cfg'
else:
    assert False,"There's nothing in slot %d right now!"%slot


S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False) 
