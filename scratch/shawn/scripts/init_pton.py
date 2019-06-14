import pysmurf
import matplotlib.pylab as plt
import numpy as np

epics_prefix = 'smurf_server_s2' 
config_file='/data/pysmurf_cfg/experiment_pc004_smurfsrv08_noExtRef_dspv3.cfg'

S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False) 
