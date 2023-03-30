import os
import pysmurf
import matplotlib.pylab as plt
import numpy as np
import sys

config_file_path='/data/pysmurf_cfg/'

slot=int(sys.argv[1])
epics_prefix = 'smurf_server_s%d'%slot

config_file='experiment_tkid_lbOnlyBay0.cfg'

config_file=os.path.join(config_file_path,config_file)    

S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False) 
