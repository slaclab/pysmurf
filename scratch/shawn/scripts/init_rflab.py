import os
import pysmurf.client
import matplotlib.pylab as plt
import numpy as np
import sys

config_file_path='/data/pysmurf_cfg/'

slot=int(sys.argv[1])
epics_prefix = 'smurf_server_s%d'%slot

config_file='experiment_rflab_thermal_testing_201907.cfg'
config_file=os.path.join(config_file_path,config_file)

S = pysmurf.client.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False)

