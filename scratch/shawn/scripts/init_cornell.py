import pysmurf
import matplotlib.pylab as plt
import numpy as np
import sys
import os

config_file_path='/data/pysmurf_cfg/'

slot=int(sys.argv[1])
epics_prefix = 'smurf_server_s%d'%slot

#config_file='experiment_cornell.cfg'
config_file='experiment_cornell_wilma_ccc2-09_hbOnlyBay0.cfg'
config_file=os.path.join(config_file_path,config_file) 

S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False)

# right now, using HB in bay 0.  Set bandCenterMHz
# manually so don't have to constantly remember the
# +2GHz.
S.set_band_center_mhz(0,6250)
S.set_band_center_mhz(1,6750)
S.set_band_center_mhz(2,7250)
S.set_band_center_mhz(3,7750)
