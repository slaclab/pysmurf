import pysmurf
import matplotlib.pylab as plt
import numpy as np

epics_prefix = 'smurf_server_s2' 
config_file='/data/pysmurf_cfg/experiment_pc004_smurfsrv08_noExtRef_dspv3.cfg'

S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False)

# right now, using HB in bay 0.  Set bandCenterMHz
# manually so don't have to constantly remember the
# +2GHz.
#S.set_band_center_mhz(0,6250)
#S.set_band_center_mhz(1,6750)
#S.set_band_center_mhz(2,7250)
#S.set_band_center_mhz(3,7750)

