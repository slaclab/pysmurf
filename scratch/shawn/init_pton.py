import pysmurf
import matplotlib.pylab as plt
import numpy as np

# the s2 means this is the slot 2 server.
epics_prefix = 'smurf_server_s2'
# point to your config file here
config_file='/data/pysmurf_cfg/experiment_pc004_smurfsrv08_noExtRef.cfg'

# don't initialize by default
S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False) 
