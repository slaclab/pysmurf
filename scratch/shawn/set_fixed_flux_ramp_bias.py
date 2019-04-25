import matplotlib
matplotlib.use('Agg')
import pysmurf
import sys

epics_prefix = 'smurf_server_s5'
config_file='/data/pysmurf_cfg/experiment_fp29_smurfsrv03_noExtRef_hbOnlyBay0.cfg'
S = pysmurf.SmurfControl(epics_root=epics_prefix,cfg_file=config_file,setup=False,make_logfile=False)

bias=float(sys.argv[1].replace('n','-'))

print(bias)
S.set_fixed_flux_ramp_bias(bias)

