import pysmurf
import numpy as np
import time
import sys

## instead of takedebugdata try relaunch PyRogue, then loopFilterOutputArray, which is 
## the integral tracking term with lmsEnable[1..3]=0

S = pysmurf.SmurfControl(make_logfile=False,setup=False,epics_root='smurf_server',cfg_file='/data/pysmurf_cfg/experiment_pc002_smurfsrv08_noExtRef.cfg')

#######
bias=None
wait_time=.0
#bias_low=-0.432
#bias_high=0.432
bias_low=-0.45
bias_high=0.45
bias_step=.0015
quiet=True

if bias is None:
    bias = np.arange(bias_low, bias_high, bias_step)

S.log('Staring to flux ramp.', S.LOG_USER)

sys.stdout.write('\rSetting flux ramp bias low at {:4.3f} V\033[K'.format(bias_low))
S.set_fixed_flux_ramp_bias(bias_low,debug=True,do_config=True)

try:
    while True:
        for b in bias:
            if not quiet:
                sys.stdout.write('\rFlux ramp bias at {:4.3f} V\033[K'.format(b))
                sys.stdout.flush()
            S.set_fixed_flux_ramp_bias(b,do_config=False,debug=False)
            time.sleep(wait_time)
            
except KeyboardInterrupt:
    sys.stdout.write('\rKeyboard interrupt -> setting flux ramp to zero')
    
    # done - zero and unset
    S.set_fixed_flux_ramp_bias(0,do_config=False,debug=False)
    S.unset_fixed_flux_ramp_bias()
