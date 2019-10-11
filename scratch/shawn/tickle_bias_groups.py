import numpy as np
import time
import matplotlib.pylab as plt
import sys

start=time.time()

ncycles=5
scale=0.25 #V
nsteps=1000
sig   = scale*np.sin(2*np.pi*np.array(range(nsteps))/nsteps);
sig   = np.tile(sig,ncycles)

zeros=np.zeros_like(S.get_tes_bias_bipolar_array())
S.set_tes_bias_bipolar_array(zeros)

S.set_smurf_to_gcp_clear(1, wait_after=0.5)
S.set_smurf_to_gcp_clear(0, wait_after=0.5)
print('Waiting 25 sec after clearing...')
time.sleep(25)

S.stream_data_on()
time.sleep(1)

#play sine wave on all bias groups, sequentially
bias_groups = S.all_groups
for bias_group in bias_groups:
    print(f'bias_group={bias_group}')
    for v in sig:
        S.set_tes_bias_bipolar(bias_group,v)

time.sleep(1)

S.stream_data_off()
S.set_tes_bias_bipolar_array(zeros)

end=time.time()
elapsed_time=(end-start)
print(f'elapsed_time={elapsed_time}')
