import numpy as np
import pysmurf

S = pysmurf.SmurfControl(make_logfile=False, output_dir_only=True)

datafile = '/home/common/data/cpu-b000-hp01/cryo_data/data2/20180921/1537561706/outputs/1537562122.dat'

print('Loading data')
timestamp, I, Q = S.read_stream_data(datafile)

ch = 0

bias = np.arange(19.9, 0, -.1)
phase = S.iq_to_phase(I[ch], Q[ch]) * S.pA_per_phi0 / (2*np.pi)


print('Running IV analysis')
S.analyze_slow_iv(bias, phase, make_plot=True, show_plot=True)