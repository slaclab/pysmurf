# Written by Matthew Hasselfield to compute DS counter bitmasks for
# Simons Observatory
from dscounters import *
import sys

# Pick a target frequency
target_f = float(sys.argv[1]) # target frequency in Hz

# Use the 16-bit config.  v3 here matches the implementation of the
# fixed rate downsampling counters in SMuRF tpg_ioc v5.0.1, the first
# stable version of the timing pattern generator software released for
# Simons Observatory.
dc = DownsampleCounters(configs['v3'])

# State the base frequency
base_f = 4000. # Hz

# What is ideal downsampling period?
n_target = base_f / target_f
if dc.get_mask(n_target) is None:
    print('This target frequency cannot be achieved exactly.')

# Get best downsampling period
n_achieved = dc.get_nearby(n_target)
if dc.get_mask(n_achieved) is None:
    print('This target frequency cannot be achieved exactly.')

# Compare achieved readout freq to the target ...
achieved_f = base_f / n_achieved
print(f'Target: {target_f}  Achieved: {achieved_f}    [Hz]')

# What's the bitmask?
bitmask = dc.get_mask(n_achieved, str)
print(f'Bit mask: {bitmask}')

# Write this to the ExternalBitmask register in SmurfProcessor via the
# pysmurf set_downsample_external_bitmask function to achieve this
# rate.  The bitmask is shifted 10 bits to the left because the first
# 10 bits of the timing system counters are the higher rate FIXEDDIV
# counters.
print(f'Put this in Downsampler>ExternalBitmask: {int(bitmask[1:],2)<<10}')
