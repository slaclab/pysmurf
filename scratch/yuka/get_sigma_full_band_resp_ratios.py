import pysmurf.client
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import epics
import os

epics_prefix = 'smurf_server_s5'
config_file  = "/usr/local/src/pysmurf/cfg_files/caltech/tkid/experiment_tkid_lbOnlyBay0.cfg"
S = pysmurf.client.SmurfControl(epics_root=epics_prefix, cfg_file=config_file, setup=False, make_logfile=True)

S.setup()

S.set_att_uc(0,0)
S.set_att_dc(0,0)

# Response with no attenuators
resp_0 = S.full_band_resp(0, n_scan=5, make_plot=True, save_plot=True,show_plot=False, return_plot_path=True)
resp_0 = resp_0[1]

# Response with uc 0, dc 18
S.set_att_dc(0,18)
uc_0_dc_18 = []
# Get 5 sets of responses
for i in range(5):
    resp = S.full_band_resp(0, n_scan=5, make_plot=True, save_plot=True,show_plot=False, return_plot_path=True)
    resp = resp[1]
    r = np.mean(resp/resp_0)
    uc_0_dc_18.append(r)
#  Get the sigma for the ratios, this is the error for the ratio (signal to noise is the (mean of the ratios)/(sigma of the ratios))
sigma = np.std(uc_0_dc_18)
# The allowable range should be maybe 3*sigma
print(sigma) # Got 0.00031287861778726226 (4/16/2021), 0.0007067760463537454 (4/19/2021)

# Repeat for uc 18, dc 0
S.set_att_dc(0,0)
S.set_att_uc(0,18)
uc_18_dc_0 = []
for i in range(5):
    resp = S.full_band_resp(0, n_scan=5, make_plot=True, save_plot=True,show_plot=False, return_plot_path=True)
    resp = resp[1]
    r = np.mean(resp/resp_0)
    uc_18_dc_0.append(r)
sigma = np.std(uc_18_dc_0)
print(sigma) # Got 0.000676165505128212 (4/16/2021), 0.00043685647636345704 (4/19/2021)

# Now uc 18, dc 18
S.set_att_dc(0,18)
uc_18_dc_18 = []
for i in range(5):
    resp = S.full_band_resp(0, n_scan=5, make_plot=True, save_plot=True,show_plot=False, return_plot_path=True)
    resp = resp[1]
    r = np.mean(resp/resp_0)
    uc_18_dc_18.append(r)
sigma = np.std(uc_18_dc_18)
print(sigma) # Got 0.0011676555792350957 (4/16/2021), 0.0014418298966569862 (4/19/2021)
