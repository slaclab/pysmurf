# Runlike this exec(open("scratch/shawn/full_band_response.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
import os
import matplotlib.pylab as plt
plt.ion()

n_scan_per_band=5
wait_btw_bands_sec=5

#Set all attenuators to 0 just to be safe
for x in range(4):
    S.set_att_uc(x,0)
    S.set_att_dc(x,0)

#Low band boards have fewer attenuation settings
BoardType = raw_input("Enter board type <high> or <low>:" ).strip()
if BoardType == "high":
    att=[0,1,2,4,8,16,31]
elif BoardType =="low":
    att=[0,1,2,4]
else:
    print("No board type selected, aborting test")
    exit

timestamp=S.get_timestamp()
bands=S.config.get('init').get('bands')
fig, ax = plt.subplots(ncols=len(att), sharex=True) #one plot per attenuation setting
pan_loc=0 #column location

for att in att:
    for x in range(4):
        S.set_att_uc(x, att)
        S.set_att_dc(x, att)
    time.sleep(.1) #wait a sec for attenuators to change
    resp_dict={}
    for band in bands:
        print(' ')
        print(' ')
        print(f'Band {band}')
        print(' ')
        print(' ')
        resp_dict[band]={}
        resp_dict[band]['fc']=S.get_band_center_mhz(band)

        f,resp=S.full_band_resp(band=band, make_plot=False, show_plot=False, n_scan=n_scan_per_band, timestamp=timestamp,correct_att=False)
        resp_dict[band]['f']=f
        resp_dict[band]['resp']=resp

        time.sleep(wait_btw_bands_sec)

    ax[0,pan_loc].set_title(f'Full band response for attenuation={x} , {timestamp}')
    last_angle=None
    for band in bands:
        f_plot=resp_dict[band]['f']/1e6
        resp_plot=resp_dict[band]['resp']
        plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))
        ax[0,pan_loc].plot(f_plot[plot_idx]+resp_dict[band]['fc'], np.log10(np.abs(resp_plot[plot_idx])),label=f'b{band}')

    ax[0,pan_loc].legend(loc='lower left',fontsize=8)
    ax[0,pan_loc].set_ylabel("log10(abs(Response))")
    ax[0,pan_loc].set_xlabel('Frequency [MHz]')

    for x in range(4):
        S.set_att_uc(x, 0)
        S.set_att_dc(x, 0)
        pan_loc = pan_loc + 1


save_name = '{}_full_band_resp_atten.png'.format(timestamp)
plt.title(save_name)

plt.tight_layout()

print(f'Saving plot to {os.path.join(S.plot_dir, save_name)}')
plt.savefig(os.path.join(S.plot_dir, save_name),
            bbox_inches='tight')
plt.show()

# log plot file
logf=open('/data/smurf_data/smurf_loop.log','a+')
logf.write(f'{os.path.join(S.plot_dir, save_name)}'+'\n')
logf.close()

print('RF Test Complete')
