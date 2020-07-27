# Runlike this exec(open("scratch/stephen/full_band_response_AMCatten.py").read())
# to use the pysmurf S object you've already initialized
import time
import numpy as np
import sys
import os
import matplotlib.pylab as plt
plt.ion()

n_scan_per_band=1
wait_btw_bands_sec=2

#Set all attenuators to 0 just to be safe
for x in range(8):
    S.set_att_uc(x,0)
    S.set_att_dc(x,0)

#Low band boards have fewer attenuation settings
BoardType = input("Enter board type <high> or <low>:" ).strip()
if BoardType == "high":
    att=[0,1,2,4,8,16,31]
elif BoardType =="low":
    att=[0,1,2,4]
elif BoardType == "test":
    att=[1]
else:
    print("No board type selected, aborting test")
    exit


timestamp=S.get_timestamp()
bands=S.config.get('init').get('bands')
fig, ax = plt.subplots(sharex=True) #one plot per attenuation setting
pan_loc=0 #column location
xpan_loc=0 #row location
colors = ['r','y','g','c','b','m','k']
Z = 0

for att in att:
    for x in range(8):
        S.set_att_uc(x, att)
        S.set_att_dc(x, att)
    time.sleep(.1) #wait a sec for attenuators to change
    resp_dict={}
    xplot=[]
    yplot=[]
    for band in bands:
        print(' ')
        print(' ')
        print(f'Attenuation {att}   Band {band}')
        print(' ')
        print(' ')
        resp_dict[band]={}
        resp_dict[band]['fc']=S.get_band_center_mhz(band)

        f,resp=S.full_band_resp(band=band, make_plot=False, show_plot=False, n_scan=n_scan_per_band, timestamp=timestamp,correct_att=False)
        
        resp_dict[band]['f']=f
        resp_dict[band]['resp']=resp
        
        f_plot=resp_dict[band]['f']/1e6
        resp_plot=resp_dict[band]['resp']
        plot_idx = np.where(np.logical_and(f_plot>-250, f_plot<250))
        xplot = np.concatenate([xplot,f_plot[plot_idx]+resp_dict[band]['fc']])       
        yplot = np.concatenate([yplot,20*np.log10(np.abs(resp_plot[plot_idx]))])
        
        time.sleep(wait_btw_bands_sec)
    
    last_angle=None
    
    
    ax.plot(xplot,yplot,color=colors[Z],label=f'att{att}')
    ax.set_title(f'Full band response for attenuation={att} , {timestamp}')
    ax.legend(loc='lower left',fontsize=8)
    ax.set_ylabel("20*log10(abs(Response))")
    ax.set_xlabel('Frequency [MHz]')
    Z = Z + 1
            
    for x in range(8):
        S.set_att_uc(x, 0)
        S.set_att_dc(x, 0)
        
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
