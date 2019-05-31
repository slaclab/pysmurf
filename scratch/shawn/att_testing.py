# Runlike this exec(open("att_testing.py").read())
# to use the pysmurf S object you've already initialized
import scipy.signal as signal
import time
import numpy as np
import sys

def check_att(ctime,which_att,att_idx,use_pysmurf=True,att_wait_after=2):
    bands=[0,1,2,3]
    slot=5

    epics_path_to_atts='smurf_server_s%d:AMCc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:ATT'%slot
    
    # Save DAC and ADC spectra
    adc_spectra={}
    labels=['all bands UC & DC->0','%s[%d]->0x1F'%(which_att,att_idx)]
    for label in labels:
        adc_spectra[label]={}

        if label is 'all bands UC & DC->0':
            print('->',label)
            for band in bands:
                if use_pysmurf:
                    S.set_att_uc(band,0)
                    time.sleep(att_wait_after)                    
                    S.set_att_dc(band,0)
                    time.sleep(att_wait_after)
                else:
                    os.system('caput %s:UC[%d] 0'%(epics_path_to_atts,band+1))
                    time.sleep(att_wait_after)
                    os.system('caput %s:DC[%d] 0'%(epics_path_to_atts,band+1))
                    time.sleep(att_wait_after)
        else:
            print('->',label)        
            if which_att is 'UC':
                if use_pysmurf:
                    S.set_att_uc(att_idx-1,0x1F,write_log=True)
                    time.sleep(att_wait_after)
                else:
                    os.system('caput %s:UC[%d] 0x1F'%(epics_path_to_atts,att_idx))
                    time.sleep(att_wait_after)
                    
            elif which_att is 'DC':
                if use_pysmurf:                
                    S.set_att_dc(att_idx-1,0x1F,write_log=True)
                    time.sleep(att_wait_after)                    
                else:
                    os.system('caput %s:DC[%d] 0x1F'%(epics_path_to_atts,att_idx))
                    time.sleep(att_wait_after)                    

        for band in bands:
            adc_spectra[label][band]=S.read_adc_data(band, do_plot=False)

    fig,axs = plt.subplots(2,2,figsize=(10,5))
    fig.subplots_adjust(left=0.08,right=0.98,wspace=0.3,hspace=0.3)
    for label in labels:
        for band in adc_spectra[label].keys():
            ax=axs[int(band%2),int(np.floor(band/2))]
            dat=adc_spectra[label][band]

            f, p_adc = signal.welch(dat, fs=614.4E6, nperseg=len(dat)/2, return_onesided=False,detrend=False)            
            f_plot = f / 1.0E6

            ax.set_ylabel('ADC{}'.format(band))
            ax.set_xlabel('Frequency [MHz]')
            #ax.set_title(timestamp)            
            ax.semilogy(f_plot, p_adc, alpha=0.5, lw=0.5, label=label)
            ax.set_ylim(1e-6,1e4)
            ax.set_xlim(-614.4/2,614.4/2)
            plt.grid()
            ax.legend(fontsize=6,loc='upper left')

    plt.savefig('/home/cryo/shawn/{}_{}{}.png'.format(ctime,which_att,att_idx))

ctime=S.get_timestamp()
    
for which_att in ['UC']:
    for att_idx in [1,2,3,4]:
        check_att(ctime,which_att,att_idx,use_pysmurf=True)
