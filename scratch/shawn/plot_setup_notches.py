import glob
import numpy as np
import matplotlib.pylab as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def plot_setup_notches(ctime,color,label,band=0):
    tunedata=glob.glob(f'/data/smurf_data/tune/{ctime}_tune.npy')[0]

    tnpy=np.load(tunedata,allow_pickle=True)
    tdata=tnpy.item()

    resd=tdata[band]['resonances']
    frs=[]
    for k in resd.keys():
        freq=resd[k]['freq_eta_scan']
        resp=resd[k]['resp_eta_scan']
        if k==0:
            plt.plot(freq,np.abs(resp), linestyle='-', marker='.',markersize=4,color=color,alpha=0.75,label=f'{ctime} : {label}')
        else:
            plt.plot(freq,np.abs(resp), linestyle='-', marker='.',markersize=4,color=color,alpha=0.75)
        # draw resonance freq
        fr=resd[k]['freq']
        frs.append((fr,np.interp(fr,freq,np.abs(resp))))
        plt.gca().scatter([fr],
                          [frs[-1][1]],
                          s=320, marker='*', color=color,
                          zorder=3, alpha=0.75)
                          
    return frs,resd

cm = plt.get_cmap('viridis')

ctimes=[1678227290,1678227391,1678235400]
#ctimes=[1678226516,1678226586]
#ctimes=[1678213822, 1678213846, 1678213870, 1678213895, 1678213918, 1678213942, 1678213966, 1678213989, 1678214014, 1678214037]
for idx,ctime in enumerate(ctimes):
    color = cm(idx/len(ctimes))
    frs,resd=plot_setup_notches(ctime,color=color,label=f'{ctime}')

plt.xlabel("frequency (MHz)")
plt.ylabel("setup_notches response")

plt.legend(loc='upper right',fontsize=6)

# find pairs
#for idx1,fr1 in enumerate(frs1):
#    (f1,r1)=fr1
#    # partner within 100kHz?
#    idx2=find_nearest([f for f,r in frs2],f1)
#    f2=frs2[idx2][0]
#    if np.abs(f1-f2)<0.1 and f1!=f2:
#print(f'{f1:0.4f} MHz -> {f2:0.4f} MHz ; Delta = {1000.*(f2-f1):0.4f} kHz')

plt.show()

