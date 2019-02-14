import epics
import time
import pylab as P
import os
import matplotlib.pyplot as plt
import numpy
from scipy import signal
import glob
from datetime import datetime

outfname = "/tmp/data.txt"
pwelch_ratio = 8;  # to set size of nperseg

def extract_data(datafp, first=0, last=0, filt=1):
    cmd = "/usr/local/controls/Applications/smurf/smurftestapps/extractdata "+ datafp + " " + outfname + " " +str(first) +" "+ str(last) +" "+ str(filt)
    print("extract command:", cmd);
    os.system(cmd)

def plot_most_recent_data(channels, labels = None, downsample = 1, flt=1):
    
    list_of_dat_files = glob.glob('/data/smurf_data/%s/*/outputs/*.dat'%datetime.now().strftime('%Y%m%d'))
    latest_file = max(list_of_dat_files, key=os.path.getctime)
    datafp=latest_file
    maskfname = datafp.split('.')[0]+'_mask.txt'

    mask = numpy.loadtxt(maskfname)

    fig = plt.figure(figsize = (10,8))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)

    ln = 0

    tmin = numpy.inf
    tmax = 0
    fmin = numpy.inf
    fmax = 0

    for chn in channels:
      try:
        channel = numpy.arange(len(mask))[mask == (chn+2*512)][0]
      except IndexError:
        continue
      print(channel)
      print("Extracting channel %d" %channel)
      extract_data(datafp, channel, channel, flt) # executes C command
      print("done  extracting")
      dat = numpy.loadtxt(outfname) # loads text data (numpy)
      print("data shape", dat.shape)
      dat = dat[::downsample,:]
      s = dat.shape
      points = s[0] #how many points in teh dataset
      tm = dat[:,0]
      tmin = min(tm.min(),tmin)
      tmax = max(tm.max(),tmax)
      tmd = numpy.diff(tm)
      tstep = sum(tmd) / len(tmd)
 
      print("min time", min(tm),  "max time", max(tm))
      print(tm[0:10])
      tmp = int(points / pwelch_ratio)
      tmp2 = P.log(tmp) / P.log(2)  # get in powers of two
      tmp3 = int(tmp2)
      np = pow(2,tmp3)
      print("nperseg = ", np)
      print("tstep = ", tstep)

      if labels is None:
        label = "Channel %d" %(chn)
      else:
        label = labels[ln]

      ax1.plot(tm,dat[:,1]-dat[:,1].mean(), label = label, alpha = 0.5)
      #ax1.plot(tm,dat[:,1], label = "Channel %d" %(chn), alpha = 0.5)
      fx, pden = signal.welch(dat[:,1]-dat[:,1].mean(), 1.0/tstep, nperseg = np)
      fmin = min(fmin,fx[1:].min())
      fmax = max(fmax,fx.max())
      #ax2.plot(fx, pden, '-', label = label, alpha = 0.5)
      ax2.plot(fx, numpy.sqrt(pden), '-', label = label, alpha = 0.5)

      ln += 1

      #for j in range(0, lastch+1-firstch):
      #  print("plotting", j)
      #  plt.plot(tm, dat[:,j+1])

    ax1.grid()
    ax1.legend()
    ax1.set_xlim(tmin,tmax)
    ax1.set_xlabel("Time (seconds)")
    #ax1.set_ylabel("Phase ($\Phi_0$)")
    ax1.set_ylabel("Phase (?)")
    ax2.grid()
    ax2.set_xlim(fmin,fmax)
    ax2.set_xlabel("Frequency (Hz)")
    #ax2.set_ylabel("Power Spectral Density ($\Phi_0^2$ / Hz)")
    ax2.set_ylabel("Spectral Density (? / $\sqrt{Hz}$)")
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    fig.tight_layout()
    plt.show()
