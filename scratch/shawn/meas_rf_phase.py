band=0
f,resp=S.find_freq(band,make_plot=True,show_plot=True)

fs=[]
resps=[]
sbs,sbcs=S.get_subband_centers(band)
bc=S.get_band_center_mhz(band)
for sb,sbc in zip(sbs,sbcs):
    if len(np.nonzero(f[sb])[0]):
        fs.extend(f[sb]+bc)
        resps.extend(resp[sb])

rfphase=np.unwrap([np.math.atan2(np.imag(r),np.real(r))+np.pi for r in resps])

fitskipids=10
a,b = np.polyfit(fs[fitskipids:-fitskipids],rfphase[fitskipids:-fitskipids],1)

plt.figure()

plt.plot(fs,rfphase)
plt.plot(fs,a*np.array(fs)+b,'r--')
plt.xlabel('Frequency (GHz)')
plt.ylabel('Unwrapped RF Phase (rad)')

plt.figure()

plt.plot(fs,rfphase-(a*np.array(fs)+b))
#plt.xlabel('Frequency (GHz)')
#plt.ylabel('Unwrapped RF Phase (rad)')
    
