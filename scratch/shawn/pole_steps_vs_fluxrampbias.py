import sys
import numpy as np
nkernel=10000
h = np.ones(nkernel)/nkernel

f2,df2,sync2=S.decode_single_channel('/data/smurf_data/pole/20191114/1573707874/outputs/1573708182.dat')  
f1,df1,sync1=S.decode_single_channel('/data/smurf_data/pole/20191114/1573707874/outputs/1573707959.dat')  
f3,df3,sync3=S.decode_single_channel('/data/smurf_data/pole/20191114/1573707874/outputs/1573708642.dat')

f1_filt=np.convolve(h, df1-np.median(df1))+np.median(df1)+np.median(f1)
f2_filt=np.convolve(h, df2-np.median(df2))+np.median(df2)+np.median(f2)
f3_filt=np.convolve(h, df3-np.median(df3))+np.median(df3)+np.median(f3)

f1_filt=f1_filt[100:]
f2_filt=f2_filt[100:]
f3_filt=f3_filt[100:]

print(len(f1_filt))
print(len(f2_filt))
print(len(f3_filt))

t1=np.array(range(len(f1_filt)))/(2.4e6)
t2=np.array(range(len(f2_filt)))/(2.4e6)
t3=np.array(range(len(f3_filt)))/(2.4e6)

#plt.plot(t1,df1-np.median(df1_filt),label='ff=0')  
#plt.plot(t2,df2-np.median(df2_filt),label='ff=0.05')  
#plt.plot(t3,df3-np.median(df3_filt),label='ff=0.18')

plt.plot(t1,f1_filt,label='ff=0')
plt.plot(t2,f2_filt,label='ff=0.05')
#plt.plot(t3,f3_filt,label='ff=0.18')  

plt.legend()  
plt.title('20191114/1573707874/outputs/1573707933_fsnt.dat')
plt.show()
#plt.savefig('/home/cryo/tmp.png')
