import sys
import matplotlib.pylab as plt

ctime=1570793119
#t,p,m=S.read_stream_data('/data/smurf_data/pole/20191011/1570792380/outputs/1570793119.dat')
#t=t/1.e9

band_chan_idx=[(2,ch,m[2,ch]) for ch in np.where(m[2]!=-1)[0]]+[(3,ch,m[3,ch]) for ch in np.where(m[3]!=-1)[0]]

t0=np.min(t)

spans=[]

for (band,chan,idx) in band_chan_idx:
    print('band=%d\tchan=%d'%(band,chan))

    plt.figure(figsize=(12,6))
    plt.plot(t-t0,p[idx])
    plt.xlim(np.min(t-t0),np.max(t-t0))
    maxp=np.max(p[idx])
    minp=np.min(p[idx])
    span=maxp-minp
    if span<1e-5:
        continue
    spans.append(span)
    
    plt.ylim(minp-span/4,maxp+span/4)
    
    plt.plot(plt.gca().get_xlim(),[maxp,maxp],'c--')
    plt.plot(plt.gca().get_xlim(),[minp,minp],'c--')
    
    ax=plt.gca()
    plt.text(0.0175,1.05,'p2p = %0.4e rad'%span,horizontalalignment='left',
             verticalalignment='top',transform=ax.transAxes,fontsize=10, color='c')

    plt.text(1-0.0175,1.05,'xtalk (%% assuming 17.48) = %0.2f %%'%(100.*(span/17.48)),horizontalalignment='right',
             verticalalignment='top',transform=ax.transAxes,fontsize=10, color='c')    

    plt.title('%d : band %d, chan %d'%(ctime,band,chan))
    
    # draw bias group boundaries
    for (bg,x0,x1) in [(1,0,47.4),
                       (2,47.4*1,47.4*2),
                       (3,47.4*2,47.4*3),
                       (4,47.4*3,47.4*4),
                       (5,47.4*4,47.4*5),
                       (7,47.4*5,47.4*6)]:                       
        plt.plot([x0,x0],plt.gca().get_ylim(),c='gray',ls='--',lw=2,alpha=0.5)
        plt.plot([x1,x1],plt.gca().get_ylim(),c='gray',ls='--',lw=2,alpha=0.5)

        plt.text((x0+x1)/2,maxp+span/8,'%d'%bg,horizontalalignment='center',
                 verticalalignment='center',fontsize=25, color='r')    
        
    plt.ylabel('Phase (rad)',fontsize=18)
    plt.xlabel('Time (sec)',fontsize=18)

    plt.savefig('pole_bias_tickle/%d_b%d_ch%d_tickle.png'%(ctime,band,chan))
    #plt.show()
    plt.close()
