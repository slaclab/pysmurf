datafiles=['/data/smurf_data/pole/20191107/1573157844/outputs/1573158573.dat','/data/smurf_data/pole/20191107/1573157844/outputs/1573159294.dat','/data/smurf_data/pole/20191107/1573157844/outputs/1573160008.dat','/data/smurf_data/pole/20191107/1573157844/outputs/1573160739.dat','/data/smurf_data/pole/20191107/1573157844/outputs/1573161483.dat','/data/smurf_data/pole/20191107/1573157844/outputs/1573162244.dat']
band=2
channel=np.array([  12, 290, 76, 354, 269, 1, 9, 366, 53 ])

#band=3
#channel=np.array([ 211, 135, 157, 439, 387, 187, 399])

bias=[0,0,-3,3,6,9]
meas_time=300.
freq_range_summary=(5,10)
fs=180

psd_dir='/tmp/psd/'

bias_group=-1
high_current_mode=False
overbias_voltage=19.9
analyze=True
psd_ylim=(10,5e3)
make_timestream_plot=True

## start noise_vs
var='bias',
var_range=bias

timestamp=S.get_timestamp()
## end noise_vs

nperseg=2**13
detrend='constant'
save_plot=True
show_plot=False
#data_timestamp=None
gcp_mode = True
smooth_len=15
show_legend=True
#freq_range_summary=None
#R_sh=None
#high_current_mode=True
#iv_data_filename=None
unit_override='(dB)'
xlabel_override='Relative tone power'
S.analyze_noise_vs_bias(var_range, datafiles,  channel=channel, 
                        band=band, 
                        bias_group = bias_group,
                        nperseg=nperseg, detrend=detrend, fs=fs, 
                        save_plot=True, show_plot=show_plot, 
                        data_timestamp=timestamp, 
                        gcp_mode=gcp_mode,psd_ylim=psd_ylim,
                        freq_range_summary=freq_range_summary,                        
                        make_timestream_plot=make_timestream_plot,
                        xlabel_override=xlabel_override, 
                        unit_override=unit_override,plot_fit=False)
