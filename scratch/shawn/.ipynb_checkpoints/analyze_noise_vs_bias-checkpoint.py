datadir='/data/smurf_data/20191007/1570485127/outputs'
datafiles=['1570491245.dat.part_00000','1570491371.dat.part_00000','1570491497.dat.part_00000','1570491623.dat.part_00000','1570491749.dat.part_00000','1570491875.dat.part_00000','1570492001.dat.part_00000','1570492126.dat.part_00000','1570492252.dat.part_00000','1570492378.dat.part_00000','1570492503.dat.part_00000','1570492629.dat.part_00000','1570492755.dat.part_00000','1570492881.dat.part_00000','1570493007.dat.part_00000','1570493133.dat.part_00000','1570493259.dat.part_00000']
channel=np.array([  0,   8,  11,  29,  61,  63,  64,  75, 103, 125, 127, 128, 189, \
                    256, 263, 267, 279, 285, 301, 307, 319, 320, 331, 343, 349, 365, \
                    371, 383, 395, 407, 429, 435])
bias=[15., 13.6, 12.3, 10.9, 9.5, 8.2, 6.8, 5.45, 4.8, 4.2, 3.6, 3.0, 2.4, 1.8, 1.2, 0.6, 0]
meas_time=90.
freq_range_summary=(10,100)
fs=4e3
plot_fit=False

#datadir='/data/smurf_data/20190926/1569476102/outputs/'
#datafiles=['1569565354.dat.part_00000','1569565455.dat.part_00000','1569565557.dat.part_00000','1569565657.dat.part_00000','1569565759.dat.part_00000','1569565860.dat.part_00000','1569565962.dat.part_00000','1569566063.dat.part_00000','1569566164.dat.part_00000']
#bias=[19.9,13.0,12.3,11.5,10.8,10.1,8.86,7.99,6.77]
#channel=np.array([  0,   8,  11,  23,  29,  39,  61,  63,  64,  75,  87, 103, 125, \
#                    127, 128, 189, 256, 263, 267, 279, 285, 301, 303, 307, 319, 320, \
#                    331, 343, 349, 365, 371, 383, 395, 407, 429, 435])
#meas_time=60.
#fs=None

psd_dir=os.path.join(datadir,'psd/')

#S.noise_vs_bias(band=2,bias_group=7,bias=[19.9,13.0,12.3,11.5,10.8,10.1,8.86,7.99,6.77],high_current_mode=False,overbias_voltage=19.9,meas_time=60.,analyze=True,channel=S.which_on(2),psd_ylim=(1,1e3),make_timestream_plot=True)
band=2
bias_group=7
high_current_mode=False
overbias_voltage=19.9
analyze=True
psd_ylim=(10,5e3)
make_timestream_plot=False

## start noise_vs
var='bias',
var_range=bias

timestamp=S.get_timestamp()
## end noise_vs

# append datadir
datafiles=[os.path.join(datadir,d) for d in datafiles]

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
unit_override=None
xlabel_override=None
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
                        unit_override=unit_override,plot_fit=plot_fit)
