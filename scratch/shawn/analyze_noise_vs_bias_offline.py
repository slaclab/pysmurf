import os
import matplotlib.pyplot as plt
import pysmurf.client

S = pysmurf.client.SmurfControl(offline=True)

datafile_path = '/mnt/d/Google Drive/Work/uMux/1582161944/outputs'
iv_path = '/mnt/d/Google Drive/Work/uMux/1582078535/outputs/1582087852_iv.npy'

datafiles = ['1582162928.dat','1582163631.dat','1582164334.dat','1582165036.dat',
    '1582165740.dat','1582166443.dat','1582167146.dat','1582167850.dat',
    '1582168553.dat','1582169256.dat','1582169959.dat','1582170662.dat',
    '1582171365.dat','1582172068.dat','1582172771.dat','1582173474.dat',
    '1582174177.dat','1582174880.dat','1582175583.dat','1582176286.dat',
    '1582176989.dat','1582177693.dat','1582178396.dat','1582179098.dat',
    '1582179801.dat','1582180504.dat','1582181207.dat','1582181910.dat',
    '1582182613.dat','1582183316.dat','1582184019.dat','1582184722.dat',
    '1582185425.dat','1582186128.dat','1582186831.dat','1582187533.dat',
    '1582188236.dat','1582188939.dat','1582189642.dat','1582190345.dat',
    '1582191047.dat','1582191750.dat','1582192453.dat','1582193155.dat',
    '1582193858.dat','1582194561.dat','1582195264.dat','1582195967.dat',
    '1582196670.dat','1582197372.dat','1582198075.dat','1582198778.dat',
    '1582199480.dat','1582200183.dat','1582200886.dat','1582201589.dat',
    '1582202292.dat','1582202995.dat','1582203697.dat','1582204400.dat',
    '1582205104.dat','1582205806.dat','1582206509.dat','1582207212.dat',
    '1582207915.dat','1582208618.dat','1582209321.dat','1582210024.dat',
    '1582210726.dat','1582211429.dat','1582212131.dat','1582212835.dat',
    '1582213537.dat','1582214241.dat','1582214943.dat','1582215646.dat',
    '1582216349.dat']

datafiles = [os.path.join(datafile_path,df) for df in datafiles]

var_range = [15,14.9,14.8,14.7,14.6,14.5,14.4,14.3,14.2,14.1,14.,13.9,13.8,
    13.7,13.6,13.5,13.4,13.3,13.2,13.1,13.,12.9,12.8,12.7,12.6,12.5,12.4,12.3,
    12.2,12.1,12.,11.9,11.8,11.7,11.6,11.5,11.4,11.3,11.2,11.1,11.,10.9,10.8,
    10.7,10.6,10.5,10.4,10.3,10.2,10.1,10.,9.9,9.8,9.7,9.6,9.5,9.4,9.3,9.2,9.1,
    9.,8.9,8.8,8.7,8.6,8.5,8.4,8.3,8.2,8.1,8.,7.9,7.8,7.7,7.6,7.5,7.4]

skip = 4
datafiles=datafiles[::skip]
var_range=var_range[::skip]

channel=None
band=1
bias_group=1
nperseg=8192
detrend='constant'
fs=None
save_plot=True
show_plot=False
timestamp=1582161944
psd_ylim=[10.0,1000.0]
make_timestream_plot=True
xlabel_override=None
unit_override=None
R_sh = 738E-6
high_current_mode=False

S.pA_per_phi0 = 9.0E6
S.high_low_current_ratio = 10.15  # CHECK ME
S.plot_dir = datafile_path.replace('outputs', 'plots')

S.analyze_noise_vs_bias(var_range, datafiles,  channel=channel, band=band,
    bias_group = bias_group, nperseg=nperseg, detrend=detrend, fs=fs,
    save_plot=save_plot, show_plot=show_plot, data_timestamp=timestamp,
    psd_ylim=psd_ylim, make_timestream_plot=make_timestream_plot,
    xlabel_override=xlabel_override, unit_override=unit_override, R_sh=R_sh,
    iv_data_filename=iv_path, high_current_mode=high_current_mode)

