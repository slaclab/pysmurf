# Runlike this exec(open("estimate_phase_delay.py").read())
# that way it will use your pysmurf S object.

import pysmurf
import numpy as np
import time
import os

# how to get AMC s/n and fw
def get_amcc_dump_bsi(S,ip='shm-smrf-sp01',slot=None):

    if slot is None:
        # attempt to guess from epics prefix
        import re
        p = re.compile('smurf_server_s([0-9])')
        m = p.match(S.epics_root)
        assert (m is not None),'Unable to determine slot number from epics_root={}'.format(S.epics_root)
        slot=int(m.group(1))

    import subprocess
    result=subprocess.check_output(['amcc_dump_bsi','--all','%s/%d'%(ip,slot)])
    result_string=result.decode('utf-8')

    # E.g.:
    # AMC 0 info: Aux: 01 Ser: 9f0000011d036a70 Type: 0a Ver: C03 BOM: 00 Tag: C03-A01-
    result_dict={}
    patterns={}
    patterns['AMC']=re.compile('AMC\s*([0-1])\s*info:\s*Aux:\s*(\d+)\s*Ser:\s*([a-z0-9]+)\s*Type:\s*([a-z0-9]+)\s*Ver:\s*(C[0-9][0-9])\s*BOM:\s*([0-9]+)\s*Tag:\s*([A-Z0-9a-z\-]+)')
    # E.g.:
    #"FW bld string: 'MicrowaveMuxBpEthGen2: Vivado v2018.3, pc95590 (x86_64), Built Tue Apr 30 13:35:05 PDT 2019 by mdewart'"
    patterns['FW']=re.compile('FW bld string:\s*\'(MicrowaveMuxBpEthGen2):\s*(Vivado)\s*(v2018.3),\s*(pc95590)\s*\((x86_64)\),\s*Built\s*(Tue)\s*(Apr)\s*(30)\s*(13):(35):(05)\s*(PDT)\s*(2019)\s*by\s*(mdewart)\'')

    # E.g.:
    patterns['FWGIThash']=re.compile('GIT hash:\s*([0-9a-z]+)')
    #'     GIT hash: 0000000000000000000000000000000000000000'

    for s in result_string.split('\n'):
        s=s.rstrip().lstrip()
        for key, p in patterns.items():
            m=p.match(s)
            if m is not None:
                if key not in result_dict.keys():
                    result_dict[key]={}

                if key is 'AMC':
                    bay=int(m.group(1))
                    result_dict[key][bay]={}
                    result_dict[key][bay]['Aux']=m.group(2)
                    result_dict[key][bay]['Ser']=m.group(3)
                    result_dict[key][bay]['Type']=m.group(4)
                    result_dict[key][bay]['Ver']=m.group(5)
                    result_dict[key][bay]['BOM']=m.group(6)
                    result_dict[key][bay]['Tag']=m.group(7)

                if key is 'FWGIThash':
                    result_dict[key]['GIThash']=m.group(1)                    

                if key is 'FW':
                    result_dict[key]['FWBranch']=m.group(1)
                    result_dict[key]['BuildSuite']=m.group(2)
                    result_dict[key]['BuildSuiteVersion']=m.group(3)
                    result_dict[key]['BuildPC']=m.group(4)
                    result_dict[key]['BuildArch']=m.group(5)
                    # skipping day spelled out
                    result_dict[key]['Month']=m.group(7)
                    result_dict[key]['Day']=m.group(8)
                    result_dict[key]['Hour']=m.group(9)
                    result_dict[key]['Minute']=m.group(10)
                    result_dict[key]['Second']=m.group(11)
                    result_dict[key]['TimeZone']=m.group(12)
                    result_dict[key]['Year']=m.group(13)
                    result_dict[key]['BuiltBy']=m.group(14)

    return result_dict

def estimate_phase_delay(S,band,n_samples=2**19,make_plot=True,show_plot=True,save_plot=True,save_data=True,n_scan=5,timestamp=None,uc_att=24,dc_att=0,freq_min=-2.5E8,freq_max=2.5E8):
    uc_att0=S.get_att_dc(band)
    dc_att0=S.get_att_uc(band)
    S.set_att_uc(band,uc_att,write_log=True)
    S.set_att_dc(band,dc_att,write_log=True)

    # only loop over dsp subbands in requested frequency range (to
    # save time)
    n_subbands = S.get_number_sub_bands(band)
    digitizer_frequency_mhz = S.get_digitizer_frequency_mhz(band)
    subband_half_width_mhz = digitizer_frequency_mhz/\
                         n_subbands
    band_center_mhz=S.get_band_center_mhz(band)
    subbands,subband_centers=S.get_subband_centers(band)
    subband_freq_min=-subband_half_width_mhz/2.
    subband_freq_max=subband_half_width_mhz/2.
    dsp_subbands=[]
    for sb,sbc in zip(subbands,subband_centers):
        # ignore unprocessed sub-bands
        if sb not in range(13,115):
            continue
        lower_sb_freq=sbc+subband_freq_min
        upper_sb_freq=sbc+subband_freq_max
        if lower_sb_freq>=(freq_min/1.e6-subband_half_width_mhz) and upper_sb_freq<=(freq_max/1.e6+subband_half_width_mhz):
            dsp_subbands.append(sb)
    
    # For some reason, pyrogue flips out if you try to set refPhaseDelay
    # to zero in 071150b0.  This allows an offset ; the offset just gets
    # subtracted off the delay measurement with DSP after it's made.
    refPhaseDelay0=1
    refPhaseDelayFine0=0

    if timestamp is None:
        timestamp=S.get_timestamp()

    if make_plot:
        import matplotlib.pyplot as plt
        if show_plot:
            plt.ion()
        else:
            plt.ioff()
    
    load_full_band_resp=False
    fbr_path='/data/smurf_data/20190702/1562052474/outputs'
    fbr_ctime=1562052477

    load_find_freq=False
    ff_path='/data/smurf_data/20190702/1562052474/outputs'
    ff_ctime=1562052881

    load_find_freq_check=False
    ff_corr_path='/data/smurf_data/20190702/1562052474/outputs'
    ff_corr_ctime=1562053274

    bay=int(band/4)
    amcc_dump_bsi_dict=get_amcc_dump_bsi(S)
    amc_dict=amcc_dump_bsi_dict['AMC'][bay]
    amc_sn=amc_dict['Tag']+amc_dict['Ver']

    fw_abbrev_sha=amcc_dump_bsi_dict['FWGIThash']['GIThash'][:7]
    #fw_dict=amcc_dump_bsi_dict['FW']
    #fw_build_date="Built {} {} {}:{}:{} {} {} by {}".format(fw_dict['Month'],
    #                                                        fw_dict['Day'],
    #                                                        fw_dict['Hour'],
    #                                                        fw_dict['Minute'],
    #                                                        fw_dict['Second'],
    #                                                        fw_dict['TimeZone'],
    #                                                        fw_dict['Year'],
    #                                                        fw_dict['BuiltBy'])

    # some special cases
    #if fw_build_date=='Built Apr 30 13:35:05 PDT 2019 by mdewart':
    #    fw_abbrev_sha='0eea5630'

    print('amc_sn={}'.format(amc_sn))
    print('fw_abbrev_sha={}'.format(fw_abbrev_sha))
    #print('fw_build_date={}'.format(fw_build_date))

    S.band_off(band)
    S.flux_ramp_off()

    freq_cable=None
    resp_cable=None
    if load_full_band_resp:
        S.log('Loading full band resp data')
        fbr_freq_file=os.path.join(fbr_path,'%d_freq_full_band_resp.txt'%fbr_ctime)
        fbr_real_resp_file=os.path.join(fbr_path,'%d_real_full_band_resp.txt'%fbr_ctime)
        fbr_complex_resp_file=os.path.join(fbr_path,'%d_imag_full_band_resp.txt'%fbr_ctime)

        freq_cable = np.loadtxt(fbr_freq_file)
        real_resp_cable = np.loadtxt(fbr_real_resp_file)
        complex_resp_cable = np.loadtxt(fbr_complex_resp_file)
        resp_cable = real_resp_cable + 1j*complex_resp_cable
    else:
        S.log('Running full band resp')
        freq_cable, resp_cable = S.full_band_resp(band, n_samples=n_samples, \
                                                  make_plot=make_plot, \
                                                  save_data=save_data, \
                                                  n_scan=n_scan)

    idx_cable = np.where( (freq_cable > freq_min) & (freq_cable < freq_max) )

    cable_z = np.polyfit(freq_cable[idx_cable], np.unwrap(np.angle(resp_cable[idx_cable])), 1)
    cable_p = np.poly1d(cable_z)
    cable_delay_us=np.abs(1.e6*cable_z[0]/2/np.pi)

    freq_cable_subset=freq_cable[idx_cable]
    resp_cable_subset=resp_cable[idx_cable]
    #### done measuring cable delay

    #### start measuring dsp delay (cable+processing)
    # Zero refPhaseDelay and refPhaseDelayFine to get uncorrected phase
    # delay.
    # max is 7
    S.set_ref_phase_delay(band,refPhaseDelay0)
    # max is 255
    S.set_ref_phase_delay_fine(band,refPhaseDelayFine0)

    freq_dsp=None
    resp_dsp=None
    if load_find_freq:
        S.log('Loading DSP frequency sweep data')
        ff_freq_file=os.path.join(ff_path,'%d_amp_sweep_freq.txt'%ff_ctime)
        ff_resp_file=os.path.join(ff_path,'%d_amp_sweep_resp.txt'%ff_ctime)

        freq_dsp=np.loadtxt(ff_freq_file)
        resp_dsp=np.loadtxt(ff_resp_file,dtype='complex')
    else:
        S.log('Running find_freq')
        freq_dsp,resp_dsp=S.find_freq(band,subband=dsp_subbands)
        ## not really faster if reduce n_step or n_read...somehow.
        #freq_dsp,resp_dsp=S.full_band_ampl_sweep(band, subband=dsp_subbands, drive=drive, n_read=2, n_step=n_step)

    # only preserve data in the subband half width
    freq_dsp_subset=[]
    resp_dsp_subset=[]
    for sb,sbc in zip(subbands,subband_centers):
        freq_subband=freq_dsp[sb]-sbc
        idx = np.where( ( freq_subband > subband_freq_min ) & (freq_subband < subband_freq_max) )
        freq_dsp_subset.extend(freq_dsp[sb][idx])
        resp_dsp_subset.extend(resp_dsp[sb][idx])

    freq_dsp_subset=np.array(freq_dsp_subset)
    resp_dsp_subset=np.array(resp_dsp_subset)

    idx_dsp = np.where( (freq_dsp_subset > freq_min) & (freq_dsp_subset < freq_max) )    

    # restrict to requested frequencies only
    freq_dsp_subset=freq_dsp_subset[idx_dsp]
    resp_dsp_subset=resp_dsp_subset[idx_dsp]
    
    # to Hz
    freq_dsp_subset=(freq_dsp_subset)*1.0E6

    # fit
    dsp_z = np.polyfit(freq_dsp_subset, np.unwrap(np.angle(resp_dsp_subset)), 1)
    dsp_p = np.poly1d(dsp_z)
    dsp_delay_us=np.abs(1.e6*dsp_z[0]/2/np.pi)

    # if refPhaseDelay0 or refPhaseDelayFine0 aren't zero, must add into
    # delay here 
    dsp_delay_us+=refPhaseDelay0/(subband_half_width_mhz/2.)
    dsp_delay_us-=refPhaseDelayFine0/(digitizer_frequency_mhz/2)

    ## compute refPhaseDelay and refPhaseDelayFine
    refPhaseDelay=int(np.ceil(dsp_delay_us*(subband_half_width_mhz/2.)))
    refPhaseDelayFine=int(np.round((digitizer_frequency_mhz/2/(subband_half_width_mhz/2.)*(refPhaseDelay-dsp_delay_us*(subband_half_width_mhz/2.)))))
    processing_delay_us=dsp_delay_us-cable_delay_us

    print('-------------------------------------------------------')
    print('Estimated refPhaseDelay={}'.format(refPhaseDelay))
    print('Estimated refPhaseDelayFine={}'.format(refPhaseDelayFine))
    print('Estimated processing_delay_us={}'.format(processing_delay_us))
    print('-------------------------------------------------------')

    #### done measuring dsp delay (cable+processing)

    #### start measuring total (DSP) delay with estimated correction applied
    # Zero refPhaseDelay and refPhaseDelayFine to get uncorrected phase
    # delay.
    # max is 7
    S.set_ref_phase_delay(band,refPhaseDelay)
    # max is 255
    S.set_ref_phase_delay_fine(band,refPhaseDelayFine)

    freq_dsp_corr=None
    resp_dsp_corr=None
    if load_find_freq_check:
        S.log('Loading delay-corrected DSP frequency sweep data')
        ff_corr_freq_file=os.path.join(ff_corr_path,'%d_amp_sweep_freq.txt'%ff_corr_ctime)
        ff_corr_resp_file=os.path.join(ff_corr_path,'%d_amp_sweep_resp.txt'%ff_corr_ctime)

        freq_dsp_corr=np.loadtxt(ff_corr_freq_file)
        resp_dsp_corr=np.loadtxt(ff_corr_resp_file,dtype='complex')
    else:
        S.log('Running find_freq')
        freq_dsp_corr,resp_dsp_corr=S.find_freq(band,dsp_subbands)

    freq_dsp_corr_subset=[]
    resp_dsp_corr_subset=[]
    for sb,sbc in zip(subbands,subband_centers):
        freq_subband=freq_dsp_corr[sb]-sbc
        idx = np.where( ( freq_subband > subband_freq_min ) & (freq_subband < subband_freq_max) )
        freq_dsp_corr_subset.extend(freq_dsp_corr[sb][idx])
        resp_dsp_corr_subset.extend(resp_dsp_corr[sb][idx])

    freq_dsp_corr_subset=np.array(freq_dsp_corr_subset)
    resp_dsp_corr_subset=np.array(resp_dsp_corr_subset)

    # restrict to requested frequency subset
    idx_dsp_corr = np.where( (freq_dsp_corr_subset > freq_min) & (freq_dsp_corr_subset < freq_max) )    

    # restrict to requested frequencies only
    freq_dsp_corr_subset=freq_dsp_corr_subset[idx_dsp_corr]
    resp_dsp_corr_subset=resp_dsp_corr_subset[idx_dsp_corr]    
    
    # to Hz
    freq_dsp_corr_subset=(freq_dsp_corr_subset)*1.0E6

    # fit
    dsp_corr_z = np.polyfit(freq_dsp_corr_subset, np.unwrap(np.angle(resp_dsp_corr_subset)), 1)
    dsp_corr_p = np.poly1d(dsp_corr_z)
    dsp_corr_delay_us=np.abs(1.e6*dsp_corr_z[0]/2/np.pi)
    #### done measuring total (DSP) delay with estimated correction applied

    # plot unwraped phase in top panel, subtracted in bottom

    fig, ax = plt.subplots(3, figsize=(6,7.5), sharex=True)

    f_cable_plot = (freq_cable_subset) / 1.0E6
    cable_phase = np.unwrap(np.angle(resp_cable_subset))

    f_dsp_plot = (freq_dsp_subset) / 1.0E6
    dsp_phase = np.unwrap(np.angle(resp_dsp_subset))

    f_dsp_corr_plot = (freq_dsp_corr_subset) / 1.0E6
    dsp_corr_phase = np.unwrap(np.angle(resp_dsp_corr_subset))

    ax[0].set_title('AMC {}, Bay {}, Band {} Cable Delay'.format(amc_sn,bay,band))
    ax[0].plot(f_cable_plot,cable_phase,label='Cable (full_band_resp)',c='g',lw=3)
    ax[0].plot(f_cable_plot,cable_p(f_cable_plot*1.0E6),'m--',label='Cable delay fit',lw=3)

    ax[1].set_title('AMC {}, Bay {}, Band {} DSP Delay'.format(amc_sn,bay,band))
    ax[1].plot(f_dsp_plot,dsp_phase,label='DSP (find_freq)',c='c',lw=3)
    ax[1].plot(f_dsp_plot,dsp_p(f_dsp_plot*1.0E6),c='orange',ls='--',label='DSP delay fit',lw=3)

    ax[0].set_ylabel("Phase [rad]")
    ax[0].set_xlabel('Frequency offset from band center [MHz]')

    ax[1].set_ylabel("Phase [rad]")
    ax[1].set_xlabel('Frequency offset from band center [MHz]')

    ax[0].legend(loc='lower left',fontsize=8)
    ax[1].legend(loc='lower left',fontsize=8)

    bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)        
    ax[0].text(.97, .90, 'cable delay={:.5f} us'.format(cable_delay_us),
               transform=ax[0].transAxes, fontsize=10,
               bbox=bbox,horizontalalignment='right')

    ax[1].text(.97, .90, 'dsp delay={:.5f} us'.format(dsp_delay_us),
               transform=ax[1].transAxes, fontsize=10,
               bbox=bbox,horizontalalignment='right')

    cable_residuals=cable_phase-(cable_p(f_cable_plot*1.0E6))
    ax[2].plot(f_cable_plot,cable_residuals-np.median(cable_residuals),label='Cable (full_band_resp)',c='g')
    dsp_residuals=dsp_phase-(dsp_p(f_dsp_plot*1.0E6))
    ax[2].plot(f_dsp_plot,dsp_residuals-np.median(dsp_residuals),label='DSP (find_freq)',c='c')
    ax[2].plot(f_dsp_corr_plot,dsp_corr_phase-np.median(dsp_corr_phase),label='DSP corrected (find_freq)',c='m')
    ax[2].set_title('AMC {}, Bay {}, Band {} Residuals'.format(amc_sn,bay,band))
    ax[2].set_ylabel("Residual [rad]")
    ax[2].set_xlabel('Frequency offset from band center [MHz]')
    ax[2].set_ylim([-5,5])

    ax[2].text(.97, .92, 'refPhaseDelay={}'.format(refPhaseDelay),
               transform=ax[2].transAxes, fontsize=8,
               bbox=bbox,horizontalalignment='right')
    ax[2].text(.97, .84, 'refPhaseDelayFine={}'.format(refPhaseDelayFine),
               transform=ax[2].transAxes, fontsize=8,
               bbox=bbox,horizontalalignment='right')
    ax[2].text(.97, .76, 'processing delay={:.5f} us (fw={})'.format(processing_delay_us,fw_abbrev_sha),
               transform=ax[2].transAxes, fontsize=8,
               bbox=bbox,horizontalalignment='right')
    ax[2].text(.97, .68, 'delay post-correction={:.3f} ns'.format(dsp_corr_delay_us*1000.),
               transform=ax[2].transAxes, fontsize=8,
               bbox=bbox,horizontalalignment='right')

    ax[2].legend(loc='upper left',fontsize=8)

    plt.tight_layout()

    if save_plot:
        save_name = '{}_b{}_delay.png'.format(timestamp,band)
        plt.savefig(os.path.join(S.plot_dir, save_name),
                    bbox_inches='tight')
        if not show_plot:
            plt.close()

    S.set_att_uc(band,uc_att0,write_log=True)
    S.set_att_dc(band,dc_att0,write_log=True)        
