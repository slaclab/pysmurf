#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf util module - SmurfUtilMixin class
#-----------------------------------------------------------------------------
# File       : pysmurf/util/smurf_util.py
# Created    : 2018-08-29
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
import numpy as np
from pysmurf.client.base import SmurfBase
from pysmurf.client.command.sync_group import SyncGroup as SyncGroup
import time
import os
from scipy import signal
import glob
import matplotlib.pyplot as plt
from matplotlib import gridspec
from contextlib import contextmanager
# for hardware logging
import threading
from pysmurf.client.util.SmurfFileReader import SmurfStreamReader
from pysmurf.client.util.pub import set_action



class SmurfUtilMixin(SmurfBase):

    @set_action()
    def take_debug_data(self, band, channel=None, nsamp=2**19, filename=None,
            IQstream=1, single_channel_readout=1, debug=False, write_log=True):
        """ Takes raw debugging data

        Args
        ----
        band : int
            The band to take data on.
        channel : int or None, optional, default None
            The channel to take debug data on in single_channel_mode.
        nsamp : int, optional, default 2**19
            The number of samples to take.
        filename : str or None, optional, default None
            The name of the file to save to.
        IQstream : int, optional, default 1
            Whether to take the raw IQ stream.
        single_channel_readout : int, optional, default 1
            Whether to look at one channel.
        debug : bool, optional, default False
            Whether to take data in debug mode.
        write_log : bool, optional, default True
            Whether to write low-level commands to the log file.

        Returns
        -------
        f : float array
            The frequency response.
        df : float array
            The frequency error.
        sync : float array
            The sync count.
        """
        # Set proper single channel readout
        if channel is not None:
            if single_channel_readout == 1:
                self.set_single_channel_readout(band, 1)
                self.set_single_channel_readout_opt2(band, 0)
            elif single_channel_readout == 2:
                self.set_single_channel_readout(band, 0)
                self.set_single_channel_readout_opt2(band, 1)
            else:
                self.log('single_channel_readout must be 1 or 2',
                    self.LOG_ERROR)
                raise ValueError('single_channel_readout must be 1 or 2')
            self.set_readout_channel_select(band, channel, write_log=write_log)
        else: # exit single channel otherwise
            self.set_single_channel_readout(band, 0, write_log=write_log)
            self.set_single_channel_readout_opt2(band, 0, write_log=write_log)

        # Set IQstream
        if IQstream==1:
            self.set_iq_stream_enable(band, 1)
        else:
            self.set_iq_stream_enable(band, 0)

        # set filename
        if filename is not None:
            data_filename = os.path.join(self.output_dir, filename+'.dat')
            self.log(f'Writing to file : {data_filename}',
                self.LOG_USER)
        else:
            timestamp = self.get_timestamp()
            data_filename = os.path.join(self.output_dir, timestamp+'.dat')
            self.log(f'Writing to file : {data_filename}',
                self.LOG_USER)

        dtype = 'debug'
        dchannel = 0 # I don't really know what this means and I'm sorry -CY
        self.setup_daq_mux(dtype, dchannel, nsamp, band=band, debug=debug)
        self.log('Data acquisition in progress...', self.LOG_USER)
        char_array = [ord(c) for c in data_filename] # convert to ascii
        write_data = np.zeros(300, dtype=int)
        for j in np.arange(len(char_array)):
            write_data[j] = char_array[j]

        self.set_streamdatawriter_datafile(write_data) # write this

        #self.set_streamdatawriter_open('True') # str and not bool
        self.set_streamdatawriter_open(True)

        bay=self.band_to_bay(band)
        self.set_trigger_daq(bay, 1, write_log=True) # this seems to = TriggerDM

        time.sleep(.1) # maybe unnecessary

        done=False
        while not done:
            done=True
            for k in range(2):
                # see pysmurf issue 161.  This call is no longer used,
                # and causes take_debug_data to crash if
                # get_waveform_wr_addr is called before the
                # acquisition completes.
                #wr_addr = self.get_waveform_wr_addr(bay, engine=0)
                empty = self.get_waveform_empty(bay, engine=k)
                if not empty:
                    done=False
            time.sleep(1)

        time.sleep(.25) # do we need all of these?

        # Close the streamdatawriter
        self.set_streamdatawriter_close(True)

        self.log('Done taking data', self.LOG_USER)

        if single_channel_readout > 0:
            f, df, sync = self.decode_single_channel(data_filename)
        else:
            f, df, sync = self.decode_data(data_filename)

        return f, df, sync

    # the JesdWatchdog will check if an instance of the JesdWatchdog is already
    # running and kill itself if there is
    def start_jesd_watchdog(self):
        import pysmurf.client.watchdog.JesdWatchdog as JesdWatchdog
        import subprocess
        import sys
        subprocess.Popen([sys.executable,JesdWatchdog.__file__])

    # Shawn needs to make this better and add documentation.
    @set_action()
    def estimate_phase_delay(self, band, n_samples=2**19, make_plot=True,
            show_plot=True, save_plot=True, save_data=True, n_scan=5,
            timestamp=None, uc_att=24, dc_att=0, freq_min=-2.5E8, freq_max=2.5E8):

        # For some reason, pyrogue flips out if you try to set refPhaseDelay
        # to zero in 071150b0.  This allows an offset ; the offset just gets
        # subtracted off the delay measurement with DSP after it's made.
        refPhaseDelay0=1
        refPhaseDelayFine0=0

        uc_att0=self.get_att_dc(band)
        dc_att0=self.get_att_uc(band)
        self.set_att_uc(band,uc_att, write_log=True)
        self.set_att_dc(band,dc_att, write_log=True)

        # only loop over dsp subbands in requested frequency range (to
        # save time)
        n_subbands = self.get_number_sub_bands(band)
        digitizer_frequency_mhz = self.get_digitizer_frequency_mhz(band)
        subband_half_width_mhz = digitizer_frequency_mhz/\
            n_subbands
        subbands,subband_centers=self.get_subband_centers(band)
        subband_freq_min=-subband_half_width_mhz/2.
        subband_freq_max=subband_half_width_mhz/2.
        dsp_subbands=[]
        for sb,sbc in zip(subbands,subband_centers):
            # ignore unprocessed sub-bands
            if sb not in subbands:
                continue
            lower_sb_freq=sbc+subband_freq_min
            upper_sb_freq=sbc+subband_freq_max
            if lower_sb_freq>=(freq_min/1.e6-subband_half_width_mhz) and \
                    upper_sb_freq<=(freq_max/1.e6+subband_half_width_mhz):
                dsp_subbands.append(sb)

        if timestamp is None:
            timestamp = self.get_timestamp()

        if make_plot:
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

        fw_abbrev_sha=self.get_fpga_git_hash_short()

        self.band_off(band)
        self.flux_ramp_off()

        freq_cable=None
        resp_cable=None
        if load_full_band_resp:
            self.log('Loading full band resp data')
            fbr_freq_file=(
                os.path.join(fbr_path,
                             f'{fbr_ctime}_freq_full_band_resp.txt'))
            fbr_real_resp_file=(
                os.path.join(fbr_path,
                             f'{fbr_ctime}_real_full_band_resp.txt'))
            fbr_complex_resp_file=(
                os.path.join(fbr_path,
                             f'{fbr_ctime}_imag_full_band_resp.txt'))

            freq_cable = np.loadtxt(fbr_freq_file)
            real_resp_cable = np.loadtxt(fbr_real_resp_file)
            complex_resp_cable = np.loadtxt(fbr_complex_resp_file)
            resp_cable = real_resp_cable + 1j*complex_resp_cable
        else:
            self.log('Running full band resp')
            freq_cable, resp_cable = self.full_band_resp(band, n_samples=n_samples,
                make_plot=make_plot,
                save_data=save_data,
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
        self.set_ref_phase_delay(band,refPhaseDelay0)
        # max is 255
        self.set_ref_phase_delay_fine(band,refPhaseDelayFine0)

        freq_dsp=None
        resp_dsp=None
        if load_find_freq:
            self.log('Loading DSP frequency sweep data')
            ff_freq_file=(
                os.path.join(ff_path,
                             f'{ff_ctime}_amp_sweep_freq.txt'))
            ff_resp_file=(
                os.path.join(ff_path,
                             f'{ff_ctime}_amp_sweep_resp.txt'))

            freq_dsp=np.loadtxt(ff_freq_file)
            resp_dsp=np.loadtxt(ff_resp_file,dtype='complex')
        else:
            self.log('Running find_freq')
            freq_dsp,resp_dsp=self.find_freq(band,subband=dsp_subbands)
            ## not really faster if reduce n_step or n_read...somehow.
            #freq_dsp,resp_dsp=self.full_band_ampl_sweep(band,
            # subband=dsp_subbands, drive=drive, n_read=2, n_step=n_step)

        # only preserve data in the subband half width
        freq_dsp_subset=[]
        resp_dsp_subset=[]
        for sb,sbc in zip(subbands,subband_centers):
            freq_subband=freq_dsp[sb]-sbc
            idx = np.where( ( freq_subband > subband_freq_min ) &
                (freq_subband < subband_freq_max) )
            freq_dsp_subset.extend(freq_dsp[sb][idx])
            resp_dsp_subset.extend(resp_dsp[sb][idx])

        freq_dsp_subset=np.array(freq_dsp_subset)
        resp_dsp_subset=np.array(resp_dsp_subset)

        idx_dsp = np.where( (freq_dsp_subset > freq_min) &
            (freq_dsp_subset < freq_max) )

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
        refPhaseDelayFine=int(np.round((digitizer_frequency_mhz/2/
            (subband_half_width_mhz/2.)*
            (refPhaseDelay-dsp_delay_us*(subband_half_width_mhz/2.)))))
        processing_delay_us=dsp_delay_us-cable_delay_us

        print('-------------------------------------------------------')
        print(f'Estimated refPhaseDelay={refPhaseDelay}')
        print(f'Estimated refPhaseDelayFine={refPhaseDelayFine}')
        print(f'Estimated processing_delay_us={processing_delay_us}')
        print('-------------------------------------------------------')

        #### done measuring dsp delay (cable+processing)

        #### start measuring total (DSP) delay with estimated correction applied
        # Zero refPhaseDelay and refPhaseDelayFine to get uncorrected phase
        # delay.
        # max is 7
        self.set_ref_phase_delay(band,refPhaseDelay)
        # max is 255
        self.set_ref_phase_delay_fine(band,refPhaseDelayFine)

        freq_dsp_corr=None
        resp_dsp_corr=None
        if load_find_freq_check:
            self.log('Loading delay-corrected DSP frequency sweep data')
            ff_corr_freq_file=(
                os.path.join(ff_corr_path,
                             f'{ff_corr_ctime}_amp_sweep_freq.txt'))
            ff_corr_resp_file=(
                os.path.join(ff_corr_path,
                             f'{ff_corr_ctime}_amp_sweep_resp.txt'))

            freq_dsp_corr=np.loadtxt(ff_corr_freq_file)
            resp_dsp_corr=np.loadtxt(ff_corr_resp_file,dtype='complex')
        else:
            self.log('Running find_freq')
            freq_dsp_corr,resp_dsp_corr=self.find_freq(band,dsp_subbands)

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

        ax[0].set_title(f'AMC in Bay {bay}, Band {band} Cable Delay')
        ax[0].plot(f_cable_plot,cable_phase,label='Cable (full_band_resp)',
            c='g', lw=3)
        ax[0].plot(f_cable_plot,cable_p(f_cable_plot*1.0E6),'m--',
            label='Cable delay fit',lw=3)

        ax[1].set_title(f'AMC in Bay {bay}, Band {band} DSP Delay')
        ax[1].plot(f_dsp_plot,dsp_phase,label='DSP (find_freq)',c='c',lw=3)
        ax[1].plot(f_dsp_plot,dsp_p(f_dsp_plot*1.0E6), c='orange', ls='--',
                   label='DSP delay fit', lw=3)

        ax[0].set_ylabel("Phase [rad]")
        ax[0].set_xlabel('Frequency offset from band center [MHz]')

        ax[1].set_ylabel("Phase [rad]")
        ax[1].set_xlabel('Frequency offset from band center [MHz]')

        ax[0].legend(loc='lower left',fontsize=8)
        ax[1].legend(loc='lower left',fontsize=8)

        bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
        ax[0].text(.97, .90, f'cable delay={cable_delay_us:.5f} us',
                   transform=ax[0].transAxes, fontsize=10,
                   bbox=bbox,horizontalalignment='right')

        ax[1].text(.97, .90, f'dsp delay={dsp_delay_us:.5f} us',
                   transform=ax[1].transAxes, fontsize=10,
                   bbox=bbox,horizontalalignment='right')

        cable_residuals=cable_phase-(cable_p(f_cable_plot*1.0E6))
        ax[2].plot(f_cable_plot,cable_residuals-np.median(cable_residuals),
            label='Cable (full_band_resp)',c='g')
        dsp_residuals=dsp_phase-(dsp_p(f_dsp_plot*1.0E6))
        ax[2].plot(f_dsp_plot,dsp_residuals-np.median(dsp_residuals),
            label='DSP (find_freq)', c='c')
        ax[2].plot(f_dsp_corr_plot,dsp_corr_phase-np.median(dsp_corr_phase),
            label='DSP corrected (find_freq)', c='m')
        ax[2].set_title(f'AMC in Bay {bay}, Band {band} Residuals'.format(bay,band))
        ax[2].set_ylabel("Residual [rad]")
        ax[2].set_xlabel('Frequency offset from band center [MHz]')
        ax[2].set_ylim([-5,5])

        ax[2].text(.97, .92, f'refPhaseDelay={refPhaseDelay}',
                   transform=ax[2].transAxes, fontsize=8,
                   bbox=bbox,horizontalalignment='right')
        ax[2].text(.97, .84, f'refPhaseDelayFine={refPhaseDelayFine}',
                   transform=ax[2].transAxes, fontsize=8,
                   bbox=bbox,horizontalalignment='right')
        ax[2].text(.97, .76,
                   f'processing delay={processing_delay_us:.5f} us (fw={fw_abbrev_sha})',
                   transform=ax[2].transAxes, fontsize=8,
                   bbox=bbox,horizontalalignment='right')
        ax[2].text(.97, .68, f'delay post-correction={dsp_corr_delay_us*1000.:.3f} ns',
                   transform=ax[2].transAxes, fontsize=8,
                bbox=bbox,horizontalalignment='right')

        ax[2].legend(loc='upper left',fontsize=8)

        plt.tight_layout()

        if save_plot:
            save_name = f'{timestamp}_b{band}_delay.png'

            path = os.path.join(self.plot_dir, save_name)
            plt.savefig(path,bbox_inches='tight')
            self.pub.register_file(path, 'delay', plot=True)

            if not show_plot:
                plt.close()

        self.set_att_uc(band,uc_att0,write_log=True)
        self.set_att_dc(band,dc_att0,write_log=True)


    def process_data(self, filename, dtype=np.uint32):
        """ Reads a file taken with take_debug_data and processes it into data
        and header.

        Args
        ----
        filename : str
            Path to file
        dtype : numpy.dtype, optional, default numpy.uint32
            datatype to cast to.

        Returns
        -------
        header : numpy.ndarray
            The header information.
        data : numpy.ndarray
            The resonator data.
        """
        n_chan = 2 # number of stream channels
        #header_size = 4 # 8 bytes in 16-bit word

        rawdata = np.fromfile(filename, dtype='<u4').astype(dtype)

        # -1 is equiv to [] in Matlab
        rawdata = np.transpose(np.reshape(rawdata, (n_chan, -1)))

        if dtype==np.uint32:
            header = rawdata[:2, :]
            data = np.delete(rawdata, (0,1), 0).astype(dtype)
        elif dtype==np.int32:
            header = np.zeros((2,2))
            header[:,0] = rawdata[:2,0].astype(np.uint32)
            header[:,1] = rawdata[:2,1].astype(np.uint32)
            data = np.double(np.delete(rawdata, (0,1), 0))
        elif dtype==np.int16:
            header1 = np.zeros((4,2))
            header1[:,0] = rawdata[:4,0].astype(np.uint16)
            header1[:,1] = rawdata[:4,1].astype(np.uint16)
            header1 = np.double(header1)
            header = header1[::2] + header1[1::2] * (2**16) # what am I doing
        else:
            raise TypeError(f'Type {dtype} not yet supported!')

        if (header[1,1]>>24 == 2) or (header[1,1]>>24 == 0):
            header = np.fliplr(header)
            data = np.fliplr(data)

        return header, data

    @set_action()
    def decode_data(self, filename, swapFdF=False, recast=True, truncate=True):
        """ Take a dataset from take_debug_data and spit out results.

        Args
        ----
        filename : str
            Path to file.
        swapFdF : bool, optional, default False
            Whether the F and dF (or I/Q) streams are flipped.
        recast : bool, optional, default True
            Whether to recast from size n_channels_processed to
            n_channels.
        truncate : bool, optional, default True
            Truncates the data if the number of elements returned is
            not an integer multiple of the sample rate.

        Returns
        -------
        f : numpy.ndarray
            If iqStreamEnable = 0. f is the tracking frequency.
            Otherwise if iqStreamEnable = 1. f is the demodulated
            in-phase tracking component.
        df : numpy.ndarray
            If iqStreamEnable = 0. df is the tracking frequency error.
            Otherwise if iqStreamEnable = 1. f is the demodulated
            quadrature tracking component.
        flux_ramp_strobe : numpy.ndarray
            The synchronizing pulse.
        """
        n_proc = self.get_number_processed_channels()
        n_chan = self.get_number_channels()

        n_subbands = self.get_number_sub_bands()
        digitizer_frequency_mhz = self.get_digitizer_frequency_mhz()
        subband_half_width_mhz = (digitizer_frequency_mhz / n_subbands)

        header, rawdata = self.process_data(filename)

        # decode strobes
        strobes = np.floor(rawdata / (2**30))
        data = rawdata - (2**30)*strobes
        ch0_strobe = np.remainder(strobes, 2)
        flux_ramp_strobe = np.floor((strobes - ch0_strobe) / 2)

        # decode frequencies
        ch0_idx = np.where(ch0_strobe[:,0] == 1)[0]
        f_first = ch0_idx[0]
        f_last = ch0_idx[-1]

        freqs = data[f_first:f_last, 0]
        neg = np.where(freqs >= 2**23)[0]
        f = np.double(freqs)
        if len(neg) > 0:
            f[neg] = f[neg] - 2**24

        if np.remainder(len(f), n_proc)!=0:
            if truncate:
                self.log(f'Number of points in f not a multiple of {n_proc}.' +
                    f' Truncating f to the nearest multiple of {n_proc}.',
                    self.LOG_USER)
                f=f[:(len(f)-np.remainder(len(f),n_proc))]
            else:
                self.log(f'Number of points in f not a multiple of {n_proc}.'+
                    ' Cannot decode', self.LOG_ERROR)
        f = np.reshape(f, (-1, n_proc)) * subband_half_width_mhz / 2**23

        # frequency errors
        ch0_idx_df = np.where(ch0_strobe[:,1] == 1)[0]
        if len(ch0_idx_df) > 0:
            d_first = ch0_idx_df[0]
            d_last = ch0_idx_df[-1]
            dfreq = data[d_first:d_last, 1]
            neg = np.where(dfreq >= 2**23)[0]
            df = np.double(dfreq)
            if len(neg) > 0:
                df[neg] = df[neg] - 2**24

            if np.remainder(len(df), n_proc)!=0:
                if truncate:
                    self.log('Number of points in df not a multiple of '+
                        f'{n_proc}. Truncating df to the nearest multiple ' +
                        f' of {n_proc}.', self.LOG_USER)
                    df=df[:(len(df)-np.remainder(len(df),n_proc))]
                else:
                    self.log(f'Number of points in df not a multiple of {n_proc}.' +
                        'Cannot decode', self.LOG_ERROR)
            df = np.reshape(df, (-1, n_proc)) * subband_half_width_mhz / 2**23

        else:
            df = []

        if recast:
            nsamp, nprocessed = np.shape(f)
            nsamp_df, _ = np.shape(df)
            if nsamp != nsamp_df:
                self.log('f and df are different sizes. Choosing the smaller'
                    ' value. Not sure why this is happening.')
                nsamp = np.min([nsamp, nsamp_df])

            ftmp = np.zeros((nsamp, n_chan))
            dftmp = np.zeros_like(ftmp)

            processed_ind = self.get_processed_channels()
            ftmp[:, processed_ind] = f[:nsamp]
            dftmp[:, processed_ind] = df[:nsamp]

            f = ftmp
            df = dftmp

        return f, df, flux_ramp_strobe

    @set_action()
    def decode_single_channel(self, filename, swapFdF=False):
        """
        decode take_debug_data file if in singlechannel mode

        Args
        ----
        filename : str
            Path to file to decode.
        swapFdF : bool, optional, default False
            Whether to swap f and df streams.

        Returns
        -------
        list
            [f, df, sync] if iq_stream_enable = False
            [I, Q, sync] if iq_stream_enable = True
        """

        n_subbands = self.get_number_sub_bands()
        digitizer_frequency_mhz = self.get_digitizer_frequency_mhz()
        subband_half_width_mhz = (digitizer_frequency_mhz / n_subbands)

        if swapFdF:
            nF = 1
            nDF = 0
        else:
            nF = 0
            nDF = 1

        header, rawdata = self.process_data(filename)

        # decode strobes
        strobes = np.floor(rawdata / (2**30))
        data = rawdata - (2**30)*strobes
        ch0_strobe = np.remainder(strobes, 2)
        flux_ramp_strobe = np.floor((strobes - ch0_strobe) / 2)

        # decode frequencies
        freqs = data[:,nF]
        neg = np.where(freqs >= 2**23)[0]
        f = np.double(freqs)
        if len(neg) > 0:
            f[neg] = f[neg] - 2**24

        f = np.transpose(f) * subband_half_width_mhz / 2**23

        dfreqs = data[:,nDF]
        neg = np.where(dfreqs >= 2**23)[0]
        df = np.double(dfreqs)
        if len(neg) > 0:
            df[neg] = df[neg] - 2**24

        df = np.transpose(df) * subband_half_width_mhz / 2**23

        return f, df, flux_ramp_strobe

    @set_action(action=None)
    def take_stream_data(self, meas_time, downsample_factor=None,
                         write_log=True, update_payload_size=True,
                         reset_unwrapper=True, reset_filter=True,
                         return_data=False, make_freq_mask=True,
                         register_file=True):
        """
        Takes streaming data for a given amount of time

        To do: move downsample_factor to config table

        Args
        ----
        meas_time : float
            The amount of time to observe for in seconds.
        downsample_factor : int or None, optional, default None
            The number of fast sample (the flux ramp reset rate -
            typically 4kHz) to skip between reporting. If None, does
            not update.
        write_log : bool, optional, default True
            Whether to write to the log file.
        update_payload_size : bool, optional, default True
            Whether to update the payload size (the number of channels
            written to disk). If the number of channels on is greater
            than the payload size, then only the first N channels are
            written. This bool will update the payload size to be the
            same as the number of channels on across all bands)
        reset_unwrapper : bool, optional, default True
            Whether to reset the unwrapper before taking data.
        reset_filter : bool, optional, default True
            Whether to reset the filter before taking data.
        return_data : bool, optional, default False
            Whether to return the data. If False, returns the full
            path to the data.
        make_freq_mask : bool, optional, default True
            Whether to write a text file with resonator frequencies.
        register_file : bool, optional, default True
            Whether to register the data file with the pysmurf
            publisher.


        Returns
        -------
        data_filename : str
            The fullpath to where the data is stored.
        """
        if write_log:
            self.log('Starting to take data.', self.LOG_USER)
        data_filename = self.stream_data_on(downsample_factor=downsample_factor,
            update_payload_size=update_payload_size, write_log=write_log,
            reset_unwrapper=reset_unwrapper, reset_filter=reset_filter,
            make_freq_mask=make_freq_mask)

        # Sleep for the full measurement time
        time.sleep(meas_time)

        # Stop acq
        self.stream_data_off(write_log=write_log, register_file=register_file)

        if write_log:
            self.log('Done taking data.', self.LOG_USER)

        if return_data:
            t, d, m = self.read_stream_data(data_filename)
            return t, d, m
        else:
            return data_filename

    @contextmanager
    def stream_data_cm(self, write_log=True, register_file=False,
                       **stream_on_kwargs):
        """
        Context manager for data streaming. If you intend to turn streaming
        on, do something, and then turn streaming off this is a safe way to make
        sure streaming is in fact stopped properly even if an error is raised.

        Args
        ----
        write_config : bool, optional, default False
            Whether to dump the entire config. Warning this can be
            slow.
        data_filename : str or None, optional, default None
            The full path to store the data. If None, it uses the
            timestamp.
        downsample_factor : int or None, optional, default None
            The number of fast samples to skip between sending.
        write_log : bool, optional, default True
            Whether to write to the log file.
        update_payload_size : bool, optional, default True
            Whether to update the payload size (the number of channels
            written to disk). If the number of channels on is greater
            than the payload size, then only the first N channels are
            written. This bool will update the payload size to be the
            same as the number of channels on across all bands)
        reset_filter : bool, optional, default True
            Whether to reset the filter before taking data.
        reset_unwrapper : bool, optional, default True
            Whether to reset the unwrapper before taking data.
        make_freq_mask : bool, optional, default True
            Whether to write a text file with resonator frequencies.
        register_file : bool, optional, default False
            If true, the stream data file will be registered through
            the publisher.

        Yields
        -------
        data_filename : str
            The fullpath to where the data is stored.
        """
        data_filename = self.stream_data_on(write_log=write_log, **stream_on_kwargs)
        try:
            yield data_filename
        finally:
            self.stream_data_off(write_log=write_log,
                                 register_file=register_file)

    @set_action()
    def stream_data_on(self, write_config=False, data_filename=None,
                       downsample_factor=None, write_log=True,
                       update_payload_size=True, reset_filter=True,
                       reset_unwrapper=True, make_freq_mask=True):
        """
        Turns on streaming data.

        Args
        ----
        write_config : bool, optional, default False
            Whether to dump the entire config. Warning this can be
            slow.
        data_filename : str or None, optional, default None
            The full path to store the data. If None, it uses the
            timestamp.
        downsample_factor : int or None, optional, default None
            The number of fast samples to skip between sending.
        write_log : bool, optional, default True
            Whether to write to the log file.
        update_payload_size : bool, optional, default True
            Whether to update the payload size (the number of channels
            written to disk). If the number of channels on is greater
            than the payload size, then only the first N channels are
            written. This bool will update the payload size to be the
            same as the number of channels on across all bands)
        reset_filter : bool, optional, default True
            Whether to reset the filter before taking data.
        reset_unwrapper : bool, optional, default True
            Whether to reset the unwrapper before taking data.
        make_freq_mask : bool, optional, default True
            Whether to write a text file with resonator frequencies.

        Returns
        -------
        data_filename : str
            The fullpath to where the data is stored.
        """
        bands = self.config.get('init').get('bands')

        if downsample_factor is not None:
            self.set_downsample_factor(downsample_factor)
        else:
            downsample_factor = self.get_downsample_factor()
            if write_log:
                self.log('Input downsample factor is None. Using '+
                     'value already in pyrogue:'+
                     f' {downsample_factor}')

        # Check payload size
        n_chan_in_mask = len(self.get_channel_mask())
        payload_size = self.get_payload_size()
        if n_chan_in_mask > payload_size:
            if update_payload_size:
                self.log('Updating payload size')
                self.set_payload_size(n_chan_in_mask,
                                      write_log=write_log)
            else:
                self.log('Warning : The payload size is smaller than ' +
                         'the number of channels that are on. Only ' +
                         f'writing the first {payload_size} channels. ')


        # Check if flux ramp is non-zero
        ramp_max_cnt = self.get_ramp_max_cnt()
        if ramp_max_cnt == 0:
            self.log('Flux ramp frequency is zero. Cannot take data.',
                self.LOG_ERROR)
        else:
            # check which flux ramp relay state we're in
            # read_ac_dc_relay_status() should be 0 in DC mode, 3 in
            # AC mode.  this check is only possible if you're using
            # one of the newer C02 cryostat cards.
            flux_ramp_ac_dc_relay_status=self.C.read_ac_dc_relay_status()
            if flux_ramp_ac_dc_relay_status == 0:
                if write_log:
                    self.log("FLUX RAMP IS DC COUPLED.", self.LOG_USER)
            elif flux_ramp_ac_dc_relay_status == 3:
                if write_log:
                    self.log("Flux ramp is AC-coupled.", self.LOG_USER)
            else:
                self.log("flux_ramp_ac_dc_relay_status = " +
                         f"{flux_ramp_ac_dc_relay_status} " +
                         "- NOT A VALID STATE.", self.LOG_ERROR)

            # start streaming before opening file
            # to avoid transient filter step
            self.set_stream_enable(1, write_log=False,
                                   wait_done=True)

            if reset_unwrapper:
                self.set_unwrapper_reset(write_log=write_log)
            if reset_filter:
                self.set_filter_reset(write_log=write_log)
            if reset_unwrapper or reset_filter:
                time.sleep(.1)


            # Make the data file
            timestamp = self.get_timestamp()
            if data_filename is None:
                data_filename = os.path.join(self.output_dir,
                                             timestamp+'.dat')

            self.set_data_file_name(data_filename)

            # Optionally write PyRogue configuration
            if write_config:
                config_filename=os.path.join(self.output_dir, timestamp+'.yml')
                if write_log:
                    self.log('Writing PyRogue configuration to file : '+
                         f'{config_filename}', self.LOG_USER)
                self.write_config(config_filename)

                # short wait
                time.sleep(5.)
            if write_log:
                self.log(f'Writing to file : {data_filename}',
                         self.LOG_USER)

            # Dictionary with all channels on in each band
            smurf_chans = {}
            for b in bands:
                smurf_chans[b] = self.which_on(b)

            output_mask = self.make_channel_mask(bands, smurf_chans)
            self.set_channel_mask(output_mask)

            # Save mask file as text file. Eventually this will be in the
            # raw data output
            mask_fname = os.path.join(data_filename.replace('.dat', '_mask.txt'))
            np.savetxt(mask_fname, output_mask, fmt='%i')
            self.pub.register_file(mask_fname, 'mask')
            self.log(mask_fname)

            if make_freq_mask:
                if write_log:
                    self.log("Writing frequency mask.")
                freq_mask = self.make_freq_mask(output_mask)
                np.savetxt(os.path.join(data_filename.replace('.dat', '_freq.txt')),
                           freq_mask, fmt='%4.4f')
                self.pub.register_file(
                    os.path.join(data_filename.replace('.dat', '_freq.txt')),
                    'mask', format='txt')

            self.open_data_file(write_log=write_log)

            return data_filename

    @set_action()
    def stream_data_off(self, write_log=True, register_file=False):
        """
        Turns off streaming data.

        Args
        ----
        write_log : bool, optional, default True
            Whether to log the CA commands or not.
        register_file : bool, optional, default False
            If true, the stream data file will be registered through
            the publisher.
        """
        self.close_data_file(write_log=write_log)

        if register_file:
            datafile = self.get_data_file_name().tostring().decode()
            if datafile:
                self.log(f"Registering File {datafile}")
                self.pub.register_file(datafile, 'data', format='dat')

        self.set_stream_enable(0, write_log=write_log, wait_after=.15)

    @set_action()
    def read_stream_data(self, datafile, channel=None,
                         n_samp=None, array_size=None,
                         return_header=False,
                         return_tes_bias=False, write_log=True,
                         n_max=2048, make_freq_mask=False,
                         gcp_mode=False):
        """
        Loads data taken with the function stream_data_on.
        Gives back the resonator data in units of phase. Also
        can optionally return the header (which has things
        like the TES bias).

        Args
        ----
        datafile : str
            The full path to the data to read.
        channel : int or int array or None, optional, default None
            Channels to load.
        n_samp : int or None, optional, default None
            The number of samples to read.
        array_size : int or None, optional, default None
            The size of the output arrays. If 0, then the size will be
            the number of channels in the data file.
        return_header : bool, optional, default False
            Whether to also read in the header and return the header
            data. Returning the full header is slow for large
            files. This overrides return_tes_bias.
        return_tes_bias : bool, optional, default False
            Whether to return the TES bias.
        write_log : bool, optional, default True
            Whether to write outputs to the log file.
        n_max : int, optional, default 2048
            The number of elements to read in before appending the
            datafile. This is just for speed.
        make_freq_mask : bool, optional, default False
            Whether to write a text file with resonator frequencies.
        gcp_mode (bool) : Indicates that the data was written in GCP mode. This
            is the legacy data mode which was depracatetd in Rogue 4.

        Ret:
        ----
        t (float array): The timestamp data
        d (float array): The resonator data in units of phi0
        m (int array): The maskfile that maps smurf num to gcp num
        h (dict) : A dictionary with the header information.
        """
        if gcp_mode:
            self.log('Data is in GCP mode.')
            return self.read_stream_data_gcp_save(datafile, channel=channel,
                unwrap=True, downsample=1, n_samp=n_samp)

        try:
            datafile = glob.glob(datafile+'*')[-1]
        except BaseException:
            self.log(f'datafile={datafile}')

        if write_log:
            self.log(f'Reading {datafile}')

        if channel is not None:
            self.log(f'Only reading channel {channel}')

        # Flag to indicate we are about the read the fist frame from the disk
        # The number of channel will be extracted from the first frame and the
        # data structures will be build based on that
        first_read = True
        with SmurfStreamReader(datafile,
                isRogue=True, metaEnable=True) as file:
            for header, data in file.records():
                if first_read:
                    # Update flag, so that we don't do this code again
                    first_read = False

                    # Read in all used channels by default
                    if channel is None:
                        channel = np.arange(header.number_of_channels)

                    channel = np.ravel(np.asarray(channel))
                    n_chan = len(channel)

                    # Indexes for input channels
                    channel_mask = np.zeros(n_chan, dtype=int)
                    for i, c in enumerate(channel):
                        channel_mask[i] = c

                    #initialize data structure
                    phase=list()
                    for i,_ in enumerate(channel):
                        phase.append(list())
                    for i,_ in enumerate(channel):
                        phase[i].append(data[i])
                    t = [header.timestamp]
                    if return_header or return_tes_bias:
                        tmp_tes_bias = np.array(header.tesBias)
                        tes_bias = np.zeros((0,16))

                    # Get header values if requested
                    if return_header or return_tes_bias:
                        tmp_header_dict = {}
                        header_dict = {}
                        for i, h in enumerate(header._fields):
                            tmp_header_dict[h] = np.array(header[i])
                            header_dict[h] = np.array([],
                                                      dtype=type(header[i]))
                        tmp_header_dict['tes_bias'] = np.array([header.tesBias])


                    # Already loaded 1 element
                    counter = 1
                else:
                    for i in range(n_chan):
                        phase[i].append(data[i])
                    t.append(header.timestamp)

                    if return_header or return_tes_bias:
                        for i, h in enumerate(header._fields):
                            tmp_header_dict[h] = np.append(tmp_header_dict[h],
                                                       header[i])
                        tmp_tes_bias = np.vstack((tmp_tes_bias, header.tesBias))

                    if counter % n_max == n_max - 1:
                        if write_log:
                            self.log(f'{counter+1} elements loaded')

                        if return_header:
                            for k in header_dict.keys():
                                header_dict[k] = np.append(header_dict[k],
                                                           tmp_header_dict[k])
                                tmp_header_dict[k] = \
                                    np.array([],
                                             dtype=type(header_dict[k][0]))
                            print(np.shape(tes_bias), np.shape(tmp_tes_bias))
                            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
                            tmp_tes_bias = np.zeros((0, 16))

                        elif return_tes_bias:
                            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
                            tmp_tes_bias = np.zeros((0, 16))

                    counter += 1

        phase=np.array(phase)
        t=np.array(t)

        if return_header:
            for k in header_dict.keys():
                header_dict[k] = np.append(header_dict[k],
                    tmp_header_dict[k])
            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
            tes_bias = np.transpose(tes_bias)

        elif return_tes_bias:
            tes_bias = np.vstack((tes_bias, tmp_tes_bias))
            tes_bias = np.transpose(tes_bias)

        # rotate and transform to phase
        phase = phase.astype(float) / 2**15 * np.pi

        if np.size(phase) == 0:
            self.log("Only 1 element in datafile. This is often an indication" +
                "that the data was taken in GCP mode. Try running this"+
                " function again with gcp_mode=True")

        # make a mask from mask file
        if ".dat.part" in datafile:
            mask = self.make_mask_lookup(datafile.split(".dat.part")[0] +
                "_mask.txt")
        else:
            mask = self.make_mask_lookup(datafile.replace('.dat', '_mask.txt'),
                                         make_freq_mask=make_freq_mask)

        # If an array_size was defined, resize the phase array
        if array_size is not None:
            phase.resize(array_size, phase.shape[1])

        if return_header:
            header_dict['tes_bias'] = tes_bias
            return t, phase, mask, header_dict
        elif return_tes_bias:
            return t, phase, mask, tes_bias
        else:
            return t, phase, mask

    @set_action()
    def read_stream_data_gcp_save(self, datafile, channel=None,
            unwrap=True, downsample=1, n_samp=None):
        """
        Reads the special data that is designed to be a copy of the GCP data.
        This was the most common data writing mode until the Rogue 4 update.
        Maintining this function for backwards compatibility.

        Args
        ----
        datafile : str
            The full path to the data made by stream_data_on.

        channel : int or list of int or None, optional, default None
            Channels to load.
        unwrap : bool, optional, default True
            Whether to unwrap units of 2pi.
        downsample : int, optional, default 1
            The amount to downsample.
        n_samp : int or None, optional, default None
            The number of samples to read.

        Returns
        -------
        t : numpy.ndarray
            The timestamp data.
        d : numpy.ndarray
            The resonator data in units of phi0.
        m : numpy.ndarray
            The maskfile that maps smurf num to gcp num.
        """
        import struct
        try:
            datafile = glob.glob(datafile+'*')[-1]
        except ValueError:
            print(f'datafile={datafile}')

        self.log(f'Reading {datafile}')

        if channel is not None:
            self.log(f'Only reading channel {channel}')


        keys = ['protocol_version','crate_id','slot_number','number_of_channels',
                'rtm_dac_config0', 'rtm_dac_config1', 'rtm_dac_config2',
                'rtm_dac_config3', 'rtm_dac_config4', 'rtm_dac_config5',
                'flux_ramp_increment','flux_ramp_start', 'rate_since_1Hz',
                'rate_since_TM', 'nanoseconds', 'seconds', 'fixed_rate_marker',
                'sequence_counter', 'tes_relay_config', 'mce_word',
                'user_word0', 'user_word1', 'user_word2'
                ]

        data_keys = [f'data{i}' for i in range(528)]

        keys.extend(data_keys)
        keys_dict = dict(zip(keys, range(len(keys))))

        # Read in all channels by default
        if channel is None:
            channel = np.arange(512)

        channel = np.ravel(np.asarray(channel))
        n_chan = len(channel)

        # Indices for input channels
        channel_mask = np.zeros(n_chan, dtype=int)
        for i, c in enumerate(channel):
            channel_mask[i] = keys_dict[f'data{c}']

        eval_n_samp = False
        if n_samp is not None:
            eval_n_samp = True

        # Make holder arrays for phase and timestamp
        phase = np.zeros((n_chan,0))
        timestamp2 = np.array([])
        counter = 0
        n = 20000  # Number of elements to load at a time
        tmp_phase = np.zeros((n_chan, n))
        tmp_timestamp2 = np.zeros(n)
        with open(datafile, mode='rb') as file:
            while True:
                chunk = file.read(2240)  # Frame size is 2240
                if not chunk:
                    # If frame is incomplete - meaning end of file
                    phase = np.hstack((phase, tmp_phase[:,:counter%n]))
                    timestamp2 = np.append(timestamp2, tmp_timestamp2[:counter%n])
                    break
                elif eval_n_samp:
                    if counter >= n_samp:
                        phase = np.hstack((phase, tmp_phase[:,:counter%n]))
                        timestamp2 = np.append(timestamp2,
                                               tmp_timestamp2[:counter%n])
                        break
                frame = struct.Struct('3BxI6Q8I5Q528i').unpack(chunk)

                # Extract detector data
                for i, c in enumerate(channel_mask):
                    tmp_phase[i,counter%n] = frame[c]

                # Timestamp data
                tmp_timestamp2[counter%n] = frame[keys_dict['rtm_dac_config5']]

                # Store the data in a useful array and reset tmp arrays
                if counter % n == n - 1 :
                    self.log(f'{counter+1} elements loaded')
                    phase = np.hstack((phase, tmp_phase))
                    timestamp2 = np.append(timestamp2, tmp_timestamp2)
                    tmp_phase = np.zeros((n_chan, n))
                    tmp_timestamp2 = np.zeros(n)
                counter = counter + 1

        phase = np.squeeze(phase)
        phase = phase.astype(float) / 2**15 * np.pi # where is decimal?  Is it in rad?

        rootpath = os.path.dirname(datafile)
        filename = os.path.basename(datafile)
        timestamp = filename.split('.')[0]

        mask = self.make_mask_lookup(os.path.join(rootpath,
                                                  f'{timestamp}_mask.txt'))

        return timestamp2, phase, mask

    @set_action()
    def header_to_tes_bias(self, header, as_volt=True,
                           n_tes_bias=15):
        """
        Takes the SmurfHeader returned from read_stream_data
        and turns it to a TES bias. The header is a 20 field,
        and each DAC is 18 bits signed. So the output of the
        data in the header is (dac_b - dac_a)/2. This function
        also takes care of the factor of 2 in the denominator.

        Args
        ----
        header : dict
            The header dictionary from read_stream_data.  This
            includes all the tes_byte data.

        as_volt : bool, optional, default True
            Whether to return the data as voltage. If False, returns
            as DAC units.
        n_tes_bias : int, optional, default 15
            The number of TES bias pairs.

        Returns
        -------
        bias : numpy.ndarray
            The tes bias data. (dac_b - dac_a) in voltage or DAC units
            depending on the as_volt opt arg.
        """
        # Numbr of total elements
        n_els = len(header['tes_byte_0'])

        # Pre-allocate array
        bias = np.zeros((n_tes_bias, n_els))

        # Iterate over bias groups
        for bias_group in np.arange(n_tes_bias):
            base_byte = int((bias_group*20) / 8)
            base_bit = int((bias_group*20) % 8)
            for i in np.arange(n_els):
                val = 0
                for idx, byte in enumerate(range(base_byte, base_byte+3)):
                    # Cast as type int instead of numpy.int64
                    val += int(header[f'tes_byte_{byte}'][i]) << idx*8

                # https://github.com/slaclab/pysmurf/blob/master/README.SmurfPacket.md
                # Dealing with the 16x20 bit in 10x32 bit words.
                tmp = (val >> base_bit) & 0xFFFFF
                if tmp & 0x80000:
                    tmp |= 0xF00000

                # Cast data into int
                ba = tmp.to_bytes(3, byteorder='little', signed=False)
                bias[bias_group,i] = int.from_bytes(ba, byteorder='little',
                                                    signed=True)

        # Take care of factor of 2 thrown away in writing to the header
        bias *= 2

        # Cast as voltage.
        if as_volt:
            bias *= self._rtm_slow_dac_bit_to_volt

        return bias

    @set_action()
    def make_mask_lookup(self, mask_file, mask_channel_offset=0,
            make_freq_mask=False):
        """ Makes an n_band x n_channel array where the elements correspond
        to the smurf_to_mce mask number. In other words, mask[band, channel]
        returns the GCP index in the mask that corresonds to band, channel.

        Args
        ----
        mask_file : str
            The full path the a mask file
        mask_channel_offset : int, optional, default 0
            Offset to remove from channel numbers in GCP mask file after
            loading.
        make_freq_mask : bool, optional, default False
            Whether to write a text file with resonator frequencies.

        Returns
        -------
        mask_lookup : int array
            An array with the GCP numbers.
        """
        if hasattr(self, 'config'):
            if self.config.get('smurf_to_mce').get('mask_channel_offset') is not None:
                mask_channel_offset = int(self.config.get('smurf_to_mce').get('mask_channel_offset'))

        # Look for .dat file and replace with mask file
        if ".dat" in mask_file:
            self.log("make_mask_lookup received a .dat file. " +
                "Replacing with mask path.")
            if ".dat.part" in mask_file:
                mask_file = mask_file.split(".dat.part")[0] + "_mask.txt"
            else:
                mask_file = mask_file.replace(".dat", "_mask.txt")

        mask = np.atleast_1d(np.loadtxt(mask_file))
        bands = np.unique(mask // 512).astype(int)
        ret = np.ones((np.max(bands)+1, 512), dtype=int) * -1
        if make_freq_mask:
            freq_mask_file = mask_file.replace("_mask.txt", "_freq.txt")
            freq_mask_ret = np.zeros_like(ret).astype(float)
            try:
                freq_mask = np.atleast_1d(np.loadtxt(freq_mask_file))
            except OSError:
                self.log(f'{freq_mask_file} does not exist.')
                make_freq_mask = False

        for gcp_chan, smurf_chan in enumerate(mask):
            b = int(smurf_chan//512)
            ch = int((smurf_chan-mask_channel_offset)%512)
            ret[b,ch] = gcp_chan

            # fill corresponding elements with frequency
            if make_freq_mask:
                freq_mask_ret[b, ch] = freq_mask[gcp_chan]

        # Append freq data if requested
        if make_freq_mask:
            ret = (ret, freq_mask_ret)

        return ret

    @set_action()
    def read_stream_data_daq(self, data_length, bay=0, hw_trigger=False,
            write_log=False):
        """
        Reads the stream data from the DAQ.

        Args
        ----
        data_length : int
            The number of samples to process.
        bay : int, optional, default 0
            The AMC bay number.
        hw_trigger : bool, optional, default False
            Whether to trigger the start of the acquistion with a
            hardware trigger.
        write_log : bool, optional, default False
            Whether to write outputs to log.
        """
        # Ask mitch why this is what it is...
        if bay == 0:
            stream0 = self.epics_root + ":AMCc:Stream0"
            stream1 = self.epics_root + ":AMCc:Stream1"
        else:
            stream0 = self.epics_root + ":AMCc:Stream2"
            stream1 = self.epics_root + ":AMCc:Stream3"

        pvs = [stream0, stream1]
        sg  = SyncGroup(pvs, skip_first=True)

        # trigger PV
        if not hw_trigger:
            self.set_trigger_daq(bay, 1, write_log=write_log)
        else:
            self.set_arm_hw_trigger(bay, 1, write_log=write_log)

        time.sleep(.1)
        sg.wait()

        vals = sg.get_values()

        r0 = vals[pvs[0]]
        r1 = vals[pvs[1]]

        return r0, r1

    @set_action()
    def check_adc_saturation(self, band):
        """
        Reads data directly off the ADC.  Checks for input saturation.

        Args
        -----
        band : int
            Which band.  Assumes adc number is band%4.

        Returns
        -------
        saturated : bool
           True if ADC is saturated, otherwise False.
        """
        adc = self.read_adc_data(band, data_length=2**12, do_plot=False,
                  save_data=False, show_plot=False, save_plot=False)
        adc_max   = int(np.max((adc.real.max(), adc.imag.max())))
        adc_min   = int(np.min((adc.real.min(), adc.imag.min())))
        saturated = ((adc_max > 31000) | (adc_min < -31000))
        self.log(f'ADC{band} max count: {adc_max}')
        self.log(f'ADC{band} min count: {adc_min}')
        if saturated:
            self.log(f'\033[91mADC{band} saturated\033[00m') # color red
        else:
            self.log(f'\033[92mADC{band} not saturated\033[00m') # color green
        return saturated

    @set_action()
    def check_dac_saturation(self, band):
        """
        Reads data directly off the DAC.  Checks for input saturation.

        Args
        ----
        band : int
            Which band.  Assumes dac number is band%4.

        Returns
        -------
        saturated : bool
            Flag if DAC is saturated.
        """
        dac = self.read_dac_data(band, data_length=2**12, do_plot=False,
                  save_data=False, show_plot=False, save_plot=False)
        dac_max   = int(np.max((dac.real.max(), dac.imag.max())))
        dac_min   = int(np.min((dac.real.min(), dac.imag.min())))
        saturated = ((dac_max > 31000) | (dac_min < -31000))
        self.log(f'DAC{band} max count: {dac_max}')
        self.log(f'DAC{band} min count: {dac_min}')
        if saturated:
            self.log(f'\033[91mDAC{band} saturated\033[00m') # color red
        else:
            self.log(f'\033[92mDAC{band} not saturated\033[00m') # color green
        return saturated

    @set_action()
    def read_adc_data(self, band, data_length=2**19,
                      hw_trigger=False, do_plot=False, save_data=True,
                      timestamp=None, show_plot=True, save_plot=True,
                      plot_ylimits=[None,None]):
        """
        Reads data directly off the ADC.

        Args
        ----
        band : int
            Which band.  Assumes adc number is band%4.
        data_length : int, optional, default 2**19
            The number of samples.
        hw_trigger : bool, optional, default False
            Whether to use the hardware trigger. If False, uses an
            internal trigger.
        do_plot : bool, optional, default False
            Whether or not to plot.
        save_data : bool, optional, default True
            Whether or not to save the data in a time stamped file.
        timestamp : int or None, optional, default None
            ctime to timestamp the plot and data with (if saved to
            file).  If None, it gets the time stamp right before
            acquiring data.
        show_plot : bool, optional, default True
            If do_plot is True, whether or not to show the plot.
        save_plot : bool, optional, default True
            Whether or not to save plot to file.
        plot_ylimits : [float or None, float or None], optional, default [None,None]
            y-axis limit (amplitude) to restrict plotting over.

        Returns
        -------
        dat : int array
            The raw ADC data.
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        bay=self.band_to_bay(band)
        adc_number=band%4

        self.setup_daq_mux('adc', adc_number, data_length,band=band)

        res = self.read_stream_data_daq(data_length, bay=bay,
            hw_trigger=hw_trigger)
        dat = res[1] + 1.j * res[0]

        if do_plot:
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

            import scipy.signal as signal
            digitizer_frequency_mhz = self.get_digitizer_frequency_mhz()
            f, p_adc = signal.welch(dat, fs=digitizer_frequency_mhz,
                nperseg=data_length/2, return_onesided=False, detrend=False)
            f_plot = f

            idx = np.argsort(f)
            f_plot = f_plot[idx]
            p_adc = p_adc[idx]

            plt.figure(figsize=(9,9))
            ax1 = plt.subplot(211)
            ax1.plot(np.real(dat), label='I')
            ax1.plot(np.imag(dat), label='Q')
            ax1.set_xlabel('Sample number')
            ax1.set_ylabel('Raw counts')
            ax1.set_title(f'{timestamp} Timeseries')
            ax1.legend()
            ax1.set_ylim((-2**15, 2**15))
            ax2 = plt.subplot(212)
            ax2.plot(f_plot, 10*np.log10(p_adc))
            ax2.set_ylabel(f'ADC{band}')
            ax2.set_xlabel('Frequency [MHz]')
            ax2.set_title(f'{timestamp} Spectrum')
            plt.grid(which='both')
            if plot_ylimits[0] is not None:
                plt.ylim(plot_ylimits[0],plt.ylim()[1])
            if plot_ylimits[1] is not None:
                plt.ylim(plt.ylim()[0],plot_ylimits[1])



            if save_plot:
                plot_fn = f'{self.plot_dir}/{timestamp}_adc{band}.png'
                plt.savefig(plot_fn)
                self.pub.register_file(plot_fn, 'adc', plot=True)
                self.log(f'ADC plot saved to {plot_fn}')

        if save_data:
            outfn=os.path.join(self.output_dir,
                               f'{timestamp}_adc{band}')
            self.log(f'Saving raw adc data to {outfn}',
                     self.LOG_USER)

            np.save(outfn, res)
            self.pub.register_file(outfn, 'adc', format='npy')

        return dat

    @set_action()
    def read_dac_data(self, band, data_length=2**19,
                      hw_trigger=False, do_plot=False, save_data=True,
                      timestamp=None, show_plot=True, save_plot=True,
                      plot_ylimits=[None,None]):
        """
        Read the data directly off the DAC.

        Args
        ----
        band : int
            Which band.  Assumes dac number is band%4.
        data_length : int, optional, default 2**19
            The number of samples.
        hw_trigger : bool, optional, default False
            Whether to use the hardware trigger. If False, uses an
            internal trigger.
        do_plot : bool, optional, default False
            Whether or not to plot.
        save_data : bool, optional, default True
            Whether or not to save the data in a time stamped file.
        timestamp : int or None, optional, default None
            ctime to timestamp the plot and data with (if saved to
            file).  If None, in which case it gets the time stamp
            right before acquiring data.
        show_plot : bool, optional, default True
            If do_plot is True, whether or not to show the plot.
        save_plot : bool, optional, default True
            Whether or not to save plot to file.
        plot_ylimits : list of float or list of None, optional, default [None,None]
            2-element list of y-axis limits (amplitude) to restrict
            plotting over.

        Returns
        -------
        dat : int array
            The raw DAC data.
        """
        if timestamp is None:
            timestamp = self.get_timestamp()

        bay=self.band_to_bay(band)
        dac_number=band%4

        self.setup_daq_mux('dac', dac_number, data_length, band=band)

        res = self.read_stream_data_daq(data_length, bay=bay, hw_trigger=hw_trigger)
        dat = res[1] + 1.j * res[0]

        if do_plot:
            if show_plot:
                plt.ion()
            else:
                plt.ioff()

            import scipy.signal as signal
            digitizer_frequency_mhz = self.get_digitizer_frequency_mhz()
            f, p_dac = signal.welch(dat, fs=digitizer_frequency_mhz,
                nperseg=data_length/2, return_onesided=False, detrend=False)
            f_plot = f

            idx = np.argsort(f)
            f_plot = f_plot[idx]
            p_dac = p_dac[idx]

            plt.figure(figsize=(9,9))
            ax1 = plt.subplot(211)
            ax1.plot(np.real(dat), label='I')
            ax1.plot(np.imag(dat), label='Q')
            ax1.set_xlabel('Sample number')
            ax1.set_ylabel('Raw counts')
            ax1.set_title(f'{timestamp} Timeseries')
            ax1.legend()
            ax1.set_ylim((-2**15, 2**15))
            ax2 = plt.subplot(212)
            ax2.plot(f_plot, 10*np.log10(p_dac))
            ax2.set_ylabel(f'ADC{band}')
            ax2.set_xlabel('Frequency [MHz]')
            ax2.set_title(f'{timestamp} Spectrum')
            plt.grid(which='both')
            if plot_ylimits[0] is not None:
                plt.ylim(plot_ylimits[0],plt.ylim()[1])
            if plot_ylimits[1] is not None:
                plt.ylim(plt.ylim()[0],plot_ylimits[1])


            if save_plot:
                plot_fn = f'{self.plot_dir}/{timestamp}_dac{band}.png'
                plt.savefig(plot_fn)
                self.pub.register_file(plot_fn, 'dac', plot=True)
                self.log(f'DAC plot saved to {plot_fn}')

        if save_data:
            outfn = os.path.join(self.output_dir,f'{timestamp}_dac{band}')
            self.log(f'Saving raw dac data to {outfn}', self.LOG_USER)

            np.save(outfn, res)
            self.pub.register_file(outfn, 'dac', format='npy')

        return dat

    @set_action()
    def setup_daq_mux(self, converter, converter_number, data_length,
                      band=0, debug=False, write_log=False):
        """
        Sets up for either ADC or DAC data taking.

        Args
        ----
        converter : str
            Whether it is the ADC or DAC. choices are 'adc', 'dac', or
            'debug'. The last one takes data on a single band.
        converter_number : int
            The ADC or DAC number to take data on.
        data_length : int
            The amount of data to take.
        band : int, optional, default 0
            which band to get data on.
        """

        bay=self.band_to_bay(band)

        if converter.lower() == 'adc':
            daq_mux_channel0 = (converter_number + 1)*2
            daq_mux_channel1 = daq_mux_channel0 + 1
        elif converter.lower() == 'dac':
            daq_mux_channel0 = (converter_number + 1)*2 + 10
            daq_mux_channel1 = daq_mux_channel0 + 1
        else:
            # In dspv3, daq_mux_channel0 and daq_mux_channel1 are now
            # the same for all eight bands.
            daq_mux_channel0 = 22
            daq_mux_channel1 = 23

        # setup buffer size
        self.set_buffer_size(bay, data_length, debug)

        # input mux select
        self.set_input_mux_sel(bay, 0, daq_mux_channel0,
                               write_log=write_log)
        self.set_input_mux_sel(bay, 1, daq_mux_channel1,
                               write_log=write_log)

        # which f,df stream to route to MUX, maybe?
        self.set_debug_select(bay, band%4, write_log=True)

    @set_action()
    def set_buffer_size(self, bay, size, debug=False,
                        write_log=False):
        """
        Sets the buffer size for reading and writing DAQs

        Args
        ----
        size : int
            The buffer size in number of points.
        """
        # Change DAQ data buffer size

        # Change waveform engine buffer size
        self.set_data_buffer_size(bay, size, write_log=True)
        for daq_num in np.arange(2):
            s = self.get_waveform_start_addr(bay, daq_num, convert=True,
                write_log=debug)
            e = s + 4*size
            self.set_waveform_end_addr(bay, daq_num, e, convert=True,
                write_log=debug)
            if debug:
                self.log(f'DAQ number {daq_num}: start {s} - end {e}')

    @set_action()
    def config_cryo_channel(self, band, channel, frequencyMHz, amplitude,
            feedback_enable, eta_phase, eta_mag):
        """
        Set parameters on a single cryo channel

        Args
        ----
        band : int
            The band for the channel.
        channel : int
            Which channel to configure.
        frequencyMHz : float
            The frequency offset from the subband center in MHz.
        amplitude : int
            Amplitude scale to set for the channel (0..15).
        feedback_enable : bool
            Whether to enable feedback for the channel.
        eta_phase : float
            Feedback eta phase, in degrees (-180..180).
        eta_mag : float
            Feedback eta magnitude.
        """

        n_subbands = self.get_number_sub_bands(band)
        digitizer_frequency_mhz = self.get_digitizer_frequency_mhz(band)
        subband_width = digitizer_frequency_mhz / (n_subbands / 2)

        # some checks to make sure we put in values within the correct ranges

        if frequencyMHz > subband_width / 2:
            self.log("frequencyMHz exceeds subband width! setting to top of subband")
            freq = subband_width / 2
        elif frequencyMHz < - subband_width / 2:
            self.log("frequencyMHz below subband width! setting to bottom of subband")
            freq = -subband_width / 2
        else:
            freq = frequencyMHz

        if amplitude > 15:
            self.log("amplitude too high! setting to 15")
            ampl = 15
        elif amplitude < 0:
            self.log("amplitude too low! setting to 0")
            ampl = 0
        else:
            ampl = amplitude

        # get phase within -180..180
        phase = eta_phase
        while phase > 180:
            phase = phase - 360
        while phase < -180:
            phase = phase + 360

        # now set all the PV's
        self.set_center_frequency_mhz_channel(band, channel, freq)
        self.set_amplitude_scale_channel(band, channel, ampl)
        self.set_eta_phase_degree_channel(band, channel, phase)
        self.set_eta_mag_scaled_channel(band, channel, eta_mag)

    @set_action()
    def which_on(self, band):
        """
        Finds all detectors that are on.

        Args
        ----
        band : int
            The band to search.

        Returns
        --------
        int array
            The channels that are on.
        """
        amps = self.get_amplitude_scale_array(band)
        return np.ravel(np.where(amps != 0))

    @set_action()
    def toggle_feedback(self, band, **kwargs):
        """
        Toggles feedbackEnable (->0->1) and lmsEnables1-3 (->0->1) for
        this band.  Only toggles back to 1 if it was 1 when asked to
        toggle, otherwise leaves it zero.

        Args
        ----
        band : int
           The band whose feedback to toggle.
        """

        # current vals?
        old_feedback_enable=self.get_feedback_enable(band)
        old_lms_enable1=self.get_lms_enable1(band)
        old_lms_enable2=self.get_lms_enable2(band)
        old_lms_enable3=self.get_lms_enable3(band)

        self.log(f'Before toggling feedback on band {band}, ' +
            f'feedbackEnable={old_feedback_enable}, ' +
            f'lmsEnable1={old_lms_enable1}, lmsEnable2={old_lms_enable2}, ' +
            f'and lmsEnable3={old_lms_enable3}.', self.LOG_USER)

        # -> 0
        self.log('Setting feedbackEnable=lmsEnable1=lmsEnable2=lmsEnable3=0'+
            ' (in that order).', self.LOG_USER)

        self.set_feedback_enable(band,0)
        self.set_lms_enable1(band,0)
        self.set_lms_enable2(band,0)
        self.set_lms_enable3(band,0)

        # -> 1
        logstr='Set '
        if old_feedback_enable:
            self.set_feedback_enable(band,1)
            logstr+='feedbackEnable='
        if old_lms_enable1:
            self.set_lms_enable1(band,1)
            logstr+='lmsEnable1='
        if old_lms_enable2:
            self.set_lms_enable2(band,1)
            logstr+='lmsEnable2='
        if old_lms_enable3:
            self.set_lms_enable3(band,1)
            logstr+='lmsEnable3='

        logstr+='1 (in that order).'
        self.log(logstr,
                 self.LOG_USER)

    @set_action()
    def band_off(self, band, **kwargs):
        """
        Turns off all tones in a band

        Args
        ----
        band : int
            The band that is to be turned off.
        """
        self.set_amplitude_scales(band, 0, **kwargs)
        self.set_feedback_enable_array(band, np.zeros(512, dtype=int), **kwargs)
        self.set_cfg_reg_ena_bit(0, wait_after=.11, **kwargs)


    def channel_off(self, band, channel, **kwargs):
        """
        Turns off the tone for a single channel by setting the amplitude to
        zero and disabling feedback.

        Args
        ----
        band : int
            The band that is to be turned off.
        channel : int
            The channel to turn off.
        """
        self.log(f'Turning off band {band} channel {channel}',
                 self.LOG_USER)
        self.set_amplitude_scale_channel(band, channel, 0, **kwargs)
        self.set_feedback_enable_channel(band, channel, 0, **kwargs)


    def set_feedback_limit_khz(self, band, feedback_limit_khz, **kwargs):
        """
        Sets the feedback limit

        Args
        ----
        band : int
            The band that is to be turned off.
        feedback_limit_khz : float
            The feedback rate in units of kHz.
        """
        digitizer_freq_mhz = self.get_digitizer_frequency_mhz(band)
        n_subband = self.get_number_sub_bands(band)

        subband_bandwidth = 2 * digitizer_freq_mhz / n_subband
        desired_feedback_limit_mhz = feedback_limit_khz/1000.

        if desired_feedback_limit_mhz > subband_bandwidth/2:
            desired_feedback_limit_mhz = subband_bandwidth/2

        desired_feedback_limit_dec = np.floor(desired_feedback_limit_mhz/
            (subband_bandwidth/2**16.))

        self.set_feedback_limit(band, desired_feedback_limit_dec, **kwargs)

    # if no guidance given, tries to reset both
    def recover_jesd(self,bay,recover_jesd_rx=True,recover_jesd_tx=True):
        if recover_jesd_rx:
            #1. Toggle JesdRx:Enable 0x3F3 -> 0x0 -> 0x3F3
            self.set_jesd_rx_enable(bay,0x0)
            self.set_jesd_rx_enable(bay,0x3F3)

        if recover_jesd_tx:
            #1. Toggle JesdTx:Enable 0x3CF -> 0x0 -> 0x3CF
            self.set_jesd_tx_enable(bay,0x0)
            self.set_jesd_tx_enable(bay,0x3CF)

            #2. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:
            # DAC[0]:JesdRstN 0x1 -> 0x0 -> 0x1
            self.set_jesd_reset_n(bay,0,0x0)
            self.set_jesd_reset_n(bay,0,0x1)

            #3. Toggle AMCcc:FpgaTopLevel:AppTop:AppCore:MicrowaveMuxCore[0]:
            # DAC[1]:JesdRstN 0x1 -> 0x0 -> 0x1
            self.set_jesd_reset_n(bay,1,0x0)
            self.set_jesd_reset_n(bay,1,0x1)

        # probably overkill...shouldn't call this function if you're not going
        # to do anything
        if (recover_jesd_rx or recover_jesd_tx):
            # powers up the SYSREF which is required to sync fpga and
            # adc/dac jesd
            self.run_pwr_up_sys_ref(bay)

        # check if Jesds recovered - enable printout
        (jesd_tx_ok,jesd_rx_ok)=self.check_jesd(bay,silent_if_valid=False)

        # raise exception if failed to recover
        if (jesd_rx_ok and jesd_tx_ok):
            self.log('Recovered Jesd.', self.LOG_USER)
        else:
            which_jesd_down='Jesd Rx and Tx are both down'
            if (jesd_rx_ok or jesd_tx_ok):
                which_jesd_down = ('Jesd Rx is down' if jesd_tx_ok else 'Jesd Tx is down')
            self.log('Failed to recover Jesds ...', self.LOG_ERROR)
            raise ValueError(which_jesd_down)


    def jesd_decorator(decorated):
        def jesd_decorator_function(self):
            # check JESDs
            (jesd_tx_ok0,jesd_rx_ok0)=self.check_jesd(silent_if_valid=True)

            # if either JESD is down, try to fix
            if not (jesd_rx_ok0 and jesd_tx_ok0):
                which_jesd_down0='Jesd Rx and Tx are both down'
                if (jesd_rx_ok0 or jesd_tx_ok0):
                    which_jesd_down0 = ('Jesd Rx is down' if
                        jesd_tx_ok0 else 'Jesd Tx is down')

                self.log(f'{which_jesd_down0} ... will attempt to recover.',
                         self.LOG_ERROR)

                # attempt to recover ; if it fails it will assert
                self.recover_jesd(recover_jesd_rx=(not jesd_rx_ok0),
                    recover_jesd_tx=(not jesd_tx_ok0))

                # rely on recover to assert if it failed
                self.log('Successfully recovered Jesd but may need to redo' +
                    ' some setup ... rerun command at your own risk.',
                    self.LOG_USER)

            # don't continue running the desired command by default.
            # just because Jesds are back doesn't mean we're in a sane
            # state.  User may need to relock/etc.
            if (jesd_rx_ok0 and jesd_tx_ok0):
                decorated()

        return jesd_decorator_function

    def check_jesd(self, bay, silent_if_valid=False):
        """
        Queries the Jesd tx and rx and compares the
        data_valid and enable bits.

        Args
        ----
        bay : int
            Which bay (0 or 1).
        silent_if_valid : bool, optional, default False
            If True, does not print anything if things are working.

        Returns
        -------
        (bool,bool)
            (JesdTx is ok, JesdRx is ok)
        """
        # JESD Tx
        jesd_tx_enable = self.get_jesd_tx_enable(bay)
        jesd_tx_valid = self.get_jesd_tx_data_valid(bay)
        jesd_tx_ok = (jesd_tx_enable==jesd_tx_valid)
        if not jesd_tx_ok:
            self.log("JESD Tx DOWN", self.LOG_ERROR)
        else:
            if not silent_if_valid:
                self.log("JESD Tx Okay", self.LOG_USER)

        # JESD Rx
        jesd_rx_enable = self.get_jesd_rx_enable(bay)
        jesd_rx_valid = self.get_jesd_rx_data_valid(bay)
        jesd_rx_ok = (jesd_rx_enable==jesd_rx_valid)
        if not jesd_rx_ok:
            self.log("JESD Rx DOWN", self.LOG_ERROR)
        else:
            if not silent_if_valid:
                self.log("JESD Rx Okay", self.LOG_USER)
        return (jesd_tx_ok,jesd_rx_ok)

    def get_fpga_status(self):
        """
        Loads FPGA status checks if JESD is ok.

        Returns
        -------
        ret : dict
            A dictionary containing uptime, fpga_version, git_hash,
            build_stamp, jesd_tx_enable, and jesd_tx_valid
        """
        uptime = self.get_fpga_uptime()
        fpga_version = self.get_fpga_version()
        git_hash = self.get_fpga_git_hash()
        build_stamp = self.get_fpga_build_stamp()

        git_hash = ''.join([chr(y) for y in git_hash]) # convert from int to ascii
        build_stamp = ''.join([chr(y) for y in build_stamp])

        self.log("Build stamp: " + str(build_stamp), self.LOG_USER)
        self.log("FPGA version: Ox" + str(fpga_version), self.LOG_USER)
        self.log("FPGA uptime: " + str(uptime), self.LOG_USER)

        jesd_tx_enable = self.get_jesd_tx_enable()
        jesd_tx_valid = self.get_jesd_tx_data_valid()
        if jesd_tx_enable != jesd_tx_valid:
            self.log("JESD Tx DOWN", self.LOG_USER)
        else:
            self.log("JESD Tx Okay", self.LOG_USER)

        jesd_rx_enable = self.get_jesd_rx_enable()
        jesd_rx_valid = self.get_jesd_rx_data_valid()
        if jesd_rx_enable != jesd_rx_valid:
            self.log("JESD Rx DOWN", self.LOG_USER)
        else:
            self.log("JESD Rx Okay", self.LOG_USER)


        # dict containing all values
        ret = {
            'uptime' : uptime,
            'fpga_version' : fpga_version,
            'git_hash' : git_hash,
            'build_stamp' : build_stamp,
            'jesd_tx_enable' : jesd_tx_enable,
            'jesd_tx_valid' : jesd_tx_valid,
            'jesd_rx_enable': jesd_rx_enable,
            'jesd_rx_valid' : jesd_rx_valid,
        }

        return ret

    def which_bays(self):
        r"""Which carrier AMC bays are enabled.

        Returns which AMC bays were enabled on pysmurf server startup.
        Each SMuRF carrier has two AMC bays, indexed by an integer,
        either 0 or 1.  If looking at an installed carrier from the
        front of a crate, bay 0 is on the right and bay 1 is on the
        left.

        A bay is enabled if the `--disable-bay#` argument is not
        provided as a startup argument to the pysmurf server where #
        is the bay number, either 0 or 1.  The pysmurf server startup
        arguments are returned by the
        :func:`~pysmurf.client.command.smurf_command.SmurfCommandMixin.get_smurf_startup_args`
        routine.

        Returns
        -------
        bays : list of int
            Which bays were enabled on pysmurf server startup.
        """
        # What arguments were passed to the pysmurf server on startup?
        smurf_startup_args=self.get_smurf_startup_args()

        # Bays are enabled unless --disable-bay{bay} is provided to
        # the pysmurf server on startup.
        bays=[]
        for bay in [0,1]:
            if f'--disable-bay{bay}' not in smurf_startup_args:
                bays.append(bay)

        return bays

    def which_bands(self):
        """Which bands the carrier firmware was built for.

        Returns
        -------
        bands : list of int
            Which bands the carrier firmware was built for.
        """
        build_dsp_g=self.get_build_dsp_g()
        bands=[b for b,x in enumerate(bin(build_dsp_g)[2:]) if x=='1']
        return bands

    def freq_to_subband(self, band, freq):
        """
        Look up subband number of a channel frequency, and its subband
        frequency offset.

        Args
        ----
        band : float
            The band to place the resonator.
        freq : float
            Frequency in MHz.

        Returns
        -------
        subband_no : int
            Subband (0..128) of the frequency within the band.
        offset : float
            Offset from subband center.

        """
        subbands, subband_centers = self.get_subband_centers(band,
            as_offset=False)

        df = np.abs(freq - subband_centers)
        idx = np.ravel(np.where(df == np.min(df)))[0]

        subband_no = subbands[idx]
        offset = freq - subband_centers[idx]

        return subband_no, offset

    def channel_to_freq(self, band, channel=None, yml=None):
        """
        Gives the frequency of the channel.

        Args
        ----
        band : int
            The band the channel is in.
        channel : int or None, optional, default none
            The channel number.

        Returns
        -------
        freq : float
            The channel frequency in MHz or an array of values if
            channel is None. In the array format, the freq list is
            aligned with self.which_on(band).
        """
        if band is None and channel is None:
            return None

        # Get subband centers
        _, sbc = self.get_subband_centers(band, as_offset=False, yml=yml)

        # Convenience function for turning band, channel into freq
        def _get_cf(band, ch):
            subband = self.get_subband_from_channel(band, channel, yml=yml)
            offset = float(self.get_center_frequency_mhz_channel(band, channel,
                                                                 yml=yml))
            return sbc[subband] + offset

        # If channel is requested
        if channel is not None:
            return _get_cf(band, channel)

        # Get all channels that are on
        else:
            channels = self.which_on(band)
            cfs = np.zeros(len(channels))
            for i, channel in enumerate(channels):
                cfs[i] = _get_cf(band, channel)

            return cfs


    def get_channel_order(self, band=None, channel_orderfile=None):
        """ produces order of channels from a user-supplied input file

        Args
        ----
        band : int or None, optional, default None
            Which band.  If None, assumes all bands have the same
            number of channels, and pulls the number of channels from
            the first band in the list of bands specified in the
            experiment.cfg.
        channelorderfile : str or None, optional, default None
            Path to a file that contains one channel per line.

        Returns
        -------
        channel_order : int array
            An array of channel orders.
        """

        if band is None:
            # assume all bands have the same channel order, and pull
            # the channel frequency ordering from the first band in
            # the list of bands specified in experiment.cfg.
            bands = self.config.get('init').get('bands')
            band = bands[0]

        tone_freq_offset = self.get_tone_frequency_offset_mhz(band)
        freqs = np.sort(np.unique(tone_freq_offset))

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)

        n_chanpersubband = int(n_channels / n_subbands)

        channel_order = np.zeros(len(tone_freq_offset), dtype=int)
        for i, f in enumerate(freqs):
            channel_order[n_chanpersubband*i:n_chanpersubband*(i+1)] = \
                np.ravel(np.where(tone_freq_offset == f))

        return channel_order

    def get_processed_channels(self, channel_orderfile=None):
        """
        take_debug_data, which is called by many functions including
        tracking_setup only returns data for the processed
        channels. Therefore every channel is not returned.

        Args
        ----
        channelorderfile : str or None, optional, default None
            Path to a file that contains one channel per line.
        """
        n_proc = self.get_number_processed_channels()
        n_chan = self.get_number_channels()
        n_cut = (n_chan - n_proc)//2
        return np.sort(self.get_channel_order(
            channel_orderfile=channel_orderfile)[n_cut:-n_cut])

    def get_subband_from_channel(self, band, channel, channelorderfile=None,
            yml=None):
        """Returns subband number given a channel number

        Args
        ----
        band : int
            Which band we're working in.
        channel : int
            Ranges 0..(n_channels-1), cryo channel number.
        channelorderfile : str or None, optional, default None
            Path to file containing order of channels.

        Returns
        -------
        subband : int
            The subband the channel lives in.
        """

        n_subbands = self.get_number_sub_bands(band, yml=yml)
        n_channels = self.get_number_channels(band, yml=yml)

        n_chanpersubband = n_channels / n_subbands

        if channel > n_channels:
            raise ValueError('channel number exceeds number of channels')

        if channel < 0:
            raise ValueError('channel number is less than zero!')

        chanOrder = self.get_channel_order(band,channelorderfile)
        idx = np.where(chanOrder == channel)[0]

        subband = idx // n_chanpersubband

        return int(subband)

    def get_subband_centers(self, band, as_offset=True, hardcode=False,
            yml=None):
        """ returns frequency in MHz of subband centers

        Args
        ----
        band : int
            Which band.
        as_offset : bool, optional, default True
            Whether to return as offset from band center.
        """

        if hardcode:
            #bandCenterMHz = 3.75 + 0.5*(band + 1)
            digitizer_frequency_mhz = 614.4
            n_subbands = 128
        else:
            digitizer_frequency_mhz = self.get_digitizer_frequency_mhz(band,
                yml=yml)
            n_subbands = self.get_number_sub_bands(band, yml=yml)

        subband_width_MHz = 2 * digitizer_frequency_mhz / n_subbands

        subbands = list(range(n_subbands))
        subband_centers = (np.arange(1, n_subbands + 1) - n_subbands/2) * \
            subband_width_MHz/2

        if not as_offset:
            subband_centers += self.get_band_center_mhz(band, yml=yml)

        return subbands, subband_centers

    def get_channels_in_subband(self, band, subband, channelorderfile=None):
        """
        Returns channels in subband

        Args
        ----
        band : int
            Which band.
        subband : int
            Subband number, ranges from 0..127.
        channelorderfile : str or None, optional, default None
            Path to file specifying channel order.

        Returns
        -------
        subband_chans : int array
            The channels in the subband.
        """

        n_subbands = self.get_number_sub_bands(band)
        n_channels = self.get_number_channels(band)
        n_chanpersubband = int(n_channels / n_subbands)

        if subband > n_subbands:
            raise ValueError("subband requested exceeds number of subbands")

        if subband < 0:
            raise ValueError("requested subband less than zero")

        chanOrder = self.get_channel_order(band,channelorderfile)
        subband_chans = chanOrder[subband * n_chanpersubband : subband *
            n_chanpersubband + n_chanpersubband]

        return subband_chans

    def iq_to_phase(self, i, q):
        """
        Changes IQ to phase
        """
        return np.unwrap(np.arctan2(q, i))


    def hex_string_to_int(self, s):
        """
        Converts hex string, which is an array of characters, into an int.

        Args
        ----
        s : character array
            An array of chars to be turned into a single int.

        Returns
        -------
        i : numpy.int
           The 64 bit int.
        """
        return np.int(''.join([chr(x) for x in s]),0)


    def int_to_hex_string(self, i):
        """
        Converts an int into a string of characters.

        Args
        ----
        i : int
            A 64 bit int to convert into hex.

        Returns
        -------
        s : char array
            A character array representing the int.
        """
        # Must be array length 300
        s = np.zeros(300, dtype=int)
        i_hex = hex(i)
        for j in np.arange(len(i_hex)):
            s[j] = ord(i_hex[j])

        return s


    def set_tes_bias_bipolar(self, bias_group, volt, do_enable=True,
            flip_polarity=False, **kwargs):
        """
        Set an individual TES bias group to the specified voltage, in
        volts.  Asserts if the requested bias group is not defined in
        the pysmurf configuration file.  The positive DAC in the bias
        group is set to +volt/2, while the negative DAC in the bias
        group is set to -volt/2.

        Args
        ----
        bias_group : int
            The bias group.
        volt : float
            The TES bias to command in volts.
        do_enable : bool, optional, default True
            Sets the enable bit. Only must be done once.
        flip_polarity : bool, optional, default False
            Sets the voltage to volt*-1.
        """

        # Make sure the requested bias group is in the list of defined
        # bias groups.
        bias_groups = self.bias_group_to_pair[:,0]
        assert (bias_group in bias_groups),\
            f'Bias group {bias_group} is not defined (available bias '+\
            f' groups are {bias_groups}).  Doing nothing!'

        bias_order = bias_groups
        dac_positives = self.bias_group_to_pair[:,1]
        dac_negatives = self.bias_group_to_pair[:,2]

        dac_idx = np.ravel(np.where(bias_order == bias_group))

        dac_positive = dac_positives[dac_idx][0]
        dac_negative = dac_negatives[dac_idx][0]

        volts_pos = volt / 2
        volts_neg = - volt / 2

        if flip_polarity:
            volts_pos *= -1
            volts_neg *= -1

        if do_enable:
            self.set_rtm_slow_dac_enable(dac_positive, 2, **kwargs)
            self.set_rtm_slow_dac_enable(dac_negative, 2, **kwargs)

        self.set_rtm_slow_dac_volt(dac_positive, volts_pos, **kwargs)
        self.set_rtm_slow_dac_volt(dac_negative, volts_neg, **kwargs)

    def set_tes_bias_bipolar_array(self, bias_group_volt_array, do_enable=True, **kwargs):
        """
        Set TES bipolar values for all DACs at once.  Set using a
        pyrogue array write, so should be much more efficient than
        setting each TES bias one at a time (a single register
        transaction vs. many).  Only DACs assigned to TES bias groups
        are touched by this function.  The enable status and output
        voltage of all DACs not assigned to a TES bias group are
        maintained.

        Args
        ----
        bias_group_volt_array : float array
            The TES bias to command in voltage for each bipolar TES
            bias group. Should be (n_bias_groups,).
        do_enable : bool, optional, default True
            Set the enable bit for both DACs for every TES bias group.
        """

        n_bias_groups = self._n_bias_groups

        # in this function we're only touching the DACs defined in TES
        # bias groups.  Need to make sure we carry along the setting
        # and enable of any DACs that are being used for something
        # else.
        dac_enable_array = self.get_rtm_slow_dac_enable_array()
        dac_volt_array = self.get_rtm_slow_dac_volt_array()

        if len(bias_group_volt_array) != n_bias_groups:
            self.log("Received the wrong number of biases. Expected " +
                     f"an array of n_bias_groups={n_bias_groups} voltages",
                     self.LOG_ERROR)
        else:
            for bg in np.arange(n_bias_groups):
                bias_order = self.bias_group_to_pair[:,0]
                dac_positives = self.bias_group_to_pair[:,1]
                dac_negatives = self.bias_group_to_pair[:,2]

                bias_group_idx = np.ravel(np.where(bias_order == bg))

                dac_positive = dac_positives[bias_group_idx][0] - 1 # freakin Mitch
                dac_negative = dac_negatives[bias_group_idx][0] - 1 # 1 vs 0 indexing

                volts_pos = bias_group_volt_array[bg] / 2
                volts_neg = - bias_group_volt_array[bg] / 2

                if do_enable:
                    dac_enable_array[dac_positive] = 2
                    dac_enable_array[dac_negative] = 2

                dac_volt_array[dac_positive] = volts_pos
                dac_volt_array[dac_negative] = volts_neg

            if do_enable:
                self.set_rtm_slow_dac_enable_array(dac_enable_array, **kwargs)

            self.set_rtm_slow_dac_volt_array(dac_volt_array, **kwargs)


    def set_tes_bias_off(self, **kwargs):
        """
        Turns off all of the DACs assigned to a TES bias group in the
        pysmurf configuration file.
        """
        self.set_tes_bias_bipolar_array(np.zeros(self._n_bias_groups), **kwargs)

    def get_tes_bias_bipolar(self, bias_group, return_raw=False, **kwargs):
        """
        Returns the bias voltage in units of Volts for the requested
        TES bias group.

        Args
        ----
        bias_group : int
            The number of the bias group.  Asserts if bias_group
            requested is not defined in the pysmurf configuration
            file.
        return_raw : bool, optional, default False
            If True, returns pos and neg terminal values.

        Returns
        -------
        val : float
            The bipolar output TES bias voltage for the requested bias
            group.  If return_raw=True, then returns a two element
            float array containing the output voltages of the two DACs
            assigned to the requested TES bias group.
        """
        # Make sure the requested bias group is in the list of defined
        # bias groups.
        bias_groups = self.bias_group_to_pair[:,0]
        assert (bias_group in bias_groups),\
            f'Bias group {bias_group} is not defined (available bias groups are {bias_groups}).  Doing nothing!'

        bias_order = bias_groups
        dac_positives = self.bias_group_to_pair[:,1]
        dac_negatives = self.bias_group_to_pair[:,2]

        dac_idx = np.ravel(np.where(bias_order == bias_group))
        dac_positive = dac_positives[dac_idx][0]-1
        dac_negative = dac_negatives[dac_idx][0]-1

        volt_array = self.get_rtm_slow_dac_volt_array(**kwargs)
        volts_pos = volt_array[dac_positive]
        volts_neg = volt_array[dac_negative]

        if return_raw:
            return volts_pos, volts_neg
        else:
            return volts_pos - volts_neg


    def get_tes_bias_bipolar_array(self, return_raw=False, **kwargs):
        """
        Returns array of bias voltages per bias group in units of volts.
        Currently hard coded to return the first 8 as (8,) array. I'm sorry -CY

        Args
        ----
        return_raw : bool, optional, default False
            If True, returns +/- terminal vals as separate arrays
            (pos, then negative)
        """

        bias_order = self.bias_group_to_pair[:,0]
        dac_positives = self.bias_group_to_pair[:,1]
        dac_negatives = self.bias_group_to_pair[:,2]

        n_bias_groups = self._n_bias_groups

        bias_vals_pos = np.zeros((n_bias_groups,))
        bias_vals_neg = np.zeros((n_bias_groups,))

        volts_array = self.get_rtm_slow_dac_volt_array(**kwargs)

        for idx in np.arange(n_bias_groups):
            dac_idx = np.ravel(np.where(bias_order == idx))
            dac_positive = dac_positives[dac_idx][0] - 1
            dac_negative = dac_negatives[dac_idx][0] - 1

            bias_vals_pos[idx] = volts_array[dac_positive]
            bias_vals_neg[idx] = volts_array[dac_negative]

        if return_raw:
            return bias_vals_pos, bias_vals_neg
        else:
            return bias_vals_pos - bias_vals_neg

    def set_amplifier_bias(self, bias_hemt=None, bias_50k=None, **kwargs):
        """
        Sets the HEMT and 50 K amp (if present) voltages.  If no
        arguments given, looks for default biases in cfg
        (amplifier:hemt_Vg and amplifier:LNA_Vg).  If nothing found in
        cfg file, does nothing to either bias.  Enable is written to
        both amplifier bias DACs regardless of whether or not they are
        set to new values - need to check that this is ok.  If user
        specifies values those override cfg file defaults.  Prints
        resulting amplifier biases at the end with a short wait in
        case there's latency between setting and reading.

        Args
        ----
        bias_hemt : float or None, optional default None
            The HEMT bias voltage in units of volts.
        bias_50k : float or None, optional, default None
            The 50K bias voltage in units of volts.
        """

        ########################################################################
        ### 4K HEMT
        self.set_hemt_enable(**kwargs)
        # if nothing specified take default from cfg file, if
        # it's specified there
        bias_hemt_from_cfg=False
        if bias_hemt is None and hasattr(self,'_hemt_Vg'):
            bias_hemt = self._hemt_Vg
            bias_hemt_from_cfg = True
        # if user gave a value or value was found in cfg file,
        # set it and tell the user
        if bias_hemt is not None:
            if bias_hemt_from_cfg:
                self.log('Setting HEMT LNA Vg from config file to ' +
                         f'Vg={bias_hemt:.3f}',
                         self.LOG_USER)
            else:
                self.log('Setting HEMT LNA Vg to requested ' +
                         f'Vg={bias_hemt:.3f}',
                         self.LOG_USER)

            self.set_hemt_gate_voltage(bias_hemt, override=True, **kwargs)

        # otherwise do nothing and warn the user
        else:
            self.log("No value specified for 4K HEMT Vg and " +
                     "didn't find a default in cfg " +
                     "(amplifier['hemt_Vg']).",
                     self.LOG_ERROR)
        ### done with 4K HEMT
        ########################################################################

        ########################################################################
        ### 50K LNA (if present - could make this smarter and more general)
        self.set_50k_amp_enable(**kwargs)
        # if nothing specified take default from cfg file, if
        # it's specified there
        bias_50k_from_cfg=False
        if bias_50k is None and hasattr(self,'_fiftyk_Vg'):
            bias_50k=self._fiftyk_Vg
            bias_50k_from_cfg=True
        # if user gave a value or value was found in cfg file,
        # set it and tell the user
        if bias_50k is not None:
            if bias_50k_from_cfg:
                self.log('Setting 50K LNA Vg from config file to ' +
                         f'Vg={bias_50k:.3f}',
                         self.LOG_USER)
            else:
                self.log('Setting 50K LNA Vg to requested '+
                         f'Vg={bias_50k:.3f}',
                         self.LOG_USER)

            self.set_50k_amp_gate_voltage(bias_50k, **kwargs)

        # otherwise do nothing and warn the user
        else:
            self.log("No value specified for 50K LNA Vg and " +
                     "didn't find a default in cfg " +
                     "(amplifier['LNA_Vg']).",
                     self.LOG_ERROR)
        ### done with 50K LNA
        ########################################################################

        # add some latency in case PIC needs it
        time.sleep(1)
        # print amplifier biases after setting Vgs
        self.get_amplifier_biases()

    def get_amplifier_biases(self, write_log=True):
        """
        Queries the amplifier biases

        Args
        ----
        write_log : bool, optional, default True
            Whether to write to the log.

        Returns
        -------
        amplifier_bias : dict
            Returns a dict with the hemt and 50K gate voltage and
            drain current.
        """
        # 4K
        hemt_Id_mA=self.get_hemt_drain_current()
        hemt_gate_bias_volts=self.get_hemt_gate_voltage()

        # 50K
        fiftyk_Id_mA=self.get_50k_amp_drain_current()
        fiftyk_amp_gate_bias_volts=self.get_50k_amp_gate_voltage()

        ret = {
            'hemt_Vg' : hemt_gate_bias_volts,
            'hemt_Id' : hemt_Id_mA,
            '50K_Vg' : fiftyk_amp_gate_bias_volts,
            '50K_Id' : fiftyk_Id_mA
        }

        if write_log:
            self.log(ret)

        return ret

    # alias
    get_amplifier_bias = get_amplifier_biases

    def get_hemt_drain_current(self):
        """Reports the inferred 4K HEMT amplifier drain current in mA,
        inferred by measuring the voltage across a resistor in series
        with the applied drain voltage (before the regulator) by the
        PIC on the cryostat card.  The conversion from the measured
        PIC ADC voltage to drain current assumes the circuit topology
        on the rev C2 cryostat card (SLAC board PC-248-103-02-C02, see
        schematic sheet 3).  The series resistor in that schematic is
        component R44.  The value of R54 can be specified in the
        pysmurf configuration file (as hemt_Vd_series_resistor in the
        amplifier block).  If not explicitly specified, pysmurf
        assumes the default in the C2 cryostat card BOM of 200 Ohm.

        Because the series resistor is before the regulator that drops
        the RF6.0V from the RTM down to the drain voltage set by
        manually adjusting a potentiometer on the cryostat card, the
        drain current inferred from just naively dividing the measured
        voltage across the series resistor by its resistance includes
        any additional current drawn by the regulator.  This
        additional current contribution must also be provided in
        pysmurf configuration file - pysmurf will not assume a default
        value for this offset (see hemt_Id_offset in the amplifier
        block).

        Returns
        -------
        cur : float
            4K HEMT amplifier drain current in mA.
        """
        # assumes circuit topology on rev C2 cryostat card
        # (PC-248-103-02-C02, sheet 3)
        hemt_Id_mA=2.*1000.*(self.get_cryo_card_hemt_bias())/self._hemt_Vd_series_resistor - self._hemt_Id_offset

        return hemt_Id_mA


    def get_50k_amp_drain_current(self):
        """Reports the inferred 50K amplifier drain current in mA,
        inferred by measuring the voltage across a resistor in series
        with the applied drain voltage (before the regulator) by the
        PIC on the cryostat card.  The conversion from the measured
        PIC ADC voltage to drain current assumes the circuit topology
        on the rev C2 cryostat card (SLAC board PC-248-103-02-C02, see
        schematic sheet 3).  The series resistor in that schematic is
        component R54.  The value of R54 can be specified in the
        pysmurf configuration file (as 50K_amp_Vd_series_resistor in
        the amplifier block).  If not explicitly specified, pysmurf
        assumes the default in the C2 cryostat card BOM of 10 Ohm.

        Because the series resistor is before the regulator that drops
        the RF6.0V from the RTM down to the drain voltage set by
        manually adjusting a potentiometer on the cryostat card, the
        drain current inferred from just naively dividing the measured
        voltage across the series resistor by its resistance includes
        any additional current drawn by the regulator.  This
        additional current contribution must also be provided in
        pysmurf configuration file - pysmurf will not assume a default
        value for this offset (see 50k_Id_offset in the amplifier
        block).

        Returns
        -------
        cur : float
            50K amplifier drain current in mA.
        """

        # assumes circuit topology on rev C2 cryostat card
        # (PC-248-103-02-C02, sheet 3)
        fiftyk_amp_Id_mA=2.*1000.*(self.get_cryo_card_50k_bias()/
                                   self._fiftyk_amp_Vd_series_resistor) - self._fiftyk_Id_offset

        return fiftyk_amp_Id_mA


    def overbias_tes(self, bias_group, overbias_voltage=19.9, overbias_wait=1.,
                     tes_bias=19.9, cool_wait=20., high_current_mode=False,
                     flip_polarity=False, actually_overbias=True):
        """
        Overbiases requested bias group at overbias_voltage in high current mode
        for overbias_wait seconds. If high_current_mode=False,
        returns to low current mode, after which it biases the TESs at
        tes_bias.  Then waits cool_wait seconds before returning
        control.

        Args
        ----
        bias_group : int
            The bias group to overbias.  Asserts if not a valid bias
            group.
        overbias_voltage : float, optional, default 19.9
            The value of the TES bias in the high current mode.
        overbias_wait : float, optional, default 1.0
            The time to stay in high current mode in seconds.
        tes_bias : float, optional, default 19.9
            The value of the TES bias when put back in low current
            mode.
        cool_wait : float, optional, default 20.0
            The time to wait after setting the TES bias for transients
            to die off.
        high_current_mode : bool, optional, default False
            Whether to keep the TES bias in high current mode after
            the kick.
        flip_polarity : bool, optional, default False
            Whether to flip the TES bias bipolar DAC polarity.
        actually_overbias : bool, optional, default True
            Whether to actaully do the overbias.
        """
        bias_groups = self.bias_group_to_pair[:,0]
        assert (bias_group in bias_groups),\
            f'Bias group {bias_group} is not defined (available bias groups are {bias_groups}).  Doing nothing!'

        if actually_overbias:
            # drive high current through the TES to attempt to drive normal
            self.set_tes_bias_bipolar(bias_group, overbias_voltage,
                flip_polarity=flip_polarity)
            time.sleep(.1)

            self.set_tes_bias_high_current(bias_group)
            self.log('Driving high current through TES. ' +
                     f'Waiting {overbias_wait}', self.LOG_USER)

            time.sleep(overbias_wait)

        if not high_current_mode:
            self.set_tes_bias_low_current(bias_group)
            time.sleep(.1)

        self.set_tes_bias_bipolar(bias_group, tes_bias,
            flip_polarity=flip_polarity)
        self.log(f'Waiting {cool_wait:1.1f} seconds to cool', self.LOG_USER)
        time.sleep(cool_wait)

        self.log('Done waiting.', self.LOG_USER)


    def overbias_tes_all(self, bias_groups=None, overbias_voltage=19.9,
            overbias_wait=1.0, tes_bias=19.9, cool_wait=20.,
            high_current_mode=False, actually_overbias=True):
        """
        Overbiases all requested bias groups (specified by the
        bias_groups array) at overbias_voltage in high current mode
        for overbias_wait seconds.  If high_current_mode=False,
        returns to low current mode, after which it biases the TESs at
        tes_bias.  Then waits cool_wait seconds before returning
        control.

        Args
        ----
        bias_groups : array or None, optional, default None
            Which bias groups to overbias. defaults to all_groups.
            Asserts if any of the bias groups listed is not a defined
            bias group.
        overbias_voltage : float, optional, default 19.9
            The value of the TES bias in the high current mode.
        overbias_wait : float, optional, default 1.0
            The time to stay in high current mode in seconds.
        tes_bias : float, optional, default 19.9
            The value of the TES bias when put back in low current
            mode.
        cool_wait : float, optional, default 20.0
            The time to wait after setting the TES bias for transients
            to die off.
        high_current_mode : bool, optional, default False
            Whether to keep the TES bias in high current mode after
            the kick.
        actually_overbias : bool, optional, default True
            Whether to actaully do the overbias.
        """
        # drive high current through the TES to attempt to drive normal
        if bias_groups is None:
            bias_groups = self.all_groups
        else:
            # assert requires array
            bias_groups = np.atleast_1d(bias_groups)

        valid_bias_groups = self.bias_group_to_pair[:,0]

        assert (all(bg in valid_bias_groups for bg in bias_groups)),\
            'Some of the bias groups requested are not valid '+\
            f'(available bias groups are {valid_bias_groups}).  Doing nothing!'

        # Set the overbias voltage
        if actually_overbias:
            voltage_overbias_array = self.get_tes_bias_bipolar_array()
            voltage_overbias_array[bias_groups] = overbias_voltage
            self.set_tes_bias_bipolar_array(voltage_overbias_array)

            self.log(f'Driving {overbias_voltage} V in high current mode '+
                f'through bias groups {bias_groups}. ' +
                f'Waiting {overbias_wait}', self.LOG_USER)

            # Set high current mode
            self.set_tes_bias_high_current(bias_groups)

            time.sleep(overbias_wait)

        # Set to low current mode
        if not high_current_mode:
            self.log('setting to low current')
            self.set_tes_bias_low_current(bias_groups)

        # Set TES bias
        voltage_bias_array = self.get_tes_bias_bipolar_array()
        voltage_bias_array[bias_groups] = tes_bias
        self.set_tes_bias_bipolar_array(voltage_bias_array)

        # Cool wait
        self.log(f'Waiting {cool_wait:3.2f} seconds to cool',
                 self.LOG_USER)
        time.sleep(cool_wait)
        self.log('Done waiting.', self.LOG_USER)


    def set_tes_bias_high_current(self, bias_group, write_log=False):
        """
        Sets all bias groups to high current mode. Note that the bias group
        number is not the same as the relay number. It also does not matter,
        because Joe's code secretly flips all the relays when you flip one.

        Args
        ----
        bias_group : int
            The bias group(s) to set to high current mode.
        """
        old_relay = self.get_cryo_card_relays()
        old_relay = self.get_cryo_card_relays()  # querey twice to ensure update
        new_relay = np.copy(old_relay)
        if write_log:
            self.log(f'Old relay {bin(old_relay)}')

        n_bias_groups = self._n_bias_groups
        bias_group = np.ravel(np.array(bias_group))
        for bg in bias_group:
            if bg < n_bias_groups:
                r = np.ravel(self.pic_to_bias_group[
                    np.where(self.pic_to_bias_group[:,1]==bg)])[0]
            else:
                r = bg
            new_relay = (1 << r) | new_relay
        if write_log:
            self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay, write_log=write_log)
        self.get_cryo_card_relays()


    def set_tes_bias_low_current(self, bias_group, write_log=False):
        """
        Sets all bias groups to low current mode. Note that the bias group
        number is not the same as the relay number. It also does not matter,
        because Joe's code secretly flips all the relays when you flip one

        Args
        ----
        bias_group : int
            The bias group to set to low current mode.
        """
        old_relay = self.get_cryo_card_relays()
        old_relay = self.get_cryo_card_relays()  # querey twice to ensure update
        new_relay = np.copy(old_relay)

        n_bias_groups = self._n_bias_groups
        bias_group = np.ravel(np.array(bias_group))
        if write_log:
            self.log(f'Old relay {bin(old_relay)}')
        for bg in bias_group:
            if bg < n_bias_groups:
                r = np.ravel(self.pic_to_bias_group[np.where(
                    self.pic_to_bias_group[:,1]==bg)])[0]
            else:
                r = bg
            if old_relay & 1 << r != 0:
                new_relay = new_relay & ~(1 << r)
        if write_log:
            self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay, write_log=write_log)
        self.get_cryo_card_relays()


    def set_mode_dc(self, write_log=False):
        """
        Sets flux ramp to DC coupling

        Args
        ----
        write_log : bool, optional, default False
            Whether to write outputs to log.
        """
        # The 16th bit (0 indexed) is the AC/DC coupling
        # self.set_tes_bias_high_current(16)
        r = 16

        old_relay = self.get_cryo_card_relays()
        # query twice to ensure update
        old_relay = self.get_cryo_card_relays()
        self.log(f'Old relay {bin(old_relay)}')

        new_relay = np.copy(old_relay)
        new_relay = (1 << r) | new_relay
        self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay, write_log=write_log)
        self.get_cryo_card_relays()


    def set_mode_ac(self, write_log=False):
        """
        Sets flux ramp to AC coupling

        Args
        ----
        write_log : bool, optional, default False
            Whether to write outputs to log.
        """
        # The 16th bit (0 indexed) is the AC/DC coupling
        # self.set_tes_bias_low_current(16)
        old_relay = self.get_cryo_card_relays()
        old_relay = self.get_cryo_card_relays()  # querey twice to ensure update
        new_relay = np.copy(old_relay)

        r = 16
        if old_relay & 1 << r != 0:
            new_relay = new_relay & ~(1 << r)

        self.log(f'New relay {bin(new_relay)}')
        self.set_cryo_card_relays(new_relay)
        self.get_cryo_card_relays()


    def att_to_band(self, att):
        """
        Gives the band associated with a given attenuator number

        Args
        ----
        att : int
            The attenuatory number.

        Returns
        -------
        band : int
            The band associated with the attenuator.
        """
        return self.att_to_band['band'][np.ravel(
            np.where(self.att_to_band['att']==att))[0]]

    def band_to_att(self, band):
        """
        Gives the att associated with a given 500 MHz band.

        Args
        ----
        band : int
            The 500 MHz band number.

        Returns
        -------
        att : int
            The attenuatory number.
        """
        # for now, mod 4 ; assumes the band <-> att correspondence is the same
        # for the LB and HB AMCs.
        band=band%4
        return self.att_to_band['att'][np.ravel(
            np.where(self.att_to_band['band']==band))[0]]


    def flux_ramp_rate_to_PV(self, val):
        """
        Convert between the desired flux ramp reset rate and the PV number
        for the timing triggers.

        Hardcoded somewhere that we can't access; this is just a lookup table
        Allowed reset rates (kHz): 1, 2, 3, 4, 5, 6, 8, 10, 12, 15

        Returns:
        rate_sel (int): the rate sel PV for the timing trigger
        """

        rates_kHz = np.array([15, 12, 10, 8, 6, 5, 4, 3, 2, 1])

        try:
            idx = np.where(rates_kHz == val)[0][0] # weird numpy thing sorry
            return idx
        except IndexError:
            self.log("Reset rate not allowed! Look up help for allowed values")
            return


    def flux_ramp_PV_to_rate(self, val):
        """
        Convert between PV number in timing triggers and output flux ramp reset rate

        Returns:
        reset_rate (int): the flux ramp reset rate, in kHz
        """

        rates_kHz = [15, 12, 10, 8, 6, 5, 4, 3, 2, 1]
        return rates_kHz[val]


    def why(self):
        """
        Why not?
        """
        util_dir = os.path.dirname(__file__)
        aphorisms = np.loadtxt(os.path.join(util_dir, 'aphorism.txt'),
            dtype='str', delimiter='\n')

        aph = np.random.choice(aphorisms)
        self.log(aph)
        return(aph)

    def make_channel_mask(self, band=None, smurf_chans=None):
        """
        Makes the channel mask. Only the channels in the
        mask will be streamed or written to disk.

        If no optional arguments are given, mask will contain all channels
        that are on. If both band and smurf_chans are supplied, a mask
        in the input order is created.

        Args
        ----
        band : int array or None, optional, default None
            An array of band numbers. Must be the same length as
            smurf_chans
        smurf_chans : int_array or None, optional, default None
            An array of SMuRF channel numbers.  Must be the same
            length as band.

        Returns
        -------
        output_chans : int array
            The output channels.
        """
        output_chans = np.array([], dtype=int)

        # If no input, build one by querying pyrogue
        if smurf_chans is None and band is not None:
            band = np.ravel(np.array(band))
            n_chan = self.get_number_channels(band)
            output_chans = np.arange(n_chan) + n_chan*band
        # Take user inputs and make the channel map
        elif smurf_chans is not None:
            keys = smurf_chans.keys()  # the band numbers
            for k in keys:
                n_chan = self.get_number_channels(k)
                for ch in smurf_chans[k]:
                    output_chans = np.append(output_chans,
                                             ch + n_chan*k)

        return output_chans


    def make_freq_mask(self, mask):
        """
        Makes the frequency mask. These are the frequencies
        associated with the channels in the channel mask.

        Args
        ----
        mask : int array
            The channel mask file.

        Returns
        -------
        freqs : float array
            An array with frequencies associated with the mask file.
        """
        freqs = np.zeros(len(mask), dtype=float)
        channels_per_band = 512  # avoid hardcoding this

        # iterate over mask channels and find their freq
        for i, mask_ch in enumerate(mask):
            b = mask_ch // channels_per_band
            ch = mask_ch % channels_per_band
            freqs[i] = self.channel_to_freq(b, ch)

        return freqs


    def set_downsample_filter(self, filter_order, cutoff_freq,
                              write_log=False):
        """
        Sets the downsample filter. This is anti-alias filter
        that filters data at the flux_ramp reset rate, which is
        before the downsampler.

        Args
        ----
        filter_order : int
            The number of poles in the filter.
        cutoff_freq : float
            The filter cutoff frequency.
        """
        # Get flux ramp frequency
        flux_ramp_freq = self.get_flux_ramp_freq()*1.0E3

        # Get filter parameters
        b, a = signal.butter(filter_order,
                             2*cutoff_freq/flux_ramp_freq)

        # Set filter parameters
        self.set_filter_order(filter_order,
                              write_log=write_log)
        self.set_filter_a(a, write_log=write_log)
        self.set_filter_b(b, write_log=write_log,
                          wait_done=True)

        self.set_filter_reset(wait_after=.1,
                              write_log=write_log)


    def get_filter_params(self):
        """
        Get the downsample filter parameters: filter order, filter
        gain, num averages, and the actual filter parameters. This
        reads the most recent smurf_to_mce_file to get the
        parameters. This is defined in self.smurf_to_mce_file.

        If filter order is -1, the downsampler is using a rectangula
        integrator. This will set filter_a, filter_b to None.

        Returns
        -------
        filter_params : dict
            A dictionary with the filter parameters.
        """
        # Get filter order, gain, and averages
        filter_order = self.get_filter_order()
        filter_gain = self.get_filter_gain()
        num_averages = self.get_downsample_factor()

        # Get filter order, gain, and averages

        if filter_order < 0:
            a = None
            b = None
        else:
            # Get filter parameters - (filter_order+1) elements
            a = self.get_filter_a()[:filter_order+1]
            b = self.get_filter_b()[:filter_order+1]

        # Cast into dictionary
        ret = {
            'filter_order' : filter_order,
            'filter_gain': filter_gain,
            'num_averages' : num_averages,
            'filter_a' : a,
            'filter_b' : b
        }

        return ret

    @set_action()
    def make_gcp_mask(self, band=None, smurf_chans=None, gcp_chans=None,
                      read_gcp_mask=True, mask_channel_offset=0):
        """
        Makes the gcp mask. Only the channels in this mask will be stored
        by GCP.

        If no optional arguments are given, mask will contain all channels
        that are on. If both band and smurf_chans are supplied, a mask
        in the input order is created.

        Args
        ----
        band : int array or None, optional, default None
            An array of band numbers. Must be the same length as
            smurf_chans
        smurf_chans : int_array or None, optional, default None
            An array of SMuRF channel numbers.  Must be the same
            length as band.
        gcp_chans : int_array or None, optional, default None
            A list of smurf numbers to be passed on as GCP channels.
        read_gcp_mask : bool, optional, default True
            Whether to read in the new GCP mask file.  If not read in,
            it will take no effect.
        mask_channel_offset : int, optional, default 0
            Offset to add to channel numbers in GCP mask file.
        """
        if self.config.get('smurf_to_mce').get('mask_channel_offset') is not None:
            mask_channel_offset=int(self.config.get('smurf_to_mce').get('mask_channel_offset'))

        gcp_chans = np.array([], dtype=int)
        if smurf_chans is None and band is not None:
            band = np.ravel(np.array(band))
            n_chan = self.get_number_channels(band)
            gcp_chans = np.arange(n_chan) + n_chan*band
        elif smurf_chans is not None:
            keys = smurf_chans.keys()
            for k in keys:
                self.log(f'Band {k}')
                n_chan = self.get_number_channels(k)
                for ch in smurf_chans[k]:

                    # optionally shift by an offset.  The offset is applied
                    # circularly within each 512 channel band
                    channel_offset = mask_channel_offset
                    if (ch+channel_offset)<0:
                        channel_offset+=n_chan
                    if (ch+channel_offset+1)>n_chan:
                        channel_offset-=n_chan

                    gcp_chans = np.append(gcp_chans, ch + n_chan*k + channel_offset)

        if len(gcp_chans) > 512:
            self.log('WARNING: too many gcp channels!')
            return

        static_mask = self.config.get('smurf_to_mce').get('static_mask')
        if static_mask:
            self.log('NOT DYNAMICALLY GENERATING THE MASK. STATIC. SET static_mask=0 '+
                     'IN CFG TO DYNAMICALLY GENERATE MASKS!!!')
        else:
            self.log(f'Generating gcp mask file. {len(gcp_chans)} ' +
                     'channels added')

            np.savetxt(self.smurf_to_mce_mask_file, gcp_chans, fmt='%i')

        if read_gcp_mask:
            self.read_smurf_to_gcp_config()
        else:
            self.log('Warning: new mask has not been read in yet.')

    @set_action()
    def bias_bump(self, bias_group, wait_time=.5, step_size=0.001,
                  duration=5.0, start_bias=None, make_plot=False,
                  skip_samp_start=10, high_current_mode=True,
                  skip_samp_end=10, plot_channels=None,
                  gcp_mode=False, gcp_wait=0.5, gcp_between=1.0,
                  dat_file=None, offset_percentile=2):
        """
        Toggles the TES bias high and back to its original state. From this, it
        calculates the electrical responsivity (sib), the optical responsivity (siq),
        and resistance.

        This is optimized for high_current_mode. For low current mode, you will need
        to step much slower. Try wait_time=1, step_size=.015, duration=10,
        skip_samp_start=50, skip_samp_end=50.

        Note that only the resistance is well defined now because the phase response
        has an un-set factor of -1. We will need to calibrate this out.

        Args
        ----
        bias_group : int of int array
            The bias groups to toggle. The response will return every
            detector that is on.
        wait_time : float, optional, default 0.5
            The time to wait between steps
        step_size : float, optional, default 0.001
            The voltage to step up and down in volts (for low current
            mode).
        duration : float, optional, default 5.0
            The total time of observation.
        start_bias : float or None, optional, default None
            The TES bias to start at. If None, uses the current TES
            bias.
        make_plot : bool, optional, default False
            Whether to make plots. Must set some channels in
            plot_channels.
        skip_samp_start : int, optional, default 10
            The number of samples to skip before calculating a DC
            level.
        high_current_mode : bool, optional, default True
            Whether to observe in high or low current mode.
        skip_samp_end : int, optional, default 10
            The number of samples to skip after calculating a DC
            level.
        plot_channels : int array or None, optional, default None
           The channels to plot.
        dat_file : str or None, optional, default None
            Filename to read bias-bump data from; if provided, data is
            read from file instead of being measured live.
        offset_percentile : float, optional, default 2.0
            Number between 0 and 100. Determines the percentile used
            to calculate the DC level of the TES data.

        Returns
        -------
        bands : int array
           The bands.
        channels : int array
           The channels.
        resistance : float array
            The inferred resistance of the TESs in Ohms.
        sib : float array
            The electrical responsivity. This may be incorrect until
            we define a phase convention. This is dimensionless.
        siq : float array
            The power responsivity. This may be incorrect until we
            define a phase convention. This is in uA/pW

        """
        if duration < 10* wait_time:
            self.log('Duration must bee 10x longer than wait_time for high enough' +
                     ' signal to noise.')
            return

        # Calculate sampling frequency
        # flux_ramp_freq = self.get_flux_ramp_freq() * 1.0E3
        # fs = flux_ramp_freq * self.get_downsample_factor()

        # Cast the bias group as an array
        bias_group = np.ravel(np.array(bias_group))

        # Fill in bias array if not provided
        if start_bias is not None:
            all_bias = self.get_tes_bias_bipolar_array()
            for bg in bias_group:
                all_bias[bg] += start_bias
            start_bias = all_bias
        else:
            start_bias = self.get_tes_bias_bipolar_array()

        step_array = np.zeros_like(start_bias)
        for bg in bias_group:
            step_array[bg] = step_size

        n_step = int(np.floor(duration / wait_time / 2))
        if high_current_mode:
            self.set_tes_bias_high_current(bias_group)

        if dat_file is None:
            filename = self.stream_data_on(make_freq_mask=False)

            if gcp_mode:
                self.log('Doing GCP mode bias bump')
                for j, bg in enumerate(bias_group):
                    self.set_tes_bias_bipolar(bg, start_bias[j] + step_size,
                                           wait_done=False)
                time.sleep(gcp_wait)
                for j, bg in enumerate(bias_group):
                    self.set_tes_bias_bipolar(bg, start_bias[j],
                                          wait_done=False)
                time.sleep(gcp_between)
                for j, bg in enumerate(bias_group):
                    self.set_tes_bias_bipolar(bg, start_bias[j] + step_size,
                                           wait_done=False)
                time.sleep(gcp_wait)
                for j, bg in enumerate(bias_group):
                    self.set_tes_bias_bipolar(bg, start_bias[j],
                                          wait_done=False)

            else:
                # Sets TES bias high then low
                for i in np.arange(n_step):
                    self.set_tes_bias_bipolar_array(start_bias + step_array,
                                                    wait_done=False)
                    time.sleep(wait_time)

                    self.set_tes_bias_bipolar_array(start_bias,
                                                    wait_done=False)
                    time.sleep(wait_time)

            self.stream_data_off(register_file=True)  # record data
        else:
            filename = dat_file

        if gcp_mode:
            return

        t, d, m, v_bias = self.read_stream_data(filename,
                                                return_tes_bias=True)

        # flag region after step
        flag = np.ediff1d(v_bias[bias_group[0]],
                          to_end=0).astype(bool)
        flag = self.pad_flags(flag, after_pad=20,
                              before_pad=2)

        # flag first full step
        s, e = self.find_flag_blocks(flag)
        flag[0:s[1]] = np.nan

        v_bias *= -2 * self._rtm_slow_dac_bit_to_volt  # FBU to V
        d *= self._pA_per_phi0/(2*np.pi*1.0E6) # Convert to microamp
        i_amp = step_size / self.bias_line_resistance * 1.0E6 # also uA
        i_bias = v_bias[bias_group[0]] / self.bias_line_resistance * 1.0E6


        # Scale the currents for high/low current
        if high_current_mode:
            i_amp *= self.high_low_current_ratio
            i_bias *= self.high_low_current_ratio

        # Demodulation timeline
        demod = (v_bias[bias_group[0]] - np.min(v_bias[bias_group[0]]))
        _amp = (np.max(np.abs(v_bias[bias_group[0]])) -
                np.min(np.abs(v_bias[bias_group[0]])))
        demod /= (_amp/2)
        demod -= 1
        demod[flag] = np.nan

        bands, channels = np.where(m!=-1)
        resp = np.zeros(len(bands))
        sib = np.zeros(len(bands))*np.nan

        timestamp = filename.split('/')[-1].split('.')[0]

        # Needs to be an array for the check later
        if plot_channels is None:
            plot_channels = np.array([])

        for i, (b, c) in enumerate(zip(bands, channels)):
            mm = m[b, c]
            offset = (np.percentile(d[mm], 100-offset_percentile) +
                      np.percentile(d[mm], offset_percentile))/2
            d[mm] -= offset

            # Calculate response amplitude and S_IB
            resp[i] = np.nanmedian(2*d[mm]*demod/i_amp)
            sib[i] = resp[i] / i_amp

            if c in plot_channels:
                fig, ax = plt.subplots(2, figsize=(4.5, 3),
                                       sharex=True)
                ax[0].plot(d[mm], label='resp')
                ax[0].plot(i_bias-np.min(i_bias), label='bias')

                ax[1].plot(2*d[mm]*demod/i_amp, color='k')
                ax[0].legend(loc='upper right')
                ax[0].set_ylabel('Current [uA]')
                ax[1].set_ylabel('Resp [A/A]')

                ax[1].set_xlabel('Samples')
                ax[0].set_title(f'Bias bump - b{b}ch{c:03}')
                plt.tight_layout()

                # Make plot name path
                plot_fn = os.path.join(self.plot_dir,
                                       f'{timestamp}_bias_bump_b{b}ch{c:03}.png')
                plt.savefig(plot_fn)
                plt.close(fig)
                self.pub.register_file(plot_fn, 'bias_bump', plot=True)

        resistance = np.abs(self.R_sh * (1-1/sib))
        siq = (2*sib-1)/(self.R_sh*i_amp) * 1.0E6/1.0E12  # convert to uA/pW

        ret = {}
        for b in np.unique(bands):
            ret[b] = {}
            idx = np.where(bands == b)[0]
            for i in idx:
                c = channels[i]
                ret[b][c] = {}
                ret[b][c]['resp'] = resp[i]
                ret[b][c]['R'] = resistance[i]
                ret[b][c]['Sib'] = sib[i]
                ret[b][c]['Siq'] = siq[i]

        return ret


    def all_off(self):
        """
        Turns off everything. Does band off, flux ramp off, then TES bias off.
        """
        self.log('Turning off tones')
        bands = self.config.get('init').get('bands')
        for b in bands:
            self.band_off(b)

        self.log('Turning off flux ramp')
        self.flux_ramp_off()

        self.log('Turning off all TES biases')
        n_bias_groups = self._n_bias_groups
        for bg in np.arange(n_bias_groups):
            self.set_tes_bias_bipolar(bg, 0)


    def mask_num_to_gcp_num(self, mask_num):
        """
        Goes from the smurf2mce mask file to a gcp number.
        Inverse of gcp_num_to_mask_num.

        Args
        ----
        mask_num : int
            The index in the mask file.

        Returns
        -------
        gcp_num : int
            The index of the channel in GCP.
        """
        return (mask_num*33)%528+mask_num//16


    def gcp_num_to_mask_num(self, gcp_num):
        """
        Goes from a GCP number to the smurf2mce index.
        Inverse of mask_num_to_gcp_num

        Args
        ----
        gcp_num : int
            The gcp index.

        Returns
        -------
        mask_num : int
            The index in the mask.
        """
        return (gcp_num*16)%528 + gcp_num//33


    def smurf_channel_to_gcp_num(self, band, channel, mask_file=None):
        """
        Converts from smurf channel (band and channel) to a gcp number

        Args
        ----
        band : int
            The smurf band number.
        channel : int
            The smurf channel number.
        mask_file : int array or None, optional, default None
            The mask file to convert between smurf channel and GCP
            number.

        Returns
        -------
        gcp_num : int
            The GCP number.
        """
        if mask_file is None:
            mask_file = self.smurf_to_mce_mask_file


        mask = self.make_mask_lookup(mask_file)

        if mask[band, channel] == -1:
            self.log(f'Band {band} Ch {channel} not in mask file')
            return None

        return self.mask_num_to_gcp_num(mask[band, channel])


    def gcp_num_to_smurf_channel(self, gcp_num, mask_file=None):
        """
        Converts from gcp number to smurf channel (band and channel).

        Args
        ----
        gcp_num : int
            The GCP number.
        mask_file : int array or None, optional, default None
            The mask file to convert between smurf channel and GCP
            number.

        Returns
        -------
        band : int
            The smurf band number.
        channel : int
            The smurf channel number.
        """
        if mask_file is None:
            mask_file = self.smurf_to_mce_mask_file
        mask = np.loadtxt(mask_file)

        mask_num = self.gcp_num_to_mask_num(gcp_num)
        return int(mask[mask_num]//512), int(mask[mask_num]%512)


    def play_sine_tes(self, bias_group, tone_amp, tone_freq, dc_amp=None):
        """
        Play a sine wave on the bias group pair.

        Tone file is in bias bit units. The bias is int20. The
        inputs of this function are in units of bias dac output
        voltage. The conversion from requested volts to bits
        is calculated in this function.

        Args
        ----
        bias_group : int
            The bias group to play a sine wave on.
        tone_amp : float
            The amplitude of the sine wave in units of out TES bias in
            volts.
        tone_freq : float
            The frequency of the tone in Hz.
        dc_amp : float or None, optional, default None
            The amplitude of the DC term of the sine wave.  If None,
            reads the current DC value and uses that.
        """
        if dc_amp is None:
            dc_amp = self.get_tes_bias_bipolar(bias_group)
            self.log(f"No dc_amp provided. Using current value: {dc_amp} V")

        # The waveform is played on 2 DACs, so amp/2. Then convert
        # to bits
        dc_amp /= (2*self._rtm_slow_dac_bit_to_volt)
        tone_amp /= (2*self._rtm_slow_dac_bit_to_volt)

        # Handles issue where it won't play faster than ~7 Hz
        freq_split = 5
        scale = 1
        if tone_freq > freq_split:
            scale = np.ceil(tone_freq / freq_split)

        # Make tone file. 2048 elements
        n_tes_samp = 2048
        sig = tone_amp * \
            np.cos(2*np.pi*scale*np.arange(n_tes_samp)/n_tes_samp) + dc_amp

        # Calculate frequency - 6.4ns * TimerSize between samples
        ts = int((tone_freq * n_tes_samp * 6.4E-9)**-1)
        ts *= scale
        self.set_rtm_arb_waveform_timer_size(ts, wait_done=True)

        self.play_tes_bipolar_waveform(bias_group, sig)


    def play_tone_file(self, band, tone_file=None, load_tone_file=True):
        """
        Plays the specified tone file on this band.  If no path provided
        for tone file, assumes the path to the correct tone file has
        already been loaded.

        Args
        ----
        band : int
            Which band to play tone file on.
        tone_file : str or None, optional, default None
            Path (including csv file name) to tone file.  If None,
            uses whatever's already been loaded.
        load_tone_file : bool, optional, default True
            Whether or not to load the tone file.  The tone file is
            loaded per DAC, so if you already loaded the tone file for
            this DAC you don't have to do it again.
        """
        # the bay corresponding to this band.
        bay = self.band_to_bay(band)

        # load the tone file
        if load_tone_file:
            self.load_tone_file(bay,tone_file)

        # play it!
        self.log(f'Playing tone file {tone_file} on band {band}',
                 self.LOG_USER)
        self.set_waveform_select(band, 1)


    def stop_tone_file(self, band):
        """
        Stops playing tone file on the specified band and reverts
        to DSP.

        Args
        ----
        band : int
            Which band to play tone file on.
        """

        self.set_waveform_select(band,0)

        # may need to do this, not sure.  Try without
        # for now.
        #self.set_dsp_enable(band,1)


    def get_gradient_descent_params(self, band):
        """
        Convenience function for getting all the serial
        gradient descent parameters

        Args
        ----
        band : int
            The band to query.

        Returns
        -------
        params : dict
            A dictionary with all the gradient descent parameters
        """
        ret = {}
        ret['averages'] = self.get_gradient_descent_averages(band)
        ret['beta'] = self.get_gradient_descent_beta(band)
        ret['converge_hz'] = self.get_gradient_descent_converge_hz(band)
        ret['gain'] = self.get_gradient_descent_gain(band)
        ret['max_iters'] = self.get_gradient_descent_max_iters(band)
        ret['momentum'] = self.get_gradient_descent_momentum(band)
        ret['step_hz'] = self.get_gradient_descent_step_hz(band)

        return ret


    def set_fixed_tone(self,freq_mhz,drive,quiet=False):
        """
        Places a fixed tone at the requested frequency.  Asserts
        without doing anything if the requested resonator frequency
        falls outside of the usable 500 MHz bands, or if there are no
        unassigned channels available in the subband the requested
        frequency falls into (where a channel is deemed "assigned" if
        it has non-zero amplitude).

        Args
        ----
        freq_mhz : float
            The frequency in MHz at which to place a fixed tone.
        drive : int
            The amplitude for the fixed tone (0-15 in recent fw
            revisions).
        """

        # Find which band the requested frequency falls into.
        bands=self.which_bands()
        band_centers_mhz=[self.get_band_center_mhz(b) for b in bands]

        band_idx=min(range(len(band_centers_mhz)), key=lambda i: abs(band_centers_mhz[i]-freq_mhz))
        band=bands[band_idx]
        band_center_mhz=band_centers_mhz[band_idx]

        # Confirm that the requested frequency falls into a 500 MHz
        # band that's usable in this fw.  If not, assert.
        assert (np.abs(freq_mhz-band_center_mhz)<250),\
            f'! Requested frequency (={freq_mhz:0.1f} MHz) outside of the ' + \
            '500 MHz band with the closest band center ' + \
            f'(={band_center_mhz:0.0f} MHz). Doing nothing!'

    # Find subband this frequency falls in, and its channels.
        subband,foff=self.freq_to_subband(band,freq_mhz)
        subband_channels=self.get_channels_in_subband(band,subband)

    # Which channels in the subband are unallocated?
        allocated_channels=self.which_on(band)
        unallocated_channels=[chan for chan in subband_channels if chan not in allocated_channels]
        # If no unallocated channels available in the subband, assert.
        assert (len(unallocated_channels)), \
            f'! No unallocated channels available in subband (={subband:d}).' + \
            ' Doing nothing!'

        # Take lowest channel number in the list of unallocated
        # channels for this subband.
        channel=sorted(unallocated_channels)[0]

    # Put a fixed tone at the requested frequency
        self.set_center_frequency_mhz_channel(band,channel,foff)
        self.set_amplitude_scale_channel(band,channel,drive)
        self.set_feedback_enable_channel(band,channel,0)

        # Unless asked to be quiet, print where we're putting a fixed
        # tone.
        if not quiet:
            self.log(f'Setting a fixed tone at {freq_mhz:.2f} MHz' +
                     f' and amplitude {drive}', self.LOG_USER)

    # SHOULD MAKE A GET FIXED TONE CHANNELS FUNCTION - WOULD MAKE IT
    # EASIER TO CHANGE THINGS FAST USING THE ARRAY GET/SETS
    def turn_off_fixed_tones(self,band):
        """
        Turns off every channel which has nonzero amplitude but
        feedback set to zero.

        Args
        ----
        band : int
            The band to query.
        """
        amplitude_scale_array=self.get_amplitude_scale_array(band)
        feedback_enable_array=self.get_feedback_enable_array(band)

    # want to turn off all channels for which the amplitude is
    # nonzero, but feedback is not enabled.
        fixed_tone_channels=np.where((amplitude_scale_array*(1-feedback_enable_array))!=0)
        new_amplitude_scale_array=amplitude_scale_array.copy()
        new_amplitude_scale_array[fixed_tone_channels]=0

    # set by array, not by channel
        self.set_amplitude_scale_array(band,new_amplitude_scale_array)

    __hardware_logging_pause_event=None

    def pause_hardware_logging(self):
        self.__hardware_logging_pause_event.set()

    def resume_hardware_logging(self):
        self.__hardware_logging_pause_event.clear()

    __hardware_log_file=None

    def get_hardware_log_file(self):
        return self.__hardware_log_file

    _hardware_logging_thread=None
    __hardware_logging_stop_event=None

    def start_hardware_logging(self,filename=None):
        # Just in case somewhere the enable got set to false, explicitly enable here
        if filename is None:
            filename=str(self.get_timestamp())+'_hwlog.dat'
        self.__hardware_log_file = os.path.join(self.output_dir, filename)
        self.log('Starting hardware logging to file : ' +
                 f'{self.__hardware_log_file}',
                 self.LOG_USER)
        self.__hardware_logging_stop_event=threading.Event()
        self.__hardware_logging_pause_event=threading.Event()
        self._hardware_logging_thread = threading.Thread(target=self._hardware_logger,
            args=(self.__hardware_logging_pause_event,
                self.__hardware_logging_stop_event,))
        self._hardware_logging_thread.daemon = True
        self._hardware_logging_thread.start()

    def stop_hardware_logging(self):
        self.__hardware_logging_stop_event.set()
        self._hardware_logging_thread.join()
        self._hardware_logging_thread=None
        self.__hardware_log_file=None

    def _hardware_logger(self,pause_event,stop_event,wait_btw_sec=5):
        filename=self.get_hardware_log_file()
        import fcntl
        #counter=0
        while not stop_event.wait(wait_btw_sec):
            if not pause_event.isSet():
                hdr,entry=self.get_hardware_log_entry()
                # only write header once, if file doesn't exist yet if
                # file *does* already exist, check to make sure header
                # will be the same, otherwise the resulting data won't
                # make sense if multiple carriers are logging to the same
                # file.
                if not os.path.exists(filename):
                    with open(filename,'a') as logf:
                        logf.write(hdr)
                else:
                    with open(filename) as logf:
                        hdr2=logf.readline()
                        if not hdr.rstrip().split() == hdr2.rstrip().split():
                            self.log('Attempting to temperature log to an ' +
                                'incompatible file.  Giving up without ' +
                                'logging any data!', self.LOG_ERROR)
                            return

                with open(filename,'a') as logf:
                    # file locking so multiple hardware loggers running in
                    # multiple pysmurf sessions can write to the same
                    # requested file if desired
                    fcntl.flock(logf, fcntl.LOCK_EX)
                    logf.write(entry)
                    fcntl.flock(logf, fcntl.LOCK_UN)
                #counter+=1

    def get_hardware_log_entry(self):

        d={}
        d['epics_root']=lambda:self.epics_root
        d['ctime']=self.get_timestamp
        d['fpga_temp']=self.get_fpga_temp
        d['fpgca_vccint']=self.get_fpga_vccint
        d['fpgca_vccaux']=self.get_fpga_vccaux
        d['fpgca_vccbram']=self.get_fpga_vccbram
        d['cc_temp']=self.get_cryo_card_temp

        # probably should check for which AMCs are in in a smarter way
        bays=[]
        bands=self.which_bands()
        if 0 in bands:
            bays.append(0)
        if 4 in bands:
            bays.append(1)

        for bay in bays:
            for dac in [0,1]:
                d[f'bay{bay}_dac{dac}_temp']=(
                    lambda:self.get_dac_temp(bay,dac))

        #AT THE MOMENT, WAY TOO SLOW
        # keep track of how many tones are on in each band
        #for band in bands:
        #    d['chans_b%d'%band]=lambda:len(self.which_on(band))

        # atca monitor
        d['atca_temp_fpga']=self.get_board_temp_fpga
        d['atca_temp_rtm']=self.get_board_temp_rtm
        d['atca_temp_amc0']=self.get_board_temp_amc0
        d['atca_temp_amc2']=self.get_board_temp_amc2
        d['atca_jct_temp_fpga']=self.get_junction_temp_fpga

        # regulator
        d['regulator_iout']=self.get_regulator_iout
        d['regulator_temp1']=self.get_regulator_temp1
        d['regulator_temp2']=self.get_regulator_temp2

        columns=[]
        names=[]
        fmt=''
        counter=0
        for key, value in d.items():
            columns.append(str(value()))
            names.append(key)
            fmt+=f'{0[{counter}]:<20}'
            counter+=1
        fmt+='\n'

        hdr=fmt.format(names)
        row=fmt.format(columns)
        return hdr,row

    def play_tes_bipolar_waveform(self, bias_group, waveform, do_enable=True,
            continuous=True, **kwargs):
        """ Play a bipolar waveform on the bias group.

        Args
        ----
        bias_group : int
            The bias group
        waveform : float array
            The waveform the play on the bias group.
        do_enable : bool, optional, default True
            Whether to enable the DACs (similar to what is required
            for TES bias).
        continuous : bool, optional, default True
            Whether to play the TES waveform continuously.
        """
        bias_order = self.bias_group_to_pair[:,0]
        dac_positives = self.bias_group_to_pair[:,1]
        dac_negatives = self.bias_group_to_pair[:,2]

        dac_idx = np.ravel(np.where(bias_order == bias_group))

        dac_positive = dac_positives[dac_idx][0]
        dac_negative = dac_negatives[dac_idx][0]

        # https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+firmware#SMuRFfirmware-RTMDACarbitrarywaveforms
        # Target the two bipolar DACs assigned to this bias group:
        self.set_dac_axil_addr(0, dac_positive)
        self.set_dac_axil_addr(1, dac_negative)

        # Enable waveform generation (3=on both DACs)
        self.set_rtm_arb_waveform_enable(3)

        # Must enable the DACs (if not enabled already)
        if do_enable:
            self.set_rtm_slow_dac_enable(dac_positive, 2, **kwargs)
            self.set_rtm_slow_dac_enable(dac_negative, 2, **kwargs)

        # Load waveform into each DAC's LUT table.  Opposite sign so
        # they combine coherently
        self.set_rtm_arb_waveform_lut_table(0, waveform)
        self.set_rtm_arb_waveform_lut_table(1, -waveform)

        # Continous mode to play the waveform continuously
        if continuous:
            self.set_rtm_arb_waveform_continuous(1)
        else:
            self.set_rtm_arb_waveform_continuous(0)

    # Readback on which DACs are selected is broken right now,
    # so has to be specified.
    def stop_tes_bipolar_waveform(self, bias_group, **kwargs):
        """
        Stop the bipolar waveform being played on a bias group.

        Args
        ----
        bias_group : int
            The bias group.
        """
        # https://confluence.slac.stanford.edu/display/SMuRF/SMuRF+firmware#SMuRFfirmware-RTMDACarbitrarywaveforms
        # Target the two bipolar DACs assigned to this bias group:
        self.set_dac_axil_addr(0,0) # Disabled
        self.set_dac_axil_addr(1,0) # Disabled

        # Enable waveform generation (3=on both DACs)
        self.set_rtm_arb_waveform_enable(0)

        # Zero TES biases on this bias group
        self.set_tes_bias_bipolar(bias_group, 0)

    @set_action()
    def get_sample_frequency(self):
        """ Gives the data rate.

        Returns
        -------
        sample_frequency : float
            The data sample rate in Hz.
        """
        flux_ramp_freq = self.get_flux_ramp_freq() * 1.0E3
        downsample_factor = self.get_downsample_factor()

        return flux_ramp_freq / downsample_factor

    def identify_bias_groups(self, bias_groups=None,
            probe_freq=2.5, probe_time=3, probe_amp=.1, make_plot=False,
            show_plot=False, save_plot=True, cutoff_frac=.05,
            update_channel_assignment=True, high_current_mode=True):
        """ Identify bias groups of all the channels that are on. Plays
        a sine wave on a bias group and looks for a response. Does
        this with the TESs superconducting so it can look for an
        response is exactly the same amplitude as the input.

        Args
        ----
        bias_groups : int array or None, optional, default None
            The bias groups to search. If None, does the first 8 bias
            groups.
        probe_freq : float, optional, default 2.5
            The frequency of the probe tone.
        probe_time : float, optional, default 3
            The length of time to probe each bias group in seconds.
        probe_amp : float, optional, default 0.1
            Amplitude of the probe signal in volts.
        make_plot : bool, optional, default False
            Whether to make the plot.
        show_plot : bool, optional, default False
            Whether to show the plot.
        save_plot : bool, optional, default True
            Whether to save the plot.
        cutoff_frac : float, optional, default 0.05
            The fraction difference the response can be away from the
            expected amplitude.
        update_channel_assignment : bool, optional, default True
            Whether to update the master channels assignment to
            contain the new bias group information.
        high_current_mode : bool, optional, default True
            Whether to use high or low current mode.

        Returns
        -------
        channels_dict : dict of {int : dict of {str : numpy.ndarray} }
            A dictionary where the first key is the bias group that is
            being probed. In each is the band, channnel pairs, and
            frequency of the channels.
        """
        # Check if probe frequency is too high
        flux_ramp_freq = self.get_flux_ramp_freq() * 1.0E3
        fs = flux_ramp_freq * self.get_downsample_factor()

        # Calculate downsample filter transfer function
        filter_params = self.get_filter_params()
        w, h = signal.freqz(filter_params['filter_b'],
                            filter_params['filter_a'],
                            fs=flux_ramp_freq)
        df = np.abs(w - probe_freq)
        df_idx = np.ravel(np.where(df == np.min(df)))[0]
        if probe_freq > fs:
            self.log('Probe frequency is higher than sample rate. Exiting',
                     self.LOG_ERROR)
            return
        elif h[df_idx] < 1 - cutoff_frac/3:
            self.log('Downsample filter cutting into the signal too much.' +
                     ' Exiting.', self.LOG_ERROR)
            return

        # There should be something smarter than this
        if bias_groups is None:
            bias_groups = np.arange(self._n_bias_groups)

        channels_dict = {}

        # Get the cryocard settings before starting this script
        cryo_card_bits = self.get_cryo_card_relays()

        timestamp = self.get_timestamp()

        for bias_group in bias_groups:
            self.log(f"Working on bias group {bias_group}")

            # Work in high current mode to bypass filter
            if high_current_mode:
                self.set_tes_bias_high_current(bias_group)
            else:
                self.set_tes_bias_low_current(bias_group)

            # Play sine wave and take data
            self.play_sine_tes(bias_group, probe_amp, probe_freq, dc_amp=0)
            datafile = self.take_stream_data(probe_time, write_log=False)

            self.stop_tes_bipolar_waveform(bias_group)

            # Read back data
            t, d, mm = self.read_stream_data(datafile, make_freq_mask=True)
            m = mm[0]  # extract mask
            m_freq = mm[1]  #frequency mask
            freq_arr = m_freq[np.where(m!=-1)]
            d *= (self._pA_per_phi0/2/np.pi)  # convert to pA
            d = np.transpose(d.T - np.mean(d.T, axis=0))

            n_det, n_samp = np.shape(d)

            # currents on lines
            if high_current_mode:
                r_inline = self.bias_line_resistance / self.high_low_current_ratio
            else:
                r_inline = self.bias_line_resistance

            i_bias = probe_amp / r_inline * 1.0E12  # Bias current in pA

            # sine/cosine decomp templates
            s = np.sin(2*np.pi*np.arange(n_samp) / n_samp*probe_freq*probe_time)
            c = np.cos(2*np.pi*np.arange(n_samp) / n_samp*probe_freq*probe_time)
            s /= np.sum(s**2)
            c /= np.sum(c**2)

            # cosine/sine decomposition
            sa = np.zeros(n_det)
            ca = np.zeros(n_det)
            for ch in np.arange(n_det):
                sa[ch] = np.dot(d[ch], s)
                ca[ch]= np.dot(d[ch], c)
            amp = np.sqrt(sa**2 + ca**2)  # amplitude calculation

            # In superconducting, amplitude of response should be 1
            norm_amp = amp/i_bias
            idx = np.where(np.logical_and(norm_amp < 1+cutoff_frac,
                           norm_amp > 1-cutoff_frac))[0]

            bands = np.zeros(len(idx), dtype=int)
            channels = np.zeros(len(idx), dtype=int)
            freqs = np.zeros(len(idx))
            for i, ii in enumerate(idx):
                bands[i], channels[i] = np.ravel(np.where(m == ii))
                freqs[i] = m_freq[bands[i], channels[i]]

            channels_dict[bias_group] = {}
            channels_dict[bias_group]['band'] = bands
            channels_dict[bias_group]['channel'] = channels
            channels_dict[bias_group]['freq'] = freqs

            if make_plot:
                # Turn off interactive plot
                if show_plot:
                    plt.ion()
                else:
                    plt.ioff()

                fig, ax = plt.subplots(1, 2, figsize=(8.5, 3),
                                       sharey=True)
                # Plot timestreams
                ax[0].plot((t-t[0])*1.0E-9, d.T,
                         color='k', alpha=.1)
                ax[0].axhline(-i_bias, linestyle='--', color='r')
                ax[0].axhline(i_bias, linestyle='--', color='r')
                ax[1].axhline(-i_bias, linestyle='--', color='r')
                ax[1].axhline(i_bias, linestyle='--', color='r')
                ax[0].set_xlabel('Time [s]')
                ax[0].set_ylabel('Amp [pA]')

                current_mode_label = 'high current'
                if not high_current_mode:
                    current_mode_label = 'low current'
                ax[0].text(.02, .98, current_mode_label,
                    transform=ax[0].transAxes, va='top', ha='left')

                ax[1].plot(freq_arr, sa, 'x', color='b',
                           label='sine', alpha=.5)
                ax[1].plot(freq_arr, ca, '+', color='y',
                           label='cos', alpha=.5)
                ax[1].plot(freq_arr, amp, 'o', color='k',
                           label='amp')
                ax[1].legend(loc='lower right')
                ax[1].set_ylim((-1.5*i_bias, 1.5*i_bias))
                ax[1].set_xlabel('Res Freq [MHz]')
                plt.tight_layout()

                fig.suptitle(f'Bias Group {bias_group}')

                if save_plot:
                    savename = f'{timestamp}_identify_bg{bias_group:02}.png'
                    plt.savefig(os.path.join(self.plot_dir, savename),
                                bbox_inches='tight')
                    self.pub.register_file(
                        os.path.join(self.plot_dir, savename),
                        'identify_bg', plot=True)
                if not show_plot:
                    plt.close(fig)

            # Set relays back to original state
            self.set_cryo_card_relays(cryo_card_bits)

        # To do - add a check for band, channels that are on two different
        # bias groups.

        for bias_group in bias_groups:
            self.log(f'Bias Group {bias_group} : ')
            self.log(f"   Bands : {np.unique(channels_dict[bias_group]['band'])}")
            n_chan = len(channels_dict[bias_group]['channel'])
            self.log("   Number of channels : "+
                     f"{n_chan}")
            if n_chan > 0:
                ff = channels_dict[bias_group]['freq']
                self.log(f"   Between freq : {np.min(ff)} and {np.max(ff)}")


        if update_channel_assignment:
            self.log('Updating channel assignment')
            self.write_group_assignment(channels_dict)

        return channels_dict


    def measure_tes_transfer(self, band, bias_group, probe_freq=None,
            probe_amp=.002, n_cycle=5, min_probe_time=2,
            overbias_tes=False, tes_bias=None,
            overbias_wait=None, cool_wait=None, overbias_voltage=19.9,
            analyze=True, high_current_mode=False, make_plot=False,
            save_plot=True, show_plot=False):
        """
        """
        f, sb, ch, bg = self.get_master_assignment(band)

        timestamp = self.get_timestamp()

        # Default probe frequency
        if probe_freq is None:
            probe_freq = 10**np.arange(0, 3, .5)


        # Overbias the TES
        if overbias_tes:
            if tes_bias is None:
                raise ValueError('Must supply tes_bias')
            elif overbias_wait is None:
                raise ValueError('Must supply overbias_wait')
            else:
                self.overbias_tes_all(bias_groups=np.array([bias_group]),
                    overbias_voltage=overbias_voltage, tes_bias=tes_bias,
                    overbias_wait=overbias_wait, cool_wait=cool_wait,
                    high_current_mode=high_current_mode)
        else:
            if high_current_mode:
                self.set_tes_bias_high_current(bias_group)
            else:
                self.set_tes_bias_low_current(bias_group)

        # Read back the TES bias voltage
        tes_bias = self.get_tes_bias_bipolar(bias_group)

        # Loop over probe frequencies and take data
        datafile = np.array([], dtype='str')
        for i, pf in enumerate(probe_freq):
            self.log(f'Playing tone at {pf} Hz')

            # Play the TES tone
            self.play_sine_tes(bias_group, probe_amp, pf, dc_amp=tes_bias)
            probe_time = np.max([min_probe_time, n_cycle/pf])

            # Take data and safe datafile name - no freq mask for faster
            df = self.take_stream_data(probe_time, make_freq_mask=False)
            datafile = np.append(datafile, df)

            # Turn off tone
            self.stop_tes_bipolar_waveform(bias_group)

        if analyze:
            # Get sample frequency
            fs = self.get_flux_ramp_freq() * 1.0E3 / self.get_downsample_factor()

            # Calculate amplitudes
            amp = self.analyze_measure_tes_transfer(datafile, probe_freq,
                probe_amp, band=band, bias_group=bias_group,
                channel=ch[bg==bias_group], high_current_mode=high_current_mode,
                fs=fs, make_plot=make_plot, save_plot=save_plot,
                show_plot=show_plot, timestamp=timestamp, tes_bias=tes_bias)

            return amp

        else:
            return datafile


    def analyze_measure_tes_transfer(self, datafile, probe_freq,
            probe_amp, band=None, bias_group=None, channel=None, fs=None,
            high_current_mode=False, tes_bias=None, make_plot=False,
            save_plot=True, show_plot=False, timestamp=None):
        """
        """
        if fs is None:
            fs = self.get_flux_ramp_freq() * 1.0E3 / self.get_downsample_factor()

        if channel is None:
            channel = self.which_on(band)
        n_det = len(channel)
        n_freq = len(probe_freq)

        if timestamp is None:
            timestamp = self.get_timestamp()

        if show_plot:
            plt.ion()
        else:
            plt.ioff()

        # currents on lines
        if high_current_mode:
            r_inline = self.bias_line_resistance / self.high_low_current_ratio
        else:
            r_inline = self.bias_line_resistance
        i_bias = probe_amp / r_inline * 1.0E12  # Bias current in pA

        sa = np.zeros((n_freq, n_det))
        ca = np.zeros((n_freq, n_det))
        dd = {}
        mask = {}
        for i, pf in enumerate(probe_freq):
            # Load data
            _, d, m = self.read_stream_data(datafile[i])
            d *= (self._pA_per_phi0/2/np.pi)  # convert to pA
            d = np.transpose(d.T - np.mean(d.T, axis=0))  # mean subtract

            _, n_samp = np.shape(d)

            # sine/cosine decomp templates
            s = np.sin(2*np.pi*np.arange(n_samp) / fs *pf)
            c = np.cos(2*np.pi*np.arange(n_samp) / fs* pf)
            s /= np.sum(s**2)
            c /= np.sum(c**2)

            # cosine/sine decomposition
            for j, ch in enumerate(channel):
                idx = m[band, ch]
                sa[i, j] = np.dot(d[idx], s)
                ca[i, j] = np.dot(d[idx], c)

            # Store timestreams if making plots
            if make_plot:
                dd[i] = d
                mask[i] = m

        amp = np.sqrt(sa**2 + ca**2)
        norm_amp = amp/i_bias

        if make_plot:
            bbox = dict(boxstyle="round", ec='w', fc='w', alpha=.65)
            ax = {}
            for j, ch in enumerate(channel):
                # Instatiate figure
                fig = plt.figure(figsize=(9, 5.5))
                gs = gridspec.GridSpec(n_freq, 2, width_ratios=[2, 1])

                # Loop over probe frequencies
                for i, pf in enumerate(probe_freq):
                    ax[i] = plt.subplot(gs[i, 0])
                    if i > 0:
                        ax[i].get_shared_x_axes().join(ax[i], ax[i-1])

                    idx = mask[i][band, ch]

                    # Plot time and data
                    tt = np.arange(len(dd[i][idx])) / fs
                    ax[i].plot(tt, dd[i][idx])
                    ax[i].text(.98, .96, f'{pf:0.2f} Hz',
                        transform=ax[i].transAxes,
                        va='top', ha='right', bbox=bbox)

                    # Suppress x-tick labels
                    if i < len(probe_freq) - 1:
                        ax[i].axes.xaxis.set_ticklabels([])

                ax[len(probe_freq)-1].set_xlabel('Time [s]')

                # Summary plot
                axsm = plt.subplot(gs[:,1])
                axsm.semilogx(probe_freq, norm_amp[:,j], '.')
                axsm.set_ylabel(r'$dI_{TES}/dI_{b}$')
                axsm.set_xlabel('Probe Freq [Hz]')

                # Text label
                text = ''
                text += r'$f_{s}$: ' + f'{fs:0.1f}' + '\n'
                if tes_bias is not None:
                    text += r'$I_b$: ' + f'{tes_bias:1.2f}' + '\n'
                text += 'high bias: ' + f'{high_current_mode}' + '\n'
                axsm.text(.02, .02, text, transform=axsm.transAxes,
                    va='bottom', ha='left', bbox=bbox)

                fig.suptitle(f'{timestamp} b{band}ch{ch:03} BG{bias_group}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])

                # Saving
                if save_plot:
                    savename = f'{timestamp}_tes_transfer_b{band}ch{ch:03}bg{bias_group}.png'
                    plt.savefig(os.path.join(self.plot_dir, savename))

                # Showing plot
                if show_plot:
                    plt.show()
                else:
                    plt.close()

        return amp