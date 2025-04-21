import os
import sys
import math
#import matplotlib.pyplot as plt # not sure we need this?
import numpy as np # pyepics reads stuff as numpy arrays if possible
import subprocess
#import epics
from smurf_setup.stage.tuningstage import SmurfStage
from smurf_setup.util.cryochannel import *
from smurf_setup.util.smurftune import *

"""scan resonances and get eta parameters for locking onto resonances
"""


SysgenCryo =  "AMCc:FpgaTopLevel:AppTop:AppCore:SysgenCryo:"
CryoChannels = "CryoChannels:"

def on_change(pvname=None, value=None, char_value=None, **kw):
    return True 

def eta_scan(epics_path, subchan, freqs, drive):
    """scan a small range and get IQ response
    
       Args:
        epics_path (str): root path for epics commands for eta scanning
        subchan (int): subchannel to scan; should be n_subchan * subband_no
        freqs (list): frequencies to scan over
        drive (int): power at which to scan
    """

    pv_list = ["etaScanFreqs", "etaScanAmplitude", "etaScanChannel", \
            "etaScanDwell"]
    pv_vals = [freqs, drive, subchan, 0] # make 0 a variable?

    for i in range(len(pv_list)):
        epics.caput(pv_list[i], pv_vals[i])


    #while False:
        # set a monitor
    #epics.camonitor(epics_path + "etaScanResultsReal", writer=None, on_change)
    epics.caput(epics_path + "runEtaScan", 1)

    I = epics.caget(epics_path + "etaScanResultsReal", count = len(freqs), use_monitor=False)
    Q = epics.caget(epics_path + "etaScanResultsImag", count = len(freqs), use_monitor=False)

    epics.camonitor_clear(epics_path + "etaScanResultsReal")

    if I > 2**23:
        I = I - 2**24
    if Q > 2**23:
        Q = Q - 2**24

    I = I / (2**23)
    Q = Q / (2**23)

    response = I + 1j * Q
    return response, freqs

def subband_off(smurfCfg, subchan, freqs):
    """turn off a single subband

       Args:
        smurfCfg (config object)
        subchan (int): channel number
        freqs (list): freqs to disable
    """
    config_cryo_channel(smurfCfg, subchan, freqs, 0, 0, 0, 0)
    return

def estimate_params(smurfStageCfg, response, freqs):
    """estimate eta parameters from a set of scanned frequencies

       Args:
        smurfStageCfg (config) : stage specific piece of a smurfCfg object
        response (list?): response (I + i*Q) from eta scan
        freqs (list): frequencies scanned. same length as response.
    """
    abs_response = np.abs(response)
    idx = abs_response.index(min(abs_response))
    center_freq = freqs[idx]

    search_width = smurfStageCfg['search_width']
    left = freqs.index(center_freq - search_width)
    right = freqs.index(center_freq + search_width)

    net_phase = np.unwrap(np.angle(response))

    latency = (net_phase[-1] - net_phase[0]) / (freqs[-1] - freqs[0])

    eta = (freqs[right] - freqs[left]) / (response[right] - response[left])
    eta_mag = np.abs(eta)
    eta_phase = np.angle(eta)
    eta_phase_deg = eta_phase * 180 / math.pi
    eta_scaled = eta_mag / 19.2 # take out this hard coded number

    return eta, center_freq, latency, eta_phase_deg, eta_scaled



class etaParams(SmurfStage):
    """class to tune the eta parameters. Inherits from the more generic tuning
       stage class
    """

    def __init__(self, tuning):
        self._nickname = "eta_params"

        super().__init__(tuning)

        #super(etaParams, self).__init__() # python 2 for testing

        initconfig = self.config.get('init')
        self.epics_root = initconfig['epics_root']
        self.bandNo = initconfig['band']
        self.stage_config = tuning.config.get('etaParams')

    def prepare(self, resfile=None):
        """prepare system to scan for eta parameters by reading in resonance
            locations
           
           Args:
            resfile (str): optional path resonance locations
            If none supplied, then most recent _resloc file is used

           Outputs:
            freqs (list): list of frequency locations
        """
        if resfile is not None:
            reslocs = resfile
        else:
            resfiles = sorted([f for f in os.listdir(self.outputdir) if
            "res_locs" in f]) # move nickname to a variable?
            reslocs = resfiles[-1] # grab the most recent

        data = np.loadtext(reslocs, dtype=float, delimiter=',', skiprows=1)
        freqs = data[:,0] # assumes frequencies are in the first column
        self.freqs = freqs
        return freqs

    def run(self):
        baseStr = "Base[{0}]:".format(self.bandNo)
        baseRootPath = self.epics_root + SysgenCryo + baseStr
        datadir = self.outputdir
        resultsDir = self.outputdir + "/etaScan/"
        if not os.path.exists(resultsDir):
            os.makedirs(resultsDir)

        FweepFHalf = self.stage_config['FSweepFHalf']
        FsweepDf = self.stage_config['FsweepDf']

        off(self.bandNo)

        bandCenterMHz = epics.caget(baseRootPath + 'bandCenterMHz', use_monitor=False)
        n_channels = epics.caget(baseRootPath + 'numberChannels', use_monitor=False)
        n_subbands = epics.caget(baseRootPath + 'numberSubBands', use_monitor=False)

        n_channelspersubband = n_channels // n_subbands
        max_channelspersubband = n_channelspersubband # not sure how long we need this constraint

        digitizerFreqMHz = epics.caget(baseRootPath, 'digitizerFrequencyMHz', use_monitor=False)
        subBandHalfWidthMHz = digitizerFreqMHz / n_subbands

        subbandchans = np.zeros((1, n_subbands))
        scanchans = np.zeros((1, n_channels))

        n_scanfreqs = len(np.arange(-FsweepFHalf, FweepFHalf + FsweepDf, FsweepDf))
        scanFreqs = np.zeros((n_scanfreqs, n_channels))

        for x in range(len(self.freqs)):




    def run_old(self):
        n_channels = epics.caget(self.epics_root + SysgenCryo \
                + "numberChannels", use_monitor=False) # should be 512 for a 500MHz band
        n_subbands = epics.caget(self.epics_root + SysgenCryo \
                + "numberSubBands", use_monitor=False) # is 32 right now

        n_subchannels = n_channels / n_subbands # 16

        epics_path = self.epics_root + CryoChannels
        
        freqs = self.freqs

        try:
            drive = self.stage_config['drive_power']
        except KeyError:
            drive = 10 # default to -6dB unless specified

        try:
            sweep_width = self.stage_config['sweep_width']
        except KeyError:
            sweep_width = 0.3 # default to 300kHz

        try: 
            sweep_df = self.stage_config['sweep_df']
        except KeyError:
            sweep_df = 0.005 # default to 5 kHz

        band_center = epics.caget(self.epics_root + SysgenCryo \
                + "bandCenterMHz", use_monitor=False) 
        subband_order = epics.caget(self.epics_root + SysgenCryo \
                + "subBandOrder", use_monitor=False)

        results = np.zeros((len(freqs), 7))

        for ii in range(len(freqs)):
            f = freqs[i]
            subband, offset = freq_to_subband(f, band_center, subband_order)
            
            scan_fs = np.arange(offset - sweep_width, offset + sweepwidth,\
                    sweep_df)

            resp, f = eta_scan(epics_path, subband * n_subchannels, scan_fs, drive)
            eta, center_freq, latency, eta_phase_deg, eta_scaled = \
                    estimate_params(self.stage_config, resp, f)

            subband_off(self.config, subband * n_subchannels, scan_fs)
            results[ii,:] = [center_freq, subband, offset, eta_phase_deg,\
                    eta_scaled, latency, eta]
        header = "center_freq, subband_no, offset, eta_phase_deg, eta_scaled,\
                latency, eta"

        self.write(results, header) # this was defined in the superclass
        self.results = results
        
        return results
    

    def analyze(self):
        """make & save some plots and stuff
        """
        
        

    def clean(self):
        pass
