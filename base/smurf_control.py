import numpy as np
import os
import sys
import time
from pysmurf.command.smurf_command import SmurfCommandMixin as SmurfCommandMixin
from pysmurf.util.smurf_util import SmurfUtilMixin as SmurfUtilMixin
from pysmurf.base.smurf_config import SmurfConfig as SmurfConfig

class SmurfControl(SmurfCommandMixin, SmurfUtilMixin):
    '''
    Base class for controlling Smurf. Loads all the mixins.
    '''

    def __init__(self, epics_root='mitch_epics', cfg_file=None, data_dir=None,
        name=None, make_logfile=True, **kwargs):
        '''
        Args:
        -----
        epics_root (string) : The epics root to be used. Default mitch_epics
        cfg_file (string) : Path the config file
        data_dir (string) : Path to the data dir
        '''
        super().__init__(epics_root=epics_root, **kwargs)

        if cfg_file is not None or data_dir is not None:
            self.initialize(cfg_file=cfg_file, data_dir=data_dir, name=name,
                make_logfile=make_logfile, **kwargs)

    def initialize(self, cfg_file, data_dir=None, name=None, 
        make_logfile=True, **kwargs):
        '''
        Initizializes SMuRF with desired parameters set in experiment.cfg.
        Largely stolen from a Cyndia/Shawns SmurfTune script
        '''

        self.config = SmurfConfig(cfg_file)

        # define data dir
        if data_dir is not None:
            self.data_dir = data_dir
        else:
            self.data_dir = self.config.get('data_dir')

        self.date = time.strftime("%Y%m%d")

        # name
        self.the_time = time.time()
        if name is None:
            name = '%10i' % (self.the_time)
        self.name = name

        self.base_dir = os.path.abspath(self.data_dir)

        # create output and plot directories
        self.output_dir = os.path.join(self.base_dir, name, 'outputs')
        self.plot_dir = os.path.join(self.base_dir, name, 'plots')
        self.make_dir(self.output_dir)
        self.make_dir(self.plot_dir)

        # name the logfile and create flags for it
        if make_logfile:
            self.log_file = os.path.join(self.output_dir, name + '.log')
            self.log.set_logfile(self.log_file)

        self.log('Initializing...', self.LOG_USER)

        self.set_defaults_pv()

        # The per band configs. May want to make available per-band values.
        smurf_init_config = self.config.get('init')
        bands = smurf_init_config['bands']
        for b in bands:
            self.set_iq_swap_in(b, smurf_init_config['iqSwapIn'], 
                write_log=True, **kwargs)
            self.set_iq_swap_out(b, smurf_init_config['iqSwapOut'], 
                write_log=True, **kwargs)
            self.set_ref_phase_delay(b, smurf_init_config['refPhaseDelay'], 
                write_log=True, **kwargs)
            self.set_ref_phase_delay_fine(b, 
                smurf_init_config['refPhaseDelayFine'], write_log=True, 
                **kwargs)
            self.set_tone_scale(b, smurf_init_config['toneScale'], 
                write_log=True, **kwargs)
            self.set_analysis_scale(b, smurf_init_config['analysisScale'], 
                write_log=True, **kwargs)
            self.set_feedback_enable(b, smurf_init_config['feedbackEnable'],
                write_log=True, **kwargs)
            self.set_feedback_gain(b, smurf_init_config['feedbackGain'], 
                write_log=True, **kwargs)
            self.set_lms_gain(b, smurf_init_config['lmsGain'], write_log=True,
                **kwargs)
            self.set_feedback_polarity(b, smurf_init_config['feedbackPolarity'], 
                write_log=True, **kwargs)
            self.set_band_center_mhz(b, smurf_init_config['bandCenterMHz'],
                write_log=True, **kwargs)
            self.set_synthesis_scale(b, smurf_init_config['synthesisScale'],
                write_log=True, **kwargs)
            self.set_dsp_enable(b, smurf_init_config['dspEnable'], 
                write_log=True, **kwargs)

    def make_dir(self, directory):
        """check if a directory exists; if not, make it

           Args:
            directory (str): path of directory to create
        """

        if not os.path.exists(directory):
            os.makedirs(directory)
