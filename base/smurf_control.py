import numpy as np
import os
import sys
import time
from pysmurf.command.smurf_command import SmurfCommandMixin as SmurfCommandMixin
from pysmurf.util.smurf_util import SmurfUtilMixin as SmurfUtilMixin
from pysmurf.tune.smurf_tune import SmurfTuneMixin as SmurfTuneMixin
from pysmurf.debug.smurf_noise import SmurfNoiseMixin as SmurfNoiseMixin
from pysmurf.debug.smurf_iv import SmurfIVMixin as SmurfIVMixin
from pysmurf.base.smurf_config import SmurfConfig as SmurfConfig

class SmurfControl(SmurfCommandMixin, SmurfUtilMixin, SmurfTuneMixin, 
    SmurfNoiseMixin, SmurfIVMixin):
    '''
    Base class for controlling Smurf. Loads all the mixins.
    '''

    def __init__(self, epics_root='mitch_epics', 
        cfg_file='/home/cryo/pysmurf/cfg_files/experiment_fp28.cfg', 
        data_dir=None, name=None, make_logfile=True, output_dir_only=False,
        **kwargs):
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
                make_logfile=make_logfile, output_dir_only=output_dir_only,
                **kwargs)

    def initialize(self, cfg_file, data_dir=None, name=None, 
        make_logfile=True, output_dir_only=False, **kwargs):
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
        self.start_time = self.get_timestamp()
        if name is None:
            name = self.start_time
        self.name = name

        self.base_dir = os.path.abspath(self.data_dir)

        # create output and plot directories
        self.output_dir = os.path.join(self.base_dir, self.date, name, 
            'outputs')
        self.plot_dir = os.path.join(self.base_dir, self.date, name, 'plots')
        self.make_dir(self.output_dir)
        self.make_dir(self.plot_dir)

        # name the logfile and create flags for it
        if make_logfile:
            self.log_file = os.path.join(self.output_dir, name + '.log')
            self.log.set_logfile(self.log_file)
        else:
            self.log.set_logfile(None)

        # Dictionary for frequency response
        self.freq_resp = {}

        if not output_dir_only:
            self.log('Initializing...', self.LOG_USER)

            self.set_defaults_pv()

            # The per band configs. May want to make available per-band values.
            smurf_init_config = self.config.get('init')
            bands = smurf_init_config['bands']
            for b in bands:
                self.set_feedback_limit_khz(b, 225)  # why 225?

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
                self.set_lms_gain(b, smurf_init_config['lmsGain'], 
                    write_log=True, **kwargs)
                self.set_feedback_polarity(b, smurf_init_config['feedbackPolarity'], 
                    write_log=True, **kwargs)
                # self.set_band_center_mhz(b, smurf_init_config['bandCenterMHz'],
                #     write_log=True, **kwargs)
                self.set_synthesis_scale(b, smurf_init_config['synthesisScale'],
                    write_log=True, **kwargs)

                # This should be part of exp.cfg
                if b == 2:
                    self.set_data_out_mux(6, "UserData", write_log=True, 
                        **kwargs)
                    self.set_data_out_mux(7, "UserData", write_log=True, 
                        **kwargs)
                    self.set_iq_swap_in(b, 1, write_log=True, **kwargs)
                    self.set_iq_swap_out(b, 0, write_log=True, **kwargs)
                elif b ==3 :
                    self.set_data_out_mux(8, "UserData", write_log=True, 
                        **kwargs)
                    self.set_data_out_mux(9, "UserData", write_log=True, 
                        **kwargs)
                    self.set_iq_swap_in(b, 0, write_log=True, **kwargs)
                    self.set_iq_swap_out(b, 0, write_log=True, **kwargs)

                self.set_dsp_enable(b, smurf_init_config['dspEnable'], 
                    write_log=True, **kwargs)

                # Make band dictionaries
                self.freq_resp[b] = {}

            self.set_cpld_reset(0, write_log=True)

            for i in np.arange(1,5):
                self.set_att_uc(i, 0, write_log=True)
                self.set_att_dc(i, 0, write_log=True)

    def make_dir(self, directory):
        """check if a directory exists; if not, make it

           Args:
            directory (str): path of directory to create
        """

        if not os.path.exists(directory):
            os.makedirs(directory)

    def get_timestamp(self):
        """
        Returns:
        timestampe (str): Timestamp as a string
        """
        return '{:10}'.format(int(time.time()))
