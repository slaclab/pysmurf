import json
import io
import re
import os
import numpy as np
'''
Stolen from Cyndia/Shawns get_config.py
read or dump a config file
'''

class SmurfConfig:
    """Initialize, read, or dump a SMuRF config file.
       Will be updated to not require a json, which is unfortunate

    """

    def __init__(self, filename=None):
        self.filename = filename
        # self.config = [] # do I need to initialize this? I don't think so
        if self.filename is not None:
            self.read(update=True)

    def read_json(self, filename, comment_char='#'):
        """Reads a json config file

           Args:
        """
        no_comments=[]        
        with open(self.filename) as config_file:
            for idx, line in enumerate(config_file):
                if line.lstrip().startswith(comment_char):
                    # line starts with comment character - remove it
                    continue
                elif comment_char in line:
                    # there's a comment character on this line.
                    # ignore it and everything that follows
                    line=line.split('#')[0]
                    no_comments.append(line)
                else:
                    # will pass on to json parser
                    no_comments.append(line)

        loaded_config = json.loads('\n'.join(no_comments))
        return loaded_config
    
    def read(self, update=False):
        """Reads config file and updates the configuration.

           Args:
              update (bool): Whether or not to update the configuration.
        """
        loaded_config=self.read_json(self.filename)

        # validate
        validated_config=self.validate_config(loaded_config)
        
        if update:
            # put in some logic here to make sure parameters in experiment file match 
            # the parameters we're looking for
            self.config = validated_config

    def update(self, key, val):
        """Updates a single key in the config

           Args:
              key (any): key to update in the config dictionary
              val (any): value to assign to the given key
        """
        self.config[key] = val

    def write(self, outputfile):
        """Dumps the current config to a file

           Args:
              outputfile (str): The name of the file to save the configuration to.
        """
        ## dump current config to outputfile ##
        with io.open(outputfile, 'w', encoding='utf8') as out_file:
            str_ = json.dumps(self.config, indent = 4, separators = (',', ': '))
            out_file.write(str_)

    def has(self, key):
        """Reports if configuration has requested key.

           Args:
              key (any): key to check for in configuration dictionary.
        """
        
        if key in self.config:
            return True
        else:
            return False

    def get(self, key):
        """Returns configuration entry for requested key.  Returns
           None if key not present in configuration.

           Args:
              key (any): key whose configuration entry to retrieve.
        """
        
        if self.has(key):
            return self.config[key]
        else:
            return None

    def get_subkey(self, key, subkey):
        """
        Get the subkey value. A dumb thing that just formats strings for you.
        Will return None if it can't find stuff

        Args:
          key (any): key in config
          subkey (any): config subkey
        """

        if self.has(key):
            sub_dict = self.config[key]
            try:
                return sub_dict[subkey]
            except KeyError:
                print("Key found, but subkey not found")
                return None
        else:
            print("Key not found")
            return None

    def update_subkey(self, key, subkey, val):
        """
        More dumb wrappers for nested dictionaries.

        Args:
          key (any): key in config
          subkey (any): config subkey
          val (any): value to write
        """

        try: 
            self.config[key][subkey] = val
        except TypeError:
            self.config[key] = {} # initialize an empty dictionary first
            self.config[key][subkey] = val

    def validate_config(self,loaded_config):
        # Useful schema objects
        from pysmurf.base.schema import Schema, And, Use, Optional, Regex

        # Start with an extremely limited validation to figure out
        # things that we need to validate the entire configuration
        # file, like which bands are being used.  This also gets used
        # to construct the init:bands list, from which band_# are
        # present.
        band_schema = Schema({ 'init' : { Regex('band_[0-7]') : {} } },ignore_extra_keys=True)
        band_validated = band_schema.validate(loaded_config)

        # Build list of bands from which band_[0-7] blocks are present.  Will
        # use this to build the rest of the validation schema.
        r = re.compile("band_([0-7])")
        bands=sorted([int(m.group(1)) for m in (r.match(s) for s in band_validated['init'].keys()) if m])

        ###################################################
        ##### Building full validation schema
        schema_dict={}

        # Only used if none specified in pysmurf instantiation
        schema_dict['epics_root'] = And(str, len)

        #### Start specifiying init schema
        # init must be present
        schema_dict['init']={}

        # Explicitly sets dspEnable to this value in pysmurf setup.  Shouldn't
        # be necessary if defaults.yml is properly configured (dspEnable can
        # be set there, instead).
        schema_dict['init'][Optional('dspEnable', default=1)] = And(int,lambda n: n in (0,1))

        # Each band has a configuration block in init.

        # Default data_out_mux definitions.  Should match fw.
        default_data_out_mux_dict={}
        default_data_out_mux_dict[0]=[2,3]
        default_data_out_mux_dict[1]=[0,1]
        default_data_out_mux_dict[2]=[6,7]
        default_data_out_mux_dict[3]=[8,9]
        default_data_out_mux_dict[4]=[2,3]
        default_data_out_mux_dict[5]=[0,1]
        default_data_out_mux_dict[6]=[6,7]
        default_data_out_mux_dict[7]=[8,9]

        for band in bands:
            schema_dict['init']['band_%d'%band] = {

                # Swap IQ channels on input
                Optional('iq_swap_in', default=0): And(int,lambda n: n in (0,1)),
                # Swap IQ channels on output
                Optional('iq_swap_out', default=0): And(int,lambda n: n in (0,1)),

                # Global feedback enable
                Optional('feedbackEnable', default=1): And(int,lambda n: n in (0,1)),
                # Global feedback polarity
                Optional('feedbackPolarity', default=1): And(int,lambda n: n in (0,1)),
                # Global feedback gain (might no longer be used in dspv3).
                "feedbackGain" :  And(int,lambda n: 0 <= n < 2**16),
                # Global feedback gain (might no longer be used in dspv3). 
                "feedbackLimitkHz" : And(Use(float),lambda f: 0 < f),

                # Number of cycles to delay phase reference
                'refPhaseDelay': And(int,lambda n: 0 <= n < 2**4),
                # Finer phase reference delay, 307.2MHz clock ticks.  This
                # goes in the opposite direction as refPhaseDelay.
                'refPhaseDelayFine': And(int,lambda n: 0 <= n < 2**8),

                # RF attenuator on SMuRF output.  UC=up convert.  0.5dB steps.
                'att_uc': And(int,lambda n: 0 <= n < 2**5),
                # RF attenuator on SMuRF input.  DC=down convert.  0.5dB steps.        
                'att_dc': And(int,lambda n: 0 <= n < 2**5),
                # Tone amplitude.  3dB steps.
                'amplitude_scale': And(int,lambda n: 0 <= n < 2**4),

                # data_out_mux
                Optional("data_out_mux",default=default_data_out_mux_dict[band]) \
                  : And([ Use(int) ], list, lambda l: len(l)==2 and l[0]!=l[1] and all(ll>= 0 and ll<= 9 for ll in l) ),

                # Matches system latency for LMS feedback (9.6 MHz
                # ticks, use multiples of 52).  For dspv3 to adjust to
                # match refPhaseDelay*4 (ignore refPhaseDelayFine for
                # this).  If not provided and lmsDelay=None, sets to
                # lmsDelay = 4 x refPhaseDelay.
                Optional('lmsDelay',default=None) : And(int,lambda n: 0 <= n < 2**5),

                # Adjust trigRstDly such that the ramp resets at the flux ramp
                # glitch.  2.4 MHz ticks.
                'trigRstDly': And(int,lambda n: 0 <= n < 2**7),

                # LMS gain, powers of 2
                'lmsGain': And(int,lambda n: 0 <= n < 2**3),
            }
        #### Done specifying init schema

        #### Start specifying attenuator schema
        # Here's another one that users probably shouldn't be touching Just
        # doing basic validation here - not checking if they're distinct, for
        # instance.
        schema_dict["attenuator"] = {
            'att1' : And(int,lambda n: 0 <= n < 4),
            'att2' : And(int,lambda n: 0 <= n < 4),
            'att3' : And(int,lambda n: 0 <= n < 4),
            'att4' : And(int,lambda n: 0 <= n < 4)
        }    
        #### Done specifying attenuator schema

        #### Start specifying cryostat card schema
        ## SHOULD MAKE IT SO THAT WE JUST LOAD AND VALIDATE A SEPARATE,
        ## HARDWARE SPECIFIC CRYOSTAT CARD CONFIG FILE.  FOR NOW, JUST DO
        ## SOME VERY BASIC VALIDATION.
        def RepresentsInt(s):
            try: 
                int(s)
                return True
            except ValueError:
                return False

        schema_dict["pic_to_bias_group"] = { And(str,RepresentsInt) : int }
        schema_dict["bias_group_to_pair"] = { And(str,RepresentsInt) : [ int, int ] }
        #### Done specifying cryostat card schema

        #### Start specifiying amplifier
        schema_dict["amplifier"] = {
            # 4K amplifier gate voltage, in volts.
            "hemt_Vg" : Use(float),

            # Conversion from bits (the digital value the RTM DAC is set to)
            # to volts for the 4K amplifier gate.  Units are volts/bit.  An
            # important dependency is the voltage division on the cryostat
            # card, which can be different from cryostat card to cryostat card    
            "bit_to_V_hemt" : And(Use(float),lambda f: 0 < f ),
            # The 4K amplifier drain current is measured before a voltage
            # regulator, which also draws current.  An accurate measurement of
            # the 4K drain current requires subtracting the current drawn by
            # that regulator.  This is the offset to subtract off the measured
            # value, in mA.
            "hemt_Id_offset" : Use(float),
            # 50K amplifier gate voltage, in volts.    
            "LNA_Vg" : Use(float),
            # Which RTM DAC is wired to the gate of the 50K amplifier.
            "dac_num_50k" : And(int,lambda n: 1 <= n <= 32),
            # Conversion from bits (the digital value the RTM DAC is set to)
            # to volts for the 50K amplifier gate.  Units are volts/bit.  An
            # important dependency is the voltage division on the cryostat
            # card, which can be different from cryostat card to cryostat card
            "bit_to_V_50k" : And(Use(float),lambda f: 0 < f ),
            # Software limit on the minimum gate voltage that can be set for the 4K amplifier.
            "hemt_gate_min_voltage" : Use(float),
            # Software limit on the maximum gate voltage that can be set for the 4K amplifier.    
            "hemt_gate_max_voltage" :  Use(float)
        }
        #### Done specifiying amplifier

        #### Start specifying chip-to-frequency schema
        schema_dict[Optional('chip_to_freq',default={})] = {
            # [chip lower frequency, chip upper frequency] in GHz
            Optional(str) : And([ Use(float) ], list, lambda l: len(l)==2
                                and l[0]<l[1] and all(ll >= 4 and ll<= 8 for ll in l) )
        }
        #### Done specifying chip-to-frequency schema

        #### Start specifying tune parameter schema
        schema_dict['tune_band'] = {
            "grad_cut" : And(Use(float),lambda f: 0 < f),
            "amp_cut" : And(Use(float),lambda f: 0 < f),

            Optional('n_samples',default=2**18) : And(int,lambda n: 0 < n),
            # Digitizer sampling rate is 614.4 MHz, so biggest range of
            # frequencies user could possibly be interested in is between
            # -614.4MHz/2 and 614.4MHz/2, or -307.2MHz to +307.2MHz.
            Optional('freq_max', default=250.e6) : And(Use(float),lambda f: -307.2e6 <= f <= 307.2e6),
            Optional('freq_min', default=-250.e6) : And(Use(float),lambda f: -307.2e6 <= f <= 307.2e6),

            'fraction_full_scale' : And(Use(float),lambda f: 0 < f <=1.),
            Optional('default_tune',default=None) : And(str,os.path.isfile)
        }

        ## Add tuning params that must be specified per band.
        per_band_tuning_params= [
            ( 'lms_freq',And(Use(float),lambda f: 0 < f) ),
            ( 'delta_freq',And(Use(float),lambda f: 0 < f) ),            
            ( 'feedback_start_frac',And(Use(float),lambda f: 0 <= f <= 1) ),
            ( 'feedback_end_frac',And(Use(float),lambda f: 0 <= f <= 1) ),
            ( 'gradient_descent_gain',And(Use(float),lambda f: 0 < f) ),
            ( 'gradient_descent_averages',And(Use(int),lambda n: 0 < n) ),
            ( 'gradient_descent_converge_hz',And(Use(float),lambda f: 0 < f) ),
            ( 'gradient_descent_momentum',And(Use(int),lambda n: 0 <= n) ),
            ( 'gradient_descent_step_hz',And(Use(float),lambda f: 0 < f) ),
            ( 'gradient_descent_beta',And(Use(float),lambda f: 0 <= f <= 1) ),
            ( 'eta_scan_averages',And(Use(int),lambda n: 0 < n) ),
            ( 'eta_scan_del_f', And(Use(int), lambda n: 0 < n) ),
            ( 'eta_scan_amplitude', And(Use(int), lambda n: 0 < n) ),
        ]

        for band in bands:
            for (p,v) in per_band_tuning_params:
                if band==bands[0]:
                    schema_dict['tune_band'][p]={}            
                schema_dict['tune_band'][p][str(band)] = v
        ## Done adding tuning params that must be specified per band.        

        #### Done specifying tune parameter schema

        #### Start specifying bad mask
        schema_dict[Optional('bad_mask', default = {})] = {
            # Why are these indexed by integers that are also strings?
            # I don't think the keys here are used at all.
            Optional(str) : And([ Use(float) ], list, lambda l: len(l)==2
                                and l[0]<l[1] and all(ll >= 4000 and ll<= 8000 for ll in l) )
        }
        #### Done specifying bad mask

        #### Start specifying TES-related
        # TES shunt resistance
        schema_dict["R_sh"] = And(Use(float),lambda f: 0 < f )

        # Round-trip resistance on TES bias lines, in low current mode.
        # Includes the resistance on the cryostat cards, and cable resistance.
        schema_dict["bias_line_resistance"] = And(Use(float),lambda f: 0 < f )

        # Ratio between the current per DAC unit in high current mode to the
        # current in low current mode.  Constained to be greater than or equal
        # to 1 since otherwise what does high current mode EVEN MEAN.
        schema_dict["high_low_current_ratio"] = And(Use(float),lambda f: 1 <= f )

        # If 1, TES biasing will *always* be in high current mode.
        schema_dict[Optional('high_current_mode_bool', default=0)] = And(int,lambda n: n in (0,1))

        # All SMuRF bias groups with TESs connected.
        schema_dict["all_bias_groups"] = And([ int ], list, lambda l: all(ll >= 0 and ll < 16 for ll in l))
        #### Done specifying TES-related

        #### Start specifying flux ramp-related
        schema_dict["flux_ramp"] = {
            # 0x1 selects fast flux ramp, 0x0 selects slow flux ramp.  The
            # slow flux ramp only existed on the first rev of RTM boards,
            # C0, and wasn't ever really used.
            Optional("select_ramp", default=1) : And(int,lambda n: n in (0,1)),
            # 20 bits for the C0 RTMs, 32 bits for the C1 RTMs.
            "num_flux_ramp_counter_bits" : And(int,lambda n: n in (20,32))
        }
        #### Done specifying flux-ramp related

        #### Start specifying constants schema
        # If all of a schema dictionary's keys are optional, must specify
        # them both in the schema key for that dictionary, and in the
        # schema for that dictionary. 
        constants_default_dict={ 'pA_per_phi0' : 9e6 }
        cdd_key=Optional("constant",default=constants_default_dict)
        schema_dict[cdd_key]={}
        # Assumes all constants default values are floats
        for key, value in constants_default_dict.items():    
            schema_dict[cdd_key][Optional(key,default=value)] = Use(float)
        #### Done specifying constants schema

        #### Start thermal schema
        # OT protection for ultrascale FPGA, in degrees C.  If None,
        # then pysmurf doesn't try to engage OT protection.  For
        # unknown reasons, enabling OT protection in the ELMA crate
        # we've been using for testing on campus at Stanford takes
        # down the carrier after the third command in the enable
        # sequence (set_ot_threshold_disable(0)), but it works in the
        # RF lab at SLAC where they've been testing with an ASIS
        # crate.  Shawn has yet to have this work for him.  Newer fw
        # versions will have OT protection enabled in the fw.
        schema_dict[Optional('ultrascale_temperature_limit_degC',default=None)] = And(Use(float),lambda f: 0 <= f <= 99)
        #### Done specifying thermal schema

        #### Start specifying timing-related schema
        schema_dict["timing"] = {
            # "ext_ref" : internal oscillator locked to an external
            #   front-panel reference, or unlocked if there is no front
            #   panel reference.  (LmkReg_0x0147 : 0x1A).  Also sets
            #   flux_ramp_start_mode=0
            # "backplane" : takes timing from timing master through
            #   backplane.  Also sets flux_ramp_start_mode=1.
            "timing_reference" : And(str,lambda s: s in ('ext_ref','backplane'))
        }
        #### Done specifying timing-related schema

        #### Start specifying smurf2mce
        # System should be smart enough to determine fs on the fly.
        schema_dict["fs"] = And(Use(float),lambda f: 0 < f )

        userHasWriteAccess = lambda dirpath : os.access(dirpath,os.W_OK)    
        def fileDirExistsAndUserHasWriteAccess(file_path):
            filedir_path=os.path.dirname(file_path)
            if not os.path.isdir(filedir_path):
                return False
            elif not userHasWriteAccess(filedir_path):
                return False
            else:
                return True

        isValidPort = lambda port_number : (1 <= int(port_number) <= 65535)        
        # Some documentation on some of these parameters is here -
        # https://confluence.slac.stanford.edu/display/SMuRF/SMURF2MCE
        schema_dict["smurf_to_mce"] = {
            # smurf2mce configuration files.  pysmurf generates these
            # files on the fly, so just need to make sure the directory #
            # exists and is writeable
            Optional("smurf_to_mce_file",default="/data/smurf2mce_config/smurf2mce.cfg") \
               : And(str,fileDirExistsAndUserHasWriteAccess),
            Optional("mask_file",default="/data/smurf2mce_config/mask.txt") \
               : And(str,fileDirExistsAndUserHasWriteAccess),

            # Whether or not to dynamically generate the gcp mask
            # everytime you stream data based on which channels have tones
            # assigned and on.
            Optional('static_mask',default=0) : And(int,lambda n: n in (0,1)),

            # 0 means use MCE sync word to determine when data is sent
            # Any positive number means average that many SMuRF frames
            #    before sending out the averaged frame
            # Any negative number means "please crash horribly". Same for
            #    anything that isn't a number
            'num_averages' : And(int,lambda n: n>=0),

            # The “file_name_extend” parameter now makes a new file every
            # “data_frames” frames.  Then files are written as long as the
            # writer is enabled.  E.g.
            #  <file_name>.part_00000
            #  <file_name>.part_00001
            #  ...
            'file_name_extend' : And(int,lambda n: n in (0,1)),

            # 0 in this field disables file writing
            # Setting a positive number causes a new file to be written
            #    after that many AVERAGED frames.
            # Setting a negative number causes the program to crash
            #    horribly
            'data_frames' : And(int,lambda n: n>=0),

            # Optional kludge to account for a circshift offset.
            # Implemented to get arounda bug in channel number in early
            # versions of the DSPv3 fw (specifically, fw version
            # mitch_4_30).  If not specified, no offset is applied.
            Optional('mask_channel_offset',default=0) : Use(int),

            # This is the port used for communication. Must match the port
            # set at the receive end
            Optional('port_number',default='3334') : And(str,RepresentsInt,isValidPort),

            # Could and should improve validation on this one if it's ever
            # used again.  This default is also ridiculous.
            Optional('receiver_ip', default='tcp://192.168.3.1:3334') : str,

            # Filter params
            'filter_freq' : And(Use(float),lambda f: f>0),
            'filter_order' : And(int,lambda n: n>=0),
            Optional('filter_gain',default=1.0) : Use(float)
        }
        #### Done specifying smurf2mce

        #### Start specifying directories
        schema_dict[Optional("default_data_dir",default="/data/smurf_data")] = And(str,os.path.isdir,userHasWriteAccess)
        schema_dict[Optional("smurf_cmd_dir",default="/data/smurf_data/smurf_cmd")] = And(str,os.path.isdir,userHasWriteAccess)
        schema_dict[Optional("tune_dir",default="/data/smurf_data/tune")] = And(str,os.path.isdir,userHasWriteAccess)
        schema_dict[Optional("status_dir",default="/data/smurf_data/status")] = And(str,os.path.isdir,userHasWriteAccess)
        #### Done specifying directories

        ##### Done building validation schema
        ###################################################

        ###################################################
        # Validate the full config
        schema = Schema(schema_dict)
        validated_config = schema.validate(loaded_config)

        ###################################################
        # Higher level/composite validation, if schema validation
        # succeeds

        # Check that no DAC has been assigned to multiple TES bias groups
        bias_group_to_pair=validated_config['bias_group_to_pair']        
        tes_bias_group_dacs=np.ndarray.flatten(np.array([bg2p[1] for bg2p in bias_group_to_pair.items()]))
        assert ( len(np.unique(tes_bias_group_dacs)) == len(tes_bias_group_dacs) ), 'Configuration failed - DACs may not be assigned to multiple TES bias groups.'
        
        # Check that the DAC specified as the 50K gate driver
        # isn't also defined as one of the DACs in a TES bias group
        # pair.
        dac_num_50k=validated_config['amplifier']['dac_num_50k']
        # Taking the first element works because we already required
        # that no DAC show up in more than one TES bias group
        # definition.
        assert (dac_num_50k not in tes_bias_group_dacs),'Configuration failed - DAC requested for driving 50K amplifier gate, %d, is also assigned to TES bias group %d.'%(int(dac_num_50k),int([bg2p[0] for bg2p in bias_group_to_pair.items() if dac_num_50k in bg2p[1]][0]))
        
        ##### Done with higher level/composite validation.
        ###################################################
        
        # Splice in the sorted init:bands key before returning
        validated_config['init']['bands']=bands
        
        return validated_config            
