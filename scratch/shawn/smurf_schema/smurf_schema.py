import json
import re
import logging
import os

#filename='c:/Users/shawn/Docuexperiment_fp29_srv10_dspv3_cc02-02_hbOnlyBay0.cfg'
filename='test.cfg'

comment_char='#'
with open(filename) as config_file:
    no_comments=[]
    for idx, line in enumerate(config_file):
        if line.lstrip().startswith(comment_char):
            # line starts with comment character - remove it
            continue
        elif comment_char in line:
            # there is a comment character on this line.
            # ignore it and everything that follows
            line=line.split('#')[0]
            no_comments.append(line)
        else:
            # will pass on to json parser
            no_comments.append(line)
            
loaded_config = json.loads('\n'.join(no_comments)) 

### schema stars here
from schema import Schema, And, Or, Use, Optional, Regex, Forbidden, Hook, Schema

class Deprecated(Hook):
    def __init__(self, *args, **kwargs):
        kwargs["handler"] = lambda key, *args: logging.warn(f"`{key}` is deprecated. " + (self._error or ""))
        super(Deprecated, self).__init__(*args, **kwargs)

# Start with an extremely limited validation to figure out things that
# we need to validate the entire configuration file, like which bands
# are being used.  
band_schema = Schema({ 'init' : { Regex('band_[0-7]') : {} } },ignore_extra_keys=True)
band_validated = band_schema.validate(loaded_config)

# Build list of bands from which band_[0-7] blocks are present.  Will
# use this to build the rest of the validation schema.
r = re.compile("band_([0-7])")
bands=sorted([int(m.group(1)) for m in (r.match(s) for s in band_validated['init'].keys()) if m])

print(bands)

schema_dict={}

##### Building validation schema

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
for band in bands:
    schema_dict['init']['band_%d'%band] = {
        
        # Swap IQ channels on input
        Optional('iq_swap_in', default=0): And(int,lambda n: n in (0,1)),
        # Swap IQ channels on output
        Optional('iq_swap_out', default=0): And(int,lambda n: n in (0,1)),
        
        # Global feedback enable
        Optional('feedbackEnable', default=1): And(int,lambda n: n in (0,1)),
        
        # Number of cycles to delay phase reference
	'refPhaseDelay': And(int,lambda n: 0 <= n < 2**3),
        # Finer phase reference delay, 307.2MHz clock ticks.  This
        # goes in the opposite direction as refPhaseDelay.
	'refPhaseDelayFine': And(int,lambda n: 0 <= n < 2**8),
        
	'att_uc': And(int,lambda n: 0 <= n < 2**5),
	'att_dc': And(int,lambda n: 0 <= n < 2**5),
	'amplitude_scale': And(int,lambda n: 0 <= n < 2**4),
        
        # Matches system latency for LMS feedback (9.6 MHz ticks, use
        # multiples of 52).  For dspv3 to adjust to match
        # refPhaseDelay*4 (ignore refPhaseDelayFine for this).
        'lmsDelay': And(int,lambda n: 0 <= n < 2**5),
        
	# Adjust trigRstDly such that the ramp resets at the flux ramp
	# glitch.  2.4 MHz ticks.
        'trigRstDly': And(int,lambda n: 0 <= n < 2**7),

        # LMS gain, powers of 2
        'lmsGain': And(int,lambda n: 0 <= n < 2**3),
        
        # The user really shouldn't be touching these - better to let
        # them be set in defaults.yml.
	#'analysisScale' : 2,
	#"synthesisScale": 2,
	#"toneScale" : 2,
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
    ( 'feedback_start_frac',And(Use(float),lambda f: 0 <= f <= 1) ),
    ( 'feedback_end_frac',And(Use(float),lambda f: 0 <= f <= 1) ),
    ( 'gradient_descent_gain',And(Use(float),lambda f: 0 < f) ),
    ( 'gradient_descent_averages',And(Use(int),lambda n: 0 < n) ),
    ( 'eta_scan_averages',And(Use(int),lambda n: 0 < n) ),        
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

#### Start specifying timing-related
schema_dict["timing"] = {
    # "ext_ref" : internal oscillator locked to an external
    #   front-panel reference, or unlocked if there is no front
    #   panel reference.  (LmkReg_0x0147 : 0x1A).  Also sets
    #   flux_ramp_start_mode=0
    # "backplane" : takes timing from timing master through
    #   backplane.  Also sets flux_ramp_start_mode=1.
    "timing_reference" : And(str,lambda s: s in ('ext_ref','backplane'))
}
#### Done specifying timing-related

#### Start specifying directories
userHasWriteAccess = lambda dirpath : os.access(dirpath,os.W_OK)
schema_dict[Optional("default_data_dir",default="/data/smurf_data")] = And(str,os.path.isdir,userHasWriteAccess)
schema_dict[Optional("smurf_cmd_dir",default="/data/smurf_data/smurf_cmd")] = And(str,os.path.isdir,userHasWriteAccess)
schema_dict[Optional("tune_dir",default="/data/smurf_data/tune")] = And(str,os.path.isdir,userHasWriteAccess)
schema_dict[Optional("status_dir",default="/data/smurf_data/status")] = And(str,os.path.isdir,userHasWriteAccess)
#### Done specifying directories

##### Done building validation schema

# Validate the full config
schema = Schema(schema_dict)
validated = schema.validate(loaded_config)
print(validated)

sys.exit(1)

#
#    "constant" : {
#	"pA_per_phi0" : Use(float)
#    }
#
#}



