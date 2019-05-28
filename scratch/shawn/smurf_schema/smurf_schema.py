import json
import re
import logging

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
        # Finer phase reference delay, 307.2MHz clock ticks
	'refPhaseDelayFine': And(int,lambda n: 0 <= n < 2**8),
        
	'att_uc': And(int,lambda n: 0 <= n < 2**5),
	'att_dc': And(int,lambda n: 0 <= n < 2**5),
	'amplitude_scale': And(int,lambda n: 0 <= n < 2**4),
    }

#### Done specifying init schema

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

#### Start specifying bad mask
schema_dict[Optional('bad_mask', default = {})] = {
    # Why are these indexed by integers that are also strings?
    # I don't think the keys here are used at all.
    Optional(str) : And([ Use(float) ], list, lambda l: len(l)==2
                        and l[0]<l[1] and all(ll >= 4000 and ll<= 8000 for ll in l) )
}
#### Done specifying bad mask

#### Start specifying TES-related 
schema_dict["R_sh"] = And(Use(float),lambda f: 0 < f )
schema_dict["bias_line_resistance"] = And(Use(float),lambda f: 0 < f )
schema_dict["high_low_current_ratio"] = And(Use(float),lambda f: 1 <= f )
schema_dict[Optional('high_current_mode_bool', default=0)] = And(int,lambda n: n in (0,1))
schema_dict["all_bias_groups"] = And([ int ], list, lambda l: all(ll >= 0 and ll < 16 for ll in l))
#### Done specifying TES-related 

##### Done building validation schema

# Validate the full config
schema = Schema(schema_dict)
validated = schema.validate(loaded_config)
print(validated)

sys.exit(1)

#    "flux_ramp" : {
#        # 0x1 selects fast flux ramp, 0x0 selects slow flux ramp.  The
#        # slow flux ramp only existed on the first rev of RTM boards,
#        # C0, and wasn't ever really used.
#	Optional("select_ramp", default=1) : int,
#        # 20 bits for the C0 RTMs, 32 bits for the C1 RTMs.
#	"num_flux_ramp_counter_bits" : And(int,lambda n: n in (20,32))
#    },
#
#    "constant" : {
#	"pA_per_phi0" : Use(float)
#    }
#
#}



