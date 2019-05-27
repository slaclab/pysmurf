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

schema = Schema({
    'epics_root': And(str, len),

    'init': {

        Optional('dspEnable', default=1): And(Use(int),lambda n: n in (0,1)),
                
        And(Regex('band_[0-7]')) : {
                Optional('iq_swap_in', default=0): And(Use(int),lambda n: n in (0,1)),
                Optional('iq_swap_out', default=0): And(Use(int),lambda n: n in (0,1)),

                Optional('feedbackEnable', default=1): And(Use(int),lambda n: n in (0,1)),
                Optional('rfEnable', default=1): And(Use(int),lambda n: n in (0,1)),                

                # Number of cycles to delay phase reference
		'refPhaseDelay': And(Use(int),lambda n: 0 <= n < 2**3),
                # Finer phase reference delay, 307.2MHz clock ticks
		'refPhaseDelayFine': And(Use(int),lambda n: 0 <= n < 2**8),
                
		'att_uc': And(Use(int),lambda n: 0 <= n < 2**5),
		'att_dc': And(Use(int),lambda n: 0 <= n < 2**5),
		'amplitude_scale': And(Use(int),lambda n: 0 <= n < 2**4),
        }
    }
})

validated = schema.validate(loaded_config)

# build list of bands from which band_[0-7] blocks are present
r = re.compile("band_([0-7])")
validated['init']['bands']=sorted([int(m.group(1)) for m in (r.match(s) for s in validated['init'].keys()) if m])

print(validated)


