import numpy as np
import os
import sys
# import smurf_control.command
from pysmurf.command.smurf_command import SmurfCommandMixin as SmurfCommandMixin
from pysmurf.util.smurf_util import SmurfUtilMixin as SmurfUtilMixin

class SmurfControl(SmurfCommandMixin, SmurfUtilMixin):
    '''
    Base class for controlling Smurf
    '''

    def __init__(self, epics_root='mitch_epics', **kwargs):
        super().__init__(epics_root=epics_root, **kwargs)