import numpy as np

class SmurfCommandMixin():

    def __init__(self, **kwargs):
        print('Initializing SMuRF commands')

    def _caput(self, cmd, log=False, execute=True):
        '''
        Args:
        -----
        cmd : The pyrogue command to be exectued. Input as a string
        log : Whether to log the data or not. Default False
        execute : Whether to actually execute the command. Defualt True.
        '''
        if log:
            self.log(cmd)
        if execute:
            print('This needs to be implemented with pyrogue')