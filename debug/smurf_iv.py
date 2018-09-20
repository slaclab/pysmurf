import numpy as np
from pysmurf.base import SmurfBase
import time

class SmurfIVMixin(SmurfBase):

    def slow_iv(self, daq, wait_time=.5, bias_high=2**15, bias_low=2**11, 
        bias_step=100):
        """
        Steps the TES bias down slowly. Starts at bias_high to bias_low with
        step size bias_step. Waits wait_time between changing steps.

        Args:
        -----
        daq (int) : The DAQ number

        Opt Args:
        ---------
        wait_time (float): The amount of time between changing TES biases in 
            seconds.
        bias_high (int): The maximum TES bias in bits.
        bias_low (int): The minimum TES bias in bits.
        bias_step (int): The step size in bits.
        """
        bias = np.arange(bias_high, bias_low, -bias_step)
        self.log('Starting TES bias ramp.', self.LOG_USER)

        for b in bias:
            self.set_tes_bias(daq, b)

        self.log('Done with TES bias ramp', self.LOG_USER)

    