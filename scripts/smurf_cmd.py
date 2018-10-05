#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pysmurf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--bias-group', action='store', default=-1,
                        help='The bias group to set the TES bias',
                        choices=np.arange(8))
    parser.add_argument('--bias-voltage', action='store', default=-1,
                        help='The bias voltage to set')

    args = parser.parse_args()

    S = pysmurf.SmurfControl(make_logfile=False, setup=False)
    
    # S.set_tes_bias()
