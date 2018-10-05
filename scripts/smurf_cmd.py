#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'cfg_files'))
import pysmurf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # TES bias commands
    parser.add_argument('--tes-bias', action='store_true', default=False,
        help='Set the tes bias. Must also set --bias-group and --bias-voltage')
    parser.add_argument('--bias-group', action='store', default=-1,
        help='The bias group to set the TES bias', choices=np.arange(8))
    parser.add_argument('--bias-voltage', action='store', default=-1,
        help='The bias voltage to set')

    args = parser.parse_args()

    S = pysmurf.SmurfControl(cfg_file='experiment_k7_17.cfg', 
        smurf_cmd_mode=True, setup=False)

    if args.tes_bias:
        S.set_tes_bias_bipolar(args.bias_group, args.bias_voltage, 
            write_log=True)

    
