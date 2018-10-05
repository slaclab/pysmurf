#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import pysmurf

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Stupid log command
    parser.add_argument('--log', help='Logs input phrase', default=None,
                       action='store')
    
    # TES bias commands
    parser.add_argument('--tes-bias', action='store_true', default=False,
        help='Set the tes bias. Must also set --bias-group and --bias-voltage')
    parser.add_argument('--bias-group', action='store', default=-1, type=int,
        help='The bias group to set the TES bias', choices=np.arange(8, dtype=int))
    parser.add_argument('--bias-voltage', action='store', default=-1,
        help='The bias voltage to set')

    args = parser.parse_args()

    S = pysmurf.SmurfControl(cfg_file=os.path.join(os.path.dirname(__file__), '..', 
                                                   'cfg_files' ,'experiment_k7_17.cfg'),
        smurf_cmd_mode=True, setup=False)

    if args.log is not None:
        S.log(args.log)

    if args.tes_bias:
        S.set_tes_bias_bipolar(args.bias_group, args.bias_voltage, 
            write_log=True)

    
