import ast
# functions and their arguments to check
# specified here: https://www.overleaf.com/project/5e837cac9659910001e5f71e
interface = {
    "python/pysmurf/client/tune/smurf_tune.py": {  # file
        "SmurfTuneMixin": {                        # class
            "plot_tune_summary": {                 # function and args
                "args": [
                    'self', 'band', 'eta_scan', 'show_plot', 'save_plot', 'eta_width', 'channels', 'plot_summary', 'plotname_append'
                ],
                "defaults": [
                    'False', 'False', 'True', '0.3', 'None', 'True', "''"
                ]
            },
            "full_band_resp": {
                "args": [
                    'self', 'band', 'n_scan', 'nsamp', 'make_plot', 'save_plot', 'show_plot', 'save_data', 'timestamp', 'save_raw_data', 'correct_att', 'swap', 'hw_trigger', 'write_log', 'return_plot_path', 'check_if_adc_is_saturated'
                ],
                "defaults": [
                    '1', '2 ** 19', 'False', 'True', 'False', 'False', 'None', 'False', 'True', 'False', 'True', 'False', 'False', 'True'
                ]
            },
            "tracking_setup": {
                "args": [
                    'self', 'band', 'channel', 'reset_rate_khz', 'write_log', 'make_plot', 'save_plot', 'show_plot', 'nsamp', 'lms_freq_hz', 'meas_lms_freq', 'meas_flux_ramp_amp', 'n_phi0', 'flux_ramp', 'fraction_full_scale', 'lms_enable1', 'lms_enable2', 'lms_enable3', 'feedback_gain', 'lms_gain', 'return_data', 'new_epics_root', 'feedback_start_frac', 'feedback_end_frac', 'setup_flux_ramp', 'plotname_append'
                ],
                "defaults": [
                    'None', 'None', 'False', 'False', 'True', 'True', '2 ** 19', 'None', 'False', 'False', '4', 'True', 'None', 'True', 'True', 'True', 'None', 'None', 'True', 'None', 'None', 'None', 'True', "''"
                ]
            },
            "flux_ramp_setup": {
                "args": [
                    'self', 'reset_rate_khz', 'fraction_full_scale', 'df_range', 'band', 'write_log', 'new_epics_root'
                ],
                "defaults": [
                    '0.1', '2', 'False', 'None'
                ]
            },
            "find_freq": {
                "args": [
                    'self', 'band', 'start_freq', 'stop_freq', 'subband', 'tone_power', 'n_read', 'make_plot', 'save_plot', 'plotname_append', 'window', 'rolling_med', 'make_subband_plot', 'show_plot', 'grad_cut', 'flip_phase', 'grad_kernel_width', 'amp_cut', 'pad', 'min_gap'
                ],
                "defaults": [
                    '-250', '250', 'None', 'None', '2', 'False', 'True', "''", '50', 'True', 'False', 'False', '0.05', 'False', '8', '0.25', '2', '2'
                ]
            },
            "setup_notches": {
               "args": [
                    'self', 'band', 'resonance', 'tone_power', 'sweep_width', 'df_sweep', 'min_offset', 'delta_freq', 'new_master_assignment', 'lock_max_derivative', 'scan_unassigned'
                ],
                "defaults": [
                    'None', 'None', '0.3', '0.002', '0.1', 'None', 'False', 'False', 'False'
                ] 
            }
        }
    },
    "python/pysmurf/client/command/smurf_command.py": {
        "SmurfCommandMixin": {
            "run_serial_eta_scan": {
                "args": [
                    'self', 'band', 'sync_group', 'timeout'
                ],
                "defaults": [
                    'True', '240'
                ]
            },
            "run_serial_gradient_descent": {
                "args": [
                    'self', 'band', 'sync_group', 'timeout'
                ],
                "defaults": [
                    'True', '240'
                ]
            },
            "set_amplitude_scale_array": {
                "args": [
                    'self', 'band', 'val'
                ],
                "defaults": [
                ]
            },
            "set_stream_enable": {
                "args": [
                    'self', 'val'
                ],
                "defaults": [
                ]
            },
            "set_amplifier_bias": {
                "args": [
                    'self', 'bias_hemt', 'bias_50k'
                ],
                "defaults": [
                    'None', 'None'
                ]
            },
            "set_cryo_card_ps_en": {
                "args": [
                    'self', 'enable', 'write_log'
                ],
                "defaults": [
                    '3', 'False'
                ]
            }
        }
    },
    "python/pysmurf/client/util/smurf_util.py": {
        "SmurfUtilMixin": {
            "take_stream_data": {
                "args": [
                    'self', 'meas_time', 'downsample_factor', 'write_log', 'update_payload_size', 'reset_unwrapper', 'reset_filter', 'return_data', 'make_freq_mask', 'register_file'
                ],
                "defaults": [
                    'None', 'True', 'True', 'True', 'True', 'False', 'True', 'False'
                ]
            },
            "stream_data_on": {
                "args": [
                    'self', 'write_config', 'data_filename', 'downsample_factor', 'write_log', 'update_payload_size', 'reset_filter', 'reset_unwrapper', 'make_freq_mask', 'channel_mask', 'make_datafile', 'filter_wait_time'
                ],
                "defaults": [
                    'False', 'None', 'None', 'True', 'True', 'True', 'True', 'True', 'None', 'True', '0.1'
                ]
            },
            "stream_data_off": {
                "args": [
                    'self', 'write_log', 'register_file'
                ],
                "defaults": [
                    'True', 'False'
                ]
            },
            "read_stream_data": {
                "args": [
                    'self', 'datafile', 'channel', 'nsamp', 'array_size', 'return_header', 'return_tes_bias', 'write_log', 'n_max', 'make_freq_mask', 'gcp_mode'
                ],
                "defaults": [
                    'None', 'None', 'None', 'False', 'False', 'True', '2048', 'False', 'False'
                ]
            },
            "which_on": {
                "args": [
                    'self', 'band'
                ],
                "defaults": [
                ]
            },
            "band_off": {
                "args": [
                    'self', 'band'
                ],
                "defaults": [
                ]
            },
            "channel_off": {
                "args": [
                    'self', 'band', 'channel'
                ],
                "defaults": [
                ]
            },
            "set_tes_bias_bipolar_array": {
                "args": [
                    'self', 'bias_group_volt_array', 'do_enable'
                ],
                "defaults": [
                    'False'
                ]
            },
            "set_tes_bias_high_current": {
                "args": [
                    'self', 'bias_group', 'write_log'
                ],
                "defaults": [
                    'False'
                ]
            },
            "set_tes_bias_low_current": {
                "args": [
                    'self', 'bias_group', 'write_log'
                ],
                "defaults": [
                    'False'
                ]
            },
            "set_downsample_filter": {
                "args": [
                    'self', 'filter_order', 'cutoff_freq', 'write_log'
                ],
                "defaults": [
                    'False'
                ]
            }
        }
    },
    "python/pysmurf/client/debug/smurf_iv.py": {
        "SmurfIVMixin": {
            "run_iv": {
                "args": [
                    'self', 'bias_groups', 'wait_time', 'bias', 'bias_high', 'bias_low', 'bias_step', 'show_plot', 'overbias_wait', 'cool_wait', 'make_plot', 'save_plot', 'plotname_append', 'channels', 'band', 'high_current_mode', 'overbias_voltage', 'grid_on', 'phase_excursion_min', 'bias_line_resistance', 'do_analysis'
                ],
                "defaults": [
                    'None', '0.1', 'None', '1.5', '0', '0.005', 'False', '2.0', '30', 'True', 'True', "''", 'None', 'None', 'True', '8.0', 'True', '3.0', 'None', 'True'
                ]
            },
            "analyze_iv": {
                "args": [
                    'self', 'v_bias', 'resp', 'make_plot', 'show_plot', 'save_plot', 'basename', 'band', 'channel', 'R_sh', 'plot_dir', 'high_current_mode', 'bias_group', 'grid_on', 'R_op_target', 'pA_per_phi0', 'bias_line_resistance', 'plotname_append'
                ],
                "defaults": [
                    'True', 'False', 'True', 'None', 'None', 'None', 'None', 'None', 'False', 'None', 'False', '0.007', 'None', 'None', "''"
                ]
            }
        }
    },
    "python/pysmurf/client/debug/smurf_noise.py": {
        "SmurfNoiseMixin": {
            "take_noise_psd": {
                "args": [
                    'self', 'meas_time', 'channel', 'nperseg', 'detrend', 'fs', 'low_freq', 'high_freq', 'make_channel_plot', 'make_summary_plot', 'save_data', 'show_plot', 'grid_on', 'datafile', 'downsample_factor', 'write_log', 'reset_filter', 'reset_unwrapper', 'return_noise_params', 'plotname_append'
                ],
                "defaults": [
                    'None', '2 ** 12', "'constant'", 'None', 'None', 'None', 'True', 'True', 'False', 'False', 'False', 'None', 'None', 'True', 'True', 'True', 'False', "''"
                ]
            }
        },
    }
}

"""
Frozen Functions:

SETUP FUNCTIONS
setup
set_amplifier_bias #smurf_command
set_cryo_card_ps_en #smurf_command
which_on #smurf_util
band_off #smurf_util
channel_off #smurf_util

TUNING FUNCTIONS
full_band_resp #smurf_tune.py
find_freq #smurf_tune.py
setup_notches #smurf_tune.py
run_serial_gradient_descent #smurf_command
run_serial_eta_scan #smurf_command
plot_tune_summary #smurf_tune.py
tracking_setup #smurf_tune.py
set_amplitude_scale_array #smurf_command

TES/FLUX RAMP FUNCTIONS
set_tes_bias_bipolar_array #smurf_util
set_tes_bias_high_current #smurf_util
set_tes_bias_low_current #smurf_util
set_mode_dc #smurf_util
set_mode_ac #smurf_util
flux_ramp_setup #smurf_tune.py

DATA ACQUISITION FUNCTIONS
set_stream_enable #smurf_command
take_stream_data #smurf_util
take_noise_psd #smurf_noise
stream_data_on #smurf_util
stream_data_off #surf_util
read_stream_data #smurf_util
set_downsample_filter #smurf_util

IV FUNCTIONS
run_iv #smurf_iv
analyze_iv #smurf_iv

DATA OUTPUTS TO DISK
tune files generated when new resonators are found
channel mapping file format
.dat noise files - generated by take_stream_data
iv_files - generated by run_iv"
"""

def compare_args(node, intdict, name):
    #check function
    if isinstance(node, ast.FunctionDef):
        spec_args = intdict['args']
        spec_defaults = intdict["defaults"]

        # Extract function arguments from the AST node
        found_args = [arg.arg for arg in node.args.args]
        found_defaults = [ast.unparse(d) if d else None for d in node.args.defaults]

        has_varargs = any(isinstance(arg, ast.arg) and arg.arg == 'args' for arg in node.args.args) #check for varargs

        has_kwargs = any(isinstance(arg, ast.arg) and arg.arg == 'kwargs' for arg in node.args.args) #check for kwargs

        if has_varargs == True:
            raise NotImplementedError("This code doesn't currently have the functionality to support varargs, " + name + " has vararg")
        if has_kwargs == True:
            raise NotImplementedError("This code doesn't currently have the functionality to support kwargs, " + name + " has kwarg")

        # Compare arguments to specified arguments in interface dict
        if found_args != spec_args:
            print(f"Mismatch in arguments for function {name}")
            print(f"Found: {found_args}")
            print(f"Expected: {spec_args}")
            return False
        
        # Compare default values to specified defaults in interface dict
        if found_defaults != spec_defaults:
            print(f"Mismatch in default values for function {name}")
            print(f"Found: {found_defaults}")
            print(f"Expected: {spec_defaults}")
            return False
    #Check class
    #elif isinstance(node, ast.ClassDef):

    return True

if __name__ == "__main__":   #if this is the main thing being run
    setspecclass = set()
    setspecfunc = set()
    foundclass = set()
    foundfunc = set()
    for fname in interface.keys():   #loop over file names "fname" as keys in dictionary called "interface"
        with open(fname, 'r') as fh:   #opens file "fname", in read mode 'r', to be used in code as "fh"
            tree = ast.parse(fh.read()) #parsing contents of file into abstract syntax tree
            dump = ast.dump(tree)
        for classname in interface[fname].keys():
            setspecclass.add(classname)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name == classname:
                        intdict = interface[fname][classname]
                        assert compare_args(node, intdict, classname)
                        foundclass.add(classname)
                for specfunc in interface[fname][classname].keys():
                    setspecfunc.add(specfunc)
                    if isinstance(node, ast.FunctionDef) and node.name == specfunc:
                        intdict = interface[fname][classname][specfunc]
                        assert compare_args(node, intdict, specfunc)
                        foundfunc.add(specfunc)
    if setspecclass != foundclass:
        print(setspecclass-foundclass)
    if setspecfunc != foundfunc:
        print(setspecfunc-foundfunc)
pass