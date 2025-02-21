import ast
# functions and their arguments to check
# specified here: https://www.overleaf.com/project/5e837cac9659910001e5f71e
interface = {
    "python/pysmurf/client/tune/smurf_tune.py": {  # file
        "SmurfTuneMixin": {                        # class
            "methods": [
                'tune', 'tune_band', 'tune_band_serial', 'plot_tune_summary', 'full_band_resp', 'find_peak', 'find_flag_blocks', 'pad_flags', 'plot_find_peak', 'eta_fit', 'plot_eta_fit', 'get_closest_subband', 'check_freq_scale', 'load_master_assignment', 'get_master_assignment', 'assign_channels', 'write_master_assignment', 'make_master_assignment_from_file', 'get_group_list', 'get_group_number', 'write_group_assignment', 'relock', 'fast_relock', '_get_eta_scan_result_from_key', 'get_eta_scan_result_freq', 'get_eta_scan_result_eta', 'get_eta_scan_result_eta_mag', 'get_eta_scan_result_eta_scaled', 'get_eta_scan_result_eta_phase', 'get_eta_scan_result_channel', 'get_eta_scan_result_subband', 'get_eta_scan_result_offset', 'eta_estimator', 'eta_scan', 'flux_ramp_check', '_feedback_frac_to_feedback', '_feedback_to_feedback_frac', 'tracking_setup', 'track_and_check', 'eta_phase_check', 'analyze_eta_phase_check', 'unset_fixed_flux_ramp_bias', 'set_fixed_flux_ramp_bias', 'flux_ramp_setup', 'get_fraction_full_scale', 'check_lock', 'check_lock_flux_ramp_off', 'find_freq', 'plot_find_freq', 'full_band_ampl_sweep', 'find_all_peak', 'fast_eta_scan', 'setup_notches', 'calculate_eta_svd', 'save_tune', 'load_tune', 'last_tune', 'optimize_lms_delay', 'estimate_lms_freq', 'estimate_flux_ramp_amp', 'flux_mod2', 'make_sync_flag', 'flux_mod', 'dump_state', 'fake_resonance_dict', 'freq_to_band'
            ],
            "plot_tune_summary": {                 # methods + function and args
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
            "methods" : [
                '_caput', '_caget', 'get_pysmurf_version', 'get_pysmurf_directory', 'get_smurf_startup_script', 'get_smurf_startup_args', 'get_enabled_bays', 'get_configuring_in_progress', 'get_system_configured', 'get_rogue_version', 'get_enable', 'get_number_sub_bands', 'get_number_channels', 'get_number_processed_channels', 'set_defaults_pv', 'set_read_all', 'run_pwr_up_sys_ref', 'get_eta_scan_in_progress', 'set_gradient_descent_max_iters', 'get_gradient_descent_max_iters', 'set_gradient_descent_averages', 'get_gradient_descent_averages', 'set_gradient_descent_gain', 'get_gradient_descent_gain', 'set_gradient_descent_converge_hz', 'get_gradient_descent_converge_hz', 'set_gradient_descent_step_hz', 'get_gradient_descent_step_hz', 'set_gradient_descent_momentum', 'get_gradient_descent_momentum', 'set_gradient_descent_beta', 'get_gradient_descent_beta', 'run_parallel_eta_scan', 'run_serial_eta_scan', 'run_serial_min_search', 'run_serial_gradient_descent', 'sel_ext_ref', 'save_state', 'save_config', 'get_tone_file_path', 'set_tone_file_path', 'load_tone_file', 'set_tune_file_path', 'get_tune_file_path', 'set_load_tune_file', 'get_load_tune_file', 'set_eta_scan_del_f', 'get_eta_scan_del_f', 'set_eta_scan_freq', 'get_eta_scan_freq', 'set_eta_scan_amplitude', 'get_eta_scan_amplitude', 'set_eta_scan_channel', 'get_eta_scan_channel', 'set_eta_scan_averages', 'get_eta_scan_averages', 'set_eta_scan_dwell', 'get_eta_scan_dwell', 'set_run_serial_find_freq', 'set_run_eta_scan', 'get_run_eta_scan', 'get_eta_scan_results_real', 'get_eta_scan_results_imag', 'set_amplitude_scales', 'get_amplitude_scales', 'set_amplitude_scale_array', 'get_amplitude_scale_array', 'set_amplitude_scale_array_currentchans', 'set_feedback_enable_array', 'get_feedback_enable_array', 'set_single_channel_readout', 'get_single_channel_readout', 'set_single_channel_readout_opt2', 'get_single_channel_readout_opt2', 'set_readout_channel_select', 'get_readout_channel_select', 'set_stream_enable', 'get_stream_enable', 'set_rf_iq_stream_enable', 'get_rf_iq_stream_enable', 'get_build_dsp_g', 'set_decimation', 'get_decimation', 'set_filter_alpha', 'get_filter_alpha', 'set_iq_swap_in', 'get_iq_swap_in', 'set_iq_swap_out', 'get_iq_swap_out', 'set_ref_phase_delay', 'get_ref_phase_delay', 'set_ref_phase_delay_fine', 'get_ref_phase_delay_fine', 'set_band_delay_us', 'get_band_delay_us', 'set_tone_scale', 'get_tone_scale', 'set_waveform_select', 'get_waveform_select', 'set_waveform_start', 'get_waveform_start', 'set_rf_enable', 'get_rf_enable', 'set_analysis_scale', 'get_analysis_scale', 'set_feedback_enable', 'get_feedback_enable', 'get_loop_filter_output_array', 'set_tone_frequency_offset_mhz', 'get_tone_frequency_offset_mhz', 'set_center_frequency_array', 'get_center_frequency_array', 'set_feedback_gain', 'get_feedback_gain', 'set_eta_phase_array', 'get_eta_phase_array', 'set_frequency_error_array', 'get_frequency_error_array', 'set_eta_mag_array', 'get_eta_mag_array', 'set_feedback_limit', 'get_feedback_limit', 'set_noise_select', 'get_noise_select', 'set_lms_delay', 'get_lms_delay', 'set_lms_gain', 'get_lms_gain', 'set_trigger_reset_delay', 'get_trigger_reset_delay', 'set_feedback_start', 'get_feedback_start', 'set_feedback_end', 'get_feedback_end', 'set_lms_enable1', 'get_lms_enable1', 'set_lms_enable2', 'get_lms_enable2', 'set_lms_enable3', 'get_lms_enable3', 'set_lms_rst_dly', 'get_lms_rst_dly', 'set_lms_freq', 'get_lms_freq', 'set_lms_freq_hz', 'get_lms_freq_hz', 'set_lms_dly_fine', 'get_lms_dly_fine', 'set_iq_stream_enable', 'get_iq_stream_enable', 'set_feedback_polarity', 'get_feedback_polarity', 'set_band_center_mhz', 'get_band_center_mhz', 'get_channel_frequency_mhz', 'get_digitizer_frequency_mhz', 'set_synthesis_scale', 'get_synthesis_scale', 'set_dsp_enable', 'get_dsp_enable', 'set_feedback_enable_channel', 'get_feedback_enable_channel', 'set_eta_mag_scaled_channel', 'get_eta_mag_scaled_channel', 'set_center_frequency_mhz_channel', 'get_center_frequency_mhz_channel', 'set_amplitude_scale_channel', 'get_amplitude_scale_channel', 'set_eta_phase_degree_channel', 'get_eta_phase_degree_channel', 'get_frequency_error_mhz', 'band_to_bay', 'set_att_uc', 'get_att_uc', 'set_att_dc', 'get_att_dc', 'set_remap', 'get_dac_temp', 'set_dac_enable', 'get_dac_enable', 'set_data_out_mux', 'get_data_out_mux', 'set_jesd_reset_n', 'set_jesd_rx_enable', 'get_jesd_rx_enable', 'get_jesd_rx_status_valid_cnt', 'get_jesd_rx_data_valid', 'set_jesd_link_disable', 'get_jesd_link_disable', 'set_jesd_tx_enable', 'get_jesd_tx_enable', 'get_jesd_tx_data_valid', 'get_jesd_tx_status_valid_cnt', 'set_check_jesd', 'get_jesd_status', 'get_fpga_uptime', 'get_fpga_version', 'get_fpga_git_hash', 'get_fpga_git_hash_short', 'get_fpga_build_stamp', 'set_input_mux_sel', 'get_input_mux_sel', 'set_data_buffer_size', 'get_data_buffer_size', 'set_waveform_start_addr', 'get_waveform_start_addr', 'set_waveform_end_addr', 'get_waveform_end_addr', 'set_waveform_wr_addr', 'get_waveform_wr_addr', 'set_waveform_empty', 'get_waveform_empty', 'set_streamdatawriter_datafile', 'get_streamdatawriter_datafile', 'set_streamdatawriter_open', 'get_streamdatawriter_open', 'set_streamdatawriter_close', 'get_streamdatawriter_close', 'set_trigger_daq', 'get_trigger_daq', 'set_arm_hw_trigger', 'set_trigger_hw_arm', 'get_trigger_hw_arm', 'set_rtm_arb_waveform_lut_table', 'get_rtm_arb_waveform_lut_table', 'get_rtm_arb_waveform_busy', 'get_rtm_arb_waveform_trig_cnt', 'set_rtm_arb_waveform_continuous', 'get_rtm_arb_waveform_continuous', 'trigger_rtm_arb_waveform', 'set_dac_axil_addr', 'get_dac_axil_addr', 'set_rtm_arb_waveform_timer_size', 'get_rtm_arb_waveform_timer_size', 'set_rtm_arb_waveform_max_addr', 'get_rtm_arb_waveform_max_addr', 'set_rtm_arb_waveform_enable', 'get_rtm_arb_waveform_enable', 'reset_rtm', 'set_cpld_reset', 'get_cpld_reset', 'cpld_toggle', 'set_k_relay', 'get_k_relay', 'set_ramp_rate', 'get_ramp_rate', 'set_trigger_delay', 'get_trigger_delay', 'set_debounce_width', 'get_debounce_width', 'set_ramp_slope', 'get_ramp_slope', 'set_flux_ramp_dac', 'get_flux_ramp_dac', 'set_mode_control', 'get_mode_control', 'set_fast_slow_step_size', 'get_fast_slow_step_size', 'set_fast_slow_rst_value', 'get_fast_slow_rst_value', 'set_enable_ramp_trigger', 'get_enable_ramp_trigger', 'set_cfg_reg_ena_bit', 'get_cfg_reg_ena_bit', 'set_rtm_slow_dac_enable', 'get_rtm_slow_dac_enable', 'set_rtm_slow_dac_enable_array', 'get_rtm_slow_dac_enable_array', 'set_rtm_slow_dac_data', 'get_rtm_slow_dac_data', 'set_rtm_slow_dac_data_array', 'get_rtm_slow_dac_data_array', 'set_rtm_slow_dac_volt', 'get_rtm_slow_dac_volt', 'set_rtm_slow_dac_volt_array', 'get_rtm_slow_dac_volt_array', 'get_amp_gate_voltage', 'set_amp_gate_voltage', 'get_amp_drain_voltage', 'get_amp_drain_enable', 'set_amp_drain_enable', 'set_amp_drain_voltage', 'get_amp_drain_current', 'get_amp_drain_current_dict', 'set_amp_defaults', 'get_amplifier_biases', 'set_hemt_enable', 'set_50k_amp_enable', 'get_50k_amp_gate_voltage', 'set_50k_amp_gate_voltage', 'set_hemt_gate_voltage', 'set_hemt_bias', 'get_hemt_bias', 'set_amplifier_bias', 'get_amplifier_bias', 'get_hemt_drain_current', 'get_50k_amp_drain_current', 'flux_ramp_on', 'flux_ramp_off', 'set_ramp_max_cnt', 'get_ramp_max_cnt', 'set_flux_ramp_freq', 'get_flux_ramp_freq', 'set_low_cycle', 'get_low_cycle', 'set_high_cycle', 'get_high_cycle', 'set_select_ramp', 'get_select_ramp', 'set_ramp_start_mode', 'get_ramp_start_mode', 'set_pulse_width', 'get_pulse_width', 'set_streaming_datafile', 'get_streaming_datafile', 'set_streaming_file_open', 'get_streaming_file_open', 'get_slot_number', 'get_crate_id', 'get_fpga_temp', 'get_fpga_vccint', 'get_fpga_vccaux', 'get_fpga_vccbram', 'get_regulator_iout', 'get_regulator_temp1', 'get_regulator_temp2', 'get_cryo_card_temp', 'get_cryo_card_cycle_count', 'get_cryo_card_relays', 'set_cryo_card_relay_bit', 'set_cryo_card_relays', 'set_cryo_card_delatch_bit', 'set_cryo_card_ps_en', 'get_cryo_card_ps_en', 'get_cryo_card_ac_dc_mode', 'get_user_config0', 'set_user_config0', 'clear_unwrapping_and_averages', 'set_trigger_width', 'set_trigger_enable', 'get_trigger_enable', 'set_evr_channel_reg_enable', 'get_evr_channel_reg_enable', 'set_evr_trigger_reg_enable', 'get_evr_channel_reg_count', 'set_evr_trigger_dest_type', 'get_evr_trigger_dest_type', 'set_evr_trigger_channel_reg_dest_sel', 'set_dbg_enable', 'get_dbg_enable', 'set_dac_reset', 'get_dac_reset', 'set_debug_select', 'get_debug_select', 'set_ultrascale_ot_upper_threshold', 'get_ultrascale_ot_upper_threshold', 'set_crossbar_output_config', 'get_crossbar_output_config', 'get_timing_link_up', 'get_timing_crc_err_cnt', 'get_timing_rx_dec_err_cnt', 'get_timing_rx_dsp_err_cnt', 'get_timing_rx_rst_cnt', 'set_lmk_enable', 'get_lmk_enable', 'set_lmk_reg', 'get_lmk_reg', 'set_mcetransmit_debug', 'get_frame_count', 'get_frame_size', 'get_frame_loss_cnt', 'get_frame_out_order_count', 'set_channel_mask', 'get_channel_mask', 'set_unwrapper_reset', 'set_filter_reset', 'set_filter_a', 'get_filter_a', 'set_filter_b', 'get_filter_b', 'set_filter_order', 'get_filter_order', 'set_filter_gain', 'get_filter_gain', 'set_downsample_mode', 'get_downsample_mode', 'set_downsample_factor', 'get_downsample_factor', 'set_downsample_external_bitmask', 'get_downsample_external_bitmask', 'set_filter_disable', 'get_filter_disable', 'set_max_file_size', 'get_max_file_size', 'set_data_file_name', 'get_data_file_name', 'open_data_file', 'close_data_file', 'get_smurf_processor_num_channels', 'set_payload_size', 'get_payload_size', 'set_predata_emulator_enable', 'get_predata_emulator_enable', 'set_predata_emulator_disable', 'get_predata_emulator_disable', 'set_predata_emulator_type', 'get_predata_emulator_type', 'set_predata_emulator_amplitude', 'get_predata_emulator_amplitude', 'set_predata_emulator_offset', 'get_predata_emulator_offset', 'set_predata_emulator_period', 'get_predata_emulator_period', 'set_postdata_emulator_enable', 'get_postdata_emulator_enable', 'set_postdata_emulator_type', 'get_postdata_emulator_type', 'set_postdata_emulator_amplitude', 'get_postdata_emulator_amplitude', 'set_postdata_emulator_offset', 'get_postdata_emulator_offset', 'set_postdata_emulator_period', 'get_postdata_emulator_period', 'set_stream_data_source_enable', 'get_stream_data_source_enable', 'set_stream_data_source_period', 'get_stream_data_source_period', 'shell_command', 'get_fru_info'
            ],
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
            "methods" : [
                'take_debug_data', 'start_jesd_watchdog', 'estimate_phase_delay', 'process_data', 'decode_data', 'decode_single_channel', 'take_stream_data', 'stream_data_cm', 'stream_data_on', 'stream_data_off', 'read_stream_data', 'read_stream_data_gcp_save', 'header_to_tes_bias', 'make_mask_lookup', 'read_stream_data_daq', 'check_adc_saturation', 'check_dac_saturation', 'read_adc_data', 'read_dac_data', 'setup_daq_mux', 'set_buffer_size', 'config_cryo_channel', 'which_on', 'toggle_feedback', 'band_off', 'channel_off', 'set_feedback_limit_khz', 'recover_jesd', 'jesd_decorator', 'check_jesd', 'get_fpga_status', 'which_bays', 'which_bands', 'freq_to_subband', 'channel_to_freq', 'get_channel_order', 'get_processed_channels', 'get_subband_from_channel', 'get_subband_centers', 'get_channels_in_subband', 'iq_to_phase', 'hex_string_to_int', 'int_to_hex_string', 'set_tes_bias_bipolar', 'set_tes_bias_bipolar_array', 'set_tes_bias_off', 'get_tes_bias_bipolar', 'get_tes_bias_bipolar_array', 'overbias_tes', 'overbias_tes_all', 'set_tes_bias_high_current', 'get_tes_bias_high_current', 'set_tes_bias_low_current', 'set_mode_dc', 'set_mode_ac', 'att_to_band', 'band_to_att', 'flux_ramp_rate_to_PV', 'flux_ramp_PV_to_rate', 'why', 'make_channel_mask', 'make_freq_mask', 'set_downsample_filter', 'get_filter_params', 'make_gcp_mask', 'bias_bump', 'all_off', 'mask_num_to_gcp_num', 'gcp_num_to_mask_num', 'smurf_channel_to_gcp_num', 'gcp_num_to_smurf_channel', 'play_sine_tes', 'play_tone_file', 'stop_tone_file', 'get_gradient_descent_params', 'set_fixed_tone', 'turn_off_fixed_tones', 'pause_hardware_logging', 'resume_hardware_logging', 'get_hardware_log_file', 'start_hardware_logging', 'stop_hardware_logging', '_hardware_logger', 'get_hardware_log_entry', 'play_tes_bipolar_waveform', 'stop_tes_bipolar_waveform', 'get_sample_frequency', 'identify_bias_groups', 'find_probe_tone_gap', 'check_full_band_resp', 'get_timing_mode', 'set_timing_mode'
            ],
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
            "methods" : [
                'run_iv', 'partial_load_curve_all', 'analyze_iv_from_file', 'analyze_iv', 'analyze_plc_from_file', 'estimate_opt_eff', 'estimate_bias_voltage'
            ],
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
            "methods" : [
                'take_noise_psd', 'turn_off_noisy_channels', 'noise_vs_tone', 'noise_vs_bias', 'noise_vs_amplitude', 'noise_vs', 'get_datafiles_from_file', 'get_biases_from_file', 'get_iv_data', 'get_si_data', 'get_NEI_to_NEP_factor', 'analyze_noise_vs_bias', 'analyze_psd', 'noise_all_vs_noise_solo', 'analyze_noise_all_vs_noise_solo', 'NET_CMB', 'analyze_noise_vs_tone', 'take_noise_high_bandwidth', 'noise_svd', 'plot_svd_summary', 'plot_svd_modes', 'remove_svd'
            ],
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

def compare_args(node, fname, classname, funcname):
    if isinstance(node, ast.FunctionDef):
        spec_args = dict[funcname]["args"]
        spec_defaults = interface[fname][classname][funcname]["defaults"]

        # Extract function arguments from the AST node
        found_args = [arg.arg for arg in node.args.args]
        found_defaults = [ast.unparse(d) if d else None for d in node.args.defaults]

        has_varargs = any(isinstance(arg, ast.arg) and arg.arg == 'args' for arg in node.args.args)

        has_kwargs = any(isinstance(arg, ast.arg) and arg.arg == 'kwargs' for arg in node.args.args)

        if has_varargs == True:
            print(funcname + "has vararg")

        if has_kwargs == True:
            print(funcname + "has kwarg")

        # Compare arguments to specified arguments in interface dict
        if found_args != spec_args:
            print(f"Mismatch in arguments for function {funcname}")
            print(f"Found: {found_args}")
            print(f"Expected: {spec_args}")
            return False
        
        # Compare default values to specified defaults in interface dict
        if found_defaults != spec_defaults:
            print(f"Mismatch in default values for function {funcname}")
            print(f"Found: {found_defaults}")
            print(f"Expected: {spec_defaults}")
            return False
    # Check class
    elif isinstance(node, ast.ClassDef):
        spec_meths = interface[fname][classname]["methods"]

        # Extract class methods from the AST node
        found_meths = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                found_meths.append(child.name)

        # Compare methods to specified methods in interface dict
        found_meths_set = set(found_meths)
        spec_meths_set = set(spec_meths)
        if spec_meths_set.difference(found_meths_set) != set():
            print(f"Mismatch in methods for class {classname}")
            print(f"Found methods: {found_meths_set.difference(spec_meths_set)}")
            print(f"Expected methods: {spec_meths_set.difference(found_meths_set)}")
            return False
    return True

if __name__ == "__main__":   #if this is the main thing being run
    for fname, classname in interface.items():   #loop over file names "fname" as keys and "spec" as values in dictionary called "interface"
        with open(fname, 'r') as fh:   #opens file "fname", in read mode 'r', to be used in code as "fh"
            tree = ast.parse(fh.read()) #parsing contents of file into abstract syntax tree
            notfound = []
            found = {}
            dump = ast.dump(tree)
        for specclass, funcname in classname.items():
            for specfunc in funcname.keys():
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == specfunc:
                        found.update({node.name: node})
                        type = "isfunc"
                        dict = interface[fname][classname]
                        assert compare_args(node, fname, dict, specfunc)
                    elif isinstance(node, ast.ClassDef) and node.name == specclass:
                        found.update({node.name: node})
                        type = "isclass"
                        assert compare_args(node, fname, specclass, specfunc)
pass