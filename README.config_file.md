The configuration file defines basic default parameters that are implementation specific. There is one configuration file per physical hardware setup.

# Organization

The configuration files are stored in the `cfg_files` folder in the pysmurf git repository ([https://github.com/slaclab/pysmurf/tree/main/cfg_files](https://github.com/slaclab/pysmurf/tree/main/cfg_files)). There are subfolders for each institution/site. Some subdirectories also have a README file that describes the config file. We encourage this to be included so the specific details of the files can be understood in the future.

# Quick Start

1. If it does not exist, make a directory for your site under `cfg_files/`.
2. Copy the template config file from `cfg_files/template/template.cfg` into your site directory.
3. Update the `amplifier` values to match your cryostat-card calibration.
4. If you know them, update the `bias_line_resistance` and `R_sh` fields.
5. Save.

You should be able to call this config file when you instantiate your `SmurfControl` object. One of the first steps you'll do is run `S.estimate_phase_delay(band)` for each band; save its output into the per-band `bandDelayUs` field of your config file so you do not need to re-run it later.

The file is JSON, with one extension: any line beginning with `#`, and any text following a `#` on a line, is stripped before parsing (see `SmurfConfig.read_json`). The schema lives in `python/pysmurf/client/base/smurf_config.py` (`SmurfConfig.validate_config`); unknown top-level keys are tolerated (`ignore_extra_keys=True`) but ignored.

# Parameter Reference

For each parameter below: `key` — type, range/units, default, short description. "Required" means schema validation rejects the file if the key is missing. "Optional" means the schema fills in the listed default.

## init

DSP initialisation, per 500 MHz band.

- `dspEnable` — int (0/1), optional, default `1`. Force-enables DSP during pysmurf setup.
- `band_N` — dict (N in 0..7), at least one required. Each band block accepts:

| Key | Type / range | Default | Description |
| --- | --- | --- | --- |
| `feedbackGain` | int, `[0, 2^16)` | required | Global tracking-feedback gain. |
| `feedbackLimitkHz` | float kHz, > 0 | required | Tracking-feedback bandwidth limit. |
| `att_uc` | int, `[0, 32)` | required | Up-converter (DAC) RF attenuator, 0.5 dB steps. |
| `att_dc` | int, `[0, 32)` | required | Down-converter (ADC) RF attenuator, 0.5 dB steps. |
| `amplitude_scale` | int, `[0, 16)` | required | Probe-tone amplitude, 3 dB steps. |
| `trigRstDly` | int, `[0, 128)` | required | Flux-ramp counter reset delay, 2.4 MHz ticks. |
| `lmsGain` | int, `[0, 8)` | required | LMS gain, powers of 2. |
| `iq_swap_in` | int (0/1) | `0` | Swap I/Q on input. |
| `iq_swap_out` | int (0/1) | `0` | Swap I/Q on output. |
| `feedbackEnable` | int (0/1) | `1` | Enable tracking feedback. |
| `feedbackPolarity` | int (0/1) | `1` | Tracking-feedback sign. |
| `bandDelayUs` | float, `[0, 30)` µs | `null` | Total round-trip delay; measure with `S.estimate_phase_delay(band)`. Replaces the deprecated `refPhaseDelay`/`refPhaseDelayFine` pair. |
| `data_out_mux` | list[int, int] in `[0, 9]`, distinct | per-band default | DSP data-out mux selection. Defaults: bands 0/4 → `[2,3]`, 1/5 → `[0,1]`, 2/6 → `[6,7]`, 3/7 → `[8,9]`. |
| `lmsDelay` | int, `[0, 64)` | falls back to `refPhaseDelay` | LMS feedback latency match. |
| `refPhaseDelay` | int, `[0, 32)` | `0` | **DEPRECATED** — use `bandDelayUs`. |
| `refPhaseDelayFine` | int, `[0, 256)` | `0` | **DEPRECATED** — use `bandDelayUs`. |

## bad_mask

Optional, default `{}`. Frequency intervals (in MHz) where resonator candidates are ignored during relock. Keys are arbitrary unique strings; values are `[f_low, f_high]` with `f_low < f_high` and both in `[4000, 8000]`.

```yaml
"bad_mask": {
  "0": [5000, 5100],
  "1": [5171.64, 5171.74]
}
```

## amplifier

4K HEMT and 50K LNA bias values plus calibration constants.

| Key | Type / units | Default | Description |
| --- | --- | --- | --- |
| `hemt_Vg` | float, V | required | 4K HEMT gate voltage. |
| `bit_to_V_hemt` | float V/bit, > 0 | required | HEMT gate DAC scale. |
| `hemt_Id_offset` | float, mA | required | HEMT drain-current regulator offset, subtracted from the measured value. |
| `LNA_Vg` | float, V | required | 50K LNA gate voltage. |
| `50k_Id_offset` | float, mA | required | 50K LNA drain-current regulator offset. |
| `dac_num_50k` | int, `[1, 32]` | required | RTM DAC wired to the 50K LNA gate. C02 cards use `32`; some early C01s use `2`. |
| `bit_to_V_50k` | float V/bit, > 0 | required | 50K gate DAC scale. |
| `hemt_gate_min_voltage` / `hemt_gate_max_voltage` | float, V | required | Software clamps for `set_hemt_gate_voltage`. |
| `hemt_Vd_series_resistor` | float Ω, > 0 | `200.0` | Inline drain-current sense resistor (R44 on C02 cards). |
| `50K_amp_Vd_series_resistor` | float Ω, > 0 | `10.0` | Inline drain-current sense resistor (R54 on C02 cards). |
| `hemt`, `50k` | dict | C02 defaults | Cryocard PIC enable wiring per amp rail. Required field if present: `gate_dac_num`. |
| `hemt1`, `hemt2`, `50k1`, `50k2` | dict | C02 defaults | Dual-amplifier crate descriptors. If a block is present, **all 14 fields** are required (see `smurf_config.py:523-663`): `drain_conversion_b/m`, `drain_dac_num`, `drain_offset`, `drain_opamp_gain`, `drain_pic_address`, `drain_resistor`, `drain_volt_default/min/max`, `gate_bit_to_volt`, `gate_volt_default/min/max`, `power_bitmask` (and `gate_dac_num` for `hemt2`/`50k1`/`50k2`). |

## attenuator

Mapping from each of the four UC/DC RF attenuators to a band index in `[0, 4)`. The same mapping applies to bay 0 (bands 0-3) and bay 1 (bands 4-7).

```yaml
"attenuator": { "att1": 0, "att2": 1, "att3": 2, "att4": 3 }
```

## chip_to_freq

Mux-chip number → `[f_min_GHz, f_max_GHz]` covered by that chip. Used by tune helpers when assigning resonators to chips. Keys are stringified chip numbers; values are length-2 float lists.

## pic_to_bias_group / bias_group_to_pair

The relation between PIC channel on the cryocard and TES bias group. See the [cryostat-board documentation](https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=SMuRF&title=Cryostat+board).

`bias_group_to_pair` defines the bipolar RTM DAC pair `[pos, neg]` driving each TES bias group. Validation enforces:

- No DAC may appear in more than one pair.
- The DAC named in `amplifier.dac_num_50k` may not appear in any pair.

## TES electrical parameters

Top-level keys, all required and `> 0` unless noted.

- `R_sh` — float, Ω. TES shunt resistance (assumed identical for every channel).
- `bias_line_resistance` — float, Ω. Round-trip TES bias-line resistance in low-current mode (cryostat-card + cable + cold resistors).
- `high_low_current_ratio` — float, `>= 1`. Current ratio between low- and high-current modes.
- `high_current_mode_bool` — int (0/1), optional, default `0`. If `1`, biasing is *always* high-current.
- `all_bias_groups` — list[int] in `[0, 16)`. TES bias groups in use.

## tune_band

Resonator tuning and tracking parameters.

| Key | Type / range / units | Notes |
| --- | --- | --- |
| `fraction_full_scale` | float, `(0, 1]` | Flux-ramp DAC swing as a fraction of full scale. |
| `reset_rate_khz` | float, `[0, 100]` kHz | Flux-ramp reset rate. |
| `default_tune` | path string, optional | Saved tune file loaded on construction. |
| `lms_freq` | dict, per-band, float Hz, > 0 | Tracking sideband frequency. |
| `delta_freq` | dict, per-band, float MHz, > 0 | Span around each resonator probed during tracking. |
| `feedback_start_frac` | dict, per-band, `[0, 1]` | Fraction of cycle where feedback turns on. |
| `feedback_end_frac` | dict, per-band, `[0, 1]` | Fraction of cycle where feedback turns off. |
| `gradient_descent_gain` | dict, per-band, float > 0 | |
| `gradient_descent_averages` | dict, per-band, int > 0 | |
| `gradient_descent_converge_hz` | dict, per-band, float Hz, > 0 | |
| `gradient_descent_step_hz` | dict, per-band, float Hz, > 0 | |
| `gradient_descent_momentum` | dict, per-band, int >= 0 | |
| `gradient_descent_beta` | dict, per-band, `[0, 1]` | |
| `eta_scan_averages` | dict, per-band, int > 0 | |
| `eta_scan_del_f` | dict, per-band, int Hz, > 0 | |
| `eta_scan_amplitude` | dict, per-band, int | Probe amplitude. |

Per-band dicts must contain one entry per band declared in `init`. Keys are stringified band indices (`"0"`, `"1"`, ...).

## flux_ramp

- `num_flux_ramp_counter_bits` — `20` (C0 RTM) or `32` (C1 RTM).

## constant

Optional, default `{ "pA_per_phi0": 9e6 }`.

- `pA_per_phi0` — float, pA / Φ₀. TES current per Φ₀ of demodulated SQUID phase. Depends on the installed mux-chip generation.

## timing

- `timing_reference` — one of `"ext_ref"`, `"backplane"`, `"fiber"`.
  - `"ext_ref"` — internal oscillator locked to the front-panel reference (`LmkReg_0x0147 : 0x1A`); sets `flux_ramp_start_mode = 0`.
  - `"backplane"` — timing taken from a timing master via the ATCA backplane; sets `flux_ramp_start_mode = 1`.
  - `"fiber"` — timing taken from a timing-system fiber input.

## fs

Float Hz, > 0. SMuRF data sampling frequency. The system reports its actual rate at runtime; this value is used for offline calculations and as a fallback.

## Output directories

All optional. Each path must exist with write permission for the running user (validated on load).

- `default_data_dir` — default `/data/smurf_data` (raw/output data).
- `smurf_cmd_dir` — default `/data/smurf_data/smurf_cmd` (remote-command files).
- `tune_dir` — default `/data/smurf_data/tune` (saved tunings).
- `status_dir` — default `/data/smurf_data/status` (run-status files).

## ultrascale_temperature_limit_degC

Optional, default `null`. Ultrascale FPGA over-temperature protection threshold in degC, range `[0, 99]`. If `null`, pysmurf does not engage OT protection. Some crates have been observed to drop the carrier mid-enable sequence; newer firmware enables OT protection in firmware instead.

# Deprecated Keys

Documented for back-compat with older site configs. Avoid in new files.

- `init.band_N.refPhaseDelay`, `init.band_N.refPhaseDelayFine` — superseded by `bandDelayUs`. Migrate by measuring `bandDelayUs` with `S.estimate_phase_delay(band)` and removing both legacy keys. Validation still accepts them (defaults `0`).
- `smurf_to_mce` — block driving the SMuRF → Keck GCP transmit path. Mostly retired; some legacy fields are still consumed by `python/pysmurf/client/util/smurf_util.py`. Replaced by the `processor` block in PR #444. Not validated by the schema (`ignore_extra_keys=True`).
- `init.band_N.lmsDelay` — retained for offline analysis and back-compat; new configs should rely on `bandDelayUs`-derived defaults.
- `init.dspEnable` — legacy override; production systems set this through `defaults.yml`.

Configuration keys present in older site configs but **not** recognised by the current schema (silently ignored):

- `epics_root` — only referenced in `scratch/cyu/cryochannel.py`; not loaded by the active client.
- `flux_ramp.select_ramp` — `set_select_ramp` is hard-coded to `0x1` in `python/pysmurf/client/base/smurf_control.py:646`; the config value is never read.
