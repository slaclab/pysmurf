The configuration file defines basic default parameters that are implementation specific. There is one configuration file per physical hardware setup.

# Organization

The configuration files are stored in the `cfg_files` folder in the pysmurf git repository ([https://github.com/slaclab/pysmurf/tree/main/cfg_files](https://github.com/slaclab/pysmurf/tree/main/cfg_files)). There are subfolders for each institution/site. Some subdirectories also have a README file that describes the config file. We encourage this to be included so the specific details of the files can be understood in the future. 

# Quick Start

1. If it does not exist, make a directory for your site. 
2. Copy the template config file from `cfg_files/template/template.cfg` into your directory you created in step 1. 
3. Update the amplifier values if needed.
4. If you know it, update the `bias_line_resistance` and `R_sh` fields.
5. Save

You should be able to call this config file when you instantiate your `SmurfControl` object. One of the first steps you'll do is run `estimate_phase_delay`. You can save the outputs into your config file so you will not need to run it in the future. 

# Important Variables

## init

### band_#

- `refPhaseDelay` - The phase delay parameter. Solve for this using the function `S.estimate_phase_delay(band)`
- `refPhaseDelayFine` - The fine phase delay parameter. Solve for this using the function `S.estimate_phase_delay(band)`
- `att_uc` - The upconverter (DAC) attenuator
- `att_dc` - The downconverter (ADC) attenuator
- `amplitude_scale` - The probe tone power.

## bad_mask

A list of bad channels. Resonators in the bad mask will not be turned on. The bad mask is defined by frequency. Below is a bad mask that excludes resonators between 5000-5100 MHz and 5171.64-5171.74 MHz.  

```yaml
"bad_mask": {
"0": [
	5000,
  5100
      ],
"1": [
	5171.64,
	5171.74
]
}
```

## amplifier

The 4K and 50K amplifier bias values. This also holds the conversion between volts and DAC bits.

## pic_to_bias_group

The relation between PIC on the cryocard and bias group pair. See the documentation [here](https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=SMuRF&title=Cryostat+board).

## bias_group_to_pair

The TES bias uses a bipolar pair of DACs. This defines the pair of DACs that form a bias group.

## constant

- `pA_per_phi0` - The conversion between units oh phi0 and pA. This is number depends on the resonator/mulitplexing chips installed.

## timing

This defines which timing master to use. The available values are "ext_ref" and "backplane". 

## Others

These are other variables that are in the top level of the config table. Several of these should probably be moved, but for now just know they exist.

- `R_sh` - The resistance of the shunt resistor in Ohms
- `bias_line_resistance`- The total round-trip TES bias-line resistance in low-current mode, in Ohms.  Includes both the inline resistance on the cryostat card and the cryocable resistance.  May be supplied directly (legacy single-value path) or derived from `Rcable` plus the cryostat-card resistance loaded from a card cfg file (see [Cryostat-card cfg files](#cryostat-card-cfg-files) below).  An explicitly set `bias_line_resistance` always takes precedence.
- `Rcable` - Optional.  Round-trip TES bias-cable resistance plus any cold resistors, in Ohms, with the cryostat-card resistance excluded.  When paired with a cryostat-card cfg that supplies `R_cryostat_card`, the loader sets `bias_line_resistance = Rcable + R_cryostat_card` automatically.  Ignored if `bias_line_resistance` is set explicitly.
- `cryostat_card_config_file` - Optional.  Path to a separate JSON cfg file holding cryostat-card-specific values (`R_cryostat_card`, `high_low_current_ratio`, `pic_to_bias_group`, `bias_group_to_pair`).  Resolved relative to the main cfg's directory if not absolute.  See [Cryostat-card cfg files](#cryostat-card-cfg-files) below.
- `tune_dir`- The directory where the tuning files are stored. Tuning files define the resonator frequencies and the channel assignments
- `default_data_dir`- The directory where the output data is stored. Output data can include anything from eta-scans to tracked resonator timestreams
- `high_low_current_ratio` - The ratio between the high current mode (no filtering) and low current mode (with lowpass filter)

# Cryostat-card cfg files

Cryostat-card-specific values can live in a separate cfg file shared
across experiment cfgs that use the same physical card.  Reference one
from the main cfg via `cryostat_card_config_file`:

```jsonc
{
    "cryostat_card_config_file" : "../cryostat_cards/cc02-06.cfg",
    "Rcable" : 29.1,
    "R_sh"   : 750e-6
    // bias_line_resistance, high_low_current_ratio, pic_to_bias_group,
    // and bias_group_to_pair are loaded from the card cfg.
}
```

The card cfg encodes the resistors loaded onto the card and the card's
PIC-to-bias-group / bias-group-to-DAC-pair maps:

```jsonc
{
    "card_id"                : "cc02-06",
    "R_cryostat_card"        : 16505.0,
    "high_low_current_ratio" : 8.084,
    "pic_to_bias_group"      : { "0": 0, "1": 1, "...": "..." },
    "bias_group_to_pair"     : { "0": [1, 2], "1": [3, 4], "...": "..." }
}
```

Two example card cfgs ship in `cfg_files/cryostat_cards/`.  See
`cfg_files/cryostat_cards/README.md` for full merge semantics.

If both the main cfg and the card cfg set the same key, the main cfg
wins (explicit override) and a notice is printed.  This makes it easy
to start from a card cfg and tweak a single value per deployment.
