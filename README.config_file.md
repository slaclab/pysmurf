The configuration file defines basic default parameters that are implementation specific. There is one configuration file per physical hardware setup.

# Organization

The configuration files are stored in the `cfg_files` folder in the pysmurf git repository ([https://github.com/slaclab/pysmurf/tree/main/cfg_files](https://github.com/slaclab/pysmurf/tree/main/cfg_files)). There are subfolders for each institution/site. Some subdirectories also have a README file that describes the config file. We encourage this to be included so the specific details of the files can be understood in the future. 

# Quick Start

1. If it does not exist, make a directory for your site. 
2. Copy the default config file
3. Check amplifier values

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

The relation between PIC on the cryocard and bias group pair. See the documenation [here](https://confluence.slac.stanford.edu/pages/viewpage.action?spaceKey=SMuRF&title=Cryostat+board).

## bias_group_to_pair

The TES bias uses a bipolar pair of DACs. This defines the pair of DACs that form a bias group.

## constant

- `pA_per_phi0` - The conversion between units oh phi0 and pA. This is number depends on the resonator/mulitplexing chips installed.

## timing

## Others

These are other variables that are in the top level of the config table. Several of these should probably be moved, but for now just know they exist.

- `R_sh` - The resistance of the shunt resistor in Ohms
- `bias_line_resistance`- The resistance of cabling from cryocard to the TES.
- `tune_dir`- The directory where the tuning files are stored. Tuning files define the resonator frequencies and the channel assignments
- `default_data_dir`- The directory where the output data is stored. Output data can include anything from eta-scans to tracked resonator timestreams
- `high_low_current_ratio` - The ratio between the high current mode (no filtering) and low current mode (with lowpass filter)
