#!/usr/bin/env python
#-----------------------------------------------------------------------------
# Title      : pysmurf base module - SmurfConfig class
#-----------------------------------------------------------------------------
# File       : pysmurf/base/smurf_config.py
# Created    : 2018-08-30
#-----------------------------------------------------------------------------
# This file is part of the pysmurf software package. It is subject to
# the license terms in the LICENSE.txt file found in the top-level directory
# of this distribution and at:
#    https://confluence.slac.stanford.edu/display/ppareg/LICENSE.html.
# No part of the pysmurf software package, including this file, may be
# copied, modified, propagated, or distributed except according to the terms
# contained in the LICENSE.txt file.
#-----------------------------------------------------------------------------
"""Defines the :class:`SmurfConfig` class."""
import io
import json
import os
import re

import numpy as np

class SmurfConfig:
    """Read, manipulate or write a pysmurf config file.

    Class for loading pysmurf configuration files.  In addition to
    functions for reading and writing pysmurf configuration files,
    contains helper functions for manipulating configuration variables
    which are stored internally in a class instance attribute
    dictionary named :attr:`config`.

    pysmurf configuration files must be in the JSON format [#json]_.
    On instantiation, attempts to read the pysmurf configuration file
    at the path provided by the `filename` argument into the
    :attr:`config` dictionary.  If the `filename` argument is not
    provided, no configuration file is loaded and the :attr:`config`
    attribute is `None`.

    If a pysmurf configuration file is successfully loaded and the
    `validate` constructor argument is `True` (which is the default
    behavior), the parameters in the configuration file will be
    validated using the 3rd party `schema` python library [#schema]_
    using the rules specified in the :meth:`validate_config` class
    method.  If the configuration file data is valid, parameters are
    loaded into the :attr:`config` dictionary.

    If `validate` is `False`, the pysmurf configuration data will be
    loaded without `schema` validation.  *Use at your own risk!*

    Args
    ----
    filename : str or None, optional, default None
        Path to pysmurf configuration file to load.
    validate : bool, optional, default True
        Whether or not to run `schema` validation on the pysmurf
        configuration file data.  If `schema` validation fails, a
        `SchemaError` exception will be raised.

    Attributes
    ----------
    filename : str or None
        Path to loaded pysmurf configuration file, or configuration
        file to load.  `None` if no pysmurf configuration file has
        been loaded.
    config : dict or None
        Loaded dictionary of pysmurf configuration data.  `None` if no
        pysmurf configuration file has been loaded.

    See Also
    --------
    :meth:`validate_config`
        Run `schema` validation on loaded configuration dictionary.

    References
    ----------
    .. [#json] https://www.json.org/json-en.html
    .. [#schema] https://github.com/keleshev/schema

    """

    def __init__(self, filename=None, validate=True):
        """SmurfConfig constructor."""
        self.filename = filename
        self.config = None
        if self.filename is not None:
            self.read(update=True, validate=validate)

    @staticmethod
    def read_json(filename, comment_char='#'):
        """Read a pysmurf configuration file.

        Opens configuration file at the path provided by the
        `filename` argument, strips off all lines that start with the
        character provided by the `comment_char` argument, and then
        parses the remaining lines in the pysmurf configuration file
        into a dictionary using the :py:func:`json.loads` routine.
        Any text after the `comment_char` on a line is also ignored.

        Args
        ----
        filename : str
            Path to pysmurf configuration file to load.
        comment_char : str, optional, default '#'
            Comments that start with this character will be ignored.

        Returns
        -------
        loaded_config : dict
            Dictionary of loaded pysmurf configuration parameters.

        Raises
        ------
        FileNotFoundError
            Raised if the configuration file does not exist.
        :py:exc:`~json.JSONDecodeError`
            Raised if the loaded configuration file data is not in JSON
            format.
        """
        no_comments = []
        try:
            with open(filename) as config_file:
                for _, line in enumerate(config_file):

                    if line.lstrip().startswith(comment_char):
                        # line starts with comment character - remove it
                        continue

                    if comment_char in line:
                        # there's a comment character on this line.
                        # ignore it and everything that follows
                        line = line.split(comment_char)[0]
                        no_comments.append(line)
                    else:
                        # will pass on to json parser
                        no_comments.append(line)

        except FileNotFoundError:
            print('No configuration file found at' +
                  f' filename={filename}')
            raise

        loaded_config = json.loads('\n'.join(no_comments))
        return loaded_config

    def read(self, update=False, validate=True):
        """Read config file and update the config dictionary.

        Reads raw configuration parameters from configuration file at
        the path specified by the :attr:`filename` class instance
        attribute using the :meth:`read_json` method.

        If the `validate` argument is `True` (which is the default
        behavior), the loaded configuration parameters are validated
        using the :meth:`validate_config` routine.

        The :attr:`config` class instance attribute is only updated
        with the loaded configuration parameters if the `update`
        argument is `True` (it is `False` by default).  Either way,
        the loaded configuration parameter dictionary is returned.

        Args
        ----
        update : bool, optional, default False
            Whether or not to update the configuration.
        validate : bool, optional, default True
            Whether or not to run `schema` validation on the pysmurf
            configuration file data.  If `schema` validation fails, a
            `SchemaError` exception will be raised.

        Returns
        -------
        config : dict
            The loaded dictionary of pysmurf configuration parameters.

        See Also
        --------
        :meth:`read_json`
            Reads raw configuration file into a dictionary.
        :meth:`validate_config`
            Run `schema` validation on loaded configuration
            dictionary.
        """
        config = self.read_json(self.filename)

        # validate
        if validate:
            config = self.validate_config(config)

        # only update config dictionary if update=True
        if update:
            self.config = config

        return config

    def update(self, key, val):
        """Update a single key in the config dictionary.

        Args
        ----
        key : any
            Key to update in the :attr:`config` dictionary.
        val : any
            Value to assign to the given key
        """
        self.config[key] = val

    def write(self, outputfile):
        """Dump the current config to a file.

        Writes the :attr:`config` dictionary to file at the path
        provided by the `outputfile` argument using the
        :py:func:`json.dumps` routine.

        Args
        ----
        outputfile : str
            The name of the file to save the configuration to.
        """
        ## dump current config to outputfile ##
        with io.open(outputfile, 'w', encoding='utf8') as out_file:
            str_ = json.dumps(
                self.config,
                indent=4,
                separators=(',', ': '))
            out_file.write(str_)

    def has(self, key):
        """Report if configuration has requested key.

        Args
        ----
        key : any
            Key to check for in :attr:`config` dictionary.

        Returns
        -------
        bool
            Returns `True` if key is in the :attr:`config` dictionary,
            `False` if it is not.
        """
        if key in self.config:
            return True
        return False

    def get(self, key):
        """Return entry in config dictionary for requested key.

        Args
        ----
        key : any
            Key whose :attr:`config` dictionary entry to retrieve.

        Returns
        -------
        any or None
            Returns value for requested key from :attr:`config`
            dictionary.  Returns `None` if key is not present in the
            :attr:`config` dictionary.
        """
        if self.has(key):
            return self.config[key]
        return None

    def get_subkey(self, key, subkey):
        """Get config dictionary subkey value.

        Args
        ----
        key : any
            Key in :attr:`config` dictionary.
        subkey : any
            Subkey in :attr:`config` dictionary.

        Returns
        -------
        any or None
            Returns value for requested subkey from :attr:`config`
            dictionary.  Returns `None` if either key or subkey are
            not present in the :attr:`config` dictionary.
        """
        if self.has(key):
            sub_dict = self.config[key]
            try:
                return sub_dict[subkey]
            except KeyError:
                print("Key found, but subkey not found")
                return None
        else:
            print("Key not found")
            return None

    def update_subkey(self, key, subkey, val):
        """Set config dictionary subkey value.

        Args
        ----
        key : any
            Key in :attr:`config` dictionary.
        subkey : any
            Subkey in :attr:`config` dictionary
        val : any
            Value to write to :attr:`config` dictionary subkey.
        """
        try:
            self.config[key][subkey] = val
        except TypeError:
            self.config[key] = {} # initialize an empty dictionary first
            self.config[key][subkey] = val

    @staticmethod
    def validate_config(loaded_config):
        """Validate pysmurf configuration dictionary.

        Validates the parameters in the configuration dictionary
        provided by the `loaded_config` argument using the 3rd party
        `schema` python library.  If the configuration data is valid,
        parameters are returned as a dictionary.

        `schema` validation does several important things to raw data
        loaded from the pysmurf configuration file:

        - Checks that all mandatory configuration variables are defined.
        - Conditions all configuration variables into the correct type
          (e.g. `float`, `int`, `str`, etc.).
        - Automatically fills in the values for missing optional
          parameters.  Optional parameters are typically parameters which
          almost never change from SMuRF system to SMuRF system.
        - Checks if parameters have valid values (e.g., some parameters
          can only be either 0 or 1, or must be in a predefined interval,
          etc.).
        - Performs validation of some known higher level configuration
          data interdependencies (e.g. prevents the user from defining an
          RTM DAC as both a TES bias and an RF amplifier bias).

        If validation fails, `schema` will raise a `SchemaError` exception
        and fail to load the configuration data, forcing the user to fix
        the cause of the `SchemaError` exception before the configuration
        file can be loaded and used.

        Args
        ----
        loaded_config : dict
            Dictionary of pysmurf configuration parameters to run
            `schema` validation on.

        Returns
        -------
        validated_config : dict
            Dictionary of validated configuration parameters.
            Parameter values are conditioned by `schema` to conform to
            the specified types.

        Raises
        ------
        SchemaError
            Raised if the configuration data fails `schema` validation.

        """
        # Import useful schema objects
        from schema import Schema, And, Use, Optional, Regex

        # Start with an extremely limited validation to figure out
        # things that we need to validate the entire configuration
        # file, like which bands are being used.  This also gets used
        # to construct the init:bands list, from which band_# are
        # present.
        band_schema = Schema({'init' : {Regex('band_[0-7]') : {}}}, ignore_extra_keys=True)
        band_validated = band_schema.validate(loaded_config)

        # Build list of bands from which band_[0-7] blocks are present.  Will
        # use this to build the rest of the validation schema.
        band_regexp = re.compile("band_([0-7])")
        bands = sorted([int(m.group(1)) for m in (band_regexp.match(s) for s in
                                                  band_validated['init'].keys()) if m])

        ###################################################
        ##### Building full validation schema
        schema_dict = {}

        # Only used if none specified in pysmurf instantiation
        schema_dict['epics_root'] = And(str, len)

        #### Start specifiying init schema
        # init must be present
        schema_dict['init'] = {}

        # Explicitly sets dspEnable to this value in pysmurf setup.  Shouldn't
        # be necessary if defaults.yml is properly configured (dspEnable can
        # be set there, instead).
        schema_dict['init'][Optional('dspEnable', default=1)] = And(int, lambda n: n in (0, 1))

        # Each band has a configuration block in init.

        # Default data_out_mux definitions.  Should match fw.
        default_data_out_mux_dict = {}
        default_data_out_mux_dict[0] = [2, 3]
        default_data_out_mux_dict[1] = [0, 1]
        default_data_out_mux_dict[2] = [6, 7]
        default_data_out_mux_dict[3] = [8, 9]
        default_data_out_mux_dict[4] = [2, 3]
        default_data_out_mux_dict[5] = [0, 1]
        default_data_out_mux_dict[6] = [6, 7]
        default_data_out_mux_dict[7] = [8, 9]

        for band in bands:
            schema_dict['init'][f'band_{band}'] = {

                # Swap IQ channels on input
                Optional('iq_swap_in', default=0): And(int, lambda n: n in (0, 1)),
                # Swap IQ channels on output
                Optional('iq_swap_out', default=0): And(int, lambda n: n in (0, 1)),

                # Global feedback enable
                Optional('feedbackEnable', default=1): And(int, lambda n: n in (0, 1)),
                # Global feedback polarity
                Optional('feedbackPolarity', default=1): And(int, lambda n: n in (0, 1)),
                # Global feedback gain (might no longer be used in dspv3).
                "feedbackGain" :  And(int, lambda n: 0 <= n < 2**16),
                # Global feedback gain (might no longer be used in dspv3).
                "feedbackLimitkHz" : And(Use(float), lambda f: f > 0),

                ## TODO remove refPhaseDelay and refPhaseDelayFine
                # refPhaseDelay and refPhaseDelayFine are deprected
                # use bandDelayUs instead

                # Number of cycles to delay phase reference
                Optional('refPhaseDelay', default=0): And(int, lambda n: 0 <= n < 2**5),
                # Finer phase reference delay, 307.2MHz clock ticks.  This
                # goes in the opposite direction as refPhaseDelay.
                Optional('refPhaseDelayFine', default=0): And(int, lambda n: 0 <= n < 2**8),

                # use bandDelayUs (microseconds) instead of refPhaseDelay(Fine)
                Optional('bandDelayUs', default=None): And(Use(float), lambda n : 0 <= n < 30),

                # RF attenuator on SMuRF output.  UC=up convert.  0.5dB steps.
                'att_uc': And(int, lambda n: 0 <= n < 2**5),
                # RF attenuator on SMuRF input.  DC=down convert.  0.5dB steps.
                'att_dc': And(int, lambda n: 0 <= n < 2**5),
                # Tone amplitude.  3dB steps.
                'amplitude_scale': And(int, lambda n: 0 <= n < 2**4),

                # data_out_mux
                Optional("data_out_mux",
                         default=default_data_out_mux_dict[band]) : \
                And([Use(int)], list, lambda l: len(l) == 2 and
                    l[0] != l[1] and all(0 <= ll <= 9 for ll in l)),

                ## TODO remove lmsDelay
                # lmsDelay is deprected use bandDelayUs instead

                # Matches system latency for LMS feedback (9.6 MHz
                # ticks, use multiples of 52).  For dspv3 to adjust to
                # match refPhaseDelay*4 (ignore refPhaseDelayFine for
                # this).  If not provided and lmsDelay=None, sets to
                # lmsDelay = 4 x refPhaseDelay.
                Optional('lmsDelay', default=None) : And(int, lambda n: 0 <= n < 2**5),

                # Adjust trigRstDly such that the ramp resets at the flux ramp
                # glitch.  2.4 MHz ticks.
                'trigRstDly': And(int, lambda n: 0 <= n < 2**7),

                # LMS gain, powers of 2
                'lmsGain': And(int, lambda n: 0 <= n < 2**3),
            }
        #### Done specifying init schema

        #### Start specifying attenuator schema
        # Here's another one that users probably shouldn't be touching Just
        # doing basic validation here - not checking if they're distinct, for
        # instance.
        schema_dict["attenuator"] = {
            'att1' : And(int, lambda n: 0 <= n < 4),
            'att2' : And(int, lambda n: 0 <= n < 4),
            'att3' : And(int, lambda n: 0 <= n < 4),
            'att4' : And(int, lambda n: 0 <= n < 4)
        }
        #### Done specifying attenuator schema

        #### Start specifying cryostat card schema
        ## SHOULD MAKE IT SO THAT WE JUST LOAD AND VALIDATE A SEPARATE,
        ## HARDWARE SPECIFIC CRYOSTAT CARD CONFIG FILE.  FOR NOW, JUST DO
        ## SOME VERY BASIC VALIDATION.
        def represents_int(string):
            try:
                int(string)
                return True
            except ValueError:
                return False

        schema_dict["pic_to_bias_group"] = {And(str, represents_int) : int}
        schema_dict["bias_group_to_pair"] = {And(str, represents_int) : [int, int]}
        #### Done specifying cryostat card schema

        #### Start specifiying amplifier
        schema_dict["amplifier"] = {

            Optional('hemt', default = {
                'drain_offset': 0.9668,
                'drain_opamp_gain': 1,
                'drain_pic_address': 0x3,
                'drain_resistor': 200.0,
                'gate_bit_to_volt': 1.93e-06,
                'gate_dac_num': 33,
                'gate_volt_default': 0.265,
                'gate_volt_min': 0,
                'gate_volt_max': 2.03,
                'power_bitmask': 0b1,
                'power_default': False
            }): {
                'gate_dac_num': Use(int)
            },

            Optional('50k', default = {
                'drain_offset': 0.2643,
                'drain_opamp_gain': 1,
                'drain_pic_address': 0x4,
                'drain_resistor': 10.0,
                'gate_bit_to_volt': 3.38e-06,
                'gate_dac_num': 32,
                'gate_volt_default': 0,
                'gate_volt_min': 0,
                'gate_volt_max': 2.03,
                'power_bitmask': 0b10,
                'power_default': False
            }): {
                'gate_dac_num': Use(int)
            },

            Optional('hemt1', default = {
                'drain_conversion_b': 1.74185,
                'drain_conversion_m': -0.259491,
                'drain_dac_num': 31,
                'drain_offset': 0.9668,
                'drain_opamp_gain': 3.874,
                'drain_pic_address': 0x3,
                'drain_resistor': 50.0,
                'drain_volt_default': 0.5,
                'drain_volt_min': 0,
                'drain_volt_max': 2,
                'gate_bit_to_volt': 3.86936e-6,
                'gate_dac_num': 33,
                'gate_volt_default': 0.265,
                'gate_volt_min': 0,
                'gate_volt_max': 2.03,
                'power_bitmask': 0b1,
                'power_default': False
            }): {
                'gate_dac_num': Use(int)
            },

            Optional('hemt2', default = {
                'drain_conversion_m': -0.259491,
                'drain_conversion_b': 1.74185,
                'drain_dac_num': 29,
                'drain_offset': 0.9668,
                'drain_opamp_gain': 3.874,
                'drain_pic_address': 0x0a,
                'drain_resistor': 50.0,
                'drain_volt_default': 0.5,
                'drain_volt_max': 2,
                'drain_volt_min': 0,
                'gate_bit_to_volt': 3.86936e-6,
                'gate_dac_num': 27,
                'gate_volt_default': 0.265,
                'gate_volt_max': 2.03,
                'gate_volt_min': 0,
                'power_bitmask': 0b100,
                'power_default': False
            }): {
                'gate_dac_num': Use(int)
            },

            Optional('50k1', default = {
                'drain_conversion_m': -0.224968,
                'drain_conversion_b': 1.74185,
                'drain_dac_num': 32,
                'drain_offset': 0.2643,
                'drain_opamp_gain': 9.929,
                'drain_pic_address': 0x04,
                'drain_resistor': 10.0,
                'drain_volt_default': 5.0,
                'drain_volt_max': 5.5,
                'drain_volt_min': 3.5,
                'gate_bit_to_volt': 3.86936e-6,
                'gate_dac_num': 30,
                'gate_volt_default': 0,
                'gate_volt_max': 2.03,
                'gate_volt_min': 0,
                'power_bitmask': 0b10,
                'power_default': False
            }): {
                'gate_dac_num': Use(int)
            },

            Optional('50k2', default = {
                'drain_conversion_m': -0.224968,
                'drain_conversion_b': 5.59815,
                'drain_dac_num': 28,
                'drain_offset': 0.2643,
                'drain_opamp_gain': 9.929,
                'drain_pic_address': 0x0b,
                'drain_resistor': 10.0,
                'drain_volt_default': 5.0,
                'drain_volt_max': 5.5,
                'drain_volt_min': 3.5,
                'gate_bit_to_volt': 3.86936e-6,
                'gate_dac_num': 26,
                'gate_volt_default': 0,
                'gate_volt_max': 2.03,
                'gate_volt_min': 0,
                'power_bitmask': 0b1000,
                'power_default': False
            }): {
                'gate_dac_num': Use(int)
            },
        }
        #### Done specifiying amplifier

        #### Start specifying tune parameter schema
        schema_dict['tune_band'] = {
            'fraction_full_scale' : And(Use(float), lambda f: 0 < f <= 1.),
            'reset_rate_khz' : And(Use(float), lambda f: 0 <= f <= 100),
            Optional('default_tune', default=None) : And(str, os.path.isfile)
        }

        ## Add tuning params that must be specified per band.
        per_band_tuning_params = [
            ('lms_freq', And(Use(float), lambda f: f > 0)),
            ('delta_freq', And(Use(float), lambda f: f > 0)),
            ('feedback_start_frac', And(Use(float), lambda f: 0 <= f <= 1)),
            ('feedback_end_frac', And(Use(float), lambda f: 0 <= f <= 1)),
            ('gradient_descent_gain', And(Use(float), lambda f: f > 0)),
            ('gradient_descent_averages', And(Use(int), lambda n: n > 0)),
            ('gradient_descent_converge_hz', And(Use(float), lambda f: f > 0)),
            ('gradient_descent_momentum', And(Use(int), lambda n: n >= 0)),
            ('gradient_descent_step_hz', And(Use(float), lambda f: f > 0)),
            ('gradient_descent_beta', And(Use(float), lambda f: 0 <= f <= 1)),
            ('eta_scan_averages', And(Use(int), lambda n: n > 0)),
            ('eta_scan_del_f', And(Use(int), lambda n: n > 0)),
        ]

        for band in bands:
            for (param, value) in per_band_tuning_params:
                if band == bands[0]:
                    schema_dict['tune_band'][param] = {}
                schema_dict['tune_band'][param][str(band)] = value
        ## Done adding tuning params that must be specified per band.

        #### Done specifying tune parameter schema

        #### Start specifying bad mask
        schema_dict[Optional('bad_mask', default={})] = {
            # Why are these indexed by integers that are also strings?
            # I don't think the keys here are used at all.
            Optional(str) : And([Use(float)], list, lambda l: len(l) == 2 and
                                l[0] < l[1] and all(4000 <= ll <= 8000 for ll in l))
        }
        #### Done specifying bad mask

        #### Start specifying TES-related
        # TES shunt resistance
        schema_dict["R_sh"] = And(Use(float), lambda f: f > 0)

        # Round-trip resistance on TES bias lines, in low current mode.
        # Includes the resistance on the cryostat cards, and cable resistance.
        schema_dict["bias_line_resistance"] = And(Use(float), lambda f: f > 0)

        # Ratio between the current per DAC unit in high current mode to the
        # current in low current mode.  Constained to be greater than or equal
        # to 1 since otherwise what does high current mode EVEN MEAN.
        schema_dict["high_low_current_ratio"] = And(Use(float), lambda f: f >= 1)

        # If 1, TES biasing will *always* be in high current mode.
        schema_dict[Optional('high_current_mode_bool', default=0)] = And(int, lambda n: n in (0, 1))

        # All SMuRF bias groups with TESs connected.
        schema_dict["all_bias_groups"] = And([int], list, lambda l:
                                             all(0 <= ll < 16 for ll in l))
        #### Done specifying TES-related

        #### Start specifying flux ramp-related
        schema_dict["flux_ramp"] = {
            # 20 bits for the C0 RTMs, 32 bits for the C1 RTMs.
            "num_flux_ramp_counter_bits" : And(int, lambda n: n in (20, 32))
        }
        #### Done specifying flux-ramp related

        #### Start specifying constants schema
        # If all of a schema dictionary's keys are optional, must specify
        # them both in the schema key for that dictionary, and in the
        # schema for that dictionary.
        constants_default_dict = {'pA_per_phi0' : 9.e6}
        cdd_key = Optional("constant", default=constants_default_dict)
        schema_dict[cdd_key] = {}
        # Assumes all constants default values are floats
        for key, value in constants_default_dict.items():
            schema_dict[cdd_key][Optional(key, default=value)] = Use(float)
        #### Done specifying constants schema

        #### Start thermal schema
        # OT protection for ultrascale FPGA, in degrees C.  If None,
        # then pysmurf doesn't try to engage OT protection.  For
        # unknown reasons, enabling OT protection in the ELMA crate
        # we've been using for testing on campus at Stanford takes
        # down the carrier after the third command in the enable
        # sequence (set_ot_threshold_disable(0)), but it works in the
        # RF lab at SLAC where they've been testing with an ASIS
        # crate.  Shawn has yet to have this work for him.  Newer fw
        # versions will have OT protection enabled in the fw.
        schema_dict[
            Optional('ultrascale_temperature_limit_degC',
                     default=None)] = And(Use(float),
                                          lambda f: 0 <= f <= 99)
        #### Done specifying thermal schema

        #### Start specifying timing-related schema
        schema_dict["timing"] = {
            # "ext_ref" : internal oscillator locked to an external
            #   front-panel reference, or unlocked if there is no front
            #   panel reference.  (LmkReg_0x0147 : 0x1A).  Also sets
            #   flux_ramp_start_mode=0
            # "backplane" : takes timing from timing master through
            #   backplane.  Also sets flux_ramp_start_mode=1.
            "timing_reference" : And(str, lambda s: s in ('ext_ref', 'backplane', 'fiber'))
        }
        #### Done specifying timing-related schema

        #### Start specifying smurf2mce
        # System should be smart enough to determine fs on the fly.
        schema_dict["fs"] = And(Use(float), lambda f: f > 0)

        def user_has_write_access(dirpath):
            return os.access(dirpath, os.W_OK)

        def dir_exists_with_write_access(file_path):
            filedir_path = os.path.dirname(file_path)
            if (not os.path.isdir(filedir_path) or
                    not user_has_write_access(filedir_path)):
                return False
            return True

        #### Start specifying directories
        schema_dict[
            Optional(
                "default_data_dir",
                default="/data/smurf_data")] = And(str,
                                                   os.path.isdir,
                                                   user_has_write_access)
        schema_dict[
            Optional(
                "smurf_cmd_dir",
                default="/data/smurf_data/smurf_cmd")] = And(str,
                                                             os.path.isdir,
                                                             user_has_write_access)
        schema_dict[
            Optional("tune_dir",
                     default="/data/smurf_data/tune")] = And(str,
                                                             os.path.isdir,
                                                             user_has_write_access)
        schema_dict[
            Optional(
                "status_dir",
                default="/data/smurf_data/status")] = And(str,
                                                          os.path.isdir,
                                                          user_has_write_access)
        #### Done specifying directories

        ##### Done building validation schema
        ###################################################

        ###################################################
        # Validate the full config
        schema = Schema(schema_dict, ignore_extra_keys=True)
        validated_config = schema.validate(loaded_config)

        ###################################################
        # Higher level/composite validation, if schema validation
        # succeeds

        # Check that no DAC has been assigned to multiple TES bias groups
        bias_group_to_pair = validated_config['bias_group_to_pair']
        tes_bias_group_dacs = np.ndarray.flatten(
            np.array([bg2p[1] for bg2p in bias_group_to_pair.items()]))
        assert (len(np.unique(tes_bias_group_dacs)) ==
                len(tes_bias_group_dacs)), (
                    'Configuration failed - DACs may not be ' +
                    'assigned to multiple TES bias groups.')

        ##### Done with higher level/composite validation.
        ###################################################

        # Splice in the sorted init:bands key before returning
        validated_config['init']['bands'] = bands

        return validated_config
