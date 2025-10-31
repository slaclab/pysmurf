#!/usr/bin/env ipython3

import os
import numpy as np
import matplotlib.pyplot as plt
import pysmurf.client

# The path to the default configuration file, if any, is defined
# in the environmental variable 'CONFIG_FILE'
config_file = os.getenv('CONFIG_FILE')
if config_file == None:
	del config_file
