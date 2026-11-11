#!/usr/bin/env ipython3
# The client session start_client.sh opens: a SmurfControl connected to the server
# SMURF_TARGET names ('host:port' or 'crate:<slot>'), with the configuration file
# CONFIG_FILE if one was given. On a failed connection the error is shown and the
# session stays open with `pysmurf`, `target` and `config_file` defined, so the
# connection can be retried by hand:
#
#   S = pysmurf.client.SmurfControl(cfg_file=config_file, server_addr=host, server_port=port)

import os

import numpy as np                    # noqa: F401  (for the interactive session)
import matplotlib.pyplot as plt       # noqa: F401

import cryodaq
import pysmurf.client

target = os.getenv('SMURF_TARGET', 'localhost:9000')
config_file = os.getenv('CONFIG_FILE')
host, port = cryodaq.endpoint_of(target)

how = f"with the configuration {config_file}" if config_file else "reading the configuration back from it"
print(f"Connecting to the server at {host}:{port}, {how}")
S = pysmurf.client.SmurfControl(cfg_file=config_file, server_addr=host, server_port=port)
