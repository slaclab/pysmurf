#!/usr/bin/env python
# Copyright 2019 Matthew Hasselfield.
# Redistribution and use of this code are permitted under the terms
# of the LICENSE in this repository.

# From here: https://github.com/mhasself/stdpub/blob/master/pub.py

import json
import os
import socket
import sys
import time

import pysmurf

DEFAULT_ENV_ROOT = 'SMURFPUB_'
DEFAULT_UDP_PORT = 8200
UDP_MAX_BYTES = 64000

class Publisher:
    seq_no = 0
    host = 'localhost'
    script_id = 'undeclared'
    pub_id = 'undeclared'
    env_root = DEFAULT_ENV_ROOT

    def __init__(self, script_id=None, options={}, env_root=None):
        """The Publisher should normally be instantiated with just the
        script_id, e.g.:

          pub = Publisher('tuning_script')

        Configuration options - these should be set in the options
        dictionary, or through environment variables using the
        appropriate prefix (see module DEFAULT_ENV_ROOT).

        ID: an ID string associated with this system in order to
          disambiguate it from others; e.g. "readout_crate_1".

        BACKEND: a string that selects the backend publishing engine.
          Options are: "null" and "udp".  The null backend is the
          default, and in that case the published messages are simply
          discarded.

        Options for BACKEND="udp":

        UDP_HOST: the target host for UDP packets.  Defaults to
          localhost.

        UDP_PORT: the target port for UDP packets.  Defaults to
          module DEFAULT_UDP_PORT.

        """
        # Basic setup.
        if env_root is not None:
            self.env_root = env_root
        self.host = socket.gethostname()
        if script_id is not None:
            self.script_id = script_id
        self.pub_id = self._getoptenv('ID', options, 'undeclared')

        # Now decode the backend-specific setup.
        backend = self._getoptenv('BACKEND', options, 'null')

        if backend == 'null':
            self._backend = self._backend_null

        elif backend == 'udp':
            self.udp_sock = socket.socket(socket.AF_INET,
                                          socket.SOCK_DGRAM)
            self.udp_ip = self._getoptenv('UDP_HOST', options, 'localhost')
            self.udp_port = int(self._getoptenv('UDP_PORT', options, DEFAULT_UDP_PORT))
            self._backend = self._backend_udp

        else:
            sys.stderr.write('%s: no backend for "%s", selecting null Publisher.\n' %
                             (self.__class__, backend))
            self._backend = self._backend_null

        # Issue the start message.
        self.log_start()

    def __del__(self):
        self.log_stop()

    def _getoptenv(self, key, options, default=None):
        """Look for `key` in dict `options`, and if it's not found then try to
        get it from the environment (including the local `env_root`
        prefix), and if it's not found then return the `default`.

        """
        if key in options:
            return options[key]
        return os.getenv(self.env_root + key, default)

    # Backend send functions.  One of these will be assigned to
    # self._backend.

    def _backend_null(self, json_msg):
        pass

    def _backend_udp(self, json_msg):
        payload = bytes(json_msg, 'utf_8')
        if len(payload) > UDP_MAX_BYTES:
            # Can't send this; write to stderr and notify consumer.
            error = 'Backend error: dropped large UDP packet (%i bytes).' % len(payload)
            sys.stderr.write('%s %s' % (self, error))
            self.publish({'message': error}, 'backend_error')
            return
        self.udp_sock.sendto(payload, (self.udp_ip, self.udp_port))

    # Publishing functions -- external interface.

    def publish(self, data, msgtype='general'):
        # Create the wrapper.
        output = {
            'host': self.host,
            'id': self.pub_id,
            'script': self.script_id,
            'time': time.time(),
            'seq_no': self.seq_no,
            'type': msgtype,
        }
        self.seq_no += 1
        # Add in the data and convert to json.
        output['payload'] = data
        jtext = json.dumps(output)
        # Send.
        self._backend(jtext)

    def log(self, message):
        return self.publish({'message': message}, 'log')

    def log_start(self):
        """Publishes start message"""
        return self.publish({}, 'start')

    def log_stop(self):
        """Publishes stop message"""
        return self.publish({}, 'stop')

    def register_file(self, path, type, format='', timestamp=None, plot=False):
        """
        Publishes file info so it can be picked up by the pysmurf-archiver.

        Args
        ----
        path : str
            Full path to file.
        type : str
            Type of data file, e.g. "tuning" or "config_snapshot".
        format : str
            File extension. E.g. "npy" or "txt".
        timestamp : float or None, optional, default None
            Unix timestamp when file was created.
        plot : bool, optional, default False
            True if file is a plot
        """
        if timestamp is None:
            timestamp = time.time()

        file_data = {
            'path': path,
            'type': type,
            'format': format,
            'timestamp': timestamp,
            'plot': plot,
            'pysmurf_version': pysmurf.__version__
        }

        return self.publish(file_data, 'data_file')
