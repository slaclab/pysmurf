.. _client_config:

config module
=============

The client's configuration: a YAML file over the packaged default, resolved by
:mod:`cryodaq.config` with this package's schema as the judge of what the keys
mean. See :ref:`configuration` for the file format.

.. automodule:: pysmurf.client.config
    :members: load, load_mapping, DEFAULT

schema
------
.. automodule:: pysmurf.client.config.schema
    :members: validate, ConfigInvalid

legacy
------
.. automodule:: pysmurf.client.config.legacy
    :members: convert, read_json_with_comments, to_yaml, DROPPED_KEYS
