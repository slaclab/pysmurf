#!/usr/bin/env bash

# Define repositories:
# ====================
# These variables define the firmware and YML configuration repositories URLs.
# Under normal conditions, you should not have to change these definitions.
# - fw_repo: points to the firmware repository, which contains both the
#   MCS and the ZIP files.
# - config_repo: points to the configuration repository, which contains the
#   the YML files.

# Firmware repo for uMUX systems
fw_repo=https://github.com/slaclab/cryo-det
# Firmware repo for TKID systems
tkid_fw_repo=https://github.com/slaclab/cryo-det-kid
# Repo for defaults.ymls
config_repo=https://github.com/slaclab/smurf_cfg

# Define the firmware version:
# ============================
# Define the firmware version to use. This is were you define which firmware
# version to include in the docker image.
# - fw_repo_tag: Set this variable to the tag version you want to use.
# - mcs_file_name: Set this variable to the MCS file name.
# The name of the MCS file is independent from the tag name, so you need to
# define it here. On the other hand, the ZIP file follows this naming convention:
# 'rogue_${fw_repo_tag}.zip', so you don't need to define it here.
# The files will be downloaded from the release list of assets.

# Firmware version for uMUX systems
fw_repo_tag=MicrowaveMuxBpEthGen2_v1.1.1
mcs_file_name=MicrowaveMuxBpEthGen2-0x01010000-20210928123941-ruckman-02ed52a.mcs.gz

# Firmware version for TKID systems
tkid_fw_repo_tag=CryoDetKid_v2.0.0
tkid_mcs_file_name=CryoDetKid-0x00000101-20230620224002-bareese-e3253bd.mcs.gz

# Define the configuration version:
# =================================
# Define the YML configuration version to use. This is were you define which
# configuration version to include in the docker image.
# - 'yml_repo_tag': Set this variable to the tag version you want to use.
config_repo_tag=v2.1.0
