#!/usr/bin/env bash

# Every firmware release the server image carries is pinned here and nowhere
# else: build.sh downloads them, validate.sh checks they exist, and the CI
# workflow reads the tags it resolves the register names against from here.
#
# Define repositories:
# ====================
# - config_repo: points to the configuration repository, which contains the YML files.
# - smurf_pcie_repo/smurf_pcie_tag: the PCIe offload card's firmware; its release archive's
#     Python package is the card's register map, placed on the image's import path (not under /tmp/fw).
# - FW_SYSTEMS: array of firmware system definitions. Each entry is a pipe-separated tuple:
#     name|repo_url|tag|fw_files[|zip_file]
#
#   - name:     short identifier used in log messages and file lookups (no spaces)
#   - repo_url: GitHub HTTPS URL for the firmware repository
#   - tag:      release tag to pull assets from
#   - fw_files: semicolon-separated list of firmware asset filenames to download from that release.
#               For ATCA/PCIe systems this is the MCS file (e.g. MyFirmware-hash.mcs).
#               For RFSoC systems these are the linux.tar.gz images (one per FPGA target variant).
#               check_rfsoc_firmware.sh auto-selects the correct image at runtime based on what
#               is actually loaded on the board.
#   - zip_file: (optional) pyrogue ZIP to use. Three forms are accepted:
#               - Omitted: defaults to rogue_<tag>.zip (the ruckus Primary:True naming convention).
#               - Filename only (e.g. rogue_MicrowaveMuxBpEthGen2_v2.3.1.zip): fetched from the
#                 GitHub release assets (ruckus Primary:False naming convention).
#               - Absolute path (e.g. /path/to/rogue_MicrowaveMuxZcu208_v3.2.0.zip): copied from
#                 local disk. Useful when the zip has been built locally before the docker image.
#
# To add a new system, append a new entry to FW_SYSTEMS.
# To disable a system, comment out its entry.

# Firmware system definitions
# Format: "name|repo_url|tag|fw_files[|zip_file]"  (fw_files is semicolon-separated)
FW_SYSTEMS=(
    # uMUX (standard microwave multiplexer) systems
    # Uses Primary:False naming -> rogue_MicrowaveMuxBpEthGen2_<tag>.zip
    # v2.7.1 is the first release whose package leaves the tuning operations to pysmurf.
    "umux|https://github.com/slaclab/cryo-det|MicrowaveMuxBpEthGen2_v2.7.1|MicrowaveMuxBpEthGen2-0x02070000-20261106193048-ruckman-d3e99291.mcs|rogue_MicrowaveMuxBpEthGen2_v2.7.1.zip"

    # NO LONGER BEING DEVELOPED
    ## TKID (transition-edge sensor kinetic inductance detector) systems
    ##"tkid|https://github.com/slaclab/cryo-det-kid|v2.1.0|CryoDetKid-0x02010000-20240920083819-ruckman-ec69acf.mcs.gz"

    # RFSoC systems (ZCU208). No release of either repository yet leaves the tuning operations to
    # pysmurf (the latest, v4.0.0 and PreSpectra v2.1.1, predate cryo-det v2.7.1), so a server on
    # these packages needs a stripped CryoDet checkout mounted in front of them until one does.
    # fw_files is a semicolon-separated list of linux.tar.gz images, one per FPGA target variant
    # (e.g. BaseBand, HighOrderNyquist). All listed images are downloaded into the docker image.
    # At runtime, checkRFSoCFW() reads the target name from the board via axiversiondump and
    # selects the matching image automatically -- so adding a new variant here is all that is
    # needed to support it. A single pyrogue ZIP covers all RFSoC target variants.
    # Currently only BaseBand is deployed. When HighOrderNyquist is ready, append it like so:
    #   ...66895bd.linux.tar.gz;MicrowaveMuxZcu208_HighOrderNyquist-0x0XXXXXXX-date-ruckman-hash.linux.tar.gz
    #
    # zip_file: the rfsoc zip follows rogue_MicrowaveMuxZcu208_<ver>.zip naming (not rogue_<tag>.zip),
    # so it must be specified explicitly. If the zip has already been built locally it can be given
    # as an absolute path (e.g. /tmp/rogue_MicrowaveMuxZcu208_v3.2.0.zip) and will be copied from
    # disk instead of downloaded. Otherwise give just the filename and it will be fetched from the
    # GitHub release assets.
    "rfsoc|https://github.com/slaclab/zcu208-cryo-det|MicrowaveMuxZcu208_v3.2.1|MicrowaveMuxZcu208_BaseBand-0x03020000-20260408094535-ruckman-66895bd.linux.tar.gz|rogue_MicrowaveMuxZcu208_v3.2.1.zip"
    # Pre-Spectra
    "rfsoc|https://github.com/slaclab/zcu208-cryo-det-pre-spectra|MicrowaveMuxZcu208_PreSpectra_v2.1.1|MicrowaveMuxZcu208_PreSpectra-0x02010000-20260416123535-ruckman-7d4fca8.linux.tar.gz|rogue_MicrowaveMuxZcu208_PreSpectra_v2.1.1.zip"

    # RFMC carrier (uMUX on the RFSoC-based ATCA carrier). One linux.tar.gz image, like the ZCU208;
    # the release zip follows rogue_<tag>.zip naming. Its register defaults are in the package itself.
    # Two firmware lines are released in parallel: version 1 (baseband, with the mixer daughter
    # cards) and version 2 (direct RF sampling). One archive per image name is all the launcher
    # accepts, so one line ships: v1.2.3, on the image of v1.2.1 (the first RFMC release whose
    # package leaves the tuning operations to pysmurf), with the host fixes since: the DSP cores
    # reset at start-up and the high-speed ADC NCO restore. The version 2 line (v2.0.0) still
    # carries the tuning operations.
    "rfmc|https://github.com/slaclab/rfmc-cryo-det|MicrowaveMuxRfmcCarrier_v1.2.3|MicrowaveMuxRfmcCarrier-0x01020000-20261106110105-ruckman-29c6f6b.linux.tar.gz|rogue_MicrowaveMuxRfmcCarrier_v1.2.3.zip"

    # Add additional systems below, one per line:
    # "mysystem|https://github.com/slaclab/my-repo|v1.2.3|MyFirmware-0x01020000-date-hash.mcs"   (the image name carries major.minor only)
)

# The PCIe offload card's firmware (public repository); the release zip is rogue_<tag>.zip.
smurf_pcie_repo=https://github.com/slaclab/smurf-pcie
smurf_pcie_tag=v3.3.1

# Repo for defaults YMLs
config_repo=https://github.com/slaclab/smurf_cfg

# Define the configuration version:
# ==================================
# - config_repo_tag: tag version of the YML configuration repo to include.
config_repo_tag=v3.2.0
