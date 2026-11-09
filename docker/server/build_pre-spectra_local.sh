#!/usr/bin/env bash

# This script builds a local pre-spectra pysmurf server docker image for development/testing.
# Unlike build.sh, it does not push to any registry, and tags the image as ':latest'
# rather than by a release tag.
#
# TO BE DELETED: a stopgap from before build.sh could build without pushing. `PUSH=0
# build.sh` now builds the same image from the same definitions, tagged by `git describe`,
# and is what to use; this script stays only until its users have moved over.

# Load the user definitions
. definitions.sh

# Call the validation script
. validate.sh

# Dockerhub repositories definitions
dockerhub_repo_stable='pre-spectra-server'
dockerhub_repo_base='pre-spectra-server-base'

# Create temporary local file directory to hold firmware and configuration
# files which will later be copied to the docker image
rm -rf local_files && mkdir local_files

# Download firmware and ZIP assets for each defined system
echo "Downloading firmware files..."
for entry in "${FW_SYSTEMS[@]}"; do
    IFS='|' read -r sys_name sys_repo sys_tag sys_fw_files sys_zip <<< "${entry}"
    # Fall back to rogue_<tag>.zip if no explicit zip filename was specified
    sys_zip="${sys_zip:-rogue_${sys_tag}.zip}"

    # Download each firmware file (semicolon-separated)
    IFS=';' read -ra fw_file_list <<< "${sys_fw_files}"
    for fw_file in "${fw_file_list[@]}"; do
        echo "  [${sys_name}] Downloading firmware: ${fw_file}"
        (cd local_files && get_private_asset "${sys_repo}" "${sys_tag}" "${fw_file}") || exit 1
    done

    # Install the ZIP: copy from disk if an absolute path was given, otherwise download
    if [[ "${sys_zip}" = /* ]]; then
        echo "  [${sys_name}] Copying ZIP from disk: ${sys_zip}"
        cp "${sys_zip}" local_files/ || exit 1
    else
        echo "  [${sys_name}] Downloading ZIP: ${sys_zip}"
        (cd local_files && get_private_asset "${sys_repo}" "${sys_tag}" "${sys_zip}") || exit 1
    fi
done

# Get the configuration files. We clone the whole repository.
echo "Downloading configuration files..."
git -C local_files clone -c advice.detachedHead=false ${config_repo} -b ${config_repo_tag} || exit 1

# Build the docker image at this checkout's HEAD, as the remote has it (commit and push
# first), with the offload card's release pinned in definitions.sh
echo "Building docker image..."
docker image build --build-arg ref="$(git rev-parse HEAD)" --build-arg smurf_pcie_tag="${smurf_pcie_tag}" \
    -t pysmurf_server . || exit 1

# Tag the image locally (no push for local builds)
docker image tag pysmurf_server ${dockerhub_repo_stable}:latest
docker image tag pysmurf_server ${dockerhub_repo_base}:latest
