#!/usr/bin/env bash

# This script builds the stable pysmurf server docker image, which includes the firmware and
# configuration files, as well as pysmurf.
#
# The firmware and configuration files versions are defined in the 'definitions.sh' file, while the
# version of pysmurf is extracted from this copy of the repository (using the
# `git describe --tags --always` command).
#
# This script downloads the firmware and configuration files into a temporary directory called
# 'local_files' which is then used by the Dockerfile to generate the final docker image.
# Finally, the docker image is pushed to the repository defined by ${dockerhub_org_name}.

# Load the user definitions
. definitions.sh

# Call the validation script
. validate.sh

# Dockerhub repositories definitions
dockerhub_org_name='ghcr.io/slaclab'
dockerhub_repo_stable='pysmurf-server'
dockerhub_repo_base='pysmurf-server-base'

# Get the git tag, which will be used to tag the docker image.
# At the moment, this script runs only on tagged releases.
tag=`git describe --tags --always`

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

# Build the docker image. This same image will be pushed to both the stable
# and the base dockerhub repositories.
echo "Building docker image..."
docker image build --build-arg branch=${tag} -t pysmurf_server . || exit 1

# Tag and push the image to the stable dockerhub repository
docker image tag pysmurf_server ${dockerhub_org_name}/${dockerhub_repo_stable}:${tag}
docker image push ${dockerhub_org_name}/${dockerhub_repo_stable}:${tag}
echo "Docker image '${dockerhub_org_name}/${dockerhub_repo_stable}:${tag}' pushed"

# Tag and push the image to the base dockerhub repository (for backwards compatibility)
docker image tag pysmurf_server ${dockerhub_org_name}/${dockerhub_repo_base}:${tag}
docker image push ${dockerhub_org_name}/${dockerhub_repo_base}:${tag}
echo "Docker image '${dockerhub_org_name}/${dockerhub_repo_base}:${tag}' pushed"
