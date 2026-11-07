#!/usr/bin/env bash

# Build the pysmurf server image and push it to ghcr.io/slaclab/pysmurf-server:<tag>:
# pysmurf and what a server needs to run, plus the firmware releases and the configuration
# files named in definitions.sh under /tmp/fw.
#
# The pysmurf built is this checkout's HEAD as the remote has it -- commit and push first,
# uncommitted changes stay out of the image; <tag> is `git describe --tags --always`
# of it, so a release is tagged by its release tag and any other commit by the
# nearest tag, its distance and its hash (v11.4.4-12-g<hash>).
#
# The firmware files are downloaded into 'local_files', which the Dockerfile
# copies to /tmp/fw. Downloading the private releases needs
# GITHUB_TOKEN in the environment.
#
# Environment:
#   PUSH=0            build and tag only (the default pushes)
#   SYSTEMS=a,b       only these FW_SYSTEMS names, e.g. SYSTEMS=umux,rfmc (default: all)
#   REF=<ref>         build this pysmurf tag, branch or commit instead of HEAD
#   TAG=<tag>         tag the images so instead of by `git describe`

cd "$(dirname "$0")" || exit 1

# Load the user definitions
. definitions.sh

# Call the validation script
. validate.sh

# Image repository definitions
dockerhub_org_name='ghcr.io/slaclab'
dockerhub_repo='pysmurf-server'

# What to build, and what to call it.
ref=${REF:-$(git rev-parse HEAD)}
tag=${TAG:-$(git describe --tags --always ${REF:+"${REF}"})}
push=${PUSH:-1}
systems=${SYSTEMS:-}

echo "pysmurf ${ref} -> ${dockerhub_repo}:${tag}"

# Create temporary local file directory to hold firmware and configuration
# files which will later be copied to the docker image
rm -rf local_files && mkdir local_files

# Download firmware and ZIP assets for each defined system
echo "Downloading firmware files..."
for entry in "${FW_SYSTEMS[@]}"; do
    IFS='|' read -r sys_name sys_repo sys_tag sys_fw_files sys_zip <<< "${entry}"
    if [ -n "${systems}" ] && [[ ",${systems}," != *",${sys_name},"* ]]; then
        echo "  [${sys_name}] skipped (SYSTEMS=${systems})"
        continue
    fi
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

# Build the image.
echo "Building docker image..."
docker image build \
    --build-arg ref="${ref}" --build-arg smurf_pcie_tag="${smurf_pcie_tag}" \
    -t pysmurf_server . || exit 1

docker image tag pysmurf_server ${dockerhub_org_name}/${dockerhub_repo}:${tag}

if [ "${push}" != 1 ]; then
    echo "Built '${dockerhub_org_name}/${dockerhub_repo}:${tag}' (PUSH=${push}: not pushed)"
    exit 0
fi

docker image push ${dockerhub_org_name}/${dockerhub_repo}:${tag} || exit 1
echo "Docker image '${dockerhub_org_name}/${dockerhub_repo}:${tag}' pushed"
