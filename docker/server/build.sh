#!/usr/bin/env bash

# This script builds the stable pysmurf server docker image, which includes the firmware and
# configuration files, as well as pysmurf.
#
# The firmware and configuration files versions are defined in the 'definitions.sh' file, while the
# version of pysmurf is extracted from this copy of the repository (using the  `git describe --tags
# --always' command).
#
# This scripts downloads the firmware and configuration files into a temporal directory called
# 'local_files' which is then used by the Dockerfile to generate the final docker image.
# Finally, the docker image is pushed to Dockerhub, to the repository defined by the variables
# ${dockerhub_org_name} and ${dockerhub_repo_stable}.
#
# A 'base' pysmurf server image is not longer generated, but in order to be backward compatible
# with scripts that use that image, the same stable image is pushed to the Dockerhub repository
# defined by ${dockerhub_org_name} and ${dockerhub_repo_base} as well; this image can be used as
# base image as well, just like previous images. Finally, the docker images are tag with the same
# tag from this repository.

# Load the user definitions
. definitions.sh

# Call the validation script
. validate.sh

# Dockerhub repositories definitions
dockerhub_org_name='tidair'
dockerhub_repo_stable='pysmurf-server'
dockerhub_repo_base='pysmurf-server-base'

# Get the git tag, which will be used to tag the docker image.
# At the moment, this script runs only on tagged releases.
#tag=`git describe --tags --always`
tag='rogue_v6'

# Create temporal local file directory to hold firmware and configuration
# files which will later be copied to the docker image
rm -rf local_files && mkdir local_files

# Get the  MCS file from the assets
echo "Downloading MCS file..."
(cd local_files && get_private_asset ${fw_repo} ${fw_repo_tag} ${mcs_file_name}) || exit 1

# Get the ZIP file from the assets
echo "Downloading ZIP file..."
(cd local_files && get_private_asset ${fw_repo} ${fw_repo_tag} ${zip_file_name}) || exit 1

#  Get the configuration files. We clone the whole repository
echo "Downloading configuration files..."
git -C local_files clone -c advice.detachedHead=false ${config_repo} -b ${config_repo_tag} || exit 1

# Build the docker image. This same image will be pushed to both the stable
# and the base dockerhub repositories.
echo "Building docker image..."
docker image build --build-arg branch=${tag} -t docker_image . || exit 1

# Tag and push the image to the stable dockerhub repository
#docker image tag docker_image ${dockerhub_org_name}/${dockerhub_repo_stable}:${tag}
#docker image push ${dockerhub_org_name}/${dockerhub_repo_stable}:${tag}
#echo "Docker image '${dockerhub_org_name}/${dockerhub_repo_stable}:${tag}' pushed"

# Tag and push the image to the stable dockerhub repository
#docker image tag docker_image ${dockerhub_org_name}/${dockerhub_repo_base}:${tag}
#docker image push ${dockerhub_org_name}/${dockerhub_repo_base}:${tag}
#echo "Docker image '${dockerhub_org_name}/${dockerhub_repo_base}:${tag}' pushed"
