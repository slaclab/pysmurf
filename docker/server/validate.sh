#!/usr/bin/env bash

# Validate definitions.sh
# RTH: Replaced AER7_TOKEN with GITHUB_TOKEN

#############
# FUNCTIONS #
#############

function error {
	echo "Error: $1"
	exit 1
}

# Check if a tag exists on a github public repository
# Arguments:
# - first: github public repository url,
# - second: tag name
check_if_public_tag_exist()
{
    local repo=$1
    local tag=$2
    git ls-remote --refs --tag ${repo} | grep -q refs/tags/${tag} > /dev/null
}

# Check if a asset file exist on a tag version on a github public repository
# Arguments:
# - first: github public repository url,
# - second: tag name,
# - third: asset file name
check_if_public_asset_exist()
{
    local repo=$1
    local tag=$2
    local file=$3
    curl --head --silent --fail ${repo}/releases/download/${tag}/${file} > /dev/null
}

# Check if a tag exists on a github pivate repository.
# It requires ssh key authentication.
# Arguments:
# - first: github private repository url (https),
# - second: tag name
check_if_private_tag_exist()
{
    # e.g. https://github.com/slaclab/cryo-det turns into
    # https://api.github.com/repos/slaclab/cryo-det/tags. Because cryo-det is
    # private, you need some authentication that can read the repository.

    local api_url=$(echo $1 | sed -e "s|github.com/|api.github.com/repos/|g")/tags
    local tag=$2

    echo "Searching for tag ${tag} in repo $1 with API URL ${api_url} ..."

    # Big blob of JSON with the tags in it.
    local tags=$(curl \
      --url $api_url \
      --header "Authorization: token ${GITHUB_TOKEN}" \
      --fail)

    # e.g. 0 if no match
    local grep_count=$(echo "$tags" | grep -c "$tag")

    if [ "$grep_count" -eq 0 ]; then
	    error "Tag ${tag} doesn't exist using API URL ${api_url} for repo $1"
    else
	    echo "Found."
    fi
}

# Check if a asset file exist on a tag version on a github private repository.
# It requires the access token to be defined in $AER7_TOKEN.
# Arguments:
# - first: github private repository url (https),
# - second: tag name,
# - third: asset file name
check_if_private_asset_exist()
{
    local repo=$(echo $1 | sed -e "s|https://github.com|https://api.github.com/repos|g")
    local tag=$2
    local file=$3

    # Search the asset ID in the specified release
    local r=$(curl --silent --header "Authorization: token ${GITHUB_TOKEN}" "${repo}/releases/tags/${tag}")
    eval $(echo "${r}" | grep -C3 "name.:.\+${file}" | grep -w id | tr : = | tr -cd '[[:alnum:]]=')

    # return is the asset tag was found
    [ "${id}" ]
}

# Download the asset file on a tagged version on a github private repository.
# It requires the access token to be defined in $AER7_TOKEN.
# Arguments:
# - first: github private repository url (https),
# - second: tag name,
# - third: asset file name
get_private_asset()
{
    local repo=$(echo $1 | sed -e "s|https://github.com|https://api.github.com/repos|g")
    local tag=$2
    local file=$3

    # Check if the asset exist, and get it's ID
    check_if_private_asset_exist ${repo} ${tag} ${file} || exit 1

    echo "Downloading ${file}..."

    # Try to download the asset
    curl --fail --location --remote-header-name --remote-name --progress-bar \
         --header "Authorization: token ${GITHUB_TOKEN}" \
         --header "Accept: application/octet-stream" \
         "${repo}/releases/assets/${id}"
}

# Check if file exist on a tag version on a github repository
# Arguments:
# - first: github repository url,
# - second: tag name,
# - third: file name (must be a full path on that repository)
check_if_file_exist()
{
    local repo=$1
    local tag=$2
    local file=$3
    curl --head --silent --fail ${repo}/blob/${tag}/${file} > /dev/null
}

# Exit with an error message, and with return code = 1
exit_on_error()
{
    echo
    echo "Validation failed! The 'definitions.sh' file is incorrect!"
    echo
    exit 1
}

# Exist with a success message and with return core = 0
exit_on_success()
{
    echo
    echo "Success! The 'definitions.sh' file is correct!"
    echo
}

#############
# MAIN BODY #
#############

# Load the user definitions
. definitions.sh

# Extra definitions, generated from the user definitions
zip_file_name=rogue_${fw_repo_tag}.zip

# Validate if repositories were defined
echo "======================================================================================"
echo "Repository names validation"
echo "======================================================================================"

printf "Checking if firmware repository was defined...      "
if [ -z ${fw_repo} ]; then
    echo "Failed!"
    echo
    echo "Firmware repository not define! Please check that the variable 'fw_repo' is defined in the 'definitions.sh' file."
    exit_on_error
fi
echo "${fw_repo}"

printf "Checking if configuration repository was defined... "
if [ -z ${config_repo} ]; then
    echo "Failed!"
    echo
    echo "Configuration repository not define! Please check that the variable 'config_repo' is defined in the 'definitions.sh' file."
    exit_on_error
fi
echo "${config_repo}"

# At this point all repository name definition are correct.
echo "Done! All repository names were defined!"
echo

# Validate the firmware version
echo "======================================================================================"
echo "Firmware version validation"
echo "======================================================================================"

printf "Checking if the tag version was defined...          "
if [ -z ${fw_repo_tag} ]; then
    echo "Failed!"
    echo
    echo "Firmware tag version not defined! Please check that the variable 'fw_repo_tag' is defined in the 'definitions.sh' file."
    exit_on_error
fi
echo "${fw_repo_tag}"

printf "Checking if MCS file name was defined...            "
if [ -z ${mcs_file_name} ]; then
    echo "Failed!"
    echo
    echo "MCS file name not defined! Please check that the variable 'mcs_file_name' is defined in the 'definitions.sh' file."
    exit_on_error
fi
echo "${mcs_file_name}"

check_if_private_tag_exist ${fw_repo} ${fw_repo_tag}

printf "Checking if MCS file is in the list of assets...    "
check_if_private_asset_exist ${fw_repo} ${fw_repo_tag} ${mcs_file_name}
if [ $? != 0 ]; then
    echo "Failed!"
    echo
    echo "File '${mcs_file_name}' does not exist in the assets list of release '${fw_repo_tag}'!"
    exit_on_error
fi
echo "File exist!"

printf "Checking if ZIP file is in the list of assets...    "
check_if_private_asset_exist ${fw_repo} ${fw_repo_tag} ${zip_file_name}
if [ $? != 0 ]; then
    echo "Failed!"
    echo
    echo "File '${zip_file_name}' does not exist in the assets list of release '${fw_repo_tag}'!"
    exit_on_error
fi
echo "File exist!"

# At this points all the definition of the firmware version are correct.
echo "Done! A correct firmware version was defined"
echo


# Validate the configuration version
echo "======================================================================================"
echo "Configuration version validation"
echo "======================================================================================"

printf "Checking if the tag version was defined...          "
if [ -z ${config_repo_tag} ]; then
    echo "Failed!"
    echo
    echo "Configuration tag version not defined! Please check that the variable 'config_repo_tag' is defined in the 'definitions.sh' file."
fi
echo "${config_repo_tag}"

printf "Checking if release exist...                        "
check_if_public_tag_exist ${config_repo} ${config_repo_tag}
if [ $? != 0 ]; then
    echo "Failed!"
    echo
    echo "Release '${config_repo_tag}' does not exist in repository '${config_repo}'!"
    exit_on_error
fi
echo "Release exist!"

# At this point all the definitions of the configuration version are correct.
echo "Done! A correct configuration version was defined"

# At this point all definition where correct
exit_on_success
