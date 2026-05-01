#!/usr/bin/env bash

# Validate definitions.sh

#############
# FUNCTIONS #
#############

function error {
	echo "Error: $1"
	exit 1
}

# Check if a tag exists on a GitHub public repository
# Arguments: repo_url tag
check_if_public_tag_exist()
{
    local repo=$1
    local tag=$2
    git ls-remote --refs --tag ${repo} | grep -q refs/tags/${tag} > /dev/null
}

# Check if a tag exists on a GitHub private repository.
# Paginates through all tags (100 per page) so repos with many releases are
# handled correctly. Requires $GITHUB_TOKEN.
# Arguments: repo_url tag
check_if_private_tag_exist()
{
    local api_base=$(echo $1 | sed -e "s|github.com/|api.github.com/repos/|g")/tags
    local tag=$2
    local page=1

    echo "Searching for tag ${tag} in repo $1 ..."

    while true; do
        local api_url="${api_base}?per_page=100&page=${page}"
        local tags=$(curl \
          --silent \
          --url "${api_url}" \
          --header "Authorization: token ${GITHUB_TOKEN}" \
          --fail)

        if [ $? -ne 0 ]; then
            error "API request failed for ${api_url}"
        fi

        # Empty array response means we've gone past the last page
        if [ "${tags}" = "[]" ]; then
            error "Tag ${tag} doesn't exist in repo $1"
        fi

        if echo "${tags}" | grep -q "\"${tag}\""; then
            echo "Found."
            return 0
        fi

        page=$((page + 1))
    done
}

# Check if an asset file exists on a tagged release of a GitHub private repository.
# Requires $GITHUB_TOKEN.
# Arguments: repo_url tag file
check_if_private_asset_exist()
{
    local repo=$(echo $1 | sed -e "s|https://github.com|https://api.github.com/repos|g")
    local tag=$2
    local file=$3

    local r=$(curl --silent --header "Authorization: token ${GITHUB_TOKEN}" "${repo}/releases/tags/${tag}")
    eval $(echo "${r}" | grep -C3 "name.:.\\+${file}" | grep -w id | tr : = | tr -cd '[[:alnum:]]=')

    [ "${id}" ]
}

# Download an asset from a tagged release of a GitHub private repository.
# Requires $GITHUB_TOKEN.
# Arguments: repo_url tag file
get_private_asset()
{
    local repo=$(echo $1 | sed -e "s|https://github.com|https://api.github.com/repos|g")
    local tag=$2
    local file=$3

    check_if_private_asset_exist ${repo} ${tag} ${file} || exit 1

    echo "Downloading ${file}..."

    curl --fail --location --progress-bar \
         --header "Authorization: token ${GITHUB_TOKEN}" \
         --header "Accept: application/octet-stream" \
         --output "${file}" \
         "${repo}/releases/assets/${id}"
}

# Check if file exists on a tagged version of a GitHub repository
# Arguments: repo_url tag file
check_if_file_exist()
{
    local repo=$1
    local tag=$2
    local file=$3
    curl --head --silent --fail ${repo}/blob/${tag}/${file} > /dev/null
}

# Exit with an error message
exit_on_error()
{
    echo
    echo "Validation failed! The 'definitions.sh' file is incorrect!"
    echo
    exit 1
}

# Exit with a success message
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

# Validate the config repo
echo "======================================================================================"
echo "Repository names validation"
echo "======================================================================================"

printf "Checking if configuration repository was defined... "
if [ -z "${config_repo}" ]; then
    echo "Failed!"
    echo
    echo "Configuration repository not defined! Please check that 'config_repo' is defined in 'definitions.sh'."
    exit_on_error
fi
echo "${config_repo}"

printf "Checking if FW_SYSTEMS array was defined...         "
if [ ${#FW_SYSTEMS[@]} -eq 0 ]; then
    echo "Failed!"
    echo
    echo "No firmware systems defined! Please add at least one entry to 'FW_SYSTEMS' in 'definitions.sh'."
    exit_on_error
fi
echo "${#FW_SYSTEMS[@]} system(s) defined"

echo "Done! All repository names were defined!"
echo

# Validate each firmware system
for entry in "${FW_SYSTEMS[@]}"; do
    IFS='|' read -r sys_name sys_repo sys_tag sys_fw_files sys_zip <<< "${entry}"
    # Fall back to rogue_<tag>.zip if no explicit zip filename was specified
    sys_zip="${sys_zip:-rogue_${sys_tag}.zip}"

    echo "======================================================================================"
    echo "Firmware validation: ${sys_name}"
    echo "======================================================================================"

    printf "Checking repo is defined...                         "
    if [ -z "${sys_repo}" ]; then
        echo "Failed!"
        echo "Repo URL missing for system '${sys_name}' in FW_SYSTEMS."
        exit_on_error
    fi
    echo "${sys_repo}"

    printf "Checking tag is defined...                          "
    if [ -z "${sys_tag}" ]; then
        echo "Failed!"
        echo "Tag missing for system '${sys_name}' in FW_SYSTEMS."
        exit_on_error
    fi
    echo "${sys_tag}"

    printf "Checking firmware filename(s) are defined...        "
    if [ -z "${sys_fw_files}" ]; then
        echo "Failed!"
        echo "Firmware filename(s) missing for system '${sys_name}' in FW_SYSTEMS."
        exit_on_error
    fi
    echo "${sys_fw_files}"

    check_if_private_tag_exist "${sys_repo}" "${sys_tag}"

    # Validate each firmware file (semicolon-separated)
    IFS=';' read -ra fw_file_list <<< "${sys_fw_files}"
    for fw_file in "${fw_file_list[@]}"; do
        printf "Checking firmware file in assets: %-30s " "${fw_file}..."
        check_if_private_asset_exist "${sys_repo}" "${sys_tag}" "${fw_file}"
        if [ $? != 0 ]; then
            echo "Failed!"
            echo
            echo "File '${fw_file}' does not exist in assets for release '${sys_tag}' of '${sys_repo}'!"
            exit_on_error
        fi
        echo "File exists!"
    done

    # Validate the ZIP: skip asset check if it's a local absolute path
    if [[ "${sys_zip}" = /* ]]; then
        printf "Checking ZIP file exists on disk...                "
        if [ ! -f "${sys_zip}" ]; then
            echo "Failed!"
            echo
            echo "Local ZIP file '${sys_zip}' not found on disk!"
            exit_on_error
        fi
        echo "Found."
    else
        printf "Checking ZIP file is in the list of assets...      "
        check_if_private_asset_exist "${sys_repo}" "${sys_tag}" "${sys_zip}"
        if [ $? != 0 ]; then
            echo "Failed!"
            echo
            echo "File '${sys_zip}' does not exist in assets for release '${sys_tag}' of '${sys_repo}'!"
            exit_on_error
        fi
        echo "File exists!"
    fi

    echo "Done! Firmware '${sys_name}' validated."
    echo
done

# Validate the configuration version
echo "======================================================================================"
echo "Configuration version validation"
echo "======================================================================================"

printf "Checking if the tag version was defined...          "
if [ -z "${config_repo_tag}" ]; then
    echo "Failed!"
    echo
    echo "Configuration tag version not defined! Please check that 'config_repo_tag' is defined in 'definitions.sh'."
    exit_on_error
fi
echo "${config_repo_tag}"

printf "Checking if release exists...                       "
check_if_public_tag_exist "${config_repo}" "${config_repo_tag}"
if [ $? != 0 ]; then
    echo "Failed!"
    echo
    echo "Release '${config_repo_tag}' does not exist in repository '${config_repo}'!"
    exit_on_error
fi
echo "Release exists!"

echo "Done! A correct configuration version was defined."

# All definitions are correct
exit_on_success
