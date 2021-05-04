# Docker images

## Description

Docker images for both the server and client applications are built automatically from this repository.

For the server application, two docker images are generated:
- An **stable** version called **pysmurf-server**. This image contains, besides the pysmurf server application, the firmware and its configuration files in it. The versions of these files are defined in the [definition.sh file](docker/server/definitions.sh).
- A **base** version called **pysmurf-server-base**. Historically, this image contained only the pysmurf server application (without any firmware-related file) and was used for testing development firmware images. However, now this image is exactly the same **estable** image described above; as you can mount an external directory with development firmware files on top of the directory contained in the stable docker image when starting the container, there is not really necessary to maintained two different images. Although this image is not really necessary, is still being published to maintain backward compatibility with existing scripts; although you should use the stable image instead as this images will be deprecated in the future.

On the other hand,
The docker image for the server is called **pysmurf-server-base**, while the docker image for the client is called **pysmurf-client**.

## Building the image

When a tag is pushed to this github repository, three new Docker images, two for the server and one for the client applications, are automatically built and push to their Dockerhub repositories using GitHub Actions. The three Dockerhub repositories are:
- Stable server repository: https://hub.docker.com/r/tidair/pysmurf-server
- Base server repository: https://hub.docker.com/r/tidair/pysmurf-server-base, and
- Client repository: https://hub.docker.com/r/tidair/pysmurf-client

The resulting docker images are tagged with the same git tag string (as returned by `git describe --tags --always`).

The files used to generate the docker images are located in the [docker/server] directory for the server docker image, and in the [docker/client] directory for the client docker image.

## How to get the container

To get the docker image, first you will need to install the docker engine in you host OS. Then you can pull a copy by running:

```
docker pull tidair/pysmurf-server:<TAG>
```

for the server, and
```
docker pull tidair/pysmurf-client:<TAG>
```

for the client.

Here, **TAG** represents the specific tagged version you want to use.

## Running the container

### Server application

The docker image entrypoint is the [start_server.sh](docker/server/scripts/start_server.sh) script. That script receives the following arguments:

```
usage: start_server.sh [-S|--shelfmanager <shelfmanager_name> -N|--slot <slot_number>]
                       [-a|--addr <FPGA_IP>] [-D|--no-check-fw] <pyrogue_server-args>
    -S|--shelfmanager <shelfmanager_name> : ATCA shelfmanager node name or IP address. Must be used with -N.
    -N|--slot         <slot_number>       : ATCA crate slot number. Must be used with -S.
    -a|--addr         <FPGA_IP>           : FPGA IP address. If defined, -S and -N are ignored.
    -c|--comm-type    <comm_type>         : Communication type ('eth' or 'pcie'). Default is 'eth'.
    -D|--no-check-fw                      : Disabled FPGA version checking.
    -E|--disable-hw-detect                : Disable hardware type auto detection.
    -H|--hard-boot                        : Do a hard boot: reboot the FPGA and load default configuration.
    -h|--help                             : Show this message.
    <pyrogue_server_args> are passed to the SMuRF pyrogue server.
```

The `start_server.sh` script is in change of:

#### FPGA IP address calculation
You can address the target FPGA either using the card's ATCA shelfmanager's name and slot number (using arguments `-S` and `-N`), or by directly giving its IP address (using argument `-a`). If you use the shelfmanager name and slot number, the script will automatically detect the FPGA's IP address by reading the crate's ID and using the slot number, following the SLAC's convention: `IP address = 10.<crate's ID higher byte>.<crate's ID lower byte>.<100 + slot number>`. On the other hand, if you use the IP address, then the shelfmanager name and slot number arguments are ignored. The final IP address, either passed by the user or auto-detected by the script, is passed to the next startup script using the `-a` argument.

#### Pyrogue ZIP file detection
The script looks a pyrogue `zip` file to be present in `/tmp/fw/`. If found, the location of that file will be passed to the next startup script using the argument `-z`. If no zip file is found, the script will then look for a local checked out repository in the same location; if found, the python directories under it will be added to the PYTHONPATH environmental variable. The python directories must match these patterns:
```
/tmp/fw/*/firmware/python/
/tmp/fw/*/firmware/submodules/*/python/
```

The docker images contains the `zip` file file in the `/tmp/fw/` directory. However, you can use a different version by mounting a local copy from the host CPU inside the container as `/tmp/fw/`, which will replace the internal files.

#### Firmware version checking
On the other hand, the scripts also looks for a MCS file in `/tmp/fw/`. The file name must include the short githash version of the firmware and an extension `mcs` or `mcs.gz`, following this expression: `*-<short-githash>.mcs[.gz]`. The script will also read the firmware short githash from the specified FPGA. If the version from the MCS file and the FPGA don't match, then the script will automatically load the MCS file into the FPGA. This automatic version checking can be disabled either by passing the argument `-D|--no-check-fw`, or by addressing the FPGA by IP address instead of ATCA's shelfmanager_name/slot_number.

The docker images contains the `mcs.gz` files in the `/tmp/fw/` directory. However, you can use a different version of these files by mounting a local copy from the host CPU inside the container as `/tmp/fw/`, which will replace the internal files.

#### Hardware type auto-detection
The script will try to auto-detect the type of carrier and AMC boards, and automatically generate server startup arguments based on the hardware type.

##### Carrier board
The script will determine the type and version of the carrier board using the `cba_fru_init` utility, and looking at the `Board Part Number` filed. These field must follow this convention: `<part_number>_C<version>`:
- The supported part number are: `PC_379_396_01` (Gen1), and `PC_379_396_38` (Gen2).
- The version is a 2-digit number.
If the carrier is a Gen2, version >= C03, then the option `--enable-em22xx` will be added to the list of startup arguments.

##### AMC daughter board
The script will determine the presence, type and version of the carrier board using the `cba_amc_init` utility, and looking at the `Board Part Number` filed. These field must follow this convention: `<part_number>_C<version>_<type>`:
- The supported part number is: `PC_379_396_30`.
- The version is a 2-digit number.
- The type is either `A01` (for low band boards) or `A02` (for high band boards)
If a board is not present in one of the bays, then the corresponding `--disable-bay[0,1]` flag will be added the list of start arguments. On the other hands, depending on the presence, type, and version of the AMC daughter boards, the corresponding defaults file will be added to the list of arguments as well; the defaults file name must follow the naming convention `defaults_<version>_<bay0_type>_<bay1_type>.yml`, where:
- **version**: is the version of the AMC daughter boards. If both boards are present, they must be of the same version.
- **bay[0,1]_type**: if the type of the AMC daughter board in each bay. Supported types are `lb` (for low band boards) and `hb` (for high band boards). In a bay is not populated, then this filed must be set to `none`.

If the script fails to detect the type of boards, then the startup process is aborted. The reason for the detection to failed are:
- A board with an unsupported part number was found.
- The board type or versions can not be extracted from the part numbers.
- The AMC daughter boards are of different versions.
- The defaults file was not found.

This hardware auto-detection feature can be disabled using the option `-E|--disable-hw-detect`, or by addressing the FPGA by IP address instead of ATCA's shelfmanager_name/slot_number. The user should provided the appropriate startup arguments based on the hardware type.

#### Hard boot
The option `-H|--hard-boot` can be used to request a hard boot. During this boot mode, the FPGA is rebooted by deactivating and activating the carrier board before starting the pyrogue server, and the default configuration file is loaded during the pyrogue server booting process.


All other arguments are passed verbatim to the next startup script.

Depending on the communication type, one the following startup scripts are called, which are located under the [server_scripts](server_scripts) folder:
- if `eth` is used, then [cmb_eth.py](server_scripts/cmb_eth.py) is called,
- if `pcie` is used, then [cmb_pcie.py](server_scripts/cmb_pcie.py) is called.

The usage of each one of these subsequent startup scripts is described here:

#### Startup script using ETH communication

The startup script when using PCIe communication, [cmb_eth.py](server_scripts/cmb_eth.py), receives the following arguments:

```
Usage: cmb_eth.py
        [-z|--zip file] [-a|--addr IP_address] [-g|--gui]
        [-n|--nopoll] [-l|--pcie-rssi-lane index]
        [-f|--stream-type data_type] [-b|--stream-size byte_size]
        [-d|--defaults config_file] [-u|--dump-pvs file_name] [--disable-gc]
        [--disable-bay0] [--disable-bay1] [-w|--windows-title title]
        [--pcie-dev-rssi pice_device] [--pcie-dev-data pice_device] [-h|--help]

    -h|--help                   : Show this message
    -z|--zip file               : Pyrogue zip file to be included in the python path.
    -a|--addr IP_address        : FPGA IP address. Required when the communication type is based on Ethernet.
    -d|--defaults config_file   : Default configuration file. If the path is relative, it refers to the zip file (i.e: file.zip/config/config_file.yml).
    -e|--epics prefix           : Start an EPICS server with PV name prefix "prefix"
    -g|--gui                    : Start the server with a GUI.
    -n|--nopoll                 : Disable all polling
    -l|--pcie-rssi-lane index   : PCIe RSSI lane (only needed withPCIe). Supported values are 0 to 5
    -b|--stream-size data_size  : Expose the stream data as EPICS PVs. Only the first "data_size" points will be exposed. Default is 2^19. (Must be used with -e)
    -f|--stream-type data_type  : Stream data type (UInt16, Int16, UInt32 or Int32). Default is Int16. (Must be used with -e and -b)
    -u|--dump-pvs file_name     : Dump the PV list to "file_name". (Must be used with -e)
    --disable-bay0              : Disable the instantiation of the devices for Bay0
    --disable-bay1              : Disable the instantiation of the devices for Bay1
    --disable-gc                : Disable python's garbage collection(enabled by default)
    -w|--windows-title title    : Set the GUI windows title. If not specified, the default windows title will be the name of this script.This value will be ignored when running in server mode.
    --pcie-dev-rssi pice_device : Set the PCIe card device name used for RSSI (defaults to '/dev/datadev_0')
    --pcie-dev-data pice_device : Set the PCIe card device name used for data (defaults to '/dev/datadev_1')
```

#### Startup script using PCIe communication

The startup script when using PCIe communication, [cmb_pcie.py](server_scripts/cmb_pcie.py), receives the following arguments:

```
Usage: cmb_pcie.py
        [-z|--zip file] [-a|--addr IP_address] [-g|--gui]
        [-n|--nopoll] [-l|--pcie-rssi-lane index]
        [-f|--stream-type data_type] [-b|--stream-size byte_size]
        [-d|--defaults config_file] [-u|--dump-pvs file_name] [--disable-gc]
        [--disable-bay0] [--disable-bay1] [-w|--windows-title title]
        [--pcie-dev-rssi pice_device] [--pcie-dev-data pice_device] [-h|--help]

    -h|--help                   : Show this message
    -z|--zip file               : Pyrogue zip file to be included in the python path.
    -a|--addr IP_address        : FPGA IP address. Required when the communication type is based on Ethernet.
    -d|--defaults config_file   : Default configuration file. If the path is relative, it refers to the zip file (i.e: file.zip/config/config_file.yml).
    -e|--epics prefix           : Start an EPICS server with PV name prefix "prefix"
    -g|--gui                    : Start the server with a GUI.
    -n|--nopoll                 : Disable all polling
    -l|--pcie-rssi-lane index   : PCIe RSSI lane (only needed withPCIe). Supported values are 0 to 5
    -b|--stream-size data_size  : Expose the stream data as EPICS PVs. Only the first "data_size" points will be exposed. Default is 2^19. (Must be used with -e)
    -f|--stream-type data_type  : Stream data type (UInt16, Int16, UInt32 or Int32). Default is Int16. (Must be used with -e and -b)
    -u|--dump-pvs file_name     : Dump the PV list to "file_name". (Must be used with -e)
    --disable-bay0              : Disable the instantiation of the devices for Bay0
    --disable-bay1              : Disable the instantiation of the devices for Bay1
    --disable-gc                : Disable python's garbage collection(enabled by default)
    -w|--windows-title title    : Set the GUI windows title. If not specified, the default windows title will be the name of this script.This value will be ignored when running in server mode.
    --pcie-dev-rssi pice_device : Set the PCIe card device name used for RSSI (defaults to '/dev/datadev_0')
    --pcie-dev-data pice_device : Set the PCIe card device name used for data (defaults to '/dev/datadev_1')
```

With all that in mind, the command to run the container looks something like this:

```
docker run -ti --rm \
    -v <local_data_dir>:/data \
    tidair/pysmurf-server:<TAG> \
    <arguments>
```

Where:
- **local_data_dir**: is a local directory in the host CPU where the persistent data can be written to if needed,
- **TAG**: is the tagged version of the container your want to run,
- **arguments**: are the arguments passed to start_server.sh.

Optionally, you can add an argument `-v <local_fw_files_dir>:/tmp/fw/` to the command above, where **local_fw_files_dir** is a local directory in the host CPU with the firmware's MCS and pyrogue zip files, if you want to use a different version respect to the ones includes in the docker image.

For example, to start the server using PCIe communication, on the carrier card located in slot 2 on a crate with shelfmanager node name `shm-smrf-sp01`, you will use the following arguments:

```
-S shm-smrf-sp01 -N 2 -c pcie <extra_args>
```

At this point, `cmb_pcie.py` is called passing the arguments:
- `-a <ip_address>`,
- if a zip file is located under `/tmp/fw` the the argument `-z /tmp/fw/file_name.zip` is passed, and
- `extra_args` are passed verbatim.

### Client application

The docker image entrypoint is the [start_client.sh](docker/client/scripts/start_client.sh) script. That script receives the following arguments:

```
usage: styart_client.sh [-e|--epics <epics_prefix>] [-c|--config-file <config_file>] [-h|--help]
    -e|--epics <epics_prefix>      : Sets the EPICS PV name prefix (defaults to 'smurf_server').
    -c|--config-file <config_file> : Path to the configuration path
    -h|--help                      : Show this message.
```

The script will then start an `ipython3` session loading the [pysmurf_startup.py](docker/client/scripts/pysmurf_startup.py) script. This script will capture the `epics_prefix` and `config_file` definition passed to the startup script, if any.

The client needs a location to write data. That locations inside the container is `/data/smurf_data`. So, you need to have those directories in the host CPU and mount then inside the container.

On the other hand, the client application runs in a `ipyhton3` session as the `cryo` user. If you want to keep the `ipython3` history you need to mount a local directory as `/home/cryo/` inside the container.

With all that in mind, the command to run the container looks something like this:

```
docker run -ti --rm \
    -v <local_data_dir>:/data \
    -v <local_home_dir>:/home/cryo \
    tidair/pysmurf:<TAG> \
    -e <epics_prefix> -c <config_file>
```

Where:
- **local_data_dir**: is a local directory in the host CPU which contains the directory `smurf_data` where the data is going to be written to,
- **local_home_dir**: is the local directory in the host CPU which will hold the cryo user home directory inside the docker, in order to maintain the ipython history,
- **TAG**: is the tagged version of the container your want to run.
