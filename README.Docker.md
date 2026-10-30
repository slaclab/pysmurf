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

The docker image entrypoint is the [start_server.sh](docker/server/scripts/start_server.sh) script. It passes every argument to the server's own command line, `python3 -m pysmurf.core.server`, with one translation: `-c|--comm-type <eth|pcie|emu>` becomes `--transport <eth|pcie|emulation>`, since the server's own `-c` is `--configure`. With `-g|--gui` it also starts a virtual display when none is available.

The server works out from a crate slot what the launcher used to. Name the slot instead of an address:

```
-S|--shelfmanager <host>   The crate's shelf manager. With -N.
-N|--slot <n>              The carrier's slot, 2 to 7. With -S.
--firmware-dir <dir>       Where the firmware release is kept (default /tmp/fw).
-E|--no-hw-detect          Do not read the board population; pass --disable-bay* and --layer yourself.
-D|--no-check-firmware     Do not compare the FPGA's firmware with the image on disk.
--program                  Load the FPGA image on disk even if the FPGA already runs it.
-H|--hard-boot             Reboot the FPGA before starting, then apply the register configuration.
```

Every value worked out from the slot is a default: an option given explicitly (`-a|--addr`, `-l|--pcie-rssi-lane`, `--disable-bay0/1`, `--enable-em22xx`, `-d|--layer`, `-z|--firmware`) is used as given. `python3 -m pysmurf.core.server --help` lists them all.

#### What the slot implies
With `-S` and `-N`, the server reads the crate id from the carrier's management controller through the shelf manager and derives the FPGA's address by SLAC's convention, `10.<crate id high byte>.<crate id low byte>.<100 + slot>` (a carrier with a processor system beside the FPGA has that at `<200 + slot>`); the PCIe RSSI lane is `slot - 2`. It reads the carrier's and each AMC bay's FRU record to decide which bays are populated (`--disable-bay0/1` for an empty one), whether the carrier carries the EM22xx power monitor (`--enable-em22xx`), and which register defaults the population calls for: `defaults_<version>_<bay 0>_<bay 1>.yml` under `<firmware-dir>/smurf_cfg/defaults/`, with a bay's board type (`lb`, `hb`, `tkid`) or `none`. The population's defaults are applied first, so a `-d|--layer` file you pass overrides them. It finds the firmware's Python package in the firmware directory and passes it as `--firmware`, and compares the short git hash the FPGA reports with the one in the FPGA image's file name (`*-<short githash>.mcs[.gz]`), loading the image when they differ.

Everything the server read is published on its status device as the hardware inventory, beside the startup arguments, so a client can see what the launcher saw. A board whose management controller does not answer is tolerated when `-a|--addr` (and `--ps-addr` where there is a processor system) is given and refused, naming the read that failed, when it is not.

#### Addressing the FPGA directly
With `-a|--addr` and no `-S`, nothing is read from the crate: the address, the bays and the register defaults are as given on the command line, and the firmware package is the one in the firmware directory unless `-z|--firmware` names another. `-H|--hard-boot` and `--program` need the shelf manager and are refused without `-S`/`-N`. This is how a bench board with no management controller, or an emulated server (`-c emu --platform <name>`), is started.

#### Hard boot
With `-H|--hard-boot` the FPGA is rebooted through its management controller before the server starts, and the register configuration is applied during start-up (`--configure`). A reboot is skipped when the firmware check has just loaded a new image, since loading one reboots the FPGA.

The firmware files the image carries are under `/tmp/fw/`; mount a directory of your own there to use different ones.

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

which runs `python3 -m pysmurf.core.server -S shm-smrf-sp01 -N 2 --transport pcie <extra_args>`: the server derives the address, lane, bays, defaults and firmware package from the slot, and `extra_args` are passed verbatim.

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