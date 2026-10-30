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

The server's command line -- addressing a board directly or naming its crate slot and letting the server work out the address, the lane, the bays, the register defaults and the firmware package through the shelf manager -- is documented with the `cryodaq.server` package in the Sphinx documentation (*Starting a server*); `python3 -m pysmurf.core.server --help` lists the options.

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