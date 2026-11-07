# Docker images

## Description

Two Docker images are built from this repository and published on `ghcr.io/slaclab`:

- **pysmurf-server**: the pysmurf server application and what it needs to run -- pysmurf and the
  `cryodaq` package, the `SmurfProcessor` module and the PCIe offload card's register map -- on the
  [smurf-rogue](https://github.com/slaclab/smurf-rogue-docker) image, plus, under `/tmp/fw`, the
  firmware releases and the register configuration files named in
  [definitions.sh](docker/server/definitions.sh): each platform's pyrogue package archive and FPGA
  image, and the [smurf_cfg](https://github.com/slaclab/smurf_cfg) checkout. To run other firmware,
  mount a directory of your own at `/tmp/fw`.
- **pysmurf-client**: the client application with its plotting and notebook dependencies.

`pysmurf-server-base`, historically the server without firmware and lately a second copy of
`pysmurf-server`, is no longer published; a mounted `/tmp/fw` is what it was for.

## Building the images

When a tag is pushed to this repository, GitHub Actions builds both and pushes them, tagged with
the git tag. The pull-request tests build both images at the commit under test, the server's with
no firmware files, and run the test suites inside them; nothing is pushed.

[build.sh](docker/server/build.sh) does the server build: it validates `definitions.sh`
([validate.sh](docker/server/validate.sh)), downloads the firmware assets into `local_files/`
(private repositories: `GITHUB_TOKEN` in the environment), builds the image at this checkout's
`HEAD` and pushes it tagged `git describe --tags --always` -- the release tag on a release,
`v11.4.4-174-g835b0ffb` on any other commit. Any commit builds; a push to `ghcr.io` needs a token
with `write:packages`. For a local build:

```
PUSH=0 SYSTEMS=umux,rfmc docker/server/build.sh
```

`PUSH=0` builds and tags only; `SYSTEMS` restricts the firmware downloaded to the named entries of
`FW_SYSTEMS`; `REF` builds another tag, branch or commit; `TAG` names the image.

## Running the server

```
docker pull ghcr.io/slaclab/pysmurf-server:<TAG>
docker run -ti --rm \
    -v <local_data_dir>:/data \
    ghcr.io/slaclab/pysmurf-server:<TAG> \
    <arguments>
```

The entrypoint is [start_server.sh](docker/server/scripts/start_server.sh). It passes every
argument to the server's own command line, `python3 -m pysmurf.core.server`, with one translation:
`-c|--comm-type <eth|pcie|emu>` becomes `--transport <eth|pcie|emulation>`, since the server's own
`-c` is `--configure`. With `-g|--gui` it also starts a virtual display when none is available.

The server's command line -- addressing a board directly, or naming its crate slot and letting the
server work out the address, the lane, the bays, the register defaults and the firmware package
through the shelf manager -- is documented with the `cryodaq.server` package in the Sphinx
documentation (*Starting a server*); `python3 -m pysmurf.core.server --help` lists the options.

The firmware is found in `/tmp/fw` (`--firmware-dir`): the one release archive or checkout of the
platform's package, its FPGA image, and the deployment's register configuration files under
`smurf_cfg/defaults/` (`--config-dir`). The image carries them; mount a directory of your own
there, `-v <local_fw_files_dir>:/tmp/fw/`, to run other ones. With several platforms' archives in the directory and no crate record to choose
by, `--platform <name>` says whose.

For example, to serve the carrier in slot 2 of the crate whose shelf manager is `shm-smrf-sp01`
over PCIe:

```
-S shm-smrf-sp01 -N 2 -c pcie <extra_args>
```

which runs `python3 -m pysmurf.core.server -S shm-smrf-sp01 -N 2 --transport pcie <extra_args>`.

## Running the client

```
docker pull ghcr.io/slaclab/pysmurf-client:<TAG>
docker run -ti --rm \
    -v <local_data_dir>:/data \
    -v <local_home_dir>:/home/cryo \
    ghcr.io/slaclab/pysmurf-client:<TAG> \
    -e <epics_prefix> -c <config_file>
```

The entrypoint is [start_client.sh](docker/client/scripts/start_client.sh):

```
usage: start_client.sh [-e|--epics <epics_prefix>] [-c|--config-file <config_file>] [-h|--help]
    -e|--epics <epics_prefix>      : Sets the EPICS PV name prefix (defaults to 'smurf_server').
    -c|--config-file <config_file> : Path to the configuration path
    -h|--help                      : Show this message.
```

It starts an `ipython3` session as the `cryo` user, loading
[pysmurf_startup.py](docker/client/scripts/pysmurf_startup.py), which takes the prefix and the
configuration file given. The client writes data under `/data/smurf_data`, so mount a host
directory holding `smurf_data` at `/data`; mount a host directory at `/home/cryo` to keep the
`ipython3` history.
