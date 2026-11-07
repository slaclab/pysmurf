# The pysmurf server image: the rogue image plus pysmurf and what a server
# needs to run -- the cryodaq package, the SmurfProcessor module, the offload
# card's register map -- and, under /tmp/fw, the firmware releases and the
# register configuration files from local_files/, which build.sh fills from
# definitions.sh. pysmurf is built at build-arg `ref` (a tag, branch or commit).
#
# /tmp/fw is copied last, so a firmware bump rebuilds one layer; a container
# that mounts its own directory at /tmp/fw runs other firmware than the image's.

FROM ghcr.io/slaclab/smurf-rogue:R5.1.1

# Install Xvfb for virtual display support when running with GUI (-g)
RUN apt-get update && apt-get install -y --no-install-recommends \
    xvfb \
    x11-utils \
    curl \
 && rm -rf /var/lib/apt/lists/*

# The PCIe offload card's register map is the smurf-pcie release's Python
# package, taken from the release archive and imported from it; the archive
# stays outside /tmp/fw so the server never takes it for a carrier's package.
# The release is pinned in definitions.sh and passed in by build.sh and CI.
ARG smurf_pcie_tag
RUN test -n "${smurf_pcie_tag}" || { echo "build-arg smurf_pcie_tag is required (see definitions.sh)" >&2; exit 1; }
RUN mkdir -p /usr/local/src/smurf-pcie \
 && curl --fail --silent --show-error --location \
      --output /usr/local/src/smurf-pcie/rogue_${smurf_pcie_tag}.zip \
      https://github.com/slaclab/smurf-pcie/releases/download/${smurf_pcie_tag}/rogue_${smurf_pcie_tag}.zip \
 && chmod a+r /usr/local/src/smurf-pcie/rogue_${smurf_pcie_tag}.zip
ENV PYTHONPATH=/usr/local/src/smurf-pcie/rogue_${smurf_pcie_tag}.zip/python:${PYTHONPATH}

# Upgrade pip - need this right now so can do an editable install of
#  pysmurf, which fixes Issue 848.  In principle, this would be better
#  to do in the base docker, but we're going to retire the base docker
#  soon.
RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    setuptools_scm

# Install pysmurf at `ref`: a tag, a branch or a commit. The clone is a full
# one so the version resolves from the tags; a commit the clone does not
# reach (a pull request's) is fetched by name first.
ARG ref
RUN test -n "${ref}" || { echo "build-arg ref is required: the pysmurf tag, branch or commit to build" >&2; exit 1; }
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/pysmurf.git \
 && cd pysmurf \
 && (git fetch --quiet origin "${ref}" 2>/dev/null || true) \
 && git checkout --quiet --detach "${ref}"
# Must add to safe directories for cryo user in order for the core
# pysmurf instance to be able to resolve its version using git.
USER cryo
RUN git config --global --add safe.directory /usr/local/src/pysmurf
USER root
WORKDIR pysmurf
RUN pip3 install -e ./
RUN mkdir build
WORKDIR build
RUN cmake -DCMAKE_PREFIX_PATH=/usr/local/lib -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make -j4
ENV PYTHONPATH=/usr/local/src/pysmurf/lib:${PYTHONPATH}
ENV SMURF_DIR=/usr/local/src/pysmurf

# Add utilities
RUN mkdir -p /usr/local/src/pysmurf_utilities
ADD scripts/* /usr/local/src/pysmurf_utilities/
ENV PATH=/usr/local/src/pysmurf_utilities:${PATH}

# The firmware releases and the configuration files, last: each release's
# package archive and FPGA image, and the smurf_cfg checkout (local_files/,
# written by build.sh from definitions.sh).
RUN mkdir -p /tmp/fw/ && chmod -R a+rw /tmp/fw/
COPY local_files /tmp/fw/

# Set the working directory to the root
WORKDIR /

# Define the entrypoint
ENTRYPOINT ["start_server.sh"]
