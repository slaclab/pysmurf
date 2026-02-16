FROM ghcr.io/slaclab/smurf-rogue:R3.10.5

# Copy all firmware related files, which are in the local_files directory
RUN mkdir -p /tmp/fw/ && chmod -R a+rw /tmp/fw/
COPY local_files /tmp/fw/

# Install the SMURF PCIe card repository
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/smurf-pcie.git -b v2.0.0.1
WORKDIR smurf-pcie
RUN git submodule sync && git submodule update --init --recursive
ENV PYTHONPATH=/usr/local/src/smurf-pcie/software/python:${PYTHONPATH}
ENV PYTHONPATH=/usr/local/src/smurf-pcie/firmware/submodules/axi-pcie-core/python:${PYTHONPATH}

# Install pysmurf
ARG branch
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/pysmurf.git -b ${branch}
# Must add to safe directories for cryo user in order for the core
# pysmurf instance to be able to resolve its version using git.
USER cryo
RUN git config --global --add safe.directory /usr/local/src/pysmurf
USER root
WORKDIR pysmurf
RUN python3 -m pip install .
RUN mkdir build
WORKDIR build
RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make -j4
ENV PYTHONPATH=/usr/local/src/pysmurf/lib:${PYTHONPATH}
ENV SMURF_DIR=/usr/local/src/pysmurf

# Add utilities
RUN mkdir -p /usr/local/src/pysmurf_utilities
ADD scripts/* /usr/local/src/pysmurf_utilities/
ENV PATH=/usr/local/src/pysmurf_utilities:${PATH}

# Set the working directory to the root
WORKDIR /

# Define the entrypoint
ENTRYPOINT ["start_server.sh"]
