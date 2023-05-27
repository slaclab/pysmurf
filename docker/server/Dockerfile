FROM tidair/smurf-rogue:R2.9.1

# Copy all firmware related files, which are in the local_files directory
RUN mkdir -p /tmp/fw/ && chmod -R a+rw /tmp/fw/
COPY local_files /tmp/fw/

# Install the SMURF PCIe card repository
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/smurf-pcie.git -b v2.0.0
WORKDIR smurf-pcie
RUN sed -i -e 's|git@github.com:|https://github.com/|g' .gitmodules
RUN git submodule sync && git submodule update --init --recursive
ENV PYTHONPATH /usr/local/src/smurf-pcie/software/python:${PYTHONPATH}
ENV PYTHONPATH /usr/local/src/smurf-pcie/firmware/submodules/axi-pcie-core/python:${PYTHONPATH}

# Install pysmurf
ARG branch
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/pysmurf.git -b ${branch}
WORKDIR pysmurf
RUN mkdir build
WORKDIR build
RUN cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo .. && make -j4
ENV PYTHONPATH /usr/local/src/pysmurf/lib:${PYTHONPATH}
ENV PYTHONPATH /usr/local/src/pysmurf/python:${PYTHONPATH}
ENV SMURF_DIR /usr/local/src/pysmurf

# Add utilities
RUN mkdir -p /usr/local/src/pysmurf_utilities
ADD scripts/* /usr/local/src/pysmurf_utilities/
ENV PATH /usr/local/src/pysmurf_utilities:${PATH}

# Set the working directory to the root
WORKDIR /

# Define the entrypoint
ENTRYPOINT ["start_server.sh"]
