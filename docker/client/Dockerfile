FROM tidair/smurf-roguev6:R3.0.2

# RTH: Allowing ubuntu to update generates an error related to the git-lfs repo.

#RUN apt-get update && \
#    DEBIAN_FRONTEND=noninteractive \
RUN apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    python3-cairocffi \
    gir1.2-gtk-3.0 \
    python3-matplotlib \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

#RUN pip3 install \
#    scipy==1.5.2 \
#    pandas==1.1.0 \
#    PyYAML==5.3.1 \
#    pillow==6.2.2 \
#    seaborn==0.11.2 \
#    jupyter==1.0.0 \
#    pytest==6.0.1 \
#    GitPython==3.1.7 \
#    schema==0.7.1

RUN pip3 install \
    scipy \
    pandas \
    PyYAML \
    pillow \
    seaborn \
    jupyter \
    pytest \
    GitPython \
    schema

# Install pysmurf
ARG branch
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/pysmurf.git -b ${branch}
ENV PYTHONPATH /usr/local/src/pysmurf/python:${PYTHONPATH}
ENV PYSMURF_TOP /usr/local/src/pysmurf

RUN mkdir -p /usr/local/src/pysmurf_utilities
ADD scripts/* /usr/local/src/pysmurf_utilities/
ENV PATH /usr/local/src/pysmurf_utilities:${PATH}

# Set GTK3Agg as the matplotlib backend
ENV MPLBACKEND GTK3Agg

# Set jupyter config stuff
COPY jupyter /.jupyter
RUN chown cryo:smurf -R /.jupyter
ENV JUPYTER_CONFIG_DIR /.jupyter

ENTRYPOINT ["start_client.sh"]
