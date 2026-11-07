FROM ghcr.io/slaclab/smurf-rogue:R5.1.1

RUN apt-get update
RUN apt-get install -y \
    python3-gi \
    python3-gi-cairo \
    python3-cairocffi \
    gir1.2-gtk-3.0 \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

RUN pip3 install --no-cache-dir \
    Cython \
    scipy \
    pandas \
    PyYAML \
    pillow \
    seaborn \
    jupyterlab \
    notebook \
    ipykernel \
    ipywidgets \
    ipympl \
    jupyter-resource-usage \
    plotly \
    tqdm \
    pytest \
    GitPython \
    schema \
    setuptools_scm

# Install pysmurf at `ref`: a tag, a branch or a commit (as the server image).
ARG ref
RUN test -n "${ref}" || { echo "build-arg ref is required: the pysmurf tag, branch or commit to build" >&2; exit 1; }
WORKDIR /usr/local/src
RUN git clone https://github.com/slaclab/pysmurf.git \
 && cd pysmurf \
 && { { git fetch --quiet origin "${ref}" 2>/dev/null && git checkout --quiet --detach FETCH_HEAD; } \
      || git checkout --quiet --detach "${ref}"; }
# Must add to safe directories for cryo user in order for the core
# pysmurf instance to be able to resolve its version using git.
USER cryo
RUN git config --global --add safe.directory /usr/local/src/pysmurf
USER root
WORKDIR /usr/local/src/pysmurf
RUN pip3 install -e .
ENV PYSMURF_TOP=/usr/local/src/pysmurf

RUN mkdir -p /usr/local/src/pysmurf_utilities
ADD scripts/* /usr/local/src/pysmurf_utilities/
ENV PATH=/usr/local/src/pysmurf_utilities:${PATH}

# Set GTK3Agg as the matplotlib backend
ENV MPLBACKEND=GTK3Agg

# Set jupyter config stuff
COPY jupyter /.jupyter
RUN chown cryo:smurf -R /.jupyter
ENV JUPYTER_CONFIG_DIR=/.jupyter

ENTRYPOINT ["start_client.sh"]
