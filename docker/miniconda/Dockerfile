ARG CUDA_VERSION=11.1.1
ARG CUDNN_VERSION=8
FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-ubuntu18.04

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Install ubuntu packages
RUN apt-get update -qq && apt-get install -y -qq \
        curl bzip2 software-properties-common pkg-config ca-certificates \
        cmake autoconf automake libtool flex sudo git tzdata openssh-server \
        libglib2.0-0 libxext6 libsm6 libxrender1 libreadline-dev \
        graphviz libgraphviz-dev build-essential python3-pip && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set timezone
ENV TZ Asia/Tokyo
RUN echo $TZ > /etc/timezone && rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
        dpkg-reconfigure -f noninteractive tzdata

# Mkdir for working user
RUN mkdir -m 777 /home/anonymous

# Install miniconda
ENV MINICONDA_VERSION py38_4.9.2
RUN curl -s -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
    bash miniconda.sh -b -p /opt/miniconda && \
    rm miniconda.sh
ENV PATH /opt/miniconda/bin:$PATH

# Create conda environments and install modules
ADD environment.yml /src/environment.yml
RUN conda env update -q -f /src/environment.yml && conda clean --all -y

# Install torch / torchvision
ARG TORCH_VERSION=1.8.1+cu111
ARG TORCHVISION_VERSION=0.9.1+cu111
RUN pip install torch==$TORCH_VERSION torchvision==$TORCHVISION_VERSION -f https://download.pytorch.org/whl/torch_stable.html

# Install JupyterLab
RUN pip install jupyterlab jupyterlab-system-monitor
ENV NODE_OPTIONS --max-old-space-size=4096
RUN jupyter labextension install jupyterlab-plotly plotlywidget

# Install nncomp
WORKDIR /work
ADD ./pkgs /work/pkgs
RUN pip install /work/pkgs/nncomp-21.4-py3-none-any.whl[dev,cv]

# Install competition-specific packages
ADD . /work
ADD .config /home/anonymous/.config
RUN chmod -R 777 /home/anonymous/.config
RUN pip install -U git+https://github.com/rwightman/pytorch-image-models.git@7077f16c6a72960dd5a2ca042ed79c8e72eeb765
RUN pip install -e /work
