FROM gcr.io/kaggle-gpu-images/python

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set timezone
ENV TZ Asia/Tokyo
RUN echo $TZ > /etc/timezone && rm /etc/localtime && \
    ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && \
        dpkg-reconfigure -f noninteractive tzdata

# Install nncomp
RUN pip3 uninstall -y typing
ADD submodules/nncomp /work/submodules/nncomp
RUN pip3 install -e /work/submodules/nncomp[dev]

# Install original packages
WORKDIR /work
ADD . /work
RUN pip3 install -e .
