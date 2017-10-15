FROM python:3.6

MAINTAINER Sara Collins <saracollins0508@gmail.com>

# TensorFlow Dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        git \
        pkg-config \
        python3-dev \
        rsync \
        software-properties-common \
        unzip \
        libgtk2.0-0 \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Anaconda
ADD https://repo.continuum.io/miniconda/Miniconda3-4.2.12-Linux-x86_64.sh tmp/Miniconda3-4.2.12-Linux-x86_64.sh
RUN bash tmp/Miniconda3-4.2.12-Linux-x86_64.sh -b
ENV PATH $PATH:/root/miniconda3/bin/
COPY lt_environment.yml .
RUN conda env create -f lt_environment.yml
RUN conda clean -tp -y

# Set working directory
RUN cd / && \
    git clone https://github.com/scollins83/language_translator.git && \
    cd language_translator && \
    git checkout dev && \
    cd / && \
    mkdir /data && \
    cd /data && \
    wget -O training-giga-fren.tar http://www.statmt.org/wmt10/training-giga-fren.tar && \
    tar -xvf training-giga-fren.tar

WORKDIR "/language_translator"

# Expose ports
# TensorBoard
EXPOSE 6006
# Flask Server
EXPOSE 4567

COPY run.sh /
RUN chmod +x /run.sh
ENTRYPOINT ["/run.sh"]
