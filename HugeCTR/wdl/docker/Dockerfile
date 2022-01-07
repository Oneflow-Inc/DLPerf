FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04-rc
ARG CMAKE_BUILD_TYPE=Release
ARG SM="70;80"
ARG VAL_MODE=OFF
ARG ENABLE_MULTINODES=ON
ARG NCCL_A2A=ON

RUN apt-get update && \
    apt-get upgrade -y

RUN apt-get install -y \
    git \
    cmake \
    vim \
    wget \
    openssh-server \
    python3-pip

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    numpy \
    pandas \
    sklearn \
    ortools \
    pynvml \
    tensorflow

RUN cp /usr/local/cuda/lib64/libnccl*  /usr/lib/x86_64-linux-gnu/ && \
    cp /usr/local/cuda-11.0/targets/x86_64-linux/include/nccl*.h  /usr/include

RUN echo 'export PS1="\s \w\$ "' >>/etc/bash.bashrc

RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz && \
    tar -zxvf openmpi-4.0.5.tar.gz && \
    cd openmpi-4.0.5 && \
    ./configure --prefix=/usr/local && \
    make all install && \
    cd .. && rm -rf openmpi-4.0.5 openmpi-4.0.5.tar.gz

RUN wget https://github.com/openucx/ucx/releases/download/v1.8.1/ucx-1.8.1.tar.gz && \
    tar -zxvf ucx-1.8.1.tar.gz && \
    cd ucx-1.8.1 && \
    ./contrib/configure-release --prefix=/usr/local --disable-numa && \
    make -j8 install && \
    cd .. && rm -rf ucx-1.8.1.tar.gz ucx-1.8.1

RUN wget https://download.open-mpi.org/release/hwloc/v2.2/hwloc-2.2.0.tar.gz && \
    tar -zxvf hwloc-2.2.0.tar.gz && \
    cd hwloc-2.2.0 && \
    ./configure --prefix=/usr/local && \
    make -j install && \
    cd ..  && rm -rf hwloc-2.2.0.tar.gz hwloc-2.2.0

RUN git clone https://github.com/NVIDIA/HugeCTR.git &&\
    cd HugeCTR && \
    git submodule update --init --recursive && \
    mkdir build && cd build &&\
    cmake -DCMAKE_BUILD_TYPE=$CMAKE_BUILD_TYPE -DSM=$SM \
    -DVAL_MODE=$VAL_MODE -DENABLE_MULTINODES=$ENABLE_MULTINODES -DNCCL_A2A=$NCCL_A2A .. && \
    make -j &&\
    mkdir /usr/local/hugectr &&\
    make install &&\
    chmod +x /usr/local/hugectr/bin/* &&\
    rm -rf HugeCTR

ENV PATH /usr/local/hugectr/bin:$PATH

ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:/usr/local/lib

RUN pip3 install mpi4py

