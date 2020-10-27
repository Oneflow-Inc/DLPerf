ARG PYTHON=3.8
ARG pip_args="-i https://pypi.tuna.tsinghua.edu.cn/simple"
FROM nvidia/cuda:10.2-base-ubuntu18.04

WORKDIR /etc/apt/sources.list.d
RUN rm cuda.list nvidia-ml.list
WORKDIR /

RUN apt-get update && \
    apt-get -y install --no-install-recommends openssh-server vim python3 python3-pip wget perl lsb-core google-perftools numactl

ENV MOFED_DIR MLNX_OFED_LINUX-4.3-1.0.1.0-ubuntu18.04-x86_64
ENV IGNOREEOF 3

RUN wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/deps/${MOFED_DIR}.tgz && \
    tar -xzvf ${MOFED_DIR}.tgz && \
    ${MOFED_DIR}/mlnxofedinstall --user-space-only --without-fw-update --all -q && \
    cd .. && \
    rm -rf ${MOFED_DIR} && \
    rm -rf *.tgz

RUN ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa
RUN cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    chmod 600 /root/.ssh/authorized_keys
RUN /etc/init.d/ssh start && \
    ssh-keyscan -H localhost >> /root/.ssh/known_hosts
RUN echo "Host *\n\tStrictHostKeyChecking no" >> /root/.ssh/config && \
    chmod 600 /root/.ssh/config

RUN echo 'ALL ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install $pip_args scikit-learn pynvml
RUN python3 -m pip install --find-links https://oneflow-public.oss-cn-beijing.aliyuncs.com/nightly.pip.index.html oneflow_cu102
