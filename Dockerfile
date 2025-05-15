FROM nvidia/cudagl:11.3.0-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i 's@http://archive.ubuntu.com@http://mirrors.aliyun.com@g' /etc/apt/sources.list && apt update -y && \
    apt-get install -y --no-install-recommends build-essential git \
    wget vim tmux libtool autoconf automake libssl-dev libx11-dev \
    zlib1g-dev libxcursor-dev libxi-dev libxrandr-dev ninja-build \
    lsb-release python3-pip libxinerama-dev iwyu && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir cmake -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    apt-get clean
