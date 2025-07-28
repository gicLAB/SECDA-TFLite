# ==============================================================================
FROM ubuntu:22.04 AS builder

# Development Tools
RUN apt update && apt install -y curl wget tree less git vim tmux htop parallel unzip rsync sudo jq

# Compilers
RUN apt install -y clang-14 lld-14 clang-format-14 && \
    ln -s /usr/bin/clang-14 /usr/bin/clang && \
    ln -s /usr/bin/clang++-14 /usr/bin/clang++  && \
    ln -s //usr/bin/clang-format-14 /usr/bin/clang-format

RUN apt install -y crossbuild-essential-armhf crossbuild-essential-arm64 libz-dev
RUN apt install -y gdb
RUN apt install -y software-properties-common

# Install Bazel
RUN apt install apt-transport-https curl gnupg -y && \
    curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
    mv bazel-archive-keyring.gpg /usr/share/keyrings && \
    echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt update && apt install bazel-6.1.0 -y && \
    ln -sf /usr/bin/bazel-6.1.0 /usr/bin/bazel6


# Install CMake
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh && \
    chmod +x cmake-3.18.2-Linux-x86_64.sh && \
    ./cmake-3.18.2-Linux-x86_64.sh --skip-license --prefix=/usr/local

# RUN wget https://github.com/ninja-build/ninja/releases/download/v1.10.1/ninja-linux.zip && \
#     unzip ninja-linux.zip && \
#     mv ninja /usr/local/bin/ninja && \
#     ln -s /usr/local/bin/ninja /usr/sbin/ninja

# Install SystemC
# RUN mkdir -p /tmp/systemc /opt/systemc/systemc-2.3.3 /tmp/systemc/systemc-2.3.3/build
# WORKDIR /tmp/systemc
# RUN wget https://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz && \
#     tar -xvf systemc-2.3.3.tar.gz
# WORKDIR /tmp/systemc/systemc-2.3.3/build
# RUN ../configure --prefix=/opt/systemc/systemc-2.3.3/ && \
#     make -j$(nproc) && \
#     make install



# ============================================================================
FROM builder AS dev-user

COPY .devcontainer/requirements.txt /working_dir/requirements.txt
WORKDIR /working_dir

ENV PATH="/root/miniconda3/bin:${PATH}"

# Install Conda
RUN apt-get update && \
    apt-get install -y wget && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda init
RUN conda create -y -n secda_tflitev2 python=3.9.19
RUN conda run -n secda_tflitev2 pip install -r /working_dir/requirements.txt
RUN conda init


RUN echo "PS1='\[\033[01;31m\][\[\033[01;30m\]\u@\h\[\033[01;36m\] \W\[\033[01;31m\]]\$\[\033[00m\] '" >> ~/.bashrc

# Select clang compiler and set SystemC home
RUN echo "export CXX=/usr/bin/clang++" >> ~/.bashrc && \
    echo "export CC=/usr/bin/clang" >> ~/.bashrc && \
    echo "export SYSTEMC_HOME=/opt/systemc/systemc-2.3.3" >> ~/.bashrc && \
    echo "export CONTAINERIZED=1" >> ~/.bashrc


# Print welcome message
# RUN echo "cd /working_dir" >> ~/.bashrc && \
#     echo "echo 'Welcome to the SECDA-TFLITE dev container'" >> ~/.bashrc && \
#     echo "echo ' '" >> ~/.bashrc && \
#     echo "echo 'NOTE:'" >> ~/.bashrc && \
#     echo "echo '    /working_dir folder was volume mounted'" >> ~/.bashrc && \
#     echo "echo ' '" >> ~/.bashrc 

RUN echo "echo 'Welcome to the SECDA-TFLITE dev container'" >> ~/.bashrc && \
    echo "echo ' '" >> ~/.bashrc && \
    echo "echo 'NOTE: Important folders are volume mounted to /working_dir '" >> ~/.bashrc 

RUN echo "conda activate secda_tflitev2" >> ~/.bashrc