# ==============================================================================
FROM ubuntu:22.04 as builder

# Development Tools
RUN apt update && apt install -y curl wget tree less git vim tmux htop parallel unzip

# Compilers
RUN apt update && apt install -y clang-14 lld-14 clang-format-14 && \
  ln -s /usr/bin/clang-14 /usr/bin/clang && \
  ln -s /usr/bin/clang++-14 /usr/bin/clang++  && \
  ln -s //usr/bin/clang-format-14 /usr/bin/clang-format

RUN apt update && apt install -y crossbuild-essential-armhf libz-dev
RUN apt update && apt install -y crossbuild-essential-arm64
RUN apt update &&  apt-get install gdb -y

# Install Bazel
RUN apt install apt-transport-https curl gnupg -y && \
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg && \
mv bazel-archive-keyring.gpg /usr/share/keyrings && \
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list

RUN apt update && apt install bazel-3.7.2 -y && \
    ln -s /usr/bin/bazel-3.7.2 /usr/bin/bazel


# Install CMake
WORKDIR /tmp
RUN wget https://github.com/Kitware/CMake/releases/download/v3.18.2/cmake-3.18.2-Linux-x86_64.sh && \
    chmod +x cmake-3.18.2-Linux-x86_64.sh && \
    ./cmake-3.18.2-Linux-x86_64.sh --skip-license --prefix=/usr/local

RUN wget https://github.com/ninja-build/ninja/releases/download/v1.10.1/ninja-linux.zip && \
    unzip ninja-linux.zip && \
    mv ninja /usr/local/bin/ninja && \
    ln -s /usr/local/bin/ninja /usr/sbin/ninja

# Install SystemC
RUN mkdir -p /tmp/systemc /opt/systemc/systemc-2.3.3 /tmp/systemc/systemc-2.3.3/build
WORKDIR /tmp/systemc
RUN wget https://accellera.org/images/downloads/standards/systemc/systemc-2.3.3.tar.gz && \
    tar -xvf systemc-2.3.3.tar.gz
WORKDIR /tmp/systemc/systemc-2.3.3/build
RUN ../configure --prefix=/opt/systemc/systemc-2.3.3/ && \
    make -j$(nproc) && \
    make install

RUN echo "PS1='\[\033[01;31m\][\[\033[01;30m\]\u@\h\[\033[01;36m\] \W\[\033[01;31m\]]\$\[\033[00m\] '" >> .bashrc

RUN echo "export CXX=/usr/bin/clang++" >> ~/.bashrc && \
    echo "export CC=/usr/bin/clang" >> ~/.bashrc && \
    echo "export SYSTEMC_HOME=/opt/systemc/systemc-2.3.3" >> ~/.bashrc

ENV SYSTEMC_HOME=/opt/systemc/systemc-2.3.3
ENV CXX=/usr/bin/clang++ 
ENV CC=/usr/bin/clang 

# ============================================================================
FROM builder AS dev-user
# Add dev user with matching UID of the user who build the image
ARG USER_ID
ARG GROUP_ID
RUN useradd -m --uid $USER_ID developer && \
    echo "developer:devpasswd" | chpasswd && \
    usermod -aG dialout developer && \
    usermod -aG sudo developer

USER developer
WORKDIR /home/developer

RUN echo "PS1='\[\033[01;31m\][\[\033[01;30m\]\u@\h\[\033[01;36m\] \W\[\033[01;31m\]]\$\[\033[00m\] '" >> .bashrc

# Select clang compiler and set SystemC home
RUN echo "export CXX=/usr/bin/clang++" >> ~/.bashrc && \
    echo "export CC=/usr/bin/clang" >> ~/.bashrc && \
    echo "export SYSTEMC_HOME=/opt/systemc/systemc-2.3.3" >> ~/.bashrc

# Print welcome message
WORKDIR /home/developer
RUN echo "echo 'Welcome to the SECDA-TFLITE dev container'" >> ~/.bashrc && \
    echo "echo ' '" >> ~/.bashrc && \
    echo "echo 'NOTE:'" >> ~/.bashrc && \
    echo "echo '    /working_dir folder was volume mounted'" >> ~/.bashrc && \
    echo "echo ' '" >> ~/.bashrc 
