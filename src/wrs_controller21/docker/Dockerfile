FROM devrt/ros-devcontainer-vscode:noetic-desktop

ARG CUDA_MAJOR=11
ARG CUDA_MINOR=0
ARG CUDNN_MAJOR=8
ARG CUDNN_MINOR=0.5.39
ARG PYTHON_VERSION=3.6.9
ARG PYENV_NAME=catkin_py3

USER root

RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin \
        > /etc/apt/preferences.d/cuda-repository-pin-600 \
    && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub \
        | gpg --no-default-keyring --keyring /usr/share/keyrings/nvidia-cuda.gpg --import \
    && curl -fsSL https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub \
        | gpg --no-default-keyring --keyring /usr/share/keyrings/nvidia-ml.gpg --import \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-cuda.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" \
        > /etc/apt/sources.list.d/cuda.list \
    && echo "deb [signed-by=/usr/share/keyrings/nvidia-ml.gpg] https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" \
        > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VER_NAME ${CUDNN_MAJOR}.${CUDNN_MINOR}-1+cuda${CUDA_MAJOR}.${CUDA_MINOR}
RUN apt-get update \
    # for cuda
    && apt-get install -y --no-install-recommends cuda-toolkit-${CUDA_MAJOR}-${CUDA_MINOR} \
    && apt-get install -y --no-install-recommends libcudnn${CUDNN_MAJOR}=${CUDNN_VER_NAME} libcudnn${CUDNN_MAJOR}-dev=${CUDNN_VER_NAME} \
    # for pyenv
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y --no-install-recommends \
        build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
        xz-utils tk-dev libffi-dev liblzma-dev python-openssl git \
    # for catkin build
    && apt-get install -y --no-install-recommends python3-catkin-tools python3-vcstool \
    # for develop
    # ros-noetic-tf2-bullet: packages.ros.org のミラーによっては削除済みの古いスナップショットを
    # 案内して404になることがあるため、動作確認済みのバージョンを直接URL指定して取得する
    && apt-get update \
    && curl -fsSL -o /tmp/ros-noetic-tf2-bullet.deb \
        http://packages.ros.org/ros/ubuntu/pool/main/r/ros-noetic-tf2-bullet/ros-noetic-tf2-bullet_0.7.10-1focal.20250502.232148_amd64.deb \
    && apt-get install -y --no-install-recommends /tmp/ros-noetic-tf2-bullet.deb netcat-openbsd \
    && rm -f /tmp/ros-noetic-tf2-bullet.deb \
    # for compatibility with the latest VSCode server
    && apt-get install -y --no-install-recommends code \
    # cleaning
    && apt-get clean

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

RUN mkdir -p /workspace \
    && chown -R developer:developer /workspace

USER developer

ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
RUN git clone https://github.com/yyuu/pyenv.git $HOME/.pyenv -b v1.2.23 \
    && git clone https://github.com/yyuu/pyenv-virtualenv.git $HOME/.pyenv/plugins/pyenv-virtualenv -b v1.1.5 \
    && echo -e '\nPYENV_ROOT=$HOME/.pyenv\nPATH=$PYENV_ROOT/bin:$PATH\neval "$(pyenv init -)"\neval "$(pyenv virtualenv-init -)"\n' >> $HOME/.bashrc

RUN eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" \
    && CONFIGURE_OPTS=--enable-shared pyenv install ${PYTHON_VERSION}
COPY requirements-*.txt /workspace/
RUN eval "$(pyenv init -)" && eval "$(pyenv virtualenv-init -)" \
    && pyenv virtualenv ${PYTHON_VERSION} ${PYENV_NAME} \
    && cd /workspace && pyenv local ${PYENV_NAME} \
    && ln -s requirements-cu${CUDA_MAJOR}${CUDA_MINOR}.txt requirements-cu.txt \
    && pip install -U pip==21.2.4 \
    # pycocotoolsをinstallするために、cythonとnumpyを入れておく
    && pip install cython numpy \
    && pip install -r /workspace/requirements-cu.txt\
    && catkin init \
    && catkin config --cmake-args \
        -DPYTHON_EXECUTABLE=$(pyenv prefix)/bin/python3 \
        -DPYTHON_LIBRARY=${HOME}/.pyenv/versions/${PYTHON_VERSION}/lib/libpython3.so \
        -DPYTHON_VERSION=3 \
    && rm /workspace/requirements-*.txt

COPY settings.json /home/developer/.vscode-server/settings.json

COPY entrypoint.sh /entrypoint.sh
