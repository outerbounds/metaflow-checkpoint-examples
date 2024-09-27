FROM docker.io/nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu20.04
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    git \
    wget \
    curl \
    vim \
    tmux \
    sudo 
RUN curl -sL https://micro.mamba.pm/api/micromamba/linux-64/1.1.0 \
    | sudo tar -xvj -C /usr/local bin/micromamba
ENV MAMBA_EXE=/usr/local/bin/micromamba \
    MAMBA_ROOT_PREFIX=/home/user/micromamba \
    CONDA_PREFIX=/home/user/micromamba \
    PATH=/home/user/micromamba/bin:$PATH
RUN micromamba create -y -n base && \
    micromamba shell init --shell=bash --prefix="$MAMBA_ROOT_PREFIX"
RUN micromamba install python=3.11 pip -c conda-forge -y && python -m pip install --upgrade pip
RUN git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
RUN cd LLaMA-Factory && python -m pip install -e ".[torch,metrics]"
RUN python -m pip install liger-kernel bitsandbytes hqq gptq aqlm vllm galore badam adam-mini qwen modelscope