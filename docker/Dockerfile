# FROM ubuntu:22.04
FROM docker.io/nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install basic apt packages.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    nano \
    vim \
    wget \
    curl \
    g++ \
    ninja-build

ARG CONDA_INSTALL_DIR=/opt/miniconda3
RUN mkdir -p ${CONDA_INSTALL_DIR}
RUN wget --no-check-certificate https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ${CONDA_INSTALL_DIR}/miniconda.sh
RUN bash ${CONDA_INSTALL_DIR}/miniconda.sh -b -u -p ${CONDA_INSTALL_DIR}
RUN rm -rf ${CONDA_INSTALL_DIR}/miniconda.sh

# Update conda and install packages
RUN ${CONDA_INSTALL_DIR}/bin/conda update -y conda && \
    ${CONDA_INSTALL_DIR}/bin/conda install -y python=3.12 pip

RUN ${CONDA_INSTALL_DIR}/bin/pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

RUN ${CONDA_INSTALL_DIR}/bin/pip install \
    mmengine-lite==0.10.4 scikit-learn==1.5.0 tqdm==4.66.5 \
    pyarrow==16.1.0 numpy==1.26.4 matplotlib==3.9.0 pandas==2.2.2
RUN ${CONDA_INSTALL_DIR}/bin/pip install requests==2.32.3
RUN ${CONDA_INSTALL_DIR}/bin/pip install optuna scipy sympy einops
RUN ${CONDA_INSTALL_DIR}/bin/pip install timm==1.0.9

RUN ${CONDA_INSTALL_DIR}/bin/pip install packaging
RUN ${CONDA_INSTALL_DIR}/bin/pip install netCDF4 xarray cartopy h5py bottleneck
RUN ${CONDA_INSTALL_DIR}/bin/pip install bd-metric pillow pybind11

ENTRYPOINT /bin/bash
