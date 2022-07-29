# hash:sha256:81cb6074b98f0d07cdc8e062f4155c7ed1ecf79207aa377e5c5b7d321c6eec2a
FROM registry.codeocean.com/codeocean/miniconda3:4.9.2-python3.8-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        fonts-liberation=1:1.07.4-11 \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -y --channel=conda-forge \
        glpk==5.0 \
        matplotlib==3.5.2 \
        numpy==1.20.3 \
        pandas==1.4.3 \
        scip==8.0.1 \
    && conda clean -ya

RUN pip3 install -U --no-cache-dir \
    -e git+https://github.com/Pyomo/pyomo.git@1538b17f738bc4c7cd6a9246bc0b8cf32fbbe07e#egg=pyomo
