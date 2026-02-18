# syntax=docker/dockerfile:1.7

ARG PYTORCH_IMAGE=pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_BIN=python3
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-venv \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ${PYTHON_BIN} -m venv --system-site-packages ${VIRTUAL_ENV}

RUN ${VIRTUAL_ENV}/bin/python -m pip install --upgrade --no-cache-dir pip setuptools wheel

RUN ${VIRTUAL_ENV}/bin/python -m pip install --no-cache-dir gymnasium

WORKDIR /workspace

CMD ["bash"]
