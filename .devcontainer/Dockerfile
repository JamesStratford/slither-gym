FROM nvidia/cuda:11.8.0-base-ubuntu22.04

RUN apt-get update && \
    apt-get install -y \
        iproute2 \
        wget \
        build-essential \
        && rm -rf /var/lib/apt/lists/*
