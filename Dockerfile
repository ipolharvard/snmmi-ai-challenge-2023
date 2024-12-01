FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

ENV TZ=America/Montreal
ARG DEBIAN_FRONTEND=noninteractive

# Add a /.local/bin and /.local/lib directories to allow editable python
# installs by any user
RUN mkdir -p -m 777 /.local/bin /.local/lib

# Install python and other useful programs
RUN apt update && apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        python-is-python3 \
        git \
        glances \
        vim \
        tmux \
        curl && \
    apt clean

COPY requirements.txt .
# Install Python requirements using the compiled version of the requirements
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip install -r requirements.txt && \
    pip install -U scipy && \
    rm requirements.txt \
