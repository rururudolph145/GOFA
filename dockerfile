FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    WANDB_MODE=offline

# Python 3.9 + build tools
RUN apt-get update && apt-get install -y \
    software-properties-common curl ca-certificates git build-essential \
 && add-apt-repository ppa:deadsnakes/ppa -y \
 && apt-get update && apt-get install -y \
    python3.9 python3.9-venv python3.9-dev python3-pip \
 && rm -rf /var/lib/apt/lists/*

# uv
RUN python3.9 -m pip install --upgrade pip && pip install uv

WORKDIR /workspace

# Create venv OUTSIDE /workspace so bind-mounts don't hide it
RUN uv venv -p python3.9 /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# NEW: ensure build tooling is present in the venv
RUN . /opt/venv/bin/activate && uv pip install --upgrade setuptools wheel packaging

# Install CUDA-tied packages first
RUN . /opt/venv/bin/activate && uv pip install \
  --extra-index-url https://download.pytorch.org/whl/cu116 \
  torch==1.13.0+cu116 torchvision==0.14.0+cu116
# If you need audio:
# RUN . /opt/venv/bin/activate && uv pip install --extra-index-url https://download.pytorch.org/whl/cu116 torchaudio==0.13.0+cu116

# PyTorch Geometric wheels matching torch==1.13.0+cu116
RUN . /opt/venv/bin/activate && uv pip install \
  --no-build-isolation \
  --find-links https://data.pyg.org/whl/torch-1.13.0+cu116.html \
  torch-scatter==2.1.0 torch-sparse==0.6.15 torch-cluster==1.6.0 \
  torch-spline-conv==1.2.2 torch-geometric==2.2.0

# ---- Torch-dependent native ext that needs torch visible at build time ----
RUN . /opt/venv/bin/activate && uv pip install --no-build-isolation causal-conv1d==1.1.3.post1
RUN . /opt/venv/bin/activate && uv pip install --no-build-isolation mamba-ssm==1.1.3.post1
  
# Project dependencies (everything else) — ensure this file does NOT re-pin torch/PyG
COPY requirements.txt /workspace/requirements.txt
RUN . /opt/venv/bin/activate && uv pip install --index-url https://pypi.org/simple -r /workspace/requirements.txt



# Bring in source after deps
COPY . /workspace

SHELL ["/bin/bash", "-lc"]
COPY . /workspace
SHELL ["/bin/bash", "-lc"]

# ensure /workspace/.venv -> /opt/venv each run
RUN printf '#!/usr/bin/env bash\nset -e\nif [ ! -e /workspace/.venv ]; then ln -s /opt/venv /workspace/.venv 2>/dev/null || true; fi\nexec "$@"\n' \
  > /usr/local/bin/entrypoint.sh && chmod +x /usr/local/bin/entrypoint.sh

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["bash"]

