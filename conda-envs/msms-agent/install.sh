#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ENV_NAME=$(grep -e '^name:' core_env.yaml | awk '{print $2}')
echo "Creating Conda environment $ENV_NAME..."

sed '/^[[:space:]]*- pip:/,$d' core_env.yaml > conda_only_env.yaml
conda env remove -n $ENV_NAME -y || true
conda env create -f conda_only_env.yaml

sed -n '/^[[:space:]]*- pip:/,$p' core_env.yaml | grep -v 'pip:' | sed 's/^[[:space:]]*- //' | tr -d '"' | tr -d "'" | grep -v '^dgl ' > uv_requirements.txt

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing dgl from custom index..."
uv pip install dgl --find-links https://data.dgl.ai/wheels/torch-2.4/repo.html

echo "Installing pip dependencies with uv..."
uv pip install -r uv_requirements.txt

# torch-scatter / torch-sparse: no generic arm64 wheels on PyPI.
# Use the PyG find-links index pinned to torch 2.4.0+cpu.
TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
echo "Installing torch-scatter and torch-sparse from PyG index (torch ${TORCH_VERSION})..."
uv pip install torch-scatter torch-sparse --find-links "$PYG_INDEX"

# Install ms_pred from GitHub.
# setup.py includes a Cython extension for massformer (not used by ICEBERG),
# so we clone, replace setup.py with a pure-Python version, then install.
echo "Installing ms_pred from GitHub..."
TMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/coleygroup/ms-pred "$TMP_DIR/ms-pred"

cat > "$TMP_DIR/ms-pred/setup.py" << 'SETUP_EOF'
from setuptools import setup, find_namespace_packages
setup(
    name='ms_pred',
    version='0.0.1',
    packages=find_namespace_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
)
SETUP_EOF

uv pip install "$TMP_DIR/ms-pred"
rm -rf "$TMP_DIR"

rm -f conda_only_env.yaml uv_requirements.txt
echo "Environment $ENV_NAME created successfully!"
