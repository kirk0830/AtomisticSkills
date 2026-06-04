#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

ENV_NAME=$(grep -e '^name:' core_env.yaml | awk '{print $2}')
echo "Creating Conda environment $ENV_NAME..."

sed '/^[[:space:]]*- pip:/,$d' core_env.yaml > conda_only_env.yaml
conda env remove -n $ENV_NAME -y || true
conda env create -f conda_only_env.yaml

sed -n '/^[[:space:]]*- pip:/,$p' core_env.yaml | grep -v 'pip:' | sed 's/^[[:space:]]*- //' | tr -d '"' | tr -d "'" > uv_requirements.txt

if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "Installing pip dependencies with uv..."
uv pip install -r uv_requirements.txt

# Install ms_pred from GitHub.
# setup.py has a Cython ext (massformer only) that breaks standard install,
# so we clone, patch setup.py to skip the ext, then install pure-Python package.
echo "Installing ms_pred from GitHub..."
TMP_DIR=$(mktemp -d)
git clone --depth 1 https://github.com/coleygroup/ms-pred "$TMP_DIR/ms-pred"

# Patch: replace setup.py with a Cython-free version
cat > "$TMP_DIR/ms-pred/setup.py" << 'SETUP_EOF'
from setuptools import setup, find_packages
setup(
    name='ms_pred',
    version='0.0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
)
SETUP_EOF

uv pip install "$TMP_DIR/ms-pred"
rm -rf "$TMP_DIR"

rm -f conda_only_env.yaml uv_requirements.txt
echo "Environment $ENV_NAME created successfully!"
