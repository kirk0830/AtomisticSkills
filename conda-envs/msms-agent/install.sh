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

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS: DGL 2.2.0 graphbolt dylibs only go up to pytorch 2.3.0.
    # torchdata>=0.8 dropped datapipes and pulls torch 2.12 as dep; pin 0.7.1 --no-deps.
    # torch_scatter/sparse must match torch 2.3.0 exactly.
    echo "macOS: installing torch 2.3.0 (CPU)..."
    pip install "torch==2.3.0" --index-url https://download.pytorch.org/whl/cpu

    echo "macOS: installing DGL from torch-2.3 index..."
    uv pip install dgl --find-links https://data.dgl.ai/wheels/torch-2.3/repo.html

    echo "macOS: installing pip dependencies..."
    uv pip install -r uv_requirements.txt

    # DGL 2.2.0 graphbolt imports torchdata which is incompatible with torch 2.3.
    # ICEBERG never uses graphbolt at runtime; stub all torchdata imports in graphbolt.
    echo "macOS: patching DGL graphbolt to remove torchdata dependency..."
    python - <<'DGLPATCH'
import pathlib, sys, re
site = pathlib.Path(sys.executable).parent.parent / "lib/python3.10/site-packages"
graphbolt = site / "dgl/graphbolt"
for f in graphbolt.glob("*.py"):
    txt = f.read_text()
    if "torchdata" not in txt:
        continue
    # Replace all torchdata imports with try/except stubs
    patched = re.sub(
        r'^((?:from|import) torchdata\S*.*)',
        r'try:\n    \1\nexcept (ImportError, ModuleNotFoundError):\n    pass',
        txt, flags=re.MULTILINE
    )
    f.write_text(patched)
    print(f"Patched {f.name}")
DGLPATCH

    echo "macOS: installing torch-scatter and torch-sparse for torch 2.3.0..."
    uv pip install --force-reinstall torch-scatter torch-sparse \
        --find-links https://data.pyg.org/whl/torch-2.3.0+cpu.html
else
    echo "Installing dgl from custom index..."
    uv pip install dgl --find-links https://data.dgl.ai/wheels/torch-2.4/repo.html

    echo "Installing pip dependencies with uv..."
    uv pip install -r uv_requirements.txt

    # torch-scatter / torch-sparse: use the PyG find-links index pinned to torch 2.4.0+cpu.
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    PYG_INDEX="https://data.pyg.org/whl/torch-${TORCH_VERSION}+cpu.html"
    echo "Installing torch-scatter and torch-sparse from PyG index (torch ${TORCH_VERSION})..."
    uv pip install torch-scatter torch-sparse --find-links "$PYG_INDEX"
fi

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

# Patch ms_pred source for upstream bugs before installing:
#   1. iceberg_elucidation.py calls predict_smis.py via a cwd-relative path
#   2. predict_smis.py uses pl.utilities.seed.seed_everything (removed in PL 2.0)
#   3. predict_smis.py calls torch.cuda.set_device(gpu_id) unconditionally (fails on CPU)
python - "$TMP_DIR/ms-pred/src/ms_pred/dag_pred" <<'PYPATCH'
import sys, pathlib
d = pathlib.Path(sys.argv[1])
el = d / "iceberg_elucidation.py"
el.write_text(el.read_text().replace(
    '{python_path} src/ms_pred/dag_pred/predict_smis.py',
    '{python_path} {Path(__file__).resolve().parent / "predict_smis.py"}'))
ps = d / "predict_smis.py"
t = ps.read_text()
t = t.replace('pl.utilities.seed.seed_everything', 'pl.seed_everything')
t = t.replace('torch.cuda.set_device(gpu_id)',
              '(torch.cuda.set_device(gpu_id) if (gpu and avail_gpu_num > 0) else None)')
ps.write_text(t)
print("Patched ms_pred dag_pred (pip-install path / PL2 seed / CPU cuda guard)")
PYPATCH

uv pip install "$TMP_DIR/ms-pred"
rm -rf "$TMP_DIR"

# macOS only: create fake torchdata package so DGL graphbolt imports succeed.
# DGL uses graphbolt only for distributed training; ICEBERG never triggers it.
if [[ "$(uname)" == "Darwin" ]]; then
    echo "macOS: creating torchdata compatibility shim for DGL..."
    python - <<'TDPATCH'
import pathlib, sys
site = pathlib.Path(sys.executable).parent.parent / "lib/python3.10/site-packages"
(site / "torchdata/datapipes/iter").mkdir(parents=True, exist_ok=True)
(site / "torchdata/dataloader2").mkdir(parents=True, exist_ok=True)
(site / "torchdata/__init__.py").write_text("")
(site / "torchdata/datapipes/__init__.py").write_text("from . import iter\n")
(site / "torchdata/datapipes/iter/__init__.py").write_text(
    "import torch.utils.data\n"
    "class IterDataPipe(torch.utils.data.IterableDataset): pass\n"
    "class IterableWrapper(IterDataPipe):\n"
    "    def __init__(self, iterable): self.iterable = iterable\n"
    "    def __iter__(self): yield from self.iterable\n"
    "class Mapper(IterDataPipe):\n"
    "    def __init__(self, datapipe=None, fn=None): self.datapipe=datapipe; self.fn=fn\n"
    "    def __iter__(self):\n"
    "        for item in self.datapipe: yield self.fn(item)\n"
)
(site / "torchdata/dataloader2/__init__.py").write_text("")
(site / "torchdata/dataloader2/graph.py").write_text(
    "def traverse_dps(dp): return {}\n"
    "def find_dps(g, t): return []\n"
    "def replace_dp(g, old, new): return g\n"
)
print("Created torchdata shim")
TDPATCH
fi

rm -f conda_only_env.yaml uv_requirements.txt
echo "Environment $ENV_NAME created successfully!"
