#!/bin/bash
set -e

# Source conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate react-ot-agent

# Download models
echo "Downloading models..."
python conda-envs/react-ot-agent/download_models.py

# Run example
echo "Running example generation..."
python .agent/skills/chem-react-ot/scripts/generate_ts.py \
    --reactants .agent/skills/chem-react-ot/examples/oxadiazole_isomerization/reactant.xyz \
    --products .agent/skills/chem-react-ot/examples/oxadiazole_isomerization/product.xyz \
    --output_dir .agent/skills/chem-react-ot/examples/oxadiazole_isomerization/output \
    --nfe 10

echo "Verification complete!"
