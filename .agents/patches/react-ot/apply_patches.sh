#!/bin/bash
# Apply patches to react-ot source code
#
# These patches are required for react-ot to work correctly:
# 1. Enable package data inclusion in pyproject.toml
# 2. Enable namespace packages in pyproject.toml
# 3. Fix ASE NEB import path (ASE API change)
#
# Usage: ./apply_patches.sh <react-ot-build-dir>
#
# Applied by: pixi run install-react-ot
# Source: https://github.com/deepprinciple/react-ot

set -euo pipefail

BUILD_DIR="${1:?Error: BUILD_DIR argument required}"

echo "Applying react-ot patches..."

# Patch pyproject.toml: enable package data
sed -i 's/include-package-data = false/include-package-data = true/' "$BUILD_DIR/pyproject.toml"

# Patch pyproject.toml: enable namespace packages
sed -i 's/namespaces = false/namespaces = true/' "$BUILD_DIR/pyproject.toml"

# Patch _utils.py: fix ASE NEB import (API changed in newer ASE versions)
sed -i 's/from ase.neb import NEB/from ase.mep import NEB/' "$BUILD_DIR/reactot/diffusion/_utils.py"

echo "Patches applied successfully."