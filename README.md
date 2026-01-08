# Simulation MCP: Machine Learning Interatomic Potential Agent

## Overview
Simulation MCP is a comprehensive framework designed to integrate Machine Learning Interatomic Potentials (MLIPs) with automated simulation workflows. It leverages the Model Context Protocol (MCP) to expose powerful atomistic simulation tools, property calculations, and fine-tuning capabilities to AI agents.

The project aims to provide a seamless bridge between natural language scientific questions and rigorous atomistic simulations.

## Key Features

### 1. Multi-Framework MLIP Support
The project provides unified wrappers for the three leading MLIP frameworks, each running in its specialized conda environment:
- **MACE**: Supports MACE-MP, MACE-MATPES, and MACE-H (Multi-Head) models.
- **MatGL**: Supports CHGNet, M3GNet, and TensorNet models.
- **FAIRCHEM**: Supports UMA (Universal) and ESEN (Organic/Catalysis) models.

### 2. High-Level Property Calculations
Expose complex simulation tasks as simple tools:
- **Structural Relaxation**: Geometry optimization using various optimizers (FIRE, BFGS, LBFGS).
- **Molecular Dynamics**: NVT, NPT, and NVE ensembles with various thermostats and barostats.
- **Barrier Calculations**: Nudged Elastic Band (NEB) for reaction pathways.
- **Thermal Properties**: Phonon calculations and Quasi-Harmonic Approximation (QHA).

### 3. Advanced Data Augmentation & Sampling
Generate robust training data for fine-tuning:
- **Off-Equilibrium Sampling**: MD-based sampling with clustering.
- **Near-Equilibrium Sampling**: Targeted sampling around ground states.
- **Order-Disorder Sampling**: Generating ordered structures from disordered inputs.

### 4. Automated Fine-tuning Pipeline
Fine-tune foundation models on custom datasets:
- automatic conversion of structures and labels (energy, forces, stress).
- Backwards propagation of metrics and training history.
- Automatic generation of training visualization plots.

### 5. DFT Integration
Bridge the gap between MLIPs and first-principles calculations:
- **VASP Preparation**: Automatically generate INCAR, KPOINTS, POSCAR, and POTCAR files with sensible defaults.
- **VASP Parsing**: Extract energy, forces, and stress from VASP outputs for labeling.
- **Mock DFT**: Use high-accuracy MLIPs (like UMA) to simulate DFT results for rapid testing.

## Environment Setup
The project uses separate conda environments to manage conflicting MLIP dependencies.

1. **Materials Tools / Main**: `conda activate mlip-agent`
2. **MatGL**: `conda activate matgl-agent`
3. **MACE**: `conda activate mace-agent`
4. **FAIRCHEM**: `conda activate fairchem-agent`

### Installation
```bash
# Clone the repository
git clone git@github.com:bowen-bd/simulation_mcp.git
cd simulation_mcp

# Setup environments (assuming .yml files are in conda-envs/)
conda env create -f conda-envs/mlip-agent.yml
conda env create -f conda-envs/matgl-environment.yml
conda env create -f conda-envs/mace-environment.yml
conda env create -f conda-envs/fairchem-environment.yml
```

## Running the MCP Servers
The project is configured to run as a set of MCP servers. The configuration is stored in `mcp_config.json`.

To run a server manually (e.g., MACE):
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
/home/bdeng/miniforge3/envs/mace-agent/bin/python -m src.mcp_server.mace_server
```

## Project Structure
- `src/`: Core package containing wrappers and servers.
  - `mcp_server/`: FastMCP server implementations.
  - `utils/`: Utility functions for structures, DFT, and MLIP management.
- `tests/`: Integration tests for servers and tools.
- `.agent/`: Agent-specific rules, instructions, and workflows.
- `research/`: Directory for storing simulation results and research logs.

## Contributing
Simulation MCP is developed as a tool for automated atomistic research. Contributions to new potentials, sampling methods, or simulation workflows are welcome.
