---
name: mat-electronic-structure
description: Calculate electronic band structure and density of states using ASE + Quantum ESPRESSO.
category: [materials]
---

# Electronic Structure

## Goal

To calculate the electronic band structure of a crystalline material, revealing the energy-momentum relationship for electrons and determining whether the material is metallic, semiconducting, or insulating. This includes computing the band gap ($E_g$), identifying direct vs. indirect transitions, and visualizing the dispersion along high-symmetry k-paths.

This skill uses **ASE + Quantum ESPRESSO** for local DFT calculations. The previous atomate2 + VASP path is retained as a legacy option (see "Legacy VASP path" below).

## Prerequisites / Environment Check

- `MP_API_KEY` (recommended) — Required when retrieving pre-computed electronic structure from Materials Project via `get_mp_electronic_structure.py` (or when using the MCP search tool). Without it, MP retrieval will fail. Get a free key at https://next-gen.materialsproject.org/api.
- `ESPRESSO_PSEUDO` (required for ASE + QE calculations) — Path to a directory containing Quantum ESPRESSO pseudopotentials (e.g. SSSP efficiency or precision libraries). The `qe` pixi environment does **not** include pseudopotentials.
- `ASE_ESPRESSO_COMMAND` or `ESPRESSO_COMMAND` (optional) — Command to run `pw.x`, e.g. `mpirun -np 4 pw.x`. If unset, ASE defaults to `pw.x`.

See `docs/api_key_guide.md`, `docs/environment_variables.md`, and `docs/hpc_job_submission.md` for setup details.

Before running this skill, verify the variables for your chosen data source and execution mode are set. If any required variable is missing, ask the user to set it before proceeding.

## Instructions

### 1. Obtain or Prepare the Input Structure

Start with a relaxed crystalline structure in CIF or POSCAR format. You can:

- Search Materials Project using the [`mcp_base_search_materials_project_by_formula`](../../../src/mcp_server/base_server.py) tool
- Use a structure from previous calculations
- Create a structure manually using pymatgen or ASE

### 2. Run Band Structure Calculation

Use the ASE + Quantum ESPRESSO command-line script:

```bash
# Env: qe
python .agents/skills/mat-electronic-structure/scripts/run_qe_band_structure.py \
    structure.cif \
    --mode line \
    --output-dir ./qe_band_structure \
    --ecutwfc 50 \
    --kpts 6 6 6 \
    --n-points 100 \
    --pseudo-dir $ESPRESSO_PSEUDO
```

**Band structure modes:**
- `"line"`: Calculate along high-symmetry k-paths (for band structure plots)
- `"uniform"`: Calculate on a uniform k-mesh (for density of states)
- `"both"`: Perform both line and uniform calculations

The workflow automatically:
1. Runs a static SCF calculation to obtain the charge density.
2. Runs a non-SCF calculation (`bands` for line mode, `nscf` for uniform mode) to compute eigenvalues.
3. Writes `results.json` with eigenvalues, k-points, Fermi energy, and band gap.

You can pass `--command` to override the `pw.x` command and `--magmoms` for spin-polarized systems.

### Alternative: Retrieve Pre-Computed Data from Materials Project

Instead of running DFT calculations, you can retrieve existing electronic structure data from Materials Project:

```bash
# Env: base
python .agents/skills/mat-electronic-structure/scripts/get_mp_electronic_structure.py \
    --material_id mp-149 \
    --output si_mp_bands.json \
    --plot
```

**This retrieves**:
- Pre-computed band structure along high-symmetry paths
- Density of states (DOS)
- Band gap (energy, direct/indirect)
- Fermi energy

**When to use MP retrieval vs. calculations**:
- **Retrieve from MP**: Quick screening, validation, known materials
- **Run calculations**: New materials, custom structures, specific DFT settings

### 3. Post-Process and Visualize Results

After the calculation completes, parse the results and generate a band structure plot:

```bash
# Env: qe
python .agents/skills/mat-electronic-structure/scripts/plot_qe_band_structure.py \
    ./qe_band_structure \
    --output band_structure.png
```

The script will:
- Read `qe_band_structure/results.json`
- Set the Fermi level as the energy reference
- Generate a publication-quality band structure plot

**For DOS (uniform mode)**:

```bash
# Env: qe
python .agents/skills/mat-electronic-structure/scripts/plot_qe_dos.py \
    ./qe_band_structure \
    --output dos.png
```

The script will:
- Read `qe_band_structure/results.json`
- Compute total DOS from uniform-k eigenvalues with Gaussian broadening
- Generate a DOS plot

**Alternative: Manual post-processing with matplotlib**

```python
import json
import numpy as np
import matplotlib.pyplot as plt

with open("qe_band_structure/results.json") as f:
    results = json.load(f)

eigenvalues = np.asarray(results["eigenvalues_line"])
kpoints = np.asarray(results["kpoints_line"])
fermi_energy = results["fermi_energy"]

# Simple band structure plot
distances = np.linalg.norm(np.diff(kpoints, axis=0, prepend=kpoints[:1]), axis=1)
distances = np.cumsum(distances)

plt.figure(figsize=(8, 6))
for band in range(eigenvalues.shape[-1]):
    plt.plot(distances, eigenvalues[:, band] - fermi_energy, color="blue")
plt.axhline(0, color="red")
plt.ylabel("Energy (eV)")
plt.savefig("band_structure.png", dpi=300, bbox_inches="tight")
```

## Examples

### Silicon Band Structure Calculation

```bash
# 1. Search for Si structure (or download Si.cif)
# 2. Run band structure calculation with ASE + QE
# Env: qe
python .agents/skills/mat-electronic-structure/scripts/run_qe_band_structure.py \
    Si.cif \
    --mode line \
    --output-dir ./Si_qe_bands \
    --ecutwfc 50 \
    --kpts 6 6 6 \
    --n-points 100 \
    --pseudo-dir $ESPRESSO_PSEUDO

# 3. Plot results
python .agents/skills/mat-electronic-structure/scripts/plot_qe_band_structure.py \
    ./Si_qe_bands --output Si_qe_bands.png
```

### Silicon Density of States (DOS)

```bash
# Env: qe
python .agents/skills/mat-electronic-structure/scripts/run_qe_band_structure.py \
    Si.cif \
    --mode uniform \
    --output-dir ./Si_qe_dos \
    --ecutwfc 50 \
    --kpts 8 8 8 \
    --pseudo-dir $ESPRESSO_PSEUDO

# Plot DOS
python .agents/skills/mat-electronic-structure/scripts/plot_qe_dos.py \
    ./Si_qe_dos --output Si_qe_dos.png
```

### Band Structure + DOS in One Run

```bash
# Env: qe
python .agents/skills/mat-electronic-structure/scripts/run_qe_band_structure.py \
    Si.cif \
    --mode both \
    --output-dir ./Si_qe_bands_dos \
    --ecutwfc 50 \
    --kpts 6 6 6 \
    --n-points 100 \
    --pseudo-dir $ESPRESSO_PSEUDO

python .agents/skills/mat-electronic-structure/scripts/plot_qe_band_structure.py \
    ./Si_qe_bands_dos --output Si_qe_bands.png
python .agents/skills/mat-electronic-structure/scripts/plot_qe_dos.py \
    ./Si_qe_bands_dos --output Si_qe_dos.png
```

See [examples/](examples/) for a complete Si band structure calculation.

## Constraints

- **Structure Requirements**: Input must be a crystalline structure with well-defined symmetry. Band structure calculations are not meaningful for amorphous or highly disordered materials.
- **QE Setup**: Requires properly configured Quantum ESPRESSO environment:
  - `ESPRESSO_PSEUDO` must point to a valid pseudopotential directory.
  - `ASE_ESPRESSO_COMMAND` or `ESPRESSO_COMMAND` should be set for parallel execution.
- **Environments**:
  - Band structure calculation: `qe` (ASE, seekpath)
  - Post-processing scripts: `qe` (numpy, matplotlib)
- **Functional Choice**:
  - PBE typically **underestimates** band gaps.
  - For accurate gaps: Use hybrid functionals (HSE06) or GW methods (not yet supported in the ASE-QE script).
- **k-point Density**: The high-symmetry k-path is generated with `seekpath`. For very accurate results, you may increase `--n-points` or provide a denser SCF k-grid.
- **Spin-Polarization**: Pass `--magmoms` to run spin-polarized calculations for magnetic materials.

## Legacy VASP Path

The previous atomate2 + VASP scripts (`plot_band_structure.py`, `plot_dos.py`) are kept for historical VASP results only. New calculations should use the ASE + Quantum ESPRESSO scripts above.

## Foundation Potential Recommendations

For exploratory band structure analysis using MLIPs (not DFT):

- **CHGNet**: Can predict band gaps, but accuracy varies
- **Note**: MACE, M3GNet, and other MLIPs trained on PES data do not predict electronic properties

For production calculations, **always use DFT** (this skill) and choose the appropriate functional:
- **Standard screening**: PBE
- **Improved gaps**: r2SCAN (not yet exposed in the ASE-QE CLI)
- **Accurate gaps**: HSE06 or GW (requires custom input)
---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
