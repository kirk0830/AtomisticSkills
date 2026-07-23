---
name: mat-magnetic-density
description: Calculate magnetic moments and spin density from spin-polarized DFT calculations using ASE + CP2K.
category: [materials]
---

# Magnetic Density

## Goal

To calculate the magnetic moments of magnetic materials using spin-polarized DFT calculations. This skill enables the characterization of magnetic ordering and local magnetic moments on individual atoms.

This skill uses **ASE + CP2K**. The previous atomate2 + VASP path is retained as a legacy option for spin-density cube extraction (see below).

## Prerequisites / Environment Check

- `MP_API_KEY` (recommended) — Required when querying the input structure from Materials Project via the MCP search tool. Without it, MP structure search will fail. Get a free key at https://next-gen.materialsproject.org/api.
- `CP2K_DATA_DIR` (required for ASE + CP2K calculations) — Path to CP2K data directory containing `BASIS_SET` and `POTENTIAL` files.
- `ASE_CP2K_COMMAND` or `CP2K_COMMAND` (optional) — Command to run CP2K, e.g. `mpirun -np 4 cp2k`. If unset, ASE defaults to `cp2k`.

See `docs/api_key_guide.md`, `docs/environment_variables.md`, and `docs/hpc_job_submission.md` for setup details.

Before running this skill, verify the variables for your chosen execution mode are set. If any required variable is missing, ask the user to set it before proceeding.

## Instructions

### 1. Structure Preparation

Obtain or prepare the structure of the magnetic material you want to study. You can:
- Query from Materials Project using the base MCP tools (recommended - already DFT-optimized)
- Load from a local CIF/POSCAR file
- Use a previously relaxed structure

**Note on relaxation**: For magnetic moment calculations, **relaxation is optional** if using high-quality experimental or Materials Project structures. Magnetic moments are relatively insensitive to small structural variations. However, **relaxation is recommended** for:
- New/hypothetical structures
- Surfaces, interfaces, or defects
- Systems where you need accurate total energies (not just magnetic moments)
- Strongly correlated oxides with significant magnetic-structural coupling

### 2. Run Spin-Polarized CP2K Calculation

Use the ASE + CP2K command-line script to run a spin-polarized calculation:

```bash
# Env: cp2k
python .agents/skills/mat-magnetic-density/scripts/run_cp2k_magnetic.py \
    Fe.cif \
    --output-dir ./Fe_magnetic \
    --spin-polarized \
    --initial-magmoms 2.2 \
    --cutoff 400 \
    --kpts 4 4 4 \
    --xc PBE
```

**Important - Functional Selection**:
- **For metallic ferromagnets** (Fe, Co, Ni): Use PBE. PBE provides good accuracy for magnetic moments at modest cost.
- **For strongly correlated oxides** (NiO, CoO, FeO): CP2K supports +U corrections via custom input, but the CLI above does not set them automatically. Standard PBE may fail to predict the correct insulating antiferromagnetic ground state.
- **Avoid r2SCAN for metallic ferromagnets**: r2SCAN can overestimate magnetic moments in itinerant ferromagnets like Fe.

### 3. Parse and Analyze Results

Use the provided script to parse the magnetic moments from `results.json`:

```bash
# Env: cp2k (or base)
python .agents/skills/mat-magnetic-density/scripts/parse_magnetic_moments.py \
    Fe_magnetic/results.json \
    --output magnetic_analysis.json
```

This script will:
- Extract site-resolved magnetic moments from the Mulliken spin population analysis
- Calculate the total magnetization
- Identify the magnetic ordering pattern
- Generate a summary report

### 4. Extract Spin Density (Optional, VASP legacy)

CP2K spin-density cube parsing is not yet implemented in this skill. The VASP spin-density extraction script is kept for historical VASP results:

```bash
# Env: base
python .agents/skills/mat-magnetic-density/scripts/extract_spin_density.py <vasp_output_dir> --output spin_density.json
```

This requires access to the CHGCAR file from the VASP calculation.

### 5. Visualize Magnetic Ordering (Optional)

Visualize the magnetic ordering by creating a structure with magnetic moment vectors:

```bash
# Env: base
python .agents/skills/mat-magnetic-density/scripts/visualize_magnetic_structure.py structure.cif magnetic_analysis.json --output magnetic_structure.png
```

## Examples

### Example 1: Calculate magnetic moments for bulk Fe

```bash
# Step 1: Get Fe structure from Materials Project (or use a local Fe.cif)

# Step 2: Run spin-polarized CP2K calculation
# Env: cp2k
python .agents/skills/mat-magnetic-density/scripts/run_cp2k_magnetic.py \
    Fe.cif \
    --output-dir Fe_magnetic \
    --spin-polarized \
    --initial-magmoms 2.2 \
    --cutoff 400 \
    --kpts 4 4 4 \
    --xc PBE

# Step 3: Parse magnetic moments
python .agents/skills/mat-magnetic-density/scripts/parse_magnetic_moments.py \
    Fe_magnetic/results.json --output Fe_moments.json
```

### Example 2: Antiferromagnetic NiO

```bash
# For antiferromagnetic ordering, provide alternating initial moments.
# Here a 2-atom NiO cell is assumed; adjust --initial-magmoms for your supercell.
# Env: cp2k
python .agents/skills/mat-magnetic-density/scripts/run_cp2k_magnetic.py \
    NiO.cif \
    --output-dir NiO_magnetic \
    --spin-polarized \
    --initial-magmoms 1.0,-1.0 \
    --cutoff 500 \
    --kpts 4 4 4 \
    --xc PBE
```

## Functional Selection Guide

Choosing the right exchange-correlation functional is critical for accurate magnetic property calculations:

### For Metallic Ferromagnets (Fe, Co, Ni, etc.)

**Recommended**: PBE
- Provides good agreement with experimental spin magnetization
- Computational efficiency superior to meta-GGA functionals
- **Be cautious**: r2SCAN can overestimate magnetic moments in itinerant ferromagnets

**Example**: Bulk Fe experimental value = 2.2 μB/atom
- PBE prediction: ~2.15 μB/atom

### For Transition Metal Oxides and Strongly Correlated Systems

**Recommended**: PBE + U via custom CP2K input (not yet automated in the CLI)

**When to use PBE+U**:
1. **Localized d-electrons**: Systems where d-electrons exhibit strong correlation (NiO, CoO, Fe₂O₃)
2. **Band gap issues**: Standard PBE underestimates band gaps significantly
3. **Magnetic ground states**: PBE gives incorrect magnetic ordering or underestimates moments

**U parameter selection**:
- U values are material-dependent and typically calibrated against experimental band gaps or magnetic moments
- Common values: U = 3-5 eV for 3d transition metals in oxides
- For NiO: U = 5-6 eV is typical
- Consult literature for your specific system

**Why standard GGA fails for oxides**:
- Self-interaction error artificially delocalizes d-electrons
- Underestimates band gaps and magnetic exchange interactions
- May predict incorrect magnetic ground states

## Constraints

- **DFT Engine**: Requires CP2K (`CP2K_DATA_DIR`). VASP parsing is retained for legacy atomate2 results.
- **Spin Polarization**: Pass `--spin-polarized` and `--initial-magmoms` to the CP2K runner.
- **Initial Magnetic Moments**: For complex magnetic ordering (e.g., antiferromagnetic), provide alternating `--initial-magmoms` values.
- **Environment**: The CP2K runner requires the `cp2k` pixi environment with ASE installed. Parsing works in `cp2k` or `base`.
- **Spin Density**: CP2K spin-density cube extraction is not yet implemented; use the legacy VASP path if 3D spin density is required.
- **Convergence**: Magnetic systems can be challenging to converge. If calculations fail to converge, try:
  - Adjusting `--initial-magmoms`
  - Increasing `--cutoff` or k-point density
  - Enabling +U for strongly correlated systems via custom CP2K input

## References

- CP2K Manual: [FORCE_EVAL / DFT / SPIN_POLARIZED](https://manual.cp2k.org/)
- Pymatgen Documentation: [Magnetic Structure Analysis](https://pymatgen.org/pymatgen.analysis.magnetism.html)

---
---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
