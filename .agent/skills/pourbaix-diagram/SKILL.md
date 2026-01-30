---
name: pourbaix-diagram
description: Calculate Pourbaix (pH-voltage) diagrams for aqueous electrochemical stability using water-corrected MLIP energies and pymatgen.
---

# Pourbaix Diagram

## Goal

To calculate thermodynamically consistent Pourbaix (pH-voltage) diagrams for assessing the aqueous electrochemical stability of materials. This skill uses Machine Learning Interatomic Potentials (MLIPs) for solid phase energies combined with Materials Project data for aqueous species, following the rigorous methodology of Persson et al. (2012)¹.

**Applications**:
- Alkaline-stable solid-state electrolytes (Li-air batteries)
- Corrosion-resistant materials
- Aqueous battery electrodes
- Electrochemical stability screening

## Background

### Pourbaix Diagrams

A Pourbaix diagram shows the thermodynamically stable phases as a function of pH and electrochemical potential (voltage vs. SHE). The diagram maps stability domains for solids and dissolved ions in aqueous environments.

### Critical: Thermodynamic Consistency

**The Challenge**: Mixing computational (MLIP/DFT) solid energies with experimental aqueous ion data creates energy scale mismatch.

**The Solution** (Persson et al. 2012)¹: **Water correction** that aligns MLIP water formation energy with experimental Gibbs free energy.
We use a robust cycle that fixes the hydrogen reference to the Standard Hydrogen Electrode (SHE) scale:

1. **Calculate $\mu_H$ from H₂ gas**:
   $$ \mu_H^{ref} \approx \frac{1}{2} (E_{H2}^{MLIP} + ZPE + \Delta H - TS)_{H2} $$
   Includes entropy ($S^\circ \approx 1.35$ meV/K) and enthalpy corrections.

2. **Derive $\mu_O$ from Water Equilibrium**:
   $$ \mu_O = G_{H2O}^{MLIP} - 2\mu_H^{ref} - \Delta G_f^{exp}(H_2O) $$
   where $\Delta G_f^{exp}(H_2O) = -2.4583$ eV.

This ensures:
- Hydrogen potential matches the standard SHE scale.
- Water formation energy is exactly -2.4583 eV.
- Oxygen potential absorbs any remaining MLIP/DFT errors (including O₂ errors).


**Is `calculate_water_correction.py` still needed?**  
**Yes**. This script is essential for calibrating the elemental chemical potentials ($\mu_H, \mu_O$) to the experimental water scale. Without it, the solid and ion energy scales would not be thermodynamically consistent.

## Instructions

### 1. Get Structures from Materials Project

Query all relevant solid phases and reference molecules:

```bash
# Env: base-agent
python .agent/skills/pourbaix-diagram/scripts/get_pourbaix_structures.py \
    --target_formula "Zn" \
    --output_dir ./structures/
```

This retrieves:
- All solid phases in the chemical system (e.g., Zn, ZnO, ZnO2, etc.)
- Reference molecules: H₂O, H₂, O₂ (for water correction)

### 2. Select Foundation Potential

Choose an appropriate MLIP based on your system (see `.agent/rules/foundation-potentials.md`).

**Recommended for Pourbaix diagrams**:
- **FairChem UMA**: `uma-s-1p1`.
- **MACE**: `MACE-MH-1` with `matpes` or `omol` head.
- **MatGL**: `CHGNet-MatPES-r2SCAN`.

**Rationale**: UMA `s`/`m` models provide excellent accuracy for both isolated molecules (H₂O, H₂, O₂) and solid phases. MatPES R2SCAN is also excellent for oxides.

### 3. Relax Reference Molecules (H₂O, H₂, O₂)

The script `get_pourbaix_structures.py` automatically copies standard reference molecule structures into the `references/` subdirectory of your output.

Relax these molecules to calculate water correction:

```python
# Example with FairChem UMA-small (recommended)
mcp_fairchem_load_model(model_name="uma-s-1p1")

# Batch relax references
mcp_fairchem_relax_structure(
    structure_data="./pourbaix_inputs/references/",  # Directory containing H2.cif, O2.cif, H2O.cif
    fmax=0.02,        
    steps=500,
    relax_cell=True,
    output_dir="./relaxed_molecules/"
)
```

**Critical**: Use `fmax ≤ 0.02 eV/Å` for reliable energies.

> [!WARNING]
> **O₂ Energy Issues with Some MLIPs**
> 
> Some MLIPs (e.g., UMA) may produce unphysical O₂ energies due to training data biases. For example, UMA-m-1p1 yields O₂ energy of +96 eV instead of the expected negative value, because its ORCA training data uses a different energy reference.
> 
> **No Action Required**: The `calculate_pourbaix.py` script automatically handles this by deriving the oxygen reference from H₂O and H₂ energies plus experimental water formation energy (ΔGf(H₂O) = -2.4583 eV), rather than directly using O₂. This ensures thermodynamic consistency regardless of MLIP O₂ accuracy.
> 
> See [`examples/Mn_pourbaix.md`](file:///home/bdeng/projects/AtomisticSkills/.agent/skills/pourbaix-diagram/examples/Mn_pourbaix.md) for a detailed example of this issue and its resolution.

### 4. Calculate Water Correction

Extract energies and calculate correction parameters:

```bash
# Get energies from MCP relaxation results
E_H2O=$(python -c "import json; print(json.load(open('./relaxed_molecules/H2O/result.json'))['energy'])")
E_H2=$(python -c "import json; print(json.load(open('./relaxed_molecules/H2/result.json'))['energy'])")
E_O2=$(python -c "import json; print(json.load(open('./relaxed_molecules/O2/result.json'))['energy'])")

# Calculate water correction (Env: base-agent)
python .agent/skills/pourbaix-diagram/scripts/calculate_water_correction.py \
    --h2o_energy $E_H2O \
    --h2_energy $E_H2 \
    --o2_energy $E_O2 \
    --output ./water_correction.json
```

This generates `water_correction.json` with μ_H^ref and μ_O values.

**Expected output**:
- ΔGf(H₂O)_MLIP ≈ -2.3 to -2.7 eV (should be close to -2.4583 eV experimental)
- Correction error < 0.5 eV indicates reasonable MLIP accuracy

### 5. Relax Solid Phases

Relax all solid phases using the **same MLIP**:

```python
# Batch relax all solids with same MLIP
# (Model already loaded from step 3)
mcp_mace_relax_structure(
    structure_data="./structures/solids/",  # Directory with all CIF files
    relax_cell=True,
    fmax=0.02,
    steps=500,
    output_dir="./relaxed_solids/"
)
```

**Critical**: Use the **same MLIP model** for molecules and solids!

### 6. Calculate Pourbaix Diagram

Construct the diagram with water-corrected energies:

```bash
# Env: base-agent
python .agent/skills/pourbaix-diagram/scripts/calculate_pourbaix.py \
    --relaxed_solids ./relaxed_solids/ \
    --water_correction ./water_correction.json \
    --target "ZnO" \
    --output ./pourbaix_results/
```

**Parameters**:
- `--relaxed_solids`: Directory with MLIP-relaxed solid structures
- `--water_correction`: Water correction JSON from step 4
- `--target`: Target material formula
- `--ion_concentration`: Ion concentration in M (default: 1e-6)
- `--output`: Output directory

**Note**: Requires `MP_API_KEY` environment variable for aqueous ion data.

### 7. Interpret Results

The script generates:
- `pourbaix_diagram.png`: Visual representation
- `stable_entries.json`: List of all stable phases in the diagram
- Stability domains for each phase across pH/V space

**Interpretation**:
- **Solid domain**: Material is thermodynamically stable as a solid
- **Ion domain**: Material dissolves into aqueous ions
- **Passivation**: Material covered by protective oxide layer

## Complete Example: ZnO Stability

```bash
# Step 1: Get structures (filters lowest energy polymorphs, copies references)
python .agent/skills/pourbaix-diagram/scripts/get_pourbaix_structures.py \
    --target_formula "ZnO" \
    --output_dir ./ZnO_project/

# Step 2: Load MLIP (UMA-small recommended)
mcp_fairchem_load_model(model_name="uma-s-1p1")

# Step 3: Relax references (Batch mode)
mcp_fairchem_relax_structure(
    structure_data="./ZnO_project/references/",
    fmax=0.02,
    steps=500,
    relax_cell=True,
    output_dir="./ZnO_relaxed_refs/"
)

# Step 4: Calculate water correction
# (Extract energies from results...)
E_H2O=$(python -c "import json; print(json.load(open('./ZnO_relaxed_refs/H2O.cif/result.json'))['energy'])")
# ... extract H2, O2 similarly ...

python .agent/skills/pourbaix-diagram/scripts/calculate_water_correction.py \
    --h2o_energy $E_H2O --h2_energy $E_H2 --o2_energy $E_O2 \
    --output ./ZnO_correction.json

# Step 5: Relax solids
mcp_fairchem_relax_structure(
    structure_data="./ZnO_project/structures/",
    fmax=0.02,
    steps=500,
    relax_cell=True,
    output_dir="./ZnO_relaxed_solids/"
)

# Step 6: Calculate Pourbaix
python .agent/skills/pourbaix-diagram/scripts/calculate_pourbaix.py \
    --relaxed_solids ./ZnO_relaxed_solids/ \
    --water_correction ./ZnO_correction.json \
    --target "ZnO" \
    --output ./ZnO_pourbaix/
```

**Expected result**: Multiple stability domains (Zn metal, ZnO, Zn²⁺, ZnO₂²⁻, etc.)

## MLIP-Agnostic Workflow

This skill works with **any MLIP** (MACE, MatGL, FairChem):

**MACE Example**:
```python
mcp_mace_load_model(model_name="MACE-OMAT-0-small")
mcp_mace_relax_structure(...)
```

**MatGL Example**:
```python
mcp_matgl_load_model(model_name="CHGNet-MatPES-PBE-2025.2.10-2.7M-PES")
mcp_matgl_relax_structure(...)
```

**FairChem Example**:
```python
mcp_fairchem_load_model(model_name="uma-s-1p1")
mcp_fairchem_relax_structure(...)
```

**Critical**: Use the **same MLIP** for all structures in a single workflow!

## Constraints

- **Energy Consistency**: All structures (molecules + solids) MUST be relaxed with the **same MLIP model**.
- **Water Correction**: REQUIRED for thermodynamic consistency. Skipping water correction will produce incorrect diagrams.
- **Convergence**: Use `fmax ≤ 0.02 eV/Å` for both molecules and solids.
- **Chemical System**: Automatically determined from target material and MP ion data.
- **Temperature**: Calculations assume 298 K (25°C).
- **Reference Electrode**: Results are vs. Standard Hydrogen Electrode (SHE).
- **Materials Project Access**: Requires `MP_API_KEY` environment variable.
- **DFT Validation**: For publication-quality results, validate critical stability boundaries with DFT.

## Theoretical Background

Following Persson et al. (2012)¹, the grand Pourbaix potential is:

```
ϕ_pbx = G - μ_H·N_H - μ_O·N_O - eV·Q
```

The water correction aligns MLIP solid energies (computational scale) with MP aqueous ion free energies (experimental scale) by enforcing:

```
ΔGf(H₂O)_corrected = -2.4583 eV (experimental)
```

This allows thermodynamically consistent mixing of:
- MLIP total energies for solids
- MP Gibbs free energies for aqueous ions

## References

1. **K. A. Persson, B. Waldwick, P. Lazic, G. Ceder.** "Prediction of solid-aqueous equilibria: Scheme to combine first-principles calculations of solids with experimental aqueous states." *Physical Review B* **85**, 235438 (2012). [DOI:10.1103/PhysRevB.85.235438](https://doi.org/10.1103/PhysRevB.85.235438)

2. **Hierarchical screening methodology**: [arXiv:2511.20964](https://arxiv.org/abs/2511.20964)

3. **Pymatgen Pourbaix module**: https://pymatgen.org/pymatgen.analysis.pourbaix_diagram.html

4. **Related skills**: 
   - [material-stability](../material-stability/SKILL.md) for convex hull calculations
   - [elemental-energies](../elemental-energies/SKILL.md) for retrieving pre-calculated elemental reference energies required for formation energy calculations.
