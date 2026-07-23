---
name: mat-phonon
description: Calculate vibrational properties (phonon dispersions, density of states, thermal properties) using MLIPs.
category: [materials]
---

# Phonon Calculation Skill

This skill provides tools for calculating vibrational properties of materials using Machine Learning Interatomic Potentials (MLIPs).

## 1. Prerequisites / Environment Check

- The appropriate MLIP wrapper must be available (`MACEWrapper`, `MatGLWrapper`, or `FAIRCHEMWrapper`).
- `matcalc`, `phonopy`, and `phono3py` must be installed in the relevant pixi environment.
- `MP_API_KEY` (required for Option B — MP retrieval) — Required to retrieve pre-computed phonon data from Materials Project via `get_mp_phonon.py`. Without it, MP retrieval will fail. Get a free key at https://next-gen.materialsproject.org/api.

See `docs/api_key_guide.md` and `docs/environment_variables.md` for setup instructions.

Before running Option B, verify `MP_API_KEY` is set. If it is missing, ask the user to set it before proceeding.

## 2. Choosing a Foundation Potential

Phonon calculations are highly sensitive to the quality of the potential energy surface (PES).

> [!IMPORTANT]
> - **Use OMAT or MatPES trained models**: These models (e.g., `MACE-OMAT-0-small`, `TensorNet-MatPES-r2SCAN`) are specifically optimized for forces and vibrational stability.
> - **Avoid MPtrj-trained models**: Models trained primarily on the `MPtrj` dataset (e.g., `CHGNet-MPtrj`) suffer from the "softening" problem, where the calculated phonon frequencies are significantly lower than DFT values.

Refer to the [foundation-potentials skill](../ml-foundation-potentials/SKILL.md) for more details.

## 3. Calculation Workflow

### Option A: Calculate with MLIPs

To calculate phonon properties using machine learning potentials, use the `calculate_phonon.py` script.

```bash
pixi shell -e mace
python .agents/skills/mat-phonon/scripts/calculate_phonon.py \
    --structure path/to/relaxed_structure.cif \
    --model_type mace \
    --model_name MACE-MP-small \
    --supercell_matrix '[[2,0,0],[0,2,0],[0,0,2]]' \
    --output_dir research/my_folder/phonon
```

### Option B: Retrieve DFT Reference Data from Materials Project

For validation and benchmarking, retrieve pre-computed DFT phonon data:

```bash
# Env: base
python .agents/skills/mat-phonon/scripts/get_mp_phonon.py \
    --material_id mp-149 \
    --phonon_method dfpt \
    --output si_phonon_mp.json \
    --plot
```

**Available phonon methods**: `dfpt`, `phonopy`, `pheasy`

**When to use MP retrieval vs. MLIP calculations**:
- **Retrieve from MP**: Get DFT reference data for validation, benchmark MLIP accuracy
- **Calculate with MLIPs**: New materials, compare different MLIPs, high-throughput screening

### Validation Workflow: Compare MLIP vs DFT

```bash
# 1. Calculate with MLIP
python .agents/skills/mat-phonon/scripts/calculate_phonon.py \
    --structure Si.cif \
    --model_type mace \
    --model_name MACE-OMAT-0-small \
    --output_dir si_mace_phonon

# 2. Get DFT reference from MP
python .agents/skills/mat-phonon/scripts/get_mp_phonon.py \
    --material_id mp-149 \
    --phonon_method dfpt \
    --output si_mp_phonon.json \
    --plot

# 3. Compare phonon frequencies (manual inspection of plots)
#    - Check if MLIP frequencies match DFT
#    - Look for imaginary modes (structural instability)
#    - Validate thermal properties
```

## 4. Output Files

- `phonon_results.json`: Summary.
- `phonon.yaml`: Phonon data.
- `band_structure.yaml`: Band structure.
- `total_dos.dat`: Density of states.

## 5. Examples

See `examples/` for detailed usage scenarios.
---

**Author:** Bowen Deng
**Contact:** [GitHub @learningmatter-mit](https://github.com/learningmatter-mit)
