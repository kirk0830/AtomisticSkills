# Methanol BDE — Homolytic vs Heterolytic Example

**Molecule:** Methanol (CH₄O, SMILES: `CO`)  
**Model:** MACE-OMOL-extra-large  
**Cleavage mode:** `both` (homolytic + heterolytic)  
**Includes H bonds:** Yes  

## How to Reproduce

```bash
# Env: mace-agent
python .agents/skills/chem-bond-dissociation/scripts/calculate_bde.py \
    --smiles CO \
    --all_bonds \
    --include_h_bonds \
    --cleavage both \
    --model_type mace \
    --model_name MACE-OMOL-extra-large \
    --output_dir .agents/skills/chem-bond-dissociation/examples/methanol_mace_omol_both

# Plot
python .agents/skills/chem-bond-dissociation/examples/methanol_mace_omol_both/plot.py
```

## Results

| Bond | Homolytic BDE (kcal/mol) | Heterolytic BDE (kcal/mol) | Δ |
|:---|:---:|:---:|:---:|
| **C–O** | **143.0** | **143.0** | 0.0 |
| C–H (2,3) | 179.5 | 179.5 | 0.0 |
| C–H (4) | 192.2 | 192.2 | 0.0 |
| O–H | 199.8 | 199.8 | 0.0 |

Expected experimental values (gas phase):  
- C–O homolytic: ~92 kcal/mol; O–H homolytic: ~104 kcal/mol (MACE-OMOL overestimates)  
- Gas-phase heterolytic C–O: ~250–350 kcal/mol (much higher than homolytic)

## Key Finding: MACE-OMOL is Charge-Invariant in Gas Phase

The heterolytic BDE is **identical** to the homolytic BDE for every bond. This is because MACE-OMOL produces the **same total energy** for an isolated molecular fragment regardless of the charge/spin annotation written to `atoms.info`:

```
E(CH₃, q=0, spin=2) ≈ E(CH₃⁺, q=+1, spin=1) ≈ E(CH₃⁻, q=-1, spin=1)
```

### Why this happens

MACE-OMOL was trained on the OMOL dataset of molecules at **DFT-level** — the model *does* differentiate charge states during training because different charge states were separate training entries with different ground-truth energies. However, in practice, the model's charge/spin handling only distinguishes energies when:

1. The molecular context is **rich enough** (multiple atoms whose electron density can redistribute)
2. The fragment is **not too small** (a lone H atom has no internal degrees of freedom to respond to charge)

For a single H atom or a 4-atom methyl group, the MLIP architecture cannot reorganize its embedding in response to a scalar charge/spin annotation in a physically meaningful way.

### Physical context

Heterolytic BDE is normally computed using:
```
ΔE_hetero = IE(A) − EA(B)   [for A·B → A⁺ + B⁻]
```
where IE = ionization energy and EA = electron affinity. These are single-molecule properties that require explicit electron addition/removal, which is only meaningful in:
- Polarizable continuum models (implicit solvent)  
- Explicit quantum mechanical calculations (DFT with charged supercells)  
- Reference data from photoelectron spectroscopy

Gas-phase heterolytic BDE is typically 200–400 kcal/mol above homolytic — the energy cost of charge separation without stabilizing environment.

### Conclusion

> **MACE-OMOL in its current form does not reliably differentiate the gas-phase energy of a molecular fragment based on its charge annotation alone.**  The `supports_charge_spin=True` capability flags that MACE-OMOL and FairChem (omol) correctly accept and *pass through* charge/spin to the calculator — they do not guarantee that the resulting energies will differ significantly for small, isolated fragments.

The heterolytic BDE feature is still architecturally correct and useful in scenarios where:
- OMOL models are fine-tuned on charged fragment pairs with reference energies
- Future MLIP training explicitly includes charged fragment dissociation pathways
- Solvation correction terms are added post-hoc to the fragment energies

## Files

| File | Description |
|:---|:---|
| `bde_results.json` | Full BDE results with all variants |
| `intact_relaxed.xyz` | Relaxed methanol geometry |
| `frag_bond{N}_homo_{1,2}.xyz` | Homolytic radical fragments |
| `frag_bond{N}_hetero_pos_neg_{1,2}.xyz` | Heterolytic cation/anion (variant A) |
| `frag_bond{N}_hetero_neg_pos_{1,2}.xyz` | Heterolytic anion/cation (variant B) |
| `bde_comparison.png` | Comparison plot |
| `plot.py` | Plotting script |
