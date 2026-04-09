---
name: drug-mmgbsa
description: Compute single-trajectory MM-GBSA binding free energy estimates from a protein-ligand MD trajectory using OpenMM GBn2 implicit solvent.
category: [drug-discovery]
---

# drug-mmgbsa

## Goal

To estimate relative binding free energies from MD trajectories using the single-trajectory MM-GBSA approach. For each trajectory frame, the method strips explicit solvent, evaluates potential energies of the complex, receptor, and ligand subsystems in GBn2 implicit solvent, and computes:

    dG = E_complex - E_receptor - E_ligand

The entropy term (-TdS) is omitted, which is standard practice when the goal is relative ranking rather than absolute binding affinity. MM-GBSA is most useful for re-ranking docked poses after MD refinement, providing an orthogonal signal to docking scores and geometric stability metrics.

## Instructions

### 1. Basic usage (protein + ligand, short HTVS-style MD)

For the 1-5 ns production runs typical in the [HTVS workflow](../../workflows/drug-hit-finding-htvs.md):

```bash
# Env: drugmd-agent
python .agents/skills/drug-mmgbsa/scripts/compute_mmgbsa.py \
  --topology md/system/complex_solvated.pdb \
  --trajectory md/run/production.dcd \
  --ligand_sdf md/ligand.sdf \
  --ligand_resname UNL \
  --skip_ns 0.5 \
  --stride 5 \
  --output_dir md/mmgbsa/
```

Key parameters:
- `--topology`: solvated complex PDB from [drug-complex-system-builder](../drug-complex-system-builder/SKILL.md)
- `--trajectory`: production DCD from [drug-protein-ligand-md](../drug-protein-ligand-md/SKILL.md)
- `--ligand_sdf`: ligand SDF with correct bond orders (for OpenFF parameterization and AM1-BCC charges)
- `--ligand_resname`: residue name of the ligand in the PDB topology (default: UNL)
- `--skip_ns`: skip the first N nanoseconds of trajectory as equilibration (default: 0.5). For longer production runs, increase proportionally: roughly 10% of production length is a reasonable rule of thumb, capped at ~5 ns for 50+ ns runs.
- `--stride`: evaluate every Nth frame after skipping (default: 5)
- `--solute_dielectric`: interior dielectric of the solute (default: 1.0). See section 3 below on when to raise this.

### 2. With a cofactor (e.g., NADPH) and longer MD

When the receptor has a bound cofactor that should be part of the receptor subsystem, and the MD production was long enough (>=10 ns) to justify a longer equilibration skip:

```bash
# Env: drugmd-agent
python .agents/skills/drug-mmgbsa/scripts/compute_mmgbsa.py \
  --topology md/system/complex_solvated.pdb \
  --trajectory md/run/production.dcd \
  --ligand_sdf md/ligand.sdf \
  --cofactor_sdf md/nadph.sdf \
  --ligand_resname UNL \
  --cofactor_resname NDP \
  --skip_ns 2.0 \
  --output_dir md/mmgbsa/
```

### 3. Tuning the solute dielectric

The interior dielectric of the solute (`--solute_dielectric`, OpenMM's `soluteDielectric` parameter) is a major knob for MM-GBSA rankings and is often the difference between sensible and nonsense results. The default of 1.0 is the most physically pure choice (treats the solute as a vacuum) and works well for **nonpolar binding sites with few charged residues**. For **polar or highly charged pockets**, raise it:

| Binding site character | Suggested `--solute_dielectric` |
|---|---|
| Mostly hydrophobic, few ionizable residues | 1.0 (default) |
| Mixed polar/nonpolar | 2.0 |
| Many charged residues (Asp/Glu/Lys/Arg clusters, salt bridges) | 4.0 |

Higher dielectrics damp electrostatic contributions and generally improve ranking agreement with experiment for charged systems, at the cost of losing sensitivity to directional electrostatic interactions. If you are unsure, run the rescoring at 1.0 and 2.0 on a small subset with known rank-order and pick the value that tracks better. The chosen value is recorded in `mmgbsa_summary.json` for provenance.

```bash
# Env: drugmd-agent
python .agents/skills/drug-mmgbsa/scripts/compute_mmgbsa.py \
  --topology md/system/complex_solvated.pdb \
  --trajectory md/run/production.dcd \
  --ligand_sdf md/ligand.sdf \
  --ligand_resname UNL \
  --solute_dielectric 2.0 \
  --output_dir md/mmgbsa/
```

### 4. Custom selections

For non-standard residue naming or multi-chain receptors:

```bash
# Env: drugmd-agent
python .agents/skills/drug-mmgbsa/scripts/compute_mmgbsa.py \
  --topology md/system/complex_solvated.pdb \
  --trajectory md/run/production.dcd \
  --ligand_sdf md/ligand.sdf \
  --ligand_sel "resname UNK and not name H*" \
  --output_dir md/mmgbsa/
```

### 5. Interpret results

The script outputs:

- `mmgbsa_frames.csv`: per-frame energies (E_complex, E_receptor, E_ligand, dG)
- `mmgbsa_summary.json`: mean, std, SEM of dG, and the dielectric/frame parameters used

```json
{
    "dG_mean_kcal_mol": -42.5,
    "dG_std_kcal_mol": 8.3,
    "dG_sem_kcal_mol": 1.2,
    "n_frames_evaluated": 50,
    "solute_dielectric": 1.0,
    "solvent_dielectric": 78.5
}
```

More negative dG indicates stronger predicted binding. **Only relative ranking within a congeneric series is interpretable.** Absolute dG values from single-trajectory MM-GBSA without entropy corrections are not binding free energies in any physical sense, and specific numbers should not be reported as "predicted affinities" or compared across chemically distinct scaffolds. When ranking compounds:

- Group comparisons by formal charge state (see Constraints below).
- Compare mean dG values, but also inspect the per-frame trace in `mmgbsa_frames.csv` for stability. A compound with a noisy or drifting dG signal is less trustworthy than one with a tight distribution, even if the mean looks favorable.
- Use MM-GBSA as an additional signal alongside geometric stability (RMSD, contact persistence) from `drug-trajectory-analysis`, not as a standalone ranking.
- **Compare replicate means, not within-trajectory SEMs.** The honest uncertainty for a compound is the spread across independent MD replicates, not the SEM from any single one. See [examples/README.md](examples/README.md) for a real CDK2 case study that demonstrates this: one compound in the example shows a 35 kcal/mol swing across three MD replicates while each individual replicate reports a tight single-run SEM. Trusting any single replicate on that compound would be deeply misleading.

### 6. Integration with the HTVS workflow

In the [HTVS workflow](../../workflows/drug-hit-finding-htvs.md), MM-GBSA is run after MD refinement (Stage 8) as an additional ranking signal. Combined with trajectory stability metrics (RMSD, H-bonds, contacts from [drug-trajectory-analysis](../drug-trajectory-analysis/SKILL.md)), it provides a more complete picture of binding quality than docking scores alone.

## Constraints

- **Environment**: Requires `drugmd-agent`.
- **Dependencies**: openmm, openmmforcefields, openff-toolkit (for SMIRNOFF parameterization), rdkit, MDAnalysis.
- **Implicit solvent**: Uses GBn2 (Generalized Born with neck correction, model 2). NoCutoff nonbonded method.
- **Force fields**: Protein: Amber ff14SB. Ligand: OpenFF 2.2.0 (Sage) with AM1-BCC charges. These must match the explicit-solvent MD force fields.
- **Single-trajectory approximation**: Receptor and ligand conformations are extracted from the complex trajectory. This neglects reorganization entropy but is standard for ranking.
- **No entropy correction**: The -TdS term is omitted. This is acceptable for relative ranking but means absolute dG values are not comparable to experimental binding affinities.
- **Reported SEM is optimistic**: `dG_sem_kcal_mol` is computed as `std / sqrt(n_frames)`, which assumes independent samples. Frames from a single trajectory are correlated, so the true statistical error is larger (often by 2-5x for typical HTVS MD lengths). For honest uncertainties, run multiple independent replicates (different seeds) and compare across them rather than trusting the single-run SEM. See Genheden & Ryde (2010) for a detailed treatment of block averaging and replicate-based error estimation.
- **Charged ligands are a known weakness**: MM-GBSA's treatment of charge desolvation penalties is approximate, and ranking accuracy degrades as the ligand formal charge magnitude grows. Compare compounds within their own charge state rather than across charge states; a dG comparison between a neutral and a dication is not meaningful even within a congeneric series.
- **Electrostatic environment mismatch**: The explicit-solvent MD samples conformations under PME long-range electrostatics with a specific ionic strength. MM-GBSA re-evaluates those conformations under `NoCutoff` electrostatics in continuum solvent with no explicit ions. This is standard practice, but it is a real source of systematic error, especially for highly charged systems or binding sites near the protein surface where ionic screening matters.
- **Computation**: Runs on CPU (no GPU needed). Energy evaluation is fast (~1-5 minutes per compound depending on frame count).

## References

- Genheden, S.; Ryde, U. The MM/PBSA and MM/GBSA Methods to Estimate Ligand-Binding Affinities. *Expert Opin. Drug Discov.* **2015**, *10*, 449-461. [doi:10.1517/17460441.2015.1032936](https://doi.org/10.1517/17460441.2015.1032936)
- Genheden, S.; Ryde, U. How to Obtain Statistically Converged MM/GBSA Results. *J. Comput. Chem.* **2010**, *31*, 837-846. [doi:10.1002/jcc.21366](https://doi.org/10.1002/jcc.21366) (on block averaging and the need for replicate trajectories for honest error bars)
- Onufriev, A.; Case, D. A. Generalized Born Implicit Solvent Models for Biomolecules. *Annu. Rev. Biophys.* **2019**, *48*, 275-296. [doi:10.1146/annurev-biophys-052118-115325](https://doi.org/10.1146/annurev-biophys-052118-115325)
- Wang, E.; et al. End-Point Binding Free Energy Calculation with MM/PBSA and MM/GBSA: Strategies and Applications in Drug Design. *Chem. Rev.* **2019**, *119*, 9478-9508. [doi:10.1021/acs.chemrev.9b00055](https://doi.org/10.1021/acs.chemrev.9b00055)
- Roux, B.; Chipot, C. Editorial: Guidelines for Computational Studies of Ligand Binding Using MM/PBSA and MM/GBSA Approximations Wisely. *J. Phys. Chem. B* **2024**, *128*, 12027-12029. [doi:10.1021/acs.jpcb.4c06614](https://doi.org/10.1021/acs.jpcb.4c06614) (cautionary framing: end-point approximations like MM/GBSA are not a substitute for rigorous FEP/ABFE and should not be compared to experiment via a pre-established standard protocol)

---

**Author:** Matthew Cox
**Contact:** [GitHub @mcox3406](https://github.com/mcox3406)
