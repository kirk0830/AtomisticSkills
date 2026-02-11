---
name: molecular-fingerprints
description: Compute Morgan/ECFP fingerprints, Tanimoto similarity, and optional Butina clusters/heatmaps for small-molecule comparison.
category: drug-discovery
---

# Molecular Fingerprints

## Goal
To compute circular Morgan fingerprints (ECFP-style; default ECFP4 with radius=2) for a set of compounds, then calculate pairwise Tanimoto similarity for library comparison. Optionally perform Butina clustering for diversity analysis and generate a similarity heatmap for small sets.

This skill is commonly used for hit expansion, SAR triage, compound library diversity assessment, and applicability-domain style analysis.

## Instructions

### 1. Compute fingerprints + similarity matrix from a SMILES list
```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles "CCO" "CCCO" "c1ccccc1" "c1ccc(cc1)O" \
  --radius 2 --nbits 2048 \
  --output similarity.json
```

### 2. Read compounds from a SMILES file

The SMILES file format is:

* One molecule per line
* `SMILES[whitespace or tab]NAME` (NAME optional)
* Blank lines and `# comments` are ignored

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles_file compounds.smi \
  --radius 2 --nbits 2048 \
  --output similarity.json
```

### 3. Optional: standardize inputs (recommended for mixed sources)

Standardization helps reduce artifacts from salts/hydrates and inconsistent charge/tautomer forms.
Choose one of:

* `none`: no standardization (canonicalize only)
* `cleanup`: RDKit cleanup/normalization
* `parent`: cleanup + fragment parent selection (removes salts/counterions)
* `uncharged`: parent + uncharge
* `tautomer`: uncharged + canonical tautomer

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles_file compounds.smi \
  --standardize parent \
  --output similarity_parent.json
```

### 4. Optional: cluster compounds (Butina)

Use `--cluster` to run Butina clustering. Provide a **similarity cutoff** (more intuitive than Butina's distance threshold).

* Example: `--cluster_cutoff 0.7` means cluster using pairs with Tanimoto >= 0.7.

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles_file compounds.smi \
  --cluster --cluster_cutoff 0.7 \
  --output clustered.json
```

### 5. Optional: generate a similarity heatmap (small-to-medium sets)

Heatmaps become unreadable and slow for large n; this is intended for quick visual inspection.

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles_file compounds.smi \
  --heatmap similarity_heatmap.png \
  --output similarity.json
```

### 6. Optional: chirality and feature Morgan

* `--use_chirality`: distinguish stereoisomers (often important in drug-like datasets)
* `--use_features`: feature-based Morgan (FCFP-like), sometimes preferred for scaffold-level similarity

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles_file compounds.smi \
  --use_chirality \
  --output similarity_chiral.json
```

## Examples

Compare similarity of common NSAIDs (and save a heatmap):

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles \
    "CC(=O)Oc1ccccc1C(=O)O" \
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl" \
  --radius 2 --nbits 2048 \
  --heatmap nsaid_similarity.png \
  --output nsaid_similarity.json
```

## Guidance: Interpreting Similarity

Tanimoto similarity values are **not meaningful without specifying the fingerprint**. The same molecule pair can produce very different scores depending on the fingerprint type and parameters (e.g., esomeprazole vs. lansoprazole gives Tanimoto 0.79 with RDKit fingerprints but 0.43 with Morgan radius=2). Always report the fingerprint alongside the value.

### Noise floors (random molecule pairs)

Different fingerprints have different baseline similarity between unrelated molecules. Below the noise floor, similarity values carry no structural signal. Empirical 90th-percentile thresholds from 50K random ChEMBL pairs ([Landrum, 2021](https://greglandrum.github.io/rdkit-blog/posts/2021-05-18-fingerprint-thresholds1.html)):

| Fingerprint | Noise floor (p90) | Notes |
|---|---|---|
| Morgan r=2, 2048 bits | ~0.27 | Default in this skill |
| Morgan r=3, 2048 bits | ~0.13 | More selective (longer range) |
| MACCS keys (166 bits) | ~0.55 | High baseline (0.55 is noise, not signal) |
| RDKit topological | ~0.51 | Also high baseline |
| Feature Morgan r=2 | ~0.35 | `--use_features` mode |

### Recommended similarity search thresholds

For finding related compounds (e.g., same chemical series), empirical thresholds for ~95% recall based on 1,047 ChEMBL chemical series ([Landrum, 2021](https://greglandrum.github.io/rdkit-blog/posts/2021-05-21-similarity-search-thresholds.html)):

| Fingerprint | Threshold (~95% recall) | Notes |
|---|---|---|
| Morgan r=2 bits | **0.4** | Good default for hit expansion |
| Morgan r=3 bits | 0.3 | Lower because FP is more specific |
| MACCS keys | 0.65 | Must be high due to noise floor |

Lowering the threshold retrieves more true positives but also more noise. For a 100K library, Morgan r=2 at threshold 0.4 returns ~19 hits per query vs. ~2.5 at 0.55, an order of magnitude difference in result set size.

### Choosing radius and nBits

* **Radius 2** (ECFP4-equivalent) is the standard default for most drug discovery applications.
* **Radius 3** (ECFP6-equivalent) is more specific, useful when you want tighter structural matches but at the cost of lower recall.
* **nBits 2048** is a reasonable default. Increasing to 4096 or 8192 reduces bit collisions for large/diverse libraries but has diminishing returns for small sets.
* **Feature Morgan** (`--use_features`) encodes pharmacophoric features rather than atom identities, producing FCFP-like fingerprints that capture scaffold-level similarity.

### References

* [Fingerprint Generator Tutorial](https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html) - API, parameters, and output formats
* [Naming Similarity Metrics](https://greglandrum.github.io/rdkit-blog/posts/2025-07-17-naming-similarity-metrics.html) - why fingerprint type must accompany Tanimoto values
* [Fingerprint Thresholds Part 1](https://greglandrum.github.io/rdkit-blog/posts/2021-05-18-fingerprint-thresholds1.html) - noise floor analysis
* [Similarity Search Thresholds Part 2](https://greglandrum.github.io/rdkit-blog/posts/2021-05-21-similarity-search-thresholds.html) - empirical recall-based thresholds

## Constraints

* **Environment**: Requires `drugdisc-agent` conda environment.
* **Dependencies**: RDKit. Heatmap requires `numpy` and `matplotlib`.
* **Fingerprint**: Morgan (ECFP-style). Default radius=2 corresponds to ECFP4-style neighborhoods (per the ECFP literature).
* **Similarity metric**: Tanimoto/Jaccard on fingerprints (range [0,1]).
* **Clustering**: Butina clustering uses a pairwise distance list (distance = 1 - Tanimoto). Threshold choice is application-dependent.
* **Scaling**: Pairwise similarity is **O(n^2)**. Large libraries can be expensive in time/memory; prefer sampling or alternative large-scale clustering methods if n is very large.
* **Data hygiene**: Salts, mixtures, metal complexes, and inconsistent tautomer/protonation states can dominate similarity results. Use `--standardize` options deliberately.

Author: Matthew Cox
Contact: github username <mcox3406>
