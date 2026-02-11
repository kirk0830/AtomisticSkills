# NSAID Fingerprint Similarity Example

This example computes ECFP4 Morgan fingerprints and pairwise Tanimoto similarity for three NSAIDs (aspirin, ibuprofen, diclofenac) plus caffeine as a negative control, with Butina clustering and a heatmap.

## Files

- `nsaid_similarity.json`: Full output including fingerprint metadata, pairwise similarity matrix, and Butina clusters
- `nsaid_similarity.png`: Tanimoto similarity heatmap

## How to reproduce

From the project root:

```bash
# Env: drugdisc-agent
python .agent/skills/molecular-fingerprints/scripts/compute_fingerprints.py \
  --smiles \
    "CC(=O)Oc1ccccc1C(=O)O" \
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O" \
    "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl" \
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" \
  --radius 2 --nbits 2048 \
  --cluster --cluster_cutoff 0.4 \
  --heatmap .agent/skills/molecular-fingerprints/examples/nsaid_similarity.png \
  --output .agent/skills/molecular-fingerprints/examples/nsaid_similarity.json
```

## Results

Pairwise Tanimoto similarity (ECFP4, 2048 bits):

|  | Aspirin | Ibuprofen | Diclofenac | Caffeine |
|---|---|---|---|---|
| Aspirin | 1.00 | 0.20 | 0.20 | 0.09 |
| Ibuprofen | 0.20 | 1.00 | 0.17 | 0.09 |
| Diclofenac | 0.20 | 0.17 | 1.00 | 0.08 |
| Caffeine | 0.09 | 0.09 | 0.08 | 1.00 |

At cutoff 0.4, Butina clustering places each compound in its own cluster (4 clusters), which is expected since these are structurally distinct scaffolds. Caffeine shows the lowest similarity to all three NSAIDs (0.08-0.09).
