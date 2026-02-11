# Common Drugs ADMET Prediction Example

This example demonstrates batch physicochemical descriptor and drug-likeness heuristic computation for three well-known drugs: caffeine, aspirin, and ibuprofen.

## Files

- `compounds.smi`: Input SMILES file (tab-separated SMILES + name)
- `compounds_admet.json`: Full ADMET output including descriptors (MW, cLogP, TPSA, HBD/HBA, QED, etc.) and heuristic evaluations (Lipinski Ro5, Veber)

## How to reproduce

From the project root:

```bash
# Env: drugdisc-agent
python .agent/skills/admet-prediction/scripts/predict_admet.py \
    --smiles_file .agent/skills/admet-prediction/examples/compounds.smi \
    --output .agent/skills/admet-prediction/examples/compounds_admet.json
```

## Results

All three compounds pass Lipinski Ro5 and Veber heuristics with zero violations:

| Molecule | MW | cLogP | TPSA | HBD/HBA | QED | Ro5 | Veber |
|---|---|---|---|---|---|---|---|
| Caffeine | 194.2 | -1.03 | 61.8 | 0 / 6 | 0.54 | pass | pass |
| Aspirin | 180.2 | 1.31 | 63.6 | 1 / 3 | 0.55 | pass | pass |
| Ibuprofen | 206.3 | 3.07 | 37.3 | 1 / 1 | 0.82 | pass | pass |
