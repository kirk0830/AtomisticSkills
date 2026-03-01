---
name: chem-db-qmof
description: Query the Quantum MOF (QMOF) database via Materials Project's MPContribs platform for DFT-computed properties (bandgap) and optimized crystal structures of Metal-Organic Frameworks.
---

# Quantum MOF (QMOF) Database Search

This skill allows you to retrieve computational data and relaxed crystal structures (.cif) for roughly 20,000 Metal-Organic Frameworks (MOFs) that have been thoroughly characterized using DFT.

QMOF is officially hosted by the Materials Project under the "MPContribs" project (`qmof`).

## Use Cases
- Retrieving the relaxed CIF structure of a specific MOF if you know its reference name or CSD refcode (e.g., `IRMOF-1`, `HKUST-1`, `ZIF-8`, `KAXQIL`).
- Finding MOFs with specific DFT-computed properties (like a specific bandgap range).

## Requirements
- This tool requires `mpcontribs-client` to be installed in the `base-agent` environment.
- It requires the standard Materials Project API key exported as `MP_API_KEY`.

## Usage Instructions

To query the QMOF database programmatically using the `mpcontribs-client`, use the provided script:

```bash
conda run -n base-agent python .agent/skills/chem-db-qmof/scripts/query_qmof.py --formula "Zn" --max-results 5 --output-dir ./research/qmof_results
```

### Available Arguments:
- `--formula`: Search by chemical formula or elements (e.g., `Zn,O,C`).
- `--identifier`: Search by specific CSD refcode or MOF common name (e.g., `KAXQIL`).
- `--max-results`: Maximum number of structures to download (default: 5).
- `--output-dir`: Directory to save the resulting .cif files.

### Note on Performance
If you cannot find a MOF in QMOF via the script, or if the API query is overly complex, you can also search the QMOF database manually through the Materials Project web interface under the "Contributions" section.
