---
name: mat-xrd-db-opxrd
description: Load and access experimental powder X-Ray diffractograms from the Open Experimental Powder X-Ray Diffraction Database (opXRD).
category: [materials, machine-learning]
---

# mat-xrd-db-opxrd

## Goal
To programmatically download, instantiate, and analyze experimental powder X-Ray diffractograms using the opXRD Python dataset wrapper.

## Dataset Contents
The opXRD database is an extensive collection of real, experimental powder X-Ray diffraction patterns. It is split into two primary subsets:
- **Unlabeled subset (~90,000 diffractograms)**: Vast quantities of raw experimental patterns intended for unsupervised or semi-supervised transfer learning tasks.
- **Labeled subset (2,179 diffractograms)**: Fully labeled experimental patterns. These include metadata like explicit chemical composition, crystallographic classification (e.g., Space Group, Crystal System), explicit lattice parameters, and structural basis data.

Because these patterns are experimental, they inherently contain real-world instrumental artifacts (peak broadening, amorphous backgrounds, mixed phases, and noise) which make them exceptionally valuable for training predictive ML models compared to purely simulated baseline data.

## Instructions

1. **Environment Setup**
   Ensure you are using the `xrd-agent` conda environment, which has been configured with `opxrd` and its underlying compiler dependencies (Boost, SWIG, GCC).

2. **Loading the Dataset**
   `opxrd` serves as a wrapper around the dataset and PyTorch's `Dataset` class. To instantiate the dataset, import `opxrd` and create an instance using `load()`.
   ```python
   # Env: xrd-agent
   from opxrd import OpXRD as OpxrdDataset
   
   # Instantiate the dataset, which automatically downloads and extracts it if not present.
   dataset = OpxrdDataset.load('./research/opxrd_data', download=True)
   ```

3. **Interacting with Data**
   Each item in the dataset is represented as a structured `XrdPattern` object containing the experimental pattern and available metadata.
   ```python
   # Env: xrd-agent
   sample = dataset.patterns[0]
   
   # Extracts the coordinate arrays directly:
   intensities = sample.intensities
   two_theta = sample.two_theta_values
   
   # Built-in image export:
   # sample.plot(save_fpath="output_pattern.png")
   ```

## Examples

Loading and plotting a single experimental pattern from the dataset:
```bash
# Env: xrd-agent
python .agents/skills/mat-xrd-db-opxrd/scripts/load_sample.py --index 0 --output .agents/skills/mat-xrd-db-opxrd/examples/load-sample/opxrd_sample.png
```

For full details, expected outputs, and sample files, refer to the [load-sample example](examples/load-sample/README.md).

## Constraints
- **Environment**: Must strictly use the `xrd-agent` conda environment, as the underlying C++ libraries required by `opxrd` are mapped and installed there.
- **Architecture**: The `opxrd` Python library (specifically the `xrdpattern` dependency) hardcodes a pre-compiled `x86_64` binary of `_xylib.so` in its PIP distribution. **It will fail with an `ImportError` on ARM64 (`aarch64`) systems** (such as AWS Graviton or Apple Silicon) because it prevents standard source recompilation.
- **Data Size**: The full dataset contains over 90,000 diffractograms. Ensure adequate disk space when initializing `OpxrdDataset` for the first time.

## References
- B. S. Ganser et al., "opXRD: Open Experimental Powder X-Ray Diffraction Database", *Advanced Intelligent Discovery*, 2025. [DOI](https://doi.org/10.1002/aidi.202500044)

---

**Author:** Bowen Deng  
**Contact:** [GitHub @Bowen-BD](https://github.com/bowen-bd)
