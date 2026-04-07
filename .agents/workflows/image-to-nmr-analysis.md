---
description: Digitize an NMR spectrum image and quantify its mixture composition — combines plot digitization with reaction product prediction and Wasserstein deconvolution.
---

# Image to NMR Mixture Analysis

This workflow guides the agent through extracting quantitative composition data from a raw image of a 1H NMR spectrum.

**Scientific Problem:**
Researchers often only have access to 1H NMR spectra in image formats (e.g., from literature, old reports, or screenshots) rather than raw numeric FID or processed `.xy` text data. To computationally analyze these spectra (e.g., for reaction yield quantification or mixture deconvolution), the image must first be accurately digitized into numeric data, and then deconvoluted to determine the relative concentrations of components. This workflow is a pipeline combining `general-plot-digitizer` with downstream NMR prediction and Wasserstein deconvolution.

## Step-by-Step Methodology

### Step 1: Digitize Spectrum Image
Convert the provided spectrum image into numeric (ppm, intensity) data.

- **Skill:** Use the `general-plot-digitizer` skill.
- **Action:** Follow the digitizer's full VLM + CV pipeline to extract a two-column `.csv` (ppm, intensity) from the image.
- **Output:** A numeric `.csv` or `.xy` spectrum file ready for downstream steps.
- **Decision:** If the digitized spectrum shows artifacts (axis labels captured as peaks, grid lines), re-run with adjusted parameters before proceeding.

### Step 2: Identify All NMR-Relevant Species
Based on the reaction description or user input, compile a complete list of compounds whose signals may appear in the crude NMR.

- **Action:** Check reactants, products, solvents, and NMR-visible reagents. Resolve compound names to SMILES using the `drug-db-pubchem` skill.

### Step 3: Predict Reaction Products (If unknown)
Use ML forward prediction to determine what products formed if the user only provides reactants.

- **Skill:** Use the `chem-nmr-analysis` skill (`predict_products.py`).
- **Action:** Pass reactant and reagent SMILES to ReactionT5.
- **Output:** JSON file with predicted product SMILES.

### Step 4: Generate NMR Reference Spectra
Predict 1H NMR spectra for all identified species.

- **Skill:** Use the `chem-nmr-predict` skill.
- **Action:** Pass all collected SMILES to `predict_nmr.py`. Match `--field_mhz` to the user's spectrometer if known (often stated in the image or caption).
- **Output:** Real-valued `.xy` reference spectra and `_signals.csv` signal tables.

### Step 5: Visual Inspection
Overlay the digitized crude spectrum with all references to verify alignment.

- **Skill:** Use the `chem-nmr-analysis` skill (`plot.py`).
- **Action:** Generate an overlay plot and inspect it to ensure major peaks in the digitized image align with reference peaks.

### Step 6: Deconvolution
Quantify component mole fractions via Wasserstein-distance deconvolution.

- **Skill:** Use the `chem-nmr-analysis` skill (`deconvolve.py`).
- **Action:** Run `deconvolve.py` with the digitized crude spectrum, reference spectra, proton counts, and component names.
- **Output:** Estimated mole fractions, Wasserstein distance (fit quality), and a multi-panel deconvolution plot.

---

## Summary Checklist

```
[ ] 1. Digitize NMR image to .csv (`general-plot-digitizer`)
[ ] 2. Resolve species names to SMILES (`drug-db-pubchem`)
[ ] 3. Predict reaction products if needed (`chem-nmr-analysis`)
[ ] 4. Generate reference spectra (`chem-nmr-predict`)
[ ] 5. Plot overlay and inspect (`chem-nmr-analysis`)
[ ] 6. Deconvolve signals to mole fractions (`chem-nmr-analysis`)
```
