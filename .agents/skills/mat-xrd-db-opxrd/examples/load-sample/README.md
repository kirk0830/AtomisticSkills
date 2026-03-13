# opXRD Dataset Example

This directory demonstrates how to use the `mat-xrd-db-opxrd` skill to programmatically download, instantiate, and visualize experimental data from the opXRD dataset.

## Goal
To retrieve an individual experimental XRD pattern from the >90k size dataset via its index, extract its associated metadata (such as crystal system or background information), and plot it as an image file.

## Expected Output
The script will save a `.png` file containing the diffractogram with `2θ (degrees)` and `Intensity (a.u.)` axes. It will also print the sample metadata to the console:

```text
Successfully saved plot for index 0 to .agents/skills/mat-xrd-db-opxrd/examples/load-sample/opxrd_sample.png

Metadata:
 - file: aimat-lab_opXRD_614b8a2e.xy
 - crystal_system: Monoclinic
 - space_group: P2_1/c
 - temperature: 298
 - wavelength: 1.5406
 - background_subtracted: False
```
*(Note: Actual metadata keys may vary depending on the specific pattern loaded from the database.)*

## Instructions
Run the following command from the project root:

```bash
# Env: xrd-agent
python .agents/skills/mat-xrd-db-opxrd/scripts/load_sample.py --index 0 --output .agents/skills/mat-xrd-db-opxrd/examples/load-sample/opxrd_sample.png
```
