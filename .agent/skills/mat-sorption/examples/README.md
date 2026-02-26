# mat-sorption examples

Run relax, Widom, and GCMC from the **project root** with the `fairchem-agent` environment active.

1. **Relax** a framework CIF; outputs are CIF/XYZ files and `relax_results.json` in `--output-dir`. Use a relaxed CIF path (e.g. `output_dir/name.relaxed.cif`) for Widom and GCMC.
2. **Widom**: Writes a single JSON to `output_dir/widom_results.json`. Use relaxed CIF as `--structure`.
3. **GCMC**: Writes `gcmc_results.json` plus `nmols.png`, `energy.png` to `--output-dir`. Use relaxed CIF as `--cif`.

See the [mat-sorption SKILL.md](../SKILL.md) for full workflow and parameters.
