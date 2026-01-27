---
name: melting-point
description: Calculate the melting temperature of a material using the solid-liquid interface (coexistence) method.
---

# Melting Point

## Goal
To determine the thermodynamic melting temperature ($T_m$) of a bulk material by equilibrating a solid-liquid interface in an NVE ensemble.

## Instructions

1.  **Property Research**: 
    - Search for the approximate melting point ($T_m$) and boiling/evaporation point ($T_{vap}$) of the material.
    - Choose a melting temperature $T_{melt}$ where $T_m < T_{melt} \ll T_{vap}$.
    - **Example**: For Aluminum, $T_m \approx 933$ K, $T_{vap} \approx 2743$ K. A safe $T_{melt}$ is around 2000 K.

2.  **Phase Preparation**: 
    - **Solid**: Create a supercell using `create_supercell.py`.
    ```bash
    python scripts/create_supercell.py [input_structure.cif] [solid_supercell.cif] --min_length 20.0
    ```
    - **Liquid**: Melt a block using 1D-NPT (with mask) to ensure matching dimensions.
    ```bash
    mcp_mace_run_md(
        structure_data="solid_supercell.cif",
        temperature=2000,  # REPLACE with T_melt from Step 1
        ensemble="npt",
        steps=5000,
        pressure=1.0,      # Apply positive pressure (1-2 bar) to prevent evaporation
        pressure_mask=[1, 0, 0], # REQUIRED: Must match the stacking axis (e.g., x-axis)
        output_dir="melt_stage"
    )
    ```
3.  **Interface Creation**: Use `create_interface.py` to concatenate the two phases.
    ```bash
    python scripts/create_interface.py solid.cif liquid.cif --axis 0 --output interface.cif
    ```
4.  **Relaxation**: Perform an ionic relaxation using the `relax_structure` MCP tool with `relax_cell=True`. This allows the unit cell to adjust (shrink/expand) to match the density, and remove the interface energy created by stacking the two cells.
    ```bash
    mcp_mace_relax_structure(structure_data="interface.cif", relax_cell=True)
    ```
5.  **Production**: Run an NVE MD simulation starting near the expected $T_m$ (e.g., 933K for Al) using the MCP tool.
    ```bash
    mcp_mace_run_md(structure_data="relaxed_structure.cif", temperature=933, ensemble="nve", steps=50000, output_dir="production_md")
    ```
6.  **Monitoring**: **Concurrent with Step 5**, run the monitor script in a separate terminal to check for convergence and automatically terminate the simulation.
    ```bash
    python scripts/monitor_melting.py production_md/production_nve.log --window 1.0 --stability_duration 5.0
    ```
6.  **Monitoring**: **Concurrent with Step 5**, run the monitor script in a separate terminal to check for convergence and automatically terminate the simulation.
    ```bash
    python scripts/monitor_melting.py production_md/production_nve.log --window 1.0 --stability_duration 5.0
    ```
    - `mcp_mace_run_md` spawns a dedicated worker process for the simulation.
    - It writes the worker's PID to `production_md/MD.pid`.
    - The monitor script reads this PID and kills the specific worker process when stability is reached.
    - This ensures the main MCP server remains active.

7.  **Phase Validation**: Verify that the solid and liquid phases still coexist at the end of the simulation.
    ```bash
    # Note: If MD was killed, use the last frame from trajectory or the restart file if available.
    python scripts/check_phase.py production_md/final_structure.cif
    ```
    - **Fully Solidified**: The NVE starting temperature was too low.
    - **Fully Melted**: The NVE starting temperature was too high.
7.  **Analysis**: If coexistence is verified, calculate $T_m$ by averaging the temperature over the last 5 ps of the simulation.



## Examples

Creating a solid-liquid interface for Aluminum:
```bash
python scripts/create_interface.py Al_solid.cif Al_liquid.cif --axis 0 --output Al_interface.cif
```

## Constraints
- **Box Dimensions**: The lattice parameters perpendicular to the stacking axis must be identical for both solid and liquid blocks.
- **Ensemble**: The final production run must be in the **NVE** ensemble to allow the temperature to evolve to $T_m$.
- **Environments**: Different MLIPs require specific Conda environments (e.g., `mace-agent`, `matgl-agent`). Ensure the scripts are run within the correct environment for the chosen model.
