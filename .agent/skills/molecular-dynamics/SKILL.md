---
name: molecular-dynamics
description: Best practices and tools for running stable Machine Learning Interatomic Potential (MLIP) molecular dynamics.
---

# Molecular Dynamics

## Goal
To perform stable and accurate molecular dynamics simulations using MLIPs, ensuring physical correctness and avoiding common "explosions" associated with neural network potentials.

## Instructions

### 1. Monitoring Stability
MD stability monitoring is integrated directly into the `run_md` tool via ASE callbacks. This ensures zero-latency response to instabilities and simplifies the simulation workflow.

- **Enable Monitoring**: Set `monitor=True` and specify `monitor_type` (single string or list).
    - `explosion`: Safety check. Stops if T > 10,000K or NaN. Recommended for all unstable simulations.
    - `equilibration`: Convergence check. Stops once temperature and potential energy stabilize (e.g., for production runs).
    - `overshoot`: Thermostat check. Stops if T deviates significantly from target (T-target > 200K).
    - `volume`: NPT stability check. Stops if volume expands by 2x or contracts to 0.2x of initial.

- **Example Usage**:
    ```python
    # MACE example with multiple monitors
    mace.run_md(structure, monitor=True, monitor_type=["explosion", "equilibration"])
    ```

- **Action**: When a monitor triggers, the simulation stops immediately with a `status: "stopped"` and a clear `stop_reason` in the result dictionary.

### 2. Parameter Initialization
- **Time Step**: 
    - **2.0 fs**: Recommended for most systems without light elements (Hydrogen).
    - **0.5 - 1.0 fs**: Use for systems containing Hydrogen, or at very high temperatures (> 2000K) to maintain stability.
- **Thermostat**: Use a coupling constant (`taut`) around $100 \times \text{timestep}$.
- **Temperature Ramp**: Start at a low temperature (50K) and ramp to the target to avoid "shock" waves from initial overlaps.

### 3. Handling Instability
If a simulation explodes:
1.  **Reduce Timestep**: Try 0.5 fs.
2.  **Ramp Temperature**: Use a slower heating rate.
3.  **Check Potential**: Consider if the chemistry is within the training range of the foundation potential. If not, use the [mlip-training](../mlip-training/SKILL.md) skill.

## Examples

Monitoring a running simulation's log:
```bash
python .agent/skills/molecular-dynamics/scripts/monitor_md.py md_run.log
```

## Constraints
- **Termination**: If `monitor_md.py` or manual inspection shows an explosion, terminate the task and adjust parameters. Do not proceed with unstable trajectories.
- **Reporting**: Always report simulation parameters (ensemble, T, timestep, duration) in the research report.
