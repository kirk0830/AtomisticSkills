---
name: molecular-dynamics
description: Best practices and tools for running stable Machine Learning Interatomic Potential (MLIP) molecular dynamics.
---

# Molecular Dynamics

## Goal
To perform stable and accurate molecular dynamics simulations using MLIPs, ensuring physical correctness and avoiding common "explosions" associated with neural network potentials.

## Instructions

### 1. Monitoring Stability
Always monitor simulation stability in real-time.
- Start the [monitor_md.py](scripts/monitor_md.py) script in the background *before* starting the MD simulation.
    ```bash
    # Run in background with target temperature (e.g., 800K)
    python scripts/monitor_md.py <log_file> --target_temp 800 --interval 60 &
    ```
- **Detection**: The script monitors starting temperature ($T_{start}$) and current temperature ($T_{curr}$):
    - **Fluctuation**: Detects excessive fluctuations in the last 4 readings.
    - **Drift**: Alerts if $T_{curr}$ overshoots or undershoots $T_{target}$ by $>100K$ based on the heating/cooling direction.
    - **Explosion**: Detects NaNs or $T > 10,000K$.
- **Action**: If any `CRITICAL` alert is printed, the agent **MUST** terminate the MD process immediately.

### 2. Parameter Initialization
- **Time Step**: Default to 1.0 fs or 0.5 fs for high temperatures ($> 1000K$) or light elements (H, Li).
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
