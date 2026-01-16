---
trigger: model_decision
description: Rules to check when workflow involves molecular dynamics simulation
---

# Molecular Dynamics (MD) Best Practices

## Monitoring and Stability

When running MLIP Molecular Dynamics, especially at high temperatures, it is critical to monitor the simulation for stability. Instabilities can manifest as "explosions" where the temperature or potential energy rises uncontrollably.

### Critical Metrics to Monitor

1.  **Temperature (T)**:
    -   **Expectation**: Should fluctuate around the target temperature.
    -   **Warning Sign**: If T rises steadily or spikes to unreasonable values (e.g., > twice of the target temperature), the simulation has exploded.
    -   **Action**: TERMINATE immediately. Do not wait for completion.

2.  **Potential Energy (E_pot)**:
    -   **Expectation**: Should be negative and stable.
    -   **Warning Sign**: If E_pot becomes positive or extremely large, the atomic structure has likely disintegrated.
    -   **Action**: TERMINATE immediately.

### Parameter Tuning for Stability

If a simulation explodes, adjust the following parameters in order:

1.  **Time Step (`timestep`)**:
    -   **Default**: 2.0 fs (often okay for heavier atoms at low T).
    -   **High T / Light Atoms**: Reduce to **1.0 fs** or **0.5 fs**.
    -   **Reasoning**: High velocities at high T mean atoms move too far in a single step, causing overlaps and massive forces that break the integrator.

2.  **Temperature Coupling (`taut` or `ttime`)**:
    -   **Definition**: The time constant for the thermostat (Berendsen, Nose-Hoover, etc.).
    -   **Recommendation**: A value of `100 * timestep` is a good starting point (e.g., 100-200 fs).
    -   **Too Small (< 50 fs)**: Can cause oscillation artifacts.
    -   **Too Large (> 1000 fs)**: Temperature control is too loose; system may overheat before thermostat catches it.

3.  **Temperature**:
    -   **Initial Temperature**: Start near 0K or a low temperature (e.g. 50K) and ramp up, rather than initializing velocities instantaneously at 2000K. This prevents "shock" waves from initial random overlaps.
    -   **Target Temperature**: For simulation that doesn't need to be performed at the given temperature (sampling, melting, amorphoization, activation energy), use lower temperature MD for the simulation
