---
name: ionic-conductivity
description: Compute ionic conductivity from MD-derived tracer diffusivities via the Nernst-Einstein relation with correlation diagnostics and Arrhenius analysis.
---

# Ionic Conductivity

## Goal
To estimate the **dc ionic conductivity** $\sigma(T)$ of an ion-conducting material by converting tracer diffusivities $D^*(T)$ (from `diffusion-analysis`) into conductivity via the **Nernst-Einstein (NE) relation**, with best-practice handling of carrier counting, unit consistency, and **correlation caveats** (Haven ratio).

This skill composes with:
- [molecular-dynamics](../molecular-dynamics/SKILL.md) (how to run stable MD),
- [diffusion-analysis](../diffusion-analysis/SKILL.md) (how to compute $D(T)$ robustly),
- [material-stability](../material-stability/SKILL.md) / [pourbaix-diagram](../pourbaix-diagram/SKILL.md) (for screening campaigns).

---

## Background

### Nernst-Einstein Conductivity
For a single dominant mobile species the NE relation is:

$$\sigma_{\mathrm{NE}} = \frac{N q^2}{V k_B T} D^*$$

where $N$ = number of mobile carriers in the simulation cell, $V$ = average cell volume, $q = z \cdot e$ (carrier charge), $T$ = temperature, $k_B$ = Boltzmann constant, $D^*$ = tracer diffusivity.

For **multiple mobile species**:

$$\sigma_{\mathrm{NE}} = \sum_i \frac{n_i q_i^2}{k_B T} D_i^*$$

with $n_i = N_i / V$ (number density).

**Important**: NE assumes effectively uncorrelated carriers. Many fast conductors show correlated motion and NE can misestimate $\sigma$.

### Haven Ratio (Correlation Diagnostic)
The Haven ratio quantifies deviation from NE:

$$H_R = \frac{D^*}{D_\sigma}$$

where $D_\sigma = \sigma k_B T / (n q^2)$ is the conductivity diffusion coefficient. With this convention $\sigma = \sigma_{\mathrm{NE}} / H_R$.

**$H_R$ can be greater or less than 1** depending on the transport mechanism:
- $H_R > 1$: Cross-correlations *suppress* charge transport (e.g., cation-anion drag in electrolytes, blocking effects). NE *overestimates* $\sigma$.
- $H_R < 1$: Correlated motion *enhances* charge transport (e.g., concerted multi-ion hopping in superionic conductors like LGPS, argyrodites). NE *underestimates* $\sigma$. Values of $H_R \approx 0.3$--$0.5$ are common in fast Li-ion conductors.
- $H_R = 1$: Uncorrelated motion; NE is exact.

**Convention warning**: Some references define $H_R$ inversely ($D_\sigma / D^*$). This skill uses $H_R = D^* / D_\sigma$ throughout.

If you can compute $\sigma$ from a correlation-aware method (collective Einstein MSD or Green-Kubo charge-current autocorrelation), report $H_R$ and use the correlation-aware $\sigma$ as the primary result. Otherwise, report $\sigma_{\mathrm{NE}}$ and explicitly state the NE assumption.

**Convergence caveat**: A well-converged $D^*$ does *not* guarantee a reliable $\sigma$. The off-diagonal (cross-correlation) terms that distinguish true $\sigma$ from $\sigma_{\mathrm{NE}}$ have much higher variance than the diagonal (self-diffusion) terms. Converging the collective conductivity typically requires $\sim$30--40 independent trajectories, whereas $D^*$ may converge from a single long run. If only NE is reported, flag this limitation explicitly.

---

## Instructions

### 1. Prerequisites
You need:
- A relaxed structure and MD trajectories at one or more temperatures.
- Robust $D^*(T)$ estimates for the mobile species from [diffusion-analysis](../diffusion-analysis/SKILL.md), saved as `diffusion_results.json` files.

**Recommended**: compute $\sigma(T)$ at $\geq 3$ temperatures and fit Arrhenius, but check that the diffusion mechanism is consistent across $T$ (no phase transition or order-disorder change).

### 2. Run MD Valid for Transport
(See [molecular-dynamics](../molecular-dynamics/SKILL.md) for stability and monitors.)

Transport best practice:
1. **Equilibrate** (often NPT to relax density), then
2. **Production** in a minimally perturbing ensemble: NVE (ideal if stable) or weakly thermostatted NVT (e.g., `nvt_bussi`).

**Rule of thumb**: you need enough time for **multiple hops per carrier**; otherwise $D$ is noise-dominated and $\sigma$ will be unreliable.

### 3. Compute Tracer Diffusivity $D^*(T)$
Use [diffusion-analysis](../diffusion-analysis/SKILL.md):
```bash
# Env: base-agent
python .agent/skills/diffusion-analysis/scripts/analyze_diffusion.py \
    results/md_800K/trajectory.traj \
    --species Li \
    --temperature 800 \
    --ignore_ps 5.0 \
    --output_dir results/md_800K
```
This produces `results/md_800K/diffusion_results.json` with keys `diffusivity` (cm$^2$/s), `diffusivity_std_dev`, `temperature`, `species`.

### 4. Convert $D^*(T)$ to $\sigma_{\mathrm{NE}}(T)$
Use the `compute_ionic_conductivity.py` script. It reads `diffusion_results.json` files and the simulation structure to extract volume and carrier counts.

**From a single diffusion result:**
```bash
# Env: base-agent
python .agent/skills/ionic-conductivity/scripts/compute_ionic_conductivity.py \
    --structure results/relaxed.cif \
    --diffusion_json results/md_800K/diffusion_results.json \
    --charges "Li=1" \
    --out results/md_800K/conductivity.json
```

**Manual specification (e.g., from literature values):**
```bash
# Env: base-agent
python .agent/skills/ionic-conductivity/scripts/compute_ionic_conductivity.py \
    --structure results/relaxed.cif \
    --temperature 800 \
    --diffusivities "Li=1.0e-6" \
    --diffusion_units "cm2/s" \
    --charges "Li=1" \
    --out results/conductivity_800K.json
```

**With a Haven ratio estimate:**
```bash
# Env: base-agent
python .agent/skills/ionic-conductivity/scripts/compute_ionic_conductivity.py \
    --structure results/relaxed.cif \
    --diffusion_json results/md_800K/diffusion_results.json \
    --charges "Li=1" \
    --haven_ratio 1.3 \
    --out results/md_800K/conductivity.json
```

**Critical caveat (defects/doping)**: If the MD cell contains an artificially high defect concentration (common in small supercells), decide whether to report "as-simulated $\sigma$" or "scaled $\sigma$" using a realistic carrier concentration. Both are valid if clearly labeled.

### 5. Multi-Temperature Arrhenius Analysis
Once conductivity JSONs exist for multiple temperatures, fit the Arrhenius relation. The standard form for NE-derived conductivity is:

$$\ln(\sigma T) = \ln(A) - \frac{E_a}{k_B T}$$

```bash
# Env: base-agent
python .agent/skills/ionic-conductivity/scripts/fit_arrhenius_conductivity.py \
    results/md_*K/conductivity.json \
    --out results/arrhenius_conductivity.json
```

This produces:
- `arrhenius_conductivity.json` with $E_a$ (eV), prefactor, and per-point data.
- `arrhenius_conductivity_plot.png` with the fit and data points.

---

## Helper Scripts

- **`compute_ionic_conductivity.py`**: Reads a structure (for volume + species counts) and diffusivities (from JSON or CLI). Outputs $\sigma_{\mathrm{NE}}$ in S/m and S/cm to a JSON file.
- **`fit_arrhenius_conductivity.py`**: Reads multiple conductivity JSON files, fits $\ln(\sigma T)$ vs $1/T$, reports $E_a$ (eV) and generates a publication-quality Arrhenius plot.

---

## Examples

### Example A: Single-Carrier NE Conductivity (Li$^+$)
Given `diffusion_results.json` from a 800 K run of Li$_{10}$GeP$_2$S$_{12}$:
```bash
# Env: base-agent
python .agent/skills/ionic-conductivity/scripts/compute_ionic_conductivity.py \
    --structure LGPS_221.cif \
    --diffusion_json md_800K/diffusion_results.json \
    --charges "Li=1" \
    --out md_800K/conductivity.json
```

### Example B: Multi-Temperature Arrhenius
After running at 600, 700, 800, 900, 1000 K:
```bash
# Env: base-agent
python .agent/skills/ionic-conductivity/scripts/fit_arrhenius_conductivity.py \
    md_600K/conductivity.json md_700K/conductivity.json md_800K/conductivity.json \
    md_900K/conductivity.json md_1000K/conductivity.json \
    --out arrhenius_conductivity.json
```

---

## Validation Checklist
- MSD shows a clear linear diffusive regime (not just vibrations).
- Each mobile ion shows multiple hops; otherwise $D$ is not converged.
- No net drift / center-of-mass drift (or removed consistently).
- Size/time convergence is discussed (at least qualitatively).
- Report whether $\sigma$ is NE-only or correlation-aware.
- If defects/doping control carriers, clearly state the carrier concentration used.
- Units are consistent: $D$ in cm$^2$/s from `diffusion-analysis`, converted internally to m$^2$/s for SI.
- **NE ≠ true $\sigma$**: Even with well-converged $D^*$, $\sigma_{\mathrm{NE}}$ sidesteps the hardest statistical problem (converging off-diagonal cross-correlation terms). If true conductivity is needed, use the collective Einstein relation with $\sim$30--40 independent trajectories.

**Common failure modes:**
- "$D$ is ~0" -> simulation too short, $T$ too low, or structure too ordered.
- "$\sigma$ is huge" -> mis-specified units (cm$^2$/s vs m$^2$/s) or wrong charge.
- "$\sigma$ changes wildly with fit window" -> not in diffusive regime or insufficient sampling.

---

## Constraints
- **Environments**: All analysis scripts require the **base-agent** conda environment.
- **Trajectory Format**: Upstream diffusion analysis requires ASE `.traj` format trajectories.
- **Diffusion Input**: Scripts consume `diffusion_results.json` from [diffusion-analysis](../diffusion-analysis/SKILL.md) (units: cm$^2$/s).
- **Charge Convention**: Default to **formal ionic charges** (Li: +1, Na: +1, O: -2, etc.). Document any deviation.
- **Atom Count**: Upstream MD should contain sufficient mobile ions (> 20 recommended) for statistical significance.

---

## Key References
- [doi:10.1038/s41524-018-0074-y](https://doi.org/10.1038/s41524-018-0074-y)
- [doi.org/10.1103/PhysRevLett.122.136001](https://doi.org/10.1103/PhysRevLett.122.136001)
- [doi:10.1103/PhysRevB.64.184307](https://doi.org/10.1103/PhysRevB.64.184307)
- [doi:10.1021/acs.jpcb.0c07704](https://pubs.acs.org/doi/10.1021/acs.jpcb.0c07704)