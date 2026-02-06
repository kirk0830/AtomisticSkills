---
name: kinetic-monte-carlo
description: Simulate long-time kinetics using rejection-free kinetic Monte Carlo (KMC): event catalog construction, rate assignment (TST/Arrhenius), detailed-balance validation, superbasin handling, and analysis of transport.
---

# Kinetic Monte Carlo (KMC)

## Goal
Run **kinetic Monte Carlo** simulations to evolve a system on **experimental (long) timescales** using a **continuous-time Markov jump process** defined by **elementary events** and their **rates**.

This skill focuses on best-practice, physics-grounded KMC:
- correct rejection-free time advancement (no time-step error),
- good event/rate bookkeeping,
- detailed balance / microreversibility checks when appropriate,
- practical handling of "flickers" / superbasins,
- and robust postprocessing (event stats, MSD -> diffusivity, Arrhenius).

This skill is designed to compose with:
- [neb-barrier](../neb-barrier/SKILL.md) (barriers for hops/reactions),
- [phonon](../phonon/SKILL.md) (attempt frequencies / hTST prefactors when feasible),
- [diffusion-analysis](../diffusion-analysis/SKILL.md) (MSD fitting and Arrhenius workflows),
- [ionic-conductivity](../ionic-conductivity/SKILL.md) (D -> sigma).

---

## When to Use KMC (and When Not)

### Use KMC when:
- Dynamics are **rare-event dominated** (activated hops/reactions separated by long waiting times).
- You can define a set of states + elementary transitions between states with rate constants.
- You need time/length scales unreachable by MD.

### Do NOT use KMC when:
- Motion is not rare-event-like (barriers ~ few kBT or less) and **recrossings** dominate.
- You cannot define a reasonably complete event set (or the event set changes too rapidly without on-the-fly discovery).
- The system is strongly non-Markovian at the state resolution you chose.

---

## Background

### Core Theory
KMC simulates a Poisson process over discrete events with rates {k_m}.

At a given state:
1. Compute total rate: R = sum_m k_m
2. Choose the next event m with probability k_m / R
3. Advance time by: dt = -ln(u) / R where u ~ Uniform(0,1)

This is equivalent to the Gillespie direct method / residence-time algorithm and the classic rejection-free "n-fold way" formulation.

Key property: No time-step bias; time is advanced by the correct exponential waiting-time distribution.

### Model-Building Choices

#### A) Lattice KMC (recommended if you can map to a lattice)
Best for:
- diffusion on a known sublattice (vacancy-mediated, intercalation on a site network),
- surface catalysis on discrete adsorption sites,
- ordering kinetics with local events.

Requires:
- a **site network** (graph: sites + neighbor relations),
- local **occupancy/state variables**,
- an **event catalog** (local patterns -> transitions).

#### B) Off-lattice / On-the-fly KMC (use for complex/disordered systems)
Best for:
- amorphous materials,
- heavily strained crystals,
- defect clusters, complex mechanisms,
- when you cannot predefine the event table.

Typical approaches:
- **AKMC** (saddle searches near current minimum + hTST rates),
- **k-ART** (topology-based self-learning off-lattice KMC),
- **SLKMC** (self-learning event discovery; often surfaces).

This skill provides guidance + validation criteria, but the included scripts implement **lattice KMC** (event table provided).

---

## Instructions

### 0. Define the Scientific Question and Minimal State Representation
Examples:
- Ion diffusion: state = occupancy of diffusion sites (carriers/vacancies).
- Surface microkinetics: state = coverage on adsorption sites.
- Defect aggregation: state = positions/connectivity of defects.

**Best practice**: choose the *coarsest* state that still makes the dynamics approximately Markovian.

### 1. Build the Site Network (Graph)
For lattice KMC you need:
- site coordinates (fractional/cartesian),
- periodic cell,
- neighbor list (including periodic image shifts).

**Validation**:
- neighbor graph is symmetric (if i neighbors j, ensure j neighbors i with opposite shift).
- hop distances are physically reasonable (cutoffs/NN shells).

```bash
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/build_lattice_from_structure.py \
    --structure relaxed.cif \
    --site_element Li \
    --cutoff 3.2 \
    --out lattice.json
```

### 2. Define Elementary Events (Event Catalog)
An "elementary event" must specify:
- precondition: local pattern (occupied -> empty neighbor, reactant adjacency, etc.)
- action: update the state (swap occupancy, change species, etc.)
- rate constant k(T)

**Best practice**:
- Always include **reverse events** (or verify they exist).
- Avoid hidden multi-step processes masquerading as "one event" unless properly coarse-grained.
- Track **degeneracy** explicitly (many symmetry-equivalent realizations).

### 3. Assign Rates in a Thermodynamically Consistent Way
Most atomistic KMC models use Arrhenius / transition-state theory:

k = nu(T) * exp(-dG_barrier(T) / kBT)

Common approximations:
- Use a constant attempt frequency nu ~ 1e12-1e13 s^-1 (document it).
- Use 0 K NEB barriers dE_barrier as dG_barrier (document missing entropic contribution).
- For high rigor: compute nu(T) via harmonic TST (Vineyard-type prefactor) and include free-energy corrections.

**Critical: detailed balance / microreversibility**

If your simulation is intended to reproduce equilibrium thermodynamics, rates must satisfy:

k_ij / k_ji = exp(-(F_j - F_i) / kBT)

At minimum (energy-only model):

k_ij / k_ji ~ exp(-(E_j - E_i) / kBT)

### 4. Validate the Event Table (Completeness + Correctness)
**Correctness checks**:
- reverse transitions exist
- detailed balance holds (if applicable)
- rates have correct units and magnitudes

```bash
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/validate_detailed_balance.py \
    --config kmc_config.json
```

**Completeness checks (modern best practice)**:
- Identify dominant events via sensitivity analysis (remove/perturb event types).
- Use on-the-fly discovery if needed.

If you see "stuck" behavior or unrealistically slow kinetics, the event catalog is likely incomplete.

### 5. Handle Flickers / Superbasins (Do NOT Ignore)
A common failure mode: the system executes extremely frequent **small-barrier back-and-forth transitions** ("flickers"), wasting steps without making physical progress.

Best-practice solutions include:
- superbasin / mean-rate / absorbing Markov chain acceleration methods,
- local superbasin methods,
- bac-MRM-style approaches (common in off-lattice/on-the-fly KMC).

At minimum:
- detect flickers (rapid repeated transitions among a small state set),
- report them in logs (so users know the model needs superbasin handling).

### 6. Run KMC with a Rejection-Free Engine
Use a rejection-free algorithm (residence-time / Gillespie / n-fold way):
- build list of enabled events + rates,
- sample event proportional to rate,
- advance time by exponential waiting time.

```bash
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/run_lattice_kmc.py \
    --config kmc_config.json
```

**Modern performance guidance**:
- Do NOT rescan the entire lattice each step for large models.
- Use local updates + a rate-sum data structure (Fenwick tree / heap / skip-list / event queue).
- Record RNG seed for reproducibility.

### 7. Postprocess: Transport + Mechanism
Typical outputs:
- event counts and residence times per event type,
- time series of MSD (for diffusion),
- Arrhenius fits across temperature.

```bash
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/analyze_kmc_msd.py \
    --trace kmc_run_T800K/kmc_trace.npz \
    --dim 3 \
    --out kmc_run_T800K/D_fit.json
```

---

## Helper Scripts

- **`build_lattice_from_structure.py`**: Builds a lattice site network from a crystal structure for vacancy/sublattice diffusion KMC. Uses ASE neighbor list with periodic image shifts.
- **`run_lattice_kmc.py`**: Rejection-free lattice KMC engine for carrier hops on a fixed site network. Implements local rate updates + Fenwick tree for O(log N) event sampling.
- **`validate_detailed_balance.py`**: Checks microreversibility for rate models and verifies neighbor graph is bidirectional with opposite shifts.
- **`analyze_kmc_msd.py`**: Computes tracer diffusivity (D_tracer), collective/charge diffusivity (D_J), and Haven ratio from KMC traces using the single-point Einstein relation D = MSD/(2dt). D_J is the physically relevant quantity for ionic conductivity via the Nernst-Einstein relation. Composable with `ionic-conductivity` workflows.

---

## Inputs/Outputs

### Input JSON (lattice diffusion example)
See `examples/kmc_config.example.json`.

The included engine supports:
- indistinguishable carriers on a site network
- hop event: occupied site i -> empty neighbor j
- two rate models:
  1. `constant`: k = nu * exp(-E_barrier / kBT)
  2. `symmetric_site_energy`: k = nu * exp(-(E0 + max(0, E_j - E_i)) / kBT) — enforces microreversibility if prefactors are equal.

### Outputs
- `kmc_trace.npz`: time, MSD, and carrier unwrapped positions (carrier_r_A, carrier_r0_A)
- `kmc_summary.json`: runtime metadata, step counts, rates, basic diagnostics
- `D_fit.json` (from `analyze_kmc_msd.py`): D_tracer (A^2/s, m^2/s, cm^2/s), D_J (collective diffusivity), Haven ratio, MSD values

---

## Examples

### Example A: Vacancy Diffusion on Li Sublattice
```bash
# 1) Build site network from relaxed structure
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/build_lattice_from_structure.py \
    --structure relaxed.cif \
    --site_element Li \
    --cutoff 3.2 \
    --out lattice.json

# 2) Validate detailed balance (if using site energies)
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/validate_detailed_balance.py \
    --config kmc_config.json

# 3) Run KMC
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/run_lattice_kmc.py \
    --config kmc_config.json

# 4) Analyze -> D_tracer, D_J, Haven ratio
# Env: base-agent
python .agent/skills/kinetic-monte-carlo/scripts/analyze_kmc_msd.py \
    --trace kmc_run_T800K/kmc_trace.npz \
    --dim 3 \
    --out kmc_run_T800K/D_fit.json
```

### Example B: Multi-Temperature Arrhenius from KMC
Run KMC at multiple temperatures, then use `ionic-conductivity` to convert D -> sigma:
```bash
# Env: base-agent
for T in 600 700 800 900 1000; do
    # update temperature in config, run KMC, analyze MSD...
    python .agent/skills/ionic-conductivity/scripts/compute_ionic_conductivity.py \
        --structure relaxed.cif \
        --temperature $T \
        --diffusivities "Li=${D_VALUE}" \
        --diffusion_units "cm2/s" \
        --charges "Li=1" \
        --out kmc_run_T${T}K/conductivity.json
done
```

---

## Common Pitfalls
1. **Wrong time**: using "MC sweeps" as time -> invalid. KMC time must come from exponential waiting-time steps.
2. **Incomplete event catalog**: missing dominant events -> wrong kinetics by orders of magnitude.
3. **Violating detailed balance**: produces unphysical steady states (when equilibrium is intended).
4. **Ignoring flickers**: step count explodes; kinetics appears "slow" but is just trapped in a superbasin.
5. **Barrier/prefactor inconsistency**: mixed methods (some barriers from NEB, others guessed) without validation.
6. **Finite-size effects**: too-small lattice gives biased diffusion and correlations.

---

## Constraints
- **Environments**: All scripts require the **base-agent** conda environment.
- **Lattice KMC only**: The included engine implements lattice KMC with fixed site networks. Off-lattice/on-the-fly KMC (AKMC, k-ART) is discussed but not implemented.
- **Rate models**: Two built-in rate models (`constant`, `symmetric_site_energy`). Custom rate models require extending the engine.
- **Barrier inputs**: Barriers are user-provided (from NEB, DFT, or literature). The scripts do not compute barriers.
- **Cluster expansion barriers**: For local cluster expansion (LCE) based rate models (e.g., NASICON-type systems), consider [kMCpy](https://github.com/caneparesearch/kMCpy) which natively integrates with fitted KECI and LCE event kernels.

---

## References
- Fichthorn & Weinberg, *J. Chem. Phys.* **1991**: theoretical foundations for dynamical/kinetic MC (Poisson process/master equation basis)
- Fichthorn & Lin, *J. Chem. Phys*. **2013**: "A local superbasin kinetic Monte Carlo method"
- Xu & Henkelman, *J. Chem. Phys.* **2008**: "Adaptive kinetic Monte Carlo for first-principles accelerated dynamics"
- Deng et al., "kMCpy: A python package to simulate transport properties in solids with kinetic Monte Carlo", *Comp. Mater. Sci.* **2023**. [doi.org/10.1016/j.commatsci.2023.112394](https://doi.org/10.1016/j.commatsci.2023.112394)
