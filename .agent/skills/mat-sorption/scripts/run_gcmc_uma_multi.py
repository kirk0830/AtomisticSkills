"""
Multicomponent BVT GCMC (N-component mixture) in a periodic framework using UMA (FairChem).

Same interface as COFclean/gcmc/uma_ase_gcmc_multicomponent.py: arbitrary mixture via
--gases, --y, --p-total-bar. Species order is canonical (e.g. CO2, N2); partial pressures
are p_i = y_i * p_total.

Usage:
    python run_gcmc_uma_multi.py --cif path/to/relaxed.cif --output-dir ./out --weights path/to/uma.pt \\
        --steps 100000 --temperature-K 298 --gases CO2 N2 --y 0.15 0.85 --p-total-bar 1.0

Requirements:
    - Conda environment: fairchem-agent
    - Required packages: ase, ase-mc, numpy, matplotlib, torch, fairchem
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from ase import Atoms, build
from ase.io import read
from ase.optimize import LBFGS

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

try:
    from .ase_mc import BVT, BVT_GCMCOnly, MonteCarlo  # type: ignore[import-not-found]
    from .gcmc_common import (  # type: ignore[import-not-found]
        B_peng_robinson_mixture_for_framework,
        analyze_and_plot_multicomponent,
        build_moveset_bvt,
        canonicalize_species_key,
        ensure_tags,
        get_radius_probe,
        load_host_atoms,
        load_restart_atoms,
        print_restart_sanity_multicomponent,
        run_mc,
        write_timing_kv,
        compute_eq_loading_from_traj_multicomponent,
        gcmc_standalone_payload_multicomponent,
        write_gcmc_results_json,
        GAS_PR_PARAMS_CO2_N2,
        gcmc_output_dir,
        _parse_kij_entry,
        _resolve_species_name,
    )
    from .uma_calculator import load_uma_calculators, set_uma_spin_info  # type: ignore[import-not-found]
except Exception:
    from ase_mc import BVT, BVT_GCMCOnly, MonteCarlo
    from gcmc_common import (
        B_peng_robinson_mixture_for_framework,
        analyze_and_plot_multicomponent,
        build_moveset_bvt,
        canonicalize_species_key,
        ensure_tags,
        get_radius_probe,
        load_host_atoms,
        load_restart_atoms,
        print_restart_sanity_multicomponent,
        run_mc,
        write_timing_kv,
        compute_eq_loading_from_traj_multicomponent,
        gcmc_standalone_payload_multicomponent,
        write_gcmc_results_json,
        GAS_PR_PARAMS_CO2_N2,
        gcmc_output_dir,
        _parse_kij_entry,
        _resolve_species_name,
    )
    from uma_calculator import load_uma_calculators, set_uma_spin_info

# Gas DB: same as single-component skill (CO2, N2). Extend here or via JSON if needed.
GAS_DB = GAS_PR_PARAMS_CO2_N2


def _load_uma_calcs(*, weights_path: str, task_name: str, device: str) -> tuple[object, object]:
    """Return (calc_probe, calc_host), supporting both loader signatures/returns (same as COFclean)."""
    try:
        ret = load_uma_calculators(weights_path=weights_path, task_name=task_name, device=device)
    except TypeError:
        ret = load_uma_calculators(model=weights_path, task_name=task_name, device=device)
    if isinstance(ret, dict):
        calc_host = ret.get("host")
        calc_probe = ret.get("probe")
        if calc_host is None or calc_probe is None:
            raise RuntimeError(f"load_uma_calculators returned dict missing keys: {list(ret.keys())}")
        return calc_probe, calc_host
    if isinstance(ret, (tuple, list)) and len(ret) >= 3:
        return ret[1], ret[2]
    raise RuntimeError(f"Unrecognized load_uma_calculators return type: {type(ret)}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multicomponent BVT GCMC (N-component mixture) in a periodic host (CIF) using UMA/FairChem."
    )
    p.add_argument("--cif", type=Path, required=True, help="Path to periodic framework CIF file.")
    p.add_argument("--steps", type=int, required=True, help="Number of Monte Carlo steps.")
    p.add_argument("--temperature-K", type=float, required=True, help="Simulation temperature in Kelvin.")
    p.add_argument("--scheme", choices=["gcmc", "hmc"], default="gcmc")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for results (gcmc_results.json + PNGs).")
    p.add_argument("--restart-traj", type=Path, default=None, help="Optional: restart from previous ASE .traj.")
    p.add_argument("--restart-frame", type=int, default=-1, help="Frame index to restart from (default -1).")
    p.add_argument("--device", type=str, default="cuda", help="Device string (e.g. 'cuda', 'cpu').")
    p.add_argument("--weights", type=Path, required=True, help="Path to UMA checkpoint (.pt).")
    p.add_argument("--task-name", type=str, default="odac", help="UMA task name (default: odac).")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path (default: output_dir/gcmc_results.json)")
    p.add_argument(
        "--starting-tag",
        type=int,
        default=1,
        help="Starting tag for species; tags assigned in canonical order: tag = starting_tag + i.",
    )
    p.add_argument("--grid-resolution", type=int, default=10, help="BVT cavity grid resolution.")
    p.add_argument("--translate-max", type=float, default=5.0, help="Translate move max_delta (Å).")
    p.add_argument("--probe-fmax", type=float, default=0.05, help="LBFGS fmax for probe relaxation.")
    p.add_argument("--probe-steps", type=int, default=200, help="LBFGS max steps for probe relaxation.")
    # Mixture: N components (same as COFclean)
    p.add_argument("--gases", nargs="+", type=str, default=None, help="Gas species list, e.g. --gases CO2 N2")
    p.add_argument(
        "--y",
        nargs="+",
        type=float,
        default=None,
        help="Mole fractions aligned with --gases, e.g. --y 0.15 0.85 (will be normalized).",
    )
    p.add_argument("--p-total-bar", type=float, default=None, help="Total pressure in bar.")
    p.add_argument("--kij", action="append", default=[], help="Optional PR kij entries like 'CO2,N2,0.0'. Repeatable.")
    return p.parse_args(args=argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.gases is None or args.y is None or args.p_total_bar is None:
        raise ValueError("Must pass --gases, --y, and --p-total-bar.")

    cif_path = args.cif
    T = float(args.temperature_K)

    # Resolve species names against GAS_DB (same as COFclean)
    gases_in = [_resolve_species_name(g, GAS_DB) for g in args.gases]
    y_in = np.asarray(args.y, dtype=float)

    if len(gases_in) != len(y_in):
        raise ValueError(f"--gases (n={len(gases_in)}) and --y (n={len(y_in)}) must match.")
    if len(set(gases_in)) != len(gases_in):
        raise ValueError(f"Duplicate gas in --gases: {gases_in}")
    if np.any(y_in < 0) or y_in.sum() <= 0:
        raise ValueError(f"Invalid --y: {y_in}")

    # Normalize composition and canonicalize ordering for restart stability
    y_in = y_in / y_in.sum()
    y_map = {g: float(yi) for g, yi in zip(gases_in, y_in)}
    species_names = sorted(y_map.keys(), key=canonicalize_species_key)
    y = np.array([y_map[nm] for nm in species_names], dtype=float)

    p_total_bar = float(args.p_total_bar)
    if p_total_bar <= 0:
        raise ValueError(f"--p-total-bar must be > 0 (got {p_total_bar})")
    p_bar_map = {nm: float(y_i * p_total_bar) for nm, y_i in zip(species_names, y)}

    out_dir = gcmc_output_dir(args.output_dir)
    gcmc_json_path = args.output if args.output is not None else out_dir / "gcmc_results.json"
    starting_tag = int(args.starting_tag)
    species_tags = [starting_tag + i for i in range(len(species_names))]

    print(f"[INFO] CIF            : {cif_path}")
    print(f"[INFO] UMA weights    : {args.weights}")
    print(f"[INFO] Task name      : {args.task_name}")
    print(f"[INFO] Device         : {args.device}")
    print(f"[INFO] Steps (MC)     : {args.steps}")
    print(f"[INFO] T [K]          : {T}")
    print(f"[INFO] Scheme         : {args.scheme}")
    print(f"[INFO] Species (canon): {species_names}")
    print(f"[INFO] y (canon)      : {y}")
    print(f"[INFO] p_total [bar]  : {p_total_bar}")
    print(f"[INFO] p_partial [bar]: {p_bar_map}")
    print(f"[INFO] Output dir     : {out_dir}")
    print(f"[INFO] starting_tag   : {starting_tag}")
    print(f"[INFO] species_tags   : {species_tags}")

    t0 = time.perf_counter()
    calc_mol, calc_host = _load_uma_calcs(
        weights_path=str(args.weights),
        task_name=args.task_name,
        device=args.device,
    )
    print(f"[INFO] Loaded UMA calculators in {time.perf_counter() - t0:.3f} s")

    # Build + relax probes (vacuum), same logic as COFclean
    probes: list[Atoms] = []
    ref_energies: list[float] = []
    radii: list[float] = []

    for nm, tag in zip(species_names, species_tags):
        gas = GAS_DB[nm]
        mol_name = gas.get("mol_name", nm)
        build_mode = str(gas.get("build", "")).lower()
        if build_mode == "atom":
            probe = Atoms(mol_name)
        else:
            try:
                probe = build.molecule(mol_name)
            except Exception:
                probe = Atoms(mol_name)
        set_uma_spin_info(probe)
        for a in probe:
            a.tag = int(tag)
        probe.calc = calc_mol
        opt = LBFGS(probe, trajectory=None, logfile=None)
        opt.run(fmax=float(args.probe_fmax), steps=int(args.probe_steps))
        E_probe = float(probe.get_potential_energy())
        r_probe = float(get_radius_probe(probe))
        probe.calc = None
        print(f"[INFO] Probe {nm}: E_ref={E_probe:.6f} eV | radius={r_probe:.3f} Å | tag={tag}")
        probes.append(probe)
        ref_energies.append(E_probe)
        radii.append(r_probe)

    rcavity = 0.8 * (min(radii) if radii else 0.0)

    host, host_natoms, restart_label = load_host_atoms(
        cif_path=cif_path,
        restart_traj=args.restart_traj,
        restart_frame=args.restart_frame,
        read_fn=read,
        load_restart_atoms_fn=load_restart_atoms,
    )
    set_uma_spin_info(host)
    host.calc = calc_host
    tags = ensure_tags(host)
    try:
        host_max = int(np.max(np.asarray(tags[:host_natoms], dtype=int))) if host_natoms > 0 else int(np.max(tags))
        if starting_tag <= host_max:
            print(
                f"[WARN] starting_tag={starting_tag} <= host_max_tag={host_max}. "
                f"Consider using --starting-tag {host_max + 1} to avoid overlap."
            )
    except Exception:
        pass

    print_restart_sanity_multicomponent(
        host,
        host_natoms,
        species_list=probes,
        species_names=species_names,
        species_tags=species_tags,
        label=restart_label,
    )
    print(f"[INFO] Host atoms={len(host)} | host_natoms(from CIF)={host_natoms} | pbc={tuple(bool(x) for x in host.pbc)}")

    # PR mixture -> per-component B (same as COFclean)
    Tc = np.array([float(GAS_DB[nm]["Tc"]) for nm in species_names], dtype=float)
    Pc = np.array([float(GAS_DB[nm]["Pc"]) for nm in species_names], dtype=float)
    omega = np.array([float(GAS_DB[nm]["omega"]) for nm in species_names], dtype=float)
    M = np.array(
        [float(GAS_DB[nm].get("M", GAS_DB[nm].get("molar_mass", 0.0))) for nm in species_names],
        dtype=float,
    )
    kij = None
    if args.kij:
        kij = np.zeros((len(species_names), len(species_names)), dtype=float)
        for entry in args.kij:
            a, b, val = _parse_kij_entry(entry)
            a = _resolve_species_name(a, GAS_DB)
            b = _resolve_species_name(b, GAS_DB)
            ia = species_names.index(a)
            ib = species_names.index(b)
            kij[ia, ib] = float(val)
            kij[ib, ia] = float(val)

    B_ads, beta_mu, phi, V_cell_m3, Z = B_peng_robinson_mixture_for_framework(
        atoms=host,
        T=T,
        p_total_bar=p_total_bar,
        y=y,
        Tc=Tc,
        Pc=Pc,
        omega=omega,
        molar_mass=M,
        kij=kij,
        phase="vapor",
        p_ref_bar=1.0,
    )
    print(f"[INFO] Framework volume: {host.get_volume():.3f} Å^3 ({V_cell_m3:.3e} m^3) | Z={Z:.6f}")
    for nm, Bv, phiv, bmu in zip(species_names, B_ads, phi, beta_mu):
        print(f"[INFO] PR mixture: {nm:>6s} phi={phiv:.6f}  beta_mu={bmu:.6f}  B={Bv:.6f}")

    exclusion_list = np.arange(host_natoms, dtype=int)
    grid_resolution = int(args.grid_resolution)
    translate_max = float(args.translate_max)
    bvt_kwargs = dict(
        temperature_K=T,
        b_parameter=[float(x) for x in B_ads],
        species=probes,
        reference_energy=[float(x) for x in ref_energies],
        rcavity=float(rcavity),
        grid_resolution=grid_resolution,
        cavity_bias=True,
        exclusion_list=exclusion_list,
        starting_tag=starting_tag,
    )
    EnsembleClass = BVT_GCMCOnly if args.scheme == "gcmc" else BVT
    ensemble = EnsembleClass(**bvt_kwargs)
    moveset = build_moveset_bvt(ensemble, scheme=args.scheme)
    moveset.adjust_parameter("Translate", "max_delta", translate_max)

    restarting = args.restart_traj is not None
    traj_path, log_path, dt_mc, steps_per_sec = run_mc(
        MonteCarlo_cls=MonteCarlo,
        atoms=host,
        moveset=moveset,
        dft_calc=calc_host,
        out_dir=out_dir,
        io_stub="mc",
        steps=int(args.steps),
        scheme=args.scheme,
        restarting=restarting,
        loginterval=20,
        perf_counter=time.perf_counter,
        info_prefix="[INFO] ",
    )
    print(f"[INFO] MC completed in {dt_mc:.3f} s | {steps_per_sec:.2f} steps/s")

    write_timing_kv(
        out_dir=out_dir,
        kv={
            "scheme": args.scheme,
            "steps": int(args.steps),
            "temperature_K": float(T),
            "p_total_bar": f"{p_total_bar:.8f}",
            "species_order": ",".join(species_names),
            "p_partial_bar": ",".join(f"{nm}:{p_bar_map[nm]:.8f}" for nm in species_names),
            "restart_traj": str(args.restart_traj) if args.restart_traj is not None else "",
            "restart_frame": int(args.restart_frame),
            "host_natoms": int(host_natoms),
            "starting_tag": int(starting_tag),
            "species_tags": ",".join(str(t) for t in species_tags),
            "wall_time_s": f"{dt_mc:.6f}",
            "steps_per_s": f"{steps_per_sec:.6f}",
        },
    )

    print("[INFO] Analyzing trajectory and writing nmols/energy plots...")
    analyze_and_plot_multicomponent(
        traj_path=traj_path,
        out_dir=out_dir,
        host_natoms=host_natoms,
        species_list=probes,
        species_names=species_names,
        species_tags=species_tags,
        combined_plot=True,
    )

    q_by_adsorbate, q_total = compute_eq_loading_from_traj_multicomponent(
        traj_path=traj_path,
        species_list=probes,
        species_names=species_names,
        species_tags=species_tags,
        host=host,
        host_natoms=host_natoms,
        frac_tail=0.05,
    )
    payload = gcmc_standalone_payload_multicomponent(
        species_names=species_names,
        temperature_K=T,
        pressure_partial_bar={nm: float(p_bar_map[nm]) for nm in species_names},
        scheme=args.scheme,
        q_by_adsorbate=q_by_adsorbate,
        q_total=q_total,
    )
    payload["wall_time_s"] = dt_mc
    payload["steps_per_s"] = steps_per_sec
    write_gcmc_results_json(gcmc_json_path, payload)
    print(f"[INFO] Wrote {gcmc_json_path}")

    for p in (traj_path, log_path):
        try:
            if p is not None and p.exists():
                p.unlink()
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
