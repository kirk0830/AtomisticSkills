"""
BVT/GCMC Monte Carlo adsorption of CO2 or N2 into a periodic framework using UMA (FairChem).

Usage:
    python run_gcmc_uma.py --cif path/to/relaxed.cif --output-dir ./out --weights path/to/uma.pt \\
        --steps 100000 --temperature-K 298 --pressure-bar 1 --cof-dir ./out/MYCOF

Requirements:
    - Conda environment: fairchem-agent
    - Required packages: ase, ase-mc, numpy, matplotlib, torch, fairchem
"""

import argparse
import sys
import time
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

import numpy as np
from ase import Atoms, build
from ase.io import read
from ase.optimize import LBFGS

try:
    # Package-relative imports (preferred when importable as a package).
    from .ase_mc import BVT, BVT_GCMCOnly, MonteCarlo  # type: ignore[import-not-found]
    from .gcmc_common import (  # type: ignore[import-not-found]
        B_peng_robinson_for_framework,
        analyze_and_plot_single,
        build_moveset_bvt,
        get_radius_probe,
        load_restart_atoms,
        print_restart_sanity_single,
        GAS_PR_PARAMS_CO2_N2,
        gcmc_output_dir,
        load_host_atoms,
        mc_traj_log_paths,
        maybe_record_starting_config,
        run_mc_with_timing,
        write_timing_kv,
        compute_eq_loading_from_traj_single,
        gcmc_standalone_payload_single,
        write_gcmc_results_json,
        ensure_tags,
    )
    from .uma_calculator import load_uma_calculators, set_uma_spin_info  # type: ignore[import-not-found]
except Exception:
    # Script execution fallback (expects scripts/ on sys.path).
    from ase_mc import BVT, BVT_GCMCOnly, MonteCarlo
    from gcmc_common import (
    B_peng_robinson_for_framework,
    analyze_and_plot_single,
    build_moveset_bvt,
    get_radius_probe,
    load_restart_atoms,
    print_restart_sanity_single,
    GAS_PR_PARAMS_CO2_N2,
    gcmc_output_dir,
    load_host_atoms,
    mc_traj_log_paths,
    maybe_record_starting_config,
    run_mc_with_timing,
    write_timing_kv,
    compute_eq_loading_from_traj_single,
    gcmc_standalone_payload_single,
    write_gcmc_results_json,
    ensure_tags,
    )
    from uma_calculator import load_uma_calculators, set_uma_spin_info


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


def retag_all_guests_single_species(atoms: Atoms, host_natoms: int) -> int | None:
    total = len(atoms)
    if total <= host_natoms:
        return None
    tags = np.asarray(atoms.get_tags(), dtype=int)
    if tags.shape[0] != total:
        tags = np.zeros(total, dtype=int)
    host_max = int(tags[:host_natoms].max()) if host_natoms > 0 else int(tags.max())
    guest_tag = host_max + 1
    tags[host_natoms:] = guest_tag
    atoms.set_tags(tags)
    return guest_tag


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="BVT Monte Carlo adsorption of a probe gas (CO2 or N2) into a periodic host (CIF) using UMA/FairChem."
    )
    p.add_argument("--cif", type=Path, required=True, help="Path to periodic framework CIF file.")
    p.add_argument("--steps", type=int, required=True, help="Number of Monte Carlo steps.")
    p.add_argument("--temperature-K", type=float, required=True, help="Simulation temperature in Kelvin.")
    p.add_argument("--pressure-bar", type=float, default=1.0, help="Gas pressure in bar for PR->B.")
    p.add_argument("--scheme", choices=["gcmc", "hmc"], default="gcmc")
    p.add_argument("--adsorbate", choices=["CO2", "N2"], default="CO2", help="Probe gas to adsorb.")
    p.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for results (gcmc_results.json + PNGs).")
    p.add_argument("--restart-traj", type=Path, default=None, help="Optional: restart from previous ASE trajectory (.traj).")
    p.add_argument("--restart-frame", type=int, default=-1, help="Frame index to restart from (default -1 = last frame).")
    p.add_argument("--device", type=str, default="cuda", help="Device string (e.g. 'cuda', 'cpu').")
    p.add_argument("--weights", type=Path, required=True, help="Path to UMA checkpoint (.pt).")
    p.add_argument("--task-name", type=str, default="odac", help="UMA task name (default: odac).")
    p.add_argument("--output", "-o", type=Path, default=None, help="Output JSON path (default: output_dir/gcmc_results.json)")
    p.add_argument("--model-tag", type=str, default="uma", help="Model tag for metadata.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cif_path = args.cif

    out_dir = gcmc_output_dir(args.output_dir)
    if args.output is not None:
        gcmc_json_path = args.output
    else:
        gcmc_json_path = out_dir / "gcmc_results.json"

    print(f"[INFO] CIF          : {cif_path}")
    print(f"[INFO] UMA weights  : {args.weights}")
    print(f"[INFO] Task name    : {args.task_name}")
    print(f"[INFO] Device       : {args.device}")
    print(f"[INFO] Steps (MC)   : {args.steps}")
    print(f"[INFO] T [K]        : {args.temperature_K}")
    print(f"[INFO] p [bar]      : {args.pressure_bar}")
    print(f"[INFO] Scheme       : {args.scheme}")
    print(f"[INFO] Adsorbate    : {args.adsorbate}")
    print(f"[INFO] Output dir   : {out_dir}")

    t0 = time.perf_counter()
    calc_mol, calc_host = _load_uma_calcs(
        weights_path=str(args.weights),
        task_name=args.task_name,
        device=args.device,
    )
    print(f"[INFO] Loaded UMA calculators in {time.perf_counter() - t0:.3f} s")

    PR_PARAMS = GAS_PR_PARAMS_CO2_N2
    gas = PR_PARAMS[args.adsorbate]
    t1 = time.perf_counter()
    probe = build.molecule(gas["mol_name"])
    set_uma_spin_info(probe)
    probe.calc = calc_mol
    out_dir.mkdir(parents=True, exist_ok=True)
    opt = LBFGS(probe, trajectory=None, logfile=None)
    opt.run(fmax=0.05, steps=200)
    E_probe = probe.get_potential_energy()
    print(f"[INFO] Probe relaxation done in {time.perf_counter() - t1:.3f} s | E_probe = {E_probe:.6f} eV")

    host, host_natoms, restart_label = load_host_atoms(
        cif_path=cif_path,
        restart_traj=args.restart_traj,
        restart_frame=args.restart_frame,
        read_fn=read,
        load_restart_atoms_fn=load_restart_atoms,
    )
    set_uma_spin_info(host)
    host.calc = calc_host

    retagged = False
    guest_tag = None
    if args.restart_traj is not None:
        guest_tag = retag_all_guests_single_species(host, host_natoms)
        retagged = guest_tag is not None
    tags = ensure_tags(host)
    host_max = int(tags[:host_natoms].max()) if host_natoms > 0 else int(tags.max())
    starting_tag = guest_tag if guest_tag is not None else (host_max + 1)

    print_restart_sanity_single(
        host, host_natoms, probe,
        label=restart_label + (" [retagged guests]" if retagged else ""),
    )
    print(f"[INFO] Host atoms={len(host)} | host_natoms(from CIF)={host_natoms} | pbc={tuple(bool(x) for x in host.pbc)}")

    T = args.temperature_K
    p_bar = args.pressure_bar
    B_val, beta_mu, phi, V_cell_m3 = B_peng_robinson_for_framework(
        host,
        T=T,
        p_bar=p_bar,
        Tc=gas["Tc"],
        Pc=gas["Pc"],
        omega=gas["omega"],
        molar_mass=gas["M"],
    )
    print(f"[INFO] Framework volume: {host.get_volume():.3f} Å^3 ({V_cell_m3:.3e} m^3)")
    print(f"[INFO] PR EOS: B = {B_val:.3f}, beta_mu = {beta_mu:.6f}, phi = {phi:.6f}")

    t3 = time.perf_counter()
    exclusion_list = np.arange(host_natoms)
    rcavity = 0.8 * get_radius_probe(probe)
    grid_resolution = 10
    translate_max = 5.0
    bvt_kwargs = dict(
        temperature_K=T,
        b_parameter=[B_val],
        species=[probe],
        reference_energy=[E_probe],
        rcavity=rcavity,
        grid_resolution=grid_resolution,
        cavity_bias=True,
        exclusion_list=exclusion_list,
        starting_tag=starting_tag,
    )
    EnsembleClass = BVT_GCMCOnly if args.scheme == "gcmc" else BVT
    ensemble = EnsembleClass(**bvt_kwargs)
    moveset = build_moveset_bvt(ensemble, scheme=args.scheme)
    moveset.adjust_parameter("Translate", "max_delta", translate_max)
    print(f"[INFO] BVT/moveset ready in {time.perf_counter() - t3:.3f} s | rcavity={rcavity:.2f} Å | grid={grid_resolution} | b={B_val:.2f} | translate_max={translate_max:.2f} Å")

    restarting = args.restart_traj is not None
    traj_path, log_path = mc_traj_log_paths(out_dir=out_dir, restarting=restarting, stub="mc")
    dyn = MonteCarlo(
        atoms=host,
        dft_calc=calc_host,
        moveset=moveset,
        trajectory=str(traj_path),
        logfile=str(log_path),
        loginterval=20,
        gcmc_energy_only=(args.scheme == "gcmc"),
    )
    maybe_record_starting_config(dyn)
    print(
        f"[INFO] Starting MC run: scheme={args.scheme}, steps={args.steps}, "
        f"traj={traj_path.name}, log={log_path.name}"
    )
    dt_mc, steps_per_sec = run_mc_with_timing(dyn=dyn, steps=args.steps, perf_counter=time.perf_counter)
    print(f"[INFO] MC completed in {dt_mc:.3f} s | {steps_per_sec:.2f} steps/s")

    write_timing_kv(
        out_dir=out_dir,
        kv={
            "scheme": args.scheme,
            "steps": args.steps,
            "temperature_K": args.temperature_K,
            "pressure_bar": args.pressure_bar,
            "restart_traj": args.restart_traj,
            "restart_frame": args.restart_frame,
            "host_natoms": host_natoms,
            "wall_time_s": f"{dt_mc:.6f}",
            "steps_per_s": f"{steps_per_sec:.6f}",
        },
    )

    print("[INFO] Analyzing trajectory and writing nmols/energy plots...")
    analyze_and_plot_single(traj_path, probe, out_dir, host_natoms=host_natoms)

    q_mmol_g = compute_eq_loading_from_traj_single(
        traj_path=traj_path,
        probe=probe,
        host=host,
        host_natoms=host_natoms,
        frac_tail=0.05,
    )
    payload = gcmc_standalone_payload_single(
        adsorbate=args.adsorbate,
        temperature_K=args.temperature_K,
        pressure_bar=args.pressure_bar,
        scheme=args.scheme,
        q_total_mmol_g=q_mmol_g,
    )
    payload["wall_time_s"] = dt_mc
    payload["steps_per_s"] = steps_per_sec
    write_gcmc_results_json(gcmc_json_path, payload)
    print(f"[INFO] Wrote {gcmc_json_path}")

    # Remove traj/log; keep JSON and PNGs (nmols.png, energy.png)
    for p in (traj_path, log_path):
        try:
            if p is not None and p.exists():
                p.unlink()
        except Exception as e:
            print(f"[WARN] Could not remove {p}: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
