"""
Compute ionic conductivity from tracer diffusivities via the Nernst-Einstein relation.

Reads a structure file (for volume and species counts) and diffusivities from either
a diffusion_results.json file (produced by diffusion-analysis) or manual CLI input.
Outputs sigma_NE in S/m and S/cm to a JSON file.

Usage (from diffusion JSON):
    python compute_ionic_conductivity.py \
        --structure relaxed.cif \
        --diffusion_json md_800K/diffusion_results.json \
        --charges "Li=1" \
        --out conductivity_800K.json

Usage (manual input):
    python compute_ionic_conductivity.py \
        --structure relaxed.cif \
        --temperature 800 \
        --diffusivities "Li=1.0e-6" \
        --diffusion_units "cm2/s" \
        --charges "Li=1" \
        --out conductivity_800K.json

Requirements:
    - Conda environment: base-agent
    - Required packages: ase, numpy
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from ase.io import read

E_CHARGE = 1.602176634e-19  # C
K_B = 1.380649e-23          # J/K
ANG3_TO_M3 = 1e-30          # 1 Angstrom^3 -> m^3
CM2_TO_M2 = 1e-4            # 1 cm^2 -> m^2


def parse_kv_string(s: str) -> Dict[str, float]:
    """
    Parse a comma-separated key=value string into a dictionary.

    Args:
        s: String like "Li=1,Na=1" or "Li=1.0e-6"

    Returns:
        Dictionary mapping species names to float values.
    """
    out: Dict[str, float] = {}
    if s is None or s.strip() == "":
        return out
    for item in s.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Bad key=value pair '{item}'. Expected format: Li=1")
        k, v = item.split("=", 1)
        out[k.strip()] = float(v.strip())
    return out


def convert_diffusivity_to_m2s(D: float, units: str) -> float:
    """
    Convert diffusivity to m^2/s.

    Args:
        D: Diffusivity value.
        units: Source units, one of "cm2/s" or "m2/s".

    Returns:
        Diffusivity in m^2/s.
    """
    units = units.lower().strip()
    if units in ("m2/s", "m^2/s"):
        return D
    if units in ("cm2/s", "cm^2/s"):
        return D * CM2_TO_M2
    raise ValueError(f"Unsupported diffusion units '{units}'. Use 'cm2/s' or 'm2/s'.")


def compute_ne_conductivity(
    volume_ang3: float,
    temperature_K: float,
    species_counts: Dict[str, int],
    diffusivities_m2s: Dict[str, float],
    charges_z: Dict[str, float],
    diffusivity_errors_m2s: Optional[Dict[str, float]] = None,
    haven_ratio: float = 1.0,
) -> dict:
    """
    Compute Nernst-Einstein ionic conductivity from tracer diffusivities.

    Args:
        volume_ang3: Simulation cell volume in Angstrom^3.
        temperature_K: Temperature in Kelvin.
        species_counts: Number of each mobile species in the cell (e.g., {"Li": 64}).
        diffusivities_m2s: Tracer diffusivity per species in m^2/s.
        charges_z: Formal charge (valence) per species (e.g., {"Li": 1}).
        diffusivity_errors_m2s: Optional 1-sigma errors on D per species in m^2/s.
        haven_ratio: Haven ratio H_R. sigma = sigma_NE / H_R.

    Returns:
        Dictionary with conductivity results and per-species breakdown.
    """
    V_m3 = volume_ang3 * ANG3_TO_M3
    per_species = {}
    sigma_ne = 0.0
    sigma_ne_var = 0.0

    for sp, D in diffusivities_m2s.items():
        if sp not in species_counts:
            raise ValueError(
                f"Species '{sp}' not found in structure. "
                f"Present: {sorted(species_counts.keys())}"
            )
        if sp not in charges_z:
            raise ValueError(f"Charge for species '{sp}' not provided in --charges")

        N_i = float(species_counts[sp])
        n_i = N_i / V_m3
        z_i = float(charges_z[sp])
        q_i = z_i * E_CHARGE

        contrib = n_i * (q_i ** 2) * D / (K_B * temperature_K)
        sigma_ne += contrib

        if diffusivity_errors_m2s and sp in diffusivity_errors_m2s:
            d_sigma_d_D = n_i * (q_i ** 2) / (K_B * temperature_K)
            sigma_ne_var += (d_sigma_d_D * diffusivity_errors_m2s[sp]) ** 2

        per_species[sp] = {
            "N": int(species_counts[sp]),
            "n_m-3": n_i,
            "z": z_i,
            "D_m2_s": D,
            "D_cm2_s": D / CM2_TO_M2,
            "sigma_contrib_S_m": contrib,
            "sigma_contrib_S_cm": contrib / 100.0,
        }

    sigma = sigma_ne / haven_ratio
    sigma_ne_err = float(np.sqrt(sigma_ne_var)) if sigma_ne_var > 0 else None
    sigma_err = (sigma_ne_err / haven_ratio) if sigma_ne_err is not None else None

    result = {
        "temperature_K": temperature_K,
        "volume_A3": volume_ang3,
        "sigma_ne_S_m": sigma_ne,
        "sigma_ne_S_cm": sigma_ne / 100.0,
        "sigma_S_m": sigma,
        "sigma_S_cm": sigma / 100.0,
        "haven_ratio": haven_ratio,
        "per_species": per_species,
    }
    if sigma_ne_err is not None:
        result["sigma_ne_err_S_m"] = sigma_ne_err
        result["sigma_ne_err_S_cm"] = sigma_ne_err / 100.0
        result["sigma_err_S_m"] = sigma_err
        result["sigma_err_S_cm"] = sigma_err / 100.0

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ionic conductivity from tracer diffusivities "
                    "via the Nernst-Einstein relation."
    )
    parser.add_argument(
        "--structure", required=True,
        help="Structure file (CIF, POSCAR, .traj). For .traj, uses last frame."
    )
    parser.add_argument(
        "--diffusion_json", default=None,
        help="Path to diffusion_results.json from diffusion-analysis. "
             "Reads species, temperature, diffusivity, and diffusivity_std_dev."
    )
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Temperature in K (overrides value from --diffusion_json if both given)"
    )
    parser.add_argument(
        "--diffusivities", default=None,
        help='Manual tracer diffusivities, e.g. "Li=1e-6,Na=2e-7". '
             "Overrides --diffusion_json values."
    )
    parser.add_argument(
        "--diffusivity_errors", default=None,
        help='Optional 1-sigma errors, e.g. "Li=1e-7,Na=5e-8"'
    )
    parser.add_argument(
        "--diffusion_units", default="cm2/s",
        help="Units for --diffusivities and --diffusivity_errors. "
             "Default: cm2/s (matches diffusion-analysis output)."
    )
    parser.add_argument(
        "--charges", required=True,
        help='Formal charges (valence), e.g. "Li=1,O=-2"'
    )
    parser.add_argument(
        "--haven_ratio", type=float, default=1.0,
        help="Haven ratio H_R. Applies sigma = sigma_NE / H_R. Default: 1.0."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSON path."
    )
    args = parser.parse_args()

    # load structure for volume and species counts
    atoms = read(args.structure, index=-1)
    volume_ang3 = float(atoms.get_volume())
    if volume_ang3 <= 0:
        print(f"Error: Non-positive volume from structure: {volume_ang3} A^3")
        sys.exit(1)

    species_counts = Counter(atoms.get_chemical_symbols())

    # resolve diffusivities and temperature from JSON and/or CLI
    D_dict: Dict[str, float] = {}
    D_err_dict: Dict[str, float] = {}
    temperature: Optional[float] = args.temperature
    diff_units = args.diffusion_units

    if args.diffusion_json:
        with open(args.diffusion_json, "r") as f:
            diff_data = json.load(f)

        sp = diff_data["species"]
        # diffusion-analysis outputs cm^2/s
        json_unit = diff_data.get("unit", "cm^2/s")
        D_val = diff_data["diffusivity"]
        D_err_val = diff_data.get("diffusivity_std_dev", 0.0)

        D_dict[sp] = convert_diffusivity_to_m2s(D_val, json_unit)
        if D_err_val and D_err_val > 0:
            D_err_dict[sp] = convert_diffusivity_to_m2s(D_err_val, json_unit)

        if temperature is None:
            temperature = float(diff_data["temperature"])

    if args.diffusivities:
        manual_D = parse_kv_string(args.diffusivities)
        for sp, val in manual_D.items():
            D_dict[sp] = convert_diffusivity_to_m2s(val, diff_units)
    if args.diffusivity_errors:
        manual_err = parse_kv_string(args.diffusivity_errors)
        for sp, val in manual_err.items():
            D_err_dict[sp] = convert_diffusivity_to_m2s(val, diff_units)

    if temperature is None or temperature <= 0:
        print("Error: Temperature must be > 0 K. Provide --temperature or --diffusion_json.")
        sys.exit(1)

    if not D_dict:
        print("Error: No diffusivities provided. Use --diffusion_json or --diffusivities.")
        sys.exit(1)

    charges_z = parse_kv_string(args.charges)

    result = compute_ne_conductivity(
        volume_ang3=volume_ang3,
        temperature_K=temperature,
        species_counts=dict(species_counts),
        diffusivities_m2s=D_dict,
        charges_z=charges_z,
        diffusivity_errors_m2s=D_err_dict if D_err_dict else None,
        haven_ratio=args.haven_ratio,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=4)

    print(f"Ionic conductivity (Nernst-Einstein):")
    print(f"  T = {temperature:.1f} K")
    print(f"  V = {volume_ang3:.3f} A^3")
    print(f"  sigma_NE = {result['sigma_ne_S_m']:.6e} S/m = {result['sigma_ne_S_cm']:.6e} S/cm")
    if "sigma_ne_err_S_m" in result:
        print(f"  sigma_NE_err = {result['sigma_ne_err_S_m']:.6e} S/m (from D errors)")
    if args.haven_ratio != 1.0:
        print(f"  Haven ratio H_R = {args.haven_ratio:.4g}")
        print(f"  sigma = sigma_NE / H_R = {result['sigma_S_m']:.6e} S/m = {result['sigma_S_cm']:.6e} S/cm")
    for sp, info in result["per_species"].items():
        print(f"  {sp}: N={info['N']}, z={info['z']}, D={info['D_cm2_s']:.3e} cm^2/s, "
              f"sigma_contrib={info['sigma_contrib_S_cm']:.6e} S/cm")
    print(f"  Wrote: {out_path}")


if __name__ == "__main__":
    main()
