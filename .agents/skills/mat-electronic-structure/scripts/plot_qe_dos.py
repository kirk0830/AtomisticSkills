"""
Plot Quantum ESPRESSO density of states (DOS) from results.json.

Usage:
    python plot_qe_dos.py <results_dir_or_json> [--output dos.png]

Requirements:
    - Pixi environment: qe (numpy, matplotlib)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _load_results(input_path: Path) -> Dict[str, Any]:
    """Load results.json from a file or directory."""
    if input_path.is_dir():
        candidate = input_path / "results.json"
        if not candidate.exists():
            raise FileNotFoundError(f"No results.json found in {input_path}")
        input_path = candidate

    with open(input_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_dos(
    eigenvalues: np.ndarray,
    fermi_energy: float,
    sigma: float = 0.05,
    n_points: int = 500,
    energy_window: Tuple[float, float] = (-10, 10),
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute total DOS from eigenvalues using Gaussian broadening.

    Args:
        eigenvalues: Array of shape (n_kpoints, n_bands) or
            (n_spin, n_kpoints, n_bands) in eV.
        fermi_energy: Fermi energy in eV.
        sigma: Gaussian broadening width in eV.
        n_points: Number of energy grid points.
        energy_window: Window around Fermi level (eV).

    Returns:
        Tuple (energies, dos) where energies are relative to the Fermi level.
    """
    ev = np.asarray(eigenvalues)
    flat = ev.reshape(-1)

    emin, emax = energy_window
    energies = np.linspace(emin, emax, n_points)
    dos = np.zeros_like(energies)

    for e in flat:
        dos += np.exp(-0.5 * ((energies - (e - fermi_energy)) / sigma) ** 2)

    # Normalize by number of k-points so DOS is per k-point (conventional).
    n_kpoints = ev.shape[-2] if ev.ndim >= 2 else 1
    dos /= n_kpoints * sigma * np.sqrt(2.0 * np.pi)
    return energies, dos


def plot_qe_dos(
    results: Dict[str, Any],
    output_path: str = "dos.png",
    sigma: float = 0.05,
    energy_window: Tuple[float, float] = (-10, 10),
) -> None:
    """
    Plot DOS from a QE results dictionary.

    Args:
        results: Dictionary produced by run_qe_band_structure.py.
        output_path: Output image path.
        sigma: Gaussian broadening width in eV.
        energy_window: Energy window around the Fermi level (eV).
    """
    eigenvalues = np.asarray(results.get("eigenvalues_uniform"))
    fermi_energy = results.get("fermi_energy", 0.0)

    if eigenvalues.size == 0:
        raise ValueError(
            "No uniform eigenvalues found in results. Use mode='uniform' or 'both'."
        )

    fig, ax = plt.subplots(figsize=(7, 6))

    if eigenvalues.ndim == 2:
        energies, dos = compute_dos(
            eigenvalues, fermi_energy, sigma=sigma, energy_window=energy_window
        )
        ax.fill_between(energies, dos, color="gray", alpha=0.4)
        ax.plot(energies, dos, color="black", linewidth=1.2, label="Total DOS")
    elif eigenvalues.ndim == 3:
        energies_up, dos_up = compute_dos(
            eigenvalues[0], fermi_energy, sigma=sigma, energy_window=energy_window
        )
        energies_dn, dos_dn = compute_dos(
            eigenvalues[1], fermi_energy, sigma=sigma, energy_window=energy_window
        )
        ax.fill_between(energies_up, dos_up, color="blue", alpha=0.3)
        ax.plot(energies_up, dos_up, color="blue", linewidth=1.2, label="Spin up")
        ax.fill_between(energies_dn, -dos_dn, color="red", alpha=0.3)
        ax.plot(energies_dn, -dos_dn, color="red", linewidth=1.2, label="Spin down")
        ax.axhline(y=0, color="black", linewidth=0.8)
    else:
        raise ValueError(f"Unexpected eigenvalues shape {eigenvalues.shape}")

    ax.axvline(x=0, color="red", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Energy (eV)", fontsize=12)
    ax.set_ylabel("DOS (states / eV)", fontsize=12)
    ax.set_title("Density of States", fontsize=14)
    ax.set_xlim(*energy_window)
    ax.legend()
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\n✓ DOS plot saved to: {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Quantum ESPRESSO DOS from results.json"
    )
    parser.add_argument(
        "results",
        help="Path to results.json or the output directory containing it",
    )
    parser.add_argument(
        "--output",
        default="dos.png",
        help="Output path for the DOS plot (default: dos.png)",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Gaussian broadening width in eV (default: 0.05)",
    )
    parser.add_argument(
        "--window",
        type=float,
        nargs=2,
        default=(-10, 10),
        metavar=("MIN", "MAX"),
        help="Energy window around the Fermi level in eV (default: -10 10)",
    )

    args = parser.parse_args()

    results = _load_results(Path(args.results))
    plot_qe_dos(results, args.output, sigma=args.sigma, energy_window=tuple(args.window))


if __name__ == "__main__":
    main()
