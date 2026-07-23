"""
Plot Quantum ESPRESSO band structure from results.json.

Usage:
    python plot_qe_band_structure.py <results_dir_or_json> [--output band_structure.png]

Requirements:
    - Pixi environment: qe (numpy, matplotlib)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def _format_label(label: str) -> str:
    """Map seekpath labels to common LaTeX-style symbols."""
    label = label.upper()
    mappings = {
        "GAMMA": r"$\\Gamma$",
        "G": r"$\\Gamma$",
    }
    return mappings.get(label, label)


def plot_qe_band_structure(
    results: Dict[str, Any],
    output_path: str = "band_structure.png",
    ylim: Optional[tuple] = (-10, 10),
) -> None:
    """
    Plot band structure from a QE results dictionary.

    Args:
        results: Dictionary produced by run_qe_band_structure.py.
        output_path: Output image path.
        ylim: Energy window around the Fermi level (eV).
    """
    eigenvalues = np.asarray(results.get("eigenvalues_line"))
    kpoints = np.asarray(results.get("kpoints_line"))
    path_indices = results.get("kpath_indices", [])
    labels = results.get("kpath_labels", [])
    fermi_energy = results.get("fermi_energy", 0.0)

    if eigenvalues.size == 0 or kpoints.size == 0:
        raise ValueError("No line eigenvalues found in results. Use mode='line' or 'both'.")

    # Build x-axis as cumulative distance along the k-path.
    rec_cell = np.eye(3)  # fractional coords treated in reciprocal lattice units
    if eigenvalues.ndim == 2:
        n_kpoints, n_bands = eigenvalues.shape
    elif eigenvalues.ndim == 3:
        n_spin, n_kpoints, n_bands = eigenvalues.shape
    else:
        raise ValueError(f"Unexpected eigenvalues shape {eigenvalues.shape}")

    distances = [0.0]
    for i in range(1, n_kpoints):
        dk = np.linalg.norm(kpoints[i] - kpoints[i - 1])
        distances.append(distances[-1] + dk)
    distances = np.array(distances)

    fig, ax = plt.subplots(figsize=(8, 6))

    def _plot_bands(ev: np.ndarray, alpha: float = 1.0, label: Optional[str] = None):
        for band in range(ev.shape[-1]):
            ax.plot(
                distances,
                ev[:, band] - fermi_energy,
                color="blue",
                linewidth=1.2,
                alpha=alpha,
                label=label if band == 0 else None,
            )

    if eigenvalues.ndim == 2:
        _plot_bands(eigenvalues)
    else:
        _plot_bands(eigenvalues[0], alpha=0.8, label="Spin up")
        _plot_bands(eigenvalues[1], alpha=0.8, label="Spin down")
        ax.legend()

    # High-symmetry vertical lines and labels.
    for idx in path_indices:
        ax.axvline(x=distances[idx], color="gray", linestyle="--", linewidth=0.8)

    if labels:
        tick_positions = [distances[i] for i in path_indices if i < len(distances)]
        tick_labels = [_format_label(lbl) for lbl in labels[: len(tick_positions)]]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels)

    ax.axhline(y=0, color="red", linestyle="-", linewidth=0.8)
    ax.set_xlabel("Wave Vector", fontsize=12)
    ax.set_ylabel("Energy (eV)", fontsize=12)
    ax.set_title("Band Structure", fontsize=14)
    ax.set_xlim(distances[0], distances[-1])
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=10)
    fig.tight_layout()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    print(f"\n✓ Band structure plot saved to: {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot Quantum ESPRESSO band structure from results.json"
    )
    parser.add_argument(
        "results",
        help="Path to results.json or the output directory containing it",
    )
    parser.add_argument(
        "--output",
        default="band_structure.png",
        help="Output path for the band structure plot (default: band_structure.png)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        default=(-10, 10),
        metavar=("MIN", "MAX"),
        help="Energy window around the Fermi level in eV (default: -10 10)",
    )

    args = parser.parse_args()

    results = _load_results(Path(args.results))
    plot_qe_band_structure(results, args.output, tuple(args.ylim))


if __name__ == "__main__":
    main()
