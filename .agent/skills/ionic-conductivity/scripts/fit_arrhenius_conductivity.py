"""
Fit Arrhenius behavior from multiple conductivity JSON files.

Default fit: ln(sigma*T) vs 1/T, which is the standard form when sigma
derives from the Nernst-Einstein relation (sigma ~ D/T).

Produces a JSON file with Ea (eV) and prefactor, and an Arrhenius plot.

Usage:
    python fit_arrhenius_conductivity.py conductivity_*.json --out fit.json

Requirements:
    - Conda environment: base-agent
    - Required packages: numpy, matplotlib
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

K_B_EV = 8.617333262e-5  # eV/K


def fit_arrhenius(
    temps: np.ndarray,
    sigmas: np.ndarray,
    sigma_errs: Optional[np.ndarray] = None,
    fit_form: str = "ln_sigmaT",
) -> dict:
    """
    Fit Arrhenius relation to conductivity data.

    For fit_form="ln_sigmaT": fits ln(sigma*T) = a + b/T, where Ea = -b * kB.
    For fit_form="ln_sigma":  fits ln(sigma) = a + b/T, where Ea = -b * kB.

    Args:
        temps: Array of temperatures in K.
        sigmas: Array of conductivities (any consistent unit).
        sigma_errs: Optional array of conductivity uncertainties.
        fit_form: "ln_sigmaT" (default) or "ln_sigma".

    Returns:
        Dictionary with Ea_eV, slope, intercept, and covariance info.
    """
    x = 1.0 / temps

    if fit_form == "ln_sigmaT":
        y = np.log(sigmas * temps)
    else:
        y = np.log(sigmas)

    # weighted fit if errors are available and non-zero
    has_weights = (
        sigma_errs is not None
        and len(sigma_errs) == len(sigmas)
        and np.all(sigma_errs > 0)
    )

    if has_weights:
        # propagate sigma_err to y-space: dy/dsigma = 1/sigma (for ln terms)
        y_err = sigma_errs / sigmas
        weights = 1.0 / y_err
        coeffs, cov = np.polyfit(x, y, 1, w=weights, cov=True)
    else:
        coeffs, cov = np.polyfit(x, y, 1, cov=True)

    slope, intercept = coeffs
    Ea_eV = -slope * K_B_EV
    Ea_err_eV = float(np.sqrt(cov[0, 0])) * K_B_EV

    return {
        "Ea_eV": float(Ea_eV),
        "Ea_err_eV": float(Ea_err_eV),
        "slope": float(slope),
        "intercept": float(intercept),
        "fit_form": fit_form,
    }


def plot_arrhenius(
    temps: np.ndarray,
    sigmas: np.ndarray,
    sigma_errs: Optional[np.ndarray],
    Ea_eV: float,
    Ea_err_eV: float,
    slope: float,
    intercept: float,
    fit_form: str,
    unit_label: str,
    output_file: str,
) -> None:
    """
    Generate an Arrhenius plot.

    Args:
        temps: Temperatures in K.
        sigmas: Conductivities.
        sigma_errs: Optional conductivity uncertainties.
        Ea_eV: Activation energy in eV.
        Ea_err_eV: Uncertainty in Ea.
        slope: Fit slope (for 1/T).
        intercept: Fit intercept.
        fit_form: "ln_sigmaT" or "ln_sigma".
        unit_label: Unit string for axis label (e.g., "S/cm").
        output_file: Path to save the plot.
    """
    inv_temp_1000 = 1000.0 / temps

    plt.figure(figsize=(8, 6))

    if fit_form == "ln_sigmaT":
        y_data = sigmas * temps
        ylabel = f"$\\sigma_T$ ({unit_label}$\\cdot$K)"
    else:
        y_data = sigmas
        ylabel = f"$\\sigma$ ({unit_label})"

    if sigma_errs is not None and np.any(sigma_errs > 0):
        if fit_form == "ln_sigmaT":
            y_errs = sigma_errs * temps
        else:
            y_errs = sigma_errs
        plt.errorbar(
            inv_temp_1000, y_data, yerr=y_errs,
            fmt='ko', capsize=5, markersize=8
        )
    else:
        plt.plot(inv_temp_1000, y_data, 'ko', markersize=8)

    t_fit = np.linspace(min(temps) - 50, max(temps) + 100, 200)
    inv_t_fit_1000 = 1000.0 / t_fit
    y_fit = np.exp(intercept + slope / t_fit)
    if fit_form == "ln_sigmaT":
        # ln(sigma*T) = intercept + slope/T  =>  sigma*T = exp(intercept + slope/T)
        pass  # y_fit is already sigma*T
    plt.plot(
        inv_t_fit_1000, y_fit, 'r-', linewidth=2,
        label=f"$E_a = {Ea_eV:.3f} \\pm {Ea_err_eV:.3f}$ eV"
    )

    plt.yscale('log')
    plt.xlabel('1000 / T (K$^{-1}$)', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=18)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(matplotlib.ticker.LogFormatterSciNotation())
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(18)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(18)

    plt.grid(True, which="both", ls="-", alpha=0.3)
    plt.legend(fontsize=18, loc='best')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close()
    print(f"Saved Arrhenius plot: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit Arrhenius relation to ionic conductivity data from "
                    "multiple conductivity JSON files."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="Conductivity JSON files from compute_ionic_conductivity.py"
    )
    parser.add_argument(
        "--use", default="sigma_S_cm",
        choices=["sigma_S_m", "sigma_S_cm", "sigma_ne_S_m", "sigma_ne_S_cm"],
        help="Which conductivity field to use. Default: sigma_S_cm."
    )
    parser.add_argument(
        "--fit_form", default="ln_sigmaT",
        choices=["ln_sigmaT", "ln_sigma"],
        help="Fit ln(sigma*T) vs 1/T (default) or ln(sigma) vs 1/T."
    )
    parser.add_argument(
        "--out", required=True,
        help="Output JSON path for fit results."
    )
    args = parser.parse_args()

    err_field = args.use.replace("sigma_", "sigma_err_").replace("sigma_ne_err_", "sigma_ne_err_")
    if "sigma_ne" in args.use:
        err_field = args.use.replace("sigma_ne_", "sigma_ne_err_")
    else:
        err_field = args.use.replace("sigma_", "sigma_err_")

    temps: List[float] = []
    sigmas: List[float] = []
    sigma_errs: List[float] = []

    for p in args.inputs:
        try:
            data = json.loads(Path(p).read_text())
        except Exception as e:
            print(f"Warning: Could not read {p}: {e}")
            continue

        val = data.get(args.use)
        if val is None or val <= 0:
            print(f"Warning: Skipping {p}, {args.use}={val}")
            continue

        temps.append(float(data["temperature_K"]))
        sigmas.append(float(val))
        sigma_errs.append(float(data.get(err_field, 0.0)))

    if len(temps) < 2:
        print(f"Error: Need at least 2 valid data points, found {len(temps)}.")
        sys.exit(1)

    T = np.array(temps)
    sigma = np.array(sigmas)
    sigma_err = np.array(sigma_errs)

    idx = np.argsort(T)
    T = T[idx]
    sigma = sigma[idx]
    sigma_err = sigma_err[idx]

    has_err = np.any(sigma_err > 0)

    print(f"Fitting Arrhenius ({args.fit_form}) to {len(T)} points...")
    fit = fit_arrhenius(
        T, sigma,
        sigma_errs=sigma_err if has_err else None,
        fit_form=args.fit_form,
    )

    unit_label = "S/cm" if "S_cm" in args.use else "S/m"
    result = {
        "fit_form": fit["fit_form"],
        "conductivity_field": args.use,
        "unit": unit_label,
        "Ea_eV": fit["Ea_eV"],
        "Ea_err_eV": fit["Ea_err_eV"],
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "T_K": T.tolist(),
        "sigma": sigma.tolist(),
    }
    if has_err:
        result["sigma_err"] = sigma_err.tolist()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=4))

    plot_path = str(out_path).replace(".json", "_plot.png")
    plot_arrhenius(
        T, sigma,
        sigma_err if has_err else None,
        fit["Ea_eV"], fit["Ea_err_eV"],
        fit["slope"], fit["intercept"],
        args.fit_form, unit_label, plot_path,
    )

    print("-" * 40)
    print(f"Ea = {fit['Ea_eV']:.4f} +/- {fit['Ea_err_eV']:.4f} eV")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {plot_path}")


if __name__ == "__main__":
    main()
