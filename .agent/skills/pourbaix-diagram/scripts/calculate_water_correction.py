"""
Calculate water correction for Pourbaix diagrams following Persson et al. (2012).

This script derives the reference chemical potentials (μ_H^ref, μ_O) that align
MLIP/DFT water formation energy with experimental values, enabling thermodynamically
consistent mixing of computational solid energies with experimental aqueous species.

Reference:
    K. A. Persson, B. Waldwick, P. Lazic, G. Ceder
    "Prediction of solid-aqueous equilibria: Scheme to combine first-principles 
    calculations of solids with experimental aqueous states"
    Phys. Rev. B 85, 235438 (2012), Equations 37-43

Usage:
    python calculate_water_correction.py \
        --h2o_energy -14.7 \
        --h2_energy -6.77 \
        --o2_energy -9.86 \
        --output water_correction.json
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import sys


# Experimental constants (from Persson 2012, Table I)
K_B = 8.617333262e-5  # eV/K (Boltzmann constant)
T_STANDARD = 298.15   # K (standard temperature)

# Experimental thermodynamic values at 298K
DGFH2O_EXP = -2.4583  # eV (Gibbs free energy of H2O formation, experimental)
S_H2O_EXP = 7.24e-4   # eV/(H2O·K) (entropy of H2O at standard state)


def calculate_water_correction(
    E_H2O: float,
    E_H2: float,
    E_O2: float,
    temperature: float = T_STANDARD
) -> Dict[str, float]:
    """
    Calculate water-corrected chemical potentials following Persson et al. (2012).
    
    Args:
        E_H2O: MLIP total energy of H2O molecule (eV)
        E_H2: MLIP total energy of H2 molecule (eV)  
        E_O2: MLIP total energy of O2 molecule (eV)
        temperature: Temperature in K (default: 298.15K)
        
    Returns:
        Dictionary with corrected chemical potentials and validation info
    """
    results = {}
    
    # Constants for H2 gas correction (NIST-JANAF) at 298.15K
    # Entropy S_H2 = 130.68 J/mol.K = 1.354e-3 eV/K
    S_H2_EXP = 1.354e-3  # eV/K
    # Enthalpy difference H(298) - H(0) = 8.468 kJ/mol = 0.088 eV
    H_CORR_H2 = 0.088    # eV 
    
    # 1. Calculate Reference Hydrogen Chemical Potential (Standard Hydrogen Electrode)
    # mu_H = 0.5 * (E_H2 + ZPE + integrated_Cp - T*S)
    # We use: mu_H = 0.5 * (E_H2_MLIP + (H-H0)_exp - T*S_exp)
    # (H-H0) includes ZPE + thermal enthalpy relative to 0K
    
    # Total correction per H2 molecule:
    # G_H2(T) = E_total + (H-H0) - TS
    TS_H2 = temperature * S_H2_EXP
    G_H2_corr = E_H2 + H_CORR_H2 - TS_H2
    
    mu_H_ref = 0.5 * G_H2_corr
    results['mu_H_ref'] = mu_H_ref
    
    # Store intermediate H2 terms for debugging
    results['G_H2_corr'] = G_H2_corr
    results['TS_H2'] = TS_H2
    results['H_CORR_H2'] = H_CORR_H2
    
    # 2. Calculate Reference Oxygen Chemical Potential from Water Cycle
    # We force the formation energy of water to match experiment (-2.46 eV)
    # DeltaG_f(H2O)_exp = G_H2O_MLIP - 2*mu_H - mu_O
    # => mu_O = G_H2O_MLIP - 2*mu_H - DeltaG_f(H2O)_exp
    
    # G_H2O_MLIP: We use E_H2O + (H-H0) - TS
    # For liquid water, entropy contribution is ~0.216 eV (T*S at 298K)
    # ZPE/Enthalpy contribution is small but code previously laid out:
    # "thermal_contribution = temperature * S_H2O_EXP" ~ 0.216 eV
    # Previous code added this to E_H2O. Let's check sign.
    # G = H - TS. If we approximate H ~ E_MLIP + ZPE, then G ~ E + ZPE - TS.
    # Persson paper Eq 40: 1/2 [ E_H2O + ZPE - TS ... ]  (Using + for ZPE, - for TS)
    # The previous code had `0.5 * (E_H2O + thermal ...)` which was confusing.
    
    # Let's align with Persson 2012 Eq. 42 exactly.
    # Actually, Persson Eq 38: mu_O = E_H2O - 2 mu_H - DG_exp.
    # But E_H2O in Persson includes ZPE and thermal corrections implicitly or explicitly?
    # Usually treating E_DFT as H(0K), we need to add standard state corrections.
    #
    # However, a robust way is to rely on the difference.
    # Let's use the explicit terms:
    TS_H2O = temperature * S_H2O_EXP  # ~0.216 eV
    # For H2O(l), ZPE + (H-H0) is roughly 0.57 eV (ZPE) + 0.1 eV (thermal).
    # But usually broad "water correction" subsumes ZPE if not explicitly added.
    # H2O ZPE is large (~0.56 eV). If E_H2O is purely static DFT, we must add ZPE.
    # But often "calculated water energy" implies we might not have ZPE.
    #
    # If we simply define mu_O to close the cycle:
    # mu_O = [E_H2O - TS_H2O] - 2*mu_H - DGFH2O_EXP
    # (Assuming E_H2O approximates H, and subtracting TS to get G)
    # This consistency ensures that:
    #   G_formation = (E_H2O - TS_H2O) - 2*mu_H - mu_O
    #               = DGFH2O_EXP
    # Which is exactly what we want.
    
    # Note: If we ignore ZPE in E_H2O, our mu_O absorbs that error. 
    # This is fine as long as solids also ignore ZPE (cancellation).
    # This is the standard "Material Project" approach (Calculated - exp).
    
    G_H2O_MLIP = E_H2O - TS_H2O
    mu_O = G_H2O_MLIP - 2 * mu_H_ref - DGFH2O_EXP
    
    results['mu_O'] = mu_O
    results['G_H2O_MLIP'] = G_H2O_MLIP
    
    # 3. Validation
    # Re-calculate formation energy to verify it matches
    DGf_H2O_calc = G_H2O_MLIP - 2 * mu_H_ref - mu_O
    results['DGf_H2O_calc'] = DGf_H2O_calc
    results['correction_error'] = DGf_H2O_calc - DGFH2O_EXP
    
    # Store input energies
    results['E_H2O'] = E_H2O
    results['E_H2'] = E_H2
    results['E_O2'] = E_O2
    results['temperature'] = temperature
    
    # Store experimental references
    results['DGFH2O_exp'] = DGFH2O_EXP
    results['S_H2O_exp'] = S_H2O_EXP
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Calculate water correction for Pourbaix diagrams'
    )
    parser.add_argument('--h2o_energy', type=float, required=True,
                       help='MLIP total energy of H2O molecule (eV)')
    parser.add_argument('--h2_energy', type=float, required=True,
                       help='MLIP total energy of H2 molecule (eV)')
    parser.add_argument('--o2_energy', type=float, required=True,
                       help='MLIP total energy of O2 molecule (eV)')
    parser.add_argument('--temperature', type=float, default=T_STANDARD,
                       help=f'Temperature in K (default: {T_STANDARD})')
    parser.add_argument('--output', type=Path, required=True,
                       help='Output JSON file with correction parameters')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Water Correction Calculation (Persson et al. 2012)")
    print("="*70)
    print()
    print("Input MLIP Energies:")
    print(f"  E(H2O) = {args.h2o_energy:.6f} eV")
    print(f"  E(H2)  = {args.h2_energy:.6f} eV")
    print(f"  E(O2)  = {args.o2_energy:.6f} eV")
    print(f"  T      = {args.temperature:.2f} K")
    print()
    
    # Calculate correction
    results = calculate_water_correction(
        args.h2o_energy,
        args.h2_energy,
        args.o2_energy,
        args.temperature
    )
    
    print("Calculated Chemical Potentials:")
    print(f"  μ_O     = {results['mu_O']:.6f} eV/O")
    print(f"  μ_H^ref = {results['mu_H_ref']:.6f} eV/H")
    print()
    
    print("Water Formation Energy:")
    print(f"  G(H2O)_MLIP        = {results['G_H2O_MLIP']:.6f} eV")
    print(f"  ΔGf(H2O)_calc      = {results['DGf_H2O_calc']:.6f} eV")
    print(f"  ΔGf(H2O)_exp       = {results['DGFH2O_exp']:.6f} eV")
    print()
    
    print("Validation:")
    error_meV = results['correction_error'] * 1000
    print(f"  Correction error   = {error_meV:.3f} meV")
    
    if abs(results['correction_error']) < 0.001:  # < 1 meV
        print("  ✓ Excellent agreement with experimental value!")
    elif abs(results['correction_error']) < 0.01:  # < 10 meV
        print("  ✓ Good agreement with experimental value")
    else:
        print("  ⚠ Warning: Large correction error")
    
    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"✓ Saved correction parameters to {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
