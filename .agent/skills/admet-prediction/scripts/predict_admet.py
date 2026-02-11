"""
ADMET-Relevant Descriptor Screening (RDKit)

Compute physicochemical descriptors and rule-based drug-likeness heuristics
from SMILES. Intended for early-stage triage, not experimental ADMET prediction.

Computes:
- Lipinski Ro5 (heuristic)
- Veber (heuristic)
- QED (drug-likeness score)
- Core RDKit physchem descriptors (MW, cLogP, TPSA, HBD/HBA, etc.)

Usage:
    python predict_admet.py --smiles "CCO" --output results.json
    python predict_admet.py --smiles "CCO" "c1ccccc1" --output results.json
    python predict_admet.py --smiles_file compounds.smi --output results.json
    python predict_admet.py --smiles "OC(=O)P(=O)(O)O" --include_sandp_tpsa --output out.json

Input file format (SMILES file):
    - One molecule per line
    - First token is SMILES; remainder is the name (optional)
    - Lines starting with '#' are ignored

Requirements:
    - Conda environment: drugdisc-agent
    - Required packages: rdkit
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rdkit import Chem, rdBase
from rdkit.Chem import Crippen, Descriptors, Lipinski, QED


def _round(x: float, ndigits: int = 3) -> float:
    return float(round(x, ndigits))


def parse_smiles_file(path: str) -> List[Tuple[str, Optional[str]]]:
    """Parse a SMILES file into (smiles, name) tuples."""
    out: List[Tuple[str, Optional[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            smi = parts[0]
            name = parts[1].strip() if len(parts) > 1 else None
            out.append((smi, name))
    return out


def compute_admet_descriptors(
    smiles: str,
    name: Optional[str] = None,
    include_sandp_tpsa: bool = False,
    ndigits: int = 3,
) -> Dict[str, Any]:
    """Compute ADMET-relevant descriptors and heuristic filters for a SMILES string."""
    warnings: List[str] = []
    if "." in smiles:
        warnings.append(
            "SMILES contains disconnected fragments ('.'). "
            "Consider desalting/standardization for library screening."
        )

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {
            "name": name or smiles,
            "smiles_input": smiles,
            "smiles": None,
            "valid": False,
            "error": "Invalid SMILES (RDKit MolFromSmiles returned None).",
            "warnings": warnings,
        }

    canonical = Chem.MolToSmiles(mol, canonical=True)

    mol_wt = _round(Descriptors.MolWt(mol), ndigits)
    exact_mol_wt = _round(Descriptors.ExactMolWt(mol), ndigits)
    logp = _round(Crippen.MolLogP(mol), ndigits)
    mr = _round(Crippen.MolMR(mol), ndigits)
    tpsa = _round(
        Descriptors.TPSA(mol, includeSandP=include_sandp_tpsa), ndigits
    )

    hbd = int(Lipinski.NumHDonors(mol))
    hba = int(Lipinski.NumHAcceptors(mol))
    rotatable = int(Lipinski.NumRotatableBonds(mol))
    ring_count = int(Lipinski.RingCount(mol))
    aromatic_rings = int(Lipinski.NumAromaticRings(mol))
    heavy_atoms = int(mol.GetNumHeavyAtoms())
    fraction_csp3 = _round(Descriptors.FractionCSP3(mol), ndigits)

    qed_value = _round(QED.qed(mol), ndigits)

    # Lipinski Ro5, uses average MW per the original paper
    lipinski_details = {
        "MW_le_500": mol_wt <= 500.0,
        "LogP_le_5": logp <= 5.0,
        "HBD_le_5": hbd <= 5,
        "HBA_le_10": hba <= 10,
    }
    lipinski_violations = sum(1 for ok in lipinski_details.values() if not ok)
    lipinski_pass = lipinski_violations <= 1

    # Veber (2002), primary: RB<=10 and TPSA<=140; alternative: HBD+HBA<=12
    veber_rb_ok = rotatable <= 10
    veber_tpsa_ok = tpsa <= 140.0
    veber_hbond_sum_ok = (hbd + hba) <= 12
    veber_pass_primary = veber_rb_ok and veber_tpsa_ok

    return {
        "name": name or canonical,
        "smiles_input": smiles,
        "smiles": canonical,
        "valid": True,
        "warnings": warnings,
        "descriptors": {
            "mol_wt": mol_wt,
            "exact_mol_wt": exact_mol_wt,
            "logp_wc": logp,
            "molar_refractivity": mr,
            "tpsa": tpsa,
            "tpsa_includes_sandp": bool(include_sandp_tpsa),
            "hbd": hbd,
            "hba": hba,
            "rotatable_bonds": rotatable,
            "ring_count": ring_count,
            "aromatic_rings": aromatic_rings,
            "heavy_atoms": heavy_atoms,
            "fraction_csp3": fraction_csp3,
            "qed": qed_value,
        },
        "heuristics": {
            "lipinski_ro5": {
                "violations": int(lipinski_violations),
                "pass": bool(lipinski_pass),
                "details": lipinski_details,
                "notes": "Pass defined as <= 1 violation (common screening convention). "
                "Uses average MW (Descriptors.MolWt) per the original Ro5 paper.",
            },
            "veber": {
                "pass": bool(veber_pass_primary),
                "rotatable_le_10": bool(veber_rb_ok),
                "tpsa_le_140": bool(veber_tpsa_ok),
                "hbd_plus_hba_le_12": bool(veber_hbond_sum_ok),
                "notes": "Primary pass uses RB<=10 and TPSA<=140; "
                "HBD+HBA<=12 is reported as an alternative criterion.",
            },
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute RDKit physchem descriptors + Ro5/Veber/QED heuristics from SMILES."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", nargs="+", help="One or more SMILES strings.")
    input_group.add_argument(
        "--smiles_file",
        help="File with one SMILES per line (optional name after whitespace).",
    )
    parser.add_argument(
        "--output",
        help="Path to save results as JSON. If omitted, JSON is written to stdout.",
        default=None,
    )
    parser.add_argument(
        "--include_sandp_tpsa",
        action="store_true",
        help="Include S/P contributions in TPSA (RDKit option).",
    )
    parser.add_argument(
        "--ndigits",
        type=int,
        default=3,
        help="Decimal digits for rounding floats in JSON output (default: 3).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.smiles:
        smiles_list: List[Tuple[str, Optional[str]]] = [
            (s, None) for s in args.smiles
        ]
    else:
        smiles_list = parse_smiles_file(args.smiles_file)

    meta = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rdkit_version": getattr(rdBase, "rdkitVersion", "unknown"),
        "settings": {
            "include_sandp_tpsa": bool(args.include_sandp_tpsa),
            "ndigits": int(args.ndigits),
            "lipinski_ro5_thresholds": {"MW": 500, "LogP": 5, "HBD": 5, "HBA": 10},
            "veber_thresholds": {
                "rotatable_bonds": 10,
                "tpsa": 140,
                "hbd_plus_hba": 12,
            },
        },
        "n_input": len(smiles_list),
    }

    results: List[Dict[str, Any]] = []
    for smi, name in smiles_list:
        results.append(
            compute_admet_descriptors(
                smiles=smi,
                name=name,
                include_sandp_tpsa=args.include_sandp_tpsa,
                ndigits=args.ndigits,
            )
        )

    valid = [r for r in results if r.get("valid")]
    lip_pass = sum(1 for r in valid if r["heuristics"]["lipinski_ro5"]["pass"])
    veber_pass = sum(1 for r in valid if r["heuristics"]["veber"]["pass"])
    print(f"Analyzed {len(results)} compound(s).", file=sys.stderr)
    print(f"Valid: {len(valid)}/{len(results)}", file=sys.stderr)
    if valid:
        print(f"Lipinski Ro5 pass: {lip_pass}/{len(valid)}", file=sys.stderr)
        print(f"Veber pass: {veber_pass}/{len(valid)}", file=sys.stderr)

    payload = {"meta": meta, "results": results}

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)
        print(f"Saved JSON to {args.output}", file=sys.stderr)
    else:
        json.dump(payload, sys.stdout, indent=4)
        sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
