"""
Molecular Fingerprints and Similarity Tool

Compute Morgan/ECFP fingerprints using RDKit's FingerprintGenerator API,
pairwise Tanimoto similarity, optional Butina clustering, and optional
heatmap visualization.

Usage:
    python compute_fingerprints.py --smiles "CCO" "CCCO" --output sim.json
    python compute_fingerprints.py --smiles_file compounds.smi --heatmap heatmap.png --output sim.json
    python compute_fingerprints.py --smiles_file compounds.smi --cluster --cluster_cutoff 0.7 --output clustered.json
    python compute_fingerprints.py --smiles_file compounds.smi --standardize parent --output sim_parent.json

SMILES file format:
    One molecule per line:
        SMILES[whitespace or tab]NAME
    NAME is optional. Lines starting with '#' and blank lines are ignored.

Requirements:
    - Conda environment: drugdisc-agent
    - Required packages: rdkit
    - Optional (for heatmap): numpy, matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


SmilesRecord = Tuple[str, Optional[str]]


def read_smiles_file(path: Path) -> List[SmilesRecord]:
    """Read a SMILES file with optional names."""
    records: List[SmilesRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else None
            records.append((smi, name))
    return records


def maybe_disable_rdkit_info_logs(disable: bool) -> None:
    """RDKit MolStandardize can emit lots of rdApp.info logs."""
    if not disable:
        return
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog("rdApp.info")
    except Exception:
        pass


def standardize_mol(mol: Chem.Mol, mode: str) -> Chem.Mol:
    """Apply an RDKit MolStandardize pipeline.

    Modes: none, cleanup, parent, uncharged, tautomer (cumulative).
    """
    if mode == "none":
        return mol

    from rdkit.Chem.MolStandardize import rdMolStandardize

    clean = rdMolStandardize.Cleanup(mol)
    if mode == "cleanup":
        return clean

    parent = rdMolStandardize.FragmentParent(clean)
    if mode == "parent":
        return parent

    uncharger = rdMolStandardize.Uncharger()
    uncharged = uncharger.uncharge(parent)
    if mode == "uncharged":
        return uncharged

    te = rdMolStandardize.TautomerEnumerator()
    canon = te.Canonicalize(uncharged)
    return canon


def make_morgan_generator(
    radius: int,
    fp_size: int,
    use_chirality: bool,
    use_features: bool,
    count_simulation: bool,
) -> Any:
    """Construct an RDKit FingerprintGenerator for Morgan fingerprints."""
    if use_features:
        invgen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        return rdFingerprintGenerator.GetMorganGenerator(
            radius=radius,
            fpSize=fp_size,
            includeChirality=use_chirality,
            countSimulation=count_simulation,
            atomInvariantsGenerator=invgen,
        )

    return rdFingerprintGenerator.GetMorganGenerator(
        radius=radius,
        fpSize=fp_size,
        includeChirality=use_chirality,
        countSimulation=count_simulation,
    )


def compute_fingerprints(
    records: Sequence[SmilesRecord],
    fpgen: Any,
    standardize: str,
) -> Tuple[List[Dict[str, Any]], List[Optional[Any]]]:
    """Compute fingerprints for SMILES records.

    Returns (compound_info_list, fingerprints_list) where fingerprint entries
    are ExplicitBitVect or None for invalid molecules.
    """
    info_list: List[Dict[str, Any]] = []
    fps: List[Optional[Any]] = []

    for smi, name in records:
        entry: Dict[str, Any] = {
            "input_smiles": smi,
            "input_name": name,
            "valid": False,
        }

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            entry["error"] = "MolFromSmiles returned None (parse failed)"
            info_list.append(entry)
            fps.append(None)
            continue

        try:
            std_mol = standardize_mol(mol, standardize)
        except Exception as e:
            entry["error"] = f"Standardization failed: {type(e).__name__}: {e}"
            info_list.append(entry)
            fps.append(None)
            continue

        try:
            can_smi = Chem.MolToSmiles(std_mol, canonical=True, isomericSmiles=True)
        except Exception as e:
            entry["error"] = f"MolToSmiles failed: {type(e).__name__}: {e}"
            info_list.append(entry)
            fps.append(None)
            continue

        fp = fpgen.GetFingerprint(std_mol)
        on_bits = int(fp.GetNumOnBits())
        fp_size = int(len(fp))

        entry.update({
            "name": name or can_smi,
            "smiles": can_smi,
            "valid": True,
            "num_on_bits": on_bits,
            "bit_density": round(on_bits / fp_size, 6) if fp_size else None,
        })
        info_list.append(entry)
        fps.append(fp)

    return info_list, fps


def pairwise_tanimoto_matrix(
    fps: Sequence[Optional[Any]],
    decimals: int = 4,
) -> List[List[Optional[float]]]:
    """Compute a full pairwise Tanimoto similarity matrix."""
    n = len(fps)
    matrix: List[List[Optional[float]]] = [[None] * n for _ in range(n)]

    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_indices]

    m = len(valid_fps)
    for ii in range(m):
        i = valid_indices[ii]
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[ii], valid_fps[ii:])
        for offset, sim in enumerate(sims):
            jj = ii + offset
            j = valid_indices[jj]
            val = round(float(sim), decimals)
            matrix[i][j] = val
            matrix[j][i] = val

    return matrix


def butina_cluster(
    fps: Sequence[Optional[Any]],
    similarity_cutoff: float,
) -> List[List[int]]:
    """Cluster compounds using Butina clustering.

    Accepts a similarity cutoff (converts internally: distance = 1 - similarity).
    """
    if not (0.0 <= similarity_cutoff <= 1.0):
        raise ValueError("similarity_cutoff must be in [0, 1]")

    valid_indices = [i for i, fp in enumerate(fps) if fp is not None]
    valid_fps = [fps[i] for i in valid_indices]
    n = len(valid_fps)

    if n == 0:
        return []

    dists: List[float] = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
        dists.extend([1.0 - float(s) for s in sims])

    from rdkit.ML.Cluster import Butina

    dist_thresh = 1.0 - similarity_cutoff
    clusters_raw = Butina.ClusterData(dists, n, dist_thresh, isDistData=True)

    clusters: List[List[int]] = []
    for cluster in clusters_raw:
        clusters.append([valid_indices[i] for i in cluster])

    return clusters


def save_heatmap(
    matrix: List[List[Optional[float]]],
    labels: Sequence[str],
    output_path: Path,
) -> None:
    """Save a similarity matrix heatmap as PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(matrix)
    data = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            v = matrix[i][j]
            data[i, j] = float(v) if v is not None else 0.0

    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.5)))
    im = ax.imshow(data, cmap="YlOrRd", vmin=0.0, vmax=1.0, aspect="equal")
    plt.colorbar(im, ax=ax, label="Tanimoto Similarity")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    trunc = [str(x)[:18] for x in labels]
    ax.set_xticklabels(trunc, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(trunc, fontsize=8)
    ax.set_title("Pairwise Tanimoto Similarity")

    if n <= 15:
        for i in range(n):
            for j in range(n):
                val = data[i, j]
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute Morgan fingerprints, pairwise Tanimoto similarity, "
                    "optional Butina clustering, and optional heatmap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", nargs="+", help="One or more SMILES strings")
    input_group.add_argument("--smiles_file", type=str, help="Path to SMILES file (.smi)")

    parser.add_argument("--radius", type=int, default=2,
                        help="Morgan radius (radius=2 corresponds to ECFP4-style neighborhoods)")
    parser.add_argument("--nbits", type=int, default=2048, help="Fingerprint size in bits")
    parser.add_argument("--use_chirality", action="store_true",
                        help="Include chirality in Morgan fingerprint")
    parser.add_argument("--use_features", action="store_true",
                        help="Use Feature Morgan (FCFP-like) invariants")
    parser.add_argument("--count_simulation", action="store_true",
                        help="Enable count simulation in bit vector fingerprints")

    parser.add_argument("--standardize",
                        choices=["none", "cleanup", "parent", "uncharged", "tautomer"],
                        default="none",
                        help="RDKit MolStandardize pipeline to apply before fingerprinting")
    parser.add_argument("--keep_rdkit_info_logs", action="store_true",
                        help="Do not disable RDKit rdApp.info logs "
                             "(MolStandardize can be noisy)")

    parser.add_argument("--no_matrix", action="store_true",
                        help="Skip computing the full similarity matrix")
    parser.add_argument("--cluster", action="store_true", help="Perform Butina clustering")
    parser.add_argument("--cluster_cutoff", type=float, default=0.7,
                        help="Butina clustering similarity cutoff (Tanimoto) in [0,1]")
    parser.add_argument("--heatmap", type=str,
                        help="Path to save similarity heatmap PNG (requires matrix)")
    parser.add_argument("--heatmap_max_n", type=int, default=250,
                        help="Max molecules allowed for heatmap rendering")

    parser.add_argument("--output", required=True, type=str, help="Path to save results JSON")

    args = parser.parse_args()

    records: List[SmilesRecord] = []
    if args.smiles:
        records = [(s, None) for s in args.smiles]
    else:
        records = read_smiles_file(Path(args.smiles_file))

    if not records:
        print("Error: No SMILES provided.", file=sys.stderr)
        sys.exit(1)

    maybe_disable_rdkit_info_logs(disable=(not args.keep_rdkit_info_logs))

    fpgen = make_morgan_generator(
        radius=args.radius,
        fp_size=args.nbits,
        use_chirality=args.use_chirality,
        use_features=args.use_features,
        count_simulation=args.count_simulation,
    )

    info_list, fps = compute_fingerprints(records, fpgen=fpgen, standardize=args.standardize)

    valid_count = sum(1 for x in info_list if x.get("valid"))
    print(f"RDKit version: {rdkit.__version__}")
    print(f"Computed fingerprints for {len(info_list)} compounds ({valid_count} valid).")

    try:
        fp_info = fpgen.GetInfoString()
    except Exception:
        fp_info = None

    matrix: Optional[List[List[Optional[float]]]] = None
    if not args.no_matrix:
        matrix = pairwise_tanimoto_matrix(fps)

    if args.heatmap:
        if matrix is None:
            print("Error: --heatmap requires the similarity matrix (remove --no_matrix).",
                  file=sys.stderr)
            sys.exit(2)
        if len(info_list) > args.heatmap_max_n:
            print(f"Error: heatmap requested for n={len(info_list)} which exceeds "
                  f"--heatmap_max_n={args.heatmap_max_n}.", file=sys.stderr)
            sys.exit(2)

        labels = [x.get("name") or x.get("input_name") or x.get("input_smiles")
                  for x in info_list]
        outpath = Path(args.heatmap)
        save_heatmap(matrix, labels, outpath)
        print(f"Saved heatmap to: {outpath}")

    clusters_out: Optional[List[Dict[str, Any]]] = None
    if args.cluster:
        clusters = butina_cluster(fps, similarity_cutoff=args.cluster_cutoff)
        names = [x.get("name") or x.get("input_name") or x.get("input_smiles")
                 for x in info_list]
        clusters_out = []
        for cid, members in enumerate(clusters):
            clusters_out.append({
                "cluster_id": cid,
                "indices": members,
                "members": [names[i] for i in members],
            })
        print(f"Butina clustering produced {len(clusters)} clusters "
              f"at cutoff={args.cluster_cutoff:.2f}")

    output: Dict[str, Any] = {
        "rdkit_version": rdkit.__version__,
        "fingerprint": {
            "type": "Morgan" + ("Feature" if args.use_features else ""),
            "radius": args.radius,
            "fp_size": args.nbits,
            "use_chirality": bool(args.use_chirality),
            "use_features": bool(args.use_features),
            "count_simulation": bool(args.count_simulation),
            "info_string": fp_info,
        },
        "standardization": {"mode": args.standardize},
        "compounds": info_list,
        "similarity_matrix": matrix,
    }
    if clusters_out is not None:
        output["clusters"] = clusters_out

    outpath = Path(args.output)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Saved results to: {outpath}")


if __name__ == "__main__":
    main()
