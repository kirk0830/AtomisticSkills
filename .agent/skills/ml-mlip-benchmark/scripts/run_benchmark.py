import argparse
import json
import logging
import os
import sys
import numpy as np

# ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from pymatgen.core import Structure

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_metrics(preds, targets):
    p = np.array(preds).flatten()
    t = np.array(targets).flatten()
    if len(p) == 0 or len(t) == 0:
        return {"mae": 0.0, "rmse": 0.0}
    mae = np.mean(np.abs(p - t))
    rmse = np.sqrt(np.mean((p - t)**2))
    return {"mae": float(mae), "rmse": float(rmse)}

def main():
    parser = argparse.ArgumentParser(description="Evaluate MLIP over labeled dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to JSON with labeled data")
    parser.add_argument("--model", type=str, required=True, help="Path or name of the model")
    parser.add_argument("--backend", type=str, required=True, choices=["mace", "fairchem", "matgl"])
    parser.add_argument("--task_name", type=str, default=None, help="Optional task name (omat_pbe, oc20, etc.)")
    parser.add_argument("--output", type=str, required=True, help="Output JSON for metrics and parity data")
    
    args = parser.parse_args()
    
    logger.info(f"Loading data from {args.data_path}...")
    with open(args.data_path, "r") as f:
        dataset = json.load(f)
    
    if not dataset:
        logger.error("Empty dataset provided.")
        sys.exit(1)
        
    logger.info(f"Loading {args.backend} model: {args.model}...")
    from src.utils.mlips.loader import load_wrapper
    wrapper = load_wrapper(args.backend, args.model, device="cuda", task_name=args.task_name, use_mcp_config=False)
    
    all_energy_preds = []
    all_energy_targs = []
    all_forces_preds = []
    all_forces_targs = []
    all_stress_preds = []
    all_stress_targs = []
    
    valid_data = []

    for i, data in enumerate(dataset):
        if "structure" not in data or ("energy" not in data and "forces" not in data):
            continue
            
        system = Structure.from_dict(data["structure"])
        num_atoms = len(system)
        
        try:
            pred = wrapper.static_calculation(system)
        except Exception as e:
            logger.warning(f"Prediction failed for structure {i}: {e}")
            continue

        valid_point = {"id": i, "num_atoms": num_atoms}
        
        # Energy
        if "energy" in data and data["energy"] is not None:
            # We standardize to energy per atom for comparison
            # data["energy"] is typically total energy string or float
            targ_e = float(data["energy"]) / num_atoms
            pred_e = float(pred["energy"]) / num_atoms
            all_energy_targs.append(targ_e)
            all_energy_preds.append(pred_e)
            valid_point["energy_target"] = targ_e
            valid_point["energy_pred"] = pred_e
            
        # Forces
        if "forces" in data and data["forces"] is not None:
            targ_f = np.array(data["forces"])
            pred_f = np.array(pred["forces"])
            if targ_f.shape == pred_f.shape:
                all_forces_targs.extend(targ_f.flatten())
                all_forces_preds.extend(pred_f.flatten())
                valid_point["forces_target"] = targ_f.tolist()
                valid_point["forces_pred"] = pred_f.tolist()
                
        # Stress
        if "stress" in data and data["stress"] is not None and "stress" in pred and pred["stress"] is not None:
            targ_s = np.array(data["stress"])
            pred_s = np.array(pred["stress"])
            # standardize formats if applicable
            all_stress_targs.extend(targ_s.flatten())
            all_stress_preds.extend(pred_s.flatten())
            valid_point["stress_target"] = targ_s.tolist()
            valid_point["stress_pred"] = pred_s.tolist()
            
        valid_data.append(valid_point)

    logger.info("Computing validation metrics...")
    
    metrics = {
        "energy": evaluate_metrics(all_energy_preds, all_energy_targs) if all_energy_preds else None,
        "forces": evaluate_metrics(all_forces_preds, all_forces_targs) if all_forces_preds else None,
        "stress": evaluate_metrics(all_stress_preds, all_stress_targs) if all_stress_preds else None
    }
    
    logger.info(f"Energy Metrics (eV/atom): {metrics['energy']}")
    logger.info(f"Forces Metrics (eV/Å):    {metrics['forces']}")
    if metrics['stress']:
         logger.info(f"Stress Metrics (eV/Å^3):  {metrics['stress']}")

    results = {
        "metrics": metrics,
        "data_points": valid_data
    }
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Benchmark completed successfully. Saved to {args.output}")

if __name__ == "__main__":
    main()
