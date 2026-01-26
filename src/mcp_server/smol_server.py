import sys
import os
import io
import logging
import contextlib
import warnings
import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from typing import Dict, Any, Optional, List, Union

# Silence warnings to prevent protocol pollution
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

# --- ROBUST STDOUT ISOLATION ---
try:
    mcp_stdout_fd = os.dup(1)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, 1)
    sys.stdout = io.TextIOWrapper(
        os.fdopen(mcp_stdout_fd, 'wb', buffering=0), 
        encoding='utf-8', 
        line_buffering=True
    )
except Exception:
    pass
# -------------------------------

logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger("Smol-Server")

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Initialize FastMCP server
mcp = FastMCP("Smol")
from src.utils.smol_utils import SmolWrapper
from src.utils.research_utils import get_current_research_dir

# Global wrapper instance
wrapper = SmolWrapper()

@mcp.tool()
def train_cluster_expansion(
    primordial_structure: Union[Dict[str, Any], str],
    training_data: Union[List[Dict[str, Any]], Dict[str, Any]],
    cutoffs: Dict[int, float],
    max_cluster_size: int = 2,
    basis_set: str = "chebyshev",
    fit_method: str = "ls",
    alpha: float = 1.0,
    ce_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Create a cluster subspace, add training data, and fit a cluster expansion.
    
    This tool orchestrates the full training workflow for a cluster expansion model.
    The resulting Cluster Expansion is saved to disk.
    
    Args:
        primordial_structure: Primordial disordered structure (dict or file path).
        training_data: List of calculation results or a single result dict (from Atomate2/MatGL).
                       Must contain 'structure' and 'energy' labels.
        cutoffs: Dict of cluster sizes to cutoff distances (e.g., {2: 5.0, 3: 4.0}).
        max_cluster_size: Maximum cluster size to include.
        basis_set: Basis set type ("chebyshev", "sinusoid", "indicator").
        fit_method: Fitting method ('ls', 'lasso', 'ridge').
        alpha: Regularization parameter for Lasso/Ridge.
        ce_file: Optional path to save the CE. If not provided, it will be saved 
                 in the current research directory as 'cluster_expansion.json'.
        
    Returns:
        Dictionary with fitting results (RMSE, coefficient count).
    """
    global wrapper
    with contextlib.redirect_stdout(sys.stderr):
        try:
            # Determine save path
            if not ce_file:
                ce_dir = get_current_research_dir() / "smol"
                os.makedirs(ce_dir, exist_ok=True)
                ce_file = str(ce_dir / "cluster_expansion.json")
            
            # 1. Create subspace
            subspace_msg = wrapper.create_subspace(
                primordial_structure=primordial_structure,
                cutoffs=cutoffs,
                max_cluster_size=max_cluster_size,
                basis_set=basis_set
            )
            
            # 2. Add training data
            data_msg = wrapper.add_training_data(training_data)
            
            # 3. Fit expansion
            kwargs = {}
            if fit_method in ["lasso", "ridge"]:
                kwargs["alpha"] = alpha
            
            fit_res = wrapper.fit_expansion(method=fit_method, **kwargs)
            
            # 4. Save CE
            save_msg = wrapper.save_ce(ce_file)
            
            # Add messages to result
            if "status" in fit_res:
                fit_res["subspace_status"] = subspace_msg
                fit_res["data_status"] = data_msg
                fit_res["save_status"] = save_msg
                fit_res["ce_file"] = ce_file
            
            return fit_res
            
        except Exception as e:
            return {"error": f"Training failed: {str(e)}"}

@mcp.tool()
def run_monte_carlo(
    supercell_matrix: List[List[int]],
    temperature: float,
    steps: int,
    ensemble_type: str = "canonical",
    ce_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a Monte Carlo simulation using a Cluster Expansion.
    
    Args:
        supercell_matrix: 3x3 diagonal matrix for supercell expansion.
        temperature: Simulation temperature in K.
        steps: Number of MC steps.
        ensemble_type: 'canonical' or 'semigrand'.
        ce_file: Optional path to the Cluster Expansion file. If not provided,
                 it will attempt to load 'cluster_expansion.json' from the current 
                 research directory or use the one currently in memory.
        
    Returns:
        Simulation results including final energy and structure.
    """
    global wrapper
    with contextlib.redirect_stdout(sys.stderr):
        try:
            # 1. Load CE if needed
            if ce_file:
                load_msg = wrapper.load_ce(ce_file)
                logger.info(load_msg)
            elif wrapper.expansion is None:
                # Try default path
                default_ce = get_current_research_dir() / "smol" / "cluster_expansion.json"
                if default_ce.exists():
                    wrapper.load_ce(str(default_ce))
                else:
                    return {"error": "No ClusterExpansion loaded or found at default path. Please train one first."}
            
            return wrapper.run_mc(
                supercell_matrix=supercell_matrix,
                temperature=temperature,
                steps=steps,
                ensemble_type=ensemble_type
            )
        except Exception as e:
            return {"error": f"Monte Carlo failed: {str(e)}"}

if __name__ == "__main__":
    mcp.run()
