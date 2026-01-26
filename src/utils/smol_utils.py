import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pymatgen.core import Structure
from pymatgen.entries.computed_entries import ComputedStructureEntry
from smol.cofe import ClusterSubspace, ClusterExpansion, StructureWrangler

# Monkey-patch pymatgen PeriodicSite for smol compatibility
from pymatgen.core.sites import PeriodicSite
if not hasattr(PeriodicSite, 'specie'):
    @property
    def get_specie(self):
        try:
            return self.species.elements[0]
        except Exception:
            raise AttributeError(f"specie not found on {type(self).__name__}")
    PeriodicSite.specie = get_specie
from smol.moca import Ensemble, Sampler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor

logger = logging.getLogger(__name__)

class SmolWrapper:
    """
    A wrapper around the smol package to facilitate common cluster expansion (CE)
    and Monte Carlo (MC) workflows.
    """
    
    def __init__(self):
        self.subspace: Optional[ClusterSubspace] = None
        self.wrangler: Optional[StructureWrangler] = None
        self.expansion: Optional[ClusterExpansion] = None
        
    def create_subspace(self, 
                        primordial_structure: Union[Structure, Dict, Atoms],
                        cutoffs: Dict[int, float],
                        max_cluster_size: int = 2,
                        basis_set: str = "chebyshev") -> str:
        """
        Create a cluster subspace from a primordial structure.
        """
        struct = self._ensure_pmg_structure(primordial_structure)
        
        # smol expects cutoffs as a dict, e.g., {2: 5.0, 3: 4.0}
        # If user only specifies one float, assume it's for pairs
        if isinstance(cutoffs, (float, int)):
            cutoffs = {i: float(cutoffs) for i in range(2, max_cluster_size + 1)}
            
        self.subspace = ClusterSubspace.from_cutoffs(
            struct,
            cutoffs=cutoffs,
            basis=basis_set
        )
        self.wrangler = StructureWrangler(self.subspace)
        return f"ClusterSubspace created with {len(self.subspace)} clusters."

    def add_training_data(self, data: Union[List[Dict[str, Any]], Dict[str, Any]]) -> str:
        """
        Add training data (structures and energies) to the wrangler.
        Supports data formats from Atomate2 and MatGL.
        """
        if self.wrangler is None:
            return "Error: ClusterSubspace must be created before adding training data."
            
        if isinstance(data, dict):
            if "results" in data:
                data_list = data["results"]
            else:
                data_list = [data]
        else:
            data_list = data
            
        count = 0
        for entry in data_list:
            try:
                struct_data = entry.get("final_structure") or entry.get("structure")
                energy = entry.get("final_energy") or entry.get("energy")
                
                if struct_data is None or energy is None:
                    continue
                
                struct = self._ensure_pmg_structure(struct_data)
                entry = ComputedStructureEntry(struct, energy)
                self.wrangler.add_entry(entry)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to add entry: {e}")
                
        return f"Successfully added {count} entries to the wrangler."

    def fit_expansion(self, method: str = "ls", **kwargs) -> Dict[str, Any]:
        """
        Fit the cluster expansion using the data in the wrangler.
        methods: 'ls' (least squares), 'lasso', 'ridge'
        """
        if self.wrangler is None or len(self.wrangler.entries) == 0:
            return {"error": "No training data available for fitting."}
            
        # Get feature matrix and target values
        feature_matrix = self.wrangler.feature_matrix
        # By default, smol uses 'energy' as the property name for ComputedStructureEntry
        try:
            energies = self.wrangler.get_property_vector('energy')
        except Exception:
            # Fallback for older versions or different property names
            energies = np.array([entry.energy for entry in self.wrangler.entries])
        
        if method == "lasso":
            model = Lasso(**kwargs)
        elif method == "ridge":
            model = Ridge(**kwargs)
        else:
            model = LinearRegression()
            
        model.fit(feature_matrix, energies)
        eci = model.coef_
        # Intercept is handled by smol if we include the empty cluster
        # but scikit-learn fits it separately.
        # smol's ClusterExpansion takes eci including the empty cluster (constant).
        # Typically the first feature is the constant.
        
        self.expansion = ClusterExpansion(self.subspace, coefficients=eci)
        
        return {
            "status": "success",
            "rmse": float(np.sqrt(np.mean((model.predict(feature_matrix) - energies)**2))),
            "coef_count": len(eci)
        }

    def save_ce(self, file_path: str) -> str:
        """
        Save the fitted cluster expansion to a file.
        """
        if self.expansion is None:
            return "Error: No ClusterExpansion to save."
        
        self.expansion.save(file_path, strict=False)
        return f"ClusterExpansion saved to {file_path}"

    def load_ce(self, file_path: str) -> str:
        """
        Load a cluster expansion from a file.
        """
        try:
            self.expansion = ClusterExpansion.load(file_path)
            self.subspace = self.expansion.cluster_subspace
            # If subspace is loaded, we can recreate a wrangler if needed
            self.wrangler = StructureWrangler(self.subspace)
            return f"ClusterExpansion loaded from {file_path}"
        except Exception as e:
            return f"Error loading ClusterExpansion: {str(e)}"

    def run_mc(self, 
               supercell_matrix: List[List[int]],
               temperature: float,
               steps: int,
               ensemble_type: str = "canonical",
               initial_occupancies: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation.
        """
        if self.expansion is None:
            return {"error": "ClusterExpansion must be fitted before running MC."}
            
        # Create Ensemble
        ensemble = Ensemble.from_cluster_expansion(
            self.expansion,
            supercell_matrix=supercell_matrix
        )
        
        # Create Sampler
        # For canonical ensemble (fixed composition), use 'swap' kernels
        # For semi-grand, use 'flip' kernels (default in smol if chemical_potentials provided)
        step_type = 'swap' if ensemble_type == 'canonical' else 'flip'
        
        sampler = Sampler.from_ensemble(
            ensemble,
            temperature=temperature,
            step_type=step_type
        )
        
        # Run simulation
        if initial_occupancies is None:
            # Try to get occupancies from the processor's default structure
            # which is the supercell used for the ensemble.
            try:
                initial_occupancies = ensemble.processor.occupancy_from_structure(
                    ensemble.processor.structure
                )
            except Exception:
                # Fallback: random occupancies (might not match composition)
                num_sites = ensemble.processor.num_sites
                # This is just a placeholder, should ideally match composition
                initial_occupancies = np.zeros(num_sites, dtype=int)
        
        sampler.run(steps, initial_occupancies=initial_occupancies)
        
        # Get results
        samples = sampler.samples
        final_occu = samples.get_occupancies()[-1]
        
        return {
            "status": "success",
            "final_energy": float(samples.get_energies()[-1]),
            "average_energy": float(np.mean(samples.get_energies())),
            "final_structure": self._occu_to_structure(ensemble, final_occu).as_dict()
        }

    def _ensure_pmg_structure(self, structure_data: Any) -> Structure:
        if isinstance(structure_data, Structure):
            return structure_data
        if isinstance(structure_data, dict):
            # Check if it's a pymatgen dict
            if "lattice" in structure_data:
                return Structure.from_dict(structure_data)
            # Check if it looks like an ASE dict (not standard but possible)
            if "numbers" in structure_data or "symbols" in structure_data:
                 atoms = Atoms.from_dict(structure_data)
                 return AseAtomsAdaptor.get_structure(atoms)
        if isinstance(structure_data, Atoms):
            return AseAtomsAdaptor.get_structure(structure_data)
        if isinstance(structure_data, str) and os.path.exists(structure_data):
            return Structure.from_file(structure_data)
        raise ValueError(f"Unsupported structure format: {type(structure_data)}")

    def _occu_to_structure(self, ensemble: Ensemble, occupancies: np.ndarray) -> Structure:
        # Use ensemble's processor to get structure from occupancy
        processor = ensemble.processor
        return processor.structure_from_occupancy(occupancies)

    def generate_sqs(self, 
                     supercell_matrix: List[List[int]],
                     target_concentrations: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate SQS structure.
        """
        if self.subspace is None:
            return {"error": "ClusterSubspace must be created first."}
            
        # smol handles SQS through samplers with target correlation/composition
        # Or using specialized tools if available.
        # Let's use the Sampler approach for most robust generation
        # or check if there's a direct SQS tool.
        # Smol has 'moca.SQSRunner' or similar?
        # Looking at members: None directly named SQS.
        # Usually one runs MC at high T or target correlation.
        
        return {"error": "SQS generation utility not fully implemented yet."}
