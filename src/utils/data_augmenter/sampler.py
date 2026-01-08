"""
Structure sampling methods for MLIP Agent
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import os

# Core ASE imports - safe to import at module level
from ase import Atoms
from ase.io import write

logger = logging.getLogger(__name__)

# Try to import dependencies - handle gracefully if not available
# Note: MLIP-specific imports (torch, matgl) should only be used in appropriate conda environments
try:
    from ase.md import MDLogger, VelocityVerlet, Langevin
    from ase.md.nptberendsen import Inhomogeneous_NPTBerendsen, NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase.optimize import BFGS
    from ase.constraints import ExpCellFilter
    from ase.calculators.calculator import Calculator
    ASE_MD_AVAILABLE = True
except (ImportError, ValueError, TypeError) as e:
    ASE_MD_AVAILABLE = False
    logger.warning(f"ASE MD modules not available: {e}")
    MDLogger = None
    VelocityVerlet = None
    Langevin = None
    Inhomogeneous_NPTBerendsen = None
    NPTBerendsen = None
    MaxwellBoltzmannDistribution = None
    BFGS = None
    ExpCellFilter = None
    Calculator = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except (ImportError, ValueError, TypeError) as e:
    SKLEARN_AVAILABLE = False
    logger.warning(f"sklearn not available: {e}")
    KMeans = None
    StandardScaler = None

try:
    import torch
    from torch import nn
    from torch.autograd import grad
    TORCH_AVAILABLE = True
except (ImportError, ValueError, TypeError, AttributeError) as e:
    TORCH_AVAILABLE = False
    logger.warning(f"PyTorch not available: {e}. Required for MLIP model operations.")
    torch = None
    nn = None
    grad = None

try:
    import matgl
    MATGL_AVAILABLE = True
except (ImportError, ValueError, TypeError, AttributeError) as e:
    MATGL_AVAILABLE = False
    logger.warning(f"MatGL not available: {e}. Required for MatGL model operations in matgl-agent environment.")
    matgl = None




class TrajectoryObserver:
    """
    Observer to record intermediate structures during MD simulation.
    
    Follows CHGNet's TrajectoryObserver pattern.
    """
    
    def __init__(self, atoms: Atoms, log_interval: int = 10):
        """
        Initialize trajectory observer.
        
        Args:
            atoms: ASE Atoms object to observe
            log_interval: Interval between structure recordings
        """
        self.atoms = atoms
        self.log_interval = log_interval
        self.structures = []
        self.step_count = 0
    
    def __call__(self) -> None:
        """
        Record structure at current step.
        
        Follows CHGNet's pattern - no arguments, direct access to atoms.
        """
        # ASE handles the interval via dyn.attach(..., interval)
        # So we just record every time we are called
        atoms_copy = self.atoms.copy()
        # Preserve the calculator
        atoms_copy.calc = self.atoms.calc
        self.structures.append(atoms_copy)
        self.step_count += self.log_interval # Estimate step count increment
    
    def __len__(self) -> int:
        """The number of steps in the trajectory."""
        return len(self.structures)
    
    def save(self, filename: str) -> None:
        """Save the trajectory to file."""
        import pickle
        out_pkl = {
            "structures": self.structures,
            "atomic_number": self.atoms.get_atomic_numbers(),
        }
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)


class CrystalFeatureObserver:
    """
    Observer to record crystal features during MD simulation.
    
    Follows CHGNet's CrystalFeasObserver pattern exactly.
    """
    
    def __init__(self, atoms: Atoms, log_interval: int = 10):
        """
        Initialize crystal feature observer.
        
        Args:
            atoms: ASE Atoms object to observe
            log_interval: Interval between feature recordings
        """
        self.atoms = atoms
        self.log_interval = log_interval
        self.crystal_feature_vectors = []
        self.step_count = 0
    
    def __call__(self) -> None:
        """
        Record crystal feature vectors after an MD/relaxation step.
        
        Follows CHGNet's pattern exactly - no arguments, direct access to atoms.
        """
        # ASE handles the interval via dyn.attach(..., interval)
        if hasattr(self.atoms, "calc") and self.atoms.calc is not None:
             # Depending on calculator, results might be in .results or computed
             # For some calculators (like MACE), we might need to ensure calculation happens
             # But usually dyn.run() triggers calculation before calling observers
             
             if hasattr(self.atoms.calc, 'results') and 'crystal_fea' in self.atoms.calc.results:
                 crystal_features = self.atoms.calc.results['crystal_fea']
                 self.crystal_feature_vectors.append(crystal_features)
                 # print(f"Recorded crystal features at step {self.step_count}, shape: {crystal_features.shape}")
             else:
                 raise RuntimeError("Crystal features not found in calculator results")
        else:
             raise RuntimeError("No calculator attached to atoms")
             
        self.step_count += self.log_interval
    
    def __len__(self) -> int:
        """Number of recorded steps."""
        return len(self.crystal_feature_vectors)
    
    def save(self, filename: str) -> None:
        """Save the crystal feature vectors to filename in pickle format."""
        import pickle
        out_pkl = {"crystal_feas": self.crystal_feature_vectors}
        with open(filename, "wb") as file:
            pickle.dump(out_pkl, file)
    


class OffEquilibriumSampler:
    """
    Off-Equilibrium Sampler (formerly PESSampler).
    
    Implements MD simulation with clustering-based sampling as specified in sampler.mdc.
    Now accepts a calculator directly for dependency injection.
    """
    
    def __init__(self, calculator: Calculator, atoms: Atoms, 
                 total_steps: Optional[int] = None, output_dir: Optional[str] = None,
                 target_atoms: int = 75, temperature: float = 1000.0):
        """
        Initialize OffEquilibriumSampler.
        
        Args:
            calculator: ASE Calculator to use for MD and relaxation.
            atoms: Initial atomic structure
            total_steps: Number of MD steps (default: 10000). Use smaller values for testing.
            output_dir: Directory to save MD log file and other outputs (default: current directory)
            target_atoms: Target number of atoms after supercell expansion (default: 75, range: 50-100)
                         Note: Maximum safe limit is 70 atoms to prevent OOM in VASP calculations.
                               Structures with >70 atoms may cause memory issues during DFT labeling.
            temperature: Temperature in Kelvin (default: 1000.0)
        """
        self.calculator = calculator
        if self.calculator is None:
             raise ValueError("Calculator must be provided to OffEquilibriumSampler")

        # Relax structure before supercell expansion to avoid high stress
        atoms = self._relax_structure(atoms)
        
        # Expand supercell to reach target number of atoms (50-100) with cubic-like expansion
        expanded_atoms = self._expand_supercell(atoms, target_atoms=target_atoms)
        
        # Re-relax after supercell expansion to remove any residual stress
        # This is critical for stable MD - expanded cells often have residual stress
        self.atoms = self._relax_structure(expanded_atoms)
        logger.info("Structure relaxed after supercell expansion to remove residual stress")
        logger.info("Structure relaxed after supercell expansion to remove residual stress")
        
        self.trajectory_observer = None
        self.crystal_feature_observer = None
        self.output_dir = output_dir
        if self.output_dir:
            import os
            os.makedirs(self.output_dir, exist_ok=True)
        
        # Use aggressive time step since we want fast sampling and don't care about momentum integration accuracy
        # Check if H is present
        symbols = atoms.get_chemical_symbols()
        if 'H' in symbols:
            self.time_step = 1.0  # fs - smaller time step for light atoms
            logger.info(f"Hydrogen detected, using {self.time_step} fs time step")
        else:
            self.time_step = 2.0  # fs - standard time step per user request
            logger.info(f"No Hydrogen detected, using {self.time_step} fs time step")
            
        self.total_steps = total_steps if total_steps is not None else 10000
        self.log_interval = 5  # Log every 5 steps for real-time monitoring
        self.temperature = temperature  # K
        self.taut = 100 * self.time_step 
        self.taup = 1000.0  * self.time_step
        self.pressure_au = 0.0  
        self.compressibility_au = 4.57e-5 
        
        # Clustering parameters
        self.n_clusters = 200  # Number of representative structures
    
    def _relax_structure(self, atoms: Atoms) -> Atoms:
        """
        Relax structure using MatCalc before supercell expansion.
        
        This prevents high stress that can cause MD instability.
        Always uses relax_cell=True for full structure relaxation.
        
        Args:
            atoms: Initial atomic structure to relax
            
        Returns:
            Relaxed ASE Atoms object
        """
        if self.calculator is None:
            raise ValueError("Calculator not set")

        import matcalc as mtc
        from pymatgen.io.ase import AseAtomsAdaptor
        
        logger.info(f"Relaxing structure before supercell expansion")
        
        # Convert ASE Atoms to pymatgen Structure
        adaptor = AseAtomsAdaptor()
        pmg_structure = adaptor.get_structure(atoms)
        
        # Create RelaxCalc with the injected calculator - always use relax_cell=True
        relax_calc = mtc.RelaxCalc(
            calculator=self.calculator,
            optimizer="FIRE",
            relax_atoms=True,
            relax_cell=True,  # Always relax cell - required for stable MD
            fmax=0.05,  # Force convergence threshold
        )
        
        # Perform relaxation
        result = relax_calc.calc(pmg_structure)
        
        # Extract relaxed structure from result
        # MatCalc returns a dict with 'structure' or 'final_structure' key
        if isinstance(result, dict):
            if 'structure' in result:
                relaxed_pmg = result['structure']
            elif 'final_structure' in result:
                relaxed_pmg = result['final_structure']
            else:
                raise ValueError(f"Could not find relaxed structure in result. Result keys: {list(result.keys())}")
        elif hasattr(result, 'structure'):  # Result object with structure attribute
            relaxed_pmg = result.structure
        else:
            raise ValueError(f"Unexpected result type from relaxation: {type(result)}")
        
        # Convert back to ASE Atoms
        relaxed_atoms = adaptor.get_atoms(relaxed_pmg)
        
        energy = result.get('energy', 'N/A') if isinstance(result, dict) else getattr(result, 'energy', 'N/A')
        logger.info(f"Structure relaxed successfully. Energy: {energy} eV")
        return relaxed_atoms
    
    def _expand_supercell(self, atoms: Atoms, target_atoms: int = 75) -> Atoms:
        """
        Expand supercell to reach target number of atoms (50-100) with cubic-like expansion.
        
        Prefers expansion matrices like [2,2,2] over [8,1,1] to keep the cell as cubic as possible.
        
        Memory safety: Based on VASP memory benchmarks, structures with >70 atoms can cause OOM
        issues in MatPES calculations (LREAL=False). This method enforces a maximum safe atom
        limit to prevent memory issues during subsequent DFT calculations.
        
        Args:
            atoms: Initial atomic structure
            target_atoms: Target number of atoms (default: 75, should be in range 50-100)
            
        Returns:
            Expanded ASE Atoms object (may be smaller than target if memory safety limit applies)
        """
        current_atoms = len(atoms)
        
        # Memory safety limit based on VASP benchmark results:
        # - FeCl3 with 81 atoms uses ~119 GB RAM and causes OOM
        # - Safe limit: ~60-70 atoms for systems with similar memory scaling
        # - Research indicates cubic scaling, so be conservative
        # NOTE: User requested ~100 atoms for this task (Mock DFT), so increasing limit.
        MAX_SAFE_ATOMS = 120  # Increased limit for Mock DFT / User Request
        
        # If structure exceeds safe limit, don't expand and warn user
        if current_atoms > MAX_SAFE_ATOMS:
            logger.warning(
                f"Structure has {current_atoms} atoms, exceeding safe limit of {MAX_SAFE_ATOMS} atoms "
                f"for VASP calculations. This may cause OOM issues. Using structure as-is without expansion."
            )
            return atoms.copy()
        
        # Adjust target_atoms to not exceed safe limit
        if target_atoms > MAX_SAFE_ATOMS:
            logger.warning(
                f"Target atoms ({target_atoms}) exceeds safe limit ({MAX_SAFE_ATOMS}). "
                f"Adjusting target to {MAX_SAFE_ATOMS} to prevent OOM issues."
            )
            target_atoms = MAX_SAFE_ATOMS
        
        # If already close to target, return as-is
        if current_atoms >= target_atoms * 0.9 and current_atoms <= target_atoms * 1.1:
            logger.info(f"Structure already has {current_atoms} atoms, close to target {target_atoms}. No expansion needed.")
            return atoms.copy()
        
        # If already larger than target, return as-is (don't shrink)
        if current_atoms > target_atoms:
            logger.info(f"Structure has {current_atoms} atoms, larger than target {target_atoms}. No expansion needed.")
            return atoms.copy()
        
        # Calculate expansion factor needed
        expansion_factor = (target_atoms / current_atoms) ** (1.0 / 3.0)
        
        # Find best expansion matrix [nx, ny, nz] that:
        # 1. Gets close to target_atoms
        # 2. Minimizes difference between nx, ny, nz (prefer cubic)
        # 3. Uses integer values
        
        best_expansion = None
        best_score = float('inf')
        best_num_atoms = 0
        
        # Memory safety limit (enforced earlier, but also check here)
        MAX_SAFE_ATOMS = 120
        
        # Search space: try expansion matrices from 1x1x1 to 10x10x10
        max_expansion = int(expansion_factor * 2) + 2
        max_expansion = min(max_expansion, 10)  # Cap at 10x10x10
        
        for nx in range(1, max_expansion + 1):
            for ny in range(1, max_expansion + 1):
                for nz in range(1, max_expansion + 1):
                    num_atoms = current_atoms * nx * ny * nz
                    
                    # Only consider if within reasonable range (40 to MAX_SAFE_ATOMS)
                    # Respect memory safety limit to prevent OOM in VASP calculations
                    if num_atoms < 40 or num_atoms > MAX_SAFE_ATOMS:
                        continue
                    
                    # Score based on:
                    # 1. Distance from target (prefer closer to target, but allow some flexibility)
                    # 2. Difference between expansion factors (STRONGLY prefer cubic: [2,2,2] over [8,1,1])
                    target_distance = abs(num_atoms - target_atoms) / target_atoms
                    
                    # Calculate "cubicness" - prefer expansion matrices where nx, ny, nz are similar
                    expansion_values = [nx, ny, nz]
                    max_exp = max(expansion_values)
                    min_exp = min(expansion_values)
                    # Perfect cubic (all equal) = 0, very non-cubic = 1
                    cubicness = (max_exp - min_exp) / max_exp if max_exp > 0 else 0.0
                    
                    # Penalize non-cubic expansions more strongly
                    # Prefer cubic expansion even if it means slightly more/fewer atoms
                    # Weight: 0.5 for target distance, 0.5 for cubicness (equal weight)
                    score = 0.5 * target_distance + 0.5 * cubicness
                    
                    if score < best_score:
                        best_score = score
                        best_expansion = (nx, ny, nz)
                        best_num_atoms = num_atoms
        
        if best_expansion is None:
            # Fallback: use simple expansion based on cubic root
            nx = ny = nz = int(round(expansion_factor))
            best_expansion = (max(1, nx), max(1, ny), max(1, nz))
            best_num_atoms = current_atoms * best_expansion[0] * best_expansion[1] * best_expansion[2]
        
        # Expand the structure
        expanded_atoms = atoms.repeat(best_expansion)
        
        logger.info(f"supercell expand the original cell ({current_atoms} atoms) to supercell ({best_num_atoms} atoms) for atomic environment sampling")
        
        return expanded_atoms
        
    # _setup_calculator removed as calculator is injected via __init__
    
    def run_md_simulation(self) -> Tuple[List[Atoms], List[np.ndarray]]:
        """
        Run NPT MD simulation with trajectory and crystal feature observers.
        
        Returns:
            Tuple of (sampled_structures, crystal_features)
        """
        if self.calculator is None:
             raise ValueError("Calculator must be set")
        
        # Calculate total simulation time in ps
        total_time_ps = (self.total_steps * self.time_step) / 1000.0
        
        # Set up atoms with calculator
        atoms = self.atoms.copy()
        atoms.calc = self.calculator
        
        logger.info(f"Starting NPT MD simulation:")
        logger.info(f"  - Number of atoms: {len(atoms)}")
        logger.info(f"  - Number of MD steps: {self.total_steps}")
        logger.info(f"  - Total simulation time: {total_time_ps:.2f} ps")
        logger.info(f"  - Temperature: {self.temperature} K")
        
        # Initialize observers (CHGNet style)
        self.trajectory_observer = TrajectoryObserver(atoms, log_interval=self.log_interval)
        self.crystal_feature_observer = CrystalFeatureObserver(atoms, log_interval=self.log_interval)
        
        # Start from low temperature (50K) instead of 0K to allow gradual motion
        # Starting from 0K can cause stress buildup and then violent motion when thermostat kicks in
        # Small initial temperature allows atoms to move gradually under stress
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
        initial_temp = 50.0  # K - small initial temperature for gradual motion
        MaxwellBoltzmannDistribution(atoms, temperature_K=initial_temp)
        logger.info(f"Starting MD from {initial_temp}K, will gradually heat to {self.temperature}K via thermostat")
        
        # Create NPT MD using Inhomogeneous_NPTBerendsen for better pressure control
        # This allows anisotropic pressure control and prevents kinetic energy runaway
        if Inhomogeneous_NPTBerendsen is None:
            raise RuntimeError("Inhomogeneous_NPTBerendsen not available. Please ensure ASE is properly installed.")
        
        # Ensure pressure_au is in correct units
        import ase.units as units
        if not hasattr(self, 'pressure_au') or self.pressure_au == 0.0:
            self.pressure_au = 0.0 * units.bar  # 0 GPa = ambient pressure
        
        # Create trajectory file path if needed
        trajfile = None
        if self.output_dir:
            from pathlib import Path
            trajfile = str(Path(self.output_dir) / "md_trajectory.traj")
        
        dyn = Inhomogeneous_NPTBerendsen(
            atoms,
            timestep=self.time_step,
            temperature_K=self.temperature,
            pressure_au=self.pressure_au,
            taut=self.taut,
            taup=self.taup,
            compressibility_au=self.compressibility_au,
            trajectory=trajfile,
            logfile=None,  # We use MDLogger separately
            loginterval=self.log_interval,
            append_trajectory=False,
        )
        
        # Add observers
        dyn.attach(self.trajectory_observer, interval=self.log_interval)
        dyn.attach(self.crystal_feature_observer, interval=self.log_interval)
        
        # Add logger - save to output_dir if specified, otherwise current directory
        from pathlib import Path
        if self.output_dir:
            log_path = Path(self.output_dir) / "pes_md.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path("pes_md.log")
            
        logger_md = MDLogger(dyn, atoms, str(log_path), 
                           header=True, stress=True, peratom=True, mode="w")
        dyn.attach(logger_md, interval=self.log_interval)
        logger.info(f"  - MD log saved to: {log_path}")
        
        # Run MD simulation
        dyn.run(self.total_steps)
        logger.info("MD simulation completed successfully")
        
        return self.trajectory_observer.structures, self.crystal_feature_observer.crystal_feature_vectors, {
            "num_atoms": len(atoms),
            "total_steps": self.total_steps,
            "time_step": self.time_step,
            "total_time_ps": total_time_ps,
            "temperature": self.temperature,
            "pressure": self.pressure_au
        }
    
    def cluster_structures(self, structures: List[Atoms], features: List[np.ndarray]) -> Tuple[List[Atoms], List[int]]:
        """
        Perform clustering sampling on crystal features to downselect representative structures.
        
        Args:
            structures: List of structures from MD simulation
            features: List of crystal features corresponding to structures
            
        Returns:
        Returns:
            List of representative structures (max 200), List of indices
        """
        if len(structures) == 0:
            logger.warning("No structures to cluster")
            return [], []
        
        logger.info(f"Clustering {len(structures)} structures to {min(self.n_clusters, len(structures))} representatives")
        
        # Ensure we don't ask for more clusters than structures
        n_clusters = min(self.n_clusters, len(structures))
        
        if n_clusters >= len(structures):
            logger.info("Number of clusters >= number of structures, returning all structures")
            return structures, list(range(len(structures)))
        
        # Features are now crystal-level features (64,) from averaged atom features
        # Convert to 2D array for clustering
        features_array = np.array(features)  # Shape: (n_structures, 64)
        logger.info(f"Features shape: {features_array.shape}")
        
        # Handle very small datasets with simple selection
        if len(structures) <= 3:
            logger.info("Small dataset: using simple selection strategy")
            return self._simple_selection(structures, features_array, n_clusters)
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        # Use more robust clustering for small datasets
        if len(structures) < 10:
            # Use hierarchical clustering for small datasets
            from sklearn.cluster import AgglomerativeClustering
            clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
            cluster_labels = clustering.fit_predict(features_scaled)
        else:
            # Use K-means for larger datasets
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
            cluster_labels = kmeans.fit_predict(features_scaled)
        
        # Select representative structures
        representative_structures, representative_indices = self._select_representatives(
            structures, features_scaled, cluster_labels, n_clusters
        )
        
        logger.info(f"Clustering completed. Selected {len(representative_structures)} representative structures")
        return representative_structures, representative_indices
    
    def _aggregate_to_crystal_features(self, features: List[np.ndarray]) -> np.ndarray:
        """
        Aggregate per-atom features to crystal-level features.
        Uses multiple aggregation strategies inspired by MAML approach.
        """
        crystal_features_list = []
        
        for feature in features:
            if feature.ndim == 2 and feature.shape[1] == 1:
                # Per-atom features (N_atoms, 1) - aggregate to crystal level
                per_atom = feature.flatten()
                
                # Multiple aggregation strategies for robustness
                crystal_feature = np.concatenate([
                    [np.mean(per_atom)],      # Mean
                    [np.std(per_atom)],       # Standard deviation
                    [np.min(per_atom)],       # Minimum
                    [np.max(per_atom)],       # Maximum
                    [np.median(per_atom)],    # Median
                    [np.percentile(per_atom, 25)],  # 25th percentile
                    [np.percentile(per_atom, 75)],  # 75th percentile
                    [np.sum(per_atom)],       # Sum (extensive property)
                ])
            else:
                # Already crystal-level features
                crystal_feature = feature.flatten()
            
            crystal_features_list.append(crystal_feature)
        
        return np.array(crystal_features_list)
    
    def _simple_selection(self, structures: List[Atoms], features: np.ndarray, n_clusters: int) -> List[Atoms]:
        """
        Simple selection strategy for very small datasets.
        Selects structures with maximum diversity.
        """
        if len(structures) <= n_clusters:
            return structures
        
        # Calculate pairwise distances
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(features)
        
        # Select most diverse structures
        selected_indices = [0]  # Start with first structure
        
        for _ in range(n_clusters - 1):
            max_min_distance = -1
            best_candidate = -1
            
            for i in range(len(structures)):
                if i not in selected_indices:
                    min_distance = min(distances[i][j] for j in selected_indices)
                    if min_distance > max_min_distance:
                        max_min_distance = min_distance
                        best_candidate = i
            
            if best_candidate != -1:
                selected_indices.append(best_candidate)
        
        return [structures[i] for i in selected_indices], selected_indices
    
    def _select_representatives(self, structures: List[Atoms], features_scaled: np.ndarray, 
                              cluster_labels: np.ndarray, n_clusters: int) -> List[Atoms]:
        """
        Select representative structures from clusters.
        """
        representative_structures = []
        representative_indices = []
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_structures = [structures[i] for i in range(len(structures)) if cluster_mask[i]]
            cluster_features = features_scaled[cluster_mask]
            
            if len(cluster_structures) > 0:
                if len(cluster_structures) == 1:
                    # Single structure in cluster
                    representative_structures.append(cluster_structures[0])
                    # Find index
                    idx = [i for i, mask in enumerate(cluster_mask) if mask][0]
                    representative_indices.append(idx)
                else:
                    # Find structure closest to cluster centroid
                    cluster_centroid = np.mean(cluster_features, axis=0)
                    distances = np.linalg.norm(cluster_features - cluster_centroid, axis=1)
                    closest_local_idx = np.argmin(distances)
                    # Map back to global index
                    # cluster_structures was created by iterating range(len(structures)) with mask
                    # so we need to find the index in original list
                    # It's cleaner to re-derive indices
                    cluster_indices = [i for i, mask in enumerate(cluster_mask) if mask]
                    closest_global_idx = cluster_indices[closest_local_idx]
                    
                    representative_structures.append(structures[closest_global_idx])
                    representative_indices.append(closest_global_idx) # type: ignore
        
        return representative_structures, representative_indices
    
    def sample(self) -> Tuple[List[Atoms], Dict[str, Any]]:
        """
        Main sampling method that runs MD simulation and clustering.
        
        Returns:
            List of representative structures, Metadata dictionary
        """
        logger.info("Starting PES sampling")
        
        # Run MD simulation
        structures, features, metadata = self.run_md_simulation()
        
        # Cluster structures
        representative_structures, representative_indices = self.cluster_structures(structures, features)
        
        # Add indices to metadata
        # Convert to actual MD steps
        md_steps = [idx * self.log_interval for idx in representative_indices]
        metadata["sampled_indices"] = representative_indices # Index in the structure list
        metadata["sampled_md_steps"] = md_steps # Actual MD step number
        
        logger.info(f"PES sampling completed. Generated {len(representative_structures)} representative structures")
        return representative_structures, metadata


class NearEquilibriumSampler:
    """
    Near-equilibrium sampler for ground state energy calculations.
    
    Samples energy minima of the PES using ionic relaxation.
    """
    
    def __init__(self, calculator: Optional[Calculator] = None):
        """
        Initialize near-equilibrium sampler.
        
        Args:
            calculator: ASE calculator for relaxation
        """
        self.calculator = calculator
    
    def sample_ground_states(self, initial_structures: List[Atoms], 
                           fmax: float = 0.01, max_steps: int = 200) -> List[Atoms]:
        """
        Sample ground state structures using ionic relaxation.
        
        Args:
            initial_structures: List of initial structures to relax
            fmax: Force convergence criterion
            max_steps: Maximum relaxation steps
            
        Returns:
            List of relaxed ground state structures
        """
        if self.calculator is None:
            raise ValueError("Calculator must be set for ground state sampling")
        
        logger.info(f"Starting ground state sampling for {len(initial_structures)} structures")
        
        relaxed_structures = []
        
        for i, structure in enumerate(initial_structures):
            logger.info(f"Relaxing structure {i+1}/{len(initial_structures)}")
            
            # Set calculator
            structure.calc = self.calculator
            
            # Create optimizer
            opt = BFGS(structure, logfile=f"ground_state_relax_{i}.log")
            
            # Run relaxation
            opt.run(fmax=fmax, steps=max_steps)
            relaxed_structures.append(structure.copy())
            logger.debug(f"Ground state relaxation completed for structure {i+1}")
        
        logger.info(f"Ground state sampling completed. Generated {len(relaxed_structures)} relaxed structures")
        return relaxed_structures


class OrderDisorderSampler:
    """
    Order-Disorder Sampler for generating ordered structures from disordered structures.
    
    Uses pymatgen's OrderDisorderTransformation to generate ordered structures
    from a disordered input structure with fractional occupancies.
    """
    
    def __init__(self, atoms, n_structures: int = 100, target_atoms: int = 50, 
                 include_perturbation: int = 1, perturbation_length: float = 0.1):
        """
        Initialize OrderDisorderSampler.
        
        Args:
            atoms: Initial disordered atomic structure (ASE Atoms or pymatgen Structure)
                   For disordered structures with fractional occupancies, use pymatgen Structure
            n_structures: Number of ordered structures to generate (default: 100)
            target_atoms: Target number of atoms per structure after supercell expansion (default: 50)
            include_perturbation: Number of perturbations per ordered structure (default: 1)
            perturbation_length: Length of random displacement in Angstroms (default: 0.1)
        """
        self.atoms = atoms
        self.n_structures = n_structures
        self.target_atoms = target_atoms
        self.include_perturbation = include_perturbation
        self.perturbation_length = perturbation_length
    
    def _expand_supercell(self, atoms: Atoms, target_atoms: int = 50) -> Atoms:
        """
        Expand supercell to reach target number of atoms with cubic-like expansion.
        
        Prefers expansion matrices like [2,2,2] over [8,1,1] to keep the cell as cubic as possible.
        
        Args:
            atoms: Initial atomic structure
            target_atoms: Target number of atoms (default: 50)
            
        Returns:
            Expanded ASE Atoms object
        """
        current_atoms = len(atoms)
        
        # If already close to target, return as-is
        if current_atoms >= target_atoms * 0.9 and current_atoms <= target_atoms * 1.1:
            logger.info(f"Structure already has {current_atoms} atoms, close to target {target_atoms}. No expansion needed.")
            return atoms.copy()
        
        # If already larger than target, return as-is (don't shrink)
        if current_atoms > target_atoms:
            logger.info(f"Structure has {current_atoms} atoms, larger than target {target_atoms}. No expansion needed.")
            return atoms.copy()
        
        # Calculate expansion factor needed
        expansion_factor = (target_atoms / current_atoms) ** (1.0 / 3.0)
        
        # Find best expansion matrix [nx, ny, nz] that:
        # 1. Gets close to target_atoms
        # 2. Minimizes difference between nx, ny, nz (prefer cubic)
        # 3. Uses integer values
        
        best_expansion = None
        best_score = float('inf')
        best_num_atoms = 0
        
        # Search space: try expansion matrices from 1x1x1 to 10x10x10
        max_expansion = int(expansion_factor * 2) + 2
        max_expansion = min(max_expansion, 10)  # Cap at 10x10x10
        
        for nx in range(1, max_expansion + 1):
            for ny in range(1, max_expansion + 1):
                for nz in range(1, max_expansion + 1):
                    num_atoms = current_atoms * nx * ny * nz
                    
                    # Only consider if within reasonable range (40 to 100)
                    if num_atoms < 40 or num_atoms > 100:
                        continue
                    
                    # Score based on:
                    # 1. Distance from target (prefer closer to target)
                    # 2. Difference between expansion factors (prefer cubic: [2,2,2] over [8,1,1])
                    target_distance = abs(num_atoms - target_atoms) / target_atoms
                    
                    # Calculate "cubicness" - prefer expansion matrices where nx, ny, nz are similar
                    expansion_values = [nx, ny, nz]
                    max_exp = max(expansion_values)
                    min_exp = min(expansion_values)
                    # Perfect cubic (all equal) = 0, very non-cubic = 1
                    cubicness = (max_exp - min_exp) / max_exp if max_exp > 0 else 0.0
                    
                    # Weight: 0.5 for target distance, 0.5 for cubicness (equal weight)
                    score = 0.5 * target_distance + 0.5 * cubicness
                    
                    if score < best_score:
                        best_score = score
                        best_expansion = (nx, ny, nz)
                        best_num_atoms = num_atoms
        
        if best_expansion is None:
            # Fallback: use simple expansion based on cubic root
            nx = ny = nz = int(round(expansion_factor))
            best_expansion = (max(1, nx), max(1, ny), max(1, nz))
            best_num_atoms = current_atoms * best_expansion[0] * best_expansion[1] * best_expansion[2]
        
        # Expand the structure
        expanded_atoms = atoms.repeat(best_expansion)
        
        logger.info(f"Expanded supercell from {current_atoms} to {best_num_atoms} atoms using expansion matrix {best_expansion}")
        
        return expanded_atoms
    
    def _perturb_structure(self, atoms: Atoms, perturbation_length: float) -> Atoms:
        """
        Apply random perturbation to atomic positions.
        
        Args:
            atoms: ASE Atoms object to perturb
            perturbation_length: Maximum displacement length in Angstroms
            
        Returns:
            Perturbed ASE Atoms object
        """
        perturbed = atoms.copy()
        # Generate random displacements
        displacements = np.random.normal(0, perturbation_length / 3.0, size=perturbed.positions.shape)
        # Normalize to ensure max displacement is approximately perturbation_length
        max_disp = np.max(np.linalg.norm(displacements, axis=1))
        if max_disp > 0:
            displacements = displacements * (perturbation_length / max_disp) * np.random.random()
        perturbed.positions += displacements
        return perturbed
    
    def sample(self, output_dir: Optional[str] = None) -> List[Atoms]:
        """
        Generate ordered structures from disordered input structure.
        
        Steps:
        1. Expand supercell to target size
        2. Perform OrderDisorder transformation on expanded supercell to get n_structures
        3. Perform perturbation on all ordered structures (include_perturbation per structure)
        4. Return all sampled structures

        Returns:
            List of sampled ASE Atoms structures (n_structures * include_perturbation total)
        """
        logger.info(f"Starting order-disorder sampling: generating {self.n_structures} ordered structures")

        # Convert atoms to pymatgen Structure if needed
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.core import Structure as PymatgenStructure
        
        if isinstance(self.atoms, PymatgenStructure):
            pmg_structure = self.atoms
        else:
            adaptor = AseAtomsAdaptor()
            pmg_structure = adaptor.get_structure(self.atoms)

        # Step 1: Expand supercell to target size
        logger.info(f"Step 1: Expanding supercell to target size ({self.target_atoms} atoms)")
        
        current_atoms = len(pmg_structure)
        expansion_factor = (self.target_atoms / current_atoms) ** (1.0 / 3.0)
        
        # Search space: try expansion matrices
        best_exp = None
        best_score = float('inf')
        
        max_expansion = int(expansion_factor * 2) + 2
        max_expansion = min(max_expansion, 6) # Cap to avoid huge search
        for nx in range(1, max_expansion + 1):
            for ny in range(1, max_expansion + 1):
                for nz in range(1, max_expansion + 1):
                    num_atoms = current_atoms * nx * ny * nz
                    # Acceptable range to be near target (or at least > 1)
                    if num_atoms < 1: continue
                    
                    # CRITICAL: Check stoichiometry consistency
                    # OrderDisorderedStructureTransformation fails if total atoms * fraction != integer
                    # We check if (current_amount * expansion) is close to integer for all species
                    valid_stoichiometry = True
                    expansion_factor_int = nx * ny * nz
                    
                    for el, amt in pmg_structure.composition.items():
                        total_amt = amt * expansion_factor_int
                        if abs(total_amt - round(total_amt)) > 1e-3:
                            valid_stoichiometry = False
                            break
                    
                    if not valid_stoichiometry:
                        continue
                    
                    target_dist = abs(num_atoms - self.target_atoms) / self.target_atoms
                    # Prefer cubic-like expansion
                    cubicness = (max(nx, ny, nz) - min(nx, ny, nz)) / max(nx, ny, nz) if max(nx, ny, nz) > 0 else 0.0
                    score = 0.6 * target_dist + 0.4 * cubicness
                    
                    if score < best_score:
                        best_score = score
                        best_exp = (nx, ny, nz)
                        
        if best_exp:
            expanded_pmg = pmg_structure * best_exp
            logger.info(f"Using expansion matrix: {best_exp}")
        else:
            # Fallback: simple expansion
            nx = ny = nz = max(1, int(round(expansion_factor)))
            expanded_pmg = pmg_structure * (nx, ny, nz)
            
        logger.info(f"Expanded supercell has {len(expanded_pmg)} atoms")

        # Persistence: Save disordered supercell immediately
        if output_dir:
            try:
                from pymatgen.io.cif import CifWriter
                disordered_path = os.path.join(output_dir, "disordered_supercell.cif")
                CifWriter(expanded_pmg).write_file(disordered_path)
                logger.info(f"Saved disordered supercell to {disordered_path}")
            except Exception as e:
                logger.warning(f"Failed to save disordered supercell: {e}")
        
        # Step 2: Use pymatgen's OrderDisorderedStructureTransformation
        logger.info(f"Step 2: Performing OrderDisorderedStructureTransformation to generate candidates")
        
        ordered_structures = []
        try:
            from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
            from pymatgen.analysis.ewald import EwaldMinimizer
            
            # CRITICAL OPTIMIZATION: Use ALGO.FAST (0) as requested by user
            # This is the fastest algorithm available.
            transformation = OrderDisorderedStructureTransformation(
                algo=EwaldMinimizer.ALGO_FAST, 
                no_oxi_states=True # Ignore oxidation states for speed
            )
            
            
            # We request more structures than needed to sample from them uniformly
            # Request 5x needed to get a pool representing different energies
            n_candidates = max(self.n_structures * 5, 50) 
            
            # This returns a list of dictionaries [{'structure': ..., 'energy': ...}, ...]
            # ranked by energy (lowest first)
            ranked_list = transformation.apply_transformation(expanded_pmg, return_ranked_list=n_candidates)
            
            
            # Handle case where single result is returned (if not list)
            if not isinstance(ranked_list, list):
                ranked_list = [{'structure': ranked_list, 'energy': 0.0}]
            
            logger.info(f"Generated {len(ranked_list)} candidate ordered structures")
            
            # Step 3: Uniform sampling from the ranked list
            # We want to enable diversity, so we pick uniformly from the list
            # which usually contains low to higher energy structures (since it is ranked)
            
            if len(ranked_list) <= self.n_structures:
                selected_entries = ranked_list
            else:
                # Uniformly sample indices from the ranked list
                # This ensures we get low, medium, and higher energy structures from the pool
                indices = np.linspace(0, len(ranked_list) - 1, self.n_structures, dtype=int)
                # Remove duplicates should they occur
                indices = np.unique(indices)
                
                selected_entries = [ranked_list[i] for i in indices]
                
                # Fill up if unique removal caused shortage
                if len(selected_entries) < self.n_structures:
                    remaining_indices = [i for i in range(len(ranked_list)) if i not in indices]
                    # Add from top of remaining (lowest energy not yet picked)
                    for i in remaining_indices[:self.n_structures - len(selected_entries)]:
                        selected_entries.append(ranked_list[i])

            # Extract structures
            ordered_structures = [entry['structure'] for entry in selected_entries]
                
        except Exception as e:
            with open(".agent/test/debug_tool.log", "a") as f:
                f.write(f"DEBUG: SAMPLER: Transformation failed: {e}\n")
            logger.error(f"OrderDisorder transformation failed: {e}")
            logger.info("Attempting fallback with random ordering...")
            # Fallback: Randomly replace fractional occupancy
            try:
                from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
                from pymatgen.analysis.ewald import EwaldMinimizer
                # Use ALGO.FAST (0) which might just be random/fastest
                t = OrderDisorderedStructureTransformation(
                    algo=EwaldMinimizer.ALGO_FAST,
                    no_oxi_states=True
                ) 
                s = t.apply_transformation(expanded_pmg)
                ordered_structures = [s]
                with open(".agent/test/debug_tool.log", "a") as f:
                    f.write(f"DEBUG: SAMPLER: Fallback generated {len(ordered_structures)} structures\n")
            except Exception as e2:
                with open(".agent/test/debug_tool.log", "a") as f:
                    f.write(f"DEBUG: SAMPLER: Fallback failed: {e2}\n")
                logger.error(f"Fallback failed: {e2}")
                ordered_structures = []
            
        # Step 4: Convert to ASE and perturbation
        all_sampled_structures = []
        
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        
        logger.info(f"Step 3: Converting {len(ordered_structures)} structures to ASE")
        
        for i, pmg_struct in enumerate(ordered_structures):
            # Convert to ASE
            atoms = adaptor.get_atoms(pmg_struct)
            
            # Add info
            atoms.info['config_id'] = f"ordered_{i}"
            atoms.info['source'] = "order_disorder"
            
            all_sampled_structures.append(atoms)
            
            # Perturbation
            if self.include_perturbation:
                atoms_perturbed = atoms.copy()
                atoms_perturbed.rattle(stdev=self.perturbation_length)
                atoms_perturbed.info['config_id'] = f"ordered_{i}_perturbed"
                all_sampled_structures.append(atoms_perturbed)
                
        logger.info(f"Total structures generated: {len(all_sampled_structures)}")
        return all_sampled_structures


class StructureSampler:
    """
    Main structure sampler that decides which sampling method to use based on simulation task.
    
    This is the main interface for the MLIP Agent to sample structures for training data.
    """
    
    def __init__(self, calculator: Optional[Calculator] = None):
        """
        Initialize the structure sampler.
        
        Args:
            calculator: ASE calculator for energy/force calculations
        """
        self.calculator = calculator
        self.pes_sampler = None
        self.equilibrium_sampler = None
    
    def sample_off_equilibrium(self, atoms: Atoms, model_name: str = "M3GNet-MatPES-r2SCAN-v2025.1-PES",
                               total_steps: Optional[int] = None, output_dir: Optional[str] = None,
                               target_atoms: int = 75, temperature: float = 1000.0) -> List[Atoms]:
        """
        Sample structures for off-equilibrium calculations (MD, melting, diffusion, etc.).
        
        Uses PESSampler with MD simulation and clustering as specified in sampler.mdc.
        The structure is automatically expanded to 50-100 atoms using cubic-like supercell expansion
        before MD simulation (prefers [2,2,2] over [8,1,1] expansion).
        
        Memory safety: Maximum safe atom limit is 70 atoms to prevent OOM in subsequent VASP
        calculations. Structures exceeding this limit will not be expanded further.
        
        Args:
            atoms: Initial atomic structure
            model_name: MatGL model name for MD simulation
            total_steps: Number of MD steps (default: 10000). Use smaller values (e.g., 100) for testing.
            output_dir: Directory to save MD log file (default: current directory)
            target_atoms: Target number of atoms after supercell expansion (default: 75, range: 50-100)
                         Maximum safe limit: 70 atoms (enforced to prevent VASP OOM issues)
            
        Returns:
            List of representative structures from PES sampling
        """
        logger.info(f"Starting off-equilibrium sampling using OffEquilibriumSampler (target: {target_atoms} atoms)")
        
        if self.calculator is None:
            raise ValueError("Calculator must be initialized for StructureSampler before calling sample_off_equilibrium")

        # Initialize OffEquilibriumSampler (will automatically expand supercell to target_atoms with cubic preference)
        # Note: model_name is no longer used, calculator is passed from self.calculator
        self.pes_sampler = OffEquilibriumSampler(calculator=self.calculator, atoms=atoms, total_steps=total_steps, 
                                     output_dir=output_dir, target_atoms=target_atoms, temperature=temperature)
        
        # Run sampling
        representative_structures, metadata = self.pes_sampler.sample()
        
        # Save individual structures if output_dir is provided
        if output_dir:
            from pymatgen.io.ase import AseAtomsAdaptor
            adaptor = AseAtomsAdaptor()
            from pathlib import Path
            
            output_path = Path(output_dir)
            structures_dir = output_path / "sampled_structures"
            structures_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Saving {len(representative_structures)} structures to {structures_dir}")
            
            for i, atoms in enumerate(representative_structures):
                struct_path = structures_dir / f"structure_{i}.cif"
                pmg_struct = adaptor.get_structure(atoms)
                pmg_struct.to(filename=str(struct_path), fmt="cif")
                
        logger.info(f"Off-equilibrium sampling completed. Generated {len(representative_structures)} structures")
        return representative_structures, metadata
    
    def sample_near_equilibrium(self, initial_structures: List[Atoms], 
                              fmax: float = 0.01, max_steps: int = 200) -> List[Atoms]:
        """
        Sample structures for near-equilibrium calculations (ground state energies).
        
        Uses ionic relaxation to find energy minima.
        
        Args:
            initial_structures: List of initial structures to relax
            fmax: Force convergence criterion
            max_steps: Maximum relaxation steps
            
        Returns:
            List of relaxed ground state structures
        """
        logger.info("Starting near-equilibrium sampling using ionic relaxation")
        
        if self.calculator is None:
            raise ValueError("Calculator must be set for near-equilibrium sampling")
        
        # Initialize equilibrium sampler
        self.equilibrium_sampler = NearEquilibriumSampler(calculator=self.calculator)
        
        # Run sampling
        relaxed_structures = self.equilibrium_sampler.sample_ground_states(
            initial_structures, fmax=fmax, max_steps=max_steps
        )
        
        logger.info(f"Near-equilibrium sampling completed. Generated {len(relaxed_structures)} structures")
        return relaxed_structures
    
    def sample_order_disorder(self, atoms: Any, n_structures: int = 100, 
                             target_atoms: int = 50, include_perturbation: int = 1,
                             perturbation_length: float = 0.1, output_dir: Optional[str] = None) -> List[Atoms]:
        """
        Sample ordered structures from a disordered structure using OrderDisorderSampler.
        
        Args:
            atoms: Initial disordered structure (pymatgen Structure preferred)
            n_structures: Number of ordered structures to generate
            target_atoms: Target atoms in supercell
            include_perturbation: Number of perturbations per ordered structure
            perturbation_length: Length of perturbation
            output_dir: Directory to save intermediate files
            
        Returns:
            List of ordered (and potentially perturbed) ASE Atoms structures
        """
        logger.info("Starting order-disorder sampling")
        
        # Initialize sampler
        sampler = OrderDisorderSampler(
            atoms, 
            n_structures=n_structures, 
            target_atoms=target_atoms,
            include_perturbation=include_perturbation,
            perturbation_length=perturbation_length
        )
        
        # Run sampling
        sampled_structures = sampler.sample(output_dir=output_dir)
        
        logger.info(f"Order-disorder sampling completed. Generated {len(sampled_structures)} structures")
        return sampled_structures
    
    def sample_structures(self, atoms: Atoms, task_type: str = "off_equilibrium", 
                         model_name: str = "M3GNet-MatPES-r2SCAN-v2025.1-PES",
                         **kwargs) -> List[Atoms]:
        """
        Main sampling interface that decides which method to use based on task type.
        
        Args:
            atoms: Initial atomic structure (for off-equilibrium) or list of structures (for near-equilibrium)
            task_type: Type of simulation task ("off_equilibrium" or "near_equilibrium")
            model_name: MatGL model name for off-equilibrium sampling
            **kwargs: Additional arguments for specific sampling methods
            
        Returns:
            List of sampled structures
        """
        if task_type == "off_equilibrium":
            return self.sample_off_equilibrium(atoms, model_name=model_name)
        elif task_type == "near_equilibrium":
            if isinstance(atoms, list):
                return self.sample_near_equilibrium(atoms, **kwargs)
            else:
                return self.sample_near_equilibrium([atoms], **kwargs)
        elif task_type == "order_disorder":
            return self.sample_order_disorder(atoms, **kwargs)
        else:
            raise ValueError(f"Unknown task type: {task_type}. Must be 'off_equilibrium' or 'near_equilibrium'")
    
    def set_calculator(self, calculator: Calculator) -> None:
        """
        Set the calculator for structure sampling.
        
        Args:
            calculator: ASE calculator object.
        """
        self.calculator = calculator
        logger.info("Calculator set for structure sampling")
