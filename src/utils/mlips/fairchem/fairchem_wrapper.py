"""
FAIRCHEM model wrapper for MLIP Agent
"""

import logging
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from fairchem.core.units.mlip_unit.api.inference import InferenceSettings, guess_inference_settings
from fairchem.core.modules.loss import DDPMTLoss, L2NormLoss, PerAtomMAELoss, MAELoss
from ..base import MLIPModel

logger = logging.getLogger(__name__)




class FAIRCHEMHistoryLogger:
    """Helper class to capture metrics from FAIRCHEM training loop."""
    def __init__(self, history):
        self.history = history
        # Buffers for training metrics to average them per epoch
        self._train_buffers = {
            'loss_train': [],
            'energy_mae_train': [],
            'force_mae_train': [],
            'stress_mae_train': []
        }
        self.last_epoch = -1
        
    def log_summary(self, summary):
        """Pass-through for log_summary if needed."""
        pass
        
    def _flush_train_buffers(self):
        """Calculate averages and append to history."""
        for key, buffer in self._train_buffers.items():
            if buffer:
                avg_val = sum(buffer) / len(buffer)
                self.history[key].append(float(avg_val))
                self._train_buffers[key] = []
            elif key in self.history and len(self.history[key]) < len(self.history.get('epoch', [])):
                # If buffer is empty but we have an epoch, append None or 0 to keep lists aligned
                # Actually, if we flushed at the end of epoch, they should align.
                pass

    def log(self, metrics, step=None, commit=True):
        """Capture metrics and store in history dictionary."""
        # epoch is often reported as val/epoch or train/epoch
        epoch = -1
        if 'val/epoch' in metrics:
            epoch = int(metrics['val/epoch'])
        elif 'train/epoch' in metrics:
            epoch = int(metrics['train/epoch'])

        if epoch != -1 and epoch > self.last_epoch:
            # New epoch detected! Flush the training buffers from the previous epoch
            if self.last_epoch != -1:
                self._flush_train_buffers()
            
            # Store the new epoch
            if epoch not in self.history['epoch']:
                self.history['epoch'].append(epoch)
            self.last_epoch = epoch
        
        # Accumulate training metrics in buffers
        if 'train/loss' in metrics:
            self._train_buffers['loss_train'].append(float(metrics['train/loss']))
            
        # Parse task-specific metrics
        # Keys look like: val/dataset,task,metric_name
        for k, v in metrics.items():
            if k.startswith('val/') and 'mae' in k:
                val = float(v)
                if 'energy' in k:
                    self.history['energy_mae_val'].append(val)
                elif 'forces' in k:
                    self.history['force_mae_val'].append(val)
                elif 'stress' in k:
                    self.history['stress_mae_val'].append(val)
            elif k.startswith('train/') and 'mae' in k:
                val = float(v)
                if 'energy' in k:
                    self._train_buffers['energy_mae_train'].append(val)
                elif 'forces' in k:
                    self._train_buffers['force_mae_train'].append(val)
                elif 'stress' in k:
                    self._train_buffers['stress_mae_train'].append(val)
        
        # Validation loss is usually logged at the end of epoch along with val/epoch
        if 'val/loss' in metrics:
            self.history['loss_val'].append(float(metrics['val/loss']))

    def finalize(self):
        """Flush any remaining training logs when training ends."""
        self._flush_train_buffers()


# Available FAIRCHEM models and checkpoints
# Model configurations, default tasks, and supported tasks
MODEL_METADATA = {
    # UMA models (Universal) - Can handle both solid state (omat) and output molecular properties (omol)
    "uma-s-1p1": {
        "default_task": "omat", 
        "supported_tasks": ["omat", "omol", "oc22"],
        "domain": "general"
    },
    "uma-m-1p1": {
        "default_task": "omat", 
        "supported_tasks": ["omat", "omol", "oc22"],
        "domain": "general"
    },
    "uma-s-1": {
        "default_task": "omat", 
        "supported_tasks": ["omat", "omol", "oc22"],
        "domain": "general"
    },
    
    # ESEN models for organic molecules (Molecular only)
    "esen-md-direct-all-omol": {
        "default_task": "omol", 
        "supported_tasks": ["omol"],
        "domain": "molecular"
    },
    "esen-sm-conserving-all-omol": {
        "default_task": "omol", 
        "supported_tasks": ["omol"],
        "domain": "molecular"
    },
    "esen-sm-direct-all-omol": {
        "default_task": "omol", 
        "supported_tasks": ["omol"],
        "domain": "molecular"
    },
    
    # ESEN models for catalysis (OC25/OC22) - Surface science
    "esen-sm-conserving-all-oc25": {
        "default_task": "oc22", 
        "supported_tasks": ["oc22", "omat"],
        "domain": "catalysis"
    }, 
    "esen-md-direct-all-oc25": {
        "default_task": "oc22", 
        "supported_tasks": ["oc22", "omat"],
        "domain": "catalysis"
    },
}

# Try to import FAIRCHEM components with clear error messages
FAIRCHEM_AVAILABLE = False

try:
    from fairchem.core import FAIRChemCalculator, pretrained_mlip
    FAIRCHEM_AVAILABLE = True
except ImportError as e:
    import os
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    # Only raise the environment error if we're clearly NOT in fairchem environment
    # If we're in fairchem but import fails, it's a dependency issue, not environment issue
    if 'fairchem' not in current_env.lower() and 'fairchem' not in str(e).lower():
        raise ImportError(
            f"FAIRCHEM is not available in the current conda environment '{current_env}'. "
            f"FAIRCHEM models (UMA, ESEN) require the 'fairchem-agent' conda environment. "
            f"Please run this code in the fairchem-agent environment:\n"
            f"  conda activate fairchem-agent\n"
            f"Or use subprocess execution via MLIPModelTool which handles this automatically.\n"
            f"Original error: {e}"
        ) from e
    # If we're in fairchem environment but import fails, it's a dependency issue
    # Allow the error to propagate so subprocess executor can handle it
    FAIRCHEM_AVAILABLE = False
    raise


class FAIRCHEMWrapper(MLIPModel):
    """
    FAIRCHEM model wrapper implementing the MLIPModel interface.
    
    Supports EquiformerV2 and other FAIRCHEM models.
    """
    
    def __init__(
        self, 
        model_name: str = "EquiformerV2", 
        model_version: str = "latest",
        device: str = "auto",
        task_name: Optional[str] = None,
        inference_settings: Optional[Union[Dict[str, Any], str, InferenceSettings]] = "default"
    ):
        """
        Initialize FAIRCHEM wrapper.
        
        Args:
            model_name: Name of the model ("EquiformerV2", "GemNet", etc.)
            model_version: Version of the model
            device: Device to run the model on ("auto", "cpu", "cuda")
            task_name: Optional task name (e.g. "omat", "omol", "oc22")
        """
        super().__init__(model_name, model_version)
        self.device = device
        self.task_name = task_name
        self.inference_settings = inference_settings
        self.model_class = None
        self.calculator_class = None
        self.trainer = None
        self._calculator = None
        self._current_task = None
        
        # Note: FAIRCHEM_AVAILABLE is False when package is not installed
        # This is expected in testing environments
        
        # Set up model and calculator classes
        self._setup_model_classes()
    
    def _setup_model_classes(self):
        """Set up model and calculator classes based on model name."""
        # Check if the model name is in our available models
        # Normalize model name for lookup (e.g. UMA-S-1P1 -> uma-s-1p1)
        normalized_name = self.model_name.lower()
        
        if normalized_name in MODEL_METADATA:
            self.model_name = normalized_name # Ensure we use the canonical key
            self.model_class = pretrained_mlip
            self.calculator_class = FAIRChemCalculator
            logger.info(f"Successfully set up classes for {self.model_name}")
        elif Path(self.model_name).exists():
             logger.info(f"Model name '{self.model_name}' appears to be a file path. Will infer base model from checkpoint.")
             self.model_class = pretrained_mlip
             self.calculator_class = FAIRChemCalculator
             # Checkpoint loading logic will handle the rest
        else:
             valid_models = list(MODEL_METADATA.keys())
             raise ValueError(f"Model {self.model_name} not found and is not a valid path. Available models: {valid_models}")
    
    def load(self, model_path: Optional[str] = None) -> None:
        """
        Load a FAIRCHEM model.
        
        Args:
            model_path: Path to model checkpoint. If None, loads default pretrained model.
                       If provided, loads checkpoint from the specified path.
        """
        # Set device first
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path is None:
            # Load pretrained model
            # Load pretrained model
            model_id = self.model_name
            
            # Parse inference settings
            if isinstance(self.inference_settings, (str, InferenceSettings)):
                settings = guess_inference_settings(self.inference_settings)
            elif isinstance(self.inference_settings, dict):
                settings = InferenceSettings(**self.inference_settings)
            else:
                settings = self.inference_settings

            # Optional: Disable compile for eSEN models if it's known to be unstable
            # but for now let's respect the settings.
            
            self.model = self.model_class.get_predict_unit(
                model_name=model_id,
                device=self.device,
                inference_settings=settings
            )
            
            self.is_loaded = True
            logger.info(f"Loaded pretrained {self.model_name} model")
        else:
            # Load from checkpoint file
            self.load_checkpoint(model_path)
            logger.info(f"Loaded model from checkpoint: {model_path}")
    
    def create_calculator(self, task_name: Optional[str] = None) -> Calculator:
        """
        Create an ASE calculator from the model.
        
        Args:
            task_name: Optional task name to override the default.
            
        Returns:
            ASE Calculator object.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # Determine target task name
        target_task = task_name or self.task_name
        if target_task is None:
            if "esen" in self.model_name.lower():
                target_task = "omol"
            else:
                target_task = "omat"
        
        # Reuse existing calculator if possible
        if self._calculator is not None and self._current_task == target_task:
            return self._calculator
            
        calculator = self.calculator_class(
            predict_unit=self.model,
            task_name=target_task
        )
        
        # Cache the calculator
        self._calculator = calculator
        self._current_task = target_task
        
        return calculator
    
    def fine_tune(self, 
                  training_data: List[Dict[str, Any]], 
                  validation_data: Optional[List[Dict[str, Any]]] = None,
                  training_config: Optional[Dict[str, Any]] = None,
                  output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Fine-tune the model using the provided training data.
        
        Args:
            training_data: List of dicts with 'structure' (Atoms) and labels ('energy', 'forces').
            validation_data: Optional list of dicts for validation.
            training_config: Configuration dict (epochs, learning_rate, batch_size, task_name).
            output_dir: Directory to save results.
            
        Returns:
            Dictionary with fine-tuning results (status, best_checkpoint_path, history).
        """
        # Unwrap configuration
        config = training_config or {}
        epochs = config.get("epochs", config.get("max_epochs", 10))
        learning_rate = config.get("learning_rate", 1e-4)
        batch_size = config.get("batch_size", 4)
        task_name = config.get("task_name", None)
        # Head selection / resumption
        # If None, creates new head "shared_efs_head" (default behavior)
        # If provided (e.g. "omat"), tries to extract weights from pretrained model
        init_head_name = config.get("init_head_name", None) 

        
        save_dir = output_dir if output_dir else "./checkpoints"
        
        # Helper to attach labels to Atoms for create_db
        def attach_labels(data_list):
            if not data_list: return []
            atoms_list = []
            
            from pymatgen.core import Structure
            from pymatgen.io.ase import AseAtomsAdaptor
            from ase import Atoms
            from ase.calculators.singlepoint import SinglePointCalculator
            import ase.units
            
            for d in data_list:
                s_obj = d['structure']
                at = None
                
                if isinstance(s_obj, dict):
                    # Try to convert from pymatgen dict
                    try:
                        struct = Structure.from_dict(s_obj)
                        at = AseAtomsAdaptor.get_atoms(struct)
                    except Exception as e:
                        logging.warning(f"Failed to convert dict to structure: {e}")
                        continue
                elif isinstance(s_obj, Atoms):
                    at = s_obj.copy()
                
                if at is None: continue

                # Attach SinglePointCalculator
                energy = d.get('energy')
                forces = d.get('forces')
                stress = d.get('stress')
                
                # Check consistency
                if forces is not None:
                     forces = np.array(forces)
                     if len(forces) != len(at):
                         logging.warning("Forces length mismatch, skipping forces")
                         forces = None
                
                if stress is not None:
                     stress = np.array(stress)
                     # Standardized labels are already in eV/A^3 (ASE standard)
                
                if energy is not None or forces is not None or stress is not None:
                     calc = SinglePointCalculator(at, energy=energy, forces=forces, stress=stress)
                     at.calc = calc

                atoms_list.append(at)
            return atoms_list

        train_structures = attach_labels(training_data)
        val_structures = attach_labels(validation_data)
        
        # Simple split if validation not provided
        if not val_structures:
            if len(train_structures) > 10:
                split_idx = int(0.9 * len(train_structures))
                val_structures = train_structures[split_idx:]
                train_structures = train_structures[:split_idx]
            else:
                val_structures = train_structures # Use train as val for very small data logic

        import os
        import tempfile
        import shutil
        import ase.db
        from omegaconf import OmegaConf
        import torch
        from torch.utils.data import DataLoader
        from torch.utils.data.distributed import DistributedSampler
        
        # internal fairchem imports
        from fairchem.core.units.mlip_unit.mlip_unit import (
            MLIPTrainEvalUnit, 
            initialize_finetuning_model, 
            Task,
            OutputSpec,
            _get_consine_lr_scheduler,
            convert_train_checkpoint_to_inference_checkpoint,
            mt_collater_adapter
        )
        from fairchem.core.components.train.train_runner import (
            TrainEvalRunner, 
            TrainCheckpointCallback
        )
        from fairchem.core.datasets.ase_datasets import AseDBDataset
        from fairchem.core.common.data_parallel import BalancedBatchSampler
        from fairchem.core.modules.normalization.normalizer import Normalizer
        from fairchem.core.modules.loss import DDPMTLoss, PerAtomMAELoss, L2NormLoss
        from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
        from fairchem.core.datasets.atomic_data import AtomicData
        from functools import partial
        import torch.distributed as dist

        from fairchem.core.datasets.mt_concat_dataset import ConcatDataset
        
        # Check for model/checkpoint override
        if training_config:
            new_model = training_config.get("foundation_model") or training_config.get("checkpoint_path")
            if new_model and new_model != self.model_name:
                logging.info(f"Reloading model for fine-tuning: {self.model_name} -> {new_model}")
                self.model_name = new_model
                self.is_loaded = False
                self.load(model_path=new_model if os.path.exists(new_model) else None)

        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)

        # Initialize process group if not already initialized (required for MLIPTrainEvalUnit)
        if not dist.is_initialized():
            # Use gloo backend which supports both CPU and GPU tensors and is more robust for tests
            # NCCL requires all tensors to be on GPU which can be strict or fail with mismatch
            backend = "gloo"
            if torch.cuda.is_available():
                try:
                    torch.cuda.set_device(0)
                except: pass
            
            # Use a free port to avoid EADDRINUSE errors
            from torch.distributed.elastic.utils.distributed import get_free_port
            port = get_free_port()
            dist.init_process_group(backend=backend, init_method=f"tcp://localhost:{port}", rank=0, world_size=1)
        
        # --- Determine Task Name ---
        # --- Determine Task Name ---
        model_key = self.model_name.lower()
        model_meta = MODEL_METADATA.get(model_key, {})
        supported_tasks = model_meta.get("supported_tasks", [])
        
        if task_name is None:
            # 1. Try to get default from metadata
            if "default_task" in model_meta:
                task_name = model_meta["default_task"]
                logging.info(f"Using default task '{task_name}' from metadata for {self.model_name}")
            else:
                # 2. Heuristic based on PBC (Fallback if no metadata - though now metadata is required)
                if train_structures and any(train_structures[0].pbc):
                    task_name = "omat" # Solid state / Bulk
                    logging.info(f"Inferred task '{task_name}' based on PBC=True")
                else:
                    task_name = "omol" # Finite / Molecule
                    logging.info(f"Inferred task '{task_name}' based on PBC=False")
        
        # Validation
        if supported_tasks and task_name not in supported_tasks:
            logging.warning(
                f"Selected fine-tuning task '{task_name}' is not explicitly listed in "
                f"supported_tasks for {self.model_name}: {supported_tasks}. "
                f"Proceeding, but this may fail."
            )
        else:
             logging.info(f"Using task name: {task_name}")

        energy_key = f"{task_name}_energy"
        forces_key = f"{task_name}_forces"
        stress_key = f"{task_name}_stress"

        # Create a temporary directory for datasets and artifacts
        with tempfile.TemporaryDirectory() as temp_dir:
            logging.info(f"Created temporary directory for fine-tuning: {temp_dir}")
            
            # --- 1. Preparation ---
            def create_db(structures, db_path, dataset_name):
                with ase.db.connect(db_path) as db:
                    for atoms in structures:
                        # Ensure we have energy and forces if possible
                        data = {}
                        if atoms.get_calculator() is not None:
                            try:
                                data['energy'] = atoms.get_potential_energy()
                                data['forces'] = atoms.get_forces()
                                try:
                                    data['stress'] = atoms.get_stress()
                                except: pass
                            except Exception:
                                pass
                        
                        kv_pairs = {}
                        if 'energy' in data:
                             kv_pairs['energy'] = data['energy']
                        elif 'energy' in atoms.info:
                             kv_pairs['energy'] = atoms.info['energy']
                             
                        if 'forces' in data:
                             atoms.info['forces'] = np.array(data['forces'])
                        def _ensure_3x3_stress(s):
                            s = np.array(s)
                            if s.shape == (3, 3): return s
                            if s.size == 9: return s.reshape(3, 3)
                            if s.size == 6:
                                # Voigt: xx, yy, zz, yz, xz, xy
                                v = s.flatten()
                                return np.array([[v[0], v[5], v[4]], [v[5], v[1], v[3]], [v[4], v[3], v[2]]])
                            return s.reshape(3, 3) # fallback to try reshape anyway

                        if 'stress' in data:
                             atoms.info['stress'] = _ensure_3x3_stress(data['stress'])
                        
                        # Add dataset name to info so it gets picked up
                        kv_pairs['dataset_name'] = dataset_name
                        
                        db.write(atoms, data=kv_pairs)

            train_db_path = os.path.join(temp_dir, "train.db")
            val_db_path = os.path.join(temp_dir, "val.db")
            
            create_db(train_structures, train_db_path, task_name)
            create_db(val_structures, val_db_path, task_name)
            
            # Check if stress is present to enable stress task
            has_stress = any(s.get_calculator() is not None and s.get_calculator().results.get('stress') is not None for s in train_structures)
            
            # --- 2. Configure Dataset ---
            # We use AseDBDataset. 
            # task_name, energy_key,            # Define key mapping for AseDBDataset
            key_mapping = {
                "energy": energy_key,
                "forces": forces_key,
                "stress": stress_key,
            }

            # Configure dataset
            base_dataset_config = {
                "src": None,  # Will be set later
                "a2g_args": {
                    "r_energy": True,
                    "r_forces": True,
                    "r_stress": True, # AtomicData will read stress as 3x3
                    "max_neigh": 300, # Standard UMA default
                    "task_name": None, # Should be None if we map keys manually? Or task_name?
                    # logic: dataset.task_name is used for dataset_name in batch.
                    # but if we use key mapping, we might not rely on dataset_name for filtering?
                    # Actually get_output_mask iterates dataset_name.
                    # So we should set it.
                },
                "key_mapping": key_mapping,
                "transforms": {
                    "common_transform": {
                        "dataset_name": task_name
                    },
                    "stress_reshape_transform": {}
                }
            }
            
            # Use ConcatDataset to follow Fairchem standard setup and handle dataset_name injection
            sampling_config = {"type": "explicit", "ratios": {task_name: 1.0}}
            
            # Create training dataset
            train_config = base_dataset_config.copy()
            train_config["src"] = train_db_path
            train_dataset = ConcatDataset(
                {task_name: AseDBDataset(train_config)}, 
                sampling=sampling_config
            )
            
            # Create validation dataset
            val_config = base_dataset_config.copy()
            val_config["src"] = val_db_path
            val_dataset = ConcatDataset(
                {task_name: AseDBDataset(val_config)}, 
                sampling=sampling_config
            )
            
            # --- 3. Define Tasks ---
            # We need to create Task objects manually
            
            # --- 3. Define Tasks ---
            tasks = []
            
            # Energy Task
            tasks.append(Task(
                name=energy_key,
                level="system",
                property="energy",
                out_spec=OutputSpec(dim=[1], dtype="float32"),
                normalizer=Normalizer(mean=0.0, rmsd=1.0),
                datasets=[task_name],
                loss_fn=DDPMTLoss(loss_fn=PerAtomMAELoss(), coefficient=config.get("energy_weight", 1.0)),
                metrics=["mae"]
            ))

            # Forces Task
            tasks.append(Task(
                name=forces_key,
                level="atom",
                property="forces",
                out_spec=OutputSpec(dim=[3], dtype="float32"),
                normalizer=Normalizer(mean=0.0, rmsd=1.0),
                datasets=[task_name],
                loss_fn=DDPMTLoss(loss_fn=L2NormLoss(), coefficient=config.get("forces_weight", 10.0)),
                metrics=["mae"],
                train_on_free_atoms=True,
                eval_on_free_atoms=True
            ))
            if has_stress:
                stress_task = Task(
                    name=stress_key,
                    level="system",
                    property="stress",
                    out_spec=OutputSpec(dim=[1, 9], dtype="float32"),
                    normalizer=Normalizer(mean=0.0, rmsd=1.0),
                    datasets=[task_name],
                    # TODO: FairChem doesn't support 3x3 MAE well in metrics yet, check this
                    loss_fn=DDPMTLoss(loss_fn=MAELoss(), coefficient=config.get("stress_weight", 1.0)),
                    metrics=["mae"]
                )
                tasks.append(stress_task)
                logging.info(f"Enabling stress task for fine-tuning with weight {config.get('stress_weight', 1.0)}")
            
            # --- 4. Dataloaders ---
            collate_fn = mt_collater_adapter(tasks)
            
            train_loader = DataLoader(
                train_dataset,
                sampler=DistributedSampler(train_dataset, shuffle=True),
                batch_size=batch_size,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            # Monkey patch for FAIRCHEM compatibility (MLIPTrainEvalUnit requires this method)
            if hasattr(train_loader, "batch_sampler"):
                train_loader.batch_sampler.set_epoch_and_start_iteration = lambda epoch, start_iteration: None

            val_loader = DataLoader(
                val_dataset,
                sampler=DistributedSampler(val_dataset, shuffle=False),
                batch_size=batch_size,
                num_workers=0,
                collate_fn=collate_fn
            )
            
            # Monkey patch for FAIRCHEM compatibility
            if hasattr(val_loader, "batch_sampler"):
                 val_loader.batch_sampler.set_epoch_and_start_iteration = lambda epoch, start_iteration: None
            
            # --- 5. Initialize Model ---
            # We need to reconstruct the model for fine-tuning
            
            # Define Heads Configuration
            heads_config = {}
            target_head_name = "shared_efs_head"
            
            # If resuming a head, we need to extract its config/state first
            pretrained_head_state_dict = None
            if init_head_name:
                logging.info(f"Attempting to resume head '{init_head_name}' from pretrained model...")
                try:
                    # Temporary load to get head info
                    temp_model, temp_ckpt = load_inference_model(ckpt_path, strict=False)
                    if hasattr(temp_model, "output_heads") and init_head_name in temp_model.output_heads:
                         # Extract config if possible (often not stored directly in head, need check model_config)
                         # But FairChem heads are usually MLP_EFS_Head. We just reuse our standard config 
                         # but load weights.
                         pretrained_head_state_dict = temp_model.output_heads[init_head_name].state_dict()
                         target_head_name = init_head_name # Use the same name
                         logging.info(f"Found head '{init_head_name}'. Weights will be reloaded after initialization.")
                    else:
                        logging.warning(f"Head '{init_head_name}' not found in pretrained model. Creating new head '{target_head_name}'.")
                    del temp_model
                    del temp_ckpt
                    import gc
                    gc.collect()
                except Exception as e:
                     logging.warning(f"Failed to extract head '{init_head_name}': {e}. Creating new head.")

            heads_config = {
                target_head_name: {
                    "module": "fairchem.core.models.uma.escn_md.MLP_EFS_Head",
                    "wrap_property": True,  # Output dict {"energy": tensor} for each key
                    "prefix": task_name,    # Output keys: take simple "omat" etc
                }
            }
            
            try:
                ckpt_path = pretrained_checkpoint_path_from_name(self.model_name)
            except Exception:
                # Fallback if it's a local path
                ckpt_path = self.model_name
                
            logging.info(f"Using checkpoint for fine-tuning: {ckpt_path}")

            # overrides for initialize_finetuning_model
            # Default to freezing backbone for efficiency
            freeze_backbone = config.get("freeze_backbone", True)
            
            overrides = {
                "backbone": {
                     "otf_graph": True,
                     # "max_neighbors": 300, # Use model default or config
                     "regress_stress": has_stress, 
                     "direct_forces": False,
                },
                "pass_through_head_outputs": True,
                "freeze_backbone": freeze_backbone
            }
            
            if freeze_backbone:
                logging.info("Enabling FAIRCHEM backbone freezing")
            
            model = initialize_finetuning_model(
                checkpoint_location=ckpt_path,
                overrides=overrides,
                heads=heads_config,
                strict=False # Often needed when changing heads
            )

            # Reload head weights if resuming
            if pretrained_head_state_dict is not None and target_head_name in model.output_heads:
                try:
                    # The prefix might have changed (e.g. from 'omat' to 'omat' it's fine)
                    # But if we are fine-tuning on a NEW task name, the logic in forward might use new keys?
                    # MLP_EFS_Head uses 'prefix' arg which is set in __init__.
                    # Since we re-initialized the head with correct prefix, loading state dict (linear weights) is safe.
                    model.output_heads[target_head_name].load_state_dict(pretrained_head_state_dict)
                    logging.info(f"Successfully reloaded weights for head '{target_head_name}'")
                except Exception as e:
                    logging.error(f"Error reloading head weights: {e}")

            # --- 6. Configure Optimization ---
            # Scheduler (Aligned with official FairChem defaults)
            # warmup_epochs=0.01, lr_min_factor=0.01, warmup_factor=0.2
            cosine_lr_fn = lambda optimizer, n_iters_per_epoch: _get_consine_lr_scheduler(
                warmup_factor=0.2,
                warmup_epochs=0.01,
                lr_min_factor=0.01,
                epochs=epochs,
                n_iters_per_epoch=n_iters_per_epoch,
                optimizer=optimizer
            )
            
            # Optimizer (AdamW defaults from log)
            # Log shows: lr=0.0002 (2e-4), weight_decay=0.001
            # Must use partial because MLIPTrainEvalUnit accesses .keywords
            optimizer_fn = partial(
                torch.optim.AdamW, 
                lr=learning_rate, 
                weight_decay=config.get("weight_decay", 0.001)
            )
            
            # --- 7. Configure Runner and Unit ---
            
            # Fake job config
            dummy_config_path = os.path.join(temp_dir, "config.yaml")
            
            # The config file must have a 'job' key
            full_config = OmegaConf.create({
                "job": {
                    "debug": False,
                    "logger": False, # Disable wandb for now
                    "scheduler": {"ranks_per_node": 1},
                    "metadata": {
                        "checkpoint_dir": temp_dir,
                        "config_path": dummy_config_path
                    }
                },
                "runner": {
                    "train_eval_unit": {
                        "model": {}, # Placeholder
                        "tasks": [
                            {
                                "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
                                "name": energy_key,
                                "level": "system",
                                "property": "energy",
                                "out_spec": {"dim": [1], "dtype": "float32"},
                            },
                             {
                                "_target_": "fairchem.core.units.mlip_unit.mlip_unit.Task",
                                "name": forces_key,
                                "level": "atom",
                                "property": "forces",
                                "out_spec": {"dim": [3], "dtype": "float32"},
                            }
                        ]
                    }
                }
            })
            
            # Save the dummy config
            OmegaConf.save(full_config, dummy_config_path)
            
            job_config = full_config.job
            OmegaConf.set_readonly(job_config, False)
            job_config.logger = False
            job_config.debug = False

            # --- 7. Initialize history and logger ---
            for key in ["epoch", "loss_train", "loss_val", "energy_mae_train", "energy_mae_val", 
                        "force_mae_train", "force_mae_val", "stress_mae_train", "stress_mae_val"]:
                if key not in self._training_history:
                    self._training_history[key] = []
            
            # Create our history logger pointing directly to wrapper's history
            history_logger = FAIRCHEMHistoryLogger(self._training_history)
            
            # --- 8. Setup Runner ---
            unit = MLIPTrainEvalUnit(
                job_config=job_config,
                model=model,
                optimizer_fn=optimizer_fn,
                # Note: cosine_lr_scheduler_fn needs partial that takes (optimizer, n_iters_per_epoch)
                # But MLIPTrainEvalUnit calls it as: self.lr_scheduler_fn(self.optimizer, self.n_iters_per_epoch)
                # So we simply pass our pre-constructed lambda.
                cosine_lr_scheduler_fn=cosine_lr_fn,
                tasks=tasks,
                print_every=1
            )
            # Clip grad norm default 100 as per Log
            unit.clip_grad_norm = config.get("clip_grad_norm", 100.0)
            
            # Overwrite logger to catch metrics
            unit.logger = history_logger

            runner = TrainEvalRunner(
                train_eval_unit=unit,
                train_dataloader=train_loader,
                eval_dataloader=val_loader,
                max_epochs=epochs,
                callbacks=[TrainCheckpointCallback(checkpoint_every_n_steps=100)],
            )
            
            # Manually attach job_config as it is not an argument in __init__ but used in run()
            runner.job_config = job_config
            
            # --- 8. Run Training ---
            logging.info("Starting fine-tuning...")
            
            # Reset model to train mode
            model.train()

            runner.run()
            
            # Flush remaining training metrics
            history_logger.finalize()
            
            # --- 9. Save Artifacts ---
            # Find best or final checkpoint
            final_dcp_path = os.path.join(temp_dir, "final")
            target_pt_path = os.path.join(save_dir, "finetuned_model.pt")
            
            if os.path.exists(final_dcp_path):
                convert_train_checkpoint_to_inference_checkpoint(final_dcp_path, target_pt_path)
                logging.info(f"Saved fine-tuned model to {target_pt_path}")
                
                # Update the current model to the fine-tuned one
                self.load(target_pt_path)
                self.is_fine_tuned = True
                
                try:
                    # Construct history dictionary compatible with MLIPModel.plot_training_history
                    # Metrics are already in self._training_history thanks to history_logger
                    # We just need to add label distributions
                    
                    # Also populate label distributions for plotting
                    try:
                        # Helper to extract dicts for base method
                        train_data_dicts = []
                        for atoms in train_structures:
                            d = {'structure': atoms}
                            # check info or calc
                            if atoms.get_calculator():
                                try:
                                    d['energy'] = atoms.get_potential_energy()
                                    d['forces'] = atoms.get_forces()
                                    d['stress'] = atoms.get_stress() if hasattr(atoms, 'get_stress') else None
                                except: pass
                            elif 'energy' in atoms.info:
                                d['energy'] = atoms.info['energy']
                                d['forces'] = atoms.arrays.get('forces')
                                d['stress'] = atoms.info.get('stress')
                            
                            train_data_dicts.append(d)
                            
                        # Call the distribution collector from base class
                        dist_data = self._collect_label_distributions(train_data_dicts)
                        self._training_history.update(dist_data)
                    except Exception as e:
                        logging.warning(f"Could not collect label distributions: {e}")
                    
                    # Update the collected history with distribution data
                    # dist_data already updated above if successful
                    history = self._training_history
                    
                    # Save numerical history to JSON
                    json_path = Path(save_dir) / "training_history.json"
                    self.save_training_history(str(json_path))
                    
                    # Plot training history
                    plot_path = Path(save_dir) / "training_history.png"
                    try:
                        self.plot_training_history(save_path=str(plot_path), show=False)
                        logging.info(f"Training history plot saved to {plot_path}")
                    except Exception as e:
                        logging.warning(f"Failed to generate training history plot: {e}")
                        
                except Exception as e:
                    logging.warning(f"Could not parse training history: {e}")
                
                return {
                    "is_fine_tuned": True,
                    "model_saved_to": str(target_pt_path),
                    "training_history": history,
                    "final_metrics": history.get("val_loss", [])[-1] if history.get("val_loss") else None,
                    "plot_path": str(plot_path) if 'plot_path' in locals() else None
                }
            else:
                logging.error("Final checkpoint not found!")
                raise RuntimeError("Fine-tuning failed to produce a final checkpoint.")
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint with model state
        if hasattr(self.model, 'state_dict'):
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name,
                'model_version': self.model_version,
                'is_fine_tuned': self.is_fine_tuned
            }
        else:
            # Basic checkpoint without model state
            checkpoint = {
                'model_name': self.model_name,
                'model_version': self.model_version,
                'is_fine_tuned': self.is_fine_tuned
            }
        
        if self.training_history:
            checkpoint['training_history'] = self.training_history
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Model checkpoint saved to {checkpoint_path}")
    
    def _safe_load_state_dict(self, state_dict):
        target_model = self.model
        if not hasattr(target_model, "load_state_dict") and hasattr(target_model, "model"):
            target_model = target_model.model
        
        if hasattr(target_model, "load_state_dict"):
             msg = target_model.load_state_dict(state_dict, strict=False)
             logging.info(f"Loaded state dict: {msg}")
        else:
             logging.warning("Model does not support load_state_dict and no underlying model found. Skipping state load (assuming load_inference_model handled it).")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load a checkpoint for the model.
        
        Args:
            checkpoint_path: Path to the checkpoint file
        """
        import torch
        from pathlib import Path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        try:
            # Try loading with weights_only=False because FAIRCHEM checkpoints contain custom classes
            # The device is not available in this scope, using 'cpu' as a fallback.
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            
            # Extract metadata early to setup correct base model
            loaded_model_name = None
            if hasattr(checkpoint, "model_config") and hasattr(checkpoint.model_config, "model_name"):
                 loaded_model_name = checkpoint.model_config.model_name
            elif isinstance(checkpoint, dict):
                 loaded_model_name = checkpoint.get('model_name')
            elif hasattr(checkpoint, 'model_name'):
                 loaded_model_name = getattr(checkpoint, 'model_name')
            
            if loaded_model_name and loaded_model_name in MODEL_METADATA:
                 logging.info(f"Inferred base model from checkpoint: {loaded_model_name}")
                 self.model_name = loaded_model_name
                 self._setup_model_classes()
            
            # Now load base model if needed
            if not self.is_loaded:
                 # If model_name is still a path and we couldn't infer, we default to UMA-S-1P1 as reasonable fallback?
                 # Or we fail. UMA-S-1P1 is a safe bet for most testing.
                 if self.model_name not in MODEL_METADATA and Path(self.model_name).exists():
                     logging.warning(f"Could not infer base model from checkpoint {checkpoint_path}. Defaulting to 'uma-s-1p1'.")
                     self.model_name = "uma-s-1p1"
                     self._setup_model_classes()
                 
                 self.load()
            
            # If it's a fine-tuned model (MLIPInferenceCheckpoint)
            if hasattr(checkpoint, "model_config") and hasattr(checkpoint, "model_state_dict"):
                self.model_config = checkpoint.model_config
                self.model_state_dict = checkpoint.model_state_dict
                self.is_fine_tuned = True
            elif "state_dict" in checkpoint:
                 self.model_state_dict = checkpoint["state_dict"]
            else:
                 # Standard FAIRCHEM checkpoint dict?
                 self.model_state_dict = checkpoint.get("model", checkpoint)
                 
        except Exception as e:
            logging.error(f"Failed to load checkpoint: {e}")
            raise FileNotFoundError(f"Checkpoint not found or invalid: {checkpoint_path} -> {e}")
        
        # Load model metadata first
        # Attributes might be missing on MLIPInferenceCheckpoint or be in model_config
        if isinstance(checkpoint, dict):
            self.model_name = checkpoint.get('model_name', self.model_name)
            self.model_version = checkpoint.get('model_version', self.model_version)
            self.training_history = checkpoint.get('training_history', None)
            self.is_fine_tuned = checkpoint.get('is_fine_tuned', False)
        else:
            # Handle object-based checkpoint (MLIPInferenceCheckpoint)
            self.model_name = getattr(checkpoint, 'model_name', self.model_name)
            self.model_version = getattr(checkpoint, 'model_version', self.model_version)
            self.training_history = getattr(checkpoint, 'training_history', None)
            # Prioritize attribute from checkpoint object, but fallback to what we detected
            self.is_fine_tuned = getattr(checkpoint, 'is_fine_tuned', self.is_fine_tuned)
            
            # MLIPInferenceCheckpoint usually stores config in model_config
            if hasattr(checkpoint, "model_config"):
                 # Try to dig info from config if needed?
                 pass

        if self.model_state_dict is None:
             raise ValueError("Could not extract model state dict from checkpoint")
        
        # Load the model if not already loaded
        # (Already handled above)
        if not self.is_loaded:
             self.load()
        
        # Load model state if available
        if self.model is not None:
            # Handle loading state dict into model
            if self.is_fine_tuned and self.model_state_dict is not None:
                logging.info("Loading fine-tuned state dictionary...")
                self._safe_load_state_dict(self.model_state_dict)
            
            elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                 self._safe_load_state_dict(checkpoint['model_state_dict'])
            elif hasattr(checkpoint, 'model_state_dict'):
                 self._safe_load_state_dict(checkpoint.model_state_dict)
            elif isinstance(checkpoint, dict):
                 # Assume checkpoint IS the state dict if keys match params?
                 # Or it contains 'state_dict' key
                 if 'state_dict' in checkpoint:
                      self._safe_load_state_dict(checkpoint['state_dict'])
                 else:
                      # Try loading as is
                      try:
                          self._safe_load_state_dict(checkpoint)
                      except Exception as e:
                          logging.warning(f"Could not load checkpoint directly as state dict: {e}")
        
        self.is_loaded = True
        logging.info(f"Model checkpoint loaded from {checkpoint_path}")
    
    def get_model_capabilities(self) -> Dict[str, bool]:
        """
        Get model capabilities.
        
        Returns:
            Dictionary of capabilities.
        """
        return {
            "energy": True,
            "forces": True,
            "stress": True,
            "optimization": True,
            "relaxation": True
        }
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple:
        """
        Prepare training data in FAIRCHEM format.
        
        Args:
            training_data: List of training samples with 'structure' key containing
                         ASE Atoms objects and 'energy', 'forces', 'stress' keys.
        
        Returns:
            Tuple of (structures, energies, forces, stresses)
        """
        structures = []
        energies = []
        forces = []
        stresses = []
        
        for data in training_data:
            structures.append(data['structure'])
            energies.append(data['energy'])
            forces.append(np.array(data['forces']))
            stresses.append(np.array(data['stress']))
        
        return structures, np.array(energies), forces, stresses
    
    @staticmethod
    def get_available_models() -> List[str]:
        """
        Get list of all available FAIRCHEM model names.
        
        Returns:
            List of available model names.
        """
        return list(MODEL_METADATA.keys())
    
    @staticmethod
    def get_standard_models() -> List[str]:
        """
        Get list of standard FAIRCHEM model types.
        
        Returns:
            List of standard model types.
        """
        # Standard models are just keys of MODEL_METADATA now, or we can define a subset if needed.
        # For now, return all available keys as "standard".
        return list(MODEL_METADATA.keys())
    
    def get_supported_elements(self) -> List[str]:
        """
        Get list of chemical elements supported by the loaded model.
        
        Returns:
            List of element symbols supported by the current model.
        """
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        from ase.data import chemical_symbols

        # Unwrap the model if necessary (e.g. AveragedModel)
        model = self.model.model
        if hasattr(model, 'module'):
            model = model.module
            
        # Check for max_num_elements in backbone (common for EquiformerV2/HydraModel used in FAIRCHEM)
        if hasattr(model, 'backbone') and hasattr(model.backbone, 'max_num_elements'):
            num_elements = model.backbone.max_num_elements
            # max_num_elements defines the size of the embedding layer (nn.Embedding(num_elements, ...))
            # which supports indices 0 to num_elements-1.
            # Since atomic numbers start at 1 (H=1), the supported Z range is 1 to num_elements-1.
            return [chemical_symbols[i] for i in range(1, num_elements) if i < len(chemical_symbols)]
        
        # Fallback: All FAIRCHEM models (UMA/ESEN) typically support 1-100
        logger.info(f"Could not determine exact elements from model structure, defaulting to 1-100 for {self.model_name}")
        return [chemical_symbols[i] for i in range(1, 101) if i < len(chemical_symbols)]


if __name__ == "__main__":
    # Test the wrapper
    wrapper = FAIRCHEMWrapper()
    print(f"FAIRCHEM available: {FAIRCHEM_AVAILABLE}")
    try:
         wrapper.load()
         print(f"Model capabilities: {wrapper.get_model_capabilities()}")
         print(f"Supported elements: {len(wrapper.get_supported_elements())}")
    except Exception as e:
         print(f"Load failed (expected if no GPU or environment): {e}")
