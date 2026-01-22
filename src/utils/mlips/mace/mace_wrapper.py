"""
MACE model wrapper for MLIP Agent
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator

from src.utils.mlips.base import MLIPModel
from src.utils.wandb_utils import init_wandb, log_training_history, finish_wandb

logger = logging.getLogger(__name__)

# Available MACE checkpoints and models
AVAILABLE_MACE_MODELS = {
    # MACE-MH (Multi-Head) models - MACE-MH-1 is the latest recommended default
    "MACE": "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model",
    "MACE-MP": "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model", 
    "MACE-MH-1": "https://huggingface.co/mace-foundations/mace-mh-1/resolve/main/mace-mh-1.model",
    "MACE-MH-0": "https://huggingface.co/mace-foundations/mace-mh-0/resolve/main/mace-mh-0.model",
    
    # MACE-MP (Materials Project) models
    "MACE-MP-small": "small",
    "MACE-MP-medium": "medium",
    "MACE-MP-large": "large",
    
    # MACE-MP second generation (recommended for fine-tuning/stability)
    "MACE-MP-small-0b": "small-0b",
    "MACE-MP-medium-0b": "medium-0b",
    "MACE-MP-small-0b2": "small-0b2",
    "MACE-MP-medium-0b2": "medium-0b2",
    "MACE-MP-large-0b2": "large-0b2",
    "MACE-MP-medium-0b3": "medium-0b3",
    
    # MACE-MPA (trained on enlarged datasets)
    "MACE-MPA-0": "medium-mpa-0",
    
    # MACE-OMAT (trained on OMAT dataset)
    "MACE-OMAT-0-small": "small-omat-0",
    "MACE-OMAT-0-medium": "medium-omat-0",
    
    # MACE-OFF (Organic Transferable Force Fields)
    "MACE-OFF23-small": "MACE-OFF23-small",
    "MACE-OFF23-medium": "MACE-OFF23-medium", 
    "MACE-OFF23-large": "MACE-OFF23-large",
    
    # MACE-MATPES (Fine-tuned on MATPES dataset)
    "MACE-MATPES-PBE-0": "mace-matpes-pbe-0",
    "MACE-MATPES-R2SCAN-0": "mace-matpes-r2scan-0",
    
    # MACE-OMOL (Organic Molecules with charges/spins)
    "MACE-OMOL-extra-large": "extra_large",
    
    # ANI-CC (ANI Coupled-Cluster accuracy)
    "MACE-ANI-CC": "mace-anicc",
}


# Try to import MACE components with clear error messages
MACE_AVAILABLE = False

try:
    import mace
    from mace.calculators import mace_mp
    from mace.tools import utils
    MACE_AVAILABLE = True
except ImportError as e:
    import os
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    if not current_env.startswith('mace'):
        raise ImportError(
            f"MACE is not available in the current conda environment '{current_env}'. "
            f"MACE models require the 'mace-agent' conda environment. "
            f"Please run this code in the mace-agent environment:\n"
            f"  conda activate mace-agent\n"
            f"Or use subprocess execution via MLIPModelTool which handles this automatically.\n"
            f"Original error: {e}"
        ) from e
    raise


class MACEWrapper(MLIPModel):
    """
    MACE model wrapper implementing the MLIPModel interface.
    """
    
    def __init__(
        self, 
        model_name: str = "MACE", 
        model_version: str = "latest",
        device: str = "auto",
        head: Optional[str] = None
    ):
        """
        Initialize MACE wrapper.
        
        Args:
            model_name: Name of the model ("MACE", "MACE-MP", etc.)
            model_version: Version of the model
            device: Device to run the model on ("auto", "cpu", "cuda")
            head: Optional head name for multi-head models (e.g. "omat_pbe")
        """
        super().__init__(model_name, model_version)
        self.device = device
        self.head = head
        self.model_class = None
        self.calculator_class = None
        self.model = None
        self.trainer = None
        self.model_path = None
        
        # Note: MACE_AVAILABLE is False when package is not installed
        # This is expected in testing environments
        
    def _get_available_models(self) -> List[str]:
        """Get list of available MACE model names (all checkpoints)."""
        return sorted(list(AVAILABLE_MACE_MODELS.keys()))
    
    def load(self, model_path: Optional[str] = None) -> None:
        """
        Load a MACE model.
        
        Args:
            model_path: Path to model checkpoint. If None, loads default pretrained model.
                       If provided, loads checkpoint from the specified path.
        """
        # Set device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if model_path is None:
            # Load pretrained model
            # Store the model name for calculator creation
            if Path(self.model_name).exists():
                self.model_path = self.model_name
                self.load_checkpoint(self.model_name)
                logger.info(f"Loaded model from path: {self.model_name}")
            else:
                self.model = "MACE"
                self.is_loaded = True
                logger.info(f"Loaded pretrained {self.model_name} model")
        else:
            # Load from checkpoint file
            self.model_path = model_path
            self.load_checkpoint(model_path)
            logger.info(f"Loaded model from checkpoint: {model_path}")
    
    def create_calculator(self) -> Calculator:
        """
        Create an ASE calculator from the model.
        
        Returns:
            ASE Calculator object.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # 1. Custom model path (e.g. fine-tuned)
        if self.model_path and Path(self.model_path).exists():
            from mace.calculators import MACECalculator
            return MACECalculator(model_paths=str(self.model_path), device=self.device, head=self.head)

        # 2. Check if model_name is a path (fallback)
        if Path(self.model_name).exists():
             from mace.calculators import MACECalculator
             return MACECalculator(model_paths=str(self.model_name), device=self.device, head=self.head)

        # 3. Standard Pretrained mapping
        model_id = AVAILABLE_MACE_MODELS.get(self.model_name)
        
        if model_id is None:
            available_models = self._get_available_models()
            raise ValueError(
                f"Unknown model name: '{self.model_name}'. "
                f"Available models: {', '.join(available_models)}. "
                f"Please use one of the available model names."
            )
        
        # Determine which calculator function to use based on model type
        model_name_upper = self.model_name.upper()
        
        # Special case for MACE-MH models (direct URL or explicit name)
        if 'MH-' in model_name_upper or model_name_upper in ['MACE', 'MACE-MP'] or model_id.startswith('http'):
            # Use direct MACECalculator for URLs or MACE-MH models
            from mace.calculators import MACECalculator
            # Download if it's a URL
            if model_id.startswith('http'):
                from mace.calculators.foundations_models import download_mace_mp_checkpoint
                model_path = download_mace_mp_checkpoint(model_id)
            else:
                model_path = model_id
            
            # Default head for MH-1 and MH-0 is omat_pbe if not specified
            head = self.head
            if head is None and ('MH-1' in model_name_upper or 'MH-0' in model_name_upper or model_name_upper in ['MACE', 'MACE-MP']):
                head = "omat_pbe"
                
            return MACECalculator(model_paths=model_path, device=self.device, head=head)

        if 'OFF' in model_name_upper:
            # Use MACE-OFF calculator
            from mace.calculators import mace_off
            off_model_mapping = {
                "MACE-OFF23-SMALL": "small",
                "MACE-OFF23-MEDIUM": "medium", 
                "MACE-OFF23-LARGE": "large"
            }
            off_model_size = off_model_mapping.get(model_name_upper, "medium")
            return mace_off(model=off_model_size, device=self.device)
            
        elif 'ANI-CC' in model_name_upper:
            # Use MACE-ANI-CC
            from mace.calculators import mace_anicc
            return mace_anicc(device=self.device)
            
        elif 'OMOL' in model_name_upper:
            # Use MACE-OMOL
            from mace.calculators import mace_omol
            return mace_omol(model=model_id, device=self.device)
            
        else:
            # Use MACE-MP calculator for materials models (including MPA, MATPES, OMAT)
            from mace.calculators import mace_mp
            return mace_mp(model=model_id, device=self.device)
    
    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save a model checkpoint.
        
        Args:
            checkpoint_path: Path to save the checkpoint.
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # For MACE, we save the calculator state
            checkpoint = {
                'model_name': self.model_name,
                'model_version': self.model_version,
                'is_fine_tuned': self.is_fine_tuned,
                'device': self.device
            }
            
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise RuntimeError(f"Failed to save checkpoint: {e}")
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load a model from a checkpoint.
        
        Args:
            checkpoint_path: Path to the checkpoint file.
        """
        try:
            checkpoint_path = Path(checkpoint_path)
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Check if checkpoint is a dict (standard checkpoint)
            if isinstance(checkpoint, dict):
                # Load metadata
                self.model_name = checkpoint.get('model_name', self.model_name)
                self.model_version = checkpoint.get('model_version', self.model_version)
                self.is_fine_tuned = checkpoint.get('is_fine_tuned', False)
                self.device = checkpoint.get('device', self.device)
            else:
                # Assume it's a compiled model or other format (e.g. TorchScript)
                # We can't extract metadata easily, but we trust the user provided a valid model
                logger.info(f"Loaded checkpoint is not a dictionary (type: {type(checkpoint)}). Assuming compiled model or direct model object.")
                # Keep existing metadata
                
            self.is_loaded = True
            
            logger.info(f"Model checkpoint loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise RuntimeError(f"Failed to load checkpoint: {e}")
    
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
    
    def fine_tune(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune the MACE model on provided training data.
        
        Args:
            training_data: List of training samples
            validation_data: Optional validation data
            training_config: Training configuration
            output_dir: Directory to save the fine-tuned model
            wandb_config: Optional wandb configuration dictionary
        
        Returns:
            Dictionary containing training history and metrics
        """
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before fine-tuning")
        
        # Default training configuration
        default_config = {
            "max_epochs": 100,
            "learning_rate": 1e-4,
            "batch_size": 32,
            "validation_split": 0.1,
            "early_stopping_patience": 10,
            "save_best_model": True
        }
        
        if training_config:
            default_config.update(training_config)
        
        logger.info("Using MACE training pipeline for fine-tuning")

        # Check for model/checkpoint override
        if training_config:
            new_model = training_config.get("foundation_model") or training_config.get("checkpoint_path")
            if new_model and new_model != self.model_name:
                logger.info(f"Reloading model for fine-tuning: {self.model_name} -> {new_model}")
                self.model_name = new_model
                self.is_loaded = False
                self.load(model_path=new_model if os.path.exists(new_model) else None)
        
        # Initialize wandb if configured
        wandb_run = None
        if wandb_config:
            # Prepare wandb config
            wandb_init_config = {
                **default_config,
                "model_name": self.model_name,
                "model_version": self.model_version,
                "num_training_samples": len(training_data),
                "num_validation_samples": len(validation_data) if validation_data else 0
            }
            
            wandb_run = init_wandb(
                project=wandb_config.get("project"),
                entity=wandb_config.get("entity"),
                name=wandb_config.get("name"),
                tags=wandb_config.get("tags", []),
                config=wandb_init_config,
                mode=wandb_config.get("mode")
            )
        
        # Convert training data to MACE format
        train_structures, train_energies, train_forces, train_stresses = self._prepare_training_data(training_data)
        
        # Prepare validation data if provided
        val_structures, val_energies, val_forces, val_stresses = None, None, None, None
        if validation_data:
            val_structures, val_energies, val_forces, val_stresses = self._prepare_training_data(validation_data)
        
        # Create temporary directory for MACE training
        if output_dir is None:
            import tempfile
            output_dir = tempfile.mkdtemp(prefix="mace_fine_tuned_")
            logger.info(f"Using temporary directory for fine-tuned model: {output_dir}")
        
        output_path = Path(output_dir).absolute()
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save training data as XYZ files for MACE
        train_xyz_path = output_path / "train.xyz"
        val_xyz_path = output_path / "valid.xyz" if val_structures else None
        
        # Write training data to XYZ file
        self._write_xyz_file(train_structures, train_energies, train_forces, train_stresses, train_xyz_path)
        
        if val_xyz_path:
            self._write_xyz_file(val_structures, val_energies, val_forces, val_stresses, val_xyz_path)
        
        # Create MACE training configuration
        # config_path = output_path / "config.yaml"
        # self._create_mace_config(config_path, default_config, train_xyz_path, val_xyz_path)
        
        # Run MACE training using the foundation model via CLI
        model_save_path = output_path / f"{self.model_name.lower()}_fine_tuned.model"
        
        # Use MACE CLI via subprocess (mace.cli.run_train is the correct CLI module)
        import subprocess
        import sys
        
        # Set up directories
        checkpoints_dir = output_path / "checkpoints"
        results_dir = output_path / "results"
        log_dir = output_path / "logs"
        checkpoints_dir.mkdir(exist_ok=True)
        results_dir.mkdir(exist_ok=True)
        log_dir.mkdir(exist_ok=True)
        
        # Determine device - use GPU if available
        import torch
        if self.device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            else:
                device = "cpu"
                logger.info("No GPU available, using CPU")
        else:
            device = self.device
        
        # Check if stress is present in training data
        has_stress = any(s is not None for s in train_stresses)
        if has_stress:
            logger.info("Stress data detected in training data - will include stress in loss function")
        
        # Build MACE training command
        multiheads_finetuning = training_config.get("multiheads_finetuning", False) if training_config else False
        
        cmd = [
            sys.executable, "-m", "mace.cli.run_train",
            "--name", f"{self.model_name.lower()}_fine_tuned",
            "--train_file", str(train_xyz_path),
            "--max_num_epochs", str(default_config["max_epochs"]),
            "--lr", str(default_config["learning_rate"]),
            "--batch_size", str(default_config["batch_size"]),
            "--valid_batch_size", str(default_config["batch_size"]),
            "--energy_key", "REF_energy",
            "--forces_key", "REF_forces",
            "--device", device,
            "--seed", "42",
            "--checkpoints_dir", str(checkpoints_dir),
            "--results_dir", str(results_dir),
            "--log_dir", str(log_dir),
            "--plot", "True",
            "--plot_frequency", "1",
            "--multiheads_finetuning", str(multiheads_finetuning)
        ]
        
        if not multiheads_finetuning:
             logger.info("Disabling MACE multi-head replay (direct fine-tuning mode)")
        
        # 2. Handle Layer Freezing (Default: True)
        freeze_backbone = training_config.get("freeze_backbone", True) if training_config else True
        
        if freeze_backbone:
            # Flexible module-based unfreezing
            # Default to unfreezing only readouts if freeze_backbone is True
            trainable_modules = training_config.get("trainable_modules", ["readouts"])
            if isinstance(trainable_modules, str):
                trainable_modules = [trainable_modules]
            
            logger.info(f"Enabling MACE flexible freezing. Trainable modules: {trainable_modules}")
            
            # Use the separate patch script created in utils/mlips/mace/
            wrapper_script_path = output_path / "mace_run_train_wrapper.py"
            patch_script_path = Path(__file__).parent / "freeze_patch.py"
            
            with open(wrapper_script_path, "w") as f:
                f.write(f"""
import sys
import os
import logging
from pathlib import Path

# Add the directory containing freeze_patch.py to sys.path
sys.path.append(r"{str(patch_script_path.parent)}")

import freeze_patch
import mace.cli.run_train

if __name__ == "__main__":
    # Apply the patch with user-specified trainable modules
    freeze_patch.apply_patch({trainable_modules})
    
    # Run the main training loop
    mace.cli.run_train.main()
""")
            cmd[1] = str(wrapper_script_path)
            cmd.pop(2) # remove "-m" or "mace.cli.run_train"
        
        # Add stress support if stress is present in training data
        if has_stress:
            # Use universal loss which supports energy, forces, and stress
            cmd.extend(["--loss", "universal"])
            # Set stress weight (default is 1.0, but make it explicit)
            stress_weight = training_config.get("stress_weight", 1.0) if training_config else 1.0
            cmd.extend(["--stress_weight", str(stress_weight)])
            
            # Pass arbitrary arguments from training_config to CLI
            reserved_keys = ["max_epochs", "learning_rate", "batch_size", "validation_split", 
                             "early_stopping_patience", "save_best_model", "use_foundation_model", 
                             "stress_weight", "project", "entity", "name", "tags", "mode"]
            
            if training_config:
                for k, v in training_config.items():
                    if k not in reserved_keys:
                        # Convert value to string format for CLI
                        # Handle dicts/lists by converting to string representation
                        val_str = str(v).replace("'", '"') if isinstance(v, (dict, list)) else str(v)
                        cmd.extend([f"--{k}", val_str])
                        logger.info(f"Passing custom arg to MACE: --{k} {val_str}")
            # Specify stress key name in XYZ file
            cmd.extend(["--stress_key", "REF_stress"])
            # Enable stress computation and reporting
            cmd.extend(["--compute_stress", "True"])
            # Use error table that includes stress/virials MAE
            cmd.extend(["--error_table", "PerAtomMAEstressvirials"])
            logger.info(f"Including stress in loss function with weight {stress_weight} and enabling stress MAE reporting")
        
        # Add validation file if provided
        if val_xyz_path:
            cmd.extend(["--valid_file", str(val_xyz_path)])
        
        # Add foundation model if specified and enabled in config
        use_foundation_model = True

        # Check if we should use foundation model for E0s
        if training_config:
            use_foundation_model = training_config.get("use_foundation_model", True)
            
        if use_foundation_model:
            # Determine foundation model name based on model_name
            model_name_upper = self.model_name.upper()
            if "MATPES-R2SCAN" in model_name_upper or "MATPES-r2SCAN" in model_name_upper:
                foundation_model_name = "mace-matpes-r2scan-0"
            elif "MATPES-PBE" in model_name_upper:
                foundation_model_name = "mace-matpes-pbe-0"
            elif "MPA" in model_name_upper:
                foundation_model_name = "mace-mpa-0"
            elif "OMAT" in model_name_upper:
                if "SMALL" in model_name_upper:
                    foundation_model_name = "small-omat-0"
                else:
                    foundation_model_name = "medium-omat-0"
            else:
                foundation_model_name = "small"
            
            cmd.extend(["--foundation_model", foundation_model_name])
            # Keep all elements from foundation model to avoid shape mismatches
            # When foundation_model_elements is True, MACE keeps all elements from the foundation model
            # This ensures the atomic_energies_fn has the correct shape (89 elements) even if training data only has 2
            # cmd.extend(["--foundation_model_elements", "False"])
            logger.info(f"Using foundation model: {foundation_model_name} for fine-tuning {self.model_name} (keeping all foundation model elements)")
            
            # When using --foundation_model, MACE should automatically extract E0s from the foundation model
            # However, MACE's training code requires E0s to be explicitly provided even with foundation models
            # We extract E0s from the foundation model to ensure they remain unchanged
            # The key is to provide them in the correct format that MACE expects
            try:
                from mace.calculators import mace_mp
                foundation_calc = mace_mp(model=foundation_model_name, dispersion=False)
                if hasattr(foundation_calc, 'models') and len(foundation_calc.models) > 0:
                    model = foundation_calc.models[0]
                    if hasattr(model, "atomic_energies_fn") and hasattr(model.atomic_energies_fn, "atomic_energies"):
                        atomic_energies_fn = model.atomic_energies_fn
                        atomic_numbers = model.atomic_numbers
                        
                        if hasattr(atomic_energies_fn, 'atomic_energies'):
                            # Extract E0s: atomic_energies is a tensor [num_elements] or [1, num_elements]
                            e0s_tensor = atomic_energies_fn.atomic_energies
                            # Convert to dict mapping atomic numbers to E0 values
                            e0s_dict = {}
                            for i, z in enumerate(atomic_numbers):
                                try:
                                    if e0s_tensor.dim() == 1:
                                        val = e0s_tensor[i].item()
                                    else:
                                        val = e0s_tensor[0, i].item()
                                    e0s_dict[int(z.item())] = float(val)
                                except (IndexError, ValueError):
                                    continue
                            if e0s_dict:
                                # Provide E0s for ALL elements in the foundation model, not just training elements
                                # Convert to string format expected by MACE CLI (Python dict literal)
                                e0s_str = str(e0s_dict)
                                cmd.extend(["--E0s", e0s_str])
                                
                                # Explicitly pass atomic numbers to ensure correct model dimensions
                                atomic_numbers_list = list(sorted(e0s_dict.keys()))
                                cmd.extend(["--atomic_numbers", str(atomic_numbers_list)])
                                
                                logger.info(f"Extracted E0s from foundation model: {len(e0s_dict)} elements")
                                logger.info(f"Passing atomic numbers: {atomic_numbers_list}")
                            else:
                                logger.warning("Could not extract E0s from foundation model")
                        else:
                            logger.warning("Foundation model atomic_energies_fn does not have atomic_energies attribute")
                    else:
                        logger.warning("Foundation model does not have required attributes for E0s extraction")
                else:
                    logger.warning("Foundation model calculator does not have models attribute")
            except Exception as e:
                logger.error(f"Failed to extract E0s from foundation model: {e}")
                raise RuntimeError(f"Cannot extract E0s from foundation model {foundation_model_name}. This is required for fine-tuning.")
        else:
            # Use average E0s from training data if not using foundation model
            cmd.extend(["--E0s", "average"])
            logger.info("Using average E0s from training data")
        
        # Run MACE training
        try:
            logger.info(f"Running MACE training with command: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=str(output_path),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Log output for debugging
            if result.stdout:
                logger.info(f"MACE training stdout (last 500 chars): {result.stdout[-500:]}")
            if result.stderr:
                logger.info(f"MACE training stderr (last 500 chars): {result.stderr[-500:]}")
            
            if result.returncode != 0:
                logger.error(f"MACE training process exited with return code {result.returncode}")
                # We'll check if a model was saved anyway, as MACE often fails at the end during plotting
            
            # Load the fine-tuned model
            model_files = []
            if checkpoints_dir.exists():
                model_files.extend(list(checkpoints_dir.glob("*.model")))
                model_files.extend(list(checkpoints_dir.glob("*_*.model")))
            if results_dir.exists():
                model_files.extend(list(results_dir.glob("*.model")))
            model_files.extend(list(output_path.glob("*.model")))
            
            if model_files:
                fine_tuned_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
                if fine_tuned_model_path != model_save_path:
                    import shutil
                    shutil.copy2(fine_tuned_model_path, model_save_path)
                    logger.info(f"Copied model from {fine_tuned_model_path} to {model_save_path}")
                self.model = "MACE"
                self.is_fine_tuned = True
                logger.info(f"Fine-tuned model saved to {model_save_path}")
            elif result.returncode != 0:
                # If no model files found and returncode was non-zero, then it's a real failure
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise RuntimeError(f"MACE training failed: {result.stderr}")
            else:
                all_files = list(output_path.rglob("*"))
                logger.warning(f"No model files found. Created files: {[str(f) for f in all_files[:20]]}")
                raise RuntimeError("MACE training completed but no model checkpoint was saved.")
                
        except (subprocess.TimeoutExpired, RuntimeError) as e:
            raise e
        except Exception as e:
            logger.error(f"Unexpected error during MACE fine-tuning: {e}")
            raise RuntimeError(f"Fine-tuning failed: {e}")
        
        # Store training history for plotting
        # Use base class method to collect label distributions consistently
        distributions = self._collect_label_distributions(training_data)
        self._training_history = {
            'energy_distribution': distributions['energy_distribution'],
            'force_distribution': distributions['force_distribution'],
            'stress_distribution': distributions['stress_distribution'],
            'energy_mae_train': [],  # Would be populated by MACE training
            'energy_mae_val': [],
            'force_mae_train': [],
            'force_mae_val': [],
            'stress_mae_train': [],
            'stress_mae_val': [],
            'loss_train': [],
            'loss_val': []
        }
        
        # Extract training history from MACE results
        # Also try to parse metrics from stdout/stderr
        if result.stdout:
            # Parse stdout for training metrics
            import re
            lines = result.stdout.split('\n')
            epoch_num = 0
            for line in lines:
                # Look for epoch information
                epoch_match = re.search(r'Epoch\s+(\d+)', line)
                if epoch_match:
                    epoch_num = int(epoch_match.group(1))
                
                # Look for MAE values in various formats
                if 'energy' in line.lower() and 'mae' in line.lower():
                    mae_match = re.search(r'([\d.]+)', line)
                    if mae_match:
                        mae_val = float(mae_match.group(1))
                        if 'train' in line.lower():
                            while len(self._training_history['energy_mae_train']) <= epoch_num:
                                self._training_history['energy_mae_train'].append(None)
                            self._training_history['energy_mae_train'][epoch_num] = mae_val
                        elif 'valid' in line.lower() or 'val' in line.lower():
                            while len(self._training_history['energy_mae_val']) <= epoch_num:
                                self._training_history['energy_mae_val'].append(None)
                            self._training_history['energy_mae_val'][epoch_num] = mae_val
                
                if 'force' in line.lower() and 'mae' in line.lower():
                    mae_match = re.search(r'([\d.]+)', line)
                    if mae_match:
                        mae_val = float(mae_match.group(1))
                        if 'train' in line.lower():
                            while len(self._training_history['force_mae_train']) <= epoch_num:
                                self._training_history['force_mae_train'].append(None)
                            self._training_history['force_mae_train'][epoch_num] = mae_val
                        elif 'valid' in line.lower() or 'val' in line.lower():
                            while len(self._training_history['force_mae_val']) <= epoch_num:
                                self._training_history['force_mae_val'].append(None)
                            self._training_history['force_mae_val'][epoch_num] = mae_val
                
                if 'loss' in line.lower() and not 'mae' in line.lower():
                    loss_match = re.search(r'loss[=:]\s*([\d.]+)', line, re.IGNORECASE)
                    if loss_match:
                        loss_val = float(loss_match.group(1))
                        if 'train' in line.lower():
                            while len(self._training_history['loss_train']) <= epoch_num:
                                self._training_history['loss_train'].append(None)
                            self._training_history['loss_train'][epoch_num] = loss_val
                        elif 'valid' in line.lower() or 'val' in line.lower():
                            while len(self._training_history['loss_val']) <= epoch_num:
                                self._training_history['loss_val'].append(None)
                            self._training_history['loss_val'][epoch_num] = loss_val
        
        
        # Extract training history from MACE results directory
        if results_dir.exists():
            # Try to read training metrics from MACE output files
            self._extract_mace_training_history(results_dir, log_dir)
        
        # Generate training history plot automatically
        plot_path = output_path / "training_history.png"
        try:
            # Temporarily set is_fine_tuned to allow plotting
            was_fine_tuned = self.is_fine_tuned
            self.is_fine_tuned = True
            self.plot_training_history(save_path=str(plot_path), show=False)
            self.is_fine_tuned = was_fine_tuned
            logger.info(f"Training history plot saved to {plot_path}")
        except Exception as e:
            logger.warning(f"Failed to generate training history plot: {e}")
        
        # Save numerical history to JSON
        json_path = output_path / "training_history.json"
        self.save_training_history(str(json_path))
        
        # Save training history to CSV
        csv_path = output_path / "training_history.csv"
        try:
            import pandas as pd
            
            # Create DataFrame from training history
            csv_data = []
            max_len = max(
                len([x for x in self._training_history.get('energy_mae_train', []) if x is not None]),
                len([x for x in self._training_history.get('energy_mae_val', []) if x is not None]),
                len([x for x in self._training_history.get('force_mae_train', []) if x is not None]),
                len([x for x in self._training_history.get('force_mae_val', []) if x is not None]),
                len([x for x in self._training_history.get('loss_train', []) if x is not None]),
                len([x for x in self._training_history.get('loss_val', []) if x is not None])
            )
            
            for epoch in range(max_len):
                row = {'epoch': epoch}
                for key in ['energy_mae_train', 'energy_mae_val', 'force_mae_train', 'force_mae_val', 
                           'stress_mae_train', 'stress_mae_val', 'loss_train', 'loss_val']:
                    arr = self._training_history.get(key, [])
                    if epoch < len(arr) and arr[epoch] is not None:
                        row[key] = arr[epoch]
                
                csv_data.append(row)
            
            if csv_data:
                df = pd.DataFrame(csv_data)
                df.to_csv(csv_path, index=False)
                logger.info(f"Training history CSV saved to {csv_path}")
        except Exception as e:
            logger.warning(f"Failed to save training history CSV: {e}")
        
        # Log training history to wandb
        if wandb_run is not None:
            # from ..utils.wandb_utils import log_training_history, finish_wandb
            final_metrics = {
                "final_energy_mae_train": self._training_history.get('energy_mae_train', [None])[-1] if self._training_history.get('energy_mae_train') else None,
                "final_energy_mae_val": self._training_history.get('energy_mae_val', [None])[-1] if self._training_history.get('energy_mae_val') else None,
                "final_force_mae_train": self._training_history.get('force_mae_train', [None])[-1] if self._training_history.get('force_mae_train') else None,
                "final_force_mae_val": self._training_history.get('force_mae_val', [None])[-1] if self._training_history.get('force_mae_val') else None,
                "final_loss_train": self._training_history.get('loss_train', [None])[-1] if self._training_history.get('loss_train') else None,
                "final_loss_val": self._training_history.get('loss_val', [None])[-1] if self._training_history.get('loss_val') else None,
                "epochs_completed": default_config["max_epochs"]
            }
            log_training_history(self._training_history, self.model_name, final_metrics)
            finish_wandb()
        
        return {
            "is_fine_tuned": True,
            "training_history": self._training_history,
            "final_metrics": {
                "epochs_completed": default_config["max_epochs"],
                "model_saved_to": str(model_save_path)
            },
            "plot_path": str(plot_path) if plot_path.exists() else None,
            "csv_path": str(csv_path) if csv_path.exists() else None
        }
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple:
        """
        Prepare training data in MACE format.
        
        Args:
            training_data: List of training samples with 'structure' key containing
                         ASE Atoms, pymatgen Structure, or dict representation
                         and 'energy', 'forces', 'stress' keys.
        
        Returns:
            Tuple of (structures, energies, forces, stresses)
        """
        from ase import Atoms
        
        structures = []
        energies = []
        forces = []
        stresses = []
        
        for data in training_data:
            # Extract and convert structure to ASE Atoms
            structure = data['structure']
            
            if isinstance(structure, Atoms):
                # Already ASE Atoms
                atoms = structure
            elif isinstance(structure, dict):
                # Dict representation - could be ASE Atoms dict or pymatgen Structure dict
                if 'positions' in structure and 'symbols' in structure:
                    # ASE Atoms dict format (from serialization)
                    atoms = Atoms(
                        symbols=structure['symbols'],
                        positions=structure['positions'],
                        cell=structure.get('cell'),
                        pbc=structure.get('pbc')
                    )
                elif '@module' in structure or '@class' in structure:
                    # pymatgen Structure dict - try to reconstruct
                    try:
                        from pymatgen.io.ase import AseAtomsAdaptor
                        from pymatgen.core import Structure
                        pmg_structure = Structure.from_dict(structure)
                        converter = AseAtomsAdaptor()
                        atoms = converter.get_atoms(pmg_structure)
                    except ImportError:
                        raise ImportError(
                            "pymatgen is required to convert pymatgen Structure dicts. "
                            "Please install pymatgen or provide structures in ASE Atoms format."
                        )
                else:
                    # Try as ASE Atoms kwargs
                    try:
                        atoms = Atoms(**structure)
                    except Exception as e:
                        raise ValueError(f"Cannot convert structure dict to ASE Atoms: {e}")
            elif hasattr(structure, 'lattice') and hasattr(structure, 'species'):
                # pymatgen Structure object
                try:
                    from pymatgen.io.ase import AseAtomsAdaptor
                    converter = AseAtomsAdaptor()
                    atoms = converter.get_atoms(structure)
                except ImportError:
                    raise ImportError(
                        "pymatgen is required to convert pymatgen Structure objects. "
                        "Please install pymatgen or provide structures in ASE Atoms format."
                    )
            else:
                raise ValueError(f"Cannot convert structure to ASE Atoms: {type(structure)}")
            
            if 'config_type' in data:
                atoms.info['config_type'] = data['config_type']
            
            import ase.units
            structures.append(atoms)
            energies.append(data['energy'])
            forces.append(np.array(data['forces']) if data.get('forces') is not None else None)
            
            stress = data.get('stress')
            if stress is not None:
                stress_array = np.array(stress)
                # Standardized labels are already in eV/A^3 (ASE standard)
                stresses.append(stress_array)
            else:
                stresses.append(None)
        
        return structures, np.array(energies), forces, stresses
    
    def _write_xyz_file(self, structures, energies, forces, stresses, filepath):
        """
        Write training data to XYZ file format for MACE.
        
        Args:
            structures: List of ASE Atoms objects
            energies: Array of energies
            forces: List of force arrays
            stresses: List of stress arrays
            filepath: Path to output XYZ file
        """
        from ase.io import write
        
        # Use ASE's write function which handles the extended XYZ format correctly
        # MACE expects REF_energy and REF_forces keys
        atoms_list = []
        for atoms, energy, force, stress in zip(structures, energies, forces, stresses):
            # Create a copy to avoid modifying original
            atoms_copy = atoms.copy()
            
            # Set energy in info dict with MACE's expected key name
            atoms_copy.info['REF_energy'] = float(energy)
            
            # Set forces as array with MACE's expected key name
            atoms_copy.arrays['REF_forces'] = np.array(force)
            
            # Set stress as flattened 3x3 matrix (9 elements) in info dict
            if stress is not None:
                stress_array = np.array(stress)
                if stress_array.shape == (3, 3):
                    # Flatten 3x3 matrix to 9-element vector
                    stress_flat = stress_array.flatten()
                elif stress_array.shape == (6,):
                    # Voigt notation - convert to 3x3 then flatten
                    # Voigt: [xx, yy, zz, yz, xz, xy]
                    stress_3x3 = np.array([
                        [stress_array[0], stress_array[5], stress_array[4]],
                        [stress_array[5], stress_array[1], stress_array[3]],
                        [stress_array[4], stress_array[3], stress_array[2]]
                    ])
                    stress_flat = stress_3x3.flatten()
                elif stress_array.shape == (9,):
                    # Already flattened
                    stress_flat = stress_array
                else:
                    logger.warning(f"Unexpected stress shape {stress_array.shape}, skipping stress for this structure")
                    stress_flat = None
                
                if stress_flat is not None:
                    atoms_copy.info['REF_stress'] = stress_flat
            
            atoms_list.append(atoms_copy)
            
        write(str(filepath), atoms_list, format='extxyz')

    def _create_mace_config(self, config_path, training_config, train_xyz_path, val_xyz_path):
        """
        Create MACE training configuration file.
        """
        config = {
            "name": f"{self.model_name.lower()}_fine_tuned",
            "train_file": str(train_xyz_path),
            "valid_file": str(val_xyz_path) if val_xyz_path else "",
            "foundation_model": "small",
            "max_num_epochs": training_config["max_epochs"],
            "lr": training_config["learning_rate"],
            "batch_size": training_config["batch_size"],
            "model": "MACE",
            "loss": "universal",
            "energy_weight": 1.0,
            "forces_weight": 10.0,
            "stress_weight": 1.0,
            "device": str(self.device),
            "seed": 42
        }
        
        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

    def _extract_mace_training_history(self, results_dir: Path, log_dir: Optional[Path] = None):
        """
        Extract training history from MACE results directory.
        """
        logger.info("Extracting training history from MACE results...")
        
        import json
        import re
        
        # MACE saves training metrics in text files with JSON lines format
        metrics_files = []
        if results_dir.exists():
            metrics_files.extend(list(results_dir.glob("*_train.txt")))
            metrics_files.extend(list(results_dir.glob("*.json")))
        
        for metrics_file in metrics_files:
            try:
                with open(metrics_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            epoch = data.get('epoch')
                            mode = data.get('mode', '')
                            if epoch is None: continue
                            epoch = int(epoch)
                            
                            if mode == 'eval':
                                if 'mae_e_per_atom' in data:
                                    mae_e = data['mae_e_per_atom'] * 1000
                                    while len(self._training_history['energy_mae_val']) <= epoch:
                                        self._training_history['energy_mae_val'].append(None)
                                    self._training_history['energy_mae_val'][epoch] = mae_e
                                if 'mae_f' in data:
                                    mae_f = data['mae_f'] * 1000
                                    while len(self._training_history['force_mae_val']) <= epoch:
                                        self._training_history['force_mae_val'].append(None)
                                    self._training_history['force_mae_val'][epoch] = mae_f
                                if 'mae_stress' in data:
                                    mae_s = data['mae_stress']
                                    if mae_s is not None:
                                        while len(self._training_history['stress_mae_val']) <= epoch:
                                            self._training_history['stress_mae_val'].append(None)
                                        self._training_history['stress_mae_val'][epoch] = mae_s
                                if 'loss' in data:
                                    while len(self._training_history['loss_val']) <= epoch:
                                        self._training_history['loss_val'].append(None)
                                    self._training_history['loss_val'][epoch] = data['loss']
                            elif mode == 'opt':
                                if 'loss' in data:
                                    while len(self._training_history['loss_train']) <= epoch:
                                        self._training_history['loss_train'].append(None)
                                    self._training_history['loss_train'][epoch] = data['loss']
                        except json.JSONDecodeError: continue
                logger.info(f"Extracted training history from {metrics_file}")
                return
            except Exception as e:
                logger.warning(f"Failed to parse metrics file {metrics_file}: {e}")

    def get_supported_elements(self) -> List[str]:
        """
        Get list of chemical elements supported by the loaded model.
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        # MACE-MP models typically support 89 elements
        try:
            temp_calc = self.create_calculator()
            if hasattr(temp_calc, 'z_table') and temp_calc.z_table is not None:
                from ase.data import chemical_symbols
                return [chemical_symbols[int(z)] for z in temp_calc.z_table.zs]
        except Exception:
            pass
            
        return ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ac", "Th", "Pa", "U", "Np", "Pu"]
