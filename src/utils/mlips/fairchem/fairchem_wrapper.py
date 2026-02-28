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

from ..base import MLIPModel

logger = logging.getLogger(__name__)







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
        
        Supports two modes:
        1. Named pretrained model (e.g., 'uma-s-1p1') — loaded via get_predict_unit
        2. Fine-tuned checkpoint file path — loaded via load_predict_unit (official API)
        
        Args:
            model_path: Path to model checkpoint. If None, uses self.model_name which
                       can be either a named model or a file path to an inference_ckpt.pt.
        """
        from pathlib import Path
        from fairchem.core.units.mlip_unit import load_predict_unit
        
        # Set device first
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Determine if we're loading a file path or a named model
        load_path = model_path or self.model_name
        is_file_path = Path(load_path).exists() and Path(load_path).is_file()
        
        if is_file_path:
            # Load fine-tuned checkpoint using official FairChem API
            logger.info(f"Loading fine-tuned checkpoint from: {load_path}")
            self.model = load_predict_unit(load_path)
            self.is_loaded = True
            self.is_fine_tuned = True
            logger.info(f"Loaded fine-tuned model from {load_path}")
        else:
            # Load pretrained model by name
            model_id = load_path
            
            # Parse inference settings
            if isinstance(self.inference_settings, (str, InferenceSettings)):
                settings = guess_inference_settings(self.inference_settings)
            elif isinstance(self.inference_settings, dict):
                settings = InferenceSettings(**self.inference_settings)
            else:
                settings = self.inference_settings
            
            self.model = self.model_class.get_predict_unit(
                model_name=model_id,
                device=self.device,
                inference_settings=settings
            )
            
            self.is_loaded = True
            logger.info(f"Loaded pretrained {self.model_name} model")
    
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
        Fine-tune the model using the official FairChem CLI pipeline.
        
        Wraps the official workflow:
        1. Convert training data to extxyz format
        2. Create LMDB datasets + YAML config via fairchem scripts
        3. Run 'fairchem -c config.yaml' as subprocess
        4. Parse CLI metrics into standard training_history format
        
        Args:
            training_data: List of dicts with 'structure', 'energy', 'forces', 'stress'.
            validation_data: Optional list of dicts for validation.
            training_config: Configuration dict (max_epochs, learning_rate, batch_size, etc.).
            output_dir: Directory to save results.
            
        Returns:
            Dictionary with fine-tuning results (status, checkpoint_path, training_history).
        """
        import os
        import re
        import subprocess
        import json as json_module
        from pathlib import Path
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        from ase import Atoms
        from ase.io import write as ase_write
        import numpy as np
        
        config = training_config or {}
        epochs = config.get("epochs", config.get("max_epochs", 10))
        learning_rate = config.get("learning_rate", 4e-4)
        batch_size = config.get("batch_size", 2)
        task_name = config.get("task_name", "omat")
        base_model = config.get("base_model", self.model_name)
        
        # Advanced config parameters
        freeze_backbone = config.get("freeze_backbone", False)
        
        # reinit_head: FairChem always re-initializes output heads for new task names
        # (via initialize_finetuning_model). This key is accepted for API consistency 
        # with MACE/MatGL but has no functional effect — heads are always fresh.
        reinit_head = config.get("reinit_head", False)
        if reinit_head:
            logging.info("reinit_head=True: FairChem always re-initializes heads for new tasks (no-op)")
        else:
            logging.info("reinit_head=False: Note — FairChem always re-initializes heads for new task names")
        weight_decay = config.get("weight_decay", 1e-3)
        warmup_factor = config.get("warmup_factor", 0.2)
        warmup_epochs = config.get("warmup_epochs", 0.01)
        lr_min_factor = config.get("lr_min_factor", 0.01)
        clip_grad_norm = config.get("clip_grad_norm", 100)
        evaluate_every_n_steps = config.get("evaluate_every_n_steps", 100)
        checkpoint_every_n_steps = config.get("checkpoint_every_n_steps", 1000)
        ema_decay = config.get("ema_decay", 0.999)
        
        save_dir = Path(output_dir if output_dir else "./fairchem_finetune")
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # ----- Step 1: Convert data to extxyz -----
        logging.info("Step 1: Converting training data to extxyz format...")
        
        def _to_atoms(data_dict: Dict[str, Any]) -> Atoms:
            """Convert a data dict to ASE Atoms with info/arrays labels."""
            s_obj = data_dict["structure"]
            if isinstance(s_obj, dict):
                struct = Structure.from_dict(s_obj)
                atoms = AseAtomsAdaptor.get_atoms(struct)
            elif isinstance(s_obj, Atoms):
                atoms = s_obj.copy()
            else:
                atoms = AseAtomsAdaptor.get_atoms(s_obj)
            
            # Attach labels as info/arrays (extxyz format)
            if "energy" in data_dict and data_dict["energy"] is not None:
                atoms.info["energy"] = float(data_dict["energy"])
            if "forces" in data_dict and data_dict["forces"] is not None:
                atoms.arrays["forces"] = np.array(data_dict["forces"], dtype=np.float64)
            if "stress" in data_dict and data_dict["stress"] is not None:
                stress = np.array(data_dict["stress"], dtype=np.float64)
                # Ensure 3x3 format
                if stress.shape == (6,):
                    # Voigt: xx, yy, zz, yz, xz, xy → 3x3
                    v = stress.flatten()
                    stress = np.array([
                        [v[0], v[5], v[4]],
                        [v[5], v[1], v[3]],
                        [v[4], v[3], v[2]]
                    ])
                elif stress.size == 9:
                    stress = stress.reshape(3, 3)
                atoms.info["stress"] = stress
            return atoms
        
        # Split into train/val
        all_data = list(training_data)
        if validation_data:
            val_data = list(validation_data)
            train_data = all_data
        else:
            split_idx = max(1, int(0.9 * len(all_data)))
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            if not val_data:
                val_data = train_data[-1:]

        # Write extxyz files
        train_dir = save_dir / "extxyz" / "train"
        val_dir = save_dir / "extxyz" / "val"
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)
        
        for i, d in enumerate(train_data):
            atoms = _to_atoms(d)
            ase_write(str(train_dir / f"train_{i:04d}.extxyz"), atoms, format="extxyz")
        
        for i, d in enumerate(val_data):
            atoms = _to_atoms(d)
            ase_write(str(val_dir / f"val_{i:04d}.extxyz"), atoms, format="extxyz")
        
        logging.info(f"  Wrote {len(train_data)} train + {len(val_data)} val extxyz files")
        
        # ----- Step 2: Create LMDB + YAML via fairchem scripts -----
        logging.info("Step 2: Creating LMDB datasets and YAML config...")
        
        
        
        lmdb_dir = save_dir / "lmdb_output"
        # Remove existing lmdb_dir if it exists (create_yaml asserts it doesn't)
        import shutil
        if lmdb_dir.exists():
            shutil.rmtree(lmdb_dir)
        
        # Determine regression task type
        has_stress = any(d.get("stress") is not None for d in train_data)
        has_forces = any(d.get("forces") is not None for d in train_data)
        if has_stress:
            regression_tasks = "efs"
        elif has_forces:
            regression_tasks = "ef"
        else:
            regression_tasks = "e"
        
        # Process train and val data into LMDB
        # NOTE: We use sequential processing (no mp.Pool) to avoid CUDA fork deadlock
        # when the model is already loaded on GPU (e.g., in MCP server context).
        # The upstream launch_processing/compute_normalizer use mp.Pool which forks,
        # and forked children inherit the CUDA context causing deadlocks.
        train_lmdb_path = lmdb_dir / "train"
        val_lmdb_path = lmdb_dir / "val"
        
        logging.info("  Creating train LMDB (sequential)...")
        self._create_lmdb_sequential(train_dir, train_lmdb_path)
        
        logging.info("  Computing normalizer and linear reference...")
        if "linref_coeff" in training_config:
            force_rms = 1.0
            linref_coeff = training_config["linref_coeff"]
            logging.info("  Using provided linref_coeff from config.")
        else:
            force_rms, linref_coeff = self._compute_normalizer_sequential(train_lmdb_path)
        
        if regression_tasks == "e":
            force_rms = 1.0
        
        logging.info("  Creating val LMDB (sequential)...")
        self._create_lmdb_sequential(val_dir, val_lmdb_path)
        
        # Generate YAML config using bundled templates
        configs_dir = Path(__file__).parent / "finetune_configs"
        self._create_finetune_yaml(
            configs_dir=configs_dir,
            train_lmdb_path=str(train_lmdb_path),
            val_lmdb_path=str(val_lmdb_path),
            force_rms=float(force_rms),
            linref_coeff=linref_coeff,
            output_dir=lmdb_dir,
            dataset_name=task_name,
            regression_tasks=regression_tasks,
            base_model_name=base_model,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            freeze_backbone=freeze_backbone,
            weight_decay=weight_decay,
            warmup_factor=warmup_factor,
            warmup_epochs=warmup_epochs,
            lr_min_factor=lr_min_factor,
            clip_grad_norm=clip_grad_norm,
            evaluate_every_n_steps=evaluate_every_n_steps,
            checkpoint_every_n_steps=checkpoint_every_n_steps,
            ema_decay=ema_decay,
        )
        
        yaml_config_path = lmdb_dir / "uma_sm_finetune_template.yaml"
        logging.info(f"  YAML config: {yaml_config_path}")
        
        # ----- Step 3: Run fairchem CLI -----
        logging.info(f"Step 3: Running fairchem CLI for {epochs} epochs...")
        
        run_dir = save_dir / "runs"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp_id = f"run_{epochs}ep"
        
        # Clean any prior checkpoint to prevent the CLI from resuming instead of training fresh
        prior_run = run_dir / timestamp_id
        if prior_run.exists():
            import shutil as _shutil
            _shutil.rmtree(prior_run)
            logging.info(f"  Removed existing run directory: {prior_run}")
        
        # Resolve the fairchem binary from the same conda env as the current Python
        import sys as _sys
        fairchem_bin = Path(_sys.executable).parent / "fairchem"
        if not fairchem_bin.exists():
            raise FileNotFoundError(
                f"fairchem CLI not found at {fairchem_bin}. "
                f"Ensure fairchem-core is installed in the active environment."
            )
        
        cmd = [
            str(fairchem_bin),
            "-c", str(yaml_config_path),
            f"job.run_dir={run_dir}",
            f"+job.timestamp_id={timestamp_id}",
        ]
        
        logging.info(f"  Command: {' '.join(cmd)}")
        
        # Build subprocess environment
        # When freeze_backbone is used, the helper module is in lmdb_dir and
        # must be importable by Hydra during instantiation.
        import os as _os
        sub_env = _os.environ.copy()
        existing_pypath = sub_env.get("PYTHONPATH", "")
        lmdb_dir_str = str(lmdb_dir.resolve())
        if existing_pypath:
            sub_env["PYTHONPATH"] = lmdb_dir_str + _os.pathsep + existing_pypath
        else:
            sub_env["PYTHONPATH"] = lmdb_dir_str
        
        # Run subprocess and capture output
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(save_dir),
            env=sub_env,
        )
        
        cli_output = process.stdout + "\n" + process.stderr
        
        # Save raw output log
        log_path = save_dir / "fairchem_cli_output.log"
        with open(log_path, "w") as f:
            f.write(cli_output)
        
        if process.returncode != 0:
            logging.error(f"fairchem CLI failed with return code {process.returncode}")
            logging.error(f"stderr: {process.stderr[-2000:]}")
            return {
                "status": "failed",
                "error": f"fairchem CLI exited with code {process.returncode}",
                "log_path": str(log_path),
            }
        
        # ----- Step 4: Parse CLI metrics -----
        logging.info("Step 4: Parsing training metrics from CLI output...")
        
        training_history = self._parse_fairchem_cli_metrics(cli_output, task_name)
        
        # Also populate label distributions for plotting
        distributions = self._collect_label_distributions(training_data)
        training_history.update(distributions)
        self._training_history = training_history
        self.is_fine_tuned = True
        
        # ----- Step 5: Save artifacts -----
        logging.info("Step 5: Saving training history and plot...")
        
        # Save training history JSON
        json_path = save_dir / "training_history.json"
        self.save_training_history(str(json_path))
        
        # Plot training history
        plot_path = save_dir / "training_history.png"
        self.plot_training_history(save_path=str(plot_path), show=False)
        logging.info(f"  Training history plot saved to {plot_path}")
        
        # Find inference checkpoint
        ckpt_dir = run_dir / timestamp_id / "checkpoints" / "final"
        inference_ckpt_path = ckpt_dir / "inference_ckpt.pt"
        
        if not inference_ckpt_path.exists():
            logging.warning(f"inference_ckpt.pt not found at {inference_ckpt_path}")
            # Try alternative locations
            for ckpt_candidate in run_dir.rglob("inference_ckpt.pt"):
                inference_ckpt_path = ckpt_candidate
                break
        
        # Extract final metrics
        final_metrics = {}
        for key in ["energy_mae_val", "force_mae_val", "stress_mae_val", "loss_val"]:
            values = [v for v in training_history.get(key, []) if v is not None]
            if values:
                final_metrics[f"final_{key}"] = values[-1]
        
        return {
            "status": "completed",
            "epochs": epochs,
            "model_saved_to": str(inference_ckpt_path) if inference_ckpt_path.exists() else None,
            "training_history_path": str(json_path),
            "training_history_plot": str(plot_path),
            "log_path": str(log_path),
            "final_metrics": final_metrics,
            "training_history": training_history,
        }
    
    def _create_finetune_yaml(
        self,
        configs_dir: "Path",
        train_lmdb_path: str,
        val_lmdb_path: str,
        force_rms: float,
        linref_coeff: list,
        output_dir: "Path",
        dataset_name: str,
        regression_tasks: str,
        base_model_name: str,
        epochs: int = 10,
        learning_rate: float = 4e-4,
        batch_size: int = 2,
        freeze_backbone: bool = False,
        weight_decay: float = 1e-3,
        warmup_factor: float = 0.2,
        warmup_epochs: float = 0.01,
        lr_min_factor: float = 0.01,
        clip_grad_norm: float = 100,
        evaluate_every_n_steps: int = 100,
        checkpoint_every_n_steps: int = 1000,
        ema_decay: float = 0.999,
    ) -> None:
        """Generate finetune YAML configs from bundled templates.
        
        Args:
            configs_dir: Path to bundled template config directory.
            train_lmdb_path: Path to train LMDB dataset.
            val_lmdb_path: Path to val LMDB dataset.
            force_rms: RMS of forces for normalizer.
            linref_coeff: Linear reference coefficients.
            output_dir: Output directory for generated YAML files.
            dataset_name: Name of the dataset (used as task key).
            regression_tasks: One of 'e', 'ef', 'efs'.
            base_model_name: Name of the base UMA model checkpoint.
            epochs: Number of training epochs.
            learning_rate: Peak learning rate for AdamW optimizer.
            batch_size: Training batch size.
            freeze_backbone: If True, freeze backbone params (only train heads).
            weight_decay: Weight decay for AdamW optimizer.
            warmup_factor: Initial LR = warmup_factor * lr during warmup.
            warmup_epochs: Fraction of total epochs used for LR warmup.
            lr_min_factor: Minimum LR = lr_min_factor * lr at end of cosine decay.
            clip_grad_norm: Max gradient norm for clipping.
            evaluate_every_n_steps: Run validation every N training steps.
            checkpoint_every_n_steps: Save checkpoint every N training steps.
            ema_decay: Exponential moving average decay rate.
        """
        import yaml
        from pathlib import Path
        
        data_yaml_dir = Path("data")
        regression_label_to_yaml = {
            "e": data_yaml_dir / "uma_conserving_data_task_energy.yaml",
            "ef": data_yaml_dir / "uma_conserving_data_task_energy_force.yaml",
            "efs": data_yaml_dir / "uma_conserving_data_task_energy_force_stress.yaml",
        }
        
        # Load and modify data task YAML
        data_task_yaml = configs_dir / regression_label_to_yaml[regression_tasks]
        with open(data_task_yaml) as f:
            template = yaml.safe_load(f)
        
        template["dataset_name"] = dataset_name
        template["normalizer_rmsd"] = force_rms
        template["elem_refs"] = linref_coeff
        template["train_dataset"]["splits"]["train"]["src"] = train_lmdb_path
        template["val_dataset"]["splits"]["val"]["src"] = val_lmdb_path
        
        output_dir = Path(output_dir)
        (output_dir / data_yaml_dir).mkdir(parents=True, exist_ok=True)
        with open(output_dir / regression_label_to_yaml[regression_tasks], "w") as f:
            yaml.dump(template, f, default_flow_style=False, sort_keys=False)
        
        # Load and modify main finetune YAML
        uma_yaml = configs_dir / "uma_sm_finetune_template.yaml"
        with open(uma_yaml) as f:
            template_ft = yaml.safe_load(f)
        
        template_ft["base_model_name"] = base_model_name
        template_ft["epochs"] = epochs
        template_ft["lr"] = learning_rate
        template_ft["batch_size"] = batch_size
        template_ft["weight_decay"] = weight_decay
        template_ft["evaluate_every_n_steps"] = evaluate_every_n_steps
        template_ft["checkpoint_every_n_steps"] = checkpoint_every_n_steps
        
        # Update cosine LR scheduler parameters
        scheduler_cfg = template_ft["runner"]["train_eval_unit"]["cosine_lr_scheduler_fn"]
        scheduler_cfg["warmup_factor"] = warmup_factor
        scheduler_cfg["warmup_epochs"] = warmup_epochs
        scheduler_cfg["lr_min_factor"] = lr_min_factor
        
        # Update training unit parameters
        template_ft["runner"]["train_eval_unit"]["clip_grad_norm"] = clip_grad_norm
        template_ft["runner"]["train_eval_unit"]["ema_decay"] = ema_decay
        
        # Freeze backbone: create a helper module that Hydra can instantiate
        if freeze_backbone:
            # Create a helper Python module next to the YAML config that Hydra can call.
            # This module wraps initialize_finetuning_model and freezes backbone params.
            helper_path = output_dir / "_freeze_backbone_helper.py"
            helper_code = (
                "import torch\n"
                "from fairchem.core.units.mlip_unit.mlip_unit import initialize_finetuning_model\n"
                "\n"
                "def initialize_finetuning_model_frozen(\n"
                "    checkpoint_location, overrides=None, heads=None, strict=True\n"
                "):\n"
                "    model = initialize_finetuning_model(\n"
                "        checkpoint_location=checkpoint_location,\n"
                "        overrides=overrides,\n"
                "        heads=heads,\n"
                "        strict=strict,\n"
                "    )\n"
                "    for param in model.backbone.parameters():\n"
                "        param.requires_grad = False\n"
                "    return model\n"
            )
            with open(helper_path, "w") as f:
                f.write(helper_code)
            
            # Point the YAML _target_ to our helper function
            model_cfg = template_ft["runner"]["train_eval_unit"]["model"]
            model_cfg["_target_"] = "_freeze_backbone_helper.initialize_finetuning_model_frozen"
            logging.info("  freeze_backbone=True: backbone params will be frozen (only heads train)")
        
        template_ft["defaults"][0]["data"] = regression_label_to_yaml[regression_tasks].stem
        template_ft["train_dataset"]["dataset_configs"][dataset_name] = template_ft[
            "train_dataset"
        ]["dataset_configs"].pop("DATASET_NAME")
        template_ft["val_dataset"]["dataset_configs"][dataset_name] = template_ft[
            "val_dataset"
        ]["dataset_configs"].pop("DATASET_NAME")
        
        with open(output_dir / "uma_sm_finetune_template.yaml", "w") as f:
            yaml.dump(template_ft, f, default_flow_style=False, sort_keys=False)
    
    def _create_lmdb_sequential(self, data_dir: "Path", output_dir: "Path") -> None:
        """Create LMDB dataset sequentially (no mp.Pool) to avoid CUDA fork deadlock.
        
        Functionally identical to fairchem's launch_processing but runs in a
        single process, safe when CUDA is already initialized.
        
        Args:
            data_dir: Directory containing extxyz files.
            output_dir: Output LMDB directory.
        """
        import glob
        import os
        import numpy as np
        from ase.db import connect
        from ase.io import read
        from tqdm import tqdm
        
        os.makedirs(output_dir, exist_ok=True)
        input_files = [
            f for f in glob.glob(os.path.join(str(data_dir), "**/*"), recursive=True)
            if os.path.isfile(f)
        ]
        
        db_file = output_dir / "data.0000.aselmdb"
        natoms = []
        successful = []
        failed = []
        
        with connect(str(db_file)) as db:
            for file in tqdm(input_files):
                atoms_list = read(file, ":")
                for i, atoms in enumerate(atoms_list):
                    if atoms.calc is not None and "energy" in atoms.calc.results and "forces" in atoms.calc.results:
                        db.write(atoms, data=atoms.info)
                        natoms.append(len(atoms))
                        successful.append(f"{file},{i}")
                    else:
                        failed.append(f"{file},{i}: missing calc/energy/forces")
        
        # Save metadata
        np.savez_compressed(output_dir / "metadata.npz", natoms=natoms)
        logging.info(f"    Created LMDB with {len(successful)} entries at {output_dir}")
    
    def _compute_normalizer_sequential(self, train_lmdb_path: "Path") -> tuple:
        """Compute normalizer and linear reference sequentially (no mp.Pool).
        
        Functionally identical to fairchem's compute_normalizer_and_linear_reference
        but runs in a single process, safe when CUDA is already initialized.
        
        Args:
            train_lmdb_path: Path to training LMDB directory.
            
        Returns:
            Tuple of (force_rms, linref_coefficients).
        """
        import random
        import numpy as np
        from fairchem.core.datasets import AseDBDataset
        from tqdm import tqdm
        from fairchem.core.scripts.create_finetune_dataset import compute_lin_ref
        
        dataset = AseDBDataset({"src": str(train_lmdb_path)})
        sample_indices = random.sample(range(len(dataset)), min(100000, len(dataset)))
        
        atomic_numbers_list = []
        energies = []
        all_forces = []
        
        for idx in tqdm(sample_indices, desc="Computing normalizer values."):
            atoms = dataset.get_atoms(idx)
            atomic_numbers_list.append(atoms.get_atomic_numbers())
            energies.append(atoms.get_potential_energy())
            forces = atoms.get_forces()
            # Mask fixed atoms
            n_atoms = len(atoms)
            fixed_idx = np.zeros(n_atoms)
            if hasattr(atoms, "constraints"):
                from ase.constraints import FixAtoms
                for constraint in atoms.constraints:
                    if isinstance(constraint, FixAtoms):
                        fixed_idx[constraint.index] = 1
            mask = fixed_idx == 0
            all_forces.extend(forces[mask].tolist())
        
        forces_arr = np.array(all_forces)
        force_rms = np.sqrt(np.mean(np.square(forces_arr)))
        coeff = compute_lin_ref(atomic_numbers_list, energies)
        
        logging.info(f"    force_rms={force_rms:.4f}, linref has {len(coeff)} coeffs")
        return force_rms, coeff
    
    def _parse_fairchem_cli_metrics(
        self, cli_output: str, task_name: str
    ) -> Dict[str, Any]:
        """
        Parse fairchem CLI stdout/stderr to extract training metrics.
        
        CLI output formats:
        - Training: INFO:root:{'train/loss': 4.82, 'train/lr': 2e-05, 'train/step': 0, 'train/epoch': 0.0, ...}
        - Validation (aggregated block):
            val/loss: 4.7798
            val/{task_name}.val,energy,per_atom_mae: 0.2756
            val/{task_name}.val,forces,mae: 0.0469
            val/{task_name}.val,stress,mae: 0.0090
        
        Maps to standard training_history keys:
        - val/{task}.val,energy,per_atom_mae → energy_mae_val
        - val/{task}.val,forces,mae → force_mae_val
        - val/{task}.val,stress,mae → stress_mae_val
        - train/loss → loss_train
        
        Returns:
            Dictionary with standard training_history keys.
        """
        import re
        import ast
        
        history: Dict[str, list] = {
            "epoch": [],
            "loss_train": [],
            "loss_val": [],
            "energy_mae_train": [],
            "energy_mae_val": [],
            "force_mae_train": [],
            "force_mae_val": [],
            "stress_mae_train": [],
            "stress_mae_val": [],
        }
        
        # Collect train loss per epoch (last value in epoch wins) and val metrics per epoch
        current_train_loss = None
        current_val_metrics: Dict[str, float] = {}
        epoch_count = 0
        
        for line in cli_output.split("\n"):
            line = line.strip()
            
            # Parse training metrics from JSON dict format:
            # INFO:root:{'train/loss': 4.82, 'train/epoch': 0.0, ...}
            train_dict_match = re.search(r"INFO:root:(\{'train/loss':.+\})", line)
            if train_dict_match:
                dict_str = train_dict_match.group(1)
                try:
                    metrics = ast.literal_eval(dict_str)
                    current_train_loss = metrics.get("train/loss")
                except (ValueError, SyntaxError):
                    pass
                continue
            
            # Parse validation metrics from aggregated block (colon-separated):
            # val/loss: 4.7798
            val_loss_match = re.search(r"val/loss:\s*([\d.eE+-]+)", line)
            if val_loss_match:
                current_val_metrics["loss_val"] = float(val_loss_match.group(1))
            
            # val/{task}.val,energy,per_atom_mae: 0.2756  (in eV/atom)
            energy_match = re.search(
                rf"val/{re.escape(task_name)}\.val,energy,per_atom_mae:\s*([\d.eE+-]+)", line
            )
            if energy_match:
                current_val_metrics["energy_mae_val"] = float(energy_match.group(1)) * 1000  # eV/atom → meV/atom
            
            # val/{task}.val,forces,mae: 0.0469  (in eV/Å)
            forces_match = re.search(
                rf"val/{re.escape(task_name)}\.val,forces,mae:\s*([\d.eE+-]+)", line
            )
            if forces_match:
                current_val_metrics["force_mae_val"] = float(forces_match.group(1)) * 1000  # eV/Å → meV/Å
            
            # val/{task}.val,stress,mae: 0.0090  (in eV/Å³)
            stress_match = re.search(
                rf"val/{re.escape(task_name)}\.val,stress,mae:\s*([\d.eE+-]+)", line
            )
            if stress_match:
                current_val_metrics["stress_mae_val"] = float(stress_match.group(1)) * 1000  # eV/Å³ → meV/Å³
            
            # Epoch boundary: after validation, the trainer logs "Ended train epoch"
            if "Ended train epoch" in line and current_val_metrics:
                history["epoch"].append(epoch_count)
                history["loss_train"].append(current_train_loss)
                for key in history:
                    if key in ("epoch", "loss_train"):
                        continue
                    history[key].append(current_val_metrics.get(key, None))
                epoch_count += 1
                current_train_loss = None
                current_val_metrics = {}
        
        # Flush any remaining metrics
        if current_val_metrics:
            history["epoch"].append(epoch_count)
            history["loss_train"].append(current_train_loss)
            for key in history:
                if key in ("epoch", "loss_train"):
                    continue
                history[key].append(current_val_metrics.get(key, None))
        
        logging.info(f"  Parsed {len(history['epoch'])} epochs of metrics from CLI output")
        
        return history
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
        """Load state dict with automatic key prefix and head name remapping.
        
        Handles two common mismatches:
        1. module. prefix: MLIPInferenceCheckpoint uses 'backbone.X' but 
           AveragedModel expects 'module.backbone.X'
        2. Head names: Fine-tuning creates 'shared_efs_head' but base model 
           has 'energyandforcehead.head' or similar
        
        Args:
            state_dict: The state dictionary to load.
        """
        target_model = self.model
        if not hasattr(target_model, "load_state_dict") and hasattr(target_model, "model"):
            target_model = target_model.model
        
        if not hasattr(target_model, "load_state_dict"):
            logging.warning("Model does not support load_state_dict. Skipping state load.")
            return
        
        model_keys = set(target_model.state_dict().keys())
        ckpt_keys = set(state_dict.keys())
        
        # Phase 1: Handle module. prefix mismatch
        model_has_module = any(k.startswith("module.") for k in model_keys)
        ckpt_has_module = any(k.startswith("module.") for k in ckpt_keys)
        
        remapped = dict(state_dict)
        if model_has_module and not ckpt_has_module:
            logging.info("Remapping checkpoint keys: adding 'module.' prefix")
            remapped = {"module." + k: v for k, v in state_dict.items()}
        elif not model_has_module and ckpt_has_module:
            logging.info("Remapping checkpoint keys: stripping 'module.' prefix")
            remapped = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        # Phase 2: Handle output head name mismatch
        # Extract head names from model and checkpoint keys
        model_head_names = set()
        ckpt_head_names = set()
        for k in model_keys:
            if "output_heads." in k:
                # e.g. "module.output_heads.energyandforcehead.head.energy_block.0.weight"
                parts = k.split("output_heads.")[1].split(".")
                model_head_names.add(parts[0])
        for k in remapped.keys():
            if "output_heads." in k:
                parts = k.split("output_heads.")[1].split(".")
                ckpt_head_names.add(parts[0])
        
        if model_head_names and ckpt_head_names and model_head_names != ckpt_head_names:
            # Find weight-name suffix patterns to match heads by structure
            # Simple approach: if there is exactly 1 head on each side, remap directly
            if len(model_head_names) == 1 and len(ckpt_head_names) == 1:
                model_head = next(iter(model_head_names))
                ckpt_head = next(iter(ckpt_head_names))
                
                # Check if the ckpt head weights have a sub-prefix that matches
                # Base model: output_heads.energyandforcehead.head.energy_block.0.weight
                # Checkpoint:  output_heads.shared_efs_head.energy_block.0.weight
                # Need to find what aligns: try mapping ckpt_head -> model_head (with or without .head sub-level)
                ckpt_head_keys = {k for k in remapped if "output_heads." + ckpt_head in k}
                
                # Extract suffixes after the head name
                ckpt_suffixes = set()
                for k in ckpt_head_keys:
                    suffix = k.split("output_heads." + ckpt_head + ".")[1]
                    ckpt_suffixes.add(suffix)
                
                # Find model head suffixes
                model_head_keys = {k for k in model_keys if "output_heads." + model_head in k}
                model_suffixes = set()
                model_head_prefix = None
                for k in model_head_keys:
                    suffix = k.split("output_heads." + model_head + ".")[1]
                    model_suffixes.add(suffix)
                    if model_head_prefix is None:
                        # Detect if model has an extra sub-prefix (e.g., "head.")
                        model_head_prefix = "output_heads." + model_head + "."
                
                # If ckpt suffix "energy_block.0.weight" matches model suffix "head.energy_block.0.weight"
                # then model has an extra "head." sub-prefix
                extra_prefix = ""
                if ckpt_suffixes and model_suffixes and ckpt_suffixes != model_suffixes:
                    # Check if model suffixes all start with a common sub-prefix
                    sample_ckpt = sorted(ckpt_suffixes)[0]
                    for ms in model_suffixes:
                        if ms.endswith(sample_ckpt) and len(ms) > len(sample_ckpt):
                            extra_prefix = ms[: len(ms) - len(sample_ckpt)]
                            break
                
                logging.info("Remapping head keys: '%s' -> '%s' (extra_prefix='%s')", 
                           ckpt_head, model_head, extra_prefix)
                new_remapped = {}
                for k, v in remapped.items():
                    if "output_heads." + ckpt_head + "." in k:
                        old_prefix = "output_heads." + ckpt_head + "."
                        new_prefix = "output_heads." + model_head + "." + extra_prefix
                        new_k = k.replace(old_prefix, new_prefix, 1)
                        new_remapped[new_k] = v
                    else:
                        new_remapped[k] = v
                remapped = new_remapped
        
        msg = target_model.load_state_dict(remapped, strict=False)
        if msg.missing_keys or msg.unexpected_keys:
            logging.warning("State dict load: %d missing, %d unexpected keys", 
                          len(msg.missing_keys), len(msg.unexpected_keys))
            if len(msg.missing_keys) <= 10:
                logging.warning("Missing keys: %s", msg.missing_keys)
            if len(msg.unexpected_keys) <= 10:
                logging.warning("Unexpected keys: %s", msg.unexpected_keys)
        else:
            logging.info("State dict loaded successfully with 0 missing, 0 unexpected keys")

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
        


    def predict_atomic_features(self, structure_data: Any) -> Dict[str, Any]:
        """
        Predict atomic latent features (descriptors) for a structure.
        
        Args:
            structure_data: Structure data.
            
        Returns:
            Dictionary containing 'atomic_features'.
        """
        atoms = self.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        calc = self.create_calculator()
        atoms.calc = calc
        
        # FairChem models usually don't have a direct latent feature API in the calculator
        # For now, return a placeholder or try to extract from the model if supported
        # Placeholder for consistency with base class
        return {
            "atomic_features": [],
            "note": "Atomic feature extraction not yet implemented for FAIRCHEMWrapper"
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
