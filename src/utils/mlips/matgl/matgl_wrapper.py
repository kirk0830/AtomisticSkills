"""
MatGL model wrapper for MLIP Agent
"""

import logging
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
import torch
import torch.serialization
from ase import Atoms
from ase.calculators.calculator import Calculator
import matplotlib
matplotlib.use('Agg')  # Must be set before importing pyplot
import matplotlib.pyplot as plt

# MATGL_BACKEND is now set at the entry point (matgl_server.py)
# We just ensure it's set here for direct library usage
if "MATGL_BACKEND" not in os.environ:
    os.environ["MATGL_BACKEND"] = "DGL"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from ..base import MLIPModel
from ..device_utils import get_best_device

logger = logging.getLogger(__name__)

# Verify backend
try:
    import matgl
    logger.info(f"MatGL initialized with backend: {matgl.config.BACKEND}")
except ImportError:
    pass

# Try to import MatGL components with clear error messages
MATGL_AVAILABLE = False
TRAINER_AVAILABLE = False

try:
    import matgl
    from matgl.ext.ase import PESCalculator
    
    # Import models safely
    try:
        from matgl.models import M3GNet
    except ImportError:
        M3GNet = None
    
    try:
        from matgl.models import CHGNet
    except ImportError:
        CHGNet = None
        
    try:
        from matgl.models import TensorNet
    except ImportError:
        TensorNet = None
        
    MATGL_AVAILABLE = True
    
    # Import training components
    TRAINER_AVAILABLE = False
    try:
        from matgl.utils.training import PotentialLightningModule
        from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
        try:
            from matgl.graph.data import split_dataset
        except ImportError:
            from dgl.data.utils import split_dataset
        TRAINER_AVAILABLE = True
    except ImportError as e:
        logger.debug(f"Primary MatGL training imports failed: {e}")
        try:
             # MatGL < 1.0 or specific versions might have it here
             from matgl.utils._training_dgl import PotentialLightningModule
             from matgl.graph.data import MGLDataLoader, MGLDataset, collate_fn_pes
             from dgl.data.utils import split_dataset
             TRAINER_AVAILABLE = True
        except ImportError as e2:
             logger.debug(f"Secondary MatGL training imports failed: {e2}")
             pass
    
    from matgl.ext.pymatgen import Structure2Graph, get_element_list
    import lightning as pl
    if not TRAINER_AVAILABLE:
        # Check if lightning is available even if matgl training isn't
        try:
            import lightning as pl
        except ImportError:
            pass
    
    # Add AtomRef to safe globals for torch.load (PyTorch 2.6+ compatibility)
    try:
        from matgl.layers._atom_ref_dgl import AtomRef
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([AtomRef])
    except (ImportError, AttributeError):
        pass

except ImportError as e:
    import os
    import traceback
    current_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    # If we are in matgl-agent but import fails, it's a real issue
    if 'matgl' in current_env.lower() or 'matgl' in str(e).lower():
        logger.warning(f"Failed to import matgl in environment {current_env}: {e}")
    logger.error(f"Failed to import matgl: {e}\n{traceback.format_exc()}")
    MATGL_AVAILABLE = False


# Available MatGL models and checkpoints
AVAILABLE_MATGL_MODELS = {
    # M3GNet models - Materials Project trained
    "M3GNet": "M3GNet-MP-2021.2.8-PES",
    "M3GNet-MP": "M3GNet-MP-2021.2.8-PES",
    "M3GNet-MP-2021.2.8-PES": "M3GNet-MP-2021.2.8-PES",
    
    # M3GNet models - MatPES trained
    "M3GNet-MatPES-r2SCAN": "M3GNet-MatPES-r2SCAN-v2025.1-PES",
    "M3GNet-MatPES-PBE": "M3GNet-MatPES-PBE-v2025.1-PES",
    
    # CHGNet models - MatPES trained
    "CHGNet": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "CHGNet-MatPES-PBE": "CHGNet-MatPES-PBE-2025.2.10-2.7M-PES",
    "CHGNet-MatPES-r2SCAN": "CHGNet-MatPES-r2SCAN-2025.2.10-2.7M-PES",
    
    # TensorNet models - MatPES trained
    "TensorNet": "TensorNet-MatPES-r2SCAN-v2025.1-PES",
}

# Update TensorNet models to DGL versions if backend is DGL
if os.environ.get("MATGL_BACKEND") == "DGL" and MATGL_AVAILABLE:
    updated_models = {}
    for k, v in AVAILABLE_MATGL_MODELS.items():
        if "TensorNet" in k and not v.startswith("TensorNetDGL") and "TensorNet-MatPES" in v:
            updated_models[k] = v.replace("TensorNet-", "TensorNetDGL-")
    AVAILABLE_MATGL_MODELS.update(updated_models)

# Standard MatGL model types
STANDARD_MATGL_MODELS = ["M3GNet", "CHGNet", "TensorNet"]


if MATGL_AVAILABLE and TRAINER_AVAILABLE:
    class TrainingHistoryCallback(pl.Callback):
        """Callback to collect training history during fine-tuning."""
        
        def __init__(self):
            super().__init__()
            self.training_history = {
                'energy_distribution': [],
                'force_distribution': [],
                'stress_distribution': [],
                'energy_mae_train': [],
                'energy_mae_val': [],
                'force_mae_train': [],
                'force_mae_val': [],
                'stress_mae_train': [],
                'stress_mae_val': [],
                'loss_train': [],
                'loss_val': []
            }
        
        def on_train_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking: return
            metrics = trainer.callback_metrics
            
            def get_metric(specific_keys, fallback_keys):
                for key in specific_keys:
                    if key in metrics:
                        val = metrics[key]
                        return float(val.item()) if hasattr(val, 'item') else float(val)
                for key in fallback_keys:
                    if key in metrics:
                        val = metrics[key]
                        return float(val.item()) if hasattr(val, 'item') else float(val)
                return None
            
            loss = get_metric(['train_Total_Loss', 'train_loss'], ['loss'])
            if loss is not None: self.training_history['loss_train'].append(loss)
            
            # MatGL reports Energy_MAE in eV/atom, Force_MAE in eV/Å, Stress_MAE in GPa.
            # Convert to standardized units: meV/atom, meV/Å, meV/Å³.
            e_mae = get_metric(['train_Energy_MAE', 'train_energy_mae'], ['energy_mae'])
            if e_mae is not None: self.training_history['energy_mae_train'].append(e_mae * 1000)  # eV/atom → meV/atom
            
            f_mae = get_metric(['train_Force_MAE', 'train_force_mae'], ['force_mae'])
            if f_mae is not None: self.training_history['force_mae_train'].append(f_mae * 1000)  # eV/Å → meV/Å
            
            s_mae = get_metric(['train_Stress_MAE', 'train_stress_mae'], ['stress_mae'])
            if s_mae is not None:
                import ase.units
                s_mae_ev_per_ang3 = s_mae * ase.units.GPa  # GPa → eV/Å³
                self.training_history['stress_mae_train'].append(s_mae_ev_per_ang3 * 1000)  # eV/Å³ → meV/Å³

        def on_validation_epoch_end(self, trainer, pl_module):
            if trainer.sanity_checking: return
            metrics = trainer.callback_metrics
            
            def get_metric(keys):
                for key in keys:
                    if key in metrics:
                        val = metrics[key]
                        return float(val.item()) if hasattr(val, 'item') else float(val)
                return None
            
            loss = get_metric(['val_Total_Loss', 'val_loss', 'loss'])
            if loss is not None: self.training_history['loss_val'].append(loss)
            
            # MatGL reports Energy_MAE in eV/atom, Force_MAE in eV/Å, Stress_MAE in GPa.
            # Convert to standardized units: meV/atom, meV/Å, meV/Å³.
            e_mae = get_metric(['val_Energy_MAE', 'val_energy_mae', 'energy_mae'])
            if e_mae is not None: self.training_history['energy_mae_val'].append(e_mae * 1000)  # eV/atom → meV/atom
            
            f_mae = get_metric(['val_Force_MAE', 'val_force_mae', 'force_mae'])
            if f_mae is not None: self.training_history['force_mae_val'].append(f_mae * 1000)  # eV/Å → meV/Å
            
            s_mae = get_metric(['val_Stress_MAE', 'val_stress_mae', 'stress_mae'])
            if s_mae is not None:
                import ase.units
                s_mae_ev_per_ang3 = s_mae * ase.units.GPa  # GPa → eV/Å³
                self.training_history['stress_mae_val'].append(s_mae_ev_per_ang3 * 1000)  # eV/Å³ → meV/Å³

        def collect_label_distributions(self, training_data: List[Dict[str, Any]]):
             # Placeholder for distribution collection (implemented in base or locally)
             from ..base import MLIPModel
             # We can't easily call base._collect_label_distributions here without an instance
             # but we can implement a simple version or just skip it if it's less critical
             pass

    def move_to_device(obj, device):
        """Recursively move tensors/graphs in nested structures to device."""
        if hasattr(obj, 'to'):
            try:
                # Special handling for DGL graphs
                if hasattr(obj, 'device') and str(obj.device) == str(device):
                    return obj
                return obj.to(device)
            except Exception:
                return obj
        elif isinstance(obj, dict):
            return {k: move_to_device(v, device) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)([move_to_device(x, device) for x in obj])
        return obj

    def ensure_dgl_device(func):
        """Decorator to ensure DGL graphs and tensors are on the correct device."""
        from functools import wraps
        import inspect
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            device = getattr(self, 'device', None)
            if device is None:
                try:
                    params = list(self.parameters())
                    if params: device = params[0].device
                except Exception: pass
            if device is None: device = torch.device('cpu')
            
            # Prioritize device from input if self is CPU
            if str(device) == 'cpu':
                for obj in list(args) + list(kwargs.values()):
                    if hasattr(obj, 'device') and str(obj.device).startswith('cuda'):
                        device = obj.device
                        break
            
            # Aggressively move self to device
            if hasattr(self, 'to'):
                try:
                    self.to(device)
                    for val in self.__dict__.values():
                        if hasattr(val, 'to'): move_to_device(val, device)
                except Exception: pass
            
            new_args = [move_to_device(arg, device) for arg in args]
            new_kwargs = {k: move_to_device(v, device) for k, v in kwargs.items()}
            
            with torch.device(device):
                return func(self, *new_args, **new_kwargs)
        return wrapper

    def apply_matgl_patches():
        """Apply essential monkey-patches to MatGL for device consistency."""
        if os.environ.get("MATGL_BACKEND") == "DGL":
            # 1. Patch DGL Potential forward
            try:
                import matgl.apps._pes_dgl as pes_dgl
                if hasattr(pes_dgl, 'Potential') and not hasattr(pes_dgl.Potential.forward, "__wrapped__"):
                    pes_dgl.Potential.forward = ensure_dgl_device(pes_dgl.Potential.forward)
            except (ImportError, AttributeError): pass

            # 2. Patch DGL training modules
            for path in ['matgl.utils.training', 'matgl.utils._training_dgl']:
                try:
                    mod = __import__(path, fromlist=['PotentialLightningModule'])
                    if hasattr(mod, 'PotentialLightningModule'):
                        cls = getattr(mod, 'PotentialLightningModule')
                        if not hasattr(cls.forward, "__wrapped__"):
                            cls.forward = ensure_dgl_device(cls.forward)
                        if hasattr(cls, 'step') and not hasattr(cls.step, "__wrapped__"):
                            cls.step = ensure_dgl_device(cls.step)
                except (ImportError, AttributeError): continue
            
            # 3. Patch AtomRef
            try:
                from matgl.layers._atom_ref_dgl import AtomRef
                if not hasattr(AtomRef.forward, "__wrapped__"):
                    AtomRef.forward = ensure_dgl_device(AtomRef.forward)
            except (ImportError, AttributeError): pass
    
    apply_matgl_patches()


class MatGLWrapper(MLIPModel):
    """
    MatGL model wrapper implementing the MLIPModel interface.
    """
    
    def __init__(
        self, 
        model_name: str = "M3GNet", 
        model_version: str = "latest",
        device: str = "auto"
    ):
        """Initialize MatGL wrapper."""
        super().__init__(model_name, model_version)
        self.device = get_best_device(device)
        self.model = None
        self.is_loaded = False
        
    def _load_model_agnostic(self, path_or_name: str) -> Any:
        """
        Load a MatGL model while automatically patching metadata to match active backend.
        This resolves the common issue where a model saved with PyG metadata fails in a DGL environment.
        Specifically, it patches both model.json and model.pt (which contains init_args).
        """
        import json
        import torch
        from pathlib import Path
        from matgl.utils.io import _get_file_paths
        
        backend = str(os.environ.get("MATGL_BACKEND", "DGL")).upper()
        target_suffix = "_dgl" if backend == "DGL" else "_pyg"
        source_suffix = "_pyg" if backend == "DGL" else "_dgl"
        
        # Get file paths (handles local or pretrained)
        fpaths = _get_file_paths(Path(path_or_name))
        
        def patch_metadata(d):
            if isinstance(d, dict):
                # We need a list of keys to avoid "dict size changed during iteration" errors
                for k in list(d.keys()):
                    v = d[k]
                    if k == "@module" and isinstance(v, str):
                        # Patch module paths to match active backend
                        if source_suffix in v:
                            d[k] = v.replace(source_suffix, target_suffix)
                        # Also handle generic matgl.apps.pes if it needs specific backend
                        if "matgl.apps.pes" in v:
                             d[k] = f"matgl.apps._pes{target_suffix}"
                    elif isinstance(v, (dict, list)):
                        patch_metadata(v)
            elif isinstance(d, list):
                for item in d:
                    patch_metadata(item)
        
        # Load and patch model.json
        with open(fpaths["model.json"]) as f:
            model_data = json.load(f)
        patch_metadata(model_data)
        
        # Load and patch model.pt (init_args)
        map_location = torch.device("cpu") if not torch.cuda.is_available() else None
        init_args = torch.load(fpaths["model.pt"], map_location=map_location, weights_only=False)
        patch_metadata(init_args)
        
        # We'll create a temporary patched directory for loading
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            # Copy state.pt directly
            shutil.copy(fpaths["state.pt"], tmpdir_path / "state.pt")
            
            # Write patched files
            with open(tmpdir_path / "model.json", "w") as f:
                json.dump(model_data, f)
            torch.save(init_args, tmpdir_path / "model.pt")
            
            # Now load from the temporary patched directory using the class from model_data
            # We must import the CORRECT Potential class (DGL or PyG)
            modname = model_data["@module"]
            classname = model_data["@class"]
            mod = __import__(modname, fromlist=[classname])
            cls_ = getattr(mod, classname)
            
            model = cls_.load(tmpdir_path)
            
        return model
        
    def load(self, model_path: Optional[str] = None) -> None:
        """Load a MatGL model."""
        if not MATGL_AVAILABLE:
            raise ImportError("MatGL is not available in the current environment.")
            
        model_name_or_path = model_path or self.model_name
        
        # Check if it looks like a custom checkpoint (file ending in .pth/.pt)
        if Path(str(model_name_or_path)).is_file() and (str(model_name_or_path).endswith('.pth') or str(model_name_or_path).endswith('.pt')):
            try:
                # Try to load as checkpoint to get base model name
                checkpoint = torch.load(str(model_name_or_path), map_location='cpu')
                if isinstance(checkpoint, dict) and 'model_name' in checkpoint:
                    base_model_name = checkpoint['model_name']
                    logger.info(f"Detected checkpoint for base model: {base_model_name}")
                    
                    # Temporarily set model_name to base model for loading architecture
                    original_name = self.model_name
                    self.model_name = base_model_name
                    
                    # Load base model (recursive call with None path)
                    self.load(model_path=None)
                    
                    # Restore original name if needed, or keep base name?
                    # self.model_name = original_name 
                    
                    # Load weights from checkpoint
                    self.load_checkpoint(str(model_name_or_path))
                    return
            except Exception as e:
                logger.warning(f"Failed to load as checkpoint, falling back to standard load: {e}")

        # Case-insensitive lookup for standard models
        if model_path is None:
            for key, val in AVAILABLE_MATGL_MODELS.items():
                if self.model_name.upper() == key.upper():
                    model_name_or_path = val
                    break
        
        try:
            logger.info(f"Loading MatGL model {model_name_or_path} on {self.device}")
            # Use map_location for initial load if it's a path
            # Use our agnostic loader to ensure backend compatibility
            self.model = self._load_model_agnostic(str(model_name_or_path))
            
            self.model.to(self.device)
            self.is_loaded = True
            logger.info(f"Successfully loaded {self.model_name} on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load MatGL model: {e}")
            raise RuntimeError(f"Failed to load MatGL model: {e}")

    def create_calculator(self) -> Calculator:
        """Create an ASE calculator from the model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        # Ensure model is on the correct device (and patch if needed)
        self.model.to(self.device)
        
        # matgl 2.0.4 with DGL backend expects calc_charge on the Potential object
        if not hasattr(self.model, "calc_charge"):
            self.model.calc_charge = False
            
        return PESCalculator(potential=self.model, device=self.device, stress_unit="eV/A3")

    def static_calculation(self, structure_data: Any) -> Dict[str, Any]:
        """
        Run static calculation for a structure. 
        Automatically detects if model is a PES (Potential) or a property predictor.
        """
        if not self.is_loaded:
            return {"error": "Model not loaded. Please call load() first."}

        # Convert to ASE Atoms
        atoms = self.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms

        # Check if model has PES capabilities
        is_pes = False
        model_class_name = type(self.model).__name__
        logger.info(f"Model class: {model_class_name}, name: {self.model_name}")
        
        if "Potential" in model_class_name:
            is_pes = True
        
        # Double check with attributes that only POTENTIAL models have
        # Many property models in MatGL also have 'model' attribute, so that's not enough
        if is_pes:
            # Verify it actually supports forces/stresses
            if not hasattr(self.model, "calc_forces") and not hasattr(self.model, "model"):
                is_pes = False

        logger.info(f"Identified is_pes: {is_pes}")

        if is_pes:
            # Standard PES calculation using ASE calculator
            return super().static_calculation(atoms)
        else:
            logger.info("Proceeding with direct property prediction")
            # Direct property prediction (e.g. MEGNet Bandgap, M3GNet Eform)
            try:
                from pymatgen.io.ase import AseAtomsAdaptor
                structure = AseAtomsAdaptor.get_structure(atoms)
                
                # Check for predict_structure or call directly
                if hasattr(self.model, "predict_structure"):
                    prediction = self.model.predict_structure(structure)
                else:
                    # Manually convert to graph if predict_structure is missing
                    # This is a fallback for some MatGL model versions
                    from matgl.ext.pymatgen import Structure2Graph
                    # Get element list from self.model if possible
                    element_types = getattr(self.model, "element_types", None)
                    if element_types is None and hasattr(self.model, "model"):
                        element_types = getattr(self.model.model, "element_types", None)
                    
                    if element_types:
                        converter = Structure2Graph(element_types=element_types, cutoff=5.0)
                        graph, _, _ = converter.get_graph(structure)
                        prediction = self.model(graph)
                    else:
                        raise AttributeError("Model has no 'predict_structure' and element_types are unknown.")

                # Result handling
                if hasattr(prediction, "item"):
                    val = float(prediction.item())
                else:
                    val = float(prediction)
                
                # Map property name
                prop_name = "property_prediction"
                if "BandGap" in self.model_name or "band_gap" in self.model_name.lower():
                    prop_name = "bandgap"
                elif "Eform" in self.model_name or "formation" in self.model_name.lower():
                    prop_name = "formation_energy"
                
                return {prop_name: val, "unit": "eV" if "BandGap" in self.model_name or "Eform" in self.model_name else "unknown"}
            except Exception as e:
                import traceback
                logger.error(f"Property prediction failed: {e}\n{traceback.format_exc()}")
                return {"error": f"Property prediction failed: {str(e)}"}

    def predict_atomic_features(self, structure_data: Any) -> Dict[str, Any]:
        """
        Predict atomic latent features (descriptors) for a structure using MatGL.
        """
        if not self.is_loaded:
            return {"error": "Model not loaded. Please call load() first."}
            
        atoms = self.check_structure_data(structure_data)
        if isinstance(atoms, dict) and "error" in atoms:
            return atoms
            
        try:
            from pymatgen.io.ase import AseAtomsAdaptor
            import torch
            import dgl
            from matgl.ext.pymatgen import get_element_list, Structure2Graph
            
            struct = AseAtomsAdaptor.get_structure(atoms)
            
            # Determine which inner model we have (PES wrappers hide the actual model)
            potential = self.model
            if hasattr(potential, "model"):
                inner_model = potential.model
            else:
                inner_model = potential

            # Create graph
            elements = inner_model.element_types if hasattr(inner_model, "element_types") else get_element_list([struct])
            cutoff = getattr(inner_model, "cutoff", 5.0)
            converter = Structure2Graph(element_types=elements, cutoff=cutoff)
            graph_data = converter.get_graph(struct)
            if len(graph_data) == 3:
                 graph, state_attr, l_g = graph_data
            else:
                 graph, state_attr = graph_data
                 l_g = None
            
            # Move to device and add batch dim
            g_batch = dgl.batch([graph]).to(self.device)
            if l_g is not None and hasattr(l_g, "is_block"):
                l_g_batch = dgl.batch([l_g]).to(self.device)
            else:
                l_g_batch = None
                
            state_attr = torch.tensor(state_attr, dtype=torch.float32, device=self.device).unsqueeze(0) if state_attr is not None else None

            # Ensure ndata has 'pos' and edata has 'pbc_offshift' for CHGNet/M3GNet
            if "pos" not in g_batch.ndata:
                 g_batch.ndata["pos"] = torch.tensor(struct.cart_coords, dtype=torch.float32, device=self.device)
            
            # Handle PBC for distance calculations (DGL backend specific)
            if os.environ.get("MATGL_BACKEND") == "DGL":
                lattice = torch.tensor(struct.lattice.matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
                g_batch.edata["lattice"] = torch.repeat_interleave(lattice, g_batch.batch_num_edges(), dim=0)
                g_batch.edata["pbc_offshift"] = (g_batch.edata["pbc_offset"].unsqueeze(dim=-1) * g_batch.edata["lattice"]).sum(dim=1)
            
            # Run model with return_all_layer_output=True to get latent features
            with torch.no_grad():
                model_output = inner_model(g=g_batch, state_attr=state_attr, l_g=l_g_batch, return_all_layer_output=True)
            
            if not isinstance(model_output, dict):
                 return {"error": f"Model of type {type(inner_model)} did not return a dict when return_all_layer_output=True."}

            # Extract site features
            # CHGNet usually uses 'node_feat' or 'atom_feat' at top level
            features = model_output.get("node_feat") or model_output.get("atom_feat")
            
            # M3GNet hides them in nested 'gc_X' layer outputs
            if features is None:
                for k in reversed(list(model_output.keys())):
                    if k.startswith("gc_") and isinstance(model_output[k], dict):
                        features = model_output[k].get("node_feat") or model_output[k].get("atom_feat")
                        if features is not None:
                            logger.info(f"Extracted MatGL features from layer: {k}")
                            break
            
            if features is None:
                 return {"error": f"Could not find node features in model output. Available keys: {list(model_output.keys())}"}
            
            if hasattr(features, "detach"):
                features = features.detach().cpu().numpy()
            
            return {
                "atomic_features": features.tolist(),
                "feature_dim": features.shape[1],
                "num_atoms": features.shape[0]
            }
        except Exception as e:
            import traceback
            logger.error(f"Failed to predict atomic features (MatGL): {e}\n{traceback.format_exc()}")
            return {"error": f"Failed to predict atomic features: {str(e)}"}

    def fine_tune(
        self,
        training_data: List[Dict[str, Any]],
        validation_data: Optional[List[Dict[str, Any]]] = None,
        training_config: Optional[Dict[str, Any]] = None,
        output_dir: Optional[str] = None,
        wandb_config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Fine-tune the MatGL model using PyTorch Lightning."""
        if not self.is_loaded:
            raise RuntimeError("Model must be loaded before fine-tuning")
        
        if not TRAINER_AVAILABLE:
            raise ImportError("Required fine-tuning components (lightning, dgl) are not available.")

        config = {
            "max_epochs": 10,
            "learning_rate": 1e-3,
            "batch_size": 4,
            "val_split": 0.1,
        }
        if training_config:
            # Unified key aliases: epochs → max_epochs
            if "epochs" in training_config and "max_epochs" not in training_config:
                training_config["max_epochs"] = training_config.pop("epochs")
            config.update(training_config)
        
        # Check for model/checkpoint override
        if training_config:
            new_model = training_config.get("foundation_model") or training_config.get("checkpoint_path")
            if new_model and new_model != self.model_name:
                logger.info(f"Reloading model for fine-tuning: {self.model_name} -> {new_model}")
                self.model_name = new_model
                self.is_loaded = False
                self.load()

        # 1. Prepare Data
        train_atoms, train_energies, train_forces, train_stresses = self._prepare_training_data(training_data)
        
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        structures = [adaptor.get_structure(a) for a in train_atoms]
        element_types = get_element_list(structures)
        
        # Get cutoff from model
        cutoff = getattr(self.model.model, 'cutoff', 5.0)
        threebody_cutoff = getattr(self.model.model, 'threebody_cutoff', 4.0)
        if not hasattr(self.model.model, 'threebody_cutoff') and hasattr(self.model.model, 'three_body_cutoff'):
            threebody_cutoff = getattr(self.model.model, 'three_body_cutoff')
            
        converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
        
        def to_list(obj):
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return obj

        labels = {
            "energies": to_list(train_energies),
            "forces": [to_list(f) for f in train_forces],
            "stresses": [to_list(s) for s in train_stresses],
        }
        
        import tempfile
        dataset_dir = tempfile.mkdtemp(prefix="mgl_dataset_")
        
        is_chgnet = "CHGNet" in self.model_name
        dataset = MGLDataset(
            structures=structures,
            converter=converter,
            labels=labels,
            threebody_cutoff=threebody_cutoff,
            include_line_graph=True,
            directed_line_graph=is_chgnet,
            save_dir=dataset_dir,
        )
        
        if validation_data:
            val_atoms, val_energies, val_forces, val_stresses = self._prepare_training_data(validation_data)
            val_structures = [adaptor.get_structure(a) for a in val_atoms]
            val_dataset = MGLDataset(
                structures=val_structures,
                converter=converter,
                labels={
                    "energies": to_list(val_energies),
                    "forces": [to_list(f) for f in val_forces],
                    "stresses": [to_list(s) for s in val_stresses],
                },
                threebody_cutoff=threebody_cutoff,
                include_line_graph=True,
                directed_line_graph=is_chgnet,
                save_dir=os.path.join(dataset_dir, "val"),
            )
            train_ds, val_ds = dataset, val_dataset
        else:
            if len(dataset) > 1:
                train_ds, val_ds = split_dataset(dataset, [1.0 - config['val_split'], config['val_split']], shuffle=True)
            else:
                train_ds, val_ds = dataset, None

        # Custom collate to move to device
        def my_collate(batch):
            return move_to_device(collate_fn_pes(batch, include_line_graph=True), self.device)

        train_loader, val_loader = MGLDataLoader(
            train_data=train_ds,
            val_data=val_ds,
            collate_fn=my_collate,
            batch_size=config['batch_size'],
            num_workers=0,
        )

        # 2. Setup Training Module
        prop_offset = getattr(self.model, 'element_refs', None)
        if prop_offset and hasattr(prop_offset, 'property_offset'):
            prop_offset = prop_offset.property_offset
            
        # Freezing backbone (Default: True)
        freeze_backbone = config.get("freeze_backbone", True)
        if freeze_backbone:
            # Freeze all parameters first
            for param in self.model.model.parameters():
                param.requires_grad = False
            
            # Unfreeze readout/head layers
            unfrozen_count = 0
            for name, module in self.model.model.named_modules():
                # Target common readout layer names in MatGL models
                if any(x in name.lower() for x in ["readout", "mlp_out", "final", "output"]):
                    for param in module.parameters():
                        param.requires_grad = True
                        unfrozen_count += 1
            
            if unfrozen_count == 0:
                logging.warning("No MatGL readout parameters found to unfreeze!")
            else:
                logging.info(f"MatGL backbone frozen. Unfrozen {unfrozen_count} parameters in readout layers.")

        # Determine stress_weight: use config value if given, else auto-detect from data
        has_stress = np.any(train_stresses)
        stress_weight = config.get("stress_weight", 1.0 if has_stress else 0.0)
        energy_weight = config.get("energy_weight", 1.0)
        force_weight = config.get("force_weight", 1.0)
        decay_steps = config.get("decay_steps", 1000)
        decay_alpha = config.get("decay_alpha", 0.01)

        lit_model = PotentialLightningModule(
            model=self.model.model,
            element_refs=prop_offset,
            lr=config['learning_rate'],
            include_line_graph=True,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            decay_steps=decay_steps,
            decay_alpha=decay_alpha,
        )
        
        # 3. Trainer
        history_callback = TrainingHistoryCallback()
        history_callback.training_history.update(self._collect_label_distributions(training_data))
        
        trainer = pl.Trainer(
            max_epochs=config['max_epochs'],
            accelerator="gpu" if self.device.startswith("cuda") else "cpu",
            devices=1,
            callbacks=[history_callback],
            enable_checkpointing=True,
            default_root_dir=output_dir,
            logger=False,
        )
        
        logger.info(f"Starting MatGL fine-tuning for {config['max_epochs']} epochs")
        trainer.fit(lit_model, train_loader, val_loader)
        
        self.is_fine_tuned = True
        self._training_history = history_callback.training_history
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.save_checkpoint(os.path.join(output_dir, "fine_tuned_model.pth"))
            
            if self.is_fine_tuned:
                 # Save numerical history to JSON
                 json_path = Path(output_dir) / "training_history.json"
                 self.save_training_history(str(json_path))
                 
                 plot_path = Path(output_dir) / "training_history.png"
                 try:
                     self.plot_training_history(save_path=str(plot_path), show=False)
                     logger.info(f"Training history plot saved to {plot_path}")
                 except Exception as e:
                     logger.warning(f"Failed to generate training history plot: {e}")

        
        return {
            "status": "success",
            "epochs": trainer.current_epoch,
            "final_loss": history_callback.training_history['loss_train'][-1] if history_callback.training_history['loss_train'] else None
        }

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """Save checkpoint."""
        if not self.is_loaded: raise RuntimeError("Model not loaded")
        
        data = {
            'model_state_dict': self.model.state_dict(),
            'model_name': self.model_name,
            'is_fine_tuned': self.is_fine_tuned,
            'training_history': getattr(self, '_training_history', None)
        }
        torch.save(data, checkpoint_path)

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load checkpoint."""
        if not self.is_loaded: self.load()
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.is_fine_tuned = checkpoint.get('is_fine_tuned', True)
        self._training_history = checkpoint.get('training_history')
        self.is_loaded = True

    def get_supported_elements(self) -> List[str]:
        """Get supported elements from the loaded model."""
        if self.is_loaded:
            # MatGL models usually store element types in the trainer or model metadata
            potential = self.model
            if hasattr(potential, 'model'):
                inner_model = potential.model
                # Check for AtomRef
                if hasattr(inner_model, 'atom_ref') and inner_model.atom_ref is not None:
                    if hasattr(inner_model.atom_ref, 'element_types'):
                        return list(inner_model.atom_ref.element_types)
                # Check directly on model
                if hasattr(inner_model, 'element_types'):
                    return list(inner_model.element_types)
                    
            # Fallback for standard models if not loaded but known
            if self.model_name.startswith("M3GNet") or self.model_name.startswith("CHGNet"):
                # Standard MP-trained models support 89 elements
                return ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Ac", "Th", "Pa", "U", "Np", "Pu"]
                
        return []

    def get_model_capabilities(self) -> Dict[str, bool]:
        """Get model capabilities."""
        return {
            "energy": True,
            "forces": True,
            "stress": True,
            "charges": self.model_name.startswith("CHGNet") if self.is_loaded else False,
            "dipole": False
        }

    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> Tuple:
        import ase.units
        from pymatgen.core import Structure
        from pymatgen.io.ase import AseAtomsAdaptor
        adaptor = AseAtomsAdaptor()
        
        atoms_list, energies, forces, stresses = [], [], [], []
        for d in training_data:
            atoms = d['structure']
            if isinstance(atoms, dict):
                # Handle Pymatgen structure dict
                try:
                    struct = Structure.from_dict(atoms)
                    atoms = adaptor.get_atoms(struct)
                except Exception:
                    continue
            elif not isinstance(atoms, Atoms):
                # Handle conversion if needed
                continue
            atoms_list.append(atoms)
            energies.append(d.get('energy', 0.0))
            forces.append(d.get('forces', np.zeros((len(atoms), 3))))
            
            s = d.get('stress')
            if s is not None:
                # MatGL's Potential module internally converts stress to GPa 
                # (see matgl.apps._pes_dgl.py). Thus, labels should be in GPa.
                # Project standard labels are in eV/A^3, so we convert to GPa for training.
                s_gpa = np.array(s) / ase.units.GPa
                # MatGL Potential.forward() outputs stress as [batch, 3, 3] tensors.
                # Convert Voigt (xx, yy, zz, yz, xz, xy) to 3x3 symmetric matrix.
                if s_gpa.shape == (6,):
                    s_3x3 = np.array([
                        [s_gpa[0], s_gpa[5], s_gpa[4]],
                        [s_gpa[5], s_gpa[1], s_gpa[3]],
                        [s_gpa[4], s_gpa[3], s_gpa[2]],
                    ])
                elif s_gpa.shape == (3, 3):
                    s_3x3 = s_gpa
                else:
                    s_3x3 = np.zeros((3, 3))
                stresses.append(s_3x3)
            else:
                stresses.append(np.zeros((3, 3)))
                
        return atoms_list, np.array(energies), forces, stresses
