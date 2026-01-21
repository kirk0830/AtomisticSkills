
"""
Feature Calculators for MLIP models.

This module provides custom ASE calculators and PyTorch modules for extracting
crystal features (embeddings/descriptors) from various MLIP models (MatGL, MACE).
These features are used for clustering and sampling in the OffEquilibriumSampler.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Any, Optional, Dict, List, Union
from ase.calculators.calculator import Calculator, all_changes
from ase import Atoms

logger = logging.getLogger(__name__)

# Try imports
try:
    import matgl
    MATGL_AVAILABLE = True
except ImportError:
    MATGL_AVAILABLE = False


class CrystalFeaturePotential(nn.Module):
    """
    A custom Potential class that returns crystal features in addition to energy, forces, and stresses.
    """

    def __init__(
        self,
        model,
        data_mean: torch.Tensor | float = 0.0,
        data_std: torch.Tensor | float = 1.0,
        element_refs: torch.Tensor | np.ndarray | None = None,
        calc_forces: bool = True,
        calc_stresses: bool = True,
        calc_hessian: bool = False,
        calc_magmom: bool = False,
        calc_repuls: bool = False,
        zbl_trainable: bool = False,
        debug_mode: bool = False,
    ):
        super().__init__()
        self.model = model
        self.calc_forces = calc_forces
        self.calc_stresses = calc_stresses
        self.calc_hessian = calc_hessian
        self.calc_magmom = calc_magmom
        self.calc_repuls = calc_repuls
        self.data_mean = data_mean
        self.data_std = data_std
        self.element_refs = element_refs
        self.debug_mode = debug_mode
        self.zbl_trainable = zbl_trainable

    def forward(
        self,
        g: Any,  # dgl.DGLGraph
        lat: torch.Tensor,
        state_attr: torch.Tensor | None = None,
        l_g: Any | None = None,  # dgl.DGLGraph
    ) -> tuple[torch.Tensor, ...]:
        """
        Args:
            g: dgl Graph
            lat: lattice
            lattice: lattice
            state_attr: state attributes
            l_g: line graph

        Returns:
            (energy, forces, stress, hessian, crystal_features)
        """
        # This logic is adapted from matgl.apps.pes.Potential.forward
        # st (strain) for stress calculations
        st = lat.new_zeros([g.batch_size, 3, 3])
        if self.calc_stresses:
            st.requires_grad_(True)
            
        lattice = lat @ (torch.eye(3, device=lat.device) + st)
        g.edata["lattice"] = torch.repeat_interleave(lattice, g.batch_num_edges(), dim=0)
        g.edata["pbc_offshift"] = (g.edata["pbc_offset"].unsqueeze(dim=-1) * g.edata["lattice"]).sum(dim=1)
        g.ndata["pos"] = (
            g.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(lattice, g.batch_num_nodes(), dim=0)
        ).sum(dim=1)
        
        if self.calc_forces:
            g.ndata["pos"].requires_grad_(True)

        # Call model with return_all_layer_output=True to get energy and features in one go
        model_output = self.model(g=g, state_attr=state_attr, l_g=l_g, return_all_layer_output=True)
        
        if not isinstance(model_output, dict):
             raise TypeError(f"Model {self.model.__class__.__name__} must return a dict when return_all_layer_output=True, instead got {type(model_output)}")

        # Extract energy
        total_energy = model_output.get("final")
        if total_energy is None:
             raise KeyError(f"Model output dict did not contain 'final' energy key. Available keys: {list(model_output.keys())}")

        # Extract crystal features
        crystal_features = model_output.get("graph_feat")
        if crystal_features is None:
            # Try to find node features to average
            node_features = None
            # Check for various keys used in different versions (M3GNet/CHGNet)
            if "gc_3" in model_output and isinstance(model_output["gc_3"], dict):
                node_features = model_output["gc_3"].get("node_feat") or model_output["gc_3"].get("atom_feat")
            
            if node_features is None:
                node_features = model_output.get("node_feat") or model_output.get("atom_feat")
                
            if node_features is not None:
                # Average node features to get crystal feature
                crystal_features = torch.mean(node_features, dim=0)
        
        if crystal_features is None:
             raise KeyError(f"Could not extract crystal features from model output keys: {list(model_output.keys())}")
        
        # Apply standard Potential post-processing
        if hasattr(self, 'element_refs') and self.element_refs is not None:
             # element_refs in Potential is often an AtomRef module
             if isinstance(self.element_refs, nn.Module):
                 property_offset = torch.squeeze(self.element_refs(g))
             else:
                 # Fallback if it's just raw data
                 node_feat = g.ndata["node_type"]
                 property_offset = torch.sum(self.element_refs[node_feat], dim=0)
             total_energy += property_offset

        total_energy = total_energy * self.data_std + self.data_mean

        # Calculate derivatives
        forces = torch.zeros(1)
        stress = torch.zeros(1)
        hessian = torch.zeros(1)

        grad_vars = [g.ndata["pos"], st] if self.calc_stresses else [g.ndata["pos"]]

        if self.calc_forces:
            from torch.autograd import grad
            grads = grad(
                total_energy,
                grad_vars,
                grad_outputs=torch.ones_like(total_energy),
                create_graph=self.calc_hessian,
                retain_graph=self.calc_hessian,
                allow_unused=True,
            )
            forces = -grads[0]
            
            if self.calc_stresses:
                sts = grads[1]
                # Volume calculation for stress conversion
                # Use det(lattice)
                vol = torch.abs(torch.det(lattice))
                scale = 1.0 / vol # eV/A^3 (standard ASE unit)
                stress = sts * scale
                # Convert to Voigt or keep as tensor
                if stress.dim() == 3:
                     # stress is [batch, 3, 3]
                     # ASE Expects [3, 3] or [6]
                     stress = stress[0] # Assuming batch size 1
                
        return total_energy, forces, stress, hessian, crystal_features


class MatGLCrystalFeatureCalculator(Calculator):
    """
    Custom MatGL calculator that stores crystal features in results during calculation.
    wraps a MatGL model and uses CrystalFeaturePotential.
    """
    
    implemented_properties = ("energy", "free_energy", "forces", "stress", "hessian", "magmoms", "crystal_fea")
    
    def __init__(self, potential, state_attr=None, stress_unit="eV/Å³", stress_weight=1.0, use_voigt=False, device="auto", **kwargs):
        """
        Args:
            potential: matgl.apps.pes.Potential or matgl.models.*
            state_attr: State attributes
            stress_unit: Unit for stress (default eV/Å³)
            stress_weight: Weight for stress
            use_voigt: Use Voigt notation for stress
            device: Device to run on
        """
        Calculator.__init__(self, **kwargs)
        
        # If potential is already a Potential class, unwrap the model
        if hasattr(potential, "model"):
            model = potential.model
            # Copy other attributes if needed
            data_mean = getattr(potential, "data_mean", 0.0)
            data_std = getattr(potential, "data_std", 1.0)
            element_refs = getattr(potential, "element_refs", None)
        else:
            model = potential
            data_mean = 0.0
            data_std = 1.0
            element_refs = None
            
        self.potential = CrystalFeaturePotential(
            model=model,
            data_mean=data_mean,
            data_std=data_std,
            element_refs=element_refs,
            calc_forces=True,
            calc_stresses=True # Always calculate stress for MD
        )
        
        self.state_attr = state_attr
        self.device = device
        self.potential.to(device)

    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate properties."""
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Convert atoms to graph
        import matgl
        from matgl.ext.pymatgen import get_element_list, Structure2Graph
        from pymatgen.io.ase import AseAtomsAdaptor
        
        adaptor = AseAtomsAdaptor()
        struct = adaptor.get_structure(atoms)
        
        # determine elements and cutoff (simplified)
        elements = self.potential.model.element_types if hasattr(self.potential.model, "element_types") else get_element_list([struct])
        cutoff = 5.0 # Default
        if hasattr(self.potential.model, "cutoff"):
             cutoff = self.potential.model.cutoff
             
        converter = Structure2Graph(element_types=elements, cutoff=cutoff)
        # get_graph returns graph, state_attr, (optional) other info depending on MatGL version
        graph_data = converter.get_graph(struct)
        if len(graph_data) == 3:
             graph, state_attr, _ = graph_data
        else:
             graph, state_attr = graph_data
        
        if self.state_attr is not None:
             state_attr = self.state_attr
             
        # Prepare inputs
        # Debug graph
        # print(f"DEBUG: MatGL graph ndata keys: {graph.ndata.keys()}")
        if "pos" not in graph.ndata:
             # In newer MatGL/DGL versions, pos might be stored differently or need setting?
             # But M3GNet usually expects it.
             # Maybe graph is not what we think?
             pass
        
        try:
            graph.ndata["pos"].requires_grad_(True)
        except Exception:
            pass
        lattice = torch.tensor(struct.lattice.matrix, dtype=matgl.float_th, device=self.device).unsqueeze(0)
        lattice.requires_grad_(True)
        
        import dgl
        g_batch = dgl.batch([graph]).to(self.device)
        state_attr = torch.tensor(state_attr, dtype=matgl.float_th, device=self.device).unsqueeze(0) if state_attr is not None else None
        
        # Run model
        energy, forces, stress, hessian, crystal_features = self.potential(g_batch, lattice, state_attr)
        
        # Store results
        self.results["energy"] = energy.detach().cpu().numpy().item()
        self.results["free_energy"] = self.results["energy"]
        # Remove squeeze(0) which was incorrect for multiple atoms
        self.results["forces"] = forces.detach().cpu().numpy()
        
        # Stress handling
        self.results["stress"] = stress.detach().cpu().numpy()
        s = self.results["stress"]
        
        # Convert to Voigt [xx, yy, zz, yz, xz, xy]
        if s.shape == (3, 3):
             s_voigt = np.array([s[0,0], s[1,1], s[2,2], s[1,2], s[0,2], s[0,1]])
             self.results["stress"] = s_voigt
        else:
             self.results["stress"] = s
        
        # Ensure forces are 2D (N, 3)
        if self.results["forces"].ndim == 1:
             self.results["forces"] = self.results["forces"].reshape(-1, 3)
             
        # Store crystal features
        self.results["crystal_fea"] = crystal_features.detach().cpu().numpy().flatten()


class MaceCrystalFeatureCalculator(Calculator):
    """
    Wrapper for MACE calculator that adds crystal feature extraction.
    Uses 'get_descriptors' from MACE to get atomic features and averages them.
    """
    
    implemented_properties = ("energy", "free_energy", "forces", "stress", "magmoms", "crystal_fea")
    
    def __init__(self, mace_calculator: Calculator, **kwargs):
        """
        Args:
            mace_calculator: An initialized mace.calculators.mace.MACECalculator
        """
        self.mace_calc = mace_calculator
        Calculator.__init__(self, **kwargs)
        
        # Copy settings from wrapped calculator
        self.parameters = self.mace_calc.parameters
        
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """Calculate properties using MACE and extract descriptors."""
        
        # Standard ASE calculator setup (sets self.atoms, clears self.results if needed)
        Calculator.calculate(self, atoms, properties, system_changes)
        
        # Use underlying MACE calculator for standard properties
        # Explicitly call getters to ensure results are populated
        # This handles cases where .calculate() might be lazy or tricky with attachment
        try:
            # We want energy and forces mostly
            # If properties is None, ASE defaults to 'energy', 'forces' usually
            req_props = properties if properties else ['energy', 'forces']
            if 'energy' in req_props:
                 _ = self.mace_calc.get_potential_energy(atoms)
            if 'forces' in req_props:
                 _ = self.mace_calc.get_forces(atoms)
                 
            # If stress is requested
            if 'stress' in req_props:
                 _ = self.mace_calc.get_stress(atoms)
        except Exception:
            pass
            
        # self.mace_calc.calculate(atoms, properties, system_changes)
        
        # Copy standard results to this calculator
        self.results.update(self.mace_calc.results)
        
        # Extract crystal features
        # get_descriptors returns (n_atoms, n_features)
        # We average over atoms to get (n_features,)
        descriptors = self.mace_calc.get_descriptors(atoms)
        
        if descriptors is not None:
            # Average over atoms (axis 0)
            crystal_features = np.mean(descriptors, axis=0)
            self.results["crystal_fea"] = crystal_features
        else:
             raise RuntimeError("MACE get_descriptors returned None")

    def get_property(self, name, atoms=None, allow_calculation=True):
        if name == "crystal_fea":
             if name not in self.results:
                  if allow_calculation:
                       self.calculate(atoms=atoms, properties=[name])
                  else:
                       return None
             return self.results[name]
             return self.results[name]
        
        # For structural properties, use standard Calculator mechanism
        # This ensures self.calculate is called, populating self.results and crystal_fea
        return Calculator.get_property(self, name, atoms, allow_calculation)

