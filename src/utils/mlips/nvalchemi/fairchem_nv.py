"""NValchemi BaseModelMixin wrapper for FairChem predict_unit models."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch import nn

from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

if TYPE_CHECKING:
    from src.utils.mlips.fairchem.fairchem_wrapper import FAIRCHEMWrapper

logger = logging.getLogger(__name__)

try:
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.models.base import (
        BaseModelMixin,
        ModelConfig,
    )

    _NV_AVAILABLE = True
except ImportError:
    _NV_AVAILABLE = False
    BaseModelMixin = object  # type: ignore[misc,assignment]


class FairChemWrapper(nn.Module, BaseModelMixin):  # type: ignore[misc]
    """NValchemi BaseModelMixin wrapper for FairChem predict_unit models.

    Builds FairChem graphs on the fly from NValchemi Batch positions using
    the GPU-accelerated neighbor list already present in FairChem
    (fairchem.core.graph.radius_graph_pbc_nvidia).  Autograd forces and
    stresses are computed inside predict_unit, so no affine-strain trick is
    needed here.

    Note: ``neighbor_config=None`` so the NValchemi NeighborListHook is
    NOT registered.  The graph is rebuilt inside ``forward()`` every step.
    """

    def __init__(
        self,
        predict_unit: Any,
        task_name: str,
        device: str = "cpu",
        cutoff: float = 12.0,
        max_neigh: int = 500,
    ) -> None:
        if not _NV_AVAILABLE:
            raise ImportError("nvalchemi-toolkit is required to use FairChemWrapper.")
        super().__init__()

        self.predict_unit = predict_unit
        self.task_name = task_name
        self._device = device
        self.cutoff = cutoff
        self.max_neigh = max_neigh

        # No neighbor_config: FairChem builds its own graph internally.
        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces", "stress"}),
            active_outputs={"energy", "forces", "stress"},
            autograd_outputs=frozenset(),  # FC handles autograd internally
            autograd_inputs=frozenset(),
            required_inputs=frozenset(),
            optional_inputs=frozenset(),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=None,
        )

    # ------------------------------------------------------------------
    # Neighbor list helper
    # ------------------------------------------------------------------

    def _build_neighbor_list(
        self,
        positions: "torch.Tensor",
        cell: "torch.Tensor",
        pbc: "torch.Tensor",
        natoms: "torch.Tensor",
        batch_idx: "torch.Tensor",
        device: "torch.device",
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """Return (edge_index [2,E], cell_offsets [E,3]) for the full batch.

        Tries the GPU-accelerated `get_neighbors_nvidia` first.  Falls back to
        FairChem's CPU-compatible `radius_graph_pbc` when nvalchemiops warp
        kernels are unavailable (e.g. testing on CPU or older GPU).
        """
        try:
            from fairchem.core.graph.radius_graph_pbc_nvidia import get_neighbors_nvidia

            c_index, n_index, offsets, _ = get_neighbors_nvidia(
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=self.cutoff,
                max_neigh=self.max_neigh,
                batch=batch_idx.int(),
                natoms=natoms,
            )
            edge_index = torch.stack([c_index.long(), n_index.long()], dim=0)
            return edge_index, offsets
        except (RuntimeError, ImportError):
            pass

        # CPU fallback: build graph for the full batch using FairChem's standard builder.
        from fairchem.core.graph.compute import radius_graph_pbc

        import types

        data_batch = types.SimpleNamespace(
            pos=positions,
            cell=cell,
            pbc=pbc[0],  # radius_graph_pbc uses a single [3] pbc tensor
            natoms=natoms,
        )
        edge_index, cell_offsets, _ = radius_graph_pbc(
            data_batch,
            radius=self.cutoff,
            max_num_neighbors_threshold=self.max_neigh,
        )
        return edge_index, cell_offsets

    # ------------------------------------------------------------------
    # BaseModelMixin interface
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {}

    def adapt_input(self, data: Any, **kwargs: Any) -> Any:
        """Build a batched FairChem graph from a NValchemi Batch."""
        from fairchem.core.datasets.atomic_data import (
            AtomicData as FCAtomicData,
            atomicdata_list_to_batch,
        )

        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        num_graphs: int = data.num_graphs
        dtype = data.positions.dtype
        device = data.positions.device

        # Extract per-graph slices
        batch_idx = data.batch_idx.long()  # [N] — must be int64
        positions = data.positions  # [N, 3]
        atomic_numbers = data.atomic_numbers.long()  # [N] — FCAtomicData requires int64

        cell_raw = getattr(data, "cell", None)
        if cell_raw is not None:
            cell = cell_raw.to(dtype=dtype, device=device)  # [B, 3, 3]
            pbc = torch.ones(num_graphs, 3, dtype=torch.bool, device=device)
        else:
            cell = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(num_graphs, -1, -1)
            )
            pbc = torch.zeros(num_graphs, 3, dtype=torch.bool, device=device)

        natoms = torch.bincount(batch_idx, minlength=num_graphs)  # [B]

        # Try GPU-accelerated neighbor list first; fall back to FairChem's CPU implementation.
        edge_index, cell_offsets = self._build_neighbor_list(
            positions, cell, pbc, natoms, batch_idx, device
        )

        # Build one FairChem AtomicData per graph
        data_list: list[FCAtomicData] = []
        atom_offset = 0
        edge_offset = 0
        for i in range(num_graphs):
            n_i = int(natoms[i].item())
            pos_i = positions[atom_offset : atom_offset + n_i]
            z_i = atomic_numbers[atom_offset : atom_offset + n_i]
            cell_i = cell[i : i + 1]  # [1, 3, 3]
            pbc_i = pbc[i : i + 1]  # [1, 3]

            # Filter edges for graph i (center atom in range)
            edge_mask = (edge_index[0] >= atom_offset) & (
                edge_index[0] < atom_offset + n_i
            )
            ei_local = edge_index[:, edge_mask] - atom_offset  # local indices
            co_i = cell_offsets[edge_mask].to(dtype)
            n_edges_i = int(edge_mask.sum().item())

            # UMA model's csd_embedding asserts dataset is not None.
            # Fall back to "omat" when task_name was not specified.
            dataset_name = self.task_name or "omat"
            fc_data = FCAtomicData(
                pos=pos_i,
                atomic_numbers=z_i,
                cell=cell_i,
                pbc=pbc_i,
                natoms=torch.tensor([n_i], device=device),
                edge_index=ei_local,
                cell_offsets=co_i,
                nedges=torch.tensor([n_edges_i], device=device),
                charge=torch.zeros(1, device=device),
                spin=torch.zeros(1, device=device),
                fixed=torch.zeros(n_i, dtype=torch.long, device=device),
                tags=torch.zeros(n_i, dtype=torch.long, device=device),
                sid=[f"nv_{i}"],
                dataset=dataset_name,
            )
            data_list.append(fc_data)
            atom_offset += n_i
            edge_offset += n_edges_i

        return atomicdata_list_to_batch(data_list)

    def adapt_output(self, model_output: Any, data: Any, **kwargs: Any) -> Any:
        """Map FairChem predict outputs back to NValchemi ModelOutputs format."""

        # FairChem predict_unit returns a dict with task-keyed results.
        # For energy/forces/stress tasks these are at top level.
        outputs = super().adapt_output({}, data)

        energy = model_output.get("energy")
        forces = model_output.get("forces")
        stress = model_output.get("stress")

        if energy is not None:
            if energy.dim() == 1:
                energy = energy.unsqueeze(-1)  # [B, 1]
            outputs["energy"] = energy

        if forces is not None:
            outputs["forces"] = forces

        if stress is not None:
            # FairChem stress is [B, 3, 3]; NValchemi wants [B, 3, 3]
            outputs["stress"] = stress

        return outputs

    def forward(self, data: Any, **kwargs: Any) -> Any:
        """Run FairChem predict_unit and return NValchemi ModelOutputs."""
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        fc_batch = self.adapt_input(data)

        # Run FairChem inference (MLIPPredictUnit does not have .eval())
        with torch.no_grad():
            raw_out = self.predict_unit.predict(fc_batch)

        return self.adapt_output(raw_out, data)

    def compute_embeddings(self, data: Any, **kwargs: Any) -> Any:
        """Not implemented for FairChem — returns data unchanged."""
        return data

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_predict_unit(
        cls,
        predict_unit: Any,
        task_name: str,
        device: str = "cpu",
        cutoff: float = 12.0,
        max_neigh: int = 500,
    ) -> "FairChemWrapper":
        """Construct from a FairChem predict_unit.

        Parameters
        ----------
        predict_unit : UMAModule or similar
            Loaded FairChem predict_unit (from pretrained_mlip or load_predict_unit).
        task_name : str
            Task head name (e.g., "omat", "omol", "s2ef").
        device : str
            Device to run on.
        cutoff : float
            Neighbor list cutoff in Å.  Should match the model's cutoff.
        max_neigh : int
            Max neighbors per atom for the GPU neighbor list.
        """
        wrapper = cls(
            predict_unit=predict_unit,
            task_name=task_name,
            device=device,
            cutoff=cutoff,
            max_neigh=max_neigh,
        )
        return wrapper.to(device)


def get_nvalchemi_fairchem_model(wrapper: "FAIRCHEMWrapper") -> Optional[Any]:
    """Return a NValchemi-compatible FairChemWrapper for the loaded model.

    Returns None if nvalchemi is unavailable, the wrapper is not loaded,
    or FairChem's GPU neighbor list support is not available.
    """
    if not NVALCHEMI_AVAILABLE:
        return None
    if not wrapper.is_loaded or wrapper.model is None:
        return None

    # Check cached instance first
    if getattr(wrapper, "_nv_model", None) is not None:
        return wrapper._nv_model

    try:
        # Try to infer the model's cutoff from the predict_unit backbone
        cutoff = 12.0
        if hasattr(wrapper.model, "backbone") and hasattr(
            wrapper.model.backbone, "cutoff"
        ):
            cutoff = float(wrapper.model.backbone.cutoff)
        elif hasattr(wrapper.model, "model") and hasattr(wrapper.model.model, "cutoff"):
            cutoff = float(wrapper.model.model.cutoff)

        nv_model = FairChemWrapper.from_predict_unit(
            predict_unit=wrapper.model,
            task_name=wrapper.task_name,
            device=wrapper.device,
            cutoff=cutoff,
        )
        wrapper._nv_model = nv_model  # cache
        return nv_model

    except Exception as e:
        logger.warning(f"Failed to build NValchemi FairChem model: {e}")
        return None
