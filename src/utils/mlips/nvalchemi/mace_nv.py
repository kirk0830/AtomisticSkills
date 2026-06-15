"""NValchemi model factory for MACE."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

import torch

from src.utils.mlips.nvalchemi.nvalchemi_utils import NVALCHEMI_AVAILABLE

if TYPE_CHECKING:
    from src.utils.mlips.mace.mace_wrapper import MACEWrapper

logger = logging.getLogger(__name__)


class _MACEHeadWrapper:
    """Thin mixin that injects a fixed `"head"` tensor into adapt_input.

    MACE multi-head models index per-head atomic-energy references and
    scale/shift parameters via ``data["head"]`` (shape [n_graphs]).  When
    "head" is absent, MACE's prepare_graph defaults to zeros (head 0), which
    is **not** the omat_pbe head for MACE-MH models (omat_pbe is head 5 in
    MACE-MH-1).  This wrapper ensures the correct head index is always
    injected, making NValchemi batch inference bit-identical to the
    sequential ASE calculator path.
    """

    _head_index: int  # set by factory; 0 is safe default for single-head models

    def adapt_input(self, data: Any, **kwargs: Any) -> dict[str, Any]:
        inputs = super().adapt_input(data, **kwargs)  # type: ignore[misc]
        # Infer n_graphs from the batch pointer already in the inputs dict.
        ptr = inputs.get("ptr")
        n_graphs = (ptr.numel() - 1) if ptr is not None else 1
        device = inputs["positions"].device
        inputs["head"] = torch.full(
            (n_graphs,), self._head_index, dtype=torch.long, device=device
        )
        return inputs


def get_nvalchemi_mace_model(wrapper: "MACEWrapper") -> Optional[Any]:
    """Return a NValchemi-compatible MACEWrapper for the loaded model.

    Extracts the raw nn.Module from the MACE calculator, wraps it in
    nvalchemi.models.mace.MACEWrapper, and patches it to inject the
    correct per-head index into every forward call.  This ensures that
    atomic-energy references and scale/shift are applied for the head
    that was selected at wrapper construction time, not the default
    head-0 fallback.

    Returns None if nvalchemi is unavailable, the wrapper is not loaded,
    or the calculator does not expose a models list.
    """
    if not NVALCHEMI_AVAILABLE:
        return None
    if not wrapper.is_loaded:
        return None

    # Check cached instance first
    if getattr(wrapper, "_nv_model", None) is not None:
        return wrapper._nv_model

    try:
        from nvalchemi.models.mace import MACEWrapper as NVMACEWrapper

        calc = wrapper.create_calculator()
        if not hasattr(calc, "models") or not calc.models:
            logger.warning(
                "MACE calculator has no .models attribute; NValchemi disabled."
            )
            return None

        raw_model = calc.models[0]

        # Resolve which head index this wrapper should use.
        # create_calculator() may default to "omat_pbe" for MH models when
        # wrapper.head is None; we read it back from the live calculator.
        head_name: str = getattr(calc, "head", None) or "Default"
        available_heads: list[str] = getattr(calc, "available_heads", ["Default"])
        if head_name in available_heads:
            head_index = available_heads.index(head_name)
        else:
            head_index = 0
            logger.warning(
                f"Head '{head_name}' not found in available_heads {available_heads}; "
                f"defaulting to head 0 ('{available_heads[0]}')"
            )

        # Build a head-aware subclass at runtime (avoids a top-level import
        # cycle while keeping the class hierarchy clean).
        HeadAwareMACE = type(
            "HeadAwareMACEWrapper",
            (_MACEHeadWrapper, NVMACEWrapper),
            {"_head_index": head_index},
        )

        nv_model = HeadAwareMACE(raw_model)
        # Enable stress computation (NValchemi MACEWrapper defaults to forces-only).
        nv_model.model_config.active_outputs.add("stress")
        # Move to the same device as the wrapper
        device = getattr(wrapper, "device", "cpu")
        nv_model = nv_model.to(device)
        wrapper._nv_model = nv_model  # cache
        return nv_model

    except Exception as e:
        logger.warning(f"Failed to build NValchemi MACE model: {e}")
        return None
