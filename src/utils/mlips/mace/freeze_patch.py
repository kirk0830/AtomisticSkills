import torch
import logging

logger = logging.getLogger("mace_freeze_patch")


def apply_flexible_freezing(model: torch.nn.Module, trainable_modules: list) -> None:
    """
    Freeze everything and then unfreeze specific modules.
    Supported keys: "readouts", "products", "interactions"
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # 2. Unfreeze selected
    unfrozen_count = 0
    total_count = sum(1 for _ in model.parameters())

    mapping = {
        "readouts": ["readouts", "readout"],
        "products": ["products"],
        "interactions": ["interactions"],
    }

    for key in trainable_modules:
        attrs = mapping.get(key, [])
        for attr in attrs:
            if hasattr(model, attr):
                module = getattr(model, attr)
                logger.info(f"Unfreezing module: {attr}")
                for param in module.parameters():
                    param.requires_grad = True
                    unfrozen_count += 1
            else:
                logger.warning(f"Module '{attr}' not found in model.")

    logger.info(f"Unfrozen {unfrozen_count}/{total_count} parameter groups.")


def apply_patch(trainable_modules: list):
    """
    Apply the monkeypatch to mace.tools.scripts_utils.get_params_options
    """
    import mace.tools.scripts_utils

    original_get_params_options = mace.tools.scripts_utils.get_params_options

    def patched_get_params_options(args, model):
        logger.info(
            f"Applying flexible unfreezing logic. Trainable modules: {trainable_modules}"
        )
        apply_flexible_freezing(model, trainable_modules)
        return original_get_params_options(args, model)

    mace.tools.scripts_utils.get_params_options = patched_get_params_options
