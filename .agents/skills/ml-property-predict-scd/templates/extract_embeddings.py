import os
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch_geometric.loader import DataLoader

from models.model_helper import load_model


REPO_PREFIX = "Ty-Perez/"
MODEL_NAME = "ct-scd-pcq"  # switch to "ct-scd-amp" for periodic materials
CACHE_DIR = "./experiments"
OUTPUT_PATH = "scd_embeddings.npz"
BATCH_SIZE = 32
RETURN_ATOM_EMBS = False


def build_dataset():
    """
    Replace this with a real torch_geometric-compatible dataset.

    The loader should yield batches with at least:
    - z
    - pos
    - batch
    Optional labels:
    - y
    """
    raise NotImplementedError


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = hf_hub_download(
        repo_id=f"{REPO_PREFIX}{MODEL_NAME}",
        filename="last.ckpt",
        cache_dir=CACHE_DIR,
    )

    model, _ema_model, _ckpt = load_model(ckpt_path, device=device)
    model.eval()

    dataset = build_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    mol_embs = []
    atom_embs = []
    targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(
                z=batch.z,
                pos=batch.pos,
                batch=batch.batch,
                graph_batch=batch,
                return_atom_embs=RETURN_ATOM_EMBS,
            )

            mol_embs.append(out["mol_emb"].cpu().numpy())

            if RETURN_ATOM_EMBS:
                atom_embs.append(out["atom_embs"].cpu().numpy())

            if hasattr(batch, "y"):
                y = batch.y.detach().cpu().reshape(batch.num_graphs, -1).numpy()
                targets.append(y)

    payload = {
        "mol_emb": np.concatenate(mol_embs, axis=0),
    }

    if atom_embs:
        payload["atom_embs"] = np.array(atom_embs, dtype=object)

    if targets:
        payload["y"] = np.concatenate(targets, axis=0)

    np.savez(OUTPUT_PATH, **payload)
    print(f"Saved embeddings to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
