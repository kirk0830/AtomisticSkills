# Transfer Recipes

Use the checkpoint that matches the domain first:

- molecules: `ct-scd-pcq`
- materials: `ct-scd-amp`

## Frozen embeddings

The upstream notebook demonstrates the core pattern:

1. download `last.ckpt` with `hf_hub_download()`
2. load the model with `models.model_helper.load_model()`
3. call the model on a PyG batch and read `out["mol_emb"]`

Recommended outputs:

- `out["mol_emb"]`: graph-level feature vector for regression or classification
- `out["atom_embs"]`: atom- or site-level feature tensor when `return_atom_embs=True`

Use `templates/extract_embeddings.py` as the starting point.

## Linear probe on frozen embeddings

For a true linear probe on the published graph embeddings:

1. freeze the SCD model by keeping it in inference mode
2. cache `mol_emb` for every example
3. train a simple external model on those cached features

Recommended defaults:

- scalar regression: `sklearn.linear_model.Ridge`
- binary or multiclass classification: `sklearn.linear_model.LogisticRegression`

Use `templates/linear_probe_from_embeddings.py` as the starting point.

Important caveat:

- `train.py` does not expose a native "freeze full backbone and optimize only a linear probe on `mol_emb`" path
- if you need that exact transfer mode, keep it outside the Lightning trainer

## Full-model finetuning

Use the native training path when you want the full pretrained checkpoint updated:

```bash
python train.py --conf configs/my_finetune.yaml --load-hf ct-scd-pcq --job-id my_run
```

or

```bash
python train.py --conf configs/my_finetune.yaml --load-hf ct-scd-amp --job-id my_run
```

Useful switches:

- `reset_head: true`
  reasonable default when switching from SCD pretraining to a new scalar property
- `reset_embeddings: false`
  keep pretrained atomic embeddings unless you have a strong reason to reinitialize
- `reset_norms: null`
  leave norms alone unless transfer is unstable

The recommendation on `reset_head: true` is an inference from the code: the pretraining path freezes `scalar_head`, so the public checkpoint head is not a trained downstream property head.

## Choosing between the three modes

- use frozen embeddings when the dataset is small, labels are scarce, or you need a quick baseline
- use a linear probe when you want the cleanest test of representation quality
- use full-model finetuning when you have enough labels and want the best task-specific accuracy
