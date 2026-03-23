# Transfer Recipes

Use the checkpoint that matches the domain first:

- molecules: `ct-scd-pcq`
- materials: `ct-scd-amp`

## Frozen backbone embeddings

The upstream notebook demonstrates the core pattern:

1. download `last.ckpt` with `hf_hub_download()`
2. load the model with `models.model_helper.load_model()`
3. call the model on a PyG batch and read `out["mol_emb"]`

Recommended outputs:

- `out["mol_emb"]`: graph-level feature vector for downstream ML
- `out["atom_embs"]`: atom- or site-level feature tensor when `return_atom_embs=True`

Recommended runtime choices:

- keep the checkpoint frozen
- set `model.eval()`
- set `model.denoise = False` when the denoising head is not needed
- configure graph mode to match the task: molecules can use the compiled path, materials require loader-side graphs

Use `templates/extract_embeddings.py` as the starting point.

## Linear probe with a frozen backbone

For a true linear probe on top of the pretrained representation:

1. load the SCD checkpoint
2. freeze the entire pretrained model
3. run each raw batch through the frozen checkpoint to get `mol_emb`
4. train only a new linear head on top of those live embeddings

Recommended probe head:

- scalar regression: `torch.nn.Linear(model.emb_dim, out_dim)`

Use `templates/train_linear_probe_head.py` as the starting point.

Important caveats:

- do not use cached embedding files as the primary linear-probe workflow
- do not use the native SCD `scalar_head` as if it were a simple linear probe
- native `train.py` does not expose a clean freeze-backbone flag, so this probe workflow is better handled by a custom script

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

- use frozen backbone embeddings when you want a pretrained encoder inside a larger downstream pipeline
- use a frozen-backbone linear probe when you want the cleanest test of representation quality while still training on raw structures end-to-end
- use full-model finetuning when you have enough labels and want the best task-specific accuracy
