---
name: ml-mlip-automl
description: Automate hyperparameter tuning for MLIPs (MACE, MatGL, FairChem) using an LLM-driven search framework.
category: [machine-learning]
---

# LLM-Coupled MLIP Hyperparameter Search (AutoML)

## Goal
To automate the discovery of optimal training hyperparameters (learning rate, batch size, backbone freezing) by coupling the agent to an iterative execution loop. Instead of conducting exhaustive grid searches, the LLM actively interprets validation MAE trajectories per epoch and strategically deduces the next optimal parameter permutation to evaluate.

## The Agentic Workflow

Instead of calling a monolithic training tool, this SKILL dictates how you (the Agent) should orchestrate a search using the individual foundation fine-tuning SKILLs (`ml-mace-finetune`, `ml-matgl-finetune`, `ml-fairchem-finetune`).

### **Execution Protocol**

1. **Initial Anchor**: Select a conservative parameter set based on the dataset size (e.g., small dataset: freeze backbone, lower LR).
2. **Execute Fine-Tuning**: Create a dedicated subdirectory for the run (e.g., `search_run_lr1e-3_frozen`) and execute the respective MLIP's `prepare_*_data.py` and training runner.
3. **Parse and Evaluate**: Read the resulting `training_history.json` directly into your context. Analyze the validation loss and MAE curves (slope, signs of overfitting/forgetting).
4. **Iterate**: Deduce the next logical parameter adjustment based on the evaluation (e.g. "Validation MAE plateaued early. Action: Unfreeze backbone & lower LR to 1e-4").
5. **Terminate**: Halt and report the best configuration once a performance target is met or degradation occurs repeatedly.

## Hyperparameter Search Spaces

Different MLIP frameworks expose different tunable parameters. When deducing your next move, restrict your search space to the available features:

| Feature | `ml-fairchem-finetune` | `ml-matgl-finetune` | `ml-mace-finetune` |
|:--------|:-----------------------|:--------------------|:-------------------|
| **Freeze backbone** | `--freeze-backbone` | `--freeze-backbone` | `--freeze-backbone` |
| **Re-init head** | N/A (Auto) | `--reinit-head` | `--reinit-head` |
| **Learning Rate** | `--lr` (Default: 4e-4) | `--lr` (Default: 1e-3) | `--lr` (Default: 1e-4) |
| **Batch Size** | `--batch-size` | `--batch-size` | `--batch-size` |
| **Scheduler** | Cosine with Warmup | Cosine or Plateau | Exponential or Plateau |
| **Weight Decay** | `--weight-decay` | N/A | `weight_decay` (YAML) |

> [!TIP]
> **Constraints & Best Practices**
> - **Freezing**: For datasets <500 structures, always begin with the backbone **frozen** to prevent catastrophic forgetting.
> - **MACE LR Sensitivity**: Never exceed `1e-4` for MACE-OMAT fine-tuning initially, as higher values trigger immediate divergence.

## Invoking the Search (User Prompt Template)

To initiate this workflow, users should provide an anchor dataset and instruction similar to:

> "I have a dataset at `private_data/my_dataset.json`. I want to fine-tune `MACE-OMAT-0-small` to achieve an energy MAE below 5 meV/atom. Please orchestrate an LLM-driven hyperparameter search over learning rates (`1e-3`, `1e-4`), batch sizes (`2`, `8`), and backbone freezing strategies. Execute one run at a time, read the `training_history.json`, explain your logic for the next permutation, and stop when you find the best combination."

---

**Author:** Bowen Deng  
**Contact:** [GitHub @bowen-bd](https://github.com/bowen-bd)
