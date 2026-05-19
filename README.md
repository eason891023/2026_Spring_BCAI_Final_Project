# Nested Learning on MLP based Models
## Proposal
> TBD!

## Project Architecture
```plain
nested_learning_sandbox/
├── README.md               # Project overview and run instructions
├── requirements.txt        # Core dependencies
├── main.py                 # The CLI entry point for running specific ablation phases
├── analyze.py              # Data visualization and metric summary pipeline
├── data/                   
│   └── results/            # Auto-generated outputs
│       ├── raw_log/        # Terminal stdout records
│       ├── metrics/        # epochwise-recorded CSVs and JSON summaries
│       └── plots/          # Charts and heatmaps
└── src/                    # Main source code directory
    ├── __init__.py
    ├── data/               # Data ingestion and task splitting
    │   ├── __init__.py
    │   └── split_mnist.py
    ├── models/             # Network architectures
    │   ├── __init__.py
    │   ├── baseline.py     # Standard monolithic MLP
    │   └── cms_mlp.py      # Continuum Memory System MLPs (SCMS, NCMS, ICMS)
    ├── optimizers/         # Optimizer definitions
    │   ├── __init__.py
    │   ├── factory.py      # Instantiation logic and Decoupled Wrappers
    │   ├── m3.py           # Multi-timescale Momentum Muon (and variants)
    │   └── muon.py         # Standard Muon implementation
    └── engine/             # Training loops and evaluation logic
        ├── __init__.py
        ├── metrics.py      # Accuracy Matrix (R), F, BWT, and Epoch tracking
        └── trainer.py      # Task-incremental training loop
```

## Build Environment
> Run the following commands inside the project directory!
```bash
conda create -n nlvm python=3.12 -y
conda activate nlvm
pip install -r requirements.txt
```

## Usage
> Make sure to setup the environment first and activate the virtual environment before running the following commands.

### 1. Manual Single Runs

```bash
python main.py --model ["scms", "icms", "ncms", "baseline"] --optimizer ['SGD', 'Adam', 'Muon', 'M3', 'M3S', 'MSGD', 'MAdam']
```

> Please choose only one option at once from the `[...]` in the options listed above! E.g.: `python main.py --model baseline --optimizer SGD`

**Universal Core Options:**
* `--model`: The network architecture. Choose `baseline` (Standard monolithic MLP) or `cms` (including three variants: `scms`, `ncms`, and `icms`).
* `--optimizer`: The optimization engine. Supports standard (`SGD`, `Adam`, `Muon`) and Multi-scale (`M3`, `M3S`, `MSGD`, `MAdam`) algorithms.
* `--epochs`: Training epochs per task. Default: `5`.
* `--batch_size`: Batch size for the dataloaders. Default: `64`.
* `--lr`: Base learning rate. Default: `1e-3`. *(Tip: Orthogonalizing optimizers like Muon and M3 often require lower learning rates than Adam/SGD).*

**Multi-Scale & Memory Options:**
* `--f`: The Temporal Split Frequency. Defines the number of high-frequency inner loops executed before triggering the slow memory update for the deeper layers. Default: `20`. *(Note: The impact of this delay scales closely with batch size).*
* `--alpha`: The Memory Force ($\alpha$). Defines the scaling multiplier applied to the delayed slow memory buffer ($O^{(2)}$) during the outer loop update. It controls how aggressively the accumulated history overwrites the deep continuum layers. Default: `0.5`.
* `--beta3`: The Memory Horizon ($\beta_3$). Controls how the slow memory accumulation buffer ($M^{(2)}$) integrates historical gradients. Its mathematical behavior changes depending on the `--stab` parameter. Default: `0.9`.
* `--stab`: The Stabilization Flag. Determines the mathematical behavior of the slow memory accumulation:
  * `1` **(Stabilized - e.g., `M3S`):** Applies an Exponential Moving Average (EMA) decay rate. Higher values safely extend the "memory horizon" by gently decaying older batches without blowing up the buffer size.
  * `0` **(Non-Stabilized - e.g., `M3`):** Uses a direct additive multiplier to perfectly match the literal pseudo-code of the original paper ($M = M + \beta_3 g$). *Warning: This unbounded accumulation can cause infinite memory horizons, gradient explosions, or vanishing effective learning rates over long training sequences.*

> [!NOTE]
> **Note:** Standard optimizers will automatically utilize a Decoupled Wrapper when paired with the `cms` variant models (including `scms`, `ncms` and `icms`), enforcing the module-wise update frequencies without applying complex momentum math.

### 2. Data Analysis & Visualization

To generate publication-ready artifacts from your experimental data, run:

```bash
python analyze.py

```

This script reads the exported JSON and CSVs to generate:

* **Summary Tables:** A formatted terminal printout comparing Average Accuracy and Forgetting across all runs.
* **Heatmaps:** $T \times T$ Class-IL and Task-IL matrix visualizations showcasing representational retention and "Softmax Suppression".
* **Line Graphs:** Epoch-by-epoch learning dynamics to visually showcase the effect of forgetting and learning.
* *Plots are saved directly to `data/results/plots/`.*

## To-Do List
- [x] Implement `multi-scale Adam` (`MAdam`) and `multi-scale SGD` (`MSGD`).
- [x] Decouple the update frequency of the outer loop of multi-scale optimizers from the `cms` models. Making the standard optimizers perform differently on `cms` models and `baseline` models.
- [x] Add implementation for different kinds of `cms` models mentioned in the original research paper.
- [x] Add more options to `main.py` to enable script-based hyperparameter sweep test.
- [ ] Make sure the implementations are all correct.