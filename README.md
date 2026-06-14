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
├── scripts/
│   └── run_baseline_grid.sh # Phase 0 baseline sweep (datasets x models x optimizers x seeds)
├── data/                   
│   └── results/            # Auto-generated outputs
│       ├── raw_log/        # Terminal stdout records
│       ├── metrics/        # epochwise-recorded CSVs and JSON summaries
│       ├── checkpoints/    # Task-boundary model snapshots (for probing/relearning analyses)
│       └── plots/          # Charts and heatmaps
└── src/                    # Main source code directory
    ├── __init__.py
    ├── data/               # Data ingestion and task splitting
    │   ├── __init__.py     # get_dataset(): unified entry point (returns task class maps + scenario)
    │   ├── split_mnist.py    # Split-MNIST   (Class-IL:  5 tasks, 2 classes each)
    │   ├── permuted_mnist.py # Permuted-MNIST (Domain-IL: fixed pixel permutation per task)
    │   └── rotating_mnist.py # Rotating-MNIST (Domain-IL: gradual rotation drift per task)
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
python main.py --dataset ["split", "permuted", "rotating"] --model ["scms", "icms", "ncms", "baseline"] --optimizer ['SGD', 'Adam', 'Muon', 'M3', 'M3S', 'MSGD', 'MAdam']
```

> Please choose only one option at once from the `[...]` in the options listed above! E.g.: `python main.py --dataset split --model baseline --optimizer SGD`

**Universal Core Options:**
* `--config`: Path to a YAML configuration file. Supports native parameter sweeping if you define parameters as lists (e.g., `seed: [42, 43]`).
* `--dataset`: The CL scenario. `split` (Split-MNIST, Class-IL), `permuted` (Permuted-MNIST, Domain-IL), or `rotating` (Rotating-MNIST, gradual Domain-IL). Default: `split`. *(Domain-IL runs skip the Task-IL masked evaluation, since the label space is shared across tasks.)*
* `--num_tasks`: Number of tasks for `permuted` / `rotating` (`split` is fixed at 5). Default: `10`.
* `--angle_step`: [rotating] Rotation increment per task in degrees. Default: `15`.
* `--seed`: Random seed; also derives the permutation sequence for `permuted`. Default: `42`.
* `--save_ckpt`: Save a model snapshot at every task boundary to `data/results/checkpoints/` (required by downstream probing/relearning analyses). Default: `1`.
* `--model`: The network architecture. Choose `baseline` (Standard monolithic MLP) or `cms` (including three variants: `scms`, `ncms`, and `icms`).
* `--optimizer`: The optimization engine. Supports standard (`SGD`, `Adam`, `Muon`) and Multi-scale (`M3`, `M3S`, `MSGD`, `MAdam`) algorithms.
* `--epochs`: Training epochs per task. Default: `5`.
* `--eval_every`: Evaluate the model every N steps, enabling high-resolution tracking of catastrophic forgetting mid-epoch. Default: `200`.
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

### 2. Batch Baseline Sweeps

```bash
bash scripts/run_baseline_grid.sh "42 123 7"
```

Runs the full Phase 0 grid ({split, permuted} × {baseline, scms, ncms, icms} × {SGD, Adam, M3, M3S}) for each given seed, logging stdout to `data/results/raw_log/`.

### 3. Native YAML Parameter Sweeping

You can natively execute massive ablation studies directly in `main.py` using a YAML configuration file. 
If you define parameters as lists within the YAML file, the script will automatically calculate the Cartesian product of all parameters and run them sequentially.

```bash
python main.py --config configs/dil_experiment.yaml
```

**Example YAML:**
```yaml
dataset: 
  - "permuted"
model: 
  - "baseline"
  - "ncms"
optimizer: 
  - "SGD"
  - "Adam"
seed:
  - 42
  - 43
```
*This config will automatically execute an 8-run grid search!*

### 4. Data Analysis & Visualization

To generate publication-ready artifacts from your experimental data, run:

```bash
python analyze.py

```

This script reads the exported JSON and CSVs to generate:

* **Summary Tables:** A formatted terminal printout comparing Average Accuracy and Forgetting across all runs.
* **Heatmaps:** $T \times T$ Class-IL and Task-IL matrix visualizations showcasing representational retention and "Softmax Suppression".
* **Line Graphs:** High-resolution Step-Level learning dynamics, including an auto-smoothed variant, to showcase the exact moment forgetting and learning occurs mid-epoch.
* *Plots are saved directly to `data/results/plots/`.*

## To-Do List
- [x] Implement `multi-scale Adam` (`MAdam`) and `multi-scale SGD` (`MSGD`).
- [x] Decouple the update frequency of the outer loop of multi-scale optimizers from the `cms` models. Making the standard optimizers perform differently on `cms` models and `baseline` models.
- [x] Add implementation for different kinds of `cms` models mentioned in the original research paper.
- [x] Add more options to `main.py` to enable script-based hyperparameter sweep test.
- [ ] Make sure the implementations are all correct.