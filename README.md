# Nested Learning Validation on MLP based Models
## Proposal
> TBD!

## Project Architecture
```plain
nested_learning_sandbox/
├── README.md               # Project overview and run instructions
├── requirements.txt        # Core dependencies
├── main.py                 # The CLI entry point for running specific ablation phases
└── src/                    # Main source code directory
    ├── __init__.py
    ├── data/               # Data ingestion and task splitting
    │   ├── __init__.py
    │   └── split_mnist.py
    ├── models/             # Network architectures
    │   ├── __init__.py
    │   ├── baseline.py     # Standard MLP
    │   └── cms_mlp.py      # Continuum Memory System MLP
    ├── optimizers/         # Optimizer definitions
    │   ├── __init__.py
    │   ├── factory.py      # Logic to instantiate the correct optimizer
    │   ├── m3.py           # Multi-timescale Momentum Muon
    │   └── muon.py         # Standard Muon implementation
    └── engine/             # Training loops and evaluation logic
        ├── __init__.py
        ├── metrics.py      # Accuracy Matrix (R), F, BWT calculations
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
> make sure to setup the environment first and activate the virtual environment before running the following commands.
```bash
python main.py --model ["cms", "baseline"] --optimizer ['SGD', 'Adam', 'Muon', 'M3', 'M3S', 'MSGD', 'MAdam']
```
> Please choose only one option at once from the `[...]` in the options listed above!
> 
> E.g.: `python main.py --model baseline --optimizer SGD` would be fine!

**Options:**
* `--model`: Currently only support `baseline` which is MLP, and `cms` which is MLP with layerwise definition (for different update frequency required by NL).
* `--optimizer`: Currently support all the above options. ~~Note that `SGD`, `Adam`, and `Muon` in current version works exactly the same on `baseline` models and `cms` models, any kind of performance differences should be coming from random initialization.~~ Fixed in this version!
* `--epochs`: Defines epoch per task. Default: `5`.
* `--batch_size`: Defines batch sizes. Default: `64`.
* `--lr`: Defines learning rate. Default: `1e-3` (Note that some optimizer might require higher or lower learning rate to perform best!)
* `--f`: Defines the number of loops for inner loop before M3/M3S update outer loop. Default: `20`. (Note that this parameter might have different impact depending on the batch size settings!)

> Note that when combining `cms` models with standard optimizers (e.g.: `SGD`, `Adam`, `Muon`) options, the optimizer will be decoupled version of the standard optimizers. Meaning that the weight update will be performed on different level of the MLP layer under different frequencies.

## To-Do List
- [x] Implement `multi-scale Adam` (`MAdam`) and `multi-scale SGD` (`MSGD`).
- [x] Decouple the update frequency of the outer loop of multi-scale optimizers from the `cms` models. Making the standard optimizers perform differently on `cms` models and `baseline` models.
- [ ] Make sure the implementations are all correct.