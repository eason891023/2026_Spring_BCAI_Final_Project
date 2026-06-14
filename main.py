import argparse
import random
import torch
import numpy as np
import os
import json
from src.data import get_dataset
from src.models.baseline import BaselineMLP
from src.models.cms_mlp import SCMS_MLP, NCMS_MLP, ICMS_MLP
from src.engine.trainer import train_cl_scenario

def parse_args():
    parser = argparse.ArgumentParser(description="Nested Learning Continual Learning Ablation")
    parser.add_argument('--config', type=str, default='', help="Path to YAML config file")
    parser.add_argument('--dataset', type=str, default='split', help="CL scenario: split (Class-IL), permuted/rotating (Domain-IL)")
    parser.add_argument('--num_tasks', type=int, default=10, help="Number of tasks for permuted/rotating (split is fixed at 5)")
    parser.add_argument('--angle_step', type=int, default=15, help="[rotating] Rotation angle increment per task (degrees)")
    parser.add_argument('--seed', type=int, default=42, help="Random seed (also derives the permutation sequence)")
    parser.add_argument('--model', type=str, default='baseline', help="Architecture to test")
    parser.add_argument('--optimizer', type=str, default='SGD', help="Optimizer to use")
    parser.add_argument('--epochs', type=int, default=5, help="Epochs per task")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--f', type=int, default=20, help="f for number of inner loop before M3 update outer loop")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha multiplier for slow memory")
    parser.add_argument('--beta3', type=float, default=0.9, help="Beta3 EMA rate for slow memory")
    parser.add_argument('--stab', type=int, default=1, help="Stabilized Version of multiscale optimizer or not.")
    parser.add_argument('--meta_lr', type=float, default=0.5, help="[ncms] Reptile step size for the meta-learned inits (Eq. 72 first-order approx)")
    parser.add_argument('--medium_period', type=int, default=2, help="[ncms] Reset the medium level every N contexts (tasks)")
    parser.add_argument('--reset_mode', type=str, default='meta', choices=['meta', 'random', 'none'], help="[ncms] What to reset fast levels to at context boundaries")
    parser.add_argument('--save_ckpt', type=int, default=1, help="Save model state at each task boundary (needed for probing/relearning analyses)")
    parser.add_argument('--eval_every', type=int, default=200, help="Evaluate every N steps")
    
    args, _ = parser.parse_known_args()
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        if config:
            parser.set_defaults(**config)
            
    return parser.parse_args()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def print_metrics(title, metrics):
    print("\n" + "="*50)
    print(f"FINAL EVALUATION METRICS ({title})")
    print("="*50)
    print("Accuracy Matrix (R):")
    print(metrics['Accuracy_Matrix'])
    print(f"\nAverage Accuracy: {metrics['Average_ACC']:.4f}")
    print(f"Average Forgetting: {metrics['Forgetting']:.4f}")
    print(f"Backward Transfer (BWT): {metrics['BWT']:.4f}")

def run_experiment(args, device):
    set_seed(args.seed)
    
    print(f"\n{'='*60}")
    print(f"Starting Experiment: Dataset={args.dataset} | Model={args.model} | Opt={args.optimizer}")
    print(f"{'='*60}")


    # 1. Load Data
    tasks_train, tasks_test, task_classes, scenario = get_dataset(
        args.dataset,
        batch_size=args.batch_size,
        num_tasks=args.num_tasks,
        seed=args.seed,
        angle_step=args.angle_step
    )
    print(f"Dataset: {args.dataset} ({scenario}) | Tasks: {len(tasks_train)}")

    # 2. Initialize Architecture
    if args.model == 'baseline':
        model = BaselineMLP()
    elif args.model == 'scms':
        model = SCMS_MLP()
    elif args.model == 'ncms':
        model = NCMS_MLP(meta_lr=args.meta_lr, medium_period=args.medium_period, reset_mode=args.reset_mode)
    elif args.model == 'icms':
        model = ICMS_MLP()
    else:
        raise NotImplementedError("Currently Not implemented to support other model architectures!")

    # Construct a unique prefix for the files
    file_prefix = f"{args.dataset}_{args.model}_{args.optimizer}_f{args.f}_a{args.alpha}_b{args.beta3}_s{args.stab}_sd{args.seed}"
    if args.model == 'ncms':
        file_prefix += f"_r{args.reset_mode}_ml{args.meta_lr}_mp{args.medium_period}"

    ckpt_dir = os.path.join("data", "results", "checkpoints", file_prefix) if args.save_ckpt else None

    # 3. Execute Scenario
    results = train_cl_scenario(
        model=model,
        tasks_train=tasks_train,
        tasks_test=tasks_test,
        device=device,
        opt_name=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        f=args.f,
        alpha=args.alpha,
        beta3=args.beta3,
        stab=bool(args.stab),
        task_classes=task_classes,
        ckpt_dir=ckpt_dir,
        eval_every=args.eval_every
    )

    # 4. Report Metrics
    np.set_printoptions(precision=4, suppress=True)
    print_metrics("CLASS-IL", results['CIL'])
    if results['TIL'] is not None:
        print_metrics("TASK-IL", results['TIL'])
    print("="*50)

    # 5. Automated Data Export
    metrics_dir = os.path.join("data", "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    # Export Standard Matrices to CSV (For Heatmaps and Summaries)
    results['evaluator_cil'].export_matrix_to_csv(os.path.join(metrics_dir, f"{file_prefix}_CIL.csv"))
    # Export High-Resolution History to CSV (For Line Graphs)
    results['evaluator_cil'].export_history_to_csv(os.path.join(metrics_dir, f"{file_prefix}_CIL_history.csv"))

    if results['evaluator_til'] is not None:
        results['evaluator_til'].export_matrix_to_csv(os.path.join(metrics_dir, f"{file_prefix}_TIL.csv"))
        results['evaluator_til'].export_history_to_csv(os.path.join(metrics_dir, f"{file_prefix}_TIL_history.csv"))

    # Compile Summary Record
    summary_record = {
        "dataset": args.dataset,
        "scenario": scenario,
        "num_tasks": len(tasks_train),
        "seed": args.seed,
        "model": args.model,
        "optimizer": args.optimizer,
        "f": args.f,
        "stabilized": bool(args.stab),
        "alpha": args.alpha,
        "beta3": args.beta3,
        "epochs": args.epochs,
        "eval_every": args.eval_every,
        "steps_per_epoch": results['steps_per_epoch'],
        "lr": args.lr,
        "ncms": {"reset_mode": args.reset_mode, "meta_lr": args.meta_lr, "medium_period": args.medium_period} if args.model == 'ncms' else None,
        "CIL": results['evaluator_cil'].export_summary_dict(),
        "TIL": results['evaluator_til'].export_summary_dict() if results['evaluator_til'] is not None else None
    }

    # Safely append to the JSON file
    json_path = os.path.join(metrics_dir, "summary_metrics.json")
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            try:
                data_log = json.load(f)
            except json.JSONDecodeError:
                data_log = []
    else:
        data_log = []

    data_log.append(summary_record)

    with open(json_path, 'w') as f:
        json.dump(data_log, f, indent=4)

    print(f"\n[INFO] Data successfully exported to {metrics_dir}/")
    if ckpt_dir:
        print(f"[INFO] Task-boundary checkpoints saved to {ckpt_dir}/")

def main():
    args = parse_args()
    
    # Dynamic device selection (supports CUDA and macOS MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Native Parameter Sweep on device: {device}")
    
    # Expand list parameters into a grid
    import itertools
    
    args_dict = vars(args)
    grid_params = {}
    
    for key, value in args_dict.items():
        if key == 'config': 
            continue
        # If the parameter is a list, keep it. Otherwise, wrap it in a list.
        if isinstance(value, list):
            grid_params[key] = value
        else:
            grid_params[key] = [value]
            
    # Generate all combinations (Cartesian product)
    keys = list(grid_params.keys())
    values = list(grid_params.values())
    combinations = list(itertools.product(*values))
    
    print(f"Detected {len(combinations)} total experiment configurations to run.")
    
    # Execute each experiment sequentially
    for combo in combinations:
        exp_args_dict = dict(zip(keys, combo))
        exp_args = argparse.Namespace(**exp_args_dict)
        run_experiment(exp_args, device)
        
    print("\nAll experiments completed successfully!")

if __name__ == "__main__":
    main()
