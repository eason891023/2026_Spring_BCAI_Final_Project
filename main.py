import argparse
import torch
import numpy as np
import os
import json
from src.data.split_mnist import get_split_mnist
from src.data.permute_mnist import get_permuted_mnist
from src.models.baseline import BaselineMLP
from src.models.cms_mlp import SCMS_MLP, NCMS_MLP, ICMS_MLP
from src.engine.trainer import train_cl_scenario

def parse_args():
    parser = argparse.ArgumentParser(description="Nested Learning Continual Learning Ablation")
    parser.add_argument('--dataset', type=str, default='split_mnist', choices=['split_mnist', 'permuted_mnist'], help="Dataset scenario to run")
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'scms', 'ncms', 'icms'], help="Architecture to test")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'Muon', 'M3', 'M3S', 'MSGD', 'MAdam'], help="Optimizer to use")
    parser.add_argument('--epochs', type=int, default=5, help="Epochs per task")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--f', type=int, default=20, help="f for number of inner loop before M3 update outer loop")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha multiplier for slow memory")
    parser.add_argument('--beta3', type=float, default=0.9, help="Beta3 EMA rate for slow memory")
    parser.add_argument('--stab', type=int, default=1, help="Stabilized Version of multiscale optimizer or not.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamic device selection (supports CUDA and macOS MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Experiment on device: {device}")
    
    # 1. Load Data
    if args.dataset == 'split_mnist':
        tasks_train, tasks_test, task_classes = get_split_mnist(batch_size=args.batch_size)
    elif args.dataset == 'permuted_mnist':
        tasks_train, tasks_test, task_classes = get_permuted_mnist(batch_size=args.batch_size)
    else:
        raise ValueError("Unknown dataset")
    
    # 2. Initialize Architecture
    if args.model == 'baseline':
        model = BaselineMLP()
    elif args.model == 'scms':
        model = SCMS_MLP()
    elif args.model == 'ncms':
        model = NCMS_MLP()
    elif args.model == 'icms':
        model = ICMS_MLP()
    else:
        raise NotImplementedError("Currently Not implemented to support other model architectures!")
        
    # 3. Execute Scenario
    results = train_cl_scenario(
        model=model,
        tasks_train=tasks_train,
        tasks_test=tasks_test,
        task_classes=task_classes,
        device=device,
        opt_name=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        f=args.f,
        alpha=args.alpha,
        beta3=args.beta3,
        stab=bool(args.stab)
    )
    
    # 4. Report Metrics
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS (CLASS-IL)")
    print("="*50)
    np.set_printoptions(precision=4, suppress=True)
    print("Accuracy Matrix (R):")
    print(results['CIL']['Accuracy_Matrix'])
    print(f"\nAverage Accuracy: {results['CIL']['Average_ACC']:.4f}")
    print(f"Average Forgetting: {results['CIL']['Forgetting']:.4f}")
    print(f"Backward Transfer (BWT): {results['CIL']['BWT']:.4f}")
    
    print("\n" + "="*50)
    print("FINAL EVALUATION METRICS (TASK-IL)")
    print("="*50)
    print("Accuracy Matrix (R):")
    print(results['TIL']['Accuracy_Matrix'])
    print(f"\nAverage Accuracy: {results['TIL']['Average_ACC']:.4f}")
    print(f"Average Forgetting: {results['TIL']['Forgetting']:.4f}")
    print(f"Backward Transfer (BWT): {results['TIL']['BWT']:.4f}")
    print("="*50)

    # 5. Automated Data Export
    metrics_dir = os.path.join("data", "results", "metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Construct a unique prefix for the files
    file_prefix = f"{args.dataset}_{args.model}_{args.optimizer}_f{args.f}_a{args.alpha}_b{args.beta3}_s{args.stab}"
    
    # Export Standard Matrices to CSV (For Heatmaps and Summaries)
    results['evaluator_cil'].export_matrix_to_csv(os.path.join(metrics_dir, f"{file_prefix}_CIL.csv"))
    results['evaluator_til'].export_matrix_to_csv(os.path.join(metrics_dir, f"{file_prefix}_TIL.csv"))
    
    # Export High-Resolution History to CSV (For Line Graphs)
    results['evaluator_cil'].export_history_to_csv(os.path.join(metrics_dir, f"{file_prefix}_CIL_history.csv"))
    results['evaluator_til'].export_history_to_csv(os.path.join(metrics_dir, f"{file_prefix}_TIL_history.csv"))
    
    # Compile Summary Record
    summary_record = {
        "dataset": args.dataset,
        "model": args.model,
        "optimizer": args.optimizer,
        "f": args.f,
        "stabilized": bool(args.stab),
        "alpha": args.alpha,
        "beta3": args.beta3,
        "epochs": args.epochs,
        "steps_per_epoch": results['steps_per_epoch'],
        "lr": args.lr,
        "CIL": results['evaluator_cil'].export_summary_dict(),
        "TIL": results['evaluator_til'].export_summary_dict()
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

if __name__ == "__main__":
    main()