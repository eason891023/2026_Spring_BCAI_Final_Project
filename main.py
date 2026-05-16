import argparse
import torch
import numpy as np
from src.data.split_mnist import get_split_mnist
from src.models.baseline import BaselineMLP
from src.models.cms_mlp import CMS_MLP
from src.engine.trainer import train_cl_scenario

def parse_args():
    parser = argparse.ArgumentParser(description="Nested Learning Continual Learning Ablation")
    parser.add_argument('--model', type=str, default='baseline', choices=['baseline', 'cms'], help="Architecture to test")
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam', 'Muon', 'M3', 'SM3', 'MSGD', 'MAdam'], help="Optimizer to use")
    parser.add_argument('--epochs', type=int, default=5, help="Epochs per task")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--f', type=float, default=20, help="f for number of inner loop before M3 update outer loop")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Dynamic device selection (supports CUDA and macOS MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Experiment on device: {device}")
    
    # 1. Load Data
    tasks_train, tasks_test = get_split_mnist(batch_size=args.batch_size)
    
    # 2. Initialize Architecture
    if args.model == 'baseline':
        model = BaselineMLP()
    elif args.model == 'cms':
        model = CMS_MLP()
        
    # 3. Execute Scenario
    results = train_cl_scenario(
        model=model,
        tasks_train=tasks_train,
        tasks_test=tasks_test,
        device=device,
        opt_name=args.optimizer,
        epochs=args.epochs,
        lr=args.lr,
        f=args.f
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

if __name__ == "__main__":
    main()