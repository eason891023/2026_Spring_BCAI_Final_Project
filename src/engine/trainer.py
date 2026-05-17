import torch
import torch.nn as nn
from .metrics import CLEvaluator
from src.optimizers.factory import get_optimizer

def train_cl_scenario(model, tasks_train, tasks_test, device, opt_name='SGD', epochs=5, lr=1e-3, f=20):
    """Executes the continual learning loop across all tasks, evaluating both CIL and TIL."""
    model = model.to(device)
    optimizer = get_optimizer(model, opt_name, lr=lr, f=f)
    criterion = nn.CrossEntropyLoss()
    
    num_tasks = len(tasks_train)
    
    # Instantiate dual evaluators
    evaluator_cil = CLEvaluator(num_tasks=num_tasks)
    evaluator_til = CLEvaluator(num_tasks=num_tasks)
    
    for task_id in range(num_tasks):
        print(f"\n[ Task {task_id + 1}/{num_tasks} | Optimizer: {opt_name} ]")
        train_loader = tasks_train[task_id]
        
        # --- Training Phase ---
        model.train()
        for epoch in range(epochs):
            model.zero_grad()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
        # --- Evaluation Phase (Dual-Tracking) ---
        model.eval()
        with torch.no_grad():
            for eval_id in range(task_id + 1):
                test_loader = tasks_test[eval_id]
                correct_cil, correct_til, total = 0, 0, 0
                
                # Determine valid classes for Task-IL masking
                valid_classes = [eval_id * 2, eval_id * 2 + 1]
                
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    
                    # 1. Class-IL Prediction (Argmax over all 10 classes)
                    pred_cil = output.argmax(dim=1, keepdim=True)
                    correct_cil += pred_cil.eq(target.view_as(pred_cil)).sum().item()
                    
                    # 2. Task-IL Prediction (Mask invalid classes, then argmax)
                    mask = torch.full_like(output, float('-inf'))
                    mask[:, valid_classes] = output[:, valid_classes]
                    pred_til = mask.argmax(dim=1, keepdim=True)
                    correct_til += pred_til.eq(target.view_as(pred_til)).sum().item()
                    
                    total += target.size(0)
                
                acc_cil = correct_cil / total
                acc_til = correct_til / total
                
                evaluator_cil.update_matrix(task_id, eval_id, acc_cil)
                evaluator_til.update_matrix(task_id, eval_id, acc_til)
                
                print(f"  -> Eval on Task {eval_id + 1} | CIL: {acc_cil:.4f} | TIL: {acc_til:.4f}")

    return {
        'CIL': evaluator_cil.compute_metrics(),
        'TIL': evaluator_til.compute_metrics()
    }