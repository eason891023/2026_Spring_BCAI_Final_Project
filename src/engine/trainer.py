import torch
import torch.nn as nn
from .metrics import CLEvaluator
from src.optimizers.factory import get_optimizer

def train_cl_scenario(model, tasks_train, tasks_test, task_classes, device, opt_name='SGD', epochs=5, lr=1e-3, f=20, alpha=0.5, beta3=0.9, stab=True):
    """Executes the continual learning loop across all tasks, evaluating both CIL and TIL."""
    model = model.to(device)
    optimizer = get_optimizer(model, opt_name, lr=lr, f=f, stabilize=stab, alpha=alpha, beta3=beta3)
    criterion = nn.CrossEntropyLoss()
    
    num_tasks = len(tasks_train)
    total_epochs = num_tasks * epochs # Calculate total epochs for the history matrix
    
    # Instantiate dual evaluators with the new total_epochs parameter
    evaluator_cil = CLEvaluator(num_tasks=num_tasks, total_epochs=total_epochs)
    evaluator_til = CLEvaluator(num_tasks=num_tasks, total_epochs=total_epochs)
    
    global_epoch = 0 # NEW: Tracks absolute time across all task transitions

    steps_per_epoch = 0
    
    for task_id in range(num_tasks):
        train_loader = tasks_train[task_id]
        steps_per_epoch = len(train_loader)

        classes_str = ", ".join(map(str, task_classes[task_id]))
        print(f"\n[ Task {task_id + 1}/{num_tasks} ({classes_str}) | Optimizer: {opt_name} | Steps/Epoch: {steps_per_epoch} ]")
        
        model.zero_grad()
        # --- Training Phase ---
        for epoch in range(epochs):
            model.train() # Make sure to set train mode inside the epoch loop
            
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
            # --- NEW: Evaluation Phase (Now runs every single epoch) ---
            model.eval()
            with torch.no_grad():
                for eval_id in range(task_id + 1):
                    test_loader = tasks_test[eval_id]
                    correct_cil, correct_til, total = 0, 0, 0
                    
                    valid_classes = task_classes[eval_id]
                    
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        
                        # 1. Class-IL Prediction
                        pred_cil = output.argmax(dim=1, keepdim=True)
                        correct_cil += pred_cil.eq(target.view_as(pred_cil)).sum().item()
                        
                        # 2. Task-IL Prediction
                        mask = torch.full_like(output, float('-inf'))
                        mask[:, valid_classes] = output[:, valid_classes]
                        pred_til = mask.argmax(dim=1, keepdim=True)
                        correct_til += pred_til.eq(target.view_as(pred_til)).sum().item()
                        
                        total += target.size(0)
                    
                    acc_cil = correct_cil / total
                    acc_til = correct_til / total
                    
                    # Log high-resolution data EVERY epoch
                    evaluator_cil.update_history(global_epoch, eval_id, acc_cil)
                    evaluator_til.update_history(global_epoch, eval_id, acc_til)
                    
                    # Log standard matrix data ONLY on the final epoch of the task
                    if epoch == epochs - 1:
                        evaluator_cil.update_matrix(task_id, eval_id, acc_cil)
                        evaluator_til.update_matrix(task_id, eval_id, acc_til)
                        print(f"  -> [Task Boundary] Eval on Task {eval_id + 1} | CIL: {acc_cil:.4f} | TIL: {acc_til:.4f}")

            # Advance absolute time
            global_epoch += 1

    return {
        'CIL': evaluator_cil.compute_metrics(),
        'TIL': evaluator_til.compute_metrics(),
        'evaluator_cil': evaluator_cil,
        'evaluator_til': evaluator_til,
        'steps_per_epoch': steps_per_epoch
    }