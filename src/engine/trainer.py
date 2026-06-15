import os
import torch
import torch.nn as nn
from .metrics import CLEvaluator
from src.optimizers.factory import get_optimizer

def train_cl_scenario(model, tasks_train, tasks_test, device, opt_name='SGD', epochs=5, lr=1e-3,
                      f=20, alpha=0.5, beta3=0.9, stab=True, task_classes=None, ckpt_dir=None,
                      eval_every=200, sg_tau=2.0, sg_tmin=1, sg_tmax=None,
                      sg_ema_rho=0.05, sg_warmup=20):
    """Executes the continual learning loop across all tasks.

    task_classes: per-task class lists used for the Task-IL masked evaluation
        (e.g. [[0,1],[2,3],...] for Split-MNIST). Pass None for Domain-IL
        datasets (Permuted/Rotating-MNIST) where the label space is shared and
        TIL masking is meaningless -- TIL evaluation is skipped entirely.
    ckpt_dir: if given, the model state_dict is saved there after each task's
        training phase (before any context reset), enabling offline probing
        and relearning analyses without retraining.
    """
    model = model.to(device)
    optimizer = get_optimizer(
        model, opt_name, lr=lr, f=f, stabilize=stab, alpha=alpha, beta3=beta3,
        sg_tau=sg_tau, sg_tmin=sg_tmin, sg_tmax=sg_tmax,
        sg_ema_rho=sg_ema_rho, sg_warmup=sg_warmup)
    criterion = nn.CrossEntropyLoss()

    num_tasks = len(tasks_train)
    eval_til = task_classes is not None
    evaluator_cil = CLEvaluator(num_tasks=num_tasks)
    evaluator_til = CLEvaluator(num_tasks=num_tasks) if eval_til else None

    if ckpt_dir is not None:
        os.makedirs(ckpt_dir, exist_ok=True)

    global_step = 0 # Tracks absolute gradient steps across all task transitions

    steps_per_epoch = 0

    for task_id in range(num_tasks):
        train_loader = tasks_train[task_id]
        steps_per_epoch = len(train_loader)

        print(f"\n[ Task {task_id + 1}/{num_tasks} | Optimizer: {opt_name} | Steps/Epoch: {steps_per_epoch} ]")

        def evaluate_and_log(is_final_epoch):
            model.eval()
            with torch.no_grad():
                for eval_id in range(task_id + 1):
                    test_loader = tasks_test[eval_id]
                    correct_cil, correct_til, total = 0, 0, 0

                    valid_classes = task_classes[eval_id] if eval_til else None

                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)

                        # 1. Class-IL Prediction
                        pred_cil = output.argmax(dim=1, keepdim=True)
                        correct_cil += pred_cil.eq(target.view_as(pred_cil)).sum().item()

                        # 2. Task-IL Prediction
                        if eval_til:
                            mask = torch.full_like(output, float('-inf'))
                            mask[:, valid_classes] = output[:, valid_classes]
                            pred_til = mask.argmax(dim=1, keepdim=True)
                            correct_til += pred_til.eq(target.view_as(pred_til)).sum().item()

                        total += target.size(0)

                    acc_cil = correct_cil / total
                    acc_til = correct_til / total if eval_til else None

                    # Log high-resolution data EVERY evaluation step
                    evaluator_cil.update_history(global_step, eval_id, acc_cil)
                    if eval_til:
                        evaluator_til.update_history(global_step, eval_id, acc_til)

                    # Log standard matrix data ONLY on the final epoch of the task
                    if is_final_epoch:
                        evaluator_cil.update_matrix(task_id, eval_id, acc_cil)
                        if eval_til:
                            evaluator_til.update_matrix(task_id, eval_id, acc_til)
                            print(f"  -> [Task Boundary] Eval on Task {eval_id + 1} | CIL: {acc_cil:.4f} | TIL: {acc_til:.4f}")
                        else:
                            print(f"  -> [Task Boundary] Eval on Task {eval_id + 1} | CIL: {acc_cil:.4f}")
            model.train()

        # --- Training Phase ---
        for epoch in range(epochs):
            model.train() # Make sure to set train mode inside the epoch loop

            for data, target in train_loader:
                global_step += 1
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                # Check for step-based evaluation
                if global_step % eval_every == 0:
                    evaluate_and_log(is_final_epoch=False)

            # At the end of the epoch, trigger final evaluation matrix if it's the last epoch
            if epoch == epochs - 1:
                if hasattr(optimizer, 'flush'):
                    optimizer.flush()
                evaluate_and_log(is_final_epoch=True)

        # Snapshot the adapted state at the task boundary BEFORE any context
        # reset, so offline analyses (probing, relearning) see the trained weights.
        if ckpt_dir is not None:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"task{task_id + 1}.pt"))

        # Nested CMS context boundary (Eq. 72): meta-update the learned inits
        # and reset the fast levels before the next task starts. Skipped after
        # the final task so the returned model keeps its adapted weights.
        if task_id < num_tasks - 1 and hasattr(model, 'end_context'):
            model.end_context()

    return {
        'CIL': evaluator_cil.compute_metrics(),
        'TIL': evaluator_til.compute_metrics() if eval_til else None,
        'evaluator_cil': evaluator_cil,
        'evaluator_til': evaluator_til,
        'steps_per_epoch': steps_per_epoch,
        'optimizer_events': optimizer.get_event_log() if hasattr(optimizer, 'get_event_log') else None
    }
