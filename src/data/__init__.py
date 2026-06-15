from .split_mnist import get_split_mnist
from .permuted_mnist import get_permuted_mnist
from .rotating_mnist import get_rotating_mnist
from .pacs import get_pacs

def get_dataset(name, batch_size=64, num_tasks=10, seed=42, angle_step=15, data_dir='./data'):
    """
    Unified dataset entry point.

    Returns:
        tasks_train, tasks_test: lists of DataLoaders, one per task.
        task_classes: per-task class lists for Task-IL masking (Split-MNIST),
                      or None for Domain-IL datasets where the label space is
                      shared and TIL masking is meaningless.
        scenario: 'CIL' or 'DIL', recorded into metrics for downstream analysis.
    """
    if name == 'split':
        tasks_train, tasks_test = get_split_mnist(batch_size=batch_size, data_dir=data_dir)
        task_classes = [[t * 2, t * 2 + 1] for t in range(len(tasks_train))]
        return tasks_train, tasks_test, task_classes, 'CIL', {"input_size": 784, "num_classes": 10}
    elif name == 'permuted':
        tasks_train, tasks_test = get_permuted_mnist(
            num_tasks=num_tasks, batch_size=batch_size, data_dir=data_dir, seed=seed)
        return tasks_train, tasks_test, None, 'DIL', {"input_size": 784, "num_classes": 10}
    elif name == 'rotating':
        tasks_train, tasks_test = get_rotating_mnist(
            num_tasks=num_tasks, batch_size=batch_size, data_dir=data_dir,
            angle_step=angle_step, seed=seed)
        return tasks_train, tasks_test, None, 'DIL', {"input_size": 784, "num_classes": 10}
    elif name == 'pacs':
        tasks_train, tasks_test, info = get_pacs(
            batch_size=batch_size, data_dir=data_dir, seed=seed)
        return tasks_train, tasks_test, None, 'DIL', info
    else:
        raise ValueError(f"Unsupported dataset: {name}")
