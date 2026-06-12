import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from .permuted_mnist import _load_flat_mnist

class _RotatedTensorDataset(Dataset):
    """Serves pre-normalized MNIST tensors rotated by a fixed per-task angle.

    Rotation happens per sample at access time so all tasks share one copy
    of the base tensors (precomputing every task would cost ~190MB each).
    """
    def __init__(self, images, labels, angle):
        self.images = images  # (N, 784) normalized
        self.labels = labels
        self.angle = angle

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx].view(1, 28, 28)
        if self.angle != 0:
            x = TF.rotate(x, self.angle)
        return x.view(784), self.labels[idx]

def get_rotating_mnist(num_tasks=10, batch_size=64, data_dir='./data', angle_step=15, seed=42):
    """
    Generates Rotating-MNIST (gradual Domain-IL): task t rotates inputs by
    t * angle_step degrees; the label space (0-9) is shared. Unlike Permuted,
    consecutive tasks are highly related, modeling slow distribution drift.
    `seed` is accepted for interface parity but the task sequence is
    deterministic given angle_step.
    """
    (train_x, train_y), (test_x, test_y) = _load_flat_mnist(data_dir)

    tasks_train, tasks_test = [], []
    for t in range(num_tasks):
        angle = t * angle_step
        tasks_train.append(DataLoader(
            _RotatedTensorDataset(train_x, train_y, angle),
            batch_size=batch_size,
            shuffle=True
        ))
        tasks_test.append(DataLoader(
            _RotatedTensorDataset(test_x, test_y, angle),
            batch_size=batch_size,
            shuffle=False
        ))

    return tasks_train, tasks_test
