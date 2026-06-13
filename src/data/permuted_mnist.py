import torch
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import numpy as np

class _PermutedTensorDataset(Dataset):
    """Serves pre-normalized flat MNIST tensors with a fixed pixel permutation.

    The base tensors are shared across tasks; only the permutation index
    differs, so memory stays at one copy of MNIST regardless of num_tasks.
    """
    def __init__(self, images, labels, permutation):
        self.images = images          # (N, 784) normalized
        self.labels = labels          # (N,)
        self.permutation = permutation

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx][self.permutation], self.labels[idx]

def _load_flat_mnist(data_dir):
    """Loads MNIST once into normalized flat tensors (skips PIL per-sample transforms)."""
    train = datasets.MNIST(data_dir, train=True, download=True)
    test = datasets.MNIST(data_dir, train=False, download=True)
    def prep(ds):
        x = ds.data.float().div(255.0).sub(0.1307).div(0.3081).view(-1, 784)
        return x, ds.targets
    return prep(train), prep(test)

def get_permuted_mnist(num_tasks=10, batch_size=64, data_dir='./data', seed=42):
    """
    Generates Permuted-MNIST (Domain-IL): each task applies a fixed random
    pixel permutation to the input; the label space (0-9) is shared.
    Task 0 uses the identity permutation; permutations are derived from `seed`
    so task sequences are reproducible across runs.
    """
    (train_x, train_y), (test_x, test_y) = _load_flat_mnist(data_dir)
    rng = np.random.RandomState(seed)

    tasks_train, tasks_test = [], []
    for t in range(num_tasks):
        if t == 0:
            perm = torch.arange(784)
        else:
            perm = torch.from_numpy(rng.permutation(784))

        tasks_train.append(DataLoader(
            _PermutedTensorDataset(train_x, train_y, perm),
            batch_size=batch_size,
            shuffle=True
        ))
        tasks_test.append(DataLoader(
            _PermutedTensorDataset(test_x, test_y, perm),
            batch_size=batch_size,
            shuffle=False
        ))

    return tasks_train, tasks_test
