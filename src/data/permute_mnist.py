import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class PermutedMNISTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, perm):
        self.dataset = dataset
        self.perm = perm
        
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # x is originally [1, 28, 28]
        # view(-1) flattens to [784]
        # [self.perm] applies permutation
        # view(1, 28, 28) reshapes back
        x_perm = x.view(-1)[self.perm].view(1, 28, 28)
        return x_perm, y

def get_permuted_mnist(batch_size=64, num_tasks=5, data_dir='./data', seed=42):
    """
    Generates Domain-Incremental Learning (DIL) tasks via Permuted MNIST.
    Every task contains all 10 classes, but the input pixels are randomly permuted.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    tasks_train, tasks_test, task_classes = [], [], []
    rng = np.random.default_rng(seed)
    
    for t in range(num_tasks):
        if t == 0:
            perm = np.arange(28 * 28)
        else:
            perm = rng.permutation(28 * 28)
            
        train_loader = DataLoader(
            PermutedMNISTDataset(train_dataset, perm),
            batch_size=batch_size,
            shuffle=True
        )
        
        test_loader = DataLoader(
            PermutedMNISTDataset(test_dataset, perm),
            batch_size=batch_size,
            shuffle=False
        )
        
        tasks_train.append(train_loader)
        tasks_test.append(test_loader)
        
        # In DIL, every task contains all 10 classes
        task_classes.append(list(range(10)))
        
    return tasks_train, tasks_test, task_classes
