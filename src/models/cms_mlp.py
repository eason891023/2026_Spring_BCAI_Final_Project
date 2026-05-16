import torch.nn as nn

class CMS_MLP(nn.Module):
    """
    Continuum Memory System MLP.
    This implementation implements fast memory and slow memory, 
    and an isolated classification head (fast) to prevent orthogonal wipeout.
    Note that the fast memory and slow memory are divided within layer, not
    multiple MLP layers as described in the original NL research.
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        
        # Fast updating layers (Shallow features)
        self.fast_memory = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        # Medium updating layers (Medium persistent features)
        self.medium_memory = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Slow updating layers (Deep persistent features)
        self.slow_memory = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        # Fast / Standard updating classification head
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        fast_features = self.fast_memory(x)
        medium_features = self.medium_memory(fast_features)
        slow_features = self.slow_memory(medium_features)
        output = self.head(slow_features)
        return output