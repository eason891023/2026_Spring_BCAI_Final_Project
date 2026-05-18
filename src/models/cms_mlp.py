import torch
import torch.nn as nn

class SCMS_MLP(nn.Module):
    """
    Sequential Continuum Memory System MLP.
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

class NCMS_MLP(nn.Module):
    """
    Nested Continuum Memory System MLP.
    Splits feature representations spatially (by width) rather than by depth.
    Mimics biological ensembles with varying parallel plasticity rates.
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Split the hidden dimension into temporal ensembles
        f_size = hidden_size // 4       # e.g., 64 (Fast Working Memory)
        m_size = hidden_size // 4       # e.g., 64  (Medium Memory)
        s_size = hidden_size - f_size - m_size # e.g., 128 (Slow Continuum Memory)
        
        # Parallel processing pathways observing the same input
        self.fast_memory = nn.Sequential(
            nn.Linear(input_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, f_size),
            nn.ReLU()
        )
        
        self.medium_memory = nn.Sequential(
            nn.Linear(input_size, m_size),
            nn.ReLU(),
            nn.Linear(m_size, m_size),
            nn.ReLU(),
            nn.Linear(m_size, m_size),
            nn.ReLU()
        )
        
        self.slow_memory = nn.Sequential(
            nn.Linear(input_size, s_size),
            nn.ReLU(),
            nn.Linear(s_size, s_size),
            nn.ReLU(),
            nn.Linear(s_size, s_size),
            nn.ReLU()
        )
        
        # The head observes all temporal scales simultaneously
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        
        # Extract features at different temporal plasticity scales
        fast_feat = self.fast_memory(x)
        med_feat = self.medium_memory(x)
        slow_feat = self.slow_memory(x)
        
        # Concatenate spatially (Width-based nesting)
        combined = torch.cat([fast_feat, med_feat, slow_feat], dim=1)
        
        return self.head(combined)

class ICMS_MLP(nn.Module):
    """
    Independent (Head-wise) Continuum Memory System.
    Each temporal ensemble (Fast, Med, Slow) maintains its own independent 
    classification head. The final prediction is a summation (ensemble vote) 
    of all temporal scales.
    """
    def __init__(self, input_size=784, hidden_size=256, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        
        # Split the hidden dimension into temporal ensembles
        f_size = hidden_size // 4       
        m_size = hidden_size // 4       
        s_size = hidden_size - f_size - m_size 
        
        # Parallel processing pathways
        self.fast_memory = nn.Sequential(
            nn.Linear(input_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, f_size),
            nn.ReLU(),
            nn.Linear(f_size, f_size),
            nn.ReLU()
        )
        
        self.medium_memory = nn.Sequential(
            nn.Linear(input_size, m_size),
            nn.ReLU(),
            nn.Linear(m_size, m_size),
            nn.ReLU(),
            nn.Linear(m_size, m_size),
            nn.ReLU()
        )
        
        self.slow_memory = nn.Sequential(
            nn.Linear(input_size, s_size),
            nn.ReLU(),
            nn.Linear(s_size, s_size),
            nn.ReLU(),
            nn.Linear(s_size, s_size),
            nn.ReLU()
        )
        
        # Independent Heads grouped in a ModuleDict so factory.py can find them all at once
        self.head = nn.ModuleDict({
            'fast': nn.Linear(f_size, num_classes),
            'medium': nn.Linear(m_size, num_classes),
            'slow': nn.Linear(s_size, num_classes)
        })

    def forward(self, x):
        x = self.flatten(x)
        
        # Extract features and compute independent logits
        fast_logits = self.head['fast'](self.fast_memory(x))
        med_logits = self.head['medium'](self.medium_memory(x))
        slow_logits = self.head['slow'](self.slow_memory(x))
        
        # Ensemble vote (Summing the logits)
        return fast_logits + med_logits + slow_logits