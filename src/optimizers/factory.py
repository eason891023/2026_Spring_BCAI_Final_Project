import torch.optim as optim
from .m3 import M3
from .muon import Muon

class DecoupledOptimizer:
    """
    Wraps standard PyTorch optimizers to enable multi-timescale gradient accumulation.
    Isolates the 'Temporal Split' from custom optimizer math.
    """
    def __init__(self, param_groups, opt_class, lr):
        self.groups = []
        for group in param_groups:
            # Create a standalone standard optimizer for this specific parameter group
            opt = opt_class([{'params': group['params']}], lr=lr)
            self.groups.append({'opt': opt, 'f': group['f']})
        self.step_idx = 0

    def zero_grad(self):
        # Override the global zero_grad() called by the training loop.
        # We DO NOT want to wipe gradients globally, because slow layers 
        # need to accumulate them across multiple batches.
        pass

    def step(self, closure=None):
        self.step_idx += 1
        for group in self.groups:
            # Only step the specific layer if its frequency target is hit
            if self.step_idx % group['f'] == 0:
                group['opt'].step()
                # Clear the gradients ONLY for the layer that just updated, 
                # leaving the slower layers to continue accumulating.
                group['opt'].zero_grad()

def get_optimizer(model, opt_name, lr=1e-3, f=20, stabilize=True, alpha=0.5, beta3=0.9):
    """Instantiates the requested optimizer."""
    if opt_name == 'SGD':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                {'params': model.fast_memory.parameters(), 'f': max(1, f//5)},
                {'params': model.medium_memory.parameters(), 'f': max(1, f//2)},
                {'params': model.slow_memory.parameters(), 'f': f},
                {'params': model.head.parameters(), 'f': max(1, f//5)}
            ]
            return DecoupledOptimizer(param_groups, optim.SGD, lr=lr)
        else:  # FALLBACK FOR BASELINE MODE
            return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'MSGD':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': max(1, f//5), 'use_muon': False, 'use_variance': False, 'stabilize': stabilize},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': max(1, f//2), 'use_muon': False, 'use_variance': False, 'stabilize': stabilize},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': max(1, f//5), 'use_muon': False, 'use_variance': False, 'stabilize': stabilize}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=False, use_variance=False, stabilize = stabilize)
    elif opt_name == 'Adam':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                {'params': model.fast_memory.parameters(), 'f': max(1, f//5)},
                {'params': model.medium_memory.parameters(), 'f': max(1, f//2)},
                {'params': model.slow_memory.parameters(), 'f': f},
                {'params': model.head.parameters(), 'f': max(1, f//5)}
            ]
            return DecoupledOptimizer(param_groups, optim.Adam, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'MAdam':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': f//2, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=False, use_variance=True, stabilize = stabilize)
    elif opt_name == 'Muon':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                {'params': model.fast_memory.parameters(), 'f': max(1, f//5)},
                {'params': model.medium_memory.parameters(), 'f': max(1, f//2)},
                {'params': model.slow_memory.parameters(), 'f': f},
                {'params': model.head.parameters(), 'f': max(1, f//5)}
            ]
            # Pass the custom Muon class into the wrapper
            return DecoupledOptimizer(param_groups, Muon, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return Muon(model.parameters(), lr=lr)
    elif opt_name == 'M3S': # Stands for stablized M3 optimizer
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': True, 'use_variance': True, 'stabilize': True, 'beta3': beta3},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': f//2, 'use_muon': True, 'use_variance': True, 'stabilize': True, 'beta3': beta3},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': True, 'use_variance': True, 'stabilize': True, 'beta3': beta3}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': True, 'beta3': beta3}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=True, use_variance=True, stabilize=True)
    elif opt_name == 'M3': # Unstabilized Version, the same as the original paper
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': True, 'use_variance': True, 'stabilize': False},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': f//2, 'use_muon': True, 'use_variance': True, 'stabilize': False},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': True, 'use_variance': True, 'stabilize': False}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True, 'stabilize': False}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=True, use_variance=True, stabilize=False)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")