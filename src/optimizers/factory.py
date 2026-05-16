import torch.optim as optim
from .m3 import M3, SM3
from .muon import Muon

def get_optimizer(model, opt_name, lr=1e-3, f=20):
    """Instantiates the requested optimizer."""
    if opt_name == 'SGD':
        return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'MSGD':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': False},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': False, 'use_variance': False},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': False, 'use_variance': False}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': False}
            ]
            return SM3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return SM3(model.parameters(), lr=lr, alpha=1.0, use_muon=False, use_variance=False)
    elif opt_name == 'Adam':
        return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'MAdam':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': False, 'use_variance': True},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': False, 'use_variance': True}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True}
            ]
            return SM3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return SM3(model.parameters(), lr=lr, alpha=1.0, use_muon=False, use_variance=True)
    elif opt_name == 'Muon':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'): 
            raise AttributeError("To use Muon with cms model, please call `--model cms --optimizer SM3` instead!")
        else: # FALLBACK FOR BASELINE MODEL
            return Muon(model.parameters(), lr=lr)
    elif opt_name == 'SM3': # Stands for stablized M3 optimizer
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': True, 'use_variance': True},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': True, 'use_variance': True},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': True, 'use_variance': True}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False, 'use_variance': True}
            ]
            return SM3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return SM3(model.parameters(), lr=lr, alpha=1.0, use_muon=True, use_variance=True)
    elif opt_name == 'M3':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': True},

                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': 0.3, 'f': f//2, 'use_muon': True},
                
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': 0.5, 'f': f, 'use_muon': True}, 
                
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': f//5, 'use_muon': False}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, alpha=1.0)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")