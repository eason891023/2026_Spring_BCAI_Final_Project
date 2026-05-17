import torch
from torch.optim import Optimizer

def newton_schulz_iteration(G, steps=5):
    """
    Computes the orthogonalization of 2D tensor G using Newton-Schulz.
    (Approximation used in modern Muon implementations).
    """
    assert len(G.shape) == 2, "Newton-Schulz requires 2D tensors."
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # Cast to bfloat16 or float32 for stable computation
    X = G.float() 
    X = X / (X.norm() + 1e-7)
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    return X.to(G.dtype)

class M3(Optimizer):
    """
    Optionally Stabilized Multi-scale Momentum Muon (M3) Optimizer.
    Fixes the unbounded variance and orthogonal-division explosion 
    inherent in the literal translation of the paper's algorithm.
    """
    def __init__(self, params, lr=1e-3, f=20, beta1=0.9, beta2=0.99, beta3=0.9, alpha=0.5, eps=1e-8, ns_steps=5, use_muon=True, use_variance=True, stabilize=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, f=f, beta1=beta1, beta2=beta2, beta3=beta3, alpha=alpha, eps=eps, ns_steps=ns_steps, use_muon=use_muon, use_variance=use_variance, stabilize=stabilize)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # Initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['M1'] = torch.zeros_like(p)
                    state['M2'] = torch.zeros_like(p)
                    state['V'] = torch.zeros_like(p)
                    state['grad_sum'] = torch.zeros_like(p)
                    state['O2'] = torch.zeros_like(p)

                state['step'] += 1
                step = state['step']
                f = group['f']

                state['grad_sum'] += grad

                # --- Outer Loop: Lower-Frequency Iteration ---
                if step % f == 0:
                    if group['stabilize']: # Stabilized EMA for Slow Memory: M2 = M2 * beta3 + (1 - beta3) * g
                        state['M2'].mul_(group['beta3']).add_(state['grad_sum'], alpha=1.0 - group['beta3'])
                    else:                  # Original Paper: M2 = M2 + beta3 * g
                        state['M2'].add_(state['grad_sum'], alpha=group['beta3'])
                    
                    if len(p.shape) >= 2:
                        orig_shape = state['M2'].shape
                        m2_2d = state['M2'].view(orig_shape[0], -1)
                        o2_2d = newton_schulz_iteration(m2_2d, steps=group['ns_steps'])
                        state['O2'] = o2_2d.view(orig_shape)
                    else:
                        state['O2'] = state['M2']
                    
                    state['grad_sum'].zero_()

                # --- Inner Loop: Higher-Frequency Iteration ---
                if group['stabilize']: # Stabilized EMA for First and Second Momentum: M1 = M1 * beta1 + (1 - beta1) * g
                    state['M1'].mul_(group['beta1']).add_(grad, alpha=1.0 - group['beta1'])
                else:                  # Original Paper: M1 = M1 + beta1 * g
                    state['M1'].add_(grad, alpha=group['beta1'])
                
                if group['use_variance']:
                    if group['stabilize']: # Stabilized EMA to avoid denom used later become 0: V = V * beta2 + (1 - beta2) * g^2
                        state['V'].mul_(group['beta2']).addcmul_(grad, grad, value=1.0 - group['beta2'])
                    else:                  # Original Paper: V = V + beta2 * g^2
                        state['V'].addcmul_(grad, grad, value=group['beta2'])

                # FIX: Check for the use_muon flag before orthogonalizing
                if len(p.shape) >= 2 and group['use_muon']:
                    orig_shape = state['M1'].shape
                    m1_2d = state['M1'].view(orig_shape[0], -1)
                    o1_2d = newton_schulz_iteration(m1_2d, steps=group['ns_steps'])
                    O1 = o1_2d.view(orig_shape)
                    
                    update_direction = O1 + group['alpha'] * state['O2']
                else:
                    if group['use_variance']: # STANDARD ADAM: Now correctly applies to the working memory and head
                        denom = state['V'].sqrt().add_(group['eps'])
                        update_direction = (state['M1'] + group['alpha'] * state['O2']) / denom
                    else: # SGD FALLBACK (No variance division)
                        update_direction = state['M1'] + group['alpha'] * state['O2']
                
                # Apply scaled update
                p.add_(update_direction, alpha=-group['lr'])

        return loss