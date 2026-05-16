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
    Multi-scale Momentum Muon (M3) Optimizer.
    Implemented exactly as specified in Algorithm 1 in the nested learning paper.
    """
    def __init__(self, params, lr=1e-3, f=20, beta1=0.9, beta2=0.99, beta3=0.9, alpha=0.5, eps=1e-8, ns_steps=5, use_muon=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, f=f, beta1=beta1, beta2=beta2, beta3=beta3, alpha=alpha, eps=eps, ns_steps=ns_steps)
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

                # --- Initialization (Line 1) ---
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

                # Accumulate gradients for the slow memory chunk
                state['grad_sum'] += grad

                # --- Outer Loop: Lower-Frequency Iteration (Lines 2-4) ---
                # Triggers at the start of a new chunk (step 1, f+1, 2f+1, ...)
                if step % f == 1:
                    # Slow Memory: M2 = M2 + beta3 * sum(gradients)
                    state['M2'].add_(state['grad_sum'], alpha=group['beta3'])
                    
                    if len(p.shape) >= 2:
                        # Flatten to 2D for Newton-Schulz
                        orig_shape = state['M2'].shape
                        m2_2d = state['M2'].view(orig_shape[0], -1)
                        o2_2d = newton_schulz_iteration(m2_2d, steps=group['ns_steps'])
                        state['O2'] = o2_2d.view(orig_shape)
                    else:
                        # Fallback for 1D parameters (biases)
                        state['O2'] = state['M2'] / (state['M2'].norm() + group['eps'])
                    
                    # Reset chunk accumulator
                    state['grad_sum'].zero_()

                # --- Inner Loop: Higher-Frequency Iteration (Lines 5-10) ---
                
                # First Momentum: M1 = M1 + beta1 * g_t
                state['M1'].add_(grad, alpha=group['beta1'])
                
                # Second Momentum: V = V + beta2 * g_t^2
                state['V'].addcmul_(grad, grad, value=group['beta2'])

                # FIX: Check for the use_muon flag before orthogonalizing
                if len(p.shape) >= 2 and group['use_muon']:
                    orig_shape = state['M1'].shape
                    m1_2d = state['M1'].view(orig_shape[0], -1)
                    o1_2d = newton_schulz_iteration(m1_2d, steps=group['ns_steps'])
                    O1 = o1_2d.view(orig_shape)
                    
                    update_direction = O1 + group['alpha'] * state['O2']
                else:
                    # STANDARD ADAM: Now correctly applies to the working memory and head
                    denom = state['V'].sqrt().add_(group['eps'])
                    update_direction = (state['M1'] + group['alpha'] * state['O2']) / denom
                
                p.add_(update_direction, alpha=-group['lr'])

        return loss

class SM3(Optimizer):
    """
    Stabilized Multi-scale Momentum Muon (M3) Optimizer.
    Fixes the unbounded variance and orthogonal-division explosion 
    inherent in the literal translation of the paper's algorithm.
    """
    def __init__(self, params, lr=1e-3, f=20, beta1=0.9, beta2=0.99, beta3=0.9, alpha=0.5, eps=1e-8, ns_steps=5, use_muon=True, use_variance=True):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
            
        defaults = dict(lr=lr, f=f, beta1=beta1, beta2=beta2, beta3=beta3, alpha=alpha, eps=eps, ns_steps=ns_steps, use_muon=use_muon, use_variance=use_variance)
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
                if step % f == 1:
                    # Stabilized EMA for Slow Memory
                    state['M2'].mul_(group['beta3']).add_(state['grad_sum'], alpha=1.0 - group['beta3'])
                    
                    if len(p.shape) >= 2:
                        orig_shape = state['M2'].shape
                        m2_2d = state['M2'].view(orig_shape[0], -1)
                        o2_2d = newton_schulz_iteration(m2_2d, steps=group['ns_steps'])
                        state['O2'] = o2_2d.view(orig_shape)
                    else:
                        state['O2'] = state['M2']
                    
                    state['grad_sum'].zero_()

                # --- Inner Loop: Higher-Frequency Iteration ---
                # Stabilized EMA for First and Second Momentum
                state['M1'].mul_(group['beta1']).add_(grad, alpha=1.0 - group['beta1'])
                
                if group['use_variance']:
                    state['V'].mul_(group['beta2']).addcmul_(grad, grad, value=1.0 - group['beta2'])

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