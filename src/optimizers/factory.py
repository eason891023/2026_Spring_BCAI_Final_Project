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

class SurpriseGatedOptimizer:
    """
    Commit-on-surprise CMS wrapper for standard PyTorch optimizers.

    Fast memory and heads update every batch. Medium/slow memories keep their
    own gradient buffers and commit when either gradient-norm surprise exceeds
    a z-score threshold or a max interval is reached. On surprise, the current
    spike gradient starts the next buffer instead of being written into the
    just-committed memory.
    """
    def __init__(self, param_groups, opt_class, lr, tau=2.0, tmin=1,
                 tmax=None, ema_rho=0.05, warmup=20, eps=1e-8):
        self.groups = []
        self.step_idx = 0
        self.event_log = []
        self.ema_rho = ema_rho
        self.warmup = warmup
        self.eps = eps

        for idx, group in enumerate(param_groups):
            params = list(group['params'])
            opt = opt_class([{'params': params}], lr=lr)
            cadence = max(1, group.get('f', 1))
            gated = bool(group.get('gated', cadence > 1))
            group_tmax = group.get('tmax', tmax)
            if group_tmax is None:
                group_tmax = cadence
            self.groups.append({
                'name': group.get('name', f'group{idx}'),
                'params': params,
                'opt': opt,
                'gated': gated,
                'tau': float(group.get('tau', tau)),
                'tmin': max(1, int(group.get('tmin', tmin))),
                'tmax': max(1, int(group_tmax)),
                'last_commit': 0,
                'mean': None,
                'var': 0.0,
                'count': 0,
                'buffers': [None for _ in params],
            })

    def zero_grad(self):
        for group in self.groups:
            for p in group['params']:
                if p.grad is not None:
                    p.grad.detach_()
                    p.grad.zero_()

    @staticmethod
    def _grad_norm(grads):
        total = 0.0
        for grad in grads:
            if grad is not None:
                total += grad.pow(2).sum().item()
        return total ** 0.5

    def _commit(self, group, reason, z_score, surprise):
        has_buffer = any(buf is not None for buf in group['buffers'])
        if not has_buffer:
            group['last_commit'] = self.step_idx
            return

        for param, buf in zip(group['params'], group['buffers']):
            param.grad = None if buf is None else buf.clone()
        group['opt'].step()
        group['opt'].zero_grad()
        group['buffers'] = [None for _ in group['params']]
        group['last_commit'] = self.step_idx
        self.event_log.append({
            'step': self.step_idx,
            'group': group['name'],
            'reason': reason,
            'z': float(z_score),
            'surprise': float(surprise),
        })

    def _update_stats(self, group, surprise):
        if group['mean'] is None:
            group['mean'] = surprise
            group['var'] = 0.0
        else:
            delta = surprise - group['mean']
            group['mean'] = (1.0 - self.ema_rho) * group['mean'] + self.ema_rho * surprise
            group['var'] = (1.0 - self.ema_rho) * group['var'] + self.ema_rho * delta * delta
        group['count'] += 1

    def _append_current_grads(self, group, grads):
        for i, grad in enumerate(grads):
            if grad is None:
                continue
            if group['buffers'][i] is None:
                group['buffers'][i] = grad.clone()
            else:
                group['buffers'][i].add_(grad)

    def step(self, closure=None):
        self.step_idx += 1
        for group in self.groups:
            if not group['gated']:
                group['opt'].step()
                group['opt'].zero_grad()
                continue

            grads = [p.grad.detach().clone() if p.grad is not None else None for p in group['params']]
            surprise = self._grad_norm(grads)
            mean = surprise if group['mean'] is None else group['mean']
            var = group['var']
            z_score = 0.0 if group['count'] == 0 else (surprise - mean) / ((var + self.eps) ** 0.5)
            elapsed = self.step_idx - group['last_commit']

            surprise_ready = group['count'] >= self.warmup and elapsed >= group['tmin']
            surprise_trigger = surprise_ready and z_score > group['tau']
            max_trigger = elapsed >= group['tmax']

            if surprise_trigger:
                self._commit(group, 'surprise', z_score, surprise)
            elif max_trigger:
                self._commit(group, 'tmax', z_score, surprise)

            self._append_current_grads(group, grads)
            self._update_stats(group, surprise)

            for param in group['params']:
                param.grad = None

    def flush(self):
        for group in self.groups:
            if group['gated']:
                self._commit(group, 'flush', 0.0, 0.0)

    def get_event_log(self):
        return list(self.event_log)

def get_optimizer(model, opt_name, lr=1e-3, f=20, stabilize=True, alpha=0.5, beta3=0.9,
                  sg_tau=2.0, sg_tmin=1, sg_tmax=None, sg_ema_rho=0.05, sg_warmup=20):
    """Instantiates the requested optimizer."""
    if opt_name == 'SGD':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                {'name': 'fast', 'params': model.fast_memory.parameters(), 'f': 1, 'gated': False},
                {'name': 'medium', 'params': model.medium_memory.parameters(), 'f': max(1, f//2), 'gated': True},
                {'name': 'slow', 'params': model.slow_memory.parameters(), 'f': f, 'gated': True},
                {'name': 'head', 'params': model.head.parameters(), 'f': 1, 'gated': False}
            ]
            if getattr(model, 'surprise_gated', False):
                return SurpriseGatedOptimizer(
                    param_groups, optim.SGD, lr=lr, tau=sg_tau, tmin=sg_tmin,
                    tmax=sg_tmax, ema_rho=sg_ema_rho, warmup=sg_warmup)
            return DecoupledOptimizer(param_groups, optim.SGD, lr=lr)
        else:  # FALLBACK FOR BASELINE MODE
            return optim.SGD(model.parameters(), lr=lr)
    elif opt_name == 'MSGD':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': max(1, f//2), 'use_muon': False, 'use_variance': False, 'stabilize': stabilize},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': False, 'use_variance': False, 'stabilize': stabilize}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=False, use_variance=False, stabilize = stabilize)
    elif opt_name == 'Adam':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                {'name': 'fast', 'params': model.fast_memory.parameters(), 'f': 1, 'gated': False},
                {'name': 'medium', 'params': model.medium_memory.parameters(), 'f': max(1, f//2), 'gated': True},
                {'name': 'slow', 'params': model.slow_memory.parameters(), 'f': f, 'gated': True},
                {'name': 'head', 'params': model.head.parameters(), 'f': 1, 'gated': False}
            ]
            if getattr(model, 'surprise_gated', False):
                return SurpriseGatedOptimizer(
                    param_groups, optim.Adam, lr=lr, tau=sg_tau, tmin=sg_tmin,
                    tmax=sg_tmax, ema_rho=sg_ema_rho, warmup=sg_warmup)
            return DecoupledOptimizer(param_groups, optim.Adam, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return optim.Adam(model.parameters(), lr=lr)
    elif opt_name == 'MAdam':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': max(1, f//2), 'use_muon': False, 'use_variance': True, 'stabilize': stabilize},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': False, 'use_variance': True, 'stabilize': stabilize}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=False, use_variance=True, stabilize = stabilize)
    elif opt_name == 'Muon':
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                {'params': model.fast_memory.parameters(), 'f': 1},
                {'params': model.medium_memory.parameters(), 'f': max(1, f//2)},
                {'params': model.slow_memory.parameters(), 'f': f},
                {'params': model.head.parameters(), 'f': 1}
            ]
            # Pass the custom Muon class into the wrapper
            return DecoupledOptimizer(param_groups, Muon, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return Muon(model.parameters(), lr=lr)
    elif opt_name == 'M3S': # Stands for stablized M3 optimizer
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': True, 'use_variance': True, 'stabilize': True, 'beta3': beta3},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': max(1, f//2), 'use_muon': True, 'use_variance': True, 'stabilize': True, 'beta3': beta3},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': True, 'use_variance': True, 'stabilize': True, 'beta3': beta3}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': False, 'use_variance': True, 'stabilize': True, 'beta3': beta3}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODE
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=True, use_variance=True, stabilize=True)
    elif opt_name == 'M3': # Unstabilized Version, the same as the original paper
        if hasattr(model, 'fast_memory') and hasattr(model, 'slow_memory') and hasattr(model, 'medium_memory'):
            param_groups = [
                # Fast Memory
                {'params': model.fast_memory.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': True, 'use_variance': True, 'stabilize': False},
                # Medium Memory
                {'params': model.medium_memory.parameters(), 'alpha': alpha * 0.6, 'f': max(1, f//2), 'use_muon': True, 'use_variance': True, 'stabilize': False},
                # Slow Memory
                {'params': model.slow_memory.parameters(), 'alpha': alpha, 'f': f, 'use_muon': True, 'use_variance': True, 'stabilize': False}, 
                # Isolated Head: Standard Adam updates (No Muon)
                {'params': model.head.parameters(), 'alpha': 0.0, 'f': 1, 'use_muon': False, 'use_variance': True, 'stabilize': False}
            ]
            return M3(param_groups, lr=lr)
        else: # FALLBACK FOR BASELINE MODEL
            return M3(model.parameters(), lr=lr, f=f, alpha=alpha, beta3=beta3, use_muon=True, use_variance=True, stabilize=False)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")
