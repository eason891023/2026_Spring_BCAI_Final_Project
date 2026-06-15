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

class SGCMS_MLP(SCMS_MLP):
    """
    Surprise-Gated CMS.

    The architecture matches SCMS exactly; the difference is in optimizer
    state. factory.py detects this marker and uses adaptive consolidation for
    medium/slow memories instead of a fixed update cadence.
    """
    surprise_gated = True

class NCMS_MLP(nn.Module):
    """
    Nested Continuum Memory System MLP (NL paper Sec. 7.1, Eq. 72).
    Forward is the same chain as SCMS_MLP (Eq. 70 is shared by the Nested and
    Sequential variants; only the Independent variant changes the output
    computation), so parameter count and FLOPs match SCMS exactly. The nested
    behavior lives in STATE MANAGEMENT:
      * Each task is treated as one context. The fast and medium levels keep a
        meta-learned initial state Phi, stored as buffers.
      * At a context boundary the trainer calls `end_context()`, which first
        meta-updates Phi toward the adapted weights with a first-order
        (Reptile-style) step  Phi <- Phi + meta_lr * (theta - Phi)  -- a
        tractable approximation of Eq. 72's argmin that avoids BPTT through the
        inner loop -- and then resets the level's weights to Phi.
      * The fast level resets every context, the medium level resets every
        `medium_period` contexts, and the slow level / head never reset (they
        are the persistent memory; resetting the head would wipe old-class
        logits under Class-IL).
    Within-level update frequencies (Eq. 71) remain the optimizer's job
    (f=1 / f//2 / f batches) and are unchanged by this variant.

    reset_mode:
      'meta'   - reset to the Reptile meta-learned init (default, approx. Eq. 72)
      'random' - reset to the frozen random init (ablates the meta-learning term)
      'none'   - never reset (state management degenerates back to SCMS)
    """
    RESET_LEVELS = ('fast_memory', 'medium_memory')

    def __init__(self, input_size=784, hidden_size=256, num_classes=10,
                 meta_lr=0.5, medium_period=2, reset_mode='meta'):
        super().__init__()
        if reset_mode not in ('meta', 'random', 'none'):
            raise ValueError(f"Unsupported reset_mode: {reset_mode}")

        # Fast updating level (inner-most loop, reset every context)
        self.fast_memory = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )

        # Medium updating level (reset every `medium_period` contexts)
        self.medium_memory = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Slow updating level (persistent memory, never reset)
        self.slow_memory = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Isolated classification head (fast updates, never reset)
        self.head = nn.Linear(hidden_size, num_classes)

        self.meta_lr = meta_lr
        self.medium_period = medium_period
        self.reset_mode = reset_mode
        self.context_count = 0

        # Snapshot each resettable level's init Phi as buffers so the
        # meta-learned inits follow the model across devices and checkpoints.
        for level in self.RESET_LEVELS:
            for key, value in getattr(self, level).state_dict().items():
                self.register_buffer(self._init_key(level, key), value.detach().clone())

    @staticmethod
    def _init_key(level, key):
        return f"init_{level}_{key.replace('.', '_')}"

    def forward(self, x):
        fast_features = self.fast_memory(x)
        medium_features = self.medium_memory(fast_features)
        slow_features = self.slow_memory(medium_features)
        return self.head(slow_features)

    @torch.no_grad()
    def end_context(self):
        """
        Context (task) boundary: meta-update each resettable level's init Phi
        toward its adapted weights (first-order Eq. 72), then reset the level.
        """
        self.context_count += 1
        if self.reset_mode == 'none':
            return
        for level in self.RESET_LEVELS:
            if level == 'medium_memory' and self.context_count % self.medium_period != 0:
                continue
            for key, weight in getattr(self, level).state_dict().items():
                init = getattr(self, self._init_key(level, key))
                if self.reset_mode == 'meta':
                    init.lerp_(weight, self.meta_lr)
                weight.copy_(init)

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
