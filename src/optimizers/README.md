# Optimizers

This directory contains the implementations for various optimization strategies used in this Nested Learning on Continual Learning project. 

## Types of Optimizers in this Repo
* **Traditional Ones:** Standard monolithic implementations (e.g., `SGD`, `Adam` and `Muon`).
* **Multiscale Ones:** Optimizers with mathematically integrated temporal memories (e.g., `MSGD`, `MAdam`, `M3`, and `M3S`).
* **Decoupled Ones:** Standard optimizers wrapped structurally for CMS models (effectively `dSGD`, `dAdam`, and `dMuon`).

---

## Why Optimizers Matter in this Project

The primary goal of this small research project is to isolate and evaluate the true source of performance gains in **Continuum Memory Systems (CMS)**. 

When deep networks attempt to learn tasks incrementally, they often suffer from Catastrophic Forgetting. Recent literature suggests that using "Multi-scale" optimizers (like M3) which maintain separate memory buffers for different update frequencies can alleviate this. 

However, multi-scale optimizers conflate two completely distinct mechanisms:
1. **Temporal Splitting:** Delaying the updates of deeper layers (only updating them every $f$ steps) so they learn slower, more generalized representations.
2. **Complex Momentum Math:** Accumulating gradients into specialized unbounded memory buffers and applying orthogonalization (Newton-Schulz).

By providing all three types of optimizers (Traditional, Decoupled, and Multiscale), this project allows us to **isolate and compare** these mechanisms. We can test whether the complex mathematical momentum of M3 is actually necessary, or if simply wrapping a standard `SGD` optimizer to enforce the **Temporal Split** (the Decoupled approach) yields the exact same Continual Learning benefits with significantly less computational overhead. Furthermore, we can compare original paper formulations against mathematically stabilized variants to study the effects of gradient explosion over long training sequences.

---

## Detailed Information & Mathematical Differences

### 1. Traditional Optimizers (`SGD`, `Adam`, `Muon`)
These are the standard PyTorch-style implementations. When applied to a standard monolithic `baseline` MLP, they update all parameters simultaneously at every single step ($f=1$). 
* **Muon:** An optimizer that applies Newton-Schulz iteration to orthogonalize the momentum matrices of 2D parameters (weights), improving conditioning.

### 2. Decoupled Optimizers (e.g., `DecoupledOptimizer` Wrapper)
When a Traditional optimizer is paired with a `cms` model architecture, it is automatically wrapped in `factory.py` into a **Decoupled Optimizer**. 
* **Mechanism:** It instantiates separate, isolated optimizers for each module (`fast_memory`, `medium_memory`, `slow_memory`). 
* **Math:** The underlying mathematical update is strictly standard (e.g., normal SGD). However, the wrapper intercepts the `step()` and `zero_grad()` calls. A specific module only steps and clears its gradients when the global step index aligns with its designated frequency target $f$. Slower layers naturally accumulate raw gradients via PyTorch's native `.backward()` graph over $f$ steps before updating.

### 3. Multiscale Optimizers (`M3`, `M3S`, `MSGD`, `MAdam`)
These are complex, monolithic optimizers that handle the temporal split mathematically within their state dictionaries, rather than structurally. They manage an "Inner Loop" (high frequency) and an "Outer Loop" (low frequency).

#### Unstabilized Formulation (Original Paper - e.g., `M3`)
The original literal translation uses direct addition for accumulation.
* **Inner Loop (Every Step):**
  * $M_1 = M_1 + \beta_1 g$ *(Fast Momentum)*
  * $V = V + \beta_2 g^2$ *(Variance)*
* **Outer Loop (Every $f$ Steps):**
  * $M_2 = M_2 + \beta_3 \sum g$ *(Slow Momentum)*
  * $O_2 = \text{NewtonSchulz}(M_2)$ *(Orthogonalized Slow Memory)*
* **Update:** $\Delta W = - \eta (O_1 + \alpha O_2)$
* **Danger:** The accumulation buffers ($M = M + \beta g$) are unbounded. Over long continual learning sequences, this can lead to infinite memory horizons, gradient explosions, or a blown-up effective learning rate.

#### Stabilized Formulation (`M3S`)
To fix the unbounded variance of the original formulation, the stabilized version replaces direct addition with an **Exponential Moving Average (EMA)**.
* **Inner Loop:** $M_1 = \beta_1 M_1 + (1 - \beta_1) g$
* **Outer Loop:** $M_2 = \beta_3 M_2 + (1 - \beta_3) \sum g$
* **Result:** This safely extends the "memory horizon" by gently decaying older batches, maintaining stable buffer sizes regardless of the training duration.

#### Ablations (`MSGD`, `MAdam`)
These variants use the same Multi-scale accumulation loops but strip away specific components to ablate the effects of Muon and Adam-style scaling:
* **MSGD:** Multi-scale SGD. Strips away Muon (Newton-Schulz) and Variance scaling. Update direction is based strictly on $M_1 + \alpha O_2$.
* **MAdam:** Multi-scale Adam. Strips away Muon, but retains the Variance scaling denominator: $(M_1 + \alpha O_2) / (\sqrt{V} + \epsilon)$.
