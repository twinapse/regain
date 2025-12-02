# Methods

This file is a collection of **implementation notes** for the methods we use in this codebase. It’s meant to capture 
the practical bits that are easy to forget later: what the original authors’ implementation assumes, what we 
implemented here, what we intentionally skipped, and what would need to change to reproduce an original setup more 
faithfully.

## Linear probe

We use a linear probe to evaluate the quality of learned representations.

See [*"Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting"*](https://arxiv.org/abs/2304.00933).

## BiC (Bias Correction)

[**Paper**](https://arxiv.org/abs/1905.13260)

### Our implementation

- We use Avalanche built-in [`BiC`](https://avalanche-api.continualai.org/en/latest/generated/avalanche.training.BiC.html#avalanche.training.BiC) 
  strategy.

## IL2M (Class Incremental Learning With Dual Memory)

[**Paper**](https://ieeexplore.ieee.org/document/9009019)

### Our implementation

- We use Avalanche built-in 
  [`IL2M`](https://avalanche-api.continualai.org/en/latest/generated/avalanche.training.IL2M.html#avalanche.training.IL2M) 
  strategy.

## CN (Continual Normalization)

[**Paper**](https://arxiv.org/abs/2203.16102)

### Our implementation

- We implement CN as the *direct BN replacement* described in the paper: **GroupNorm (no affine) → BatchNorm (with the
  original BN affine + running stats)**. Concretely, our CN forward does:
  - `group_norm(x, G, weight=None, bias=None)` (so GN contributes *only* normalization, no learned scale/shift), then
  - `batch_norm(...)` using the carried-over `running_mean`, `running_var`, `weight`, and `bias`.
- We replace **only `nn.BatchNorm2d`** modules (`replace_batchnorm2d`). If a model uses `BatchNorm1d` (e.g., MLPs) we 
  do **not** rewrite those layers.
- We expose **`groups`** as a parameter and provide common presets (4/8/16/32/64). The paper typically uses a fixed 
  group count (commonly 32 in vision backbones); in this repo you can change it via controller config.
- We intentionally keep CN as a **pure backbone rewrite** (no extra losses, no extra buffers/datasets, no task-specific
  logic). This matches the “drop-in normalization replacement” spirit, but we do not attempt to reproduce every training
  detail/setup from the paper’s experiments (online CL protocols, specific baselines, etc.).

## TBBN (Task-Balanced Batch Normalization)

[**Paper**](https://arxiv.org/abs/2201.12559)

### Our implementation

- We implement TBBN as a **drop-in `BatchNorm2d` replacement** (`TaskBalancedBatchNorm`) and a controller that:
  1) rewrites all `BatchNorm2d` layers, and  
  2) calls `set_number_of_task(t)` at the **start of each experience** (0-indexed), as required by the method.
- Training-time behavior follows the paper’s core idea: correct the BN statistics/gradient bias caused by the
  **current-vs-replay minibatch imbalance** by constructing a “task-balanced” batch using reshape/repeat operations
  (current samples are split; replay samples are repeated across splits).
- Inference-time behavior is intentionally **BN-like**: once in eval mode, the layer behaves like vanilla BN using its
  running statistics (the “balancing” is a training-only mechanism).
- Assumptions/differences to be aware of:
  - **Minibatch layout + partition:** minibatches must be `[current-task samples | replay samples]` with
    `B_c = train_mb_size` (current) and `B_p = replay_batch_size` (Avalanche `mem_mb_size` / `batch_size_mem`) (replay).
    TBBN is only wired for replay-based strategies (Replay, BiC, IL2M) and validates the partition accordingly.
  - **Explicit batch sizes only:** we no longer derive `(B_c, B_p)` from a single “total batch size”; both minibatch
    sizes must be configured directly in the experiment/controller to match the strategy dataloaders.
  - **Last minibatch robustness:** some strategies / dataloaders can yield a last minibatch whose current-part size is
    not divisible by the configured split factor. We reduce splits via `gcd` per minibatch to avoid invalid reshapes; a
    stricter reference implementation may simply assume divisibility and error out.

## BaCE

[**Paper**](https://openreview.net/forum?id=EOTgj37XNM)

### Our implementation

We implement BaCE as a **training-loss controller** (`BaCEController`) that plugs into Avalanche training via
`TrainingObjectiveControllerInterface`.

What we implement:

- **Teacher / student setup with epoch-level teacher updates.**
  - We keep a frozen teacher copy initialized at the start of each experience.
  - We update teacher parameters via **EMA at the end of each training epoch** (rather than per-iteration).
- **Effect\_new (learning new classes with “old” causal effect included) via a joint-score KNN objective:**
  - We build a **KNN feature bank** in the *teacher’s* feature space over the **current experience dataset**.
    - For efficiency, the bank is built once per epoch and can be subsampled (`bank_max_samples`).
  - For current-task samples in a minibatch, we compute:
    - student self-scores (`softmax(logits)`), and
    - neighbor contributions from the **student’s predictions on retrieved neighbors**, weighted and mixed with 
      self-score via `w0`.
  - Neighbor weights are computed by an **inverse-distance** rule (with eps for stability) and we exclude self-matches /
    duplicates using a small distance threshold.
  - The loss is a negative log-likelihood on the target probability under these **joint scores**.
- **Effect\_old (stabilizing old classes) via distillation on old-class scores:**
  - On current-task samples, we compute **KL divergence** between student and teacher softmax scores restricted to the
    **old-class subset**.
  - If replay/buffer samples are present in the minibatch, we switch to a different weighting (`alpha_with_buffer` vs
    `alpha_no_buffer`) consistent with the idea that buffer availability changes the required strength of the effect.
- **Replay enhancement (when replay samples exist) in a DER++-style form:**
  - CE on replay samples (standard supervised loss),
  - plus MSE distillation between student and teacher **old-class logits** on replay samples.

Important differences / simplifications vs a faithful reproduction:

- Our implementation is **scoped to image classification-style models** that expose a `.backbone` (or `.encoder`) 
  returning 2D features; we do not reproduce the paper’s broader PTM/NLP setups.
- The KNN bank in this repo is built over the **current experience dataset** (with the teacher’s representation), not a
  specialized multi-source retrieval system. This is a pragmatic approximation of BaCE’s “use prior/old knowledge to
  assist adaptation” idea.
- We update the teacher **once per epoch** (EMA). If an official implementation updates more frequently, results may
  differ.
- We detect “current vs replay” samples by **class membership in the minibatch labels**, not by relying on a particular
  sampler ordering. This makes the controller more robust across Avalanche strategies, but it may not match an exact
  paper protocol if the protocol assumes a strict batch construction process.
