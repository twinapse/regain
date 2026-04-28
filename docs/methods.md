# Methods

This file is a collection of **implementation notes** for the methods we use in this codebase. It’s meant to capture
the practical bits that are easy to forget later: what the original authors’ implementation assumes, what we
implemented here, what we intentionally skipped, and what would need to change to reproduce an original setup more
faithfully.

Methods marked **Not yet implemented** are included to document the repair-family taxonomy used here.

<!-- toc -->

- [Post-hoc repair controllers](#post-hoc-repair-controllers)
  - [Readout repair](#readout-repair)
    - [Linear probe](#linear-probe)
    - [Ridge probe](#ridge-probe)
    - [Low-rank probe](#low-rank-probe)
    - [Prototype classifier](#prototype-classifier)
  - [Bias / calibration repair](#bias--calibration-repair)
    - [Logit bias](#logit-bias)
    - [BiC (bias correction)](#bic-bias-correction)
    - [IL2M (class incremental learning with dual memory)](#il2m-class-incremental-learning-with-dual-memory)
    - [Temperature scaling](#temperature-scaling)
    - [T-CIL / DATS-style calibration](#t-cil--dats-style-calibration)
  - [Statistical drift repair](#statistical-drift-repair)
    - [MACIL](#macil)
    - [DPCR-style repair](#dpcr-style-repair)
  - [Modulation repair](#modulation-repair)
    - [Gain controllers](#gain-controllers)
    - [TMCL](#tmcl)
  - [Test-time repair](#test-time-repair)
    - [ARC (adaptive retention and correction)](#arc-adaptive-retention-and-correction)
- [Training-time / prevention controllers](#training-time--prevention-controllers)
  - [CN (continual normalization)](#cn-continual-normalization)
  - [TBBN (task-balanced batch normalization)](#tbbn-task-balanced-batch-normalization)
  - [BaCE](#bace)

## Post-hoc repair controllers

Post-hoc repair controllers act after backbone training boundaries and do not update the backbone trajectory. They are
used to measure which failure modes are recoverable under constrained repair data, controller capacity, and inference
cost.

### Readout repair

Readout repair methods test whether useful class information remains in the frozen representation but is no longer
expressed well by the trained classifier head.

#### Linear probe

We use a linear probe to evaluate the quality of learned representations and to measure readout-repairability.

See [*"Knowledge Accumulation and Feature Forgetting"*](https://arxiv.org/abs/2304.00933).

#### Ridge probe

**Status:** Not yet implemented in this repository.

##### Method role

- Ridge-regularized linear readout baseline on frozen representations.
- Useful for testing whether modest shrinkage improves readout repair under small repair budgets.
- In REGAIN-style comparisons, it is a readout-repair baseline rather than a separate controller family.

#### Low-rank probe

**Status:** Not yet implemented in this repository.

##### Method role

- Parameter-efficient readout baseline that constrains probe updates to a low-rank factorization.
- Useful when frontier comparisons need a readout method with lower parameter count than a full linear probe.
- In REGAIN-style comparisons, it bridges full probes and very low-capacity logit-space repair.

#### Prototype classifier

**Status:** Not yet implemented in this repository.

##### Method role

- Feature-space classifier built from class prototypes estimated on repair data.
- Useful for testing whether forgetting is primarily a prototype-drift problem rather than a classifier-head problem.
- In REGAIN-style comparisons, it is tracked under readout repair as a lightweight feature-to-label baseline.

### Bias / calibration repair

Bias and calibration repair methods test whether forgetting can be recovered by correcting logits, temperatures,
class-level bias, or confidence structure with fixed post-hoc parameters rather than changing the backbone
representation.

#### Logit bias

##### Our implementation

- We implement `logit_bias` as a **post-hoc repair controller** (`LogitBiasController`).
- The controller learns an additive per-class bias vector on repair data and applies `logits' = logits + b` during
  posthoc evaluation.
- The backbone remains frozen during fitting, making this a simple low-capacity learned logit-space baseline in the
  controller registry.

#### BiC (bias correction)

[**Paper**](https://arxiv.org/abs/1905.13260)

##### Our implementation

- We implement BiC as a **post-hoc repair controller** (`BiCController`) adapted from Avalanche's `BiCPlugin`
  stage-2 bias-correction procedure.
- Backbone training is still provided by the shared experiment backbone strategy; BiC fits a bias layer on the repair
  stream and applies it during posthoc evaluation.

#### IL2M (class incremental learning with dual memory)

[**Paper**](https://ieeexplore.ieee.org/document/9009019)

##### Our implementation

- We implement IL2M as a **post-hoc repair controller** (`IL2MController`) adapted from Avalanche's `IL2MPlugin`.
- The controller accumulates IL2M statistics from repair data after each experience and applies IL2M rectification
  during posthoc evaluation.

#### Temperature scaling

**Status:** Not yet implemented in this repository.

##### Method role

- One-parameter calibration baseline that rescales logits uniformly.
- Useful as a minimal-capacity reference point below per-class bias methods such as `logit_bias` or BiC.
- In REGAIN-style comparisons, it tests how much forgetting is recoverable through global confidence calibration alone.

#### T-CIL / DATS-style calibration

**Status:** Not yet implemented in this repository.

##### Method role

- Placeholder for class-incremental calibration methods that adjust temperatures or confidence structure more flexibly
  than a single global scalar.
- Useful for documenting the space between global temperature scaling and richer class-specific correction methods.
- No T-CIL- or DATS-style calibration controller is currently implemented in this repository.

### Statistical drift repair

Statistical drift repair methods test whether forgetting is recoverable by correcting shifts in feature or logit
statistics, such as means, covariances, prototypes, or analytically reconstructed classifiers.

#### MACIL

[**Paper**](https://openreview.net/forum?id=M6L7Eaw9BW)

**Status:** Not yet implemented in this repository.

##### Method role

- MACIL is a statistical drift repair baseline for task-agnostic class-incremental learning.
- It frames semantic drift as mean/covariance shift and applies lightweight calibration to compensate for that drift.
- In REGAIN-style comparisons, MACIL represents a mid-capacity repair family between logit-only calibration and fully
  learned readout repair.

#### DPCR-style repair

**Status:** Not yet implemented in this repository.

##### Method role

- Placeholder for prototype- or distribution-reconstruction repair methods that correct drift using class statistics.
- Useful for documenting statistical repair baselines that sit between logit-only correction and fitting a new probe.
- No DPCR-style controller is currently implemented in this repository.

### Modulation repair

Modulation repair methods test whether forgetting is recoverable by applying input-, layer-,
channel-, or state-dependent
modulations rather than replacing only the final readout.

#### Gain controllers

##### Our implementation

- These modulation-family controllers are repository-native repair baselines rather than reproductions of one
  published method.
- `scalar_stage` and `scalar_block` learn one multiplicative gain per resolved stage or block and apply those gains via
  temporary forward hooks during posthoc evaluation.
- `channel_stage` and `channel_block` learn grouped per-channel gains for each resolved stage or block.
- `conditioned_stage` and `conditioned_block` predict per-example stage/block gains from backbone features with a small
  MLP, then re-run the model with those input-conditioned gains applied.
- All six controllers are fit on repair data with the backbone frozen and act only during post-hoc evaluation.
- These gain controllers serve as exploratory modulation probes for testing whether intermediate-feature modulation can
  recover forgetting under constrained controller budgets.

#### TMCL

[**Paper**](https://arxiv.org/abs/2505.14125)

**Status:** Not yet implemented in this repository.

##### Method role

- TMCL is a modulation-based continual learning baseline built around top-down modulations and consolidation.
- It is relevant as a comparator for controllers that apply task-agnostic or input-conditioned gains to intermediate
  representations.
- In REGAIN-style comparisons, TMCL represents the modulation family rather than a pure logit-level or readout-only
  repair mechanism.

### Test-time repair

Test-time repair methods test whether forgetting can be recovered by making online control decisions during
evaluation, such as detecting past-task samples or applying input-conditioned correction, rather than relying only on a
pre-fit repair module learned from a labeled repair stream.

#### ARC (adaptive retention and correction)

[**Paper**](https://openreview.net/forum?id=9bLdbp46Q1)

**Status:** Not yet implemented in this repository.

##### Method role

- ARC is a test-time repair baseline for classifier bias and task-confusion failure modes.
- It detects whether a sample is likely to come from a past task and applies adaptive retention/correction at test
  time.
- In REGAIN-style comparisons, ARC should be treated as a repair method with different assumptions from fixed
  repair-set controllers, because it can use test-time adaptation behavior rather than only a pre-collected labeled
  repair stream.

## Training-time / prevention controllers

Training-time controllers modify the learning trajectory itself. They are not purely post-hoc repair methods, but they
provide prevention-style baselines for understanding when repair is sufficient and when training-time intervention is
needed.

### CN (continual normalization)

[**Paper**](https://arxiv.org/abs/2203.16102)

#### Our implementation

- We implement CN as the *direct BN replacement* described in the paper: **GroupNorm (no affine) → BatchNorm (with the
  original BN affine + running stats)**. Concretely, our CN forward does:
  - `group_norm(x, G, weight=None, bias=None)` (so GN contributes *only* normalization, no learned scale/shift), then
  - `batch_norm(...)` using the carried-over `running_mean`, `running_var`, `weight`, and `bias`.
- We replace **only `nn.BatchNorm2d`** modules (`replace_batchnorm2d`). If a model uses `BatchNorm1d` (e.g., MLPs) we
  do **not** rewrite those layers.
- We expose **`groups`** as a parameter and provide common presets (4/8/16/32/64). The paper typically uses a fixed
  group count (commonly 32 in vision backbones); in this repo you can change it via controller config.
- We intentionally keep CN as a **pure backbone rewrite** (no extra losses, no extra buffers/datasets, no task-specific
  logic). This matches the “drop-in normalization replacement” spirit, but we do not attempt to
  reproduce every training
  detail/setup from the paper’s experiments (online CL protocols, specific baselines, etc.).

### TBBN (task-balanced batch normalization)

[**Paper**](https://arxiv.org/abs/2201.12559)

#### Our implementation

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
    `B_c = train_mb_size` (current) and `B_p` equal to the replay mini-batch size
    (Avalanche `mem_mb_size` / `batch_size_mem`).
    TBBN is only wired for replay-backed backbone training (`backbone.training.strategy.name: replay`) and validates
    the partition accordingly.
  - **Explicit partition (no total-batch inference):** `(B_c, B_p)` are not derived from a
    single “total batch size”.
    `B_c` comes from `backbone.training.batch_size`; `B_p` comes from
    `backbone.training.strategy.kwargs.batch_size_mem` when provided, and otherwise falls back to
    `backbone.training.batch_size` (matching Replay defaults).
  - **Last minibatch robustness:** some strategies / dataloaders can yield a last minibatch whose current-part size is
    not divisible by the configured split factor. We reduce splits via `gcd` per minibatch to avoid invalid reshapes; a
    stricter reference implementation may simply assume divisibility and error out.

### BaCE

[**Paper**](https://openreview.net/forum?id=EOTgj37XNM)

#### Our implementation

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
- The KNN bank in this repo is built over the **current experience dataset** (with the teacher’s
  representation), not a
  specialized multi-source retrieval system. This is a pragmatic approximation of BaCE’s “use prior/old knowledge to
  assist adaptation” idea.
- We update the teacher **once per epoch** (EMA). If an official implementation updates more frequently, results may
  differ.
- We detect “current vs replay” samples by **class membership in the minibatch labels**, not by
  relying on a particular
  sampler ordering. This makes the controller more robust across Avalanche strategies, but it may not match an exact
  paper protocol if the protocol assumes a strict batch construction process.
