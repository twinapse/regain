# Methods

This document summarizes the controller methods used in this codebase. Each family is organized in two parts: a
`Method summary` table that captures registry metadata and references, followed by `Implementation notes` that record
the practical details, approximations, and status caveats that are easy to forget later.

Methods marked **(not implemented)** remain in the document to preserve the repair-family taxonomy
used throughout the repository.

<!-- toc -->

- [Post-hoc repair controllers](#post-hoc-repair-controllers)
  - [Readout repair](#readout-repair)
    - [Linear probe](#linear-probe)
    - [Low-rank probe (not implemented)](#low-rank-probe-not-implemented)
    - [Prototype classifier](#prototype-classifier)
    - [Ridge probe (not implemented)](#ridge-probe-not-implemented)
  - [Bias / calibration repair](#bias--calibration-repair)
    - [BiC (bias correction)](#bic-bias-correction)
    - [IL2M (class incremental learning with dual memory)](#il2m-class-incremental-learning-with-dual-memory)
    - [Logit bias](#logit-bias)
    - [T-CIL-Lite calibration](#t-cil-lite-calibration)
    - [Temperature scaling](#temperature-scaling)
    - [Weight Aligning](#weight-aligning)
  - [Statistical drift repair](#statistical-drift-repair)
    - [DPCR-style repair (not implemented)](#dpcr-style-repair-not-implemented)
    - [MACIL (not implemented)](#macil-not-implemented)
  - [Modulation repair](#modulation-repair)
    - [Channel block gain](#channel-block-gain)
    - [Channel stage gain](#channel-stage-gain)
    - [Input-conditioned block gain](#input-conditioned-block-gain)
    - [Input-conditioned stage gain](#input-conditioned-stage-gain)
    - [Scalar block gain](#scalar-block-gain)
    - [Scalar stage gain](#scalar-stage-gain)
    - [TMCL (not implemented)](#tmcl-not-implemented)
  - [Test-time repair](#test-time-repair)
    - [ARC (adaptive retention and correction) (not implemented)](#arc-adaptive-retention-and-correction-not-implemented)
- [Training-time / prevention controllers](#training-time--prevention-controllers)
  - [BaCE](#bace)
  - [CN (continual normalization)](#cn-continual-normalization)
  - [TBBN (task-balanced batch normalization)](#tbbn-task-balanced-batch-normalization)

## Post-hoc repair controllers

Post-hoc repair controllers act after backbone training boundaries and do not update the backbone
trajectory. They are used to measure which failure modes are recoverable under constrained repair data,
controller capacity, and inference cost.

### Readout repair

Readout repair methods test whether useful class information remains in the frozen representation
but is no longer expressed well by the trained classifier head.

#### Method summary

| Method | Registry name | Class name | Reference | Purpose |
| --- | --- | --- | --- | --- |
| Linear probe | `linear_probe` | `LinearProbeController` | [*Knowledge Accumulation and Feature Forgetting*](https://arxiv.org/abs/2304.00933) | Evaluate representation quality and readout-repairability. |
| Low-rank probe | `N/A` | `N/A` | `N/A` | Low-rank readout baseline for lower-parameter repair comparisons. |
| Prototype classifier | `prototype_blend` | `PrototypeBlendController` | iCaRL nearest-mean-of-exemplars [classifier](https://arxiv.org/abs/1611.07725) | Blend frozen-head logits with class-prototype similarity scores. |
| Ridge probe | `N/A` | `N/A` | `N/A` | Ridge-regularized frozen-feature readout baseline for small repair budgets. |

#### Implementation notes

##### Linear probe

- We use this method to evaluate the quality of learned representations and to measure readout-repairability.

##### Low-rank probe (not implemented)

- Parameter-efficient readout baseline that constrains probe updates to a low-rank factorization.
- Useful when frontier comparisons need a readout method with lower parameter count than a full linear probe.

##### Prototype classifier

This is a generic nearest-class-prototype baseline rather than a faithful reproduction of one method.
It is related to nearest-mean classifiers used in class-incremental learning, for example iCaRL's
[nearest-mean-of-exemplars classifier](https://arxiv.org/abs/1611.07725), but `prototype_blend`
keeps the trained head and blends prototype similarity scores into its logits.

- Feature-space classifier built from class prototypes estimated on repair data.
- Useful for testing whether forgetting is primarily a prototype-drift problem rather than a classifier-head problem.
- It blends frozen classifier logits with frozen-feature prototype similarity scores
  estimated from repair examples.

##### Ridge probe (not implemented)

- Ridge-regularized linear readout baseline on frozen representations.
- Useful for testing whether modest shrinkage improves readout repair under small repair budgets.

### Bias / calibration repair

Bias and calibration repair methods test whether forgetting can be recovered by correcting logits,
temperatures, class-level bias, or confidence structure with fixed post-hoc parameters rather than
changing the backbone representation.

#### Method summary

| Method | Registry name | Class name | Reference | Purpose |
| --- | --- | --- | --- | --- |
| BiC (bias correction) | `bic` | `BiCController` | [arXiv:1905.13260](https://arxiv.org/abs/1905.13260) | Fit a bias layer on repair data to correct classifier bias. |
| IL2M (class incremental learning with dual memory) | `il2m` | `IL2MController` | [IEEE:9009019](https://ieeexplore.ieee.org/document/9009019) | Accumulate class statistics and rectify logits after each experience. |
| Logit bias | `logit_bias` | `LogitBiasController` | N/A (repository baseline) | Learn an additive per-class bias vector on repair data. |
| T-CIL-Lite calibration | `tcil_lite` | `TCILLiteController` | [arXiv:2503.22163](https://arxiv.org/abs/2503.22163) | Fit one temperature per experience class group as a lightweight T-CIL approximation. |
| Temperature scaling | `temperature_scaling` | `TemperatureScalingController` | [arXiv:1706.04599](https://arxiv.org/abs/1706.04599) | Calibrate frozen logits with one global temperature. |
| Weight Aligning | `weight_aligning` | `WeightAligningController` | [arXiv:1911.07053](https://arxiv.org/abs/1911.07053) | Rescale new-class logits using classifier-head weight norms. |

#### Implementation notes

##### BiC (bias correction)

- Adapts Avalanche's `BiCPlugin` stage-2 bias-correction procedure.
- Backbone training is still provided by the shared experiment backbone strategy; BiC fits
  a bias layer on the repair stream and applies it during post-hoc evaluation.

##### IL2M (class incremental learning with dual memory)

- Adapts Avalanche's `IL2MPlugin`.
- The controller accumulates IL2M statistics from repair data after each experience and
  applies IL2M rectification during post-hoc evaluation.

##### Logit bias

- Learns an additive per-class bias vector on repair data and applies `logits' = logits + b`
  during post-hoc evaluation.

##### T-CIL-Lite calibration

- Placeholder for class-incremental calibration methods that adjust temperatures or
  confidence structure more flexibly than a single global scalar.
- Useful for documenting the space between global temperature scaling and richer class-specific correction methods.
- It is a lightweight approximation of T-CIL: it fits one temperature per observed
  experience class group. It should not be considered a faithful T-CIL reproduction.

##### Temperature scaling

- One-parameter calibration baseline that rescales logits uniformly.
- Useful as a minimal-capacity reference point below per-class bias methods such as `logit_bias` or BiC.
- Optimizes one scalar temperature on frozen repair-set logits using NLL.

##### Weight Aligning

- Estimates the classifier-head weight-norm ratio between old and newly introduced classes
  and applies that near-zero-cost scalar to the logits of the new classes.

### Statistical drift repair

Statistical drift repair methods test whether forgetting is recoverable by correcting shifts in
feature or logit statistics, such as means, covariances, prototypes, or analytically reconstructed
classifiers.

#### Method summary

| Method | Registry name | Class name | Reference | Purpose |
| --- | --- | --- | --- | --- |
| DPCR-style repair | `N/A` | `N/A` | `N/A` | Placeholder for prototype/distribution-reconstruction repair methods. |
| MACIL | `N/A` | `N/A` | `N/A` | Compensate semantic drift through mean/covariance-shift calibration. |

#### Implementation notes

##### DPCR-style repair (not implemented)

- Placeholder for prototype- or distribution-reconstruction repair methods that correct drift
  using class statistics.
- Useful for documenting statistical repair baselines that sit between logit-only correction
  and fitting a new probe.
- No DPCR-style controller is currently implemented in this repository.

##### MACIL (not implemented)

- MACIL is a statistical drift repair baseline for task-agnostic class-incremental learning.
- It frames semantic drift as mean/covariance shift and applies lightweight calibration to compensate for that drift.

### Modulation repair

Modulation repair methods test whether forgetting is recoverable by applying input-, layer-,
channel-, or state-dependent modulations rather than replacing only the final readout.

#### Method summary

| Method | Registry name | Class name | Reference | Purpose |
| --- | --- | --- | --- | --- |
| Channel block gain | `channel_block` | `ChannelBlockGainController` | N/A (repository-native modulation baseline) | Learn grouped per-channel gains for each resolved block. |
| Channel stage gain | `channel_stage` | `ChannelStageGainController` | N/A (repository-native modulation baseline) | Learn grouped per-channel gains for each resolved stage. |
| Input-conditioned block gain | `conditioned_block` | `InputConditionedBlockGainController` | N/A (repository-native modulation baseline) | Predict per-example block gains from backbone features. |
| Input-conditioned stage gain | `conditioned_stage` | `InputConditionedStageGainController` | N/A (repository-native modulation baseline) | Predict per-example stage gains from backbone features. |
| Scalar block gain | `scalar_block` | `ScalarBlockGainController` | N/A (repository-native modulation baseline) | Learn one multiplicative gain per resolved block. |
| Scalar stage gain | `scalar_stage` | `ScalarStageGainController` | N/A (repository-native modulation baseline) | Learn one multiplicative gain per resolved stage. |
| TMCL | `N/A` | `N/A` | `N/A` | Top-down modulation baseline for continual learning. |

#### Implementation notes

All six gain-controller variants are repository-native repair baselines rather than reproductions
of one published method. They are fit on repair data with the backbone frozen and act only during
post-hoc evaluation.

##### Channel block gain

- Learns grouped per-channel gains for each resolved block.
- It serves as an exploratory modulation probe for testing whether block-level channel
  modulation can recover forgetting under constrained controller budgets.

##### Channel stage gain

- Learns grouped per-channel gains for each resolved stage.
- It serves as an exploratory modulation probe for testing whether stage-level channel
  modulation can recover forgetting under constrained controller budgets.

##### Input-conditioned block gain

- Predicts per-example block gains from backbone features with a small MLP, then re-runs
  the model with those input-conditioned gains applied.
- It serves as an exploratory modulation probe for testing whether per-example block-level
  modulation can recover forgetting under constrained controller budgets.

##### Input-conditioned stage gain

- Predicts per-example stage gains from backbone features with a small MLP, then re-runs
  the model with those input-conditioned gains applied.
- It serves as an exploratory modulation probe for testing whether per-example stage-level
  modulation can recover forgetting under constrained controller budgets.

##### Scalar block gain

- Learns one multiplicative gain per resolved block and applies those gains via temporary
  forward hooks during post-hoc evaluation.
- It serves as an exploratory modulation probe for testing whether block-level feature
  modulation can recover forgetting under constrained controller budgets.

##### Scalar stage gain

- Learns one multiplicative gain per resolved stage and applies those gains via temporary
  forward hooks during post-hoc evaluation.
- It serves as an exploratory modulation probe for testing whether stage-level feature
  modulation can recover forgetting under constrained controller budgets.

##### TMCL (not implemented)

- TMCL is a modulation-based continual learning baseline built around top-down modulations and consolidation.
- It is relevant as a comparator for controllers that apply task-agnostic or
  input-conditioned gains to intermediate representations.

### Test-time repair

Test-time repair methods test whether forgetting can be recovered by making online control
decisions during evaluation, such as detecting past-task samples or applying input-conditioned
correction, rather than relying only on a pre-fit repair module learned from a labeled repair
stream.

#### Method summary

| Method | Registry name | Class name | Reference | Purpose |
| --- | --- | --- | --- | --- |
| ARC (adaptive retention and correction) | `N/A` | `N/A` | `N/A` | Detect past-task samples and apply online correction at test time. |

#### Implementation notes

##### ARC (adaptive retention and correction) (not implemented)

- ARC is a test-time repair baseline for classifier bias and task-confusion failure modes.
- It detects whether a sample is likely to come from a past task and applies adaptive
  retention/correction at test time.
- Unlike fixed repair-set controllers, it can use test-time adaptation behavior rather than
  only a pre-collected labeled repair stream.

## Training-time / prevention controllers

Training-time controllers modify the learning trajectory itself. They are not purely post-hoc
repair methods, but they provide prevention-style baselines for understanding when repair is
sufficient and when training-time intervention is needed.

### Method summary

| Method | Registry name | Class name | Reference | Purpose |
| --- | --- | --- | --- | --- |
| BaCE | `bace` | `BaCEController` | [OpenReview:EOTgj37XNM](https://openreview.net/forum?id=EOTgj37XNM) | Combine KNN-assisted adaptation, distillation, and replay-aware loss terms during training. |
| CN (continual normalization) | `cn` | `ContinualNormalizationController` | [arXiv:2203.16102](https://arxiv.org/abs/2203.16102) | Replace BatchNorm2d layers with continual-normalization layers during training. |
| TBBN (task-balanced batch normalization) | `tbbn` | `TaskBalancedBatchNormController` | [arXiv:2201.12559](https://arxiv.org/abs/2201.12559) | Rebalance current-vs-replay batch statistics during training. |

### Implementation notes

#### BaCE

BaCE plugs into Avalanche training via `TrainingObjectiveControllerInterface`.

What we implement:

- **Teacher / student setup with epoch-level teacher updates.**
  - We keep a frozen teacher copy initialized at the start of each experience.
  - We update teacher parameters via **EMA at the end of each training epoch** (rather than per-iteration).
- **Effect\_new (learning new classes with “old” causal effect included) via a joint-score KNN objective:**
  - We build a **KNN feature bank** in the *teacher’s* feature space over the **current experience dataset**.
    - For efficiency, the bank is built once per epoch and can be subsampled (`bank_max_samples`).
  - For current-task samples in a minibatch, we compute:
    - student self-scores (`softmax(logits)`), and
    - neighbor contributions from the **student’s predictions on retrieved neighbors**, weighted
      and mixed with self-score via `w0`.
  - Neighbor weights are computed by an **inverse-distance** rule (with eps for stability)
    and we exclude self-matches / duplicates using a small distance threshold.
  - The loss is a negative log-likelihood on the target probability under these **joint scores**.
- **Effect\_old (stabilizing old classes) via distillation on old-class scores:**
  - On current-task samples, we compute **KL divergence** between student and teacher softmax
    scores restricted to the **old-class subset**.
  - If replay/buffer samples are present in the minibatch, we switch to a different weighting
    (`alpha_with_buffer` vs `alpha_no_buffer`) consistent with the idea that buffer availability
    changes the required strength of the effect.
- **Replay enhancement (when replay samples exist) in a DER++-style form:**
  - CE on replay samples (standard supervised loss),
  - plus MSE distillation between student and teacher **old-class logits** on replay samples.

Important differences / simplifications vs a faithful reproduction:

- Our implementation is **scoped to image classification-style models** that expose a
  `.backbone` (or `.encoder`) returning 2D features; we do not reproduce the paper’s broader
  PTM/NLP setups.
- The KNN bank in this repo is built over the **current experience dataset** (with the teacher’s
  representation), not a specialized multi-source retrieval system. This is a pragmatic
  approximation of BaCE’s “use prior/old knowledge to assist adaptation” idea.
- We update the teacher **once per epoch** (EMA). If an official implementation updates more
  frequently, results may differ.
- We detect “current vs replay” samples by **class membership in the minibatch labels**, not
  by relying on a particular sampler ordering. This makes the controller more robust across
  Avalanche strategies, but it may not match an exact paper protocol if the protocol assumes a
  strict batch construction process.

#### CN (continual normalization)

- Implements the *direct BN replacement* described in the paper:
  **GroupNorm (no affine) → BatchNorm (with the original BN affine + running stats)**.
  Concretely, our CN forward does:
  - `group_norm(x, G, weight=None, bias=None)` (so GN contributes *only* normalization,
    no learned scale/shift), then
  - `batch_norm(...)` using the carried-over `running_mean`, `running_var`, `weight`, and `bias`.
- We replace **only `nn.BatchNorm2d`** modules (`replace_batchnorm2d`). If a model uses
  `BatchNorm1d` (e.g., MLPs) we do **not** rewrite those layers.
- We expose **`groups`** as a parameter and provide common presets (4/8/16/32/64).
  The paper typically uses a fixed group count (commonly 32 in vision backbones); in this repo
  you can change it via controller config.
- We intentionally keep CN as a **pure backbone rewrite** (no extra losses, no extra
  buffers/datasets, no task-specific logic). This matches the “drop-in normalization replacement”
  spirit, but we do not attempt to reproduce every training detail/setup from the paper’s
  experiments (online CL protocols, specific baselines, etc.).

#### TBBN (task-balanced batch normalization)

- Builds on the `TaskBalancedBatchNorm` drop-in layer and:
  1) rewrites all `BatchNorm2d` layers, and
  2) calls `set_number_of_task(t)` at the **start of each experience** (0-indexed), as required
     by the method.
- Training-time behavior follows the paper’s core idea: correct the BN statistics/gradient bias
  caused by the **current-vs-replay minibatch imbalance** by constructing a “task-balanced”
  batch using reshape/repeat operations (current samples are split; replay samples are repeated
  across splits).
- Inference-time behavior is intentionally **BN-like**: once in eval mode, the layer behaves
  like vanilla BN using its running statistics (the “balancing” is a training-only mechanism).
- Assumptions/differences to be aware of:
  - **Minibatch layout + partition:** minibatches must be `[current-task samples | replay
    samples]` with `B_c = train_mb_size` (current) and `B_p` equal to the replay mini-batch size
    (Avalanche `mem_mb_size` / `batch_size_mem`).
    TBBN is only wired for replay-backed backbone training
    (`backbone.training.strategy.name: replay`) and validates the partition accordingly.
  - **Explicit partition (no total-batch inference):** `(B_c, B_p)` are not derived from a
    single “total batch size”. `B_c` comes from `backbone.training.batch_size`; `B_p` comes
    from `backbone.training.strategy.kwargs.batch_size_mem` when provided, and otherwise falls
    back to `backbone.training.batch_size` (matching Replay defaults).
  - **Last minibatch robustness:** some strategies / dataloaders can yield a last minibatch
    whose current-part size is not divisible by the configured split factor. We reduce splits
    via `gcd` per minibatch to avoid invalid reshapes; a stricter reference implementation may
    simply assume divisibility and error out.
