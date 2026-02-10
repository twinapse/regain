# Debugging

This document covers debug instrumentation for repair controllers, including configuration, utilities, and
MLflow metrics.

---

## Contents

1) [Enable debug mode](#1-enable-debug-mode)  
2) [Debug utilities (code)](#2-debug-utilities-code)  
3) [Logging conventions (MLflow)](#3-logging-conventions-mlflow)  
4) [Repair diagnostics (per fit event)](#4-repair-diagnostics-per-fit-event)  
5) [Repair health score](#5-repair-health-score)  
6) [Interpretation tips](#6-interpretation-tips)  
7) [Implementation notes](#7-implementation-notes)

---

## 1) Enable debug mode

Add the `debug` flag to the experiment YAML:

```yaml
debug: true
```

Notes:
- Debug mode instruments repair controllers only; prevention controllers are not instrumented.
- If `debug: true` but no repair controller is configured, the run logs
  `debug.skip_reason=no_repair_controller`.
- If `debug` is omitted, debug instrumentation is disabled.

---

## 2) Debug utilities (code)

### Package
`regain.debug` provides the instrumentation helpers.

### Key components
- `regain.debug.avalanche_utils.DebugRepairControllerPlugin`:
  wraps `RepairControllerPlugin` and runs diagnostics pre/post controller fitting.
- `regain.debug.metrics.compute_repair_diagnostics(...)`:
  evaluates repair datasets in base or controller mode and returns diagnostics.
- `regain.debug.metrics.compute_repair_health_score(...)`:
  computes the repair health score and its components from pre/post diagnostics.

---

## 3) Logging conventions (MLflow)

### Namespace
All debug metrics are logged as:
- `debug.<metric_name>`
- Per-experience fit events suffix with `.exp###` (zero-padded).

Examples:
- `debug.repair_ce_ctrl_pre.exp003`
- `debug.repair_health.exp003`

### Steps
Metrics are logged with numeric steps to keep curves readable:
- Per-experience fit after exp `exp_idx`:
  `step = (exp_idx + 1) * num_epochs_per_experience`
- Final fit at end of training:
  `step = num_experiences * num_epochs_per_experience`

---

## 4) Repair diagnostics (per fit event)

Diagnostics are run on the same repair dataset used for fitting (combined repair set so far).

### Invariants and label-space alignment
> ℹ️ **Class label invariant:** class label == logit index. Logit column `c` corresponds to global class ID `c`;
> controllers must not permute the class axis.

Debug diagnostics may evaluate in a shared label space `K` (chosen per fit event). In that case, logits are sliced to
`logits[:, :K]` and samples with labels outside `[0, K-1]` are ignored to avoid invalid CE targets. The
`debug.repair_n_samples_*` metrics therefore count the number of **valid evaluated samples** in that label space.

> ℹ️ **Valid sample:** a repair example whose target label falls within the evaluated range after capping:
> `0 <= y < K`. Samples outside `[0, K-1]` are ignored and do not contribute to CE, accuracy, or histogram denominators.

### Pre-fit
- Base (controller corrections not applied)
- Ctrl (controller corrections applied)

### Post-fit
- Ctrl (controller corrections applied)
- Base post-fit is not logged (base model does not change during pure repair fitting)

### Metrics logged (ctrl)
All are logged pre/post and as deltas (post - pre):

- `debug.repair_ce_ctrl_pre[.exp###]`: Mean cross-entropy on repair set before fitting (controller corrections applied).
- `debug.repair_ce_ctrl_post[.exp###]`: Mean cross-entropy after fitting (controller corrections applied).
- `debug.repair_ce_ctrl_delta[.exp###]`: Post - pre CE (negative is better).
- `debug.repair_top1_ctrl_pre[.exp###]`: Top-1 accuracy before fitting (controller corrections applied).
- `debug.repair_top1_ctrl_post[.exp###]`: Top-1 accuracy after fitting (controller corrections applied).
- `debug.repair_top1_ctrl_delta[.exp###]`: Post - pre top-1 (positive is better).
- `debug.repair_logit_l2_ctrl_pre[.exp###]`: Mean L2 norm of logits per sample before fitting.
- `debug.repair_logit_l2_ctrl_post[.exp###]`: Mean L2 norm after fitting.
- `debug.repair_logit_l2_ctrl_delta[.exp###]`: Post - pre logit L2 (large negative can signal collapse).
- `debug.repair_entropy_ctrl_pre[.exp###]`: Mean predictive entropy before fitting.
- `debug.repair_entropy_ctrl_post[.exp###]`: Mean predictive entropy after fitting.
- `debug.repair_entropy_ctrl_delta[.exp###]`: Post - pre entropy (large positive can signal collapse).
- `debug.repair_pred_unique_ctrl_pre[.exp###]`: Number of unique predicted classes before fitting.
- `debug.repair_pred_unique_ctrl_post[.exp###]`: Number of unique predicted classes after fitting.
- `debug.repair_pred_unique_ctrl_delta[.exp###]`: Post - pre unique class count (negative is worse).
- `debug.repair_pred_max_frac_ctrl_pre[.exp###]`: Max fraction of samples predicted as a single class before fitting.
- `debug.repair_pred_max_frac_ctrl_post[.exp###]`: Max fraction after fitting.
- `debug.repair_pred_max_frac_ctrl_delta[.exp###]`: Post - pre max fraction (positive is worse).
- `debug.repair_pred_entropy_ctrl_pre[.exp###]`: Entropy of predicted-class histogram before fitting.
- `debug.repair_pred_entropy_ctrl_post[.exp###]`: Histogram entropy after fitting.
- `debug.repair_pred_entropy_ctrl_delta[.exp###]`: Post - pre histogram entropy (negative is worse).
- `debug.repair_n_samples_ctrl_pre[.exp###]`: Number of valid repair samples evaluated pre-fit.
- `debug.repair_n_samples_ctrl_post[.exp###]`: Number of valid repair samples evaluated post-fit.

### Optional base pre metrics
Logged when available:
- `debug.repair_ce_base_pre[.exp###]`: Mean cross-entropy before fitting (controller corrections not applied).
- `debug.repair_top1_base_pre[.exp###]`: Top-1 accuracy before fitting (controller corrections not applied).
- `debug.repair_logit_l2_base_pre[.exp###]`: Mean logit L2 norm before fitting (controller corrections not applied).
- `debug.repair_entropy_base_pre[.exp###]`: Mean predictive entropy before fitting (controller corrections not applied).
- `debug.repair_pred_unique_base_pre[.exp###]`: Unique predicted classes before fitting (controller corrections not applied).
- `debug.repair_pred_max_frac_base_pre[.exp###]`: Max predicted-class fraction before fitting (controller corrections not applied).
- `debug.repair_pred_entropy_base_pre[.exp###]`: Predicted-class histogram entropy before fitting (controller corrections not applied).
- `debug.repair_n_samples_base_pre[.exp###]`: Number of valid repair samples evaluated (controller corrections not applied).

### Optional artifacts
Prediction histogram JSONs may be logged:
- `debug_pred_hist_pre_exp###.json`
- `debug_pred_hist_post_exp###.json`
- For end-of-training fits (no exp index): `debug_pred_hist_pre.json`, `debug_pred_hist_post.json`

### Skip markers
Diagnostics can be skipped for multiple reasons (for example: empty repair set, no valid samples after shared
label-space capping, or an inability to build comparable pre/post diagnostics). Health score is skipped whenever
diagnostics are unavailable/non-comparable, or if health computation fails:
- `debug.repair_diagnostics_skipped[.exp###] = 1`: Diagnostics did not run or were not comparable.
- `debug.repair_health_skipped[.exp###] = 1`: Health score did not run (missing/non-comparable diagnostics, or error).

---

## 5) Repair health score

The repair health score is a single scalar in `[0, 1]` that combines:
1) performance improvement (CE / top-1),
2) confidence stability (logit norm / entropy),
3) prediction diversity stability (class coverage / max class fraction / pred-entropy).

### Per-fit metrics
Logged per fit event:
- `debug.repair_health[.exp###]`: Health score in [0, 1] (higher is healthier).
- `debug.repair_health_delta[.exp###]`: Health - neutral baseline (signed).
- `debug.repair_health_neutral[.exp###]`: Neutral baseline score (no-change scenario).

Component scores:
- `debug.repair_health_s1_perf[.exp###]`: Performance subscore (CE + top-1 improvement).
- `debug.repair_health_s2_conf[.exp###]`: Confidence stability subscore (logit norm + entropy).
- `debug.repair_health_s3_div[.exp###]`: Diversity stability subscore (class coverage + max fraction + pred-entropy).

Raw deltas:
- `debug.repair_health_r_ce[.exp###]`: Relative CE improvement (positive is better).
- `debug.repair_health_d_acc[.exp###]`: Top-1 accuracy delta (positive is better).
- `debug.repair_health_r_norm[.exp###]`: Logit norm ratio post/pre (ideal near 1.0).
- `debug.repair_health_d_ent[.exp###]`: Entropy delta post - pre (positive is worse).
- `debug.repair_health_d_maxfrac[.exp###]`: Max fraction delta post - pre (positive is worse).
- `debug.repair_health_d_unique[.exp###]`: Unique-class fraction drop (positive is worse; only when available).
- `debug.repair_health_d_predent[.exp###]`: Pred entropy drop (positive is worse; only when available).

### End-of-run aggregates
Logged once per run:
- `debug.repair_health_mean`: Mean health score across fit events.
- `debug.repair_health_min`: Minimum health score across fit events.
- `debug.repair_health_final`: Health score from the final fit event.

---

## 6) Interpretation tips

- CE improves + top-1 improves: controller behaving normally.
- CE improves without top-1: potential soft degeneracy.
- Huge logit norm drop or entropy spike: confidence collapse.
- Predicted classes collapse to a few classes: label mapping or fitting instability.

---

## 7) Implementation notes

- Diagnostics use the same repair dataloader builder as fitting (`build_repair_dataloader`) with `shuffle=False`.
- Labels are coerced to `torch.long` in the diagnostics path to avoid dtype errors in `F.cross_entropy`.
- By default, diagnostics are capped at 2048 samples to limit overhead.
- Debug diagnostics preserve Python/NumPy/Torch RNG state to avoid affecting subsequent training.
