# Debugging

This document covers repair-controller debug instrumentation and its MLflow metrics.

<!-- toc -->

- [Namespace](#namespace)
- [Diagnostic metrics](#diagnostic-metrics)
- [Health metrics](#health-metrics)

## Namespace

All debug metrics are logged under:

- `run.debug.*`

Per-experience entries append `.exp###` (zero-padded). For base/ctrl variants, the variant is always the final segment.

Examples:

- `run.debug.repair.ce.pre.exp003.ctrl`
- `run.debug.repair.top1.delta.exp003.ctrl`
- `run.debug.repair.ce.pre.exp003.base`
- `run.debug.repair.health.exp003`

## Diagnostic metrics

Per fit event, the plugin logs pre/post controller diagnostics and deltas:

- `run.debug.repair.ce.pre[.exp###].ctrl`
- `run.debug.repair.ce.post[.exp###].ctrl`
- `run.debug.repair.ce.delta[.exp###].ctrl`
- `run.debug.repair.top1.pre[.exp###].ctrl`
- `run.debug.repair.top1.post[.exp###].ctrl`
- `run.debug.repair.top1.delta[.exp###].ctrl`
- `run.debug.repair.logit_l2.pre[.exp###].ctrl`
- `run.debug.repair.logit_l2.post[.exp###].ctrl`
- `run.debug.repair.logit_l2.delta[.exp###].ctrl`
- `run.debug.repair.entropy.pre[.exp###].ctrl`
- `run.debug.repair.entropy.post[.exp###].ctrl`
- `run.debug.repair.entropy.delta[.exp###].ctrl`
- `run.debug.repair.pred_unique.pre[.exp###].ctrl`
- `run.debug.repair.pred_unique.post[.exp###].ctrl`
- `run.debug.repair.pred_unique.delta[.exp###].ctrl`
- `run.debug.repair.pred_max_frac.pre[.exp###].ctrl`
- `run.debug.repair.pred_max_frac.post[.exp###].ctrl`
- `run.debug.repair.pred_max_frac.delta[.exp###].ctrl`
- `run.debug.repair.pred_entropy.pre[.exp###].ctrl`
- `run.debug.repair.pred_entropy.post[.exp###].ctrl`
- `run.debug.repair.pred_entropy.delta[.exp###].ctrl`
- `run.debug.repair.n_samples.pre[.exp###].ctrl`
- `run.debug.repair.n_samples.post[.exp###].ctrl`

Optional base pre-fit diagnostics:

- `run.debug.repair.<metric>.pre[.exp###].base` for the same diagnostic set above.

Skip markers:

- `run.debug.repair.diagnostics_skipped[.exp###] = 1`
- `run.debug.repair.health.skipped[.exp###] = 1`

## Health metrics

Per fit event:

- `run.debug.repair.health[.exp###]`
- `run.debug.repair.health.delta[.exp###]`
- `run.debug.repair.health.neutral[.exp###]`
- `run.debug.repair.health.s1_perf[.exp###]`
- `run.debug.repair.health.s2_conf[.exp###]`
- `run.debug.repair.health.s3_div[.exp###]`
- `run.debug.repair.health.r_ce[.exp###]`
- `run.debug.repair.health.d_acc[.exp###]`
- `run.debug.repair.health.r_norm[.exp###]`
- `run.debug.repair.health.d_ent[.exp###]`
- `run.debug.repair.health.d_maxfrac[.exp###]`
- `run.debug.repair.health.d_unique[.exp###]` (when available)
- `run.debug.repair.health.d_predent[.exp###]` (when available)

Run-level aggregates:

- `run.debug.repair.health.avg`
- `run.debug.repair.health.min`
- `run.debug.repair.health.final`
