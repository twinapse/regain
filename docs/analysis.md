# Analysis guide

This document summarizes the analysis artifact contract produced by `python -m regain.cli.run_analysis`.

## Collected tables

The `collect` stage writes:

- `tables/run_metrics.jsonl`: one row per MLflow run with run metadata, repair-budget metadata, and aggregate metrics.
- `tables/experience_metrics.jsonl`: one row per `(run, experience)` pair with per-task accuracies, calibration
  diagnostics, and baseline-only repair diagnostics.

## Derived repair outcomes

The `frontier` stage joins `experience_metrics.jsonl` with `run_metrics.jsonl` on `run_id` and writes:

- `tables/repair_outcomes.jsonl`

`repair_outcomes.jsonl` is the canonical controller-on outcome table. Each row contains:

- run and experiment identity fields
- scenario, strategy, seed, controller, and budget metadata
- `is_no_op_action`, `action_repair_budget_fraction`, and `action_repair_budget_total`
- normalized `A_ref`, `A_post`, and `A_ctrl`
- row-level repair metrics such as `forgetting`, `absolute_recovery`, `residual_forgetting`, `rho`, `helped`, `harmed`,
  and `harm_magnitude`
- `source_stage = "collect"` for experiment-logged repair rows and `source_stage = "no_op"` for synthesized no-op
  actions

Rows without valid controller-on outcomes are excluded from this table. Raw prevention rows, backbone rows, and raw
no-op rows remain in the collected tables and do not contribute directly to the derived frontier summaries. The frontier
stage then synthesizes a no-op action for each observed repair-comparison setting. These no-op rows use `A_ctrl =
A_post`, zero recovery, zero harm, zero repair cost, `run.latency.ms_ratio = 1.0`, and `source_stage = "no_op"`. They
are included in `repair_outcomes`, frontier summaries, Pareto analysis, and `repair_selection` so downstream routers can
learn when not to repair. No-op rows receive a deterministic MLflow-compatible `run_id` derived from the normalized
repair setting and shared across all synthesized task rows in that setting, plus a run name formatted as
`no_op-<scenario>-<backbone>-<strategy>-budget_<0_to_100>-seed_<seed>`.

## Frontier outputs

The canonical frontier-stage artifacts are:

- `frontier/repair_frontier.csv`
- `frontier/repair_pareto.csv`
- `frontier/repair_impact.csv`
- `frontier/repair_selection.csv`
- `frontier/manifest.json`

Highlights:

- `repair_frontier.csv` contains one row per
  `experiment_id × scenario × backbone_name × strategy_name × seed × controller × b × repair_budget_fraction × repair_budget_total`
  setting plus utility and Pareto annotations.
- `repair_pareto.csv` summarizes Pareto frequency and aggregate utility by controller and budget.
- `repair_impact.csv` aggregates seed-level frontier rows into scenario/strategy impact summaries.
- `repair_selection.csv` is the repair-selection dataset with pre-repair inputs, pivoted controller outcome columns,
  oracle labels, and `oracle_margin_vs_best_static_controller`.
- Repair-selection rows are keyed by
  `experiment_id × scenario × backbone_name × strategy_name × seed × b × repair_budget_fraction × repair_budget_total`.
- `manifest.json` records controller-id normalization, suspicious accuracy values, Pareto warnings, and plot
  save/skip details.

## Predictive outputs

The `predictive` stage writes:

- `predictive/predictive_correlations.csv`

`predictive_correlations.csv` contains diagnostic-association summaries for each `(controller, budget)` group, including
correlation coefficients and valid-task counts.

## Curves and plots

The `curves` stage writes:

- `curves/recoverability_curve.csv`
- `curves/task_age_rho.csv`
- `curves/calibration_vs_budget.csv`
- `curves/latency_vs_budget.csv`

The plotting entrypoints consume the frontier artifacts and save:

- `plots/recovery_vs_budget__<scenario>__<backbone_name>__<strategy_name>.png`
- `plots/harm_vs_recovery.png`
- `plots/utility_vs_cost.png`
- `plots/utility_delta__<controller_a>__<controller_b>.png`
- `plots/harm_vs_budget__<scenario>__<backbone_name>.png`

When a requested plot cannot be generated because the required data is missing, the reason is written to
`frontier/manifest.json`.
Saved plot filenames are recorded in `frontier/manifest.json` only when saving into the analysis output and the
corresponding plot artifacts are publishable; `generate_plots --output-dir` writes plots externally without mutating
the source analysis manifest.
