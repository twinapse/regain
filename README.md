# REGAIN: A Repairability Framework for Class-Incremental Learning

<!-- toc -->

- [Project overview](#project-overview)
- [Architecture](#architecture)
- [Protocol](#protocol)
- [Methods](#methods)
- [Experimental framework](#experimental-framework)
- [CLI usage](#cli-usage)
- [Debugging](#debugging)
- [Coding agent setup](#coding-agent-setup)

## Project overview

See [docs/overview.md](docs/overview.md) for research, domain, and technology-stack context.

See [docs/concepts.md](docs/concepts.md) for the glossary of core concepts.

## Architecture

See [docs/architecture.md](docs/architecture.md) for repository architecture patterns and implementation guidance.

## Protocol

See [docs/protocol.md](docs/protocol.md) for details on the research protocol: offline repairability mapping and
frontier construction, plus deployment-time repair selection and decision gates.

## Methods

See [docs/methods.md](docs/methods.md) for implementation notes on specific methods used in this codebase.

## Experimental framework

See [docs/experimental-framework.md](docs/experimental-framework.md) for details on the benchmarks and metrics used in
the experiments.

## CLI usage

The typical workflow is:

1) **Run experiments** (logs metrics to MLflow).
2) **Run analysis** on the logged runs (writes analysis artifacts, and optionally plots).
3) **Export** runs/analysis.

### 1) Run experiments (log metrics)

Provide one or more YAML configs (comma-separated) and run:

```bash
python -m regain.cli.run_experiment --config-files ./config/experiment_a.yaml,./config/experiment_b.yaml
```

You can also discover configs recursively from a directory:

```bash
python -m regain.cli.run_experiment --config-dir ./config
```

Optional MLflow overrides:

```bash
python -m regain.cli.run_experiment --config-dir ./config --tracking-uri file:./mlruns --artifact-location file:./mlruns
```

If omitted, MLflow uses its default behavior (environment variables such as `MLFLOW_TRACKING_URI`, configured profile,
or local default store).

### 2) Analyze logged runs (write artifacts, optionally plot)

`run_analysis` has subcommands:

* `collect`: download/aggregate finished parent MLflow runs into tidy JSONL tables
* `curves`: compute recoverability, task-age, calibration-vs-budget, and latency-vs-budget curves (requires `collect`)
* `frontier`: compute the efficiency frontier from the curve CSV (requires `curves` output)
* `predictive`: compute diagnostic-vs-repairability correlations (requires `collect`)
* `all`: run `collect + curves + frontier + predictive`

Experiment selection is standardized via one required selector:
* `--experiments`: comma-separated MLflow experiment names/ids
* `--config-files`: comma-separated experiment config files
* `--config-dir`: directory recursively searched for experiment config files

Notes:
* For repair-controller runs, predictive summaries and run-level `run.calibration.max_ece` in `runs_table` are
  baseline-only: sourced from base values in `analysis_artifacts.json`. `run.calibration.max_ece` is defined as
  `max(run.calibration.ece)` over artifact baseline vectors.
* Analysis collection requires each run to include `controller.type` (`none`, `repair`, or `prevention`) and
  `repair.split_fraction` parameters.
* Runs that fail collection validation are skipped and reported as run-level failures. Use `--allow-partial` if you want
  successful outputs to still be published when some runs/stages fail. If zero runs are successfully collected, no
  analysis outputs are published (exit code `1`).
* `repair_set_total` in analysis tables is:
  - `0` when `repair.split_fraction == 0.0`;
  - otherwise, the exact total non-empty line count across `repair/exp_*.txt` in `splits.tar.gz`.

Common flags:

* `--experiments` / `--config-files` / `--config-dir`: one required selector
* `--output-dir`: output directory root (experiment subdirectory is created under this path)
* `--tracking-uri`: optional MLflow tracking URI override
* `--show-plots`: display plots interactively
* `--save-plots`: save plots to `<output-dir>/<experiment>/plots`
* `--perf-key`: metric key to maximize for the frontier and plot in curves
* `--allow-partial`: publish successful outputs even if some stages fail
* `--overwrite`: replace existing `<output-dir>/<experiment>/*`

Examples:

```bash
# Step-by-step
python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results collect
python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results --show-plots curves
python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results --perf-key analysis.repair.rho.avg --save-plots frontier
python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results predictive

# One-shot (recommended for most use)
python -m regain.cli.run_analysis --experiments experiment_1 --output-dir ./analysis_results --show-plots --save-plots all
```

### 3) Export run tables

To export all MLflow runs in selected experiments to CSVs:

```bash
python -m regain.cli.export_runs --config-files ./config/experiment_a.yaml,./config/experiment_b.yaml --output-dir ./exports
python -m regain.cli.export_runs --config-dir ./config --output-dir ./exports
python -m regain.cli.export_runs --experiments experiment_1,experiment_2 --output-dir ./exports
```

Outputs are written under:
* `./exports/<experiment>/run_metadata.csv`
* `./exports/<experiment>/run_params.csv`
* `./exports/<experiment>/run_metrics.csv`

### 4) Export analysis bundles

To export from existing `./analysis_results/<experiment>/tables/*.jsonl` and derived CSVs:

```bash
python -m regain.cli.export_analysis --experiments experiment_1 --analysis-dir ./analysis_results --output-dir ./exports
```

If `./exports/<experiment>/analysis.json` already exists, add `--overwrite`.

Like `export_runs`, `export_analysis` supports `--allow-partial` (best-effort publishing) and `--overwrite`.

Outputs are organized under:

* `./analysis_results/<experiment>/tables/` (from `collect`; JSONL: `runs_table.jsonl`, `experiences_table.jsonl`)
* `./analysis_results/<experiment>/curves/` (from `curves`; CSV: `recoverability_curve.csv`, `task_age_rho.csv`,
  `calibration_vs_budget.csv`, `latency_vs_budget.csv`)
* `./analysis_results/<experiment>/frontier/` (from `frontier`)
* `./analysis_results/<experiment>/predictive/` (from `predictive`; CSV: `predictive_correlations.csv`)
* `./analysis_results/<experiment>/plots/` (when `--save-plots` is used)

### Plot later (if you didn’t plot during analysis)

If you ran analysis without `--show-plots` / `--save-plots`, you can render plots afterwards from the saved CSVs:

```bash
python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --show
python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --save
python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --save --perf-key analysis.acc.final.avg.ctrl
python -m regain.cli.generate_plots --analysis-dir ./analysis_results --experiments experiment_1 --show --save --output-dir ./plots
```

`generate_plots` defaults to interactive display when neither `--show` nor `--save` is provided. When saving, add
`--overwrite` to replace existing target plot directories. `--allow-partial` enables best-effort publishing on errors.

## Debugging

See [docs/debugging.md](docs/debugging.md) for details on the repair-controller debug instrumentation, MLflow metrics,
and health score diagnostics.

## Coding agent setup

See [docs/coding-agent-setup.md](docs/coding-agent-setup.md) for the project-local coding-agent setup used in this
repository.
