# REGAIN: Retrieval-Based Gain Assessment for Incremental Networks

This project asks:

> **How much of catastrophic forgetting in neural networks is actually “repairable” by tiny retrieval-only 
> interventions—and when do we truly need heavyweight continual-learning machinery?**

The work has two tightly linked parts:

1. **REGAIN analysis tool (frozen-backbone, offline):**

   * Define a **family of small retrieval controllers** (scalar → layer → channel → input-conditioned) that act only 
     at test-time / repair time (no backbone updates).
   * For each controller capacity and repair data budget, measure how much forgetting is **retrieval-correctable**, 
     producing **recoverability curves** and a **repair efficiency frontier**.
   * Explicitly show how this generalizes **linear-probe feature forgetting** and **BiC-style logit bias correction**.

2. **Algorithmic (sequential class-incremental CL):**

   * Design and evaluate a **task-agnostic, input-conditioned retrieval controller** (a tiny gating network that 
     outputs per-layer gains from the current input) in **class-incremental learning**.
   * Compare it to replay, calibration, and normalization baselines under realistic compute and memory budgets.

Conceptually, the project connects:

* The **dual form** view of neural networks as key–value memories over training patterns.
* **Key–value memory in the brain**, where forgetting is often framed as retrieval failure and “silent engrams” can be 
  reactivated.
* **Feature forgetting** and knowledge accumulation in continually learned representations.
* **Small post-hoc corrections** like BiC and normalization fixes for class-incremental learning.
* **Modulation-based CL** and **NTK reactivation** as mechanistic lenses.

## Methods

See [docs/methods.md](docs/methods.md) for implementation notes on specific methods used in this codebase.

## Experimental framework

See [docs/experimental-framework.md](docs/experimental-framework.md) for details on the benchmarks and metrics used in 
the experiments.

## Debugging

See [docs/debugging.md](docs/debugging.md) for details on the repair-controller debug instrumentation, MLflow metrics,
and health score diagnostics.

## CLI usage

The typical workflow is:

1) **Run experiments** (logs metrics to MLflow).  
2) **Run analysis** on the logged runs (writes analysis artifacts, and optionally plots).

### 1) Run experiments (log metrics)

Provide one or more YAML configs (comma-separated) and run:

```bash
python -m regain.cli.run_experiment --config_files ./config/experiment_a.yaml,./config/experiment_b.yaml
```

This produces MLflow runs under your configured tracking URI. Artifact storage follows MLflow defaults unless you set 
`mlflow_artifact_uri` to a filesystem path or artifact URI (for example `file:///path/to/mlruns`).

To export all MLflow runs in each experiment to CSVs (one row per run, written to
`./<export_dir>/<experiment>/run_metadata.csv`, `./<export_dir>/<experiment>/run_params.csv`, and
`./<export_dir>/<experiment>/run_metrics.csv`, overwrites existing exports to capture the latest snapshot/state):

```bash
python -m regain.cli.run_experiment --config_files ./config/experiment_a.yaml,./config/experiment_b.yaml --export-dir ./exports
```

### 2) Analyze logged runs (write artifacts, optionally plot)

`run_analysis` has subcommands:

* `collect`: download/aggregate finished parent MLflow runs into tidy JSONL tables
* `curves`: compute recoverability, task-age, calibration-vs-budget, and latency-vs-budget curves (requires `collect`)
* `frontier`: compute the efficiency frontier from the curve CSV (requires `curves` output)
* `predictive`: compute diagnostic-vs-repairability correlations (requires `collect`)
* `all`: run `collect + curves + frontier + predictive`

Notes:
* For repair-controller runs, predictive summaries and run-level `run.calibration.max_ece` in `runs_table` are baseline-only:
  sourced from base values in `analysis_artifacts.json`.
  `run.calibration.max_ece` is defined as `max(run.calibration.ece)` over artifact baseline vectors.
* Analysis collection requires each run to include `controller.type` (`none`, `prevention`, or `repair`) and
  `repair.split_fraction` parameters.
* `repair_set_total` in analysis tables is:
  - `0` when `repair.split_fraction == 0.0`;
  - otherwise, the exact total non-empty line count across `repair/exp_*.txt` in `splits.tar.gz`.

Common flags:

* `--experiment`: MLflow experiment name or id (e.g. `experiment_1`)
* `--output-dir`: output directory root (experiment subdirectory is created under this path)
* `--export-dir`: export directory root (writes `<export-dir>/<experiment>/analysis.json`, overwrites to capture the latest snapshot/state)
* `--tracking-uri`: MLflow tracking URI
* `--artifact-uri`: MLflow artifact URI or filesystem path
* `--show-plots`: display plots interactively
* `--save-plots`: save plots to `<output-dir>/<experiment>/plots`
* `--perf-key`: metric key to maximize for the frontier and plot in curves

Examples:

```bash
# Step-by-step
python -m regain.cli.run_analysis collect  --experiment experiment_1 --output-dir ./analysis_results
python -m regain.cli.run_analysis curves   --experiment experiment_1 --output-dir ./analysis_results --show-plots
python -m regain.cli.run_analysis frontier --experiment experiment_1 --output-dir ./analysis_results --perf-key analysis.repair.rho.avg --save-plots
python -m regain.cli.run_analysis predictive --experiment experiment_1 --output-dir ./analysis_results

# One-shot (recommended for most use)
python -m regain.cli.run_analysis all --experiment experiment_1 --output-dir ./analysis_results --show-plots --save-plots

# Export a self-contained JSON bundle
python -m regain.cli.run_analysis all --experiment experiment_1 --output-dir ./analysis_results --export-dir ./exports
```

Outputs are organized under:

* `./analysis_results/<experiment>/tables/` (from `collect`; JSONL: `runs_table.jsonl`, `experiences_table.jsonl`)
* `./analysis_results/<experiment>/curves/` (from `curves`; CSV: `recoverability_curve.csv`, `task_age_rho.csv`, `calibration_vs_budget.csv`, `latency_vs_budget.csv`)
* `./analysis_results/<experiment>/frontier/` (from `frontier`)
* `./analysis_results/<experiment>/predictive/` (from `predictive`; CSV: `predictive_correlations.csv`)
* `./analysis_results/<experiment>/plots/` (when `--save-plots` is used)

### Plot later (if you didn’t plot during analysis)

If you ran analysis without `--show-plots` / `--save-plots`, you can render plots afterwards from the saved CSVs:

```bash
python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --show
python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --save
python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --save --perf-key analysis.accuracy.final.avg.ctrl
python -m regain.cli.generate_plots --analysis-dir ./analysis_results/experiment_1 --show --save --save-dir ./plots
```

## Code style and formatting

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) and the 
[Google Python Style Guide](https://github.com/google/styleguide/blob/gh-pages/pyguide.md). Check [pylintrc](pylintrc) 
and [.style.yapf](.style.yapf) for the specific formatting rules.

To ensure code consistency, we use the following tools:

1. [isort](https://pycqa.github.io/isort/): Organizes the imports. Run:

   ```
   isort regain tests --profile google --line-length 120
   ```

2. [YAPF](https://github.com/google/yapf): Formats the code. Run:

   ```
   yapf --recursive --in-place regain tests
   ```

3. [Ruff](https://github.com/astral-sh/ruff): Quickly lints and fixes the code. Run:

   ```
   ruff check regain tests --fix
   ```

4. [Pylint](https://pypi.org/project/pylint/): Thoroughly lints the code. Run:

   ```
   pylint --rcfile=pylintrc regain
   ```

   to inspect the main [`regain`](regain) package, and

   ```
   pylint --rcfile=pylintrc --disable=redefined-outer-name,unused-argument tests
   ```

   to inspect the [`tests`](tests) package. We use different configurations for the main package and the tests to 
   avoid Pytest-related false positives.
