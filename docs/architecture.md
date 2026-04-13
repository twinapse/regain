# Architecture guide

This guide records repository-specific architecture patterns for REGAIN.

<!-- toc -->

- [Configuration](#configuration)
- [Registries and builders](#registries-and-builders)
- [Experiment orchestration](#experiment-orchestration)
- [Controllers](#controllers)
- [Evaluation and metrics](#evaluation-and-metrics)
- [Analysis pipeline](#analysis-pipeline)
- [CLI and output management](#cli-and-output-management)
- [Logging](#logging)

## Configuration

- Experiment YAML files are parsed through `regain.experiments.config`. Keep config validation centralized there rather
  than scattering ad hoc validation across CLI entry points.
- Required experiment-level concepts are `experiment_name`, `scenario`, `num_experiences`, `backbone`, `repair`,
  `evaluation`, and `runs`.
- Treat `repair.split_fraction` as part of the backbone-training protocol. It changes what data the backbone sees, so
  it must stay experiment-wide instead of becoming a per-run override.
- Keep run-specific config focused on controller identity and controller kwargs. Shared runtime settings such as device,
  repair budget, evaluation batch size, strategy settings, and checkpoint behavior belong at the experiment level.
- Do not allow user configs to set reserved MLflow parameters directly. Reserved runtime params are owned by logging and
  orchestration code.

## Registries and builders

- Public config names are resolved through `regain.registry`. Add new scenarios, backbones, controllers, learning-rate
  schedulers, and repair-buffer policies there before using them in YAML.
- Builder code lives in `regain.experiments.builders`. Use builders to validate constructor signatures, inject reserved
  parameters such as class counts, and create Avalanche strategies.
- Scenario construction belongs in `regain.avalanche_utils.scenarios`. Scenario builders should return Avalanche
  `NCScenario` objects and preserve the class-id and split-integrity invariants documented in the experimental
  framework.
- Keep fully qualified import paths absolute. Registry entries should point at importable classes or callables and be
  covered by tests when a new extension point is added.

## Experiment orchestration

- `regain.experiments.orchestrator.run_experiment()` owns the end-to-end run flow: MLflow setup, scenario creation,
  controller construction, strategy construction, training, evaluation, artifact logging, and shared-backbone reuse.
- The reserved `backbone` run is the only controller-free training trajectory for repair comparisons. Repair-controller
  runs should load its checkpoints instead of retraining the backbone.
- Source-experiment reuse must load both backbone checkpoints and the baseline analysis payload. Metric-only source
  runs are not sufficient for repair analysis.
- Prevention controllers participate in training and cannot be compared through the same controller-off post-hoc toggle
  used by repair controllers.
- Keep checkpoint paths ordered by experience index and validate that a checkpoint exists for every training experience
  before using them in repair runs.

## Controllers

- Controller base classes and lifecycle contracts live in `regain.models.controllers.base`.
- Use `PreventionController` for training-time interventions that can modify the model, objective, or training
  trajectory.
- Use `RepairController` for post-hoc interventions that fit on repair data and correct outputs during evaluation
  without changing backbone training dynamics.
- Use `BackboneControllerInterface` when a prevention controller rewrites model structure or state, and
  `TrainingObjectiveControllerInterface` when it modifies the training loss.
- Repair controllers should implement `requires_per_experience_fitting()` when final-only fitting would be invalid.
- Shared repair-data sampling and repair-buffer behavior belongs in `regain.models.controllers.sampling` or
  controller-specific helpers, not in the orchestration layer.

## Evaluation and metrics

- Custom evaluation behavior belongs in `regain.avalanche_utils.evaluation` and `regain.avalanche_utils.plugins`.
  Avoid bypassing these layers with one-off evaluation loops inside controllers.
- `MetricContextPlugin` owns the current phase, namespace, experience, epoch, and logging step used by custom metrics.
- `RegainEvaluationPlugin` runs the post-hoc evaluator that records reference accuracies, final base/ctrl accuracies,
  calibration metrics, prediction artifacts, forgetting metrics, and repair metrics.
- Evaluation code must preserve controller-off base measurements for repair-controller runs. Base diagnostic vectors
  used by repair analysis come from the reserved backbone baseline payload.
- Keep metric names aligned with constants in `regain.constants`. Analysis collectors depend on exact namespaces and
  experience-token placement.
- Use `regain.evaluation.guards.frozen_model_state()` around evaluation paths that must not mutate backbone state.

## Analysis pipeline

- Analysis code is table-first. `regain.analysis.collectors.collect_experiment_tables()` converts MLflow runs into
  `runs_table` and `experiences_table` rows before downstream calculations run.
- Curves, frontiers, predictive associations, plots, and JSON exports should consume saved tables or derived CSVs rather
  than reaching back into MLflow unnecessarily.
- Keep schema changes explicit. When `analysis.json` changes shape, update the export contract and tests that assert the
  payload contract.
- Missing or invalid runs should be reported through the CLI output helpers. `--allow-partial` controls whether
  successful staged outputs are still published after failures.
- Derived analysis outputs should remain reproducible from MLflow metrics, logged artifacts, and the analysis tables.

## CLI and output management

- CLI entry points live under `regain.cli`. Keep argument parsing thin and delegate behavior to experiment, analysis,
  or export helpers.
- Shared experiment selectors belong in `regain.cli._utils.selector_helpers`. Use the standard selector options instead
  of implementing new `--experiments`, `--config-files`, or `--config-dir` parsing.
- Staged output publishing belongs in `regain.cli._utils.output_helpers`. Use it for analysis and export commands that
  write directory trees and need rollback or best-effort behavior.
- Preserve the existing `--overwrite` and `--allow-partial` semantics when adding new output-producing CLI commands.
- Keep generated analysis directories organized under the existing `tables`, `curves`, `frontier`, `predictive`, and
  `plots` subdirectories unless a new artifact family is needed.

## Logging

- Use `regain.utils.get_logger()` for project logging.
- Log actionable warnings with enough run, experiment, stage, or artifact context to diagnose the problem.
- Use MLflow for experiment parameters, scalar metrics, run artifacts, prediction artifacts, split archives,
  checkpoints, and fatal error context.
- Avoid `print` statements outside CLI user-facing summaries.
