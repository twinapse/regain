# Project overview

This document captures research, domain, and stack context for contributors working on REGAIN.

<!-- toc -->

- [What is REGAIN?](#what-is-regain)
- [Research framing](#research-framing)
- [Core capabilities](#core-capabilities)
- [Domain model](#domain-model)
- [Tech stack](#tech-stack)

## What is REGAIN?

REGAIN is a research codebase for measuring how much catastrophic forgetting in class-incremental
neural networks is repairable by constrained post-training interventions.

The project studies repairability under a shared experimental and analysis framework. It trains or reuses shared
backbone trajectories, evaluates representative repair mechanisms under held-out repair-data and resource constraints,
and logs enough information to compute recoverability curves, efficiency frontiers, calibration summaries, and
diagnostic associations.

## Research framing

This project asks:

> **How much observed forgetting is repairable under constrained controller capacity and repair data, and what kind of
> repair mechanism is sufficient in each regime?**

The central object is a **repairability frontier**: a capacity-, data-, and cost-aware map of how much performance can
be recovered after continual learning without updating the backbone. The frontier treats different repair mechanisms as
comparable interventions under a common evaluation protocol.

REGAIN has a single goal: characterize the repairability frontier for continual forgetting under shared experimental
constraints. The repository provides the experimental protocol, representative controller implementations, and analysis
pipeline needed to compare repair mechanisms across controller capacity, repair data budget, compute, calibration, and
latency tradeoffs. Those comparisons are meant to reveal which failure modes look repairable through logit/bias
correction, readout repair, modulation, or deeper intervention, rather than to position new controller development as a
separate project objective.
Training-time / prevention controllers are included only as comparison baselines: they help locate the boundary between
forgetting that remains repairable after training and forgetting that instead requires intervention during backbone
training.

Conceptually, the project connects:

- The **dual form** view of neural networks as key-value memories over training patterns.
- **Key-value memory in the brain**, where forgetting is often framed as retrieval failure and silent engrams can be
  reactivated.
- **Feature forgetting** and knowledge accumulation in continually learned representations.
- **Small post-hoc corrections** such as bias correction, calibration, statistical drift correction, and readout repair.
- **Modulation-based continual learning** and **NTK reactivation** as mechanistic lenses.

## Core capabilities

- **Experiment execution**: YAML experiment configs define the scenario, backbone, Avalanche strategy, repair split,
  controller runs, evaluation schedule, and MLflow output behavior.
- **Shared-backbone controller comparisons**: Repair-controller experiments can train one reserved `backbone` run and
  reuse its checkpoints so post-hoc controllers are compared on the same model trajectory.
- **Repair data management**: Scenario builders can split each training experience into a backbone-training stream and
  a disjoint repair stream. Repair controllers fit only on the configured repair budget.
- **Controller extensibility**: The registry exposes backbones, scenarios, learning-rate schedulers, repair-buffer
  policies, repair controllers, and prevention controllers by stable configuration names.
- **Evaluation and diagnostics**: Custom Avalanche plugins record reference and final accuracies, controller-on and
  controller-off outputs, calibration metrics, prediction artifacts, forgetting metrics, latency metrics, and optional
  repair debug health scores.
- **Analysis automation**: Analysis commands collect finished MLflow runs into tables, compute recoverability curves,
  task-age summaries, efficiency frontiers, predictive correlations, plots, and self-contained JSON exports.

## Domain model

- **Experiments**: MLflow experiments selected by name or id. A config file describes one experiment and its run set.
- **Runs**: MLflow runs inside an experiment. The reserved `backbone` run is controller-free and stores the shared
  checkpoints and baseline analysis vectors. User-configured runs attach one controller each.
- **Scenarios**: Class-incremental benchmarks built as Avalanche `NCScenario` instances. Supported registry names are
  `cifar100`, `cub200`, `tiny_imagenet`, and `imagenet_r`.
- **Experiences**: Sequential class-incremental tasks. Each experience has train/test data, seen-class state, and
  optional repair data carved out before backbone training.
- **Backbones**: Classification models such as `resnet18`, `vit_small`, and `vit_base`. A backbone config either trains
  locally or points to a source experiment that contains reusable backbone checkpoints.
- **Controllers**: Modules attached to a run. Repair controllers fit after training boundaries and correct outputs
  during evaluation, while prevention controllers can modify training dynamics.
- **Metrics**: Runtime values logged under stable MLflow namespaces. Analysis code depends on the canonical
  `run.eval.*`, `run.repair.*`, `run.calibration.*`, `run.diagnostics.*`, `run.latency.*`, and `run.debug.*` naming
  schemes.
- **Analysis artifacts**: Table-first outputs written under `tables`, `curves`, `frontier`, `predictive`, and `plots`
  directories, plus optional `analysis.json` export bundles.

## Tech stack

- Python 3.11 in CI
- PyTorch, TorchVision, and timm for model definitions and training
- Avalanche for continual-learning scenarios, strategies, and plugin integration
- MLflow for experiment tracking, metrics, parameters, and artifacts
- NumPy for numerical utilities and analysis helpers
- PyYAML for experiment configuration
- Matplotlib for optional analysis plots
- Pytest for automated tests
