# Project overview

This document captures research, domain, and stack context for contributors working on REGAIN.

<!-- toc -->

- [What is REGAIN?](#what-is-regain)
- [Research framing](#research-framing)
- [Core capabilities](#core-capabilities)
- [Domain model](#domain-model)
- [Tech stack](#tech-stack)

## What is REGAIN?

REGAIN is a research codebase for measuring how much catastrophic forgetting in class-incremental neural networks is
retrieval-correctable by small controllers.

The project compares a frozen-backbone, post-training repair setting against prevention-style continual-learning
baselines. It trains or reuses a shared backbone trajectory, fits lightweight controllers on held-out repair data, and
logs enough information to compute recoverability curves, efficiency frontiers, calibration summaries, and diagnostic
associations.

## Research framing

This project asks:

> **How much of catastrophic forgetting in neural networks is actually "repairable" by tiny retrieval-only
> interventions, and when do we truly need heavyweight continual-learning machinery?**

The work has two tightly linked parts:

1. **REGAIN analysis tool (frozen-backbone, offline)**:

   - Define a **family of small retrieval controllers** (scalar, layer, channel, input-conditioned) that act only at
     test time or repair time, without backbone updates.
   - For each controller capacity and repair data budget, measure how much forgetting is **retrieval-correctable**,
     producing **recoverability curves** and a **repair efficiency frontier**.
   - Explicitly show how this generalizes **linear-probe feature forgetting** and **BiC-style logit bias correction**.

2. **Algorithmic (sequential class-incremental continual learning)**:

   - Design and evaluate a **task-agnostic, input-conditioned retrieval controller**: a tiny gating network that outputs
     per-layer gains from the current input in **class-incremental learning**.
   - Compare it to replay, calibration, and normalization baselines under realistic compute and memory budgets.

Conceptually, the project connects:

- The **dual form** view of neural networks as key-value memories over training patterns.
- **Key-value memory in the brain**, where forgetting is often framed as retrieval failure and silent engrams can be
  reactivated.
- **Feature forgetting** and knowledge accumulation in continually learned representations.
- **Small post-hoc corrections** like BiC and normalization fixes for class-incremental learning.
- **Modulation-based continual learning** and **NTK reactivation** as mechanistic lenses.

## Core capabilities

- **Experiment execution**: YAML experiment configs define the scenario, backbone, Avalanche strategy, repair split,
  controller runs, evaluation schedule, and MLflow output behavior.
- **Shared-backbone controller comparisons**: Repair-controller experiments can train one reserved `backbone` run and
  reuse its checkpoints so post-hoc controllers are compared on the same model trajectory.
- **Repair data management**: Scenario builders can split each training experience into a backbone-training stream and
  a disjoint repair stream. Repair controllers fit only on the configured repair budget.
- **Controller extensibility**: The registry exposes backbones, scenarios, learning-rate schedulers, repair-buffer
  policies, prevention controllers, and repair controllers by stable configuration names.
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
- **Controllers**: Lightweight modules attached to a run. Prevention controllers can modify training dynamics, while
  repair controllers fit after training boundaries and correct outputs during evaluation.
- **Metrics**: Runtime values logged under stable MLflow namespaces. Analysis code depends on the canonical
  `run.eval.*`, `run.repair.*`, `run.calibration.*`, `run.diagnostic.*`, `run.latency.*`, and `run.debug.*` naming
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
