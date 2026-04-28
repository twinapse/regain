# Research rotocol

This document defines the research protocol: how we map repairability offline, how we evaluate practical repair 
selection, and how we avoid confusing oracle controller comparison with deployable decision making.

## Table of contents

- [1. Thesis](#1-thesis)
- [2. Two-phase protocol](#2-two-phase-protocol)
- [3. Controller families](#3-controller-families)
- [4. Frontier construction](#4-frontier-construction)
- [5. Utility target](#5-utility-target)
- [6. Policy input contract](#6-policy-input-contract)
- [7. Policy evaluation](#7-policy-evaluation)
- [8. Validation levels](#8-validation-levels)
- [9. Cheap racing variant](#9-cheap-racing-variant)
- [10. Failure-mode interpretation](#10-failure-mode-interpretation)
- [11. Decision gates](#11-decision-gates)
- [12. Reporting rule](#12-reporting-rule)

## 1. Thesis

REGAIN is a repairability framework for continual forgetting.

It measures, predicts, and explains which parts of forgetting are recoverable by constrained post-hoc interventions
under explicit data, capacity, latency, and harm constraints.

The goal is not to claim that one controller is always best. The goal is to identify which failure regimes are
repairable, what kind of repair they need, and when post-hoc repair should not be trusted.

## 2. Two-phase protocol

REGAIN separates two phases.

### 2.1 Offline repairability mapping

The offline phase may fit many controllers.

Purpose:

* construct the repairability frontier;
* estimate controller utility under shared constraints;
* identify failure regimes;
* produce labels for policy/router evaluation.

This phase answers scientific questions such as:

* How much forgetting is repairable?
* Which repair families recover which failure modes?
* Where do simple methods saturate?
* Where do high-capacity methods harm?
* Which diagnostics predict repairability?

### 2.2 Deployment-time repair selection

The deployment phase must not fit every candidate controller and then choose the winner.

A practical policy must:

1. observe the trained base continual learner;
2. compute pre-repair diagnostics;
3. choose one allowed action;
4. fit only the selected controller;
5. optionally reject the controller using a cheap validation or harm check.

The rejection check must use only repair-validation data, predeclared validation tasks, or deployment-available safety
metrics. It must not use the held-out test accuracy used for final reporting.

The practical claim is therefore not:

> We can cheaply know the best controller after training all controllers.

The practical claim is:

> Offline sweeps reveal diagnostic signatures that let future settings select a safe repair family without exhaustive
> controller fitting.

## 3. Controller families

Controller families are compared as repair mechanisms, not only as named methods.

| Family                   | Examples                             | Main question                                 |
| ------------------------ | ------------------------------------ | --------------------------------------------- |
| No repair                | no-op                                | Is repair needed at all?                      |
| Bias/logit repair        | BiC, logit bias                      | Is forgetting mostly classifier bias?         |
| Calibration repair       | temperature scaling, CIL calibration | Is confidence distortion the issue?           |
| Statistical drift repair | prototype/moment correction          | Is feature or semantic drift recoverable?     |
| Readout repair           | linear/ridge/low-rank probe          | Are frozen features still separable?          |
| Modulation repair        | gain/residual controllers            | Is local/input-conditioned correction useful? |
| Test-time repair         | ARC-like methods                     | Is online correction allowed and helpful?     |

These controller families do not all share the same evaluation contract. The table below makes those contracts explicit
so frontier comparisons can stay clear about which methods rely on a labeled repair set, an unlabeled test stream, or
test-time adaptation.

| Evaluation contract            | Uses labeled repair set? | Uses unlabeled test stream? | Updates or adapts at test time? |
| ------------------------------ | -----------------------: | --------------------------: | ------------------------------: |
| Fixed REGAIN repair controller | Yes                      | No                          | No                              |
| Cheap racing / budgeted selection | Yes, partially       | No                          | No                              |
| ARC-like test-time repair      | Usually no               | Yes                         | Yes                             |
| Training-time prevention baseline | Not as post-hoc repair | No                        | During backbone training        |

Frontier comparisons should report these evaluation contracts separately. ARC-like methods are useful comparators, but
they should not be described as having the same deployment contract as fixed repair-set controllers.

Some examples in this taxonomy are protocol targets rather than implemented controllers. Implementation status is
tracked in [docs/methods.md](methods.md); experiment configs should only use controllers present in the registry.

See [docs/methods.md](methods.md) for implementation-specific method notes.

## 4. Frontier construction

The frontier phase evaluates recovery as a function of:

* repair data budget;
* controller capacity;
* fit time;
* inference latency;
* parameter overhead;
* calibration behavior;
* harm risk.

Recommended repair budgets:

* `0.02`
* `0.05`
* `0.10`
* `0.25`
* `0.50`
* `1.00`

Primary metrics:

* absolute recovered accuracy;
* mean repairable fraction rho;
* final controller accuracy;
* harm-adjusted utility;
* fraction of tasks helped;
* fraction of tasks harmed;
* worst-task harm;
* oldest-task recovery;
* newest-task recovery;
* ECE / AECE / NLL;
* max task ECE;
* latency ratio;
* repair time;
* parameter overhead;
* Pareto membership.

A controller should be promoted to the main frontier only if:

1. it appears on the Pareto frontier in more than one setting family;
2. it improves utility over a simpler method;
3. it does not introduce unacceptable worst-task harm.

## 5. Utility target

Router and policy evaluation should optimize utility, not raw rho alone.

Example utility:

```text
U(c, b) =
    absolute_recovery(c, b)
  - lambda_harm    * harm(c, b)
  - lambda_latency * latency(c, b)
  - lambda_fit     * fit_time(c, b)
  - lambda_params  * parameter_overhead(c, b)
```

Where:

* `c` is the controller family;
* `b` is the repair budget;
* `harm` may include fraction harmed, worst-task harm, or both.

When a repair-budget cap is fixed externally, `b` is part of the setting and may appear in policy inputs. When budget is
part of the policy action, the action is `(c, b)` and the chosen budget must not also be used as a policy input for
that same decision.

The exact lambdas should be reported with each analysis.

## 6. Policy input contract

A policy is any decision rule that selects a repair action, including fixed baselines, threshold rules, cheap racing,
exhaustive validation selection, and learned selectors. A router is specifically a learned policy that maps pre-repair
diagnostics to a repair action.

A policy input is any pre-repair diagnostic, metadata field, or cheap pilot summary used by a policy or router to
choose a repair action.

A policy input is valid only if it is available before fitting the candidate controller being selected.

Allowed inputs include:

* dataset/scenario metadata;
* backbone metadata;
* replay or memory metadata;
* externally imposed repair-budget cap, if budget is fixed before policy selection;
* repair samples per class;
* base final accuracy;
* estimated forgetting/headroom;
* out-of-task error rate;
* old-vs-new logit bias;
* logit drift;
* ECE / AECE / NLL;
* max task ECE;
* feature/prototype drift;
* representation separability proxies;
* cheap pilot-fit summaries, if explicitly reported as such.

Forbidden inputs include:

* final test accuracy of candidate controllers;
* candidate-controller rho;
* candidate-controller absolute recovery;
* oracle-best controller identity;
* any statistic that requires fully fitting all candidate controllers at decision time.

Offline controller outcomes may be used as training labels or evaluation targets. They may not be used as
deployment-time policy inputs.

If repair budget is part of the policy action, the selected budget must not also be used as a policy input for that
decision. In that case, the policy chooses `(controller_family, budget)` from the allowed action set using only
information available before that choice.

The tables below summarize examples of valid pre-repair information and the practical value of routing under common
diagnostic regimes.

| Signal | Available before controller selection? | Why it is useful |
| ------ | -------------------------------------: | ---------------- |
| Base final accuracy | Yes | Measures headroom and over-repair risk. |
| Estimated forgetting / headroom | Yes, when reference or validation trajectories are available | Indicates whether repair is needed. |
| ECE / AECE / NLL | Yes | Signals calibration failure. |
| Out-of-task error rate | Yes, when task or class partitions are known | Signals classifier bias or task confusion. |
| Old-vs-new logit bias | Yes | Signals recency bias. |
| Logit drift | Yes, when checkpoints or logs are available | Signals decision-boundary instability. |
| Feature/prototype drift | Yes, when feature statistics are logged | Signals semantic or statistical drift. |
| Representation separability proxy | Yes, if computed with a cheap proxy rather than a full candidate controller | Signals whether readout repair may work. |
| Repair samples per class | Yes | Signals data sufficiency and overfitting risk. |
| Candidate-controller test accuracy | No | Requires fitting and evaluating the candidate controller. |
| Candidate-controller rho or absolute recovery | No | Leaks the outcome the router is supposed to predict. |
| Oracle-best controller identity | No | Uses final evaluation outcomes. |

| Diagnostic regime | Likely router decision | Practical value |
| ----------------- | ---------------------- | --------------- |
| Low forgetting, strong base model, small headroom | no-op or conservative calibration | Avoids over-repair. |
| High recent-task bias or out-of-task errors | BiC / logit-bias repair | Uses cheap correction for classifier bias. |
| High ECE / AECE / NLL without severe separability loss | temperature scaling or CIL calibration | Repairs confidence distortion with low capacity. |
| Feature/prototype drift with moderate probe risk | prototype or moment-based statistical repair | Uses intermediate repair instead of high-capacity readout retraining. |
| Severe forgetting but frozen features remain separable | ridge / linear / low-rank probe | Spends readout capacity only when justified. |
| Fine-grained setting, small repair set, high overfitting risk | avoid unconstrained probe; prefer ridge/prototype/BiC | Reduces CUB-style over-repair harm. |
| Poor separability and low recovery across cheap probes | no post-hoc claim or training-time intervention | Identifies likely representation degradation. |

## 7. Policy evaluation

Evaluate policies as if only the selected controller were fitted.

Compare against:

* always no-op;
* always BiC;
* always best simple calibration method;
* always linear/ridge probe;
* simple threshold policy;
* learned router;
* exhaustive model selection;
* oracle best controller.

Report:

* selected action distribution;
* utility;
* regret vs oracle;
* improvement over always-BiC;
* harm reduction vs always-linear-probe;
* fraction of tasks harmed;
* worst-task harm;
* number of controllers fitted at decision time.

Definitions:

* exhaustive model selection: fits all candidate controllers and selects using validation utility;
* oracle best controller: selects using final evaluation utility and is reported only as an unattainable upper bound.

The oracle baseline is an unattainable upper bound. Exhaustive model selection is an expensive validation-selected
baseline, not a practical deployment policy.

## 8. Validation levels

Use three validation levels where possible:

1. held-seed validation;
2. held-setting validation;
3. held-dataset validation.

Held-dataset validation is the strongest evidence that the router is learning a mechanism rather than memorizing
configurations.

## 9. Cheap racing variant

A policy may use cheap pilot signals, but this must be reported separately from pure pre-repair prediction.

Examples:

* fit BiC on a tiny repair subset;
* fit a prototype classifier;
* run a ridge probe for a few epochs;
* estimate probe train/validation gap;
* stop high-capacity repair early if validation harm appears.

This is a budgeted controller-selection policy, not a zero-cost router.

## 10. Failure-mode interpretation

Use controller outcomes to interpret forgetting regimes.

| Evidence                             | Interpretation                                  |
| ------------------------------------ | ----------------------------------------------- |
| BiC helps                            | classifier bias / shallow logit repair          |
| temperature scaling helps            | calibration distortion                          |
| prototype or moment correction helps | statistical or semantic drift                   |
| ridge/linear probe helps             | readout mismatch with separable frozen features |
| probe trains well but harms test     | overcapacity / over-repair                      |
| no low-capacity method helps         | possible representation degradation             |

## 11. Decision gates

### Gate 1: Is routing viable?

Pass if the router beats always-BiC in utility while reducing harm relative to always-linear-probe.

### Gate 2: Are calibration methods distinct from BiC?

Pass if calibration methods occupy useful frontier points.

### Gate 3: Is regularized readout repair safer than unconstrained readout repair?

Pass if ridge, low-rank, prototype, class-balanced, or early-stopped probes recover accuracy without large harm.

### Gate 4: Does statistical drift repair fill the middle?

Pass if it beats BiC where readout repair is risky.

### Gate 5: Is a new modulation controller needed?

Pass only if the frontier shows a gap that simpler families do not cover.

## 12. Reporting rule

Every REGAIN result should state which setting it belongs to:

* offline frontier mapping;
* deployment-time repair selection;
* oracle/exhaustive upper bound;
* cheap racing / budgeted selection;
* failure-mode analysis.

Do not describe oracle or exhaustive controller selection as a practical deployment policy.
