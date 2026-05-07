# Core concepts

Short glossary of core concepts.

- **Action set** — The set of repair choices available to a repair-selection policy, such as no repair, a repair family, or a repair
  family paired with a budget.

- **Baseline-only diagnostics** — The rule that diagnostics used to analyze repair runs should describe the unrepaired
  backbone, not controller-modified behavior.

- **Budgeted selection** — A repair-selection setting where limited pilot fitting or validation is allowed before the
  final repair action is chosen.

- **Cheap pilot signal** — A low-cost signal from a small preliminary fit, proxy computation, or validation check
  used to guide repair selection.

- **Cheap racing** — A budgeted selection protocol that tries lightweight or partial candidate repairs before
  committing to one repair action.

- **Controller family** — A mechanism-level category of controllers grouped by the kind of repair or intervention they
  perform.

- **Controller-off evaluation** — Evaluation with a repair controller disabled, exposing the underlying unrepaired
  backbone behavior.

- **Controller-on evaluation** — Evaluation with a repair controller enabled, exposing repaired behavior.

- **Decision gate** — A predeclared criterion used to decide whether a research claim, controller family, or next
  implementation direction is justified.

- **Deployment-time repair selection** — The practical protocol in which a repair-selection policy observes pre-repair
  information, chooses one repair action, fits only that action, and may reject it using allowed validation.

- **Diagnostic association** — A relationship between pre-repair signals and later repair outcomes.

- **Diagnostic regime** — A recurring pattern of pre-repair signals that suggests a likely failure type or appropriate
  repair choice.

- **Diagnostic signal** — Information available before controller selection that may help predict repair
  benefit, repair risk, or failure mode.

- **Disjoint repair data** — Repair data held out from backbone training so post-hoc repair is evaluated
  without reusing the backbone's training samples.

- **Efficiency frontier** — The best observed tradeoff surface among repair benefit, resource cost, and harm.

- **Evaluation contract** — The assumptions under which a controller is evaluated, including what data it may use and
  whether it may adapt during evaluation.

- **Exhaustive controller selection** — A comparison protocol that fits all candidate controllers and chooses
  using validation utility; it is not treated as practical deployment-time selection.

- **Failure-mode analysis** — Interpretation of repair outcomes to infer what kind of degradation occurred and which
  repair mechanisms are sufficient.

- **Fit schedule** — The rule specifying when a repair controller is fit, such as after each experience or
  only after the full training sequence.

- **Frontier construction** — The offline process of evaluating repair options across benefit, cost,
  capacity, data, and harm dimensions.

- **Harm check** — A validation or safety check used to reject a repair action that appears likely to damage behavior.

- **Harm constraint** — A limit on the degradation a repair action is allowed to introduce.

- **Harm risk** — The possibility that a repair action improves some behavior while worsening other behavior.

- **Headroom** — The apparent room for recovery between degraded post-sequence behavior and a stronger reference point.

- **Held-dataset validation** — Repair-selection evaluation on a dataset not used to build or tune the repair-selection policy.

- **Held-seed validation** — Repair-selection evaluation on random seeds not used to build or tune the repair-selection policy.

- **Held-setting validation** — Repair-selection evaluation on experimental settings not used to build or tune the repair-selection policy.

- **Label-space regime** — The prediction space used during evaluation, such as restricting predictions to
  seen classes or allowing all classes.

- **No-op action** — The decision to apply no repair, used both as a baseline and as a valid conservative
  repair-selection policy choice.

- **Offline repairability mapping** — The research protocol in which many controllers and budgets may be evaluated to
  estimate repairability frontiers and study failure regimes.

- **Oracle upper bound** — An unattainable comparison point that chooses the best repair action using final evaluation
  outcomes.

- **Repair-selection policy** — Any decision rule that selects a repair action from available information.

- **Repair-selection input** — Any diagnostic, metadata field, budget cap, or pilot summary used by a repair-selection policy to choose a repair
  action.

- **Post-hoc repair** — Repair that fits or applies a controller after training boundaries while leaving the backbone
  training trajectory unchanged.

- **Pre-repair diagnostic** — A diagnostic signal available before the candidate controller being selected
  has been fully fit.

- **Prevention controller** — A controller that changes the training trajectory and therefore serves as a prevention
  baseline rather than a purely post-hoc repair.

- **Repair action** — The concrete choice made by a repair-selection policy, usually a repair family, budget, or no-op.

- **Repair budget** — The amount or fraction of held-out repair data available for fitting a repair controller.

- **Repair controller** — A post-hoc controller that uses repair data or inference-time correction without changing
  backbone training dynamics.

- **Repair data** — Data reserved for fitting or validating post-hoc controllers rather than training the backbone.

- **Repair family** — A broad mechanism category for repair controllers used in frontier and repair-selection policy comparisons.

- **Repair fit subset** — The subset of the repair set actually used for controller fitting under the configured budget.

- **Repair mechanism** — The kind of intervention a repair family uses to recover from degradation.

- **Repair split** — The experiment-wide rule that divides each experience's training data into backbone-training and
  repair portions.

- **Repair stream** — The sequence of held-out repair sets aligned with the training sequence.

- **Repairability** — The degree to which observed degradation can be recovered by allowed post-hoc
  interventions under explicit constraints.

- **Repairability frontier** — A constraint-aware map of which repair mechanisms recover how much degradation at what
  cost.

- **Repairable forgetting** — The portion of observed forgetting that can be recovered by a valid repair action.

- **Reserved backbone run** — The controller-free run used to produce the shared trajectory, checkpoints, and baseline
  artifacts for repair comparisons.

- **Repair router** — A learned repair-selection policy that maps pre-repair diagnostics to a repair action.

- **Shared backbone** — A single trained backbone trajectory reused across repair-controller runs so that
  controllers are compared on the same underlying states.

- **Split integrity** — The guarantee that training, repair, and evaluation data roles remain separate
  according to the protocol.

- **Test-time repair contract** — An evaluation contract where the method may use evaluation-time information or adapt
  during testing, making it distinct from fixed repair-set controllers.

- **Training-time prevention baseline** — A non-post-hoc comparison point that changes training dynamics to show when
  repair may be insufficient.

- **Utility target** — The decision objective used for repair-selection evaluation, combining repair benefit with penalties for
  harm and resource cost.

- **Validation task** — A predeclared non-test evaluation source used to accept, reject, or choose a repair action.
