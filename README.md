# REGAIN: Retrieval-Based Gain Assessment for Incremental Networks

This project asks:

> **How much of catastrophic forgetting in neural networks is actually “repairable” by tiny retrieval-only interventions—and when do we truly need heavyweight continual-learning machinery?**

The work has two tightly linked parts:

1. **Diagnostic (frozen-backbone, offline):**

   * Define a **family of small retrieval controllers** (scalar → layer → channel → input-conditioned) that act only at test-time / repair time (no backbone updates).
   * For each controller capacity and repair data budget, measure how much forgetting is **retrieval-correctable**, producing **recoverability curves** and a **repair efficiency frontier**.
   * Explicitly show how this generalizes **linear-probe feature forgetting** and **BiC-style logit bias correction**.

2. **Algorithmic (sequential class-incremental CL):**

   * Design and evaluate a **task-agnostic, input-conditioned retrieval controller** (a tiny gating network that outputs per-layer gains from the current input) in **class-incremental learning**.
   * Compare it to replay, calibration, and normalization baselines under realistic compute and memory budgets.

Conceptually, the project connects:

* The **dual form** view of neural networks as key–value memories over training patterns.
* **Key–value memory in the brain**, where forgetting is often framed as retrieval failure and “silent engrams” can be reactivated.
* **Feature forgetting** and knowledge accumulation in continually learned representations.
* **Small post-hoc corrections** like BiC and normalization fixes for class-incremental learning.
* **Modulation-based CL** and **NTK reactivation** as mechanistic lenses.
